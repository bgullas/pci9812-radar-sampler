"""
plot_radar.py  —  Real-time PPI radar from PCI-9812 + Furuno FAR-2xx8
                  with dynamic VIDEO sampling rate (1 – 20 MS/s)

━━  HARDWARE LIMIT  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  The PCI-9812 samples all 4 channels simultaneously at one shared clock.
  Maximum is 20 MS/s (not 50 MS/s).  Changing the rate at runtime stops
  the DMA, reallocates the buffer at the new rate, and restarts — the
  display freezes on the last frame for ~1 s during the switchover.

  Digital channels (HD CH0, TRIG CH2) are decimated in software to an
  effective 1 MS/s for pulse detection so their accuracy is unchanged
  at any VIDEO rate.  Only VIDEO (CH3) uses the full clock rate.

━━  RATE / RANGE RESOLUTION TABLE  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Slider pos   Rate      Range res   Bins / 3 NM   DMA buf / half
  ──────────   ────────  ─────────   ───────────   ─────────────
      1        1  MS/s   149.9 m          41          ~  8 MB
      2        2  MS/s    75.0 m          80          ~ 16 MB
      5        5  MS/s    30.0 m         189          ~ 40 MB
     10       10  MS/s    15.0 m         374          ~ 80 MB
     20       20  MS/s     7.5 m         745          ~160 MB

Channel wiring (Furuno FAR-2xx8, J510):
  CH0 = HD    — heading / north pulse
  CH1 = BP    — bearing pulse (not used)
  CH2 = TRIG  — range trigger
  CH3 = VIDEO — radar echo

Usage:  python plot_radar.py
        Move the slider at the bottom to change VIDEO rate live.
"""

import sys
import os
import re
import ctypes
import ctypes.util
import time
import threading
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Slider

if sys.platform != 'win32':
    raise EnvironmentError(
        'plot_radar.py requires Windows and the ADLINK PCIS-DASK driver.'
    )

# ─────────────────────────────────────────────────────────────────────────────
# Channel assignments
# ─────────────────────────────────────────────────────────────────────────────
CH_HD    = 0
CH_BP    = 1
CH_TRIG  = 2
CH_VIDEO = 3
N_CH     = 4

# ─────────────────────────────────────────────────────────────────────────────
# Fixed settings  (independent of VIDEO rate)
# ─────────────────────────────────────────────────────────────────────────────
CARD_NUM         = 0
DISPLAY_RANGE_NM = 3.0
PERSIST          = 0.92       # polar buffer decay per revolution
THRESH_HD        = 0.30       # V  HD rising-edge threshold
THRESH_TRIG      = 0.30       # V  TRIG rising-edge threshold
HALF_PERIOD_S    = 0.5        # target time per DMA half-buffer (s)
N_BEARINGS       = 512
PPI_SIZE         = 600

ADC_BITS  = 12
ADC_MID   = 2 ** (ADC_BITS - 1)   # 2048
VRANGE    = 5.0                    # ±5 V
C         = 299_792_458            # m/s
CTRL_RATE = 1_000_000              # effective rate for digital channels (Hz)
MAX_RANGE_M   = DISPLAY_RANGE_NM * 1852
METERS_PER_NM = 1852

# Selectable VIDEO rates  (must be multiples of CTRL_RATE)
RATE_OPTIONS  = [1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000]
RATE_LABELS   = ['1 MS/s', '2 MS/s', '5 MS/s', '10 MS/s', '20 MS/s']
DEFAULT_RATE  = 1_000_000     # startup rate

# Pre-allocate polar buffer at the MAXIMUM possible number of range bins
# (20 MS/s → 7.5 m/bin for 3 NM → ~745 bins).  Lower rates just use fewer.
_MAX_RANGE_RES = C / (2 * max(RATE_OPTIONS))
MAX_RANGE_BINS = int(MAX_RANGE_M / _MAX_RANGE_RES) + 8

# Minimum pulse gaps in CTRL_RATE (1 MHz) decimated scan space
_GAP_HD_DC   = int(0.50 * CTRL_RATE)   # 0.5 s
_GAP_TRIG_DC = int(5e-4 * CTRL_RATE)   # 0.5 ms

# ─────────────────────────────────────────────────────────────────────────────
# Colour map
# ─────────────────────────────────────────────────────────────────────────────
RADAR_CMAP = LinearSegmentedColormap.from_list('radar', [
    (0.00, (0.00, 0.00, 0.00)),
    (0.14, (0.00, 0.09, 0.02)),
    (0.38, (0.00, 0.40, 0.10)),
    (0.70, (0.05, 0.72, 0.20)),
    (0.88, (0.42, 0.92, 0.44)),
    (1.00, (0.90, 1.00, 0.82)),
])


# ─────────────────────────────────────────────────────────────────────────────
# RateConfig  — all parameters that change with VIDEO rate
# ─────────────────────────────────────────────────────────────────────────────

class RateConfig:
    """
    Encapsulates every quantity derived from VIDEO_RATE_HZ.
    Build a new RateConfig when the user changes the rate; the old one
    stays valid for any in-flight process_half() call that holds a reference.
    """

    def __init__(self, video_rate_hz: int):
        self.video_rate_hz = int(video_rate_hz)
        self.range_res_m   = C / (2 * self.video_rate_hz)
        self.n_range_bins  = min(
            int(MAX_RANGE_M / self.range_res_m) + 4,
            MAX_RANGE_BINS,
        )
        self.decimate_n    = max(1, self.video_rate_hz // CTRL_RATE)
        # DMA buffer sizes (total uint16 values across all channels)
        raw_half = int(HALF_PERIOD_S * self.video_rate_hz * N_CH)
        self.half_count = (raw_half // (N_CH * 2)) * (N_CH * 2)
        self.read_count = self.half_count * 2
        # Min pulse gap for TRIG in decimated-scan space
        self._gap_trig_dc = _GAP_TRIG_DC
        # Polar-to-Cartesian lookup (built once per config)
        self._build_lookup()

    def _build_lookup(self):
        px = np.linspace(-MAX_RANGE_M,  MAX_RANGE_M, PPI_SIZE)
        py = np.linspace( MAX_RANGE_M, -MAX_RANGE_M, PPI_SIZE)
        PX, PY = np.meshgrid(px, py)
        PR = np.sqrt(PX**2 + PY**2)
        PB = np.degrees(np.arctan2(PX, PY)) % 360
        self.ridx = np.clip(
            (PR / self.range_res_m).astype(np.int32), 0, self.n_range_bins - 1
        )
        self.bidx = (PB / 360.0 * N_BEARINGS).astype(np.int32) % N_BEARINGS
        self.mask = PR > MAX_RANGE_M

    @property
    def rate_str(self):
        return f'{self.video_rate_hz // 1_000_000} MS/s'

    @property
    def dec_str(self):
        return (f'  ·  digital ÷{self.decimate_n} → 1 MS/s'
                if self.decimate_n > 1 else '')


# ─────────────────────────────────────────────────────────────────────────────
# ADLINK PCIS-DASK driver wrapper
# ─────────────────────────────────────────────────────────────────────────────

_ADLINK_DIRS = [
    os.path.dirname(os.path.abspath(__file__)),
    r'C:\ADLINK\PCI-DASK',       r'C:\ADLINK\PCIS-DASK',
    r'C:\Program Files\ADLINK\PCI-DASK',
    r'C:\Program Files\ADLINK\PCIS-DASK',
    r'C:\Program Files (x86)\ADLINK\PCI-DASK',
    r'C:\Program Files (x86)\ADLINK\PCIS-DASK',
    r'C:\Windows\System32',      r'C:\Windows\SysWOW64',
]

def _find_dll():
    for d in _ADLINK_DIRS:
        p = os.path.join(d, 'PCI-Dask64.dll')
        if os.path.isfile(p):
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(d)
            return p
    found = ctypes.util.find_library('PCI-Dask64')
    if found:
        return found
    raise FileNotFoundError('PCI-Dask64.dll not found.')

def _load_const():
    needed = ['PCI_9812','AD_B_5_V','P9812_TRGMOD_SOFT','P9812_TRGSRC_CH0',
              'P9812_TRGSLP_POS','P9812_AD2_GT_PCI','P9812_CLKSRC_INT','ASYNCH_OP']
    pat = re.compile(
        r'^\s*#define\s+(' + '|'.join(re.escape(n) for n in needed) +
        r')\s+(0[xX][0-9a-fA-F]+|\d+)', re.MULTILINE)
    for d in _ADLINK_DIRS:
        for sub in ('', 'Include', 'include'):
            hdr = os.path.join(d, sub, 'dask.h')
            if os.path.isfile(hdr):
                out = {}
                for m in pat.finditer(open(hdr).read()):
                    v = m.group(2)
                    out[m.group(1)] = int(v, 16) if v.lower().startswith('0x') else int(v)
                return out
    return {}

_H = _load_const()
def _c(n, fb): return _H.get(n, fb)

PCI_9812          = _c('PCI_9812',          30)
AD_B_5_V          = _c('AD_B_5_V',           1)
P9812_TRGMOD_SOFT = _c('P9812_TRGMOD_SOFT',  0)
P9812_TRGSRC_CH0  = _c('P9812_TRGSRC_CH0',   0)
P9812_TRGSLP_POS  = _c('P9812_TRGSLP_POS',   0)
P9812_AD2_GT_PCI  = _c('P9812_AD2_GT_PCI',   0x0002)
P9812_CLKSRC_INT  = _c('P9812_CLKSRC_INT',   0x0000)
ASYNCH_OP         = _c('ASYNCH_OP',           1)


class DASK:
    def __init__(self):
        self._dll = ctypes.WinDLL(_find_dll())
        d = self._dll
        I16, U16, U32, F64 = (ctypes.c_int16, ctypes.c_uint16,
                               ctypes.c_uint32, ctypes.c_double)
        PU16, PU32 = ctypes.POINTER(U16), ctypes.POINTER(U32)
        d.Register_Card.argtypes              = [U16, U16];         d.Register_Card.restype  = I16
        d.AI_9812_Config.argtypes             = [I16,U16,U16,U16,U16,U16,U32]
        d.AI_9812_Config.restype              = I16
        d.AI_AsyncDblBufferMode.argtypes      = [I16, U16];         d.AI_AsyncDblBufferMode.restype  = I16
        d.AI_ContScanChannels.argtypes        = [I16, U16, U16, PU16, U32, F64, U16]
        d.AI_ContScanChannels.restype         = I16
        d.AI_AsyncDblBufferHalfReady.argtypes = [I16, PU16, PU16];  d.AI_AsyncDblBufferHalfReady.restype = I16
        d.AI_AsyncClear.argtypes              = [I16, PU32];        d.AI_AsyncClear.restype  = I16
        d.Release_Card.argtypes               = [I16];              d.Release_Card.restype   = I16

    def Register_Card(self, ct, cn):
        h = self._dll.Register_Card(ctypes.c_uint16(ct), ctypes.c_uint16(cn))
        if h < 0: raise RuntimeError(f'Register_Card error={h}')
        return h

    def AI_9812_Config(self, card, tm, ts, tsl, at, tl, pc):
        e = self._dll.AI_9812_Config(ctypes.c_int16(card),
            ctypes.c_uint16(tm), ctypes.c_uint16(ts), ctypes.c_uint16(tsl),
            ctypes.c_uint16(at), ctypes.c_uint16(tl), ctypes.c_uint32(pc))
        if e: raise RuntimeError(f'AI_9812_Config error={e}')

    def AI_AsyncDblBufferMode(self, card, enable):
        e = self._dll.AI_AsyncDblBufferMode(
            ctypes.c_int16(card), ctypes.c_uint16(1 if enable else 0))
        if e: raise RuntimeError(f'AI_AsyncDblBufferMode error={e}')

    def AI_ContScanChannels(self, card, ch, rng, buf_ptr, count, sr, synch):
        e = self._dll.AI_ContScanChannels(
            ctypes.c_int16(card), ctypes.c_uint16(ch), ctypes.c_uint16(rng),
            buf_ptr, ctypes.c_uint32(count), ctypes.c_double(sr),
            ctypes.c_uint16(synch))
        if e: raise RuntimeError(f'AI_ContScanChannels error={e}')

    def AI_AsyncDblBufferHalfReady(self, card):
        hr = ctypes.c_uint16(0); fs = ctypes.c_uint16(0)
        self._dll.AI_AsyncDblBufferHalfReady(
            ctypes.c_int16(card), ctypes.byref(hr), ctypes.byref(fs))
        return bool(hr.value), bool(fs.value)

    def AI_AsyncClear(self, card):
        cnt = ctypes.c_uint32(0)
        e = self._dll.AI_AsyncClear(ctypes.c_int16(card), ctypes.byref(cnt))
        if e: raise RuntimeError(f'AI_AsyncClear error={e}')
        return cnt.value

    def Release_Card(self, card):
        e = self._dll.Release_Card(ctypes.c_int16(card))
        if e: raise RuntimeError(f'Release_Card error={e}')


# ─────────────────────────────────────────────────────────────────────────────
# Pulse-edge helper
# ─────────────────────────────────────────────────────────────────────────────

def _rising_edges(sig, threshold, min_gap):
    above = (sig > threshold).astype(np.uint8)
    edges = np.where(np.diff(above) == 1)[0]
    if len(edges) > 1:
        keep  = np.concatenate(([True], np.diff(edges) > min_gap))
        edges = edges[keep]
    return edges


# ─────────────────────────────────────────────────────────────────────────────
# Radar processor  (polar buffer pre-allocated at MAX_RANGE_BINS)
# ─────────────────────────────────────────────────────────────────────────────

class RadarProcessor:
    """
    Thread-safe polar accumulator.

    The polar array is fixed at (N_BEARINGS × MAX_RANGE_BINS) so no
    reallocation is needed when the rate changes.  Only the first
    cfg.n_range_bins columns are used at any given rate.
    """

    def __init__(self, cfg: RateConfig):
        self._cfg    = cfg
        self._lock   = threading.Lock()
        # Pre-allocate at max possible size
        self.polar   = np.zeros((N_BEARINGS, MAX_RANGE_BINS), np.float32)

        # Bearing state (full-rate global scan index)
        self._gscan      = 0
        self._last_hd_gs = -1
        self._hd_interval = None

        # Public stats
        self.n_revs          = 0
        self.n_radials       = 0
        self.prf_hz          = 0.0
        self.cur_bearing_deg = 0.0

    # ── Called from main thread on rate change ────────────────────────────────

    def apply_new_rate(self, new_cfg: RateConfig):
        """
        Switch to a new RateConfig.  Clears the portion of the polar buffer
        beyond the new active range and resets bearing state.
        Must be called only when the acquisition thread is stopped.
        """
        with self._lock:
            self._cfg = new_cfg
            # Zero columns beyond the new active range
            self.polar[:, new_cfg.n_range_bins:] = 0.0
            # Reset bearing reference so display doesn't show stale angles
            self._gscan       = 0
            self._last_hd_gs  = -1
            self._hd_interval = None
            self.n_revs       = 0
            self.n_radials    = 0
            self.prf_hz       = 0.0

    # ── Called from acquisition thread ───────────────────────────────────────

    def process_half(self, raw_uint16: np.ndarray):
        """
        raw_uint16 : 1-D uint16 array, HALF_COUNT values (N_CH interleaved).
        Takes a local snapshot of _cfg at entry so a mid-call config change
        is safe — the previous half is finished with the old config.
        """
        cfg     = self._cfg       # atomic read (GIL); consistent throughout
        n_scans = len(raw_uint16) // N_CH
        raw     = raw_uint16[: n_scans * N_CH]

        # Voltage arrays at full VIDEO rate
        ch = {c: ((raw[c::N_CH].astype(np.float32) - ADC_MID) / ADC_MID * VRANGE)
              for c in range(N_CH)}

        # ── Decimate digital channels to 1 MHz ────────────────────────────────
        dec_hd   = ch[CH_HD]  [::cfg.decimate_n]
        dec_trig = ch[CH_TRIG][::cfg.decimate_n]

        # ── HD pulse detection ────────────────────────────────────────────────
        for hd_d in _rising_edges(dec_hd, THRESH_HD, _GAP_HD_DC):
            hd_gs = self._gscan + hd_d * cfg.decimate_n
            if self._last_hd_gs >= 0:
                self._hd_interval = hd_gs - self._last_hd_gs
                self.n_revs += 1
                with self._lock:
                    self.polar *= PERSIST
            self._last_hd_gs = hd_gs

        # ── TRIG pulse detection ──────────────────────────────────────────────
        trig_dec = _rising_edges(dec_trig, THRESH_TRIG, cfg._gap_trig_dc)
        if len(trig_dec) > 1:
            half_s = n_scans / cfg.video_rate_hz
            self.prf_hz = self.prf_hz * 0.85 + 0.15 * len(trig_dec) / max(half_s, 1e-9)

        if self._last_hd_gs < 0 or self._hd_interval is None:
            self._gscan += n_scans
            return

        # ── Build radials at full VIDEO rate ──────────────────────────────────
        new_rows: dict[int, np.ndarray] = {}
        nb = cfg.n_range_bins

        for trig_d in trig_dec:
            ti = trig_d * cfg.decimate_n          # full-rate scan index
            if ti + nb >= n_scans:
                continue                           # skip boundary-crossing radials

            ti_gs       = self._gscan + ti
            elapsed     = ti_gs - self._last_hd_gs
            bearing_deg = (elapsed % self._hd_interval) / self._hd_interval * 360.0
            b_idx       = int(bearing_deg / 360.0 * N_BEARINGS) % N_BEARINGS
            self.cur_bearing_deg = bearing_deg

            radial = ch[CH_VIDEO][ti: ti + nb].copy()
            dc     = float(radial[:min(4, nb)].mean())
            radial = np.clip(radial - dc, 0.0, None).astype(np.float32)

            if b_idx in new_rows:
                np.maximum(new_rows[b_idx], radial, out=new_rows[b_idx])
            else:
                new_rows[b_idx] = radial
            self.n_radials += 1

        if new_rows:
            with self._lock:
                for b_idx, radial in new_rows.items():
                    np.maximum(self.polar[b_idx, :nb], radial,
                               out=self.polar[b_idx, :nb])

        self._gscan += n_scans

    # ── Called from display thread ────────────────────────────────────────────

    def cart_image(self) -> np.ndarray:
        cfg = self._cfg
        with self._lock:
            p = self.polar[:, : cfg.n_range_bins].copy()
        nz  = p[p > 0]
        p99 = float(np.percentile(nz, 99)) if len(nz) > 0 else 1.0
        p   = np.clip(p / (p99 + 1e-9), 0.0, 1.0)
        img = p[cfg.bidx, cfg.ridx]
        img[cfg.mask] = 0.0
        return img.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Acquisition thread
# ─────────────────────────────────────────────────────────────────────────────

class AcquisitionThread(threading.Thread):

    def __init__(self, processor: RadarProcessor, cfg: RateConfig):
        super().__init__(daemon=True, name='PCI9812-Acq')
        self.processor = processor
        self.cfg       = cfg
        self.status    = 'starting'
        self._stop_evt = threading.Event()

        # Allocate DMA buffer sized for this specific rate config
        self._buf     = np.zeros(cfg.read_count, dtype=np.uint16)
        self._buf_ptr = self._buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        self._dask    = None
        self._card    = None

    def run(self):
        try:
            self._dask = DASK()
            self._card = self._dask.Register_Card(PCI_9812, CARD_NUM)
            self.status = 'registered'

            self._dask.AI_9812_Config(
                self._card,
                tm  = P9812_TRGMOD_SOFT,
                ts  = P9812_TRGSRC_CH0,
                tsl = P9812_TRGSLP_POS,
                at  = P9812_AD2_GT_PCI | P9812_CLKSRC_INT,
                tl  = 0x80,
                pc  = 0,
            )
            self._dask.AI_AsyncDblBufferMode(self._card, enable=True)
            self._dask.AI_ContScanChannels(
                self._card,
                ch     = N_CH - 1,
                rng    = AD_B_5_V,
                buf_ptr= self._buf_ptr,
                count  = self.cfg.read_count,
                sr     = float(self.cfg.video_rate_hz),
                synch  = ASYNCH_OP,
            )
            self.status = 'acquiring'

            sleep_s      = HALF_PERIOD_S * 0.40
            current_half = 0

            while not self._stop_evt.is_set():
                ready, f_stop = self._dask.AI_AsyncDblBufferHalfReady(self._card)
                if ready:
                    hc    = self.cfg.half_count
                    chunk = (self._buf[:hc].copy() if current_half == 0
                             else self._buf[hc:].copy())
                    current_half ^= 1
                    self.processor.process_half(chunk)
                else:
                    time.sleep(sleep_s)
                if f_stop:
                    self.status = 'card stopped'
                    break

            self.status = 'idle'

        except Exception as exc:
            self.status = f'ERROR: {exc}'
            print(f'\n[AcqThread] {exc}')
        finally:
            self._cleanup()

    def stop(self):
        self._stop_evt.set()

    def _cleanup(self):
        if self._dask and self._card is not None:
            try:
                self._dask.AI_AsyncClear(self._card)
                self._dask.Release_Card(self._card)
            except Exception:
                pass
            self._card = None


# ─────────────────────────────────────────────────────────────────────────────
# Build display  (PPI + rate slider)
# ─────────────────────────────────────────────────────────────────────────────

def build_display(init_rate_idx: int):
    fig = plt.figure(figsize=(9, 10), facecolor='black')
    fig.canvas.manager.set_window_title('PPI — PCI-9812 Live Radar')

    # PPI axes — leave room at the bottom for the slider
    ax = fig.add_axes([0.02, 0.10, 0.96, 0.87], facecolor='black')
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Radar image ────────────────────────────────────────────────────────
    ppi_im = ax.imshow(
        np.zeros((PPI_SIZE, PPI_SIZE), np.float32),
        cmap=RADAR_CMAP, vmin=0, vmax=1, origin='upper',
        extent=[-MAX_RANGE_M, MAX_RANGE_M, -MAX_RANGE_M, MAX_RANGE_M],
        interpolation='bilinear', zorder=0,
    )

    # ── Range rings ────────────────────────────────────────────────────────
    theta = np.linspace(0, 2 * np.pi, 360)
    for r_nm in np.arange(0.5, DISPLAY_RANGE_NM + 0.01, 0.5):
        r_m = r_nm * METERS_PER_NM
        ax.plot(r_m * np.sin(theta), r_m * np.cos(theta),
                color='#183018', lw=0.7, ls='--', zorder=2)
        ax.text(r_m * np.sin(np.radians(34)), r_m * np.cos(np.radians(34)),
                f'{r_m:.0f} m', color='#285028', fontsize=7,
                ha='center', va='center', zorder=3)

    ax.plot(MAX_RANGE_M * np.sin(theta), MAX_RANGE_M * np.cos(theta),
            color='#254825', lw=1.4, zorder=2)

    # ── Bearing spokes ─────────────────────────────────────────────────────
    for b_deg in range(0, 360, 30):
        bx, by = np.sin(np.radians(b_deg)), np.cos(np.radians(b_deg))
        ax.plot([0, MAX_RANGE_M * 0.97 * bx], [0, MAX_RANGE_M * 0.97 * by],
                color='#162416', lw=0.5, zorder=2)
        ax.text(MAX_RANGE_M * 1.06 * bx, MAX_RANGE_M * 1.06 * by,
                f'{b_deg}°', color='#2e4a2e', fontsize=6.5,
                ha='center', va='center', zorder=3)

    ax.plot([0, 0], [0, MAX_RANGE_M * 0.97],
            color='#35aa35', lw=1.0, ls=':', alpha=0.65, zorder=4)

    for lbl, bx, by in [('N',0,1.10),('S',0,-1.10),('E',1.10,0),('W',-1.10,0)]:
        ax.text(bx*MAX_RANGE_M, by*MAX_RANGE_M, lbl,
                color='#55b855', fontsize=11, fontweight='bold',
                ha='center', va='center', zorder=5)

    ax.plot(0, 0, 'o', color='#60d060', ms=4, zorder=6)

    sweep_ln, = ax.plot([], [], color='#c0ffc0', lw=1.2, alpha=0.85, zorder=7)

    # ── Title & status line ────────────────────────────────────────────────
    fig.suptitle('PPI  —  Furuno FAR-2xx8 / PCI-9812  (live)',
                 color='#70d070', fontsize=12, fontweight='bold', y=0.99)
    status_txt = fig.text(0.50, 0.095, 'Waiting for first HD pulse …',
                          color='#3a5a3a', fontsize=7.5, ha='center')

    # ── Rate slider ────────────────────────────────────────────────────────
    ax_sl = fig.add_axes([0.15, 0.03, 0.70, 0.030], facecolor='#0d180d')

    rate_slider = Slider(
        ax      = ax_sl,
        label   = 'Video rate',
        valmin  = 0,
        valmax  = len(RATE_OPTIONS) - 1,
        valinit = init_rate_idx,
        valstep = 1,
        color   = '#205020',
    )

    # Tick marks at each valid rate
    ax_sl.set_xticks(range(len(RATE_OPTIONS)))
    ax_sl.set_xticklabels(RATE_LABELS, color='#70c070', fontsize=8)
    ax_sl.tick_params(axis='x', length=4, color='#305030')

    # Show selected rate as the slider's value text
    rate_slider.valtext.set_text(RATE_LABELS[init_rate_idx])
    rate_slider.valtext.set_color('#a0e0a0')
    rate_slider.label.set_color('#70c070')

    # Slider annotation: hardware limit note
    fig.text(0.50, 0.005,
             f'PCI-9812 max: {max(RATE_OPTIONS)//1_000_000} MS/s  '
             f'(hardware limit)  —  '
             f'digital channels (HD, TRIG) always decimated to 1 MS/s',
             color='#2a4a2a', fontsize=6.5, ha='center')

    arts = dict(
        ppi_im=ppi_im, sweep_ln=sweep_ln,
        status_txt=status_txt, rate_slider=rate_slider,
    )
    return fig, arts


# ─────────────────────────────────────────────────────────────────────────────
# Animation updater
# ─────────────────────────────────────────────────────────────────────────────

def make_updater(proc: RadarProcessor, acq_ref: list, arts: dict,
                 pending: list):
    """
    acq_ref  : [AcquisitionThread]  — mutable single-element list so the
               closure can replace the thread on rate change.
    pending  : [bool, int]  — [change_requested, new_rate_hz]
    """
    frame_ctr = [0]

    def update(_frame):
        fc = frame_ctr[0]
        frame_ctr[0] += 1

        # ── Handle rate change request (queued by slider callback) ───────────
        if pending[0]:
            pending[0] = False
            new_hz = pending[1]
            arts['status_txt'].set_text(
                f'Restarting at {new_hz // 1_000_000} MS/s …'
            )
            # Stop current acquisition
            acq_ref[0].stop()
            acq_ref[0].join(timeout=3.0)
            # Switch config and restart
            new_cfg = RateConfig(new_hz)
            proc.apply_new_rate(new_cfg)
            new_acq = AcquisitionThread(proc, new_cfg)
            new_acq.start()
            acq_ref[0] = new_acq
            print(f'\n[Rate] → {new_hz/1e6:.0f} MS/s  '
                  f'| {new_cfg.range_res_m:.1f} m/bin  '
                  f'| {new_cfg.n_range_bins} bins  '
                  f'| buf {new_cfg.read_count * 2 / 1e6:.0f} MB')

        # ── PPI and sweep line ────────────────────────────────────────────────
        arts['ppi_im'].set_data(proc.cart_image())
        b_rad = np.radians(proc.cur_bearing_deg)
        arts['sweep_ln'].set_data(
            [0, MAX_RANGE_M * 0.97 * np.sin(b_rad)],
            [0, MAX_RANGE_M * 0.97 * np.cos(b_rad)],
        )

        # ── Status footer (every 10 frames) ──────────────────────────────────
        if fc % 10 == 0:
            cfg = proc._cfg
            arts['status_txt'].set_text(
                f'{acq_ref[0].status}  ·  '
                f'VIDEO {cfg.rate_str}{cfg.dec_str}  ·  '
                f'{cfg.range_res_m:.1f} m/bin  ·  '
                f'{cfg.n_range_bins} bins/radial  ·  '
                f'{proc.n_revs} revs  ·  {proc.n_radials:,} radials  ·  '
                f'PRF ≈ {proc.prf_hz:.0f} Hz'
            )

    return update


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('─' * 65)
    print('Real-time PPI  —  PCI-9812 + Furuno FAR-2xx8')
    print('─' * 65)
    print(f'  Hardware max  : {max(RATE_OPTIONS)//1_000_000} MS/s  '
          f'(PCI-9812 spec)')
    print(f'  Start-up rate : {DEFAULT_RATE//1_000_000} MS/s  →  '
          f'{C/(2*DEFAULT_RATE):.1f} m/bin')
    print(f'  Display range : {DISPLAY_RANGE_NM} NM  ({MAX_RANGE_M:.0f} m)')
    print(f'  Available rates: {", ".join(RATE_LABELS)}')
    print()

    init_idx  = RATE_OPTIONS.index(DEFAULT_RATE)
    init_cfg  = RateConfig(DEFAULT_RATE)
    proc      = RadarProcessor(init_cfg)
    acq       = AcquisitionThread(proc, init_cfg)
    acq_ref   = [acq]      # mutable reference for the updater closure
    pending   = [False, DEFAULT_RATE]   # [change_requested, new_rate_hz]

    print('Starting acquisition …')
    acq.start()
    time.sleep(0.5)
    print(f'  Card status   : {acq.status}')
    if 'ERROR' in acq.status:
        print('\nAcquisition failed — see error above.')
        sys.exit(1)

    print('Opening display  (move slider to change rate; close window to stop) …')
    fig, arts = build_display(init_idx)

    # ── Wire slider callback ──────────────────────────────────────────────────
    def on_slider(val):
        idx    = int(round(val))
        new_hz = RATE_OPTIONS[idx]
        arts['rate_slider'].valtext.set_text(RATE_LABELS[idx])
        if new_hz != proc._cfg.video_rate_hz:
            pending[0] = True
            pending[1] = new_hz

    arts['rate_slider'].on_changed(on_slider)

    updater = make_updater(proc, acq_ref, arts, pending)
    ani = animation.FuncAnimation(          # noqa: F841
        fig, updater, interval=40,          # ~25 fps
        blit=False, cache_frame_data=False,
    )

    try:
        plt.show()
    finally:
        print('\nStopping acquisition …')
        acq_ref[0].stop()
        acq_ref[0].join(timeout=3.0)
        print('Done.')


if __name__ == '__main__':
    main()
