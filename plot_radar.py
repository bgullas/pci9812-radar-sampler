"""
plot_radar.py  —  Real-time PPI radar from PCI-9812 + Furuno FAR-2xx8

Acquires data directly from the card in a background thread and
continuously updates a live PPI display in the main thread.

━━  HARDWARE NOTE  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  The PCI-9812 uses a single shared clock for all 4 channels — it is not
  possible to sample different channels at different hardware rates in the
  same acquisition.  To give VIDEO the best range resolution while keeping
  pulse detection on HD / TRIG reliable:

    • All 4 channels run at VIDEO_RATE_HZ  (the one setting you change).
    • HD (CH0) and TRIG (CH2) are decimated in software back to 1 MHz
      before pulse detection — their timing accuracy is unaffected.
    • Only the VIDEO (CH3) radials are processed at full VIDEO_RATE_HZ
      resolution, giving finer range bins at higher rates.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Channel wiring (Furuno FAR-2xx8, J510):
  CH0 = HD    — heading / north pulse  (1 per revolution, ~1.4 s @ 42 RPM)
  CH1 = BP    — bearing pulse          (not used here)
  CH2 = TRIG  — range trigger          (one per radial sweep, ~600-1200 Hz)
  CH3 = VIDEO — radar echo video       (amplitude vs range)

Requirements:
  Windows + ADLINK PCIS-DASK driver (PCI-Dask64.dll)
  pip install numpy matplotlib
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

if sys.platform != 'win32':
    raise EnvironmentError(
        'plot_radar.py requires Windows and the ADLINK PCIS-DASK driver.'
    )

# ─────────────────────────────────────────────────────────────────────────────
# Channel assignments
# ─────────────────────────────────────────────────────────────────────────────
CH_HD    = 0   # heading / north pulse
CH_BP    = 1   # bearing pulse  (not used — time interpolation used instead)
CH_TRIG  = 2   # range trigger
CH_VIDEO = 3   # radar echo
N_CH     = 4


# ─────────────────────────────────────────────────────────────────────────────
# ╔══════════════════════════════════════════════════════════════════╗
# ║                     USER SETTINGS                               ║
# ╚══════════════════════════════════════════════════════════════════╝
# ─────────────────────────────────────────────────────────────────────────────

# VIDEO sampling rate — the ONLY line you need to change for range resolution.
# PCI-9812 samples all 4 channels simultaneously at this clock rate.
# Digital channels (HD, TRIG) are decimated back to 1 MHz in software.
#
#   Rate (Hz)   Range res   Bins for 3 NM   Buffer / half
#   ─────────   ─────────   ─────────────   ─────────────
#   1_000_000   149.9 m          41          ~  8 MB
#   2_000_000    75.0 m          80          ~ 16 MB
#   5_000_000    30.0 m         189          ~ 40 MB
#  10_000_000    15.0 m         374          ~ 80 MB
#  20_000_000     7.5 m         745          ~160 MB
#
VIDEO_RATE_HZ = 5_000_000      # ← change this  (1_000_000 – 20_000_000)

CARD_NUM         = 0           # 0 = first PCI-9812 card
DISPLAY_RANGE_NM = 3.0         # set to match the radar's range selection
PERSIST          = 0.92        # polar buffer decay per revolution (0 = no trail, 1 = infinite)

# Pulse detection thresholds (Volts)
# Adjust if HD or TRIG pulses are missed / falsely triggered
THRESH_HD   = 0.30
THRESH_TRIG = 0.30

# ─────────────────────────────────────────────────────────────────────────────
# Derived constants  (all computed from VIDEO_RATE_HZ — do not edit)
# ─────────────────────────────────────────────────────────────────────────────
ADC_BITS  = 12
ADC_MID   = 2 ** (ADC_BITS - 1)       # 2048  (12-bit unsigned midpoint)
VRANGE    = 5.0                        # ±5 V  (AD_B_5_V)
C         = 299_792_458                # m/s

# Clamp and validate VIDEO rate
_RATE_MIN, _RATE_MAX = 1_000_000, 20_000_000
VIDEO_RATE_HZ = max(_RATE_MIN, min(_RATE_MAX, int(VIDEO_RATE_HZ)))
if VIDEO_RATE_HZ % 1_000_000 != 0:
    VIDEO_RATE_HZ = round(VIDEO_RATE_HZ / 1_000_000) * 1_000_000

# Range resolution and bins
RANGE_RES_M  = C / (2 * VIDEO_RATE_HZ)
MAX_RANGE_M  = DISPLAY_RANGE_NM * 1852
METERS_PER_NM = 1852
N_RANGE_BINS  = int(MAX_RANGE_M / RANGE_RES_M) + 4   # slight margin

# Decimation factor for digital channels (HD, TRIG) → effective 1 MHz
CTRL_RATE_HZ = 1_000_000
DECIMATE_N   = max(1, VIDEO_RATE_HZ // CTRL_RATE_HZ)

# Minimum pulse gaps in DECIMATED (1 MHz) scan space
_GAP_HD_DC   = int(0.50 * CTRL_RATE_HZ)   # 0.50 s  (antenna period ~1.4 s)
_GAP_TRIG_DC = int(5e-4 * CTRL_RATE_HZ)   # 0.50 ms (PRF ~600–1200 Hz)

# DMA double-buffer: target ~0.5 s per half at VIDEO_RATE_HZ
# READ_COUNT = total uint16 values in the full buffer (all channels × all scans)
HALF_PERIOD_S = 0.5
_half_raw     = int(HALF_PERIOD_S * VIDEO_RATE_HZ * N_CH)
HALF_COUNT    = (_half_raw // (N_CH * 2)) * (N_CH * 2)   # round to even N_CH multiple
READ_COUNT    = HALF_COUNT * 2

# Display
N_BEARINGS = 512
PPI_SIZE   = 600


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
# ADLINK PCIS-DASK driver wrapper
# ─────────────────────────────────────────────────────────────────────────────

_ADLINK_DIRS = [
    os.path.dirname(os.path.abspath(__file__)),
    r'C:\ADLINK\PCI-DASK',
    r'C:\ADLINK\PCIS-DASK',
    r'C:\Program Files\ADLINK\PCI-DASK',
    r'C:\Program Files\ADLINK\PCIS-DASK',
    r'C:\Program Files (x86)\ADLINK\PCI-DASK',
    r'C:\Program Files (x86)\ADLINK\PCIS-DASK',
    r'C:\Windows\System32',
    r'C:\Windows\SysWOW64',
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
    raise FileNotFoundError(
        'PCI-Dask64.dll not found.  '
        'Copy it next to this script or add its folder to the Windows PATH.'
    )

def _load_constants():
    needed = ['PCI_9812', 'AD_B_5_V', 'P9812_TRGMOD_SOFT', 'P9812_TRGSRC_CH0',
              'P9812_TRGSLP_POS', 'P9812_AD2_GT_PCI', 'P9812_CLKSRC_INT', 'ASYNCH_OP']
    pat = re.compile(
        r'^\s*#define\s+(' + '|'.join(re.escape(n) for n in needed) +
        r')\s+(0[xX][0-9a-fA-F]+|\d+)', re.MULTILINE,
    )
    for d in _ADLINK_DIRS:
        for sub in ('', 'Include', 'include'):
            hdr = os.path.join(d, sub, 'dask.h')
            if os.path.isfile(hdr):
                found = {}
                for m in pat.finditer(open(hdr).read()):
                    v = m.group(2)
                    found[m.group(1)] = int(v, 16) if v.lower().startswith('0x') else int(v)
                return found
    return {}

_H = _load_constants()
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
    """Minimal ctypes wrapper — AI_ContScanChannels (in-memory DMA, no file)."""

    def __init__(self):
        self._dll = ctypes.WinDLL(_find_dll())
        self._proto()

    def _proto(self):
        d = self._dll
        I16, U16, U32, F64 = (ctypes.c_int16, ctypes.c_uint16,
                               ctypes.c_uint32, ctypes.c_double)
        PU16 = ctypes.POINTER(U16)
        PU32 = ctypes.POINTER(U32)
        d.Register_Card.argtypes              = [U16, U16];        d.Register_Card.restype  = I16
        d.AI_9812_Config.argtypes             = [I16,U16,U16,U16,U16,U16,U32]
        d.AI_9812_Config.restype              = I16
        d.AI_AsyncDblBufferMode.argtypes      = [I16, U16];        d.AI_AsyncDblBufferMode.restype = I16
        d.AI_ContScanChannels.argtypes        = [I16, U16, U16, PU16, U32, F64, U16]
        d.AI_ContScanChannels.restype         = I16
        d.AI_AsyncDblBufferHalfReady.argtypes = [I16, PU16, PU16]; d.AI_AsyncDblBufferHalfReady.restype = I16
        d.AI_AsyncClear.argtypes              = [I16, PU32];       d.AI_AsyncClear.restype  = I16
        d.Release_Card.argtypes               = [I16];             d.Release_Card.restype   = I16

    def Register_Card(self, card_type, card_num):
        h = self._dll.Register_Card(ctypes.c_uint16(card_type),
                                    ctypes.c_uint16(card_num))
        if h < 0: raise RuntimeError(f'Register_Card error={h}')
        return h

    def AI_9812_Config(self, card, trig_mod, trig_src, trig_slp,
                       ad_timing, trig_level, post_count):
        e = self._dll.AI_9812_Config(
            ctypes.c_int16(card),
            ctypes.c_uint16(trig_mod), ctypes.c_uint16(trig_src),
            ctypes.c_uint16(trig_slp), ctypes.c_uint16(ad_timing),
            ctypes.c_uint16(trig_level), ctypes.c_uint32(post_count))
        if e: raise RuntimeError(f'AI_9812_Config error={e}')

    def AI_AsyncDblBufferMode(self, card, enable):
        e = self._dll.AI_AsyncDblBufferMode(
            ctypes.c_int16(card), ctypes.c_uint16(1 if enable else 0))
        if e: raise RuntimeError(f'AI_AsyncDblBufferMode error={e}')

    def AI_ContScanChannels(self, card, channel, ad_range,
                            buf_ptr, count, sample_rate, synch):
        e = self._dll.AI_ContScanChannels(
            ctypes.c_int16(card), ctypes.c_uint16(channel),
            ctypes.c_uint16(ad_range), buf_ptr,
            ctypes.c_uint32(count), ctypes.c_double(sample_rate),
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
# Polar → Cartesian lookup tables  (built once using VIDEO_RATE_HZ-derived bins)
# ─────────────────────────────────────────────────────────────────────────────
_px = np.linspace(-MAX_RANGE_M,  MAX_RANGE_M, PPI_SIZE)
_py = np.linspace( MAX_RANGE_M, -MAX_RANGE_M, PPI_SIZE)   # y-flip for imshow
_PX, _PY = np.meshgrid(_px, _py)
_PR  = np.sqrt(_PX**2 + _PY**2)
_PB  = np.degrees(np.arctan2(_PX, _PY)) % 360             # 0=N, clockwise
_RIDX = np.clip((_PR / RANGE_RES_M).astype(np.int32), 0, N_RANGE_BINS - 1)
_BIDX = (_PB / 360.0 * N_BEARINGS).astype(np.int32) % N_BEARINGS
_MASK = _PR > MAX_RANGE_M


# ─────────────────────────────────────────────────────────────────────────────
# Pulse-edge detection helper
# ─────────────────────────────────────────────────────────────────────────────

def _rising_edges(sig, threshold, min_gap):
    """Indices of upward threshold crossings with minimum spacing enforced."""
    above = (sig > threshold).astype(np.uint8)
    edges = np.where(np.diff(above) == 1)[0]
    if len(edges) > 1:
        keep  = np.concatenate(([True], np.diff(edges) > min_gap))
        edges = edges[keep]
    return edges


# ─────────────────────────────────────────────────────────────────────────────
# Radar data processor
# ─────────────────────────────────────────────────────────────────────────────

class RadarProcessor:
    """
    Thread-safe polar accumulator.

    Decimation strategy
    ───────────────────
    All 4 channels arrive at VIDEO_RATE_HZ.  Before pulse detection:

      HD  (CH0) : decimated by DECIMATE_N  → effective 1 MHz
      TRIG(CH2) : decimated by DECIMATE_N  → effective 1 MHz
      VIDEO(CH3): used at FULL VIDEO_RATE_HZ for maximum range resolution

    Bearing is tracked in full-rate scan counts so the two rates stay
    consistent: decimated-space indices are multiplied by DECIMATE_N before
    being stored in global-scan variables.
    """

    def __init__(self):
        self.polar   = np.zeros((N_BEARINGS, N_RANGE_BINS), np.float32)
        self._lock   = threading.Lock()

        # Full-rate global scan counter and bearing state
        self._gscan       = 0      # full-rate scans processed so far
        self._last_hd_gs  = -1     # global scan index of last HD rising edge
        self._hd_interval = None   # full-rate scans between last two HD pulses

        # Stats exposed to display thread
        self.n_revs          = 0
        self.n_radials       = 0
        self.prf_hz          = 0.0
        self.cur_bearing_deg = 0.0
        self.video_rate_hz   = VIDEO_RATE_HZ
        self.range_res_m     = RANGE_RES_M
        self.n_range_bins    = N_RANGE_BINS
        self.decimate_n      = DECIMATE_N

    # ── Called from acquisition thread ───────────────────────────────────────

    def process_half(self, raw_uint16):
        """
        raw_uint16 : 1-D uint16 array, HALF_COUNT values,
                     N_CH channels interleaved at VIDEO_RATE_HZ.
        """
        n_scans = len(raw_uint16) // N_CH
        raw     = raw_uint16[: n_scans * N_CH]

        # Convert to volts for every channel (full VIDEO_RATE_HZ resolution)
        ch = {}
        for c in range(N_CH):
            ch[c] = ((raw[c::N_CH].astype(np.float32) - ADC_MID)
                     / ADC_MID * VRANGE)

        # ── Decimate digital channels to 1 MHz ───────────────────────────────
        dec_hd   = ch[CH_HD]  [::DECIMATE_N]   # length ≈ n_scans / DECIMATE_N
        dec_trig = ch[CH_TRIG][::DECIMATE_N]

        # ── HD pulse detection (in decimated space) ───────────────────────────
        hd_dec = _rising_edges(dec_hd, THRESH_HD, _GAP_HD_DC)
        for hd_d in hd_dec:
            # Convert decimated index → full-rate global scan
            hd_gs = self._gscan + hd_d * DECIMATE_N
            if self._last_hd_gs >= 0:
                self._hd_interval = hd_gs - self._last_hd_gs
                self.n_revs += 1
                with self._lock:
                    self.polar *= PERSIST
            self._last_hd_gs = hd_gs

        # ── TRIG pulse detection (in decimated space) ─────────────────────────
        trig_dec = _rising_edges(dec_trig, THRESH_TRIG, _GAP_TRIG_DC)

        # PRF estimate (Hz)
        if len(trig_dec) > 1:
            half_s = n_scans / VIDEO_RATE_HZ
            self.prf_hz = (self.prf_hz * 0.85
                           + 0.15 * len(trig_dec) / max(half_s, 1e-9))

        # Skip radial building until bearing reference is established
        if self._last_hd_gs < 0 or self._hd_interval is None:
            self._gscan += n_scans
            return

        # ── Build radials at full VIDEO_RATE_HZ resolution ────────────────────
        new_rows: dict[int, np.ndarray] = {}

        for trig_d in trig_dec:
            # Full-rate scan index of this trigger
            ti_full = trig_d * DECIMATE_N

            # Skip if VIDEO radial would run past end of buffer
            if ti_full + N_RANGE_BINS >= n_scans:
                continue

            # Bearing: linear interpolation between HD pulses (full-rate scans)
            ti_gs        = self._gscan + ti_full
            elapsed      = ti_gs - self._last_hd_gs
            bearing_deg  = (elapsed % self._hd_interval) / self._hd_interval * 360.0
            b_idx        = int(bearing_deg / 360.0 * N_BEARINGS) % N_BEARINGS
            self.cur_bearing_deg = bearing_deg

            # Extract VIDEO at full rate and rectify (remove DC, clip ≥ 0)
            radial = ch[CH_VIDEO][ti_full: ti_full + N_RANGE_BINS].copy()
            dc     = float(radial[:min(4, N_RANGE_BINS)].mean())
            radial = np.clip(radial - dc, 0.0, None).astype(np.float32)

            # Max-hold within this chunk (multiple triggers may share b_idx)
            if b_idx in new_rows:
                np.maximum(new_rows[b_idx], radial, out=new_rows[b_idx])
            else:
                new_rows[b_idx] = radial

            self.n_radials += 1

        # Flush new radials into the shared polar buffer
        if new_rows:
            with self._lock:
                for b_idx, radial in new_rows.items():
                    np.maximum(self.polar[b_idx], radial, out=self.polar[b_idx])

        self._gscan += n_scans

    # ── Called from display thread ────────────────────────────────────────────

    def cart_image(self):
        """PPI_SIZE × PPI_SIZE float32 image, normalised [0, 1]. Thread-safe."""
        with self._lock:
            p = self.polar.copy()
        nz  = p[p > 0]
        p99 = float(np.percentile(nz, 99)) if len(nz) > 0 else 1.0
        p   = np.clip(p / (p99 + 1e-9), 0.0, 1.0)
        img = p[_BIDX, _RIDX]
        img[_MASK] = 0.0
        return img.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Acquisition thread
# ─────────────────────────────────────────────────────────────────────────────

class AcquisitionThread(threading.Thread):

    def __init__(self, processor: RadarProcessor):
        super().__init__(daemon=True, name='PCI9812-Acq')
        self.processor = processor
        self.status    = 'starting'
        self._stop_evt = threading.Event()

        # DMA buffer — stays alive for the lifetime of this object
        self._buf     = np.zeros(READ_COUNT, dtype=np.uint16)
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
                trig_mod   = P9812_TRGMOD_SOFT,       # continuous / free-running
                trig_src   = P9812_TRGSRC_CH0,
                trig_slp   = P9812_TRGSLP_POS,
                ad_timing  = P9812_AD2_GT_PCI | P9812_CLKSRC_INT,
                trig_level = 0x80,
                post_count = 0,
            )
            self._dask.AI_AsyncDblBufferMode(self._card, enable=True)
            self._dask.AI_ContScanChannels(
                self._card,
                channel     = N_CH - 1,               # scan CH0–CH3
                ad_range    = AD_B_5_V,
                buf_ptr     = self._buf_ptr,
                count       = READ_COUNT,
                sample_rate = float(VIDEO_RATE_HZ),    # ALL channels at VIDEO rate
                synch       = ASYNCH_OP,
            )
            self.status = 'acquiring'

            sleep_s      = HALF_PERIOD_S * 0.40        # yield CPU between polls
            current_half = 0

            while not self._stop_evt.is_set():
                ready, f_stop = self._dask.AI_AsyncDblBufferHalfReady(self._card)
                if ready:
                    chunk = (self._buf[:HALF_COUNT].copy() if current_half == 0
                             else self._buf[HALF_COUNT:].copy())
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
# Display
# ─────────────────────────────────────────────────────────────────────────────

def build_display():
    fig, ax = plt.subplots(figsize=(9, 9.5), facecolor='black')
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.axis('off')

    # Radar image
    ppi_im = ax.imshow(
        np.zeros((PPI_SIZE, PPI_SIZE), np.float32),
        cmap=RADAR_CMAP, vmin=0, vmax=1, origin='upper',
        extent=[-MAX_RANGE_M, MAX_RANGE_M, -MAX_RANGE_M, MAX_RANGE_M],
        interpolation='bilinear', zorder=0,
    )

    # Range rings
    theta = np.linspace(0, 2 * np.pi, 360)
    for r_nm in np.arange(0.5, DISPLAY_RANGE_NM + 0.01, 0.5):
        r_m = r_nm * METERS_PER_NM
        ax.plot(r_m * np.sin(theta), r_m * np.cos(theta),
                color='#183018', lw=0.7, ls='--', zorder=2)
        ax.text(r_m * np.sin(np.radians(34)),
                r_m * np.cos(np.radians(34)),
                f'{r_m:.0f} m', color='#285028', fontsize=7,
                ha='center', va='center', zorder=3)

    # Outer ring
    ax.plot(MAX_RANGE_M * np.sin(theta), MAX_RANGE_M * np.cos(theta),
            color='#254825', lw=1.4, zorder=2)

    # Bearing spokes every 30°
    for b_deg in range(0, 360, 30):
        bx = np.sin(np.radians(b_deg))
        by = np.cos(np.radians(b_deg))
        ax.plot([0, MAX_RANGE_M * 0.97 * bx],
                [0, MAX_RANGE_M * 0.97 * by],
                color='#162416', lw=0.5, zorder=2)
        ax.text(MAX_RANGE_M * 1.06 * bx, MAX_RANGE_M * 1.06 * by,
                f'{b_deg}°', color='#2e4a2e', fontsize=6.5,
                ha='center', va='center', zorder=3)

    # North indicator
    ax.plot([0, 0], [0, MAX_RANGE_M * 0.97],
            color='#35aa35', lw=1.0, ls=':', alpha=0.65, zorder=4)

    # Cardinal labels
    for lbl, bx, by in [('N', 0, 1.10), ('S', 0, -1.10),
                         ('E', 1.10, 0), ('W', -1.10, 0)]:
        ax.text(bx * MAX_RANGE_M, by * MAX_RANGE_M, lbl,
                color='#55b855', fontsize=11, fontweight='bold',
                ha='center', va='center', zorder=5)

    # Centre dot
    ax.plot(0, 0, 'o', color='#60d060', ms=4, zorder=6)

    # Sweep line
    sweep_ln, = ax.plot([], [], color='#c0ffc0', lw=1.2,
                        alpha=0.85, zorder=7)

    # Title
    fig.suptitle('PPI  —  Furuno FAR-2xx8 / PCI-9812  (live)',
                 color='#70d070', fontsize=12, fontweight='bold', y=0.98)

    # Status / info footer
    status_txt = fig.text(
        0.50, 0.012,
        'Waiting for first HD pulse …',
        color='#3a5a3a', fontsize=8, ha='center',
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig, dict(ppi_im=ppi_im, sweep_ln=sweep_ln, status_txt=status_txt)


def make_updater(acq: AcquisitionThread, proc: RadarProcessor, arts: dict):
    frame_ctr = [0]

    def update(_frame):
        fc = frame_ctr[0]
        frame_ctr[0] += 1

        arts['ppi_im'].set_data(proc.cart_image())

        b_rad = np.radians(proc.cur_bearing_deg)
        arts['sweep_ln'].set_data(
            [0, MAX_RANGE_M * 0.97 * np.sin(b_rad)],
            [0, MAX_RANGE_M * 0.97 * np.cos(b_rad)],
        )

        if fc % 10 == 0:
            rate_str  = (f'{proc.video_rate_hz/1e6:.0f} MS/s'
                         if proc.video_rate_hz >= 1_000_000
                         else f'{proc.video_rate_hz/1e3:.0f} kS/s')
            dec_str   = (f'  ·  digital ÷{proc.decimate_n} → 1 MS/s'
                         if proc.decimate_n > 1 else '')
            arts['status_txt'].set_text(
                f'{acq.status}  ·  VIDEO {rate_str}{dec_str}  ·  '
                f'{proc.range_res_m:.1f} m/bin  ·  '
                f'{proc.n_range_bins} bins/radial  ·  '
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
    print(f'  VIDEO rate   : {VIDEO_RATE_HZ/1e6:.0f} MS/s  →  '
          f'{RANGE_RES_M:.1f} m/bin  ·  {N_RANGE_BINS} bins per radial')
    print(f'  Digital dec. : ÷{DECIMATE_N}  →  effective 1 MS/s for HD / TRIG'
          if DECIMATE_N > 1 else
          f'  Digital dec. : none  (VIDEO_RATE = CTRL_RATE = 1 MS/s)')
    print(f'  Buffer       : {READ_COUNT:,} uint16  '
          f'({READ_COUNT * 2 / 1e6:.0f} MB)  —  '
          f'half fires every ~{HALF_PERIOD_S:.1f} s')
    print(f'  Display      : {DISPLAY_RANGE_NM:.1f} NM  '
          f'({DISPLAY_RANGE_NM * METERS_PER_NM:.0f} m)  ·  '
          f'persistence {PERSIST}')
    print()
    print('Starting acquisition …')

    proc = RadarProcessor()
    acq  = AcquisitionThread(proc)
    acq.start()

    time.sleep(0.5)   # let the card initialise
    print(f'  Card status  : {acq.status}')
    if 'ERROR' in acq.status:
        print('\nAcquisition failed — see error above.')
        sys.exit(1)

    print('Opening display  (close window to stop) …')
    fig, arts = build_display()
    updater   = make_updater(acq, proc, arts)

    ani = animation.FuncAnimation(          # noqa: F841
        fig, updater, interval=40,          # ~25 fps
        blit=False, cache_frame_data=False,
    )
    try:
        plt.show()
    finally:
        print('\nStopping acquisition …')
        acq.stop()
        acq.join(timeout=3.0)
        print('Done.')


if __name__ == '__main__':
    main()
