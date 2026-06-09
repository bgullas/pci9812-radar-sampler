"""
ADLINK PCI-9812 Radar Data Sampler
Uses the ADLINK PCIS-DASK driver (PCI-DASK.dll) on Windows.

Translated from the working C reference program — identical API calls:
  Register_Card → AI_9812_Config → AI_AsyncDblBufferMode
  → AI_ContScanChannelsToFile → poll AI_AsyncDblBufferHalfReady
  → AI_AsyncClear → Release_Card

Requirements:
  pip install numpy matplotlib
  ADLINK PCIS-DASK driver installed (provides PCI-DASK.dll)
"""

import os
import sys
import ctypes
import ctypes.util
import time
import numpy as np
import matplotlib.pyplot as plt

if sys.platform != 'win32':
    raise EnvironmentError('This script must run on Windows (PCI-DASK.dll required)')


_ADLINK_DIRS = [
    os.path.dirname(os.path.abspath(__file__)),
    r'C:\ADLINK\PCI-DASK',
    r'C:\ADLINK\PCIS-DASK',
    r'C:\Program Files\ADLINK\PCI-DASK',
    r'C:\Program Files (x86)\ADLINK\PCI-DASK',
    r'C:\Program Files\ADLINK\PCIS-DASK',
    r'C:\Program Files (x86)\ADLINK\PCIS-DASK',
    r'C:\Windows\System32',
    r'C:\Windows\SysWOW64',
]


def _find_dask_dll():
    """Locate PCI-Dask64.dll, searching the script folder first."""
    dll_name = 'PCI-Dask64.dll'
    for d in _ADLINK_DIRS:
        candidate = os.path.join(d, dll_name)
        if os.path.isfile(candidate):
            print(f'Found {dll_name} at: {candidate}')
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(d)
            return candidate
    found = ctypes.util.find_library('PCI-Dask64')
    if found:
        print(f'Found via system PATH: {found}')
        return found
    raise FileNotFoundError(
        f'{dll_name} not found.\n'
        'Copy PCI-Dask64.dll into the same folder as this script, or add its '
        'location to the Windows PATH environment variable.'
    )


def _load_dask_constants():
    """
    Parse all required #define values from dask.h.
    Returns a dict of name -> int.  Prints each value found.
    """
    import re
    needed = [
        'PCI_9812',
        'AD_B_5_V', 'AD_B_1_V',
        'P9812_TRGMOD_SOFT', 'P9812_TRGMOD_POST', 'P9812_TRGMOD_PRE',
        'P9812_TRGMOD_MIDL', 'P9812_TRGMOD_DELY',
        'P9812_TRGSRC_CH0', 'P9812_TRGSRC_CH1', 'P9812_TRGSRC_CH2',
        'P9812_TRGSRC_CH3', 'P9812_TRGSRC_EXT',
        'P9812_TRGSLP_POS', 'P9812_TRGSLP_NEG',
        'P9812_CLKSRC_INT', 'P9812_CLKSRC_SIN', 'P9812_CLKSRC_SQR',
        'P9812_AD2_GT_PCI',
        'SYNCH_OP', 'ASYNCH_OP',
    ]
    pattern = re.compile(
        r'^\s*#define\s+(' + '|'.join(re.escape(n) for n in needed) + r')\s+(0[xX][0-9a-fA-F]+|\d+)',
        re.MULTILINE,
    )

    for d in _ADLINK_DIRS:
        for subdir in ('', 'Include', 'include'):
            header = os.path.join(d, subdir, 'dask.h')
            if os.path.isfile(header):
                print(f'Loading constants from: {header}')
                text  = open(header).read()
                found = {}
                for m in pattern.finditer(text):
                    name, val = m.group(1), m.group(2)
                    found[name] = int(val, 16) if val.lower().startswith('0x') else int(val)
                for name in needed:
                    if name in found:
                        print(f'  {name} = {found[name]}')
                    else:
                        print(f'  WARNING: {name} not found in header')
                return found

    print('WARNING: dask.h not found — using built-in fallback constants (may be wrong).')
    return {}


_H = _load_dask_constants()

def _c(name, fallback):
    if name not in _H:
        print(f'  WARNING: using fallback {name}={fallback}')
    return _H.get(name, fallback)


# ---------------------------------------------------------------------------
# Constants — sourced from dask.h (fallbacks shown if header missing)
# ---------------------------------------------------------------------------

PCI_9812          = _c('PCI_9812',          30)

AD_B_5_V          = _c('AD_B_5_V',           1)
AD_B_1_V          = _c('AD_B_1_V',           3)

P9812_TRGMOD_SOFT = _c('P9812_TRGMOD_SOFT',  0)
P9812_TRGMOD_POST = _c('P9812_TRGMOD_POST',  1)
P9812_TRGMOD_PRE  = _c('P9812_TRGMOD_PRE',   2)
P9812_TRGMOD_MIDL = _c('P9812_TRGMOD_MIDL',  3)
P9812_TRGMOD_DELY = _c('P9812_TRGMOD_DELY',  4)

P9812_TRGSRC_CH0  = _c('P9812_TRGSRC_CH0',   0)
P9812_TRGSRC_CH1  = _c('P9812_TRGSRC_CH1',   1)
P9812_TRGSRC_CH2  = _c('P9812_TRGSRC_CH2',   2)
P9812_TRGSRC_CH3  = _c('P9812_TRGSRC_CH3',   3)
P9812_TRGSRC_EXT  = _c('P9812_TRGSRC_EXT',   4)

P9812_TRGSLP_POS  = _c('P9812_TRGSLP_POS',   0)
P9812_TRGSLP_NEG  = _c('P9812_TRGSLP_NEG',   1)

P9812_CLKSRC_INT  = _c('P9812_CLKSRC_INT',   0x0000)
P9812_CLKSRC_SIN  = _c('P9812_CLKSRC_SIN',   0x0004)
P9812_CLKSRC_SQR  = _c('P9812_CLKSRC_SQR',   0x0008)
P9812_AD2_GT_PCI  = _c('P9812_AD2_GT_PCI',   0x0002)

SYNCH_OP          = _c('SYNCH_OP',           0)
ASYNCH_OP         = _c('ASYNCH_OP',          1)

ADC_MID           = 2048   # 12-bit midpoint for voltage conversion


# ---------------------------------------------------------------------------
# DASK DLL wrapper
# ---------------------------------------------------------------------------

class DASK:
    """ctypes wrapper around PCI-DASK.dll — mirrors the C API exactly."""

    def __init__(self):
        dll_path = _find_dask_dll()
        try:
            self._dll = ctypes.WinDLL(dll_path)
        except OSError as exc:
            raise RuntimeError(f'Failed to load DLL: {dll_path}') from exc
        self._set_prototypes()

    def Register_Card(self, card_type, card_num):
        handle = self._dll.Register_Card(
            ctypes.c_uint16(card_type),
            ctypes.c_uint16(card_num),
        )
        if handle < 0:
            raise RuntimeError(f'Register_Card error={handle}')
        return handle

    def AI_9812_Config(self, card, trig_mod, trig_src, trig_slp,
                       ad_timing, trig_level, post_count):
        err = self._dll.AI_9812_Config(
            ctypes.c_int16(card),
            ctypes.c_uint16(trig_mod),
            ctypes.c_uint16(trig_src),
            ctypes.c_uint16(trig_slp),
            ctypes.c_uint16(ad_timing),
            ctypes.c_uint16(trig_level),
            ctypes.c_uint32(post_count),
        )
        self._check(err, 'AI_9812_Config')

    def AI_AsyncDblBufferMode(self, card, enable):
        err = self._dll.AI_AsyncDblBufferMode(
            ctypes.c_int16(card),
            ctypes.c_uint16(1 if enable else 0),
        )
        self._check(err, 'AI_AsyncDblBufferMode')

    def AI_ContScanChannelsToFile(self, card, channel, ad_range,
                                   file_name, count, sample_rate, synch):
        if isinstance(file_name, str):
            file_name = file_name.encode('ascii')
        err = self._dll.AI_ContScanChannelsToFile(
            ctypes.c_int16(card),
            ctypes.c_uint16(channel),
            ctypes.c_uint16(ad_range),
            ctypes.c_char_p(file_name),
            ctypes.c_uint32(count),
            ctypes.c_double(sample_rate),
            ctypes.c_uint16(synch),
        )
        self._check(err, 'AI_ContScanChannelsToFile')

    def AI_AsyncDblBufferHalfReady(self, card):
        half_ready = ctypes.c_uint16(0)
        f_stop     = ctypes.c_uint16(0)
        self._dll.AI_AsyncDblBufferHalfReady(
            ctypes.c_int16(card),
            ctypes.byref(half_ready),
            ctypes.byref(f_stop),
        )
        return bool(half_ready.value), bool(f_stop.value)

    def AI_AsyncDblBufferTransfer(self, card, buf=None):
        ptr = ctypes.cast(buf, ctypes.c_void_p) if buf is not None else None
        self._dll.AI_AsyncDblBufferTransfer(ctypes.c_int16(card), ptr)

    def AI_AsyncClear(self, card):
        count = ctypes.c_uint32(0)
        err   = self._dll.AI_AsyncClear(
            ctypes.c_int16(card), ctypes.byref(count)
        )
        self._check(err, 'AI_AsyncClear')
        return count.value

    def Release_Card(self, card):
        err = self._dll.Release_Card(ctypes.c_int16(card))
        self._check(err, 'Release_Card')

    def _set_prototypes(self):
        d   = self._dll
        I16 = ctypes.c_int16
        U16 = ctypes.c_uint16
        U32 = ctypes.c_uint32
        F64 = ctypes.c_double

        d.Register_Card.argtypes              = [U16, U16]
        d.Register_Card.restype               = I16
        d.AI_9812_Config.argtypes             = [I16, U16, U16, U16, U16, U16, U32]
        d.AI_9812_Config.restype              = I16
        d.AI_AsyncDblBufferMode.argtypes      = [I16, U16]
        d.AI_AsyncDblBufferMode.restype       = I16
        d.AI_ContScanChannelsToFile.argtypes  = [I16, U16, U16, ctypes.c_char_p, U32, F64, U16]
        d.AI_ContScanChannelsToFile.restype   = I16
        d.AI_AsyncDblBufferHalfReady.argtypes = [I16, ctypes.POINTER(U16), ctypes.POINTER(U16)]
        d.AI_AsyncDblBufferHalfReady.restype  = I16
        d.AI_AsyncDblBufferTransfer.argtypes  = [I16, ctypes.c_void_p]
        d.AI_AsyncDblBufferTransfer.restype   = I16
        d.AI_AsyncClear.argtypes              = [I16, ctypes.POINTER(U32)]
        d.AI_AsyncClear.restype               = I16
        d.Release_Card.argtypes               = [I16]
        d.Release_Card.restype                = I16

    @staticmethod
    def _check(err, fname):
        if err != 0:
            raise RuntimeError(f'{fname} error={err}')


# ---------------------------------------------------------------------------
# Acquisition — identical logic to the C program
# ---------------------------------------------------------------------------

def acquire(channel, ad_range, file_name, read_count, sample_rate,
            card_num=0, duration_s=5.0):
    """
    Acquire radar data to file, then return a dict of voltage arrays.

    channel     : last channel index (e.g. 3 → scans CH0–CH3)
    ad_range    : AD_B_5_V or AD_B_1_V
    file_name   : output base name (driver appends .dat)
    read_count  : circular buffer size in samples
    sample_rate : samples/second per channel
    duration_s  : how long to run (replaces kbhit() from C)
    """
    n_ch   = channel + 1
    vrange = 5.0 if ad_range == AD_B_5_V else 1.0

    print(f'PCI-9812  CH0–CH{channel}  {sample_rate:.0f} S/s  '
          f'buffer={read_count}  → {file_name}.dat')

    dask = DASK()
    card = dask.Register_Card(PCI_9812, card_num)

    dask.AI_9812_Config(
        card,
        trig_mod   = P9812_TRGMOD_SOFT,
        trig_src   = P9812_TRGSRC_CH0,
        trig_slp   = P9812_TRGSLP_POS,
        ad_timing  = P9812_AD2_GT_PCI | P9812_CLKSRC_INT,
        trig_level = 0x80,
        post_count = 0,
    )
    dask.AI_AsyncDblBufferMode(card, enable=True)
    dask.AI_ContScanChannelsToFile(
        card, channel, ad_range, file_name,
        read_count, sample_rate, ASYNCH_OP,
    )

    count      = 0
    half_count = read_count // 2
    deadline   = time.monotonic() + duration_s
    # half-buffer period in seconds: used to sleep just under one half-buffer
    # to avoid busy-spinning while the DMA fills (saves ~100 % CPU)
    half_period_s = (read_count // 2) / (sample_rate * (channel + 1))
    sleep_s = max(0.005, half_period_s * 0.45)

    print(f'Acquiring for {duration_s}s  '
          f'(half-buffer fires every ~{half_period_s*1000:.0f} ms) …')
    while time.monotonic() < deadline:
        half_ready, f_stop = dask.AI_AsyncDblBufferHalfReady(card)
        if half_ready:
            dask.AI_AsyncDblBufferTransfer(card, buf=None)
            count += half_count
            elapsed = duration_s - (deadline - time.monotonic())
            print(f'\r  {count:>12,} samples  |  {elapsed:.1f} / {duration_s:.0f} s',
                  end='', flush=True)
        else:
            time.sleep(sleep_s)   # yield CPU until next half-buffer is ready
        if f_stop:
            break

    total = dask.AI_AsyncClear(card)
    dask.Release_Card(card)
    print(f'\n{count} samples written to {file_name}.dat  (total={total})')

    # Read back and convert to volts
    raw    = np.fromfile(f'{file_name}.dat', dtype=np.int16)
    result = {}
    for ch in range(n_ch):
        raw_ch     = raw[ch::n_ch].astype(np.float32)
        result[ch] = (raw_ch - ADC_MID) / ADC_MID * vrange
    return result, sample_rate


# ---------------------------------------------------------------------------
# Radar channel roles (Furuno FAR-2xx8, port J510)
# ---------------------------------------------------------------------------
# CH0 — VIDEO  : radar echo video,  4 Vp-p coax → ÷6 divider → ~0.67 Vp-p
# CH1 — TRIG   : range trigger pulse, 0-12 V → ÷12 divider → ~1 V, 5-15 µs wide
# CH2 — BP     : bearing pulse, 0-12 V → ÷12 divider → ~1 V, one pulse per 0.18°
# CH3 — HD     : heading pulse, 0-12 V → ÷12 divider → ~1 V, one pulse per revolution
#
# All digital lines (TRIG / BP / HD) pass through an 11 kΩ / 1 kΩ voltage
# divider before the card input to step 12 V down to ≈ 1 V.

CH_LABELS = {
    0: 'VIDEO  (radar echo)',
    1: 'TRIG   (range trigger)',
    2: 'BP     (bearing pulse)',
    3: 'HD     (heading / north)',
}

# ---------------------------------------------------------------------------
# Trigger-pulse detection (CH1 — TRIG)
# ---------------------------------------------------------------------------

def detect_triggers(trig_ch, sample_rate, threshold_v=0.30, min_gap_us=500):
    """
    Find rising-edge positions of radar range-trigger pulses in CH1.

    Returns
    -------
    trig_samples : np.ndarray  (int64)  — sample indices of rising edges
    prf_hz       : float               — estimated pulse-repetition frequency
    """
    above     = (trig_ch > threshold_v).astype(np.uint8)
    edges     = np.where(np.diff(above) == 1)[0]           # rising edges
    min_gap   = int(min_gap_us * 1e-6 * sample_rate)
    # Remove spurious edges closer than min_gap
    if len(edges) > 1:
        keep = np.concatenate(([True], np.diff(edges) > min_gap))
        edges = edges[keep]
    prf = float(sample_rate / np.median(np.diff(edges))) if len(edges) > 1 else 0.0
    return edges, prf


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

_OVERVIEW_DECIMATE = 200   # plot every Nth sample in the 15-s overview
_ZOOM_MS           = 50    # milliseconds shown in the zoom panel

def plot_radar_signals(ch_data, sample_rate, vrange, duration_s):
    """
    Three-figure radar signal display:
      Fig 1 — Overview  : full 15-s, all 4 channels, decimated for speed
      Fig 2 — Zoom      : first _ZOOM_MS ms, all 4 channels, full resolution
      Fig 3 — FFT       : VIDEO channel (CH0) frequency spectrum
    """
    n_ch   = len(ch_data)
    dt     = 1.0 / sample_rate
    total  = len(ch_data[0])

    # Detect trigger pulses for annotation
    if 1 in ch_data:
        trig_idx, prf = detect_triggers(ch_data[1], sample_rate)
        prf_str = f'PRF ≈ {prf:.0f} Hz' if prf > 0 else 'PRF: n/a'
    else:
        trig_idx, prf_str = np.array([]), ''

    # ── Figure 1 : Overview ──────────────────────────────────────────────────
    fig1, axes1 = plt.subplots(n_ch, 1, figsize=(15, 2.6 * n_ch),
                               sharex=True, facecolor='#0a0a14')
    fig1.suptitle(
        f'PCI-9812  Radar Signals  —  Overview  '
        f'({sample_rate/1e6:.1f} MS/s  ·  {duration_s:.0f} s  ·  {prf_str})',
        fontsize=11, color='#d0e8d0',
    )
    if n_ch == 1:
        axes1 = [axes1]

    t_ov = np.arange(0, total, _OVERVIEW_DECIMATE) * dt   # decimated time axis (s)

    for ch, ax in enumerate(axes1):
        ax.set_facecolor('#04060e')
        v_dec = ch_data[ch][::_OVERVIEW_DECIMATE]
        ax.plot(t_ov, v_dec, lw=0.5, color='#40c060' if ch == 0 else '#60a0ff')
        ax.set_ylabel(CH_LABELS.get(ch, f'CH{ch}') + '\n(V)',
                      color='#80b080', fontsize=7.5)
        ax.set_ylim(-vrange * 1.05, vrange * 1.05)
        ax.axhline(0, color='#1a2a1a', lw=0.4, ls='--')
        ax.tick_params(colors='#304050', labelsize=7)
        ax.spines[:].set_color('#1a2a2a')
        ax.grid(True, alpha=0.15, color='#304050')

        # Mark heading pulses (CH3) on all channels as vertical bands
        if ch == 3 and len(trig_idx) > 0:
            for ti in trig_idx[::max(1, len(trig_idx) // 200)]:
                ax.axvline(ti * dt, color='#ff8030', lw=0.4, alpha=0.35)

    axes1[-1].set_xlabel('Time (s)', color='#5080a0', fontsize=8)
    # Mark approximate antenna revolution period
    rev_period = 60.0 / 42          # 42 RPM → 1.43 s
    for rev in np.arange(rev_period, duration_s, rev_period):
        for ax in axes1:
            ax.axvline(rev, color='#604020', lw=0.6, ls=':', alpha=0.6)
    axes1[0].text(rev_period / 2, vrange * 0.88, '← 1 rev →',
                  color='#806040', fontsize=7, ha='center')

    plt.tight_layout()

    # ── Figure 2 : Zoom ───────────────────────────────────────────────────────
    zoom_n  = min(int(_ZOOM_MS * 1e-3 * sample_rate), total)
    t_zoom  = np.arange(zoom_n) * dt * 1e3            # ms

    fig2, axes2 = plt.subplots(n_ch, 1, figsize=(15, 2.6 * n_ch),
                               sharex=True, facecolor='#0a0a14')
    fig2.suptitle(
        f'PCI-9812  Radar Signals  —  First {_ZOOM_MS} ms  '
        f'(range res. = {299792458 / (2*sample_rate):.1f} m/sample)',
        fontsize=11, color='#d0e8d0',
    )
    if n_ch == 1:
        axes2 = [axes2]

    for ch, ax in enumerate(axes2):
        ax.set_facecolor('#04060e')
        ax.plot(t_zoom, ch_data[ch][:zoom_n],
                lw=0.7, color='#40c060' if ch == 0 else '#60a0ff')
        ax.set_ylabel(CH_LABELS.get(ch, f'CH{ch}') + '\n(V)',
                      color='#80b080', fontsize=7.5)
        ax.set_ylim(-vrange * 1.05, vrange * 1.05)
        ax.axhline(0, color='#1a2a1a', lw=0.4, ls='--')
        ax.tick_params(colors='#304050', labelsize=7)
        ax.spines[:].set_color('#1a2a2a')
        ax.grid(True, alpha=0.15, color='#304050')

        # Mark TRIG pulses in the zoom window
        for ti in trig_idx[trig_idx < zoom_n]:
            ax.axvline(ti * dt * 1e3, color='#ff4040', lw=0.8,
                       alpha=0.55, zorder=5)

    axes2[-1].set_xlabel('Time (ms)', color='#5080a0', fontsize=8)

    # Range-calibration ticks on VIDEO (CH0) x-axis
    c      = 299_792_458
    r_res  = c / (2 * sample_rate)          # m per sample
    r_nms  = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for r_nm in r_nms:
        r_m    = r_nm * 1852
        t_ms   = r_m / (c / 2) * 1e3       # two-way travel time in ms
        if t_ms < _ZOOM_MS:
            axes2[0].axvline(t_ms, color='#808020', lw=0.7,
                             ls='--', alpha=0.55)
            axes2[0].text(t_ms, vrange * 0.75, f'{r_nm} NM',
                          color='#a0a030', fontsize=6, ha='center')

    plt.tight_layout()

    # ── Figure 3 : FFT of VIDEO channel ───────────────────────────────────────
    v       = ch_data[0]
    N_fft   = min(len(v), 4 * 1024 * 1024)   # cap at 4 M points for speed
    win     = np.hanning(N_fft)
    sp      = np.abs(np.fft.rfft(v[:N_fft] * win)) * 2 / N_fft
    f_khz   = np.fft.rfftfreq(N_fft, d=dt) / 1e3

    fig3, ax3 = plt.subplots(figsize=(13, 4), facecolor='#0a0a14')
    ax3.set_facecolor('#04060e')
    ax3.plot(f_khz, 20 * np.log10(sp + 1e-9), lw=0.6, color='#40c060')
    ax3.set_xlim(0, sample_rate / 2 / 1e3)
    ax3.set_ylim(-90, 5)
    ax3.set_xlabel('Frequency (kHz)', color='#5080a0', fontsize=9)
    ax3.set_ylabel('Amplitude (dBV)', color='#5080a0', fontsize=9)
    ax3.set_title('VIDEO Channel (CH0) — Frequency Spectrum  |  Hanning window',
                  color='#80b0d0', fontsize=10)
    ax3.tick_params(colors='#304050', labelsize=7)
    ax3.spines[:].set_color('#1a2a2a')
    ax3.grid(True, alpha=0.15, color='#304050')
    fig3.tight_layout()

    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # ── Acquisition parameters ────────────────────────────────────────────────
    card_num    = 0              # 0 = first card

    # Scan CH0–CH3 simultaneously (VIDEO, TRIG, BP, HD)
    channel     = 3              # last channel index → scans CH0-CH3

    # ±5 V range covers the full VIDEO signal (≈4 Vp-p) and digital pulses
    ad_range    = AD_B_5_V

    file_name   = '9812d'        # driver writes 9812d.dat (binary int16, interleaved)

    # 1 MHz scan rate ×4 ch = 4 MS/s aggregate  (max card: 20 MS/s)
    # Range resolution = c / (2 × 1 MHz) = 149.9 m / sample
    sample_rate = 1_000_000.0

    # Double-buffer: 4 000 000 samples = 1 s total, 0.5 s per half-buffer
    # (4 ch × 1 M scans/s × 0.5 s = 2 M samples per half → fires every 0.5 s)
    read_count  = 4_000_000

    duration_s  = 15.0           # capture 15 seconds = 60 M samples = ~120 MB

    # ── Run acquisition ───────────────────────────────────────────────────────
    ch_data, sr = acquire(
        channel, ad_range, file_name, read_count, sample_rate,
        card_num=card_num, duration_s=duration_s,
    )

    vrange = 5.0 if ad_range == AD_B_5_V else 1.0
    print('\nChannel statistics:')
    for ch, v in ch_data.items():
        print(f'  CH{ch}  {CH_LABELS.get(ch,""):28s}'
              f'min={v.min():+.3f} V   max={v.max():+.3f} V   '
              f'rms={np.sqrt(np.mean(v**2)):.4f} V')

    # Detect triggers and report PRF
    if 1 in ch_data:
        trig_idx, prf = detect_triggers(ch_data[1], sr)
        print(f'\n  Detected {len(trig_idx)} TRIG pulses  →  PRF ≈ {prf:.0f} Hz')
        if prf > 0:
            sweep_samples = int(sr / prf)
            r_max_m = sweep_samples * (299_792_458 / (2 * sr))
            print(f'  Samples per sweep: {sweep_samples}  '
                  f'→  unambiguous range ≈ {r_max_m/1852:.2f} NM')

    plot_radar_signals(ch_data, sr, vrange, duration_s)
