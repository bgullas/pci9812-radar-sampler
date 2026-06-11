"""
ADLINK PCI-9812 Radar Data Sampler
Uses the ADLINK PCIS-DASK driver (PCI-DASK.dll) on Windows.

In-memory double-buffer DMA path (same as plot_radar.py):
  Register_Card → AI_9812_Config → AI_AsyncDblBufferMode
  → AI_ContScanChannels (DMA into a numpy buffer)
  → poll AI_AsyncDblBufferHalfReady, copy each ready half
  → AI_AsyncClear → Release_Card

NOTE: the file-mode variant AI_ContScanChannelsToFile does not acquire on this
card (half-ready never fires, nothing reaches disk), so we DMA to memory.

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


# ===========================================================================
# USER SETTINGS  ← edit these three values, nothing else needs to change
# ===========================================================================

SAMPLE_RATE     = 44_100       # S/s — same for ALL 4 channels (HD, BP, TRIG, VIDEO)
                                #   44_100 = exactly matches the proven-working C
                                #   reference — start here to confirm the capture
                                #   pipeline, then raise it for finer range res:
                                #   1_000_000 = 1 MS/s  |  149.9 m range res
                                #   5_000_000 = 5 MS/s  |   30.0 m range res
                                #  20_000_000 = 20 MS/s |    7.5 m range res

VOLTAGE_RANGE_V = 5.0          # V — input range per channel: 5.0 or 1.0
                                #   5.0 → AD_B_5_V  (covers 12 V digital + VIDEO)
                                #   1.0 → AD_B_1_V  (more resolution for low-level signals)

DURATION_S      = 15.0         # s — total capture duration

# Driver DMA buffer limit — must match the ADLINK Device Manager setting
# (PCI-9812 → Properties → DMA Buffer Size).  Default after install is 1 MB.
# Increase both here and in the driver when moving to higher sample rates:
#   1 MS/s requires  8 MB  |  5 MS/s → 64 MB  |  20 MS/s → 256 MB
DRIVER_DMA_LIMIT_MB = 1        # MB

# ===========================================================================


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

    def AI_ContScanChannels(self, card, channel, ad_range,
                            buf_ptr, count, sample_rate, synch):
        """In-memory continuous scan — DMA fills buf_ptr directly.

        This is the proven path used by plot_radar.py.  The file-mode variant
        (AI_ContScanChannelsToFile) does not actually acquire on this card —
        AI_AsyncDblBufferHalfReady never fires and nothing reaches disk.
        """
        err = self._dll.AI_ContScanChannels(
            ctypes.c_int16(card),
            ctypes.c_uint16(channel),
            ctypes.c_uint16(ad_range),
            buf_ptr,
            ctypes.c_uint32(count),
            ctypes.c_double(sample_rate),
            ctypes.c_uint16(synch),
        )
        self._check(err, 'AI_ContScanChannels')

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

    def AI_AsyncCheck(self, card):
        """Return (stopped, access_count) — the live count of samples the DMA
        engine has actually transferred.  Lets us see whether the hardware is
        sampling even when the half-ready flag has not fired yet."""
        stopped = ctypes.c_uint16(0)
        acc     = ctypes.c_uint32(0)
        self._dll.AI_AsyncCheck(
            ctypes.c_int16(card), ctypes.byref(stopped), ctypes.byref(acc)
        )
        return bool(stopped.value), acc.value

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
        d.AI_ContScanChannels.argtypes        = [I16, U16, U16, ctypes.POINTER(U16), U32, F64, U16]
        d.AI_ContScanChannels.restype         = I16
        d.AI_ContScanChannelsToFile.argtypes  = [I16, U16, U16, ctypes.c_char_p, U32, F64, U16]
        d.AI_ContScanChannelsToFile.restype   = I16
        d.AI_AsyncDblBufferHalfReady.argtypes = [I16, ctypes.POINTER(U16), ctypes.POINTER(U16)]
        d.AI_AsyncDblBufferHalfReady.restype  = I16
        d.AI_AsyncCheck.argtypes              = [I16, ctypes.POINTER(U16), ctypes.POINTER(U32)]
        d.AI_AsyncCheck.restype               = I16
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

def _dma_align(n_scans_per_half, n_ch):
    """
    Round n_scans_per_half up to the nearest power-of-2 multiple of 512.
    The PCI-9812 DMA engine requires the half-buffer size (in scans) to be a
    multiple of 512 for reliable operation; examples in the ADLINK SDK always
    use power-of-2 counts.  Returns total uint16 samples (both halves).
    """
    import math
    # Align scans_per_half to the next power of 2 that is also ≥ 512
    k = max(9, math.ceil(math.log2(max(n_scans_per_half, 1))))  # 2^9 = 512 minimum
    aligned = 1 << k
    return aligned * 2 * n_ch   # total uint16 in the double-buffer


def acquire(channel, ad_range, file_name, read_count, sample_rate,
            card_num=0, duration_s=5.0):
    """
    Acquire radar data and return a dict of per-channel voltage arrays.

    Mirrors the proven working C reference exactly:
        AI_9812_Config → AI_AsyncDblBufferMode → AI_ContScanChannelsToFile
        loop: wait AI_AsyncDblBufferHalfReady → AI_AsyncDblBufferTransfer(NULL)
        AI_AsyncClear → Release_Card → read back the .dat file

    The AI_AsyncDblBufferTransfer(card, NULL) call after every ready half is
    ESSENTIAL — it flushes the completed half to disk and lets the double-buffer
    engine advance.  Without it the engine stalls after a tiny fill and the
    half-ready flag never fires again (the bug that produced 0–13 samples).

    channel     : last channel index (e.g. 3 → scans CH0–CH3)
    ad_range    : AD_B_5_V or AD_B_1_V
    file_name   : output base name WITHOUT extension; driver appends '.dat'.
    read_count  : circular buffer size in total uint16 samples (both halves,
                  all channels).
    sample_rate : samples/second per channel
    duration_s  : how long to run
    """
    n_ch   = channel + 1
    vrange = 5.0 if ad_range == AD_B_5_V else 1.0

    # read_count must split evenly into two halves across all channels.
    read_count = (read_count // (2 * n_ch)) * (2 * n_ch)
    half_count    = read_count // 2                       # uint16 per half
    half_period_s = half_count / (sample_rate * n_ch)

    # ── Resolve file path to absolute and clear any stale capture ─────────────
    if not os.path.isabs(file_name):
        file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 file_name)
    dat_path = file_name + '.dat'
    if os.path.exists(dat_path):
        try:
            os.remove(dat_path)
        except OSError as e:
            print(f'  [warning] could not remove stale {dat_path}: {e}')

    print(f'PCI-9812  CH0–CH{channel}  {sample_rate:.0f} S/s  '
          f'buffer={read_count} ({half_period_s*1000:.0f} ms/half)  → {dat_path}')

    dask = DASK()
    card = dask.Register_Card(PCI_9812, card_num)

    # Clear any acquisition state left by a previous run that did not exit
    # cleanly (card stuck "running" causes the next start to return -15).
    try:
        dask.AI_AsyncClear(card)
    except Exception:
        pass   # ignore — card may not have been running

    dask.AI_9812_Config(
        card,
        trig_mod   = P9812_TRGMOD_SOFT,
        trig_src   = P9812_TRGSRC_CH2,   # matches the working C reference
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

    # ── Poll loop — match the C: wait for a ready half, then TRANSFER it ───────
    count        = 0
    deadline     = time.monotonic() + duration_s
    sleep_s      = max(0.001, half_period_s * 0.40)
    hw_count     = 0
    f_stop_seen  = False

    print(f'Acquiring for {duration_s:.0f}s  '
          f'(half-buffer ≈ {half_period_s*1000:.0f} ms) …')
    while time.monotonic() < deadline:
        half_ready, f_stop = dask.AI_AsyncDblBufferHalfReady(card)
        try:
            _stopped, hw_count = dask.AI_AsyncCheck(card)
        except Exception:
            pass
        if half_ready:
            # THE essential call — flush the ready half to the .dat file and
            # release it back to the double-buffer engine.
            dask.AI_AsyncDblBufferTransfer(card, None)
            count += half_count
            elapsed = duration_s - (deadline - time.monotonic())
            print(f'\r  {count // n_ch:>12,} scans  |  hw={hw_count:>12,}  |  '
                  f'{elapsed:.1f} / {duration_s:.0f} s', end='', flush=True)
        else:
            time.sleep(sleep_s)
        if f_stop and not f_stop_seen:
            f_stop_seen = True   # note overrun but keep going, like the C

    total = dask.AI_AsyncClear(card)
    dask.Release_Card(card)

    # ── Read back, deinterleave, convert to volts ─────────────────────────────
    raw = np.fromfile(dat_path, dtype=np.uint16)
    raw = raw[: len(raw) // n_ch * n_ch]
    samples_per_ch = len(raw) // n_ch
    file_kb = (os.path.getsize(dat_path) // 1024) if os.path.exists(dat_path) else 0
    print(f'\n  captured: {len(raw)} uint16  →  {samples_per_ch} scans/ch  ≈  '
          f'{samples_per_ch / sample_rate:.2f} s  '
          f'(file {file_kb} KB, AI_AsyncClear total={total}'
          f'{", overrun seen" if f_stop_seen else ""})')
    result = {}
    for ch in range(n_ch):
        raw_ch     = raw[ch::n_ch].astype(np.float32)
        result[ch] = (raw_ch - ADC_MID) / ADC_MID * vrange
    return result, sample_rate


# ---------------------------------------------------------------------------
# Radar channel roles (Furuno FAR-2xx8, port J510)
# ---------------------------------------------------------------------------
# CH0 — HD     : heading pulse,      0-12 V → ÷12 divider → ~1 V, 1 pulse/revolution
# CH1 — BP     : bearing pulse,      0-12 V → ÷12 divider → ~1 V, 1 pulse per 0.18°
# CH2 — TRIG   : range trigger pulse,0-12 V → ÷12 divider → ~1 V, 5-15 µs wide
# CH3 — VIDEO  : radar echo video,   4 Vp-p coax → voltage divider → ~0.67 Vp-p
#
# All digital lines (HD / BP / TRIG) pass through an 11 kΩ / 1 kΩ voltage
# divider before the card input to step 12 V down to ≈ 1 V.

CH_LABELS = {
    0: 'HD     (heading / north)',
    1: 'BP     (bearing pulse)',
    2: 'TRIG   (range trigger)',
    3: 'VIDEO  (radar echo)',
}

# Channel index aliases — use these everywhere instead of bare numbers
CH_HD    = 0
CH_BP    = 1
CH_TRIG  = 2
CH_VIDEO = 3

# ---------------------------------------------------------------------------
# Trigger-pulse detection (CH1 — TRIG)
# ---------------------------------------------------------------------------

def detect_triggers(trig_ch, sample_rate, threshold_v=0.30, min_gap_us=500):
    """
    Find rising-edge positions of radar range-trigger pulses in CH2 (TRIG).

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
    """Single time-series figure — all channels vs time for the full capture."""
    n_ch  = len(ch_data)
    dt    = 1.0 / sample_rate
    total = len(ch_data[0])

    if CH_TRIG in ch_data:
        trig_idx, prf = detect_triggers(ch_data[CH_TRIG], sample_rate)
        prf_str = f'PRF ≈ {prf:.0f} Hz' if prf > 0 else 'PRF: n/a'
    else:
        trig_idx, prf_str = np.array([]), ''

    def _fmt_rate(hz):
        if hz >= 1e6:  return f'{hz/1e6:.4g} MS/s'
        if hz >= 1e3:  return f'{hz/1e3:.4g} kS/s'
        return f'{hz:.0f} S/s'

    # Decimate for plotting only — keeps rendering fast for long captures.
    # Build the time axis and the y-samples from the SAME index array so their
    # lengths always match and the x-axis always spans the full capture.
    decimate = max(1, total // 200_000)
    idx = np.arange(0, total, decimate)
    t   = idx * dt

    fig, axes = plt.subplots(n_ch, 1, figsize=(15, 2.6 * n_ch),
                              sharex=True, facecolor='#0a0a14')
    actual_duration = total * dt
    fig.suptitle(
        f'PCI-9812  Radar Signals  —  {_fmt_rate(sample_rate)}  ·  '
        f'{actual_duration:.1f} s captured  ·  {prf_str}',
        fontsize=11, color='#d0e8d0',
    )
    if n_ch == 1:
        axes = [axes]

    for ch, ax in enumerate(axes):
        ax.set_facecolor('#04060e')
        ax.plot(t, ch_data[ch][idx],
                lw=0.6, color='#40c060' if ch == CH_VIDEO else '#60a0ff')
        ax.set_ylabel(CH_LABELS.get(ch, f'CH{ch}') + '\n(V)',
                      color='#80b080', fontsize=9)
        ax.set_ylim(-vrange * 1.05, vrange * 1.05)
        ax.axhline(0, color='#1a2a1a', lw=0.4, ls='--')
        ax.tick_params(axis='y', colors='#304050', labelsize=8)
        ax.spines[:].set_color('#1a2a2a')
        ax.grid(True, alpha=0.15, color='#304050')

        if len(trig_idx) > 0:
            for ti in trig_idx[::max(1, len(trig_idx) // 200)]:
                ax.axvline(ti * dt, color='#ff8030', lw=0.3, alpha=0.25)

    axes[-1].set_xlabel('Time (s)', color='#5080a0', fontsize=12)
    axes[-1].tick_params(axis='x', colors='#7090b0', labelsize=11)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # ── Resolve user settings ─────────────────────────────────────────────────
    sample_rate = float(SAMPLE_RATE)
    ad_range    = AD_B_5_V if VOLTAGE_RANGE_V >= 5.0 else AD_B_1_V
    vrange      = 5.0      if VOLTAGE_RANGE_V >= 5.0 else 1.0
    duration_s  = float(DURATION_S)

    card_num  = 0
    channel   = 3     # scans CH0–CH3 simultaneously
    file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), '9812d')

    # ── Circular DMA buffer (recirculates continuously to file) ───────────────
    # read_count is the size of the double-buffer, NOT the total capture length.
    # The buffer recirculates and the driver appends each completed half to the
    # .dat file for as long as we poll.  Total capture length = duration_s.
    # Keep the buffer modest (~0.25 s) and within the driver's DMA limit so it
    # turns over often and never hits error -15.
    n_ch = channel + 1
    driver_max_scans_half = (DRIVER_DMA_LIMIT_MB * 1024 * 1024 // 2) // (n_ch * 2)
    target_scans_half     = int(0.25 * sample_rate)
    half_scans            = max(1, min(target_scans_half, driver_max_scans_half))
    read_count            = _dma_align(half_scans, n_ch)

    def _fmt_rate(hz):
        if hz >= 1e6:  return f'{hz/1e6:.4g} MS/s'
        if hz >= 1e3:  return f'{hz/1e3:.4g} kS/s'
        return f'{hz:.0f} S/s'

    half_period_ms = (read_count // 2) / (sample_rate * n_ch) * 1000
    print(f'  rate = {_fmt_rate(sample_rate)}  |  '
          f'±{vrange:.0f} V  |  '
          f'duration = {duration_s:.0f} s  |  '
          f'buf = {read_count} ({read_count*2//1024} KB, {half_period_ms:.0f} ms/half)')

    # ── Run acquisition ───────────────────────────────────────────────────────
    ch_data, sr = acquire(
        channel, ad_range, file_name, read_count, sample_rate,
        card_num=card_num, duration_s=duration_s,
    )

    # Guard against an empty capture (card stalled / rate out of range) so we
    # print a useful message instead of crashing on v.min() of a 0-size array.
    n_samples = len(ch_data[0]) if ch_data else 0
    if n_samples == 0:
        print('\n  NO DATA CAPTURED — the DMA never delivered a buffer.')
        print('  Most likely the sample rate is outside the PCI-9812 timebase '
              'range (it is a 20 MS/s card; very low rates such as 1 kS/s stall).')
        print(f'  Try SAMPLE_RATE = 1_000_000 (1 MS/s) or higher.  '
              f'Current = {_fmt_rate(sample_rate)}.')
        raise SystemExit(1)

    print('\nChannel statistics:')
    for ch, v in ch_data.items():
        print(f'  CH{ch}  {CH_LABELS.get(ch,""):28s}'
              f'min={v.min():+.3f} V   max={v.max():+.3f} V   '
              f'rms={np.sqrt(np.mean(v**2)):.4f} V')

    # Detect triggers and report PRF  (CH2 = TRIG)
    if CH_TRIG in ch_data:
        trig_idx, prf = detect_triggers(ch_data[CH_TRIG], sr)
        print(f'\n  Detected {len(trig_idx)} TRIG pulses  →  PRF ≈ {prf:.0f} Hz')
        if prf > 0:
            sweep_samples = int(sr / prf)
            r_max_m = sweep_samples * (299_792_458 / (2 * sr))
            print(f'  Samples per sweep: {sweep_samples}  '
                  f'→  unambiguous range ≈ {r_max_m/1852:.2f} NM')

    plot_radar_signals(ch_data, sr, vrange, duration_s)
