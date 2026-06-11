"""
Real-time VIDEO (CH3) radar echo viewer — 20 kHz, 1-second rolling window.

Standalone: does NOT import from pci9812_sampler.py.
Scans all 4 channels simultaneously (PCI-9812 requirement) and displays
only CH3 (VIDEO / radar echo) as a 1-second rolling time series.

Usage:  python video_viewer.py
Press Ctrl-C or close the window to stop.
"""
import ctypes
import os
import sys
import threading
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ── Settings ──────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 20_000     # S/s — scans per second (all 4 ch simultaneously)
VOLTAGE_RANGE = 5.0        # ±5 V
WINDOW_S      = 1.0        # seconds shown in rolling window
CARD_NUM      = 0
# ─────────────────────────────────────────────────────────────────────────────

# Constants verified from dask.h on this system (C:\ADLINK\PCIS-DASK\Include\)
PCI_9812          = 30
AD_B_5_V          = 2
P9812_TRGMOD_SOFT = 0
P9812_TRGSRC_CH2  = 16     # 0x10
P9812_TRGSLP_POS  = 0
P9812_AD2_GT_PCI  = 128    # 0x80
P9812_CLKSRC_INT  = 0
ASYNCH_OP         = 2
ADC_MID           = 2048

N_CH          = 4
VIDEO_CH      = 3           # CH3 = VIDEO (radar echo)
ADC_MID_F     = float(ADC_MID)

# Double-buffer — matches the working C reference (read_count=4000)
# At 20 kHz / 4 ch: 4000 uint16 = 500 scans/half = 25 ms per half
READ_COUNT    = 4_000
HALF_COUNT    = READ_COUNT // 2
HALF_SCANS    = HALF_COUNT // N_CH

WINDOW_SAMPS  = int(SAMPLE_RATE * WINDOW_S)


# ── Minimal DLL wrapper (standalone — no pci9812_sampler needed) ──────────────

_DLL_SEARCH = [
    os.path.dirname(os.path.abspath(__file__)),
    r'C:\Windows\System32',
    r'C:\Windows\SysWOW64',
    r'C:\ADLINK\PCIS-DASK',
    r'C:\ADLINK\PCI-DASK',
]

def _load_dll():
    for d in _DLL_SEARCH:
        p = os.path.join(d, 'PCI-Dask64.dll')
        if os.path.isfile(p):
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(d)
            print(f'Found PCI-Dask64.dll: {p}')
            return ctypes.WinDLL(p)
    raise FileNotFoundError(
        'PCI-Dask64.dll not found. Copy it next to this script or add its '
        'folder to PATH.'
    )


class _DLL:
    """Thin ctypes wrapper — only the calls this viewer needs."""

    def __init__(self):
        d = _load_dll()
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
        d.AI_ContScanChannels.argtypes        = [I16, U16, U16,
                                                 ctypes.POINTER(U16),
                                                 U32, F64, U16]
        d.AI_ContScanChannels.restype         = I16
        d.AI_AsyncDblBufferHalfReady.argtypes = [I16,
                                                 ctypes.POINTER(U16),
                                                 ctypes.POINTER(U16)]
        d.AI_AsyncDblBufferHalfReady.restype  = I16
        d.AI_AsyncDblBufferTransfer.argtypes  = [I16, ctypes.c_void_p]
        d.AI_AsyncDblBufferTransfer.restype   = I16
        d.AI_AsyncClear.argtypes              = [I16, ctypes.POINTER(U32)]
        d.AI_AsyncClear.restype               = I16
        d.Release_Card.argtypes               = [I16]
        d.Release_Card.restype                = I16
        self._d = d

    def register_card(self):
        h = self._d.Register_Card(ctypes.c_uint16(PCI_9812),
                                   ctypes.c_uint16(CARD_NUM))
        if h < 0:
            raise RuntimeError(f'Register_Card error={h}')
        return h

    def config(self, card):
        e = self._d.AI_9812_Config(
            ctypes.c_int16(card),
            ctypes.c_uint16(P9812_TRGMOD_SOFT),
            ctypes.c_uint16(P9812_TRGSRC_CH2),
            ctypes.c_uint16(P9812_TRGSLP_POS),
            ctypes.c_uint16(P9812_AD2_GT_PCI | P9812_CLKSRC_INT),
            ctypes.c_uint16(0x80),
            ctypes.c_uint32(0),
        )
        if e: raise RuntimeError(f'AI_9812_Config error={e}')

    def dbl_buf_mode(self, card):
        e = self._d.AI_AsyncDblBufferMode(ctypes.c_int16(card),
                                           ctypes.c_uint16(1))
        if e: raise RuntimeError(f'AI_AsyncDblBufferMode error={e}')

    def start_scan(self, card, buf_ptr):
        e = self._d.AI_ContScanChannels(
            ctypes.c_int16(card),
            ctypes.c_uint16(N_CH - 1),
            ctypes.c_uint16(AD_B_5_V),
            buf_ptr,
            ctypes.c_uint32(READ_COUNT),
            ctypes.c_double(float(SAMPLE_RATE)),
            ctypes.c_uint16(ASYNCH_OP),
        )
        if e: raise RuntimeError(f'AI_ContScanChannels error={e}')

    def half_ready(self, card):
        hr = ctypes.c_uint16(0)
        fs = ctypes.c_uint16(0)
        self._d.AI_AsyncDblBufferHalfReady(ctypes.c_int16(card),
                                            ctypes.byref(hr),
                                            ctypes.byref(fs))
        return bool(hr.value)

    def transfer(self, card, dst_ptr):
        self._d.AI_AsyncDblBufferTransfer(ctypes.c_int16(card), dst_ptr)

    def clear(self, card):
        n = ctypes.c_uint32(0)
        self._d.AI_AsyncClear(ctypes.c_int16(card), ctypes.byref(n))

    def release(self, card):
        self._d.Release_Card(ctypes.c_int16(card))


# ── Capture thread ────────────────────────────────────────────────────────────

class _Capture(threading.Thread):

    def __init__(self):
        super().__init__(daemon=True)
        self.buf    = deque(np.zeros(WINDOW_SAMPS, dtype=np.float32),
                            maxlen=WINDOW_SAMPS)
        self.lock   = threading.Lock()
        self._quit  = threading.Event()
        self.error  = None
        self.halves = 0

    def stop(self):
        self._quit.set()

    def run(self):
        try:
            self._run()
        except Exception as exc:
            self.error = exc
            import traceback
            traceback.print_exc()

    def _run(self):
        dll  = _DLL()
        card = dll.register_card()

        try:
            dll.clear(card)
        except Exception:
            pass

        dll.config(card)
        dll.dbl_buf_mode(card)

        main_buf = np.zeros(READ_COUNT, dtype=np.uint16)
        main_ptr = main_buf.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint16))

        half_buf = np.zeros(HALF_COUNT, dtype=np.uint16)
        half_ptr = half_buf.ctypes.data_as(ctypes.c_void_p)

        dll.start_scan(card, main_ptr)

        sleep_s = (HALF_SCANS / SAMPLE_RATE) * 0.10

        try:
            while not self._quit.is_set():
                if dll.half_ready(card):
                    dll.transfer(card, half_ptr)
                    raw   = half_buf[VIDEO_CH::N_CH].astype(np.float32)
                    volts = (raw - ADC_MID_F) / ADC_MID_F * VOLTAGE_RANGE
                    with self.lock:
                        self.buf.extend(volts)
                    self.halves += 1
                else:
                    time.sleep(sleep_s)
        finally:
            try:
                dll.clear(card)
            except Exception:
                pass
            dll.release(card)


# ── Plot ──────────────────────────────────────────────────────────────────────

def main():
    print(f'CH3 VIDEO  |  {SAMPLE_RATE/1e3:.0f} kS/s  |  ±{VOLTAGE_RANGE:.0f} V  '
          f'|  {WINDOW_S:.0f} s window')
    print(f'Half-buffer: {HALF_SCANS} scans = {HALF_SCANS/SAMPLE_RATE*1000:.0f} ms  '
          f'→  refreshes ~{1000*HALF_SCANS/SAMPLE_RATE:.0f} ms')

    cap = _Capture()
    cap.start()
    time.sleep(0.3)

    t_axis = np.linspace(-WINDOW_S, 0.0, WINDOW_SAMPS)

    fig, ax = plt.subplots(figsize=(13, 4), facecolor='#04060e')
    ax.set_facecolor('#04060e')
    (line,) = ax.plot(t_axis, np.zeros(WINDOW_SAMPS), color='#40c060', lw=0.8)
    ax.set_xlim(-WINDOW_S, 0.0)
    ax.set_ylim(-VOLTAGE_RANGE * 1.1, VOLTAGE_RANGE * 1.1)
    ax.set_xlabel('Time (s)', color='#5080a0', fontsize=12)
    ax.set_ylabel('CH3  VIDEO\n(V)', color='#80b080', fontsize=10)
    title = ax.set_title(
        f'PCI-9812  CH3 VIDEO  ·  {SAMPLE_RATE/1e3:.0f} kS/s  ·  '
        f'{WINDOW_S:.0f} s rolling  —  waiting…',
        fontsize=11, color='#d0e8d0',
    )
    ax.tick_params(axis='x', colors='#7090b0', labelsize=11)
    ax.tick_params(axis='y', colors='#304050', labelsize=9)
    ax.spines[:].set_color('#1a2a2a')
    ax.grid(True, alpha=0.15, color='#304050')
    ax.axhline(0, color='#1a2a1a', lw=0.4, ls='--')
    plt.tight_layout()

    def _update(_frame):
        if cap.error:
            title.set_text(f'ERROR: {cap.error}')
            title.set_color('#ff4040')
            return (line,)
        with cap.lock:
            data = np.array(cap.buf, dtype=np.float32)
        if len(data) == WINDOW_SAMPS:
            line.set_ydata(data)
        if cap.halves > 0:
            title.set_text(
                f'PCI-9812  CH3 VIDEO  ·  {SAMPLE_RATE/1e3:.0f} kS/s  ·  '
                f'{WINDOW_S:.0f} s rolling  ·  halves={cap.halves}'
            )
        return (line,)

    ani = animation.FuncAnimation(
        fig, _update, interval=200, blit=True, cache_frame_data=False,
    )

    try:
        plt.show()
    finally:
        cap.stop()
        cap.join(timeout=3.0)
        if cap.error:
            sys.exit(1)


if __name__ == '__main__':
    main()
