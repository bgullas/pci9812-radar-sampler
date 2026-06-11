"""
Real-time VIDEO (CH3) radar echo viewer — 1 kHz, 1-second rolling window.

Scans all 4 channels simultaneously (required by the PCI-9812 hardware) at
1 kHz and displays only CH3 (VIDEO / radar echo) as a 1-second rolling plot.

Half-buffer = 500 ms at 1 kHz → display refreshes every ~500 ms.
Press Ctrl-C or close the window to stop.

Usage:  python video_viewer.py
"""
import ctypes
import sys
import threading
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pci9812_sampler import (
    DASK, PCI_9812, AD_B_5_V, ADC_MID,
    P9812_TRGMOD_SOFT, P9812_TRGSRC_CH2, P9812_TRGSLP_POS,
    P9812_AD2_GT_PCI, P9812_CLKSRC_INT, ASYNCH_OP,
)

# ── Settings ──────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 20_000     # S/s — scans per second (all 4 channels simultaneously)
VOLTAGE_RANGE = 5.0        # ±5 V
WINDOW_S      = 1.0        # seconds shown in the rolling window
CARD_NUM      = 0
# ─────────────────────────────────────────────────────────────────────────────

N_CH         = 4
VIDEO_CH     = 3           # CH3 = VIDEO (radar echo)
ADC_MID_F    = float(ADC_MID)

# Double-buffer size — matches the proven-working C reference (read_count=4000).
# At 1 kHz / 4 ch: 4000 uint16 = 500 scans/half = 500 ms per half.
READ_COUNT   = 4_000
HALF_COUNT   = READ_COUNT // 2          # uint16 per half
HALF_SCANS   = HALF_COUNT // N_CH      # 500 scans per half

WINDOW_SAMPS = int(SAMPLE_RATE * WINDOW_S)   # 1000 samples for 1-second window


class _Capture(threading.Thread):
    """Background DMA thread — continuously appends CH3 voltage data to buf."""

    def __init__(self):
        super().__init__(daemon=True)
        self.buf     = deque(np.zeros(WINDOW_SAMPS, dtype=np.float32),
                             maxlen=WINDOW_SAMPS)
        self.lock    = threading.Lock()
        self._quit   = threading.Event()
        self.error   = None
        self.halves  = 0      # total half-buffers received (for display title)

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
        dask = DASK()
        card = dask.Register_Card(PCI_9812, CARD_NUM)

        try:
            dask.AI_AsyncClear(card)
        except Exception:
            pass

        dask.AI_9812_Config(
            card,
            trig_mod   = P9812_TRGMOD_SOFT,
            trig_src   = P9812_TRGSRC_CH2,
            trig_slp   = P9812_TRGSLP_POS,
            ad_timing  = P9812_AD2_GT_PCI | P9812_CLKSRC_INT,
            trig_level = 0x80,
            post_count = 0,
        )
        dask.AI_AsyncDblBufferMode(card, enable=True)

        # Main circular DMA buffer — the driver writes interleaved CH0–CH3 here
        main_buf = np.zeros(READ_COUNT, dtype=np.uint16)
        main_ptr = main_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))

        # Destination for each completed half (Transfer copies into this)
        half_buf = np.zeros(HALF_COUNT, dtype=np.uint16)
        half_ptr = half_buf.ctypes.data_as(ctypes.c_void_p)

        # AI_ContScanChannels is present in current pci9812_sampler.py but may
        # be missing from older local copies — call the DLL directly as fallback.
        if hasattr(dask, 'AI_ContScanChannels'):
            dask.AI_ContScanChannels(
                card, N_CH - 1, AD_B_5_V, main_ptr,
                READ_COUNT, float(SAMPLE_RATE), ASYNCH_OP,
            )
        else:
            fn = dask._dll.AI_ContScanChannels
            fn.argtypes = [
                ctypes.c_int16, ctypes.c_uint16, ctypes.c_uint16,
                ctypes.POINTER(ctypes.c_uint16),
                ctypes.c_uint32, ctypes.c_double, ctypes.c_uint16,
            ]
            fn.restype = ctypes.c_int16
            err = fn(
                ctypes.c_int16(card), ctypes.c_uint16(N_CH - 1),
                ctypes.c_uint16(AD_B_5_V), main_ptr,
                ctypes.c_uint32(READ_COUNT), ctypes.c_double(float(SAMPLE_RATE)),
                ctypes.c_uint16(ASYNCH_OP),
            )
            if err:
                raise RuntimeError(f'AI_ContScanChannels error={err}')

        sleep_s = (HALF_SCANS / SAMPLE_RATE) * 0.10   # poll at 10× the half rate

        try:
            while not self._quit.is_set():
                half_ready, _ = dask.AI_AsyncDblBufferHalfReady(card)
                if half_ready:
                    # Copy the ready half out and release it back to the engine
                    dask.AI_AsyncDblBufferTransfer(card, half_ptr)
                    # Deinterleave: CH3 is every 4th sample starting at index 3
                    raw   = half_buf[VIDEO_CH::N_CH].astype(np.float32)
                    volts = (raw - ADC_MID_F) / ADC_MID_F * VOLTAGE_RANGE
                    with self.lock:
                        self.buf.extend(volts)
                    self.halves += 1
                else:
                    time.sleep(sleep_s)
        finally:
            try:
                dask.AI_AsyncClear(card)
            except Exception:
                pass
            dask.Release_Card(card)


def main():
    print(f'CH3 VIDEO viewer  |  {SAMPLE_RATE} S/s  |  ±{VOLTAGE_RANGE:.0f} V  |  '
          f'{WINDOW_S:.0f} s window')
    print(f'Half-buffer: {HALF_SCANS} scans = {HALF_SCANS / SAMPLE_RATE * 1000:.0f} ms  '
          f'→  display refreshes every ~{HALF_SCANS / SAMPLE_RATE * 1000:.0f} ms')

    cap = _Capture()
    cap.start()
    time.sleep(0.3)   # let the card start before first draw

    t_axis = np.linspace(-WINDOW_S, 0.0, WINDOW_SAMPS)

    fig, ax = plt.subplots(figsize=(13, 4), facecolor='#04060e')
    ax.set_facecolor('#04060e')
    (line,) = ax.plot(t_axis, np.zeros(WINDOW_SAMPS), color='#40c060', lw=0.8)
    ax.set_xlim(-WINDOW_S, 0.0)
    ax.set_ylim(-VOLTAGE_RANGE * 1.1, VOLTAGE_RANGE * 1.1)
    ax.set_xlabel('Time (s)', color='#5080a0', fontsize=12)
    ax.set_ylabel('CH3  VIDEO\n(V)', color='#80b080', fontsize=10)
    title = ax.set_title(
        f'PCI-9812  CH3 VIDEO  ·  {SAMPLE_RATE/1e3:.1f} kS/s  ·  '
        f'{WINDOW_S:.0f} s rolling window  —  waiting…',
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
            title.set_text(f'CAPTURE ERROR: {cap.error}')
            title.set_color('#ff4040')
            return (line,)
        with cap.lock:
            data = np.array(cap.buf, dtype=np.float32)
        if len(data) == WINDOW_SAMPS:
            line.set_ydata(data)
        if cap.halves > 0:
            title.set_text(
                f'PCI-9812  CH3 VIDEO  ·  {SAMPLE_RATE/1e3:.1f} kS/s  ·  '
                f'{WINDOW_S:.0f} s rolling  ·  {cap.halves} halves received'
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
            print(f'\nCapture error: {cap.error}', file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()
