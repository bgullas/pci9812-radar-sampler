"""
Real-time VIDEO (CH3) radar echo viewer — 20 kHz, 1-second rolling window.

Uses AI_ContScanChannelsToFile (the only DMA path confirmed working on this
card, same as the original C reference).  Scans CH0-CH3, streams the .dat
file as it grows, and displays only CH3 (radar echo VIDEO).

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
SAMPLE_RATE   = 20_000     # S/s  (matches the working C reference)
VOLTAGE_RANGE = 5.0        # ±5 V
WINDOW_S      = 1.0        # seconds shown in rolling window
CARD_NUM      = 0
DAT_BASE      = 'video_live'   # driver writes video_live.dat next to this script
# ─────────────────────────────────────────────────────────────────────────────

# Constants from dask.h  (C:\ADLINK\PCIS-DASK\Include\)
PCI_9812          = 30
AD_B_5_V          = 2
P9812_TRGMOD_SOFT = 0
P9812_TRGSRC_CH0  = 0
P9812_TRGSLP_POS  = 0
P9812_AD2_GT_PCI  = 128    # 0x80
P9812_CLKSRC_INT  = 0
ASYNCH_OP         = 2
ADC_MID_F         = 2048.0

N_CH        = 4
VIDEO_CH    = 3            # CH3 = VIDEO

# read_count must be divisible by 2*N_CH; 4000 matches the working C exactly
READ_COUNT  = 4_000        # uint16 — full double-buffer
HALF_SCANS  = READ_COUNT // (2 * N_CH)   # 500 scans/half  → 25 ms at 20 kHz
HALF_MS     = HALF_SCANS / SAMPLE_RATE * 1000

WINDOW_SAMPS = int(SAMPLE_RATE * WINDOW_S)   # 20 000 samples


# ── DLL wrapper ───────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DLL_SEARCH = [
    _SCRIPT_DIR,
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
    raise FileNotFoundError('PCI-Dask64.dll not found')


class _DLL:
    def __init__(self):
        d = _load_dll()
        I16 = ctypes.c_int16
        U16 = ctypes.c_uint16
        U32 = ctypes.c_uint32
        F64 = ctypes.c_double

        d.Register_Card.argtypes              = [U16, U16];            d.Register_Card.restype = I16
        d.AI_9812_Config.argtypes             = [I16, U16, U16, U16, U16, U16, U32]; d.AI_9812_Config.restype = I16
        d.AI_AsyncDblBufferMode.argtypes      = [I16, U16];            d.AI_AsyncDblBufferMode.restype = I16
        d.AI_ContScanChannelsToFile.argtypes  = [I16, U16, U16, ctypes.c_char_p, U32, F64, U16]; d.AI_ContScanChannelsToFile.restype = I16
        d.AI_AsyncDblBufferHalfReady.argtypes = [I16, ctypes.POINTER(U16), ctypes.POINTER(U16)]; d.AI_AsyncDblBufferHalfReady.restype = I16
        d.AI_AsyncDblBufferTransfer.argtypes  = [I16, ctypes.c_void_p]; d.AI_AsyncDblBufferTransfer.restype = I16
        d.AI_AsyncCheck.argtypes              = [I16, ctypes.POINTER(U16), ctypes.POINTER(U32)]; d.AI_AsyncCheck.restype = I16
        d.AI_AsyncClear.argtypes              = [I16, ctypes.POINTER(U32)]; d.AI_AsyncClear.restype = I16
        d.Release_Card.argtypes               = [I16];                 d.Release_Card.restype = I16
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
            ctypes.c_uint16(P9812_TRGSRC_CH0),
            ctypes.c_uint16(P9812_TRGSLP_POS),
            ctypes.c_uint16(P9812_AD2_GT_PCI | P9812_CLKSRC_INT),
            ctypes.c_uint16(0x80),
            ctypes.c_uint32(0),
        )
        if e: raise RuntimeError(f'AI_9812_Config error={e}')

    def dbl_buf_mode(self, card):
        e = self._d.AI_AsyncDblBufferMode(ctypes.c_int16(card), ctypes.c_uint16(1))
        if e: raise RuntimeError(f'AI_AsyncDblBufferMode error={e}')

    def start_to_file(self, card, base_path):
        name = base_path.encode('ascii')
        e = self._d.AI_ContScanChannelsToFile(
            ctypes.c_int16(card),
            ctypes.c_uint16(N_CH - 1),      # scan CH0-CH3
            ctypes.c_uint16(AD_B_5_V),
            ctypes.c_char_p(name),
            ctypes.c_uint32(READ_COUNT),
            ctypes.c_double(float(SAMPLE_RATE)),
            ctypes.c_uint16(ASYNCH_OP),
        )
        if e: raise RuntimeError(f'AI_ContScanChannelsToFile error={e}')

    def half_ready(self, card):
        hr = ctypes.c_uint16(0)
        fs = ctypes.c_uint16(0)
        self._d.AI_AsyncDblBufferHalfReady(ctypes.c_int16(card),
                                            ctypes.byref(hr), ctypes.byref(fs))
        return bool(hr.value)

    def transfer_to_file(self, card):
        """Flush the ready half to the .dat file (NULL dest = file mode)."""
        self._d.AI_AsyncDblBufferTransfer(ctypes.c_int16(card), None)

    def hw_count(self, card):
        stopped = ctypes.c_uint16(0)
        count   = ctypes.c_uint32(0)
        self._d.AI_AsyncCheck(ctypes.c_int16(card),
                               ctypes.byref(stopped), ctypes.byref(count))
        return count.value

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
        print(f'  Register_Card OK  (card={card})')

        try:
            dll.clear(card)
        except Exception:
            pass

        dll.config(card)
        print('  AI_9812_Config OK  (SOFT / free-run)')
        dll.dbl_buf_mode(card)
        print('  AI_AsyncDblBufferMode OK')

        # Resolve absolute base path (driver appends .dat)
        base = os.path.join(_SCRIPT_DIR, DAT_BASE)
        dat  = base + '.dat'
        if os.path.exists(dat):
            try:
                os.remove(dat)
            except OSError:
                pass

        dll.start_to_file(card, base)
        print(f'  AI_ContScanChannelsToFile started → {dat}')
        print(f'  read_count={READ_COUNT}  {SAMPLE_RATE} S/s  half≈{HALF_MS:.0f} ms')

        # Wait 500 ms then check whether the file exists and hw_count is rising
        time.sleep(0.5)
        _fsize0 = os.path.getsize(dat) if os.path.exists(dat) else -1
        _hw0    = dll.hw_count(card)
        print(f'  [+500 ms]  hw_count={_hw0}  dat_size={_fsize0} B')
        if _fsize0 < 0:
            print('  WARNING: .dat file was NOT created — driver may have rejected the path')
        if _hw0 == 0 and _fsize0 <= 0:
            print('  DMA has not started. Try running as Administrator, or check')
            print('  the ADLINK Device Manager test panel.')

        print('  Polling...')
        sleep_s  = HALF_MS / 1000 * 0.10
        last_hb  = time.monotonic()
        HB_S     = 2.0
        file_pos = 0   # bytes already consumed from the .dat file

        try:
            while not self._quit.is_set():
                if dll.half_ready(card):
                    dll.transfer_to_file(card)   # flush half → .dat
                    self.halves += 1

                    # Read the new bytes the driver just wrote
                    try:
                        fsize = os.path.getsize(dat)
                        if fsize > file_pos:
                            with open(dat, 'rb') as f:
                                f.seek(file_pos)
                                chunk = f.read(fsize - file_pos)
                            file_pos = fsize

                            raw = np.frombuffer(chunk, dtype=np.uint16)
                            raw   = raw[: (len(raw) // N_CH) * N_CH]
                            ch3   = raw[VIDEO_CH::N_CH].astype(np.float32)
                            volts = (ch3 - ADC_MID_F) / ADC_MID_F * VOLTAGE_RANGE
                            with self.lock:
                                self.buf.extend(volts)

                            hw = dll.hw_count(card)
                            print(f'\r  half #{self.halves:>4}  '
                                  f'hw={hw:>8}  '
                                  f'new={len(ch3):>5} samps  '
                                  f'CH3 {volts.min():+.3f}…{volts.max():+.3f} V   ',
                                  end='', flush=True)
                    except OSError:
                        pass
                else:
                    now = time.monotonic()
                    if now - last_hb >= HB_S:
                        hw    = dll.hw_count(card)
                        fsize = os.path.getsize(dat) if os.path.exists(dat) else -1
                        print(f'  waiting...  hw_count={hw}  '
                              f'halves={self.halves}  dat={fsize} B',
                              flush=True)
                        last_hb = now
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
          f'|  {WINDOW_S:.0f} s window  |  half≈{HALF_MS:.0f} ms')

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
