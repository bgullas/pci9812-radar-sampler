"""
ADLINK PCI-9812 Radar Data Sampler
Uses the ADLINK DASK driver (DASK.dll) on Windows.

Translated from the working C reference program — identical API calls:
  Register_Card → AI_9812_Config → AI_AsyncDblBufferMode
  → AI_ContScanChannelsToFile → poll AI_AsyncDblBufferHalfReady
  → AI_AsyncClear → Release_Card

Requirements:
  pip install numpy matplotlib
  ADLINK DASK driver installed (provides DASK.dll)
"""

import sys
import ctypes
import time
import numpy as np
import matplotlib.pyplot as plt

if sys.platform != 'win32':
    raise EnvironmentError('This script must run on Windows (DASK.dll required)')


# ---------------------------------------------------------------------------
# Constants — from dask.h
# ---------------------------------------------------------------------------

PCI_9812 = 17

AD_B_5_V  = 1   # Bipolar ±5 V
AD_B_1_V  = 3   # Bipolar ±1 V

P9812_TRGMOD_SOFT = 0
P9812_TRGMOD_POST = 1
P9812_TRGMOD_PRE  = 2
P9812_TRGMOD_MIDL = 3
P9812_TRGMOD_DELY = 4

P9812_TRGSRC_CH0  = 0
P9812_TRGSRC_CH1  = 1
P9812_TRGSRC_CH2  = 2
P9812_TRGSRC_CH3  = 3
P9812_TRGSRC_EXT  = 4

P9812_TRGSLP_POS  = 0
P9812_TRGSLP_NEG  = 1

P9812_CLKSRC_INT  = 0x0000
P9812_CLKSRC_SIN  = 0x0004
P9812_CLKSRC_SQR  = 0x0008
P9812_AD2_GT_PCI  = 0x0002

SYNCH_OP   = 0
ASYNCH_OP  = 1

ADC_MID    = 2048   # 12-bit midpoint for voltage conversion


# ---------------------------------------------------------------------------
# DASK DLL wrapper
# ---------------------------------------------------------------------------

class DASK:
    """ctypes wrapper around DASK.dll — mirrors the C API exactly."""

    DLL_NAME = 'PCI-DASK.dll'

    def __init__(self):
        try:
            self._dll = ctypes.windll.LoadLibrary(self.DLL_NAME)
        except OSError as exc:
            raise RuntimeError(
                f'Cannot load {self.DLL_NAME}. '
                'Install the ADLINK DASK driver package.'
            ) from exc
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

    print(f'Acquiring for {duration_s}s …')
    while time.monotonic() < deadline:
        half_ready, f_stop = dask.AI_AsyncDblBufferHalfReady(card)
        if half_ready:
            dask.AI_AsyncDblBufferTransfer(card, buf=None)
            count += half_count
            print(f'\r{count} samples', end='', flush=True)
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
# Plot helpers
# ---------------------------------------------------------------------------

def plot_time(ch_data, sample_rate, vrange, title='PCI-9812 Radar'):
    n = len(ch_data)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ch, ax in enumerate(axes):
        t = np.arange(len(ch_data[ch])) / sample_rate * 1e3
        ax.plot(t, ch_data[ch], lw=0.6)
        ax.set_ylabel(f'CH{ch} (V)')
        ax.set_ylim(-vrange * 1.05, vrange * 1.05)
        ax.axhline(0, color='gray', lw=0.4, ls='--')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (ms)')
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_fft(ch_data, sample_rate):
    n = len(ch_data)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ch, ax in enumerate(axes):
        v   = ch_data[ch]
        N   = len(v)
        win = np.hanning(N)
        sp  = np.abs(np.fft.rfft(v * win)) * 2 / N
        f   = np.fft.rfftfreq(N, d=1.0 / sample_rate) / 1e3
        ax.plot(f, 20 * np.log10(sp + 1e-9), lw=0.6)
        ax.set_ylabel(f'CH{ch} (dBV)')
        ax.set_ylim(-100, 10)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Frequency (kHz)')
    fig.suptitle('Frequency Spectrum  |  Hanning window', fontsize=13)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point — matches the C program parameters exactly
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    channel     = 3
    ad_range    = AD_B_5_V
    file_name   = '9812d'
    read_count  = 4000
    sample_rate = 20000.0
    duration_s  = 5.0

    card_num = int(input('Please input a card number: '))

    ch_data, sr = acquire(
        channel, ad_range, file_name, read_count, sample_rate,
        card_num=card_num, duration_s=duration_s,
    )

    vrange = 5.0 if ad_range == AD_B_5_V else 1.0
    print('\nChannel statistics:')
    for ch, v in ch_data.items():
        print(f'  CH{ch}: min={v.min():.4f} V  max={v.max():.4f} V  '
              f'rms={np.sqrt(np.mean(v**2)):.4f} V')

    plot_time(ch_data, sr, vrange)
    plot_fft(ch_data, sr)
