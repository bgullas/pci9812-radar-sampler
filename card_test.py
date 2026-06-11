"""
PCI-9812 acquisition sanity test — run this FIRST when no data appears.

Both plot_radar.py and pci9812_sampler.py drive the card the same way
(AI_9812_Config → AI_ContScanChannels).  When neither shows data, the
question is no longer "which program" but "does the card digitize at all?"

This script does ONE blocking, SYNCHRONOUS read of a small fixed block —
the same DMA path the real programs use, but without the asynchronous
double-buffer / half-ready polling that can mask what is really happening.

Outcomes
--------
  * Completes with varying voltages   → the card, driver, wiring and
    AI_9812_Config are fine; the problem is the ASYNC double-buffer polling
    in the main programs, and we fix it there.
  * TIMES OUT (never completes)        → the timebase/trigger never starts a
    continuous conversion — a hardware / AI_9812_Config issue shared by both
    programs.  This is the case to chase next (trigger mode, clock source,
    or the original C reference's exact config arguments).
  * Flat / identical values            → the ADC is converting but the input
    isn't reaching it (wiring, range, or the radar not transmitting).

Usage:  python card_test.py
"""
import ctypes
import threading
import time

import numpy as np

from pci9812_sampler import (
    DASK, PCI_9812, AD_B_5_V,
    P9812_TRGMOD_SOFT, P9812_TRGSRC_CH0, P9812_TRGSLP_POS,
    P9812_AD2_GT_PCI, P9812_CLKSRC_INT, SYNCH_OP, ADC_MID,
)

N_CH        = 4
SCANS       = 4096                       # samples per channel (2^12 — DMA aligned)
SAMPLE_RATE = 1_000_000.0                # 1 MS/s — well inside the card's range
COUNT       = SCANS * N_CH               # total uint16 across all channels
TIMEOUT_S   = 5.0
CH_NAMES    = ['HD', 'BP', 'TRIG', 'VIDEO']


def main():
    buf = np.zeros(COUNT, dtype=np.uint16)
    ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))

    dask = DASK()
    card = dask.Register_Card(PCI_9812, 0)

    # Clear any state left by a previous crashed run.
    try:
        dask.AI_AsyncClear(card)
    except Exception:
        pass

    dask.AI_9812_Config(
        card,
        trig_mod   = P9812_TRGMOD_SOFT,
        trig_src   = P9812_TRGSRC_CH0,
        trig_slp   = P9812_TRGSLP_POS,
        ad_timing  = P9812_AD2_GT_PCI | P9812_CLKSRC_INT,
        trig_level = 0x80,
        post_count = 0,
    )

    print(f'\nSynchronous (blocking) read: {SCANS} scans/ch @ '
          f'{SAMPLE_RATE/1e6:.1f} MS/s  (expect ~{SCANS/SAMPLE_RATE*1000:.0f} ms)…')

    # Run the blocking call in a worker thread so a stalled card cannot hang
    # the test forever — ctypes releases the GIL during the C call, so the
    # join() timeout below is honoured.
    done = {}

    def _go():
        try:
            dask.AI_ContScanChannels(
                card, N_CH - 1, AD_B_5_V, ptr,
                COUNT, SAMPLE_RATE, SYNCH_OP,
            )
            done['ok'] = True
        except Exception as exc:          # noqa: BLE001 — report any error
            done['err'] = exc

    worker = threading.Thread(target=_go, daemon=True)
    t0 = time.monotonic()
    worker.start()
    worker.join(timeout=TIMEOUT_S)
    dt = time.monotonic() - t0

    if worker.is_alive():
        print(f'\n  ✗ TIMEOUT after {dt:.1f}s — the card never completed a '
              f'{SCANS}-scan synchronous read.')
        print('    The conversion is stalled at the hardware/config level, NOT '
              'in the async polling.')
        print('    Next: verify the card in the ADLINK Device Manager test '
              'panel, and check the AI_9812_Config trigger/clock arguments '
              'against the original working C program.')
    elif 'err' in done:
        print(f'\n  ✗ ERROR: {done["err"]}')
    else:
        print(f'\n  ✓ completed in {dt*1000:.0f} ms')
        print('    per-channel result (±5 V range):')
        for ch in range(N_CH):
            raw = buf[ch::N_CH].astype(np.float32)
            v   = (raw - ADC_MID) / ADC_MID * 5.0
            print(f'    CH{ch} {CH_NAMES[ch]:<6} '
                  f'min={v.min():+7.3f}  max={v.max():+7.3f}  '
                  f'mean={v.mean():+7.3f} V   '
                  f'raw[{int(raw.min())}..{int(raw.max())}]')
        spread = max(
            float(buf[ch::N_CH].astype(np.int32).ptp()) for ch in range(N_CH)
        )
        if spread < 4:
            print('\n    ⚠ values are essentially flat — the ADC is converting '
                  'but no signal is reaching the inputs (wiring / range / the '
                  'radar not transmitting).')
        else:
            print('\n    ✓ the card digitizes real, varying data — the fault is '
                  'in the asynchronous double-buffer polling, not the card.')

    try:
        dask.AI_AsyncClear(card)
    except Exception:
        pass
    dask.Release_Card(card)


if __name__ == '__main__':
    main()
