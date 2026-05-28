"""
ADLINK PCI-9812 Radar Data Sampler
Uses the PCIS-DASK driver DLL (Windows) via ctypes.

Hardware specs:
  - 4 simultaneous analog input channels, 12-bit ADC
  - Up to 20 MS/s sampling rate (40 MHz internal clock / divisor)
  - Input voltage range: ±1V or ±5V (programmable)
  - 32k-sample onboard FIFO, bus-mastering DMA transfer

Driver: PCIS-DASK.dll  (install ADLINK PCIS-DASK driver package first)
"""

import ctypes
import ctypes.wintypes
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# PCIS-DASK constants
# ---------------------------------------------------------------------------

# Trigger modes
TRIG_SOFTWARE   = 0
TRIG_POST       = 1
TRIG_PRE        = 2
TRIG_MIDDLE     = 3
TRIG_DELAY      = 4

# Trigger sources
TRIG_SRC_CH0    = 0   # Analog trigger from channel 0
TRIG_SRC_CH1    = 1
TRIG_SRC_CH2    = 2
TRIG_SRC_CH3    = 3
TRIG_SRC_EXT    = 4   # External digital trigger

# Trigger slope
TRIG_SLOPE_POS  = 0   # Rising edge
TRIG_SLOPE_NEG  = 1   # Falling edge

# Clock sources
INT_CLK         = 0   # Internal 40 MHz clock
SIN_CLK         = 1   # External sine wave clock
SQR_CLK         = 2   # External square wave clock

# Voltage ranges (for raw→voltage conversion)
VRANGE_1V       = 1.0   # ±1V
VRANGE_5V       = 5.0   # ±5V

# DMA status flags
DMA_DONE        = 0
DMA_RUNNING     = 1

# ADC resolution
ADC_BITS        = 12
ADC_COUNTS      = 2 ** ADC_BITS        # 4096
ADC_MIDPOINT    = ADC_COUNTS // 2      # 2048 (zero-voltage point)

# Internal clock frequency
MASTER_CLK_HZ   = 40_000_000          # 40 MHz


# ---------------------------------------------------------------------------
# Helper: compute clock divisor for a desired sample rate
# ---------------------------------------------------------------------------

def sample_rate_to_divisor(sample_rate_hz: int) -> int:
    """
    Sampling rate = MASTER_CLK_HZ / (divisor + 1)
    Divisor must be in range [1, 65535].
    """
    divisor = round(MASTER_CLK_HZ / sample_rate_hz) - 1
    divisor = max(1, min(divisor, 65535))
    actual = MASTER_CLK_HZ / (divisor + 1)
    if abs(actual - sample_rate_hz) / sample_rate_hz > 0.01:
        print(f"[warn] Requested {sample_rate_hz/1e6:.3f} MS/s, "
              f"actual will be {actual/1e6:.3f} MS/s (divisor={divisor})")
    return divisor


def raw_to_voltage(raw: np.ndarray, vrange: float = VRANGE_5V) -> np.ndarray:
    """Convert 12-bit ADC raw values to volts."""
    return ((raw.astype(np.float32) - ADC_MIDPOINT) / ADC_MIDPOINT) * vrange


# ---------------------------------------------------------------------------
# PCI9812 driver wrapper
# ---------------------------------------------------------------------------

class PCI9812:
    """
    Context-manager wrapper around the ADLINK PCIS-DASK Windows DLL.

    Usage:
        with PCI9812(card_no=0) as daq:
            data = daq.acquire(
                channels=[0, 1, 2, 3],
                samples_per_channel=8192,
                sample_rate_hz=10_000_000,
            )
    """

    DLL_NAME = "PCIS-DASK.dll"

    def __init__(self, card_no: int = 0, vrange: float = VRANGE_5V):
        self.card_no  = card_no
        self.vrange   = vrange
        self._dll     = None
        self._dma_buf = None   # ctypes array allocated for DMA
        self._open    = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self):
        """Load DLL and initialise the card."""
        try:
            self._dll = ctypes.WinDLL(self.DLL_NAME)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot load {self.DLL_NAME}. "
                "Ensure the ADLINK PCIS-DASK driver is installed."
            ) from exc

        self._configure_prototypes()

        op_base = ctypes.c_uint16(0)
        pt_base = ctypes.c_uint16(0)
        irq_no  = ctypes.c_uint16(0)
        pci_master = ctypes.c_uint16(0)

        ret = self._dll.W_9812_Initial(
            self.card_no,
            ctypes.byref(op_base),
            ctypes.byref(pt_base),
            ctypes.byref(irq_no),
            ctypes.byref(pci_master),
        )
        self._check(ret, "W_9812_Initial")
        self._open = True
        print(f"[PCI-9812] card {self.card_no} opened  "
              f"(base=0x{op_base.value:04X}, IRQ={irq_no.value}, "
              f"master={pci_master.value})")

    def close(self):
        if self._dma_buf is not None:
            self._dll.W_9812_Free_DMA_Mem(ctypes.byref(self._dma_buf))
            self._dma_buf = None
        if self._open and self._dll is not None:
            self._dll.W_9812_Close(self.card_no)
            self._open = False
            print(f"[PCI-9812] card {self.card_no} closed")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_clock(self, sample_rate_hz: int, clk_src: int = INT_CLK):
        divisor = sample_rate_to_divisor(sample_rate_hz)
        ret = self._dll.W_9812_Set_Clk_Rate(
            self.card_no, clk_src, 0, ctypes.c_uint16(divisor)
        )
        self._check(ret, "W_9812_Set_Clk_Rate")
        actual = MASTER_CLK_HZ / (divisor + 1)
        print(f"[PCI-9812] sample rate set to {actual/1e6:.3f} MS/s "
              f"(divisor={divisor})")
        return actual

    def set_trigger(
        self,
        mode:  int   = TRIG_SOFTWARE,
        src:   int   = TRIG_SRC_EXT,
        slope: int   = TRIG_SLOPE_POS,
        level: int   = 0,
        post_samples: int = 0,
    ):
        """
        level     : trigger level as raw ADC count (0–4095), ignored for
                    software/external triggers.
        post_samples : number of samples to collect after trigger (post/delay
                    trigger modes). Pass total_samples for post-trigger mode.
        """
        ret = self._dll.W_9812_Set_Trig(
            self.card_no,
            mode,
            src,
            slope,
            ctypes.c_uint16(level),
            ctypes.c_uint16(post_samples),
        )
        self._check(ret, "W_9812_Set_Trig")

    # ------------------------------------------------------------------
    # DMA acquisition
    # ------------------------------------------------------------------

    def acquire(
        self,
        channels: list[int],
        samples_per_channel: int,
        sample_rate_hz: int = 10_000_000,
        trigger_mode: int = TRIG_SOFTWARE,
        timeout_s: float = 5.0,
    ) -> dict[int, np.ndarray]:
        """
        Acquire data from the specified channels.

        Returns a dict  {channel_index: voltage_array}  where each array
        contains `samples_per_channel` float32 voltage values.

        channels            : list of channel indices, e.g. [0, 1, 2, 3]
        samples_per_channel : number of samples per channel
        sample_rate_hz      : desired aggregate sampling rate
        trigger_mode        : one of TRIG_SOFTWARE, TRIG_POST, etc.
        timeout_s           : maximum wait time for DMA completion
        """
        if not self._open:
            raise RuntimeError("Card not opened. Call open() first.")

        n_ch     = len(channels)
        total_samples = samples_per_channel * n_ch  # interleaved in DMA buf

        # --- Channel enable bitmask (bits 0-3) ---
        ch_mask = 0
        for ch in channels:
            if ch not in range(4):
                raise ValueError(f"Invalid channel {ch}. Must be 0–3.")
            ch_mask |= (1 << ch)

        # --- Configure clock ---
        actual_rate = self.set_clock(sample_rate_hz)

        # --- Configure trigger ---
        self.set_trigger(
            mode=trigger_mode,
            post_samples=samples_per_channel,
        )

        # --- Allocate DMA buffer ---
        buf_size = total_samples * ctypes.sizeof(ctypes.c_uint16)
        DMABuf = ctypes.c_uint16 * total_samples
        dma_buf = DMABuf()

        ret = self._dll.W_9812_Alloc_DMA_Mem(
            self.card_no,
            buf_size,
            ctypes.byref(dma_buf),
        )
        self._check(ret, "W_9812_Alloc_DMA_Mem")
        self._dma_buf = dma_buf

        # --- Start DMA acquisition ---
        ret = self._dll.W_9812_AD_DMA_Start(
            self.card_no,
            ctypes.c_uint16(ch_mask),
            total_samples,
            ctypes.byref(dma_buf),
        )
        self._check(ret, "W_9812_AD_DMA_Start")
        print(f"[PCI-9812] DMA started — {n_ch} ch × {samples_per_channel} "
              f"samples @ {actual_rate/1e6:.3f} MS/s")

        # --- Poll for completion ---
        status   = ctypes.c_uint16(DMA_RUNNING)
        deadline = time.monotonic() + timeout_s
        while True:
            ret = self._dll.W_9812_AD_DMA_Status(
                self.card_no, ctypes.byref(status)
            )
            self._check(ret, "W_9812_AD_DMA_Status")
            if status.value == DMA_DONE:
                break
            if time.monotonic() > deadline:
                self._dll.W_9812_AD_DMA_Stop(self.card_no)
                raise TimeoutError(
                    f"DMA acquisition timed out after {timeout_s}s"
                )
            time.sleep(0.001)

        print("[PCI-9812] DMA complete — converting samples")

        # --- Extract and de-interleave channels ---
        raw_all = np.frombuffer(dma_buf, dtype=np.uint16).copy()

        result = {}
        for idx, ch in enumerate(channels):
            raw_ch = raw_all[idx::n_ch][:samples_per_channel]
            result[ch] = raw_to_voltage(raw_ch, self.vrange)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _configure_prototypes(self):
        """Set ctypes argtypes/restype for each DLL function."""
        dll = self._dll
        U16 = ctypes.c_uint16
        PU16 = ctypes.POINTER(U16)
        INT = ctypes.c_int

        dll.W_9812_Initial.argtypes = [INT, PU16, PU16, PU16, PU16]
        dll.W_9812_Initial.restype  = INT

        dll.W_9812_Close.argtypes  = [INT]
        dll.W_9812_Close.restype   = INT

        dll.W_9812_Set_Clk_Rate.argtypes = [INT, INT, INT, U16]
        dll.W_9812_Set_Clk_Rate.restype  = INT

        dll.W_9812_Set_Trig.argtypes = [INT, INT, INT, INT, U16, U16]
        dll.W_9812_Set_Trig.restype  = INT

        dll.W_9812_Alloc_DMA_Mem.argtypes = [INT, INT, ctypes.c_void_p]
        dll.W_9812_Alloc_DMA_Mem.restype  = INT

        dll.W_9812_Free_DMA_Mem.argtypes  = [ctypes.c_void_p]
        dll.W_9812_Free_DMA_Mem.restype   = INT

        dll.W_9812_AD_DMA_Start.argtypes = [INT, U16, INT, ctypes.c_void_p]
        dll.W_9812_AD_DMA_Start.restype  = INT

        dll.W_9812_AD_DMA_Status.argtypes = [INT, PU16]
        dll.W_9812_AD_DMA_Status.restype  = INT

        dll.W_9812_AD_DMA_Stop.argtypes = [INT]
        dll.W_9812_AD_DMA_Stop.restype  = INT

    @staticmethod
    def _check(ret: int, fname: str):
        if ret != 0:
            raise RuntimeError(f"{fname} failed with error code {ret:#06x}")


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------

def plot_channels(
    data: dict[int, np.ndarray],
    sample_rate_hz: float,
    vrange: float = VRANGE_5V,
    title: str = "PCI-9812 Radar Acquisition",
):
    n_ch = len(data)
    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 3 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    for ax, (ch, volts) in zip(axes, sorted(data.items())):
        t_us = np.arange(len(volts)) / sample_rate_hz * 1e6
        ax.plot(t_us, volts, linewidth=0.6)
        ax.set_ylabel(f"CH{ch} (V)")
        ax.set_ylim(-vrange * 1.05, vrange * 1.05)
        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (µs)")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main — edit parameters here
# ---------------------------------------------------------------------------

def main():
    CARD_NO           = 0               # First installed PCI-9812
    CHANNELS          = [0, 1, 2, 3]   # All 4 channels
    SAMPLES_PER_CH    = 8192            # Samples per channel (≤ 8k for 4-ch @ 20MS/s)
    SAMPLE_RATE_HZ    = 10_000_000      # 10 MS/s  (max 20 MS/s)
    VOLTAGE_RANGE     = VRANGE_5V       # ±5V  (or VRANGE_1V for ±1V)
    TRIGGER           = TRIG_SOFTWARE   # Change to TRIG_POST for hardware trigger

    print("=" * 60)
    print("ADLINK PCI-9812  |  PCIS-DASK driver")
    print("=" * 60)

    with PCI9812(card_no=CARD_NO, vrange=VOLTAGE_RANGE) as daq:
        data = daq.acquire(
            channels=CHANNELS,
            samples_per_channel=SAMPLES_PER_CH,
            sample_rate_hz=SAMPLE_RATE_HZ,
            trigger_mode=TRIGGER,
            timeout_s=5.0,
        )

    # Print quick stats per channel
    print("\nChannel statistics:")
    for ch, v in sorted(data.items()):
        print(f"  CH{ch}: min={v.min():.4f} V  max={v.max():.4f} V  "
              f"mean={v.mean():.4f} V  rms={np.sqrt(np.mean(v**2)):.4f} V")

    # Save raw numpy data
    out_file = "pci9812_data.npz"
    np.savez(out_file, **{f"ch{ch}": v for ch, v in data.items()},
             sample_rate_hz=np.float64(SAMPLE_RATE_HZ))
    print(f"\nData saved to {out_file}")

    # Plot
    plot_channels(data, SAMPLE_RATE_HZ, vrange=VOLTAGE_RANGE)


if __name__ == "__main__":
    main()
