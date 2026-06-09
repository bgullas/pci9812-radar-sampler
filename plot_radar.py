"""
plot_radar.py  —  Simple PPI Radar Image from PCI-9812 captured data

Reads the binary .dat file written by pci9812_sampler.py and reconstructs
a Plan Position Indicator (PPI) image:

  1. Detect HD pulses  (CH0) — each marks one full antenna revolution (North)
  2. Detect TRIG pulses (CH2) — each marks the start of one radial sweep
  3. Assign a bearing to every TRIG by linear interpolation between HD pulses
  4. Extract VIDEO (CH3) samples after each TRIG → one range profile per radial
  5. Accumulate all radials with max-hold persistence and display as a PPI

Channel wiring (Furuno FAR-2xx8, J510 connector):
  CH0 = HD    — heading / north pulse  (1 per antenna revolution)
  CH1 = BP    — bearing pulse          (1 per 0.18°)
  CH2 = TRIG  — range trigger          (1 per radial sweep)
  CH3 = VIDEO — radar echo video

Usage:
  python plot_radar.py                  # reads 9812d.dat
  python plot_radar.py myfile.dat       # reads specified file
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────────────────────
# Channel assignments
# ─────────────────────────────────────────────────────────────────────────────
CH_HD    = 0   # heading / north pulse
CH_BP    = 1   # bearing pulse (not used for bearing calc — time interpolation used)
CH_TRIG  = 2   # range trigger
CH_VIDEO = 3   # radar echo
N_CH     = 4

# ─────────────────────────────────────────────────────────────────────────────
# Hardware / capture parameters  — must match pci9812_sampler.py settings
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE_HZ = 1_000_000      # Hz
ADC_BITS       = 12
ADC_MID        = 2 ** (ADC_BITS - 1)   # 2048 for 12-bit
VRANGE         = 5.0                    # ±5 V  (AD_B_5_V)
C              = 299_792_458            # m/s
RANGE_RES_M    = C / (2 * SAMPLE_RATE_HZ)   # 149.9 m / sample

# ─────────────────────────────────────────────────────────────────────────────
# Display settings
# ─────────────────────────────────────────────────────────────────────────────
DISPLAY_RANGE_NM = 3.0           # change to match your radar's range setting
MAX_RANGE_M      = DISPLAY_RANGE_NM * 1852
METERS_PER_NM    = 1852
PPI_SIZE         = 600           # output image resolution (pixels × pixels)

# ─────────────────────────────────────────────────────────────────────────────
# Pulse detection thresholds
# ─────────────────────────────────────────────────────────────────────────────
THRESH_HD     = 0.30   # V  — rising-edge threshold for HD pulse
THRESH_TRIG   = 0.30   # V  — rising-edge threshold for TRIG pulse
THRESH_BP     = 0.30   # V  — rising-edge threshold for BP pulse

# Minimum gap between successive pulses of the same type
MIN_GAP_HD_S   = 0.50   # s  (antenna period ≈ 1.43 s at 42 RPM)
MIN_GAP_TRIG_S = 5e-4   # s  (PRF typically 600–1200 Hz → gap 0.8–1.7 ms)
MIN_GAP_BP_S   = 1e-4   # s  (BP ~2000/rev → gap ≈ 0.7 ms at 42 RPM)

# ─────────────────────────────────────────────────────────────────────────────
# Radar green colour map
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
# 1.  Load data file
# ─────────────────────────────────────────────────────────────────────────────

def load_dat(path):
    """
    Read interleaved int16 binary file → dict of float32 voltage arrays.
    Layout: [CH0, CH1, CH2, CH3, CH0, CH1, ...] (one scan per 4 values)
    """
    size_mb = os.path.getsize(path) / 1e6
    print(f'  Loading {path}  ({size_mb:.1f} MB) …', flush=True)
    raw = np.fromfile(path, dtype=np.int16)

    # Trim to a whole number of scans
    n_scans = len(raw) // N_CH
    raw     = raw[: n_scans * N_CH]

    ch = {}
    for c in range(N_CH):
        ch[c] = ((raw[c::N_CH].astype(np.float32) - ADC_MID)
                 / ADC_MID * VRANGE)

    duration = n_scans / SAMPLE_RATE_HZ
    print(f'  {n_scans:,} scans  ·  {duration:.2f} s  ·  '
          f'{n_scans / SAMPLE_RATE_HZ * 1e3:.0f} ms of data per channel')
    return ch


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Pulse detection
# ─────────────────────────────────────────────────────────────────────────────

def _rising_edges(sig, threshold, min_gap_samples):
    """Return sample indices of upward threshold crossings."""
    above = (sig > threshold).astype(np.uint8)
    edges = np.where(np.diff(above) == 1)[0]
    if len(edges) > 1:
        keep  = np.concatenate(([True], np.diff(edges) > min_gap_samples))
        edges = edges[keep]
    return edges


def detect_pulses(ch):
    """
    Detect HD, TRIG, and BP pulse positions in the captured data.
    Returns (hd_idx, trig_idx, bp_idx) as integer sample arrays.
    Prints a summary so you can verify signal quality.
    """
    n     = len(ch[CH_HD])
    dur_s = n / SAMPLE_RATE_HZ

    hd_idx   = _rising_edges(ch[CH_HD],   THRESH_HD,
                              int(MIN_GAP_HD_S   * SAMPLE_RATE_HZ))
    trig_idx = _rising_edges(ch[CH_TRIG], THRESH_TRIG,
                              int(MIN_GAP_TRIG_S * SAMPLE_RATE_HZ))
    bp_idx   = _rising_edges(ch[CH_BP],   THRESH_BP,
                              int(MIN_GAP_BP_S   * SAMPLE_RATE_HZ))

    prf   = len(trig_idx) / dur_s
    bp_pr = len(bp_idx)   / max(len(hd_idx), 1)
    rpm   = len(hd_idx)   / dur_s * 60

    print(f'  HD    : {len(hd_idx):>5} pulses  →  {rpm:.1f} RPM')
    print(f'  TRIG  : {len(trig_idx):>5} pulses  →  PRF ≈ {prf:.0f} Hz')
    print(f'  BP    : {len(bp_idx):>5} pulses  →  ~{bp_pr:.0f} per revolution')

    if len(hd_idx) == 0:
        print('\n  *** WARNING: No HD pulses detected.')
        print('      Check CH0 connection and threshold (THRESH_HD).')
        print('      Bearing will be estimated from TRIG timing only.')
    if len(trig_idx) == 0:
        print('\n  *** ERROR: No TRIG pulses detected.')
        print('      Cannot reconstruct PPI without range triggers.')
        print('      Check CH2 connection and threshold (THRESH_TRIG).')

    return hd_idx, trig_idx, bp_idx


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PPI reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def build_ppi(ch, hd_idx, trig_idx):
    """
    Build a polar backscatter array  (N_BEARINGS × n_range_bins)
    by processing every detected TRIG pulse.

    Bearing assignment:
      If HD pulses are available: bearing = (t_trig - t_hd_start)
                                            / (t_hd_end - t_hd_start) × 360°
      Fallback (no HD): uniform spread over the full dataset.

    Range bins per radial:
      Computed from median TRIG-to-TRIG interval so the display range
      automatically matches whatever range the radar was set to.

    Returns (polar, n_range_bins).
    """
    video      = ch[CH_VIDEO]
    N_total    = len(video)
    N_BEARINGS = 512     # internal polar grid resolution

    # ── Determine range bins per radial ──────────────────────────────────────
    if len(trig_idx) > 1:
        sweep_samples = int(np.median(np.diff(trig_idx)))
    else:
        # Fallback: assume PRF=1000 Hz
        sweep_samples = SAMPLE_RATE_HZ // 1000

    # Only keep samples up to the display range (don't exceed one sweep)
    bins_to_display = min(int(MAX_RANGE_M / RANGE_RES_M) + 2, sweep_samples - 2)
    n_range_bins    = max(bins_to_display, 4)

    print(f'  Sweep interval : {sweep_samples} samples  '
          f'({sweep_samples / SAMPLE_RATE_HZ * 1e3:.2f} ms  '
          f'→  unambiguous range {sweep_samples * RANGE_RES_M / METERS_PER_NM:.2f} NM)')
    print(f'  Range bins     : {n_range_bins}  '
          f'({n_range_bins * RANGE_RES_M / METERS_PER_NM:.2f} NM displayed)')

    # ── Build polar accumulator ───────────────────────────────────────────────
    polar      = np.zeros((N_BEARINGS, n_range_bins), dtype=np.float32)
    n_radials  = 0

    # ── HD-bounded revolutions ────────────────────────────────────────────────
    if len(hd_idx) >= 2:
        for rev in range(len(hd_idx) - 1):
            t0 = int(hd_idx[rev])
            t1 = int(hd_idx[rev + 1])
            rev_dur = t1 - t0

            # TRIG pulses inside this revolution
            mask  = (trig_idx >= t0) & (trig_idx < t1)
            trigs = trig_idx[mask]
            if len(trigs) == 0:
                continue

            for ti in trigs:
                # Bearing: 0° (North) at HD pulse, grows clockwise
                bearing_frac = (int(ti) - t0) / rev_dur
                b_idx = int(bearing_frac * N_BEARINGS) % N_BEARINGS

                _accumulate(polar, video, ti, n_range_bins, b_idx, N_total)
                n_radials += 1

    else:
        # No HD pulses: spread all TRIG pulses uniformly over 360°
        print('  (no HD pulses — spreading all radials uniformly over 360°)')
        n_t = len(trig_idx)
        for i, ti in enumerate(trig_idx):
            b_idx = int(i / n_t * N_BEARINGS) % N_BEARINGS
            _accumulate(polar, video, ti, n_range_bins, b_idx, N_total)
            n_radials += 1

    print(f'  Accumulated    : {n_radials} radials  '
          f'({len(hd_idx) - 1 if len(hd_idx) >= 2 else 0} revolutions)')

    # ── Normalise ─────────────────────────────────────────────────────────────
    p99 = float(np.percentile(polar[polar > 0], 99)) if np.any(polar > 0) else 1.0
    polar = np.clip(polar / (p99 + 1e-9), 0.0, 1.0)

    return polar, n_range_bins


def _accumulate(polar, video, ti, n_range_bins, b_idx, N_total):
    """
    Extract one radial from VIDEO after sample ti, rectify, and max-hold
    into the polar buffer at bearing index b_idx.
    """
    end    = min(int(ti) + n_range_bins, N_total)
    n      = end - int(ti)
    if n < 3:
        return
    radial = video[int(ti): end].copy()

    # Rectify: radar video is a positive-envelope signal;
    # remove near-range DC and keep only positive excursions
    dc     = float(radial[:min(5, n)].mean())
    radial = np.clip(radial - dc, 0.0, None)

    if n < n_range_bins:
        radial = np.pad(radial, (0, n_range_bins - n))

    # Max-hold persistence across revolutions
    np.maximum(polar[b_idx], radial.astype(np.float32), out=polar[b_idx])


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Polar → Cartesian rendering
# ─────────────────────────────────────────────────────────────────────────────

def polar_to_cart(polar, n_range_bins):
    """
    Re-sample the polar backscatter grid onto a PPI_SIZE × PPI_SIZE
    Cartesian image using pre-computed nearest-neighbour lookup tables.
    """
    px   = np.linspace(-MAX_RANGE_M,  MAX_RANGE_M, PPI_SIZE)
    py   = np.linspace( MAX_RANGE_M, -MAX_RANGE_M, PPI_SIZE)  # flipped for imshow
    PX, PY = np.meshgrid(px, py)
    PR   = np.sqrt(PX**2 + PY**2)
    PB   = np.degrees(np.arctan2(PX, PY)) % 360   # 0=North, clockwise

    N_BEARINGS = polar.shape[0]
    ridx = np.clip((PR / RANGE_RES_M).astype(np.int32), 0, n_range_bins - 1)
    bidx = (PB / 360.0 * N_BEARINGS).astype(np.int32) % N_BEARINGS

    img         = polar[bidx, ridx]
    img[PR > MAX_RANGE_M] = 0.0
    return img.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_ppi(img, n_radials, n_revs, prf_hz):
    fig, ax = plt.subplots(figsize=(9, 9), facecolor='black')
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Radar image ────────────────────────────────────────────────────────────
    ax.imshow(
        img, cmap=RADAR_CMAP, vmin=0, vmax=1,
        origin='upper',
        extent=[-MAX_RANGE_M, MAX_RANGE_M, -MAX_RANGE_M, MAX_RANGE_M],
        interpolation='bilinear',
    )

    # ── Range rings ────────────────────────────────────────────────────────────
    theta = np.linspace(0, 2 * np.pi, 360)
    for r_nm in np.arange(0.5, DISPLAY_RANGE_NM + 0.01, 0.5):
        r_m = r_nm * METERS_PER_NM
        ax.plot(r_m * np.sin(theta), r_m * np.cos(theta),
                color='#1a381a', lw=0.7, ls='--', zorder=2)
        # Range label at 35° bearing
        lx = r_m * np.sin(np.radians(35))
        ly = r_m * np.cos(np.radians(35))
        ax.text(lx, ly, f'{r_m:.0f} m',
                color='#2a542a', fontsize=7, ha='center', va='center', zorder=3)

    # Outer hard ring
    ax.plot(MAX_RANGE_M * np.sin(theta), MAX_RANGE_M * np.cos(theta),
            color='#2a4a2a', lw=1.4, zorder=2)

    # ── Bearing spokes every 30° ───────────────────────────────────────────────
    for b_deg in range(0, 360, 30):
        bx = MAX_RANGE_M * np.sin(np.radians(b_deg))
        by = MAX_RANGE_M * np.cos(np.radians(b_deg))
        ax.plot([0, bx * 0.97], [0, by * 0.97],
                color='#162416', lw=0.5, zorder=2)
        ax.text(bx * 1.06, by * 1.06, f'{b_deg}°',
                color='#304830', fontsize=6.5, ha='center', va='center', zorder=3)

    # ── North indicator (bright line) ─────────────────────────────────────────
    ax.plot([0, 0], [0, MAX_RANGE_M * 0.97],
            color='#40c040', lw=1.2, ls=':', zorder=4, alpha=0.7)

    # ── Cardinal labels ────────────────────────────────────────────────────────
    for lbl, bx, by in [('N', 0, 1.09), ('S', 0, -1.09),
                         ('E', 1.09, 0), ('W', -1.09, 0)]:
        ax.text(bx * MAX_RANGE_M, by * MAX_RANGE_M, lbl,
                color='#58b858', fontsize=11, fontweight='bold',
                ha='center', va='center', zorder=5)

    # ── Centre dot ────────────────────────────────────────────────────────────
    ax.plot(0, 0, 'o', color='#60d060', ms=4, zorder=6)

    # ── Titles ────────────────────────────────────────────────────────────────
    fig.suptitle(
        f'PPI  —  Furuno FAR-2xx8 / PCI-9812',
        color='#70d070', fontsize=13, fontweight='bold', y=0.97,
    )
    fig.text(
        0.50, 0.015,
        f'{n_revs} revolutions  ·  {n_radials} radials  ·  '
        f'PRF ≈ {prf_hz:.0f} Hz  ·  '
        f'{SAMPLE_RATE_HZ / 1e6:.1f} MS/s  ·  '
        f'{RANGE_RES_M:.0f} m/bin  ·  '
        f'display {DISPLAY_RANGE_NM:.1f} NM',
        color='#3a5a3a', fontsize=8, ha='center',
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Resolve file path ─────────────────────────────────────────────────────
    dat_file = sys.argv[1] if len(sys.argv) > 1 else '9812d.dat'
    if not os.path.isfile(dat_file):
        alt = dat_file + '.dat'
        if os.path.isfile(alt):
            dat_file = alt
        else:
            print(f'ERROR: data file not found — {dat_file}')
            print()
            print('Usage:  python plot_radar.py [datafile.dat]')
            print('        (default: 9812d.dat in the current directory)')
            sys.exit(1)

    print('─' * 60)
    print('PPI Radar Image Reconstruction')
    print('─' * 60)
    print(f'  File        : {os.path.abspath(dat_file)}')
    print(f'  Sample rate : {SAMPLE_RATE_HZ / 1e6:.1f} MHz  →  '
          f'{RANGE_RES_M:.1f} m/bin')
    print(f'  Display     : {DISPLAY_RANGE_NM:.1f} NM  '
          f'({DISPLAY_RANGE_NM * METERS_PER_NM:.0f} m)')
    print()

    # ── Load ──────────────────────────────────────────────────────────────────
    ch = load_dat(dat_file)
    print()

    # ── Detect pulses ─────────────────────────────────────────────────────────
    print('Detecting pulses …')
    hd_idx, trig_idx, bp_idx = detect_pulses(ch)

    if len(trig_idx) == 0:
        print('\nAborting — no TRIG pulses found. Nothing to reconstruct.')
        sys.exit(1)

    prf_hz = len(trig_idx) / (len(ch[CH_TRIG]) / SAMPLE_RATE_HZ)
    n_revs = max(len(hd_idx) - 1, 0)
    print()

    # ── Build PPI ─────────────────────────────────────────────────────────────
    print('Building PPI …')
    polar, n_range_bins = build_ppi(ch, hd_idx, trig_idx)
    img = polar_to_cart(polar, n_range_bins)
    n_radials = int(np.sum(polar.max(axis=1) > 0))   # bearings with data
    print()

    # ── Plot ─────────────────────────────────────────────────────────────────
    print('Rendering …')
    plot_ppi(img, n_radials, n_revs, prf_hz)


if __name__ == '__main__':
    main()
