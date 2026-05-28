"""
Furuno FAR-2xx8 Radar PPI Simulator
Simulates 20 MS/s ADC acquisition and renders a live PPI (Plan Position Indicator).

Signals simulated (matching J510 connector wiring):
  CH0 - HD    (Heading pulse: once per revolution)
  CH1 - BP    (Bearing pulse: ~357 pulses/revolution, one per ~1 degree)
  CH2 - TRIG  (Trigger: once per radar pulse, marks start of range sweep)
  CH3 - VIDEO (Radar echo amplitude vs range after each trigger)

Replace simulate_scan() return value with data from pci9812_sampler.py
to switch from simulation to live hardware.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')        # change to 'Qt5Agg' if TkAgg is not available
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# Hardware / radar constants  (Furuno FAR-2xx8, Port J510)
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE_HZ  = 20_000_000    # PCI-9812 max: 20 MS/s
RPM             = 42             # antenna rotation speed
PRF_HZ          = 2000           # pulse repetition frequency (Hz)
PULSE_WIDTH_US  = 0.3            # µs (short pulse mode)
RANGE_NM        = 3.0            # display range in nautical miles

# Scene parameters
SEA_STATE       = 3              # 0 = calm, 9 = hurricane
WAVE_DIR_DEG    = 315            # dominant wave direction (NW)

# ─────────────────────────────────────────────────────────────────────────────
# Derived constants
# ─────────────────────────────────────────────────────────────────────────────
C               = 299_792_458    # m/s
METERS_PER_NM   = 1852
MAX_RANGE_M     = RANGE_NM * METERS_PER_NM          # 5556 m
RANGE_RES_M     = C / (2 * SAMPLE_RATE_HZ)          # 7.49 m / sample
N_RANGE_BINS    = int(MAX_RANGE_M / RANGE_RES_M)    # ~741 bins

REV_PER_S       = RPM / 60                          # 0.7 rev/s
PERIOD_S        = 1.0 / REV_PER_S                   # 1.429 s / revolution
PULSES_PER_REV  = int(PRF_HZ / REV_PER_S)           # ~2857 pulses/revolution
N_BEARINGS      = 512            # angular resolution (steps per revolution)
STEPS_PER_FRAME = 6              # bearing steps advanced per animation frame


# ─────────────────────────────────────────────────────────────────────────────
# Simulated targets in the scene
# ─────────────────────────────────────────────────────────────────────────────
TARGETS = [
    {'name': 'VESSEL A', 'range_m': 2100, 'bearing_deg':  42, 'rcs': 1.0},
    {'name': 'VESSEL B', 'range_m': 3400, 'bearing_deg': 128, 'rcs': 0.75},
    {'name': 'BUOY',     'range_m':  900, 'bearing_deg': 215, 'rcs': 0.35},
    {'name': 'VESSEL C', 'range_m': 4800, 'bearing_deg': 287, 'rcs': 0.85},
]


# ─────────────────────────────────────────────────────────────────────────────
# Radar green color map  (black → dark green → bright green → white-green)
# ─────────────────────────────────────────────────────────────────────────────
RADAR_CMAP = LinearSegmentedColormap.from_list('radar_green', [
    (0.00, (0.000, 0.000, 0.000)),
    (0.15, (0.000, 0.120, 0.040)),
    (0.45, (0.000, 0.500, 0.160)),
    (0.75, (0.100, 0.820, 0.280)),
    (1.00, (0.750, 1.000, 0.650)),
])


# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute polar → Cartesian index lookup  (done once at startup)
# ─────────────────────────────────────────────────────────────────────────────
GRID_SIZE = 700

_x = np.linspace(-MAX_RANGE_M, MAX_RANGE_M, GRID_SIZE)
_y = np.linspace( MAX_RANGE_M,-MAX_RANGE_M, GRID_SIZE)   # y flipped: N at top
XX, YY = np.meshgrid(_x, _y)

RR_GRID   = np.sqrt(XX**2 + YY**2)
# Bearing: 0° = North (y-axis positive), clockwise
BB_GRID   = np.degrees(np.arctan2(XX, YY)) % 360

RANGE_IDX = np.clip((RR_GRID / RANGE_RES_M).astype(np.int32), 0, N_RANGE_BINS - 1)
BEAR_IDX  = (BB_GRID / 360.0 * N_BEARINGS).astype(np.int32) % N_BEARINGS
CIRC_MASK = RR_GRID > MAX_RANGE_M   # outside display circle


# ─────────────────────────────────────────────────────────────────────────────
# Signal simulation
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_kernel_1d(sigma, truncate=3):
    """Simple 1-D Gaussian kernel (avoids scipy dependency)."""
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _conv_rows(arr, kernel):
    """Convolve each row of a 2-D array with a 1-D kernel."""
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        out[i] = np.convolve(arr[i], kernel, mode='same')
    return out


def simulate_scan(rng=None):
    """
    Simulate one 360° radar scan.
    Returns polar array shape (N_BEARINGS, N_RANGE_BINS), values 0–1.

    To use live hardware data instead, replace the body of this function
    with a call to pci9812_sampler.acquire() and decode the Video channel
    using bearing/trigger timestamps.
    """
    if rng is None:
        rng = np.random.default_rng()

    r = np.arange(N_RANGE_BINS, dtype=np.float32)
    bearings = np.linspace(0, 360, N_BEARINGS, endpoint=False)

    # ── Sea clutter ──────────────────────────────────────────────────────────
    clutter_range = int(25 + SEA_STATE * 18)          # ~79 bins ≈ 590 m
    range_env     = (SEA_STATE / 9.0) * np.exp(-r / clutter_range)

    # Directional sea: stronger toward wave direction
    dir_mod = (0.4 + 0.6 * np.cos(np.radians(bearings - WAVE_DIR_DEG)))

    # Rayleigh speckle (realistic amplitude statistics for sea clutter)
    speckle = rng.rayleigh(1.0, (N_BEARINGS, N_RANGE_BINS)).astype(np.float32)

    sea = range_env[np.newaxis, :] * dir_mod[:, np.newaxis] * speckle

    # Smooth slightly along range (pulse broadening effect)
    sea = _conv_rows(sea, _gaussian_kernel_1d(sigma=1.5))

    # ── Wave swell pattern ───────────────────────────────────────────────────
    # Periodic returns from wave crests (every ~80-150m downwind)
    wavelength_bins = int(100 / RANGE_RES_M)        # ~13 bins ≈ 100 m swells
    wave_pattern = (0.5 + 0.5 * np.sin(2 * np.pi * r / wavelength_bins)) ** 4
    wave_dir_mod  = np.clip(
        np.cos(np.radians(bearings - WAVE_DIR_DEG)), 0, 1
    ) ** 2
    swell = (SEA_STATE / 9.0) * 0.4 * wave_pattern[np.newaxis, :] * wave_dir_mod[:, np.newaxis]

    # ── Point targets (ships / buoys) ────────────────────────────────────────
    targets = np.zeros((N_BEARINGS, N_RANGE_BINS), dtype=np.float32)

    for t in TARGETS:
        r_bin = int(t['range_m'] / RANGE_RES_M)
        b_bin = int(t['bearing_deg'] / 360 * N_BEARINGS)
        if not (0 <= r_bin < N_RANGE_BINS):
            continue
        # Spread: wider in bearing (beamwidth), narrower in range (pulse width)
        for dr in range(-4, 5):
            for db in range(-3, 4):
                rr = r_bin + dr
                bb = (b_bin + db) % N_BEARINGS
                if 0 <= rr < N_RANGE_BINS:
                    w = np.exp(-0.5 * ((dr / 2.0) ** 2 + (db / 1.5) ** 2))
                    targets[bb, rr] += t['rcs'] * w

    # ── Noise floor ──────────────────────────────────────────────────────────
    noise = rng.exponential(0.015, (N_BEARINGS, N_RANGE_BINS)).astype(np.float32)

    # ── Time-varied gain (TVG): compensate free-space loss ───────────────────
    tvg = np.clip(r / (N_RANGE_BINS * 0.15), 0, 1) ** 0.4 + 0.3
    tvg = np.clip(tvg, 0, 1)

    data = (sea + swell + targets + noise) * tvg[np.newaxis, :]

    # Normalise to 0–1
    peak = np.percentile(data, 99.5)
    return np.clip(data / (peak + 1e-9), 0, 1).astype(np.float32)


def simulate_raw_signals(bearing_idx, n_samples=2000):
    """
    Simulate the raw 20 MS/s ADC data for CH0-CH3 over a small time window.
    Returns dict {channel: voltage_array} (after 11:1 voltage divider, 0→1V).
    Used for the signal strip at the bottom of the display.
    """
    t = np.arange(n_samples) / SAMPLE_RATE_HZ   # time axis (s)
    noise = np.random.normal(0, 0.01, n_samples).astype(np.float32)

    # CH0 - HD: single wide pulse at start of revolution (only at bearing 0)
    hd_period = int(PERIOD_S * SAMPLE_RATE_HZ)
    hd = np.zeros(n_samples, dtype=np.float32)

    # CH1 - BP: square wave at bearing-pulse rate
    bp_period_s = PERIOD_S / N_BEARINGS
    bp_period   = int(bp_period_s * SAMPLE_RATE_HZ)
    bp = ((np.arange(n_samples) % max(bp_period, 1)) < (bp_period // 2)).astype(np.float32)

    # CH2 - TRIG: short trigger pulses at PRF rate
    trig_period = int(SAMPLE_RATE_HZ / PRF_HZ)
    trig_width  = int(PULSE_WIDTH_US * 1e-6 * SAMPLE_RATE_HZ)
    trig = ((np.arange(n_samples) % trig_period) < trig_width).astype(np.float32)

    # CH3 - VIDEO: radar echo after each trigger (exponential decay with speckle)
    video = np.zeros(n_samples, dtype=np.float32)
    for pulse_start in range(0, n_samples, trig_period):
        echo_len = min(trig_period - trig_width, n_samples - pulse_start - trig_width)
        if echo_len <= 0:
            continue
        idx = np.arange(echo_len)
        amplitude = np.random.rayleigh(0.25, echo_len) * np.exp(-idx / (echo_len * 0.4))
        start = pulse_start + trig_width
        end   = start + echo_len
        if end <= n_samples:
            video[start:end] += amplitude

    # Scale to ±1V (after divider)
    return {
        'HD':    np.clip(hd    * 0.9 + noise * 0.3, 0, 1),
        'BP':    np.clip(bp    * 0.9 + noise * 0.3, 0, 1),
        'TRIG':  np.clip(trig  * 0.9 + noise * 0.2, 0, 1),
        'VIDEO': np.clip(video        + noise * 0.5, -0.1, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PPI animator
# ─────────────────────────────────────────────────────────────────────────────

class PPIAnimator:
    PERSISTENCE = 0.993    # per-frame fade factor  (1.0 = no fade)

    def __init__(self):
        self.rng          = np.random.default_rng(0)
        self.polar        = np.zeros((N_BEARINGS, N_RANGE_BINS), np.float32)
        self.scan         = simulate_scan(self.rng)
        self.bearing_step = 0    # current antenna bearing (0 … N_BEARINGS-1)

    def advance(self, steps=STEPS_PER_FRAME):
        self.polar *= self.PERSISTENCE
        for _ in range(steps):
            self.polar[self.bearing_step] = self.scan[self.bearing_step]
            self.bearing_step = (self.bearing_step + 1) % N_BEARINGS
            if self.bearing_step == 0:
                self.scan = simulate_scan(self.rng)   # new scan each revolution

    @property
    def bearing_deg(self):
        return self.bearing_step / N_BEARINGS * 360.0

    def cartesian_image(self):
        img = self.polar[BEAR_IDX, RANGE_IDX]
        img = img.copy()
        img[CIRC_MASK] = 0
        return img


# ─────────────────────────────────────────────────────────────────────────────
# Build display
# ─────────────────────────────────────────────────────────────────────────────

def build_display():
    fig = plt.figure(figsize=(13, 9), facecolor='#0a0a0a')
    fig.canvas.manager.set_window_title('Furuno FAR-2xx8 — PPI Radar Simulator')

    # Layout: PPI (left, large) + signal strip (right, narrow)
    gs = fig.add_gridspec(
        5, 2,
        left=0.02, right=0.98, top=0.95, bottom=0.04,
        wspace=0.08, hspace=0.3,
        width_ratios=[2.8, 1],
        height_ratios=[1, 1, 1, 1, 0.4],
    )

    ax_ppi = fig.add_subplot(gs[:, 0], facecolor='black')
    ax_hd  = fig.add_subplot(gs[0, 1], facecolor='#0a0a0a')
    ax_bp  = fig.add_subplot(gs[1, 1], facecolor='#0a0a0a')
    ax_trg = fig.add_subplot(gs[2, 1], facecolor='#0a0a0a')
    ax_vid = fig.add_subplot(gs[3, 1], facecolor='#0a0a0a')

    # ── PPI axes ─────────────────────────────────────────────────────────────
    ax_ppi.set_aspect('equal')
    ax_ppi.set_xlim(-MAX_RANGE_M, MAX_RANGE_M)
    ax_ppi.set_ylim(-MAX_RANGE_M, MAX_RANGE_M)
    ax_ppi.axis('off')

    # Range rings + labels
    ring_step_nm = 0.5
    for r_nm in np.arange(ring_step_nm, RANGE_NM + 0.01, ring_step_nm):
        r_m = r_nm * METERS_PER_NM
        circle = plt.Circle((0, 0), r_m, color='#1a3a1a', linewidth=0.6,
                             fill=False, linestyle='--')
        ax_ppi.add_patch(circle)
        ax_ppi.text(0, r_m + MAX_RANGE_M * 0.015, f'{r_nm:.1f} NM',
                    color='#3a7a3a', fontsize=7, ha='center', va='bottom')

    # Bearing tick marks every 10°, labels every 30°
    for b in range(0, 360, 10):
        rad   = np.radians(b)
        inner = 0.93
        outer = 0.99
        x0 = MAX_RANGE_M * inner * np.sin(rad)
        y0 = MAX_RANGE_M * inner * np.cos(rad)
        x1 = MAX_RANGE_M * outer * np.sin(rad)
        y1 = MAX_RANGE_M * outer * np.cos(rad)
        ax_ppi.plot([x0, x1], [y0, y1], color='#2a5a2a', linewidth=0.8)
        if b % 30 == 0:
            xl = MAX_RANGE_M * 1.035 * np.sin(rad)
            yl = MAX_RANGE_M * 1.035 * np.cos(rad)
            ax_ppi.text(xl, yl, f'{b}°', color='#5aaa5a', fontsize=7.5,
                        ha='center', va='center')

    # Cardinal labels
    for label, bx, by in [('N', 0, 1), ('E', 1, 0), ('S', 0, -1), ('W', -1, 0)]:
        ax_ppi.text(bx * MAX_RANGE_M * 1.08, by * MAX_RANGE_M * 1.08,
                    label, color='#80e080', fontsize=10, fontweight='bold',
                    ha='center', va='center')

    # Outer border circle
    border = plt.Circle((0, 0), MAX_RANGE_M, color='#2a5a2a', linewidth=1.5,
                         fill=False)
    ax_ppi.add_patch(border)

    # Own ship marker (centre cross)
    ms = MAX_RANGE_M * 0.018
    ax_ppi.plot([-ms, ms], [0, 0], color='#40ff40', linewidth=1.5)
    ax_ppi.plot([0, 0], [-ms, ms], color='#40ff40', linewidth=1.5)
    ax_ppi.plot(0, 0, 'o', color='#40ff40', markersize=3)

    # Target labels (static annotation)
    for t in TARGETS:
        rad = np.radians(t['bearing_deg'])
        x   = t['range_m'] * np.sin(rad)
        y   = t['range_m'] * np.cos(rad)
        ax_ppi.text(x + MAX_RANGE_M * 0.025, y + MAX_RANGE_M * 0.025,
                    t['name'], color='#ffff60', fontsize=6.5,
                    fontweight='bold', alpha=0.85)

    # ── PPI image (will be updated each frame) ───────────────────────────────
    img_handle = ax_ppi.imshow(
        np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32),
        cmap=RADAR_CMAP, vmin=0, vmax=1, origin='upper',
        extent=[-MAX_RANGE_M, MAX_RANGE_M, -MAX_RANGE_M, MAX_RANGE_M],
        interpolation='bilinear', zorder=1,
    )

    # Sweep line (will be updated)
    sweep_line, = ax_ppi.plot([], [], color='#60ff60', linewidth=1.2,
                              alpha=0.9, zorder=3)

    # Sweep sector (faint glow behind the sweep line)
    sweep_sector, = ax_ppi.plot([], [], color='#20ff20', linewidth=6,
                                alpha=0.15, zorder=2)

    # Title / status text
    title_txt = ax_ppi.text(
        0, MAX_RANGE_M * 1.16,
        f'PPI RADAR SIMULATOR  |  {SAMPLE_RATE_HZ/1e6:.0f} MS/s  |  '
        f'{RANGE_NM:.0f} NM  |  {RPM} RPM',
        color='#80ff80', fontsize=9, ha='center', va='center', fontweight='bold',
    )

    bearing_txt = ax_ppi.text(
        -MAX_RANGE_M * 0.98, MAX_RANGE_M * 1.16,
        'HDG  000°', color='#60e060', fontsize=8, va='center',
    )

    # ── Signal strip axes ─────────────────────────────────────────────────────
    sig_axes  = [ax_hd, ax_bp, ax_trg, ax_vid]
    sig_names = ['CH0  HD', 'CH1  BP', 'CH2  TRIG', 'CH3  VIDEO']
    sig_cols  = ['#ff6060', '#ffff40', '#ffffff', '#40e0ff']
    sig_lines = []

    n_sig = 2000
    t_us  = np.arange(n_sig) / SAMPLE_RATE_HZ * 1e6

    for ax, name, col in zip(sig_axes, sig_names, sig_cols):
        ax.set_facecolor('#050505')
        ax.set_xlim(0, t_us[-1])
        ax.set_ylim(-0.15, 1.15)
        ax.tick_params(colors='#3a6a3a', labelsize=6)
        ax.spines[:].set_color('#1a3a1a')
        ax.set_ylabel(name, color=col, fontsize=7, rotation=0,
                      ha='right', va='center', labelpad=40)
        ax.yaxis.set_visible(False)
        ax.axhline(0, color='#1a3a1a', linewidth=0.5)
        ax.axhline(1, color='#1a3a1a', linewidth=0.5, linestyle='--')
        line, = ax.plot(t_us, np.zeros(n_sig), color=col, linewidth=0.8)
        sig_lines.append(line)

    ax_vid.set_xlabel('Time (µs)', color='#3a7a3a', fontsize=7)
    ax_vid.xaxis.set_tick_params(labelcolor='#3a7a3a')

    # ── Right-panel title ─────────────────────────────────────────────────────
    fig.text(0.72, 0.97, 'RAW ADC SIGNALS  (20 MS/s sample)',
             color='#60a060', fontsize=7.5, ha='center')
    fig.text(0.72, 0.945, 'CH0=HD  CH1=BP  CH2=TRIG  CH3=VIDEO',
             color='#3a7a3a', fontsize=6.5, ha='center')

    return fig, img_handle, sweep_line, sweep_sector, bearing_txt, sig_lines


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f'Furuno FAR-2xx8 PPI Simulator')
    print(f'  Sample rate : {SAMPLE_RATE_HZ/1e6:.0f} MS/s')
    print(f'  Range res   : {RANGE_RES_M:.1f} m/bin')
    print(f'  Range bins  : {N_RANGE_BINS}')
    print(f'  Max range   : {RANGE_NM:.1f} NM  ({MAX_RANGE_M:.0f} m)')
    print(f'  Bearings    : {N_BEARINGS} steps/revolution')
    print(f'  PRF         : {PRF_HZ} Hz  ({PULSES_PER_REV} pulses/rev)')
    print()

    ppi   = PPIAnimator()
    n_sig = 2000

    fig, img_handle, sweep_line, sweep_sector, bearing_txt, sig_lines = build_display()

    frame_counter = [0]

    def update(_frame):
        ppi.advance(STEPS_PER_FRAME)

        # ── Update PPI image ──────────────────────────────────────────────────
        img_handle.set_data(ppi.cartesian_image())

        # ── Sweep line ────────────────────────────────────────────────────────
        b_rad = np.radians(ppi.bearing_deg)
        sweep_line.set_data(
            [0, MAX_RANGE_M * 0.98 * np.sin(b_rad)],
            [0, MAX_RANGE_M * 0.98 * np.cos(b_rad)],
        )

        # Sector glow (trailing ~15°)
        sector_angles = np.radians(np.linspace(ppi.bearing_deg - 15, ppi.bearing_deg, 30))
        sweep_sector.set_data(
            MAX_RANGE_M * 0.98 * np.sin(sector_angles),
            MAX_RANGE_M * 0.98 * np.cos(sector_angles),
        )

        bearing_txt.set_text(f'HDG  {ppi.bearing_deg:05.1f}°')

        # ── Update signal strips (every 10 frames) ────────────────────────────
        if frame_counter[0] % 10 == 0:
            sigs = simulate_raw_signals(ppi.bearing_step, n_sig)
            for line, key in zip(sig_lines, ['HD', 'BP', 'TRIG', 'VIDEO']):
                line.set_ydata(sigs[key])

        frame_counter[0] += 1

        return img_handle, sweep_line, sweep_sector, bearing_txt, *sig_lines

    ani = animation.FuncAnimation(
        fig, update,
        interval=30,          # ms between frames (~33 fps)
        blit=True,
        cache_frame_data=False,
    )

    plt.show()


if __name__ == '__main__':
    main()
