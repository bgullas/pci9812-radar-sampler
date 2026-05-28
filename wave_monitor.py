"""
wave_monitor.py  —  Sea Wave Monitoring from Radar Backscatter

Technique: wave crests reflect more radar energy than troughs. The periodic
amplitude pattern in the range profile encodes wavelength and propagation.

Four analysis panels:
  1. PPI          — full radar image, wave monitoring sector highlighted
  2. Wave Profile — 1D amplitude slice along wave direction, peaks counted
  3. Temporal     — amplitude at fixed range over time, shows wave period
  4. Waterfall    — range × time, diagonal bands = wave propagation speed
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
# Radar / hardware constants
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE_HZ = 20_000_000
RPM            = 42
RANGE_NM       = 3.0

C              = 299_792_458
METERS_PER_NM  = 1852
MAX_RANGE_M    = RANGE_NM * METERS_PER_NM        # 5 556 m
RANGE_RES_M    = C / (2 * SAMPLE_RATE_HZ)        # 7.49 m / bin
N_RANGE_BINS   = int(MAX_RANGE_M / RANGE_RES_M)  # ~741
REV_PER_S      = RPM / 60                        # 0.7
PERIOD_S       = 1.0 / REV_PER_S                 # 1.43 s / rev
N_BEARINGS     = 512

# ─────────────────────────────────────────────────────────────────────────────
# Wave field definition
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: wavelength (m), origin direction (deg, where waves come FROM),
# relative amplitude (0–1), period (s).
# Deep-water dispersion: T = 2π sqrt(λ / (2π g))
WAVE_PARAMS = [
    {'λ': 120, 'dir': 315, 'amp': 0.70, 'T':  8.8},  # primary NW swell
    {'λ':  65, 'dir': 290, 'amp': 0.30, 'T':  6.5},  # secondary W swell
    {'λ':  28, 'dir': 330, 'amp': 0.15, 'T':  4.2},  # short wind chop
]

DOMINANT_WAVE = WAVE_PARAMS[0]    # used for profile / counting
SEA_CLUTTER   = 0.18              # background clutter level (0–1)
SIM_SPEED     = 4                 # simulated seconds per real revolution
                                  # (increase to speed up wave motion)

MONITOR_RANGE_M   = 1500          # fixed range for temporal trace (m)
MONITOR_BEARING   = DOMINANT_WAVE['dir']   # look along wave origin direction
N_HISTORY_REVS    = 55            # waterfall rows (revolutions kept)
N_TRACE_REVS      = 70            # temporal trace length (revolutions)

# ─────────────────────────────────────────────────────────────────────────────
# Color maps
# ─────────────────────────────────────────────────────────────────────────────
RADAR_CMAP = LinearSegmentedColormap.from_list('radar_green', [
    (0.00, (0.00, 0.00, 0.00)),
    (0.15, (0.00, 0.12, 0.04)),
    (0.45, (0.00, 0.50, 0.16)),
    (0.75, (0.10, 0.82, 0.28)),
    (1.00, (0.75, 1.00, 0.65)),
])

WATER_CMAP = LinearSegmentedColormap.from_list('water', [
    (0.00, (0.00, 0.03, 0.12)),
    (0.35, (0.00, 0.20, 0.50)),
    (0.65, (0.00, 0.55, 0.85)),
    (0.85, (0.40, 0.85, 1.00)),
    (1.00, (1.00, 1.00, 1.00)),
])

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute polar → Cartesian lookup for PPI
# ─────────────────────────────────────────────────────────────────────────────
PPI_SIZE   = 420
_x = np.linspace(-MAX_RANGE_M,  MAX_RANGE_M, PPI_SIZE)
_y = np.linspace( MAX_RANGE_M, -MAX_RANGE_M, PPI_SIZE)
_XX, _YY   = np.meshgrid(_x, _y)
_RR        = np.sqrt(_XX**2 + _YY**2)
_BB        = np.degrees(np.arctan2(_XX, _YY)) % 360
_RIDX      = np.clip((_RR / RANGE_RES_M).astype(np.int32), 0, N_RANGE_BINS - 1)
_BIDX      = (_BB / 360.0 * N_BEARINGS).astype(np.int32) % N_BEARINGS
_MASK      = _RR > MAX_RANGE_M

RANGE_AXIS = np.arange(N_RANGE_BINS) * RANGE_RES_M   # metres


# ─────────────────────────────────────────────────────────────────────────────
# Wave field generation
# ─────────────────────────────────────────────────────────────────────────────

def _wave_echo_2d(sim_time_s, rng):
    """
    Return polar array (N_BEARINGS × N_RANGE_BINS) of wave backscatter.

    Physics: η(r, θ, t) = Σ Aᵢ · clip(cos(−2π/λᵢ · r·cos(θ−dirᵢ) − 2π/Tᵢ · t), 0)²
    Only wave crests (positive surface elevation) produce strong backscatter.
    """
    bearings = np.linspace(0, 2 * np.pi, N_BEARINGS, endpoint=False)
    r        = np.arange(N_RANGE_BINS, dtype=np.float32)

    wave = np.zeros((N_BEARINGS, N_RANGE_BINS), dtype=np.float32)
    for w in WAVE_PARAMS:
        k    = 2 * np.pi / w['λ']
        om   = 2 * np.pi / w['T']
        orig = np.radians(w['dir'])
        # Projection: -r·cos(bearing − origin)  →  crests approach radar
        cos_fac = -np.cos(bearings - orig)          # (N_BEARINGS,)
        phase   = k * r[np.newaxis, :] * cos_fac[:, np.newaxis] - om * sim_time_s
        crest   = np.clip(np.cos(phase).astype(np.float32), 0, None) ** 2
        wave   += w['amp'] * crest

    # Add sea clutter (Rayleigh speckle, range-decaying)
    r_decay  = np.exp(-r / 120).astype(np.float32)
    speckle  = rng.rayleigh(1.0, (N_BEARINGS, N_RANGE_BINS)).astype(np.float32)
    clutter  = SEA_CLUTTER * r_decay[np.newaxis, :] * speckle

    data = wave + clutter + rng.normal(0, 0.02, wave.shape).astype(np.float32)
    peak = np.percentile(data, 99.0)
    return np.clip(data / (peak + 1e-9), 0, 1)


def _wave_profile_1d(polar_data, bearing_deg):
    """Extract 1D range profile at given bearing. Returns (N_RANGE_BINS,) array."""
    b_idx = int(bearing_deg / 360 * N_BEARINGS) % N_BEARINGS
    # Average ±2 bearing bins for smoother profile
    indices = [(b_idx + d) % N_BEARINGS for d in range(-2, 3)]
    return polar_data[indices, :].mean(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Peak detection  (pure numpy — no scipy)
# ─────────────────────────────────────────────────────────────────────────────

def _smooth(x, w=7):
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='same')


def find_wave_peaks(profile, min_height=0.25):
    """
    Detect wave crests in a 1-D range profile.
    Returns array of bin indices where peaks occur.
    min_distance is auto-set to 60% of the dominant wavelength.
    """
    s     = _smooth(profile, w=5)
    min_d = max(4, int(DOMINANT_WAVE['λ'] * 0.6 / RANGE_RES_M))
    peaks = []
    for i in range(1, len(s) - 1):
        if s[i] > s[i - 1] and s[i] >= s[i + 1] and s[i] >= min_height:
            if not peaks or (i - peaks[-1]) >= min_d:
                peaks.append(i)
    return np.array(peaks, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# Wave statistics
# ─────────────────────────────────────────────────────────────────────────────

def _wave_stats(peaks, temporal_trace):
    """
    Return dict of wave statistics from peak positions and temporal data.
    """
    stats = {}

    # Wavelength from spatial peak spacing
    if len(peaks) >= 2:
        spacings = np.diff(peaks) * RANGE_RES_M
        stats['lambda_m']    = float(np.median(spacings))
        stats['wave_count']  = len(peaks)
    else:
        stats['lambda_m']    = DOMINANT_WAVE['λ']
        stats['wave_count']  = len(peaks)

    # Theoretical period from deep-water dispersion: T = 2π√(λ/2πg)
    g = 9.81
    lam = stats['lambda_m']
    stats['period_s_theory'] = 2 * np.pi * np.sqrt(lam / (2 * np.pi * g))

    # Phase speed
    stats['celerity_ms'] = lam / stats['period_s_theory']

    # Measured period from temporal trace (count zero-crossings of mean-removed signal)
    if len(temporal_trace) > 10:
        tr    = np.array(temporal_trace)
        tr_dm = tr - tr.mean()
        zc    = np.where(np.diff(np.sign(tr_dm)))[0]
        if len(zc) >= 2:
            avg_half_period_revs = np.mean(np.diff(zc))
            period_revs          = 2 * avg_half_period_revs
            stats['period_revs'] = float(period_revs)
            stats['period_s_meas'] = period_revs * PERIOD_S * SIM_SPEED
        else:
            stats['period_revs']   = stats['period_s_theory'] / (PERIOD_S * SIM_SPEED)
            stats['period_s_meas'] = stats['period_s_theory']
    else:
        stats['period_revs']   = stats['period_s_theory'] / (PERIOD_S * SIM_SPEED)
        stats['period_s_meas'] = stats['period_s_theory']

    stats['swh_estimate'] = f'~{SEA_CLUTTER * 5:.1f}–{SEA_CLUTTER * 8:.1f} m'

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Simulator state
# ─────────────────────────────────────────────────────────────────────────────

class WaveSimulator:
    PERSIST = 0.992

    def __init__(self):
        self.rng          = np.random.default_rng(7)
        self.sim_time     = 0.0
        self.polar        = np.zeros((N_BEARINGS, N_RANGE_BINS), np.float32)
        self.scan         = _wave_echo_2d(self.sim_time, self.rng)
        self.bearing_step = 0
        self.rev_count    = 0

        self.waterfall    = deque(maxlen=N_HISTORY_REVS)
        self.temporal     = deque(maxlen=N_TRACE_REVS)
        self.monitor_bin  = int(MONITOR_RANGE_M / RANGE_RES_M)

    # ── Advance one animation frame ────────────────────────────────────────

    def step(self, steps=6):
        self.polar *= self.PERSIST
        for _ in range(steps):
            self.polar[self.bearing_step] = self.scan[self.bearing_step]
            self.bearing_step = (self.bearing_step + 1) % N_BEARINGS
            if self.bearing_step == 0:
                self._on_revolution()

    def _on_revolution(self):
        self.sim_time += PERIOD_S * SIM_SPEED
        self.rev_count += 1
        self.scan = _wave_echo_2d(self.sim_time, self.rng)

        # Record wave profile for waterfall
        profile = _wave_profile_1d(self.scan, MONITOR_BEARING)
        self.waterfall.append(profile.copy())

        # Record amplitude at monitoring range for temporal trace
        self.temporal.append(float(profile[self.monitor_bin]))

    # ── Accessors ──────────────────────────────────────────────────────────

    @property
    def bearing_deg(self):
        return self.bearing_step / N_BEARINGS * 360.0

    def ppi_image(self):
        img = self.polar[_BIDX, _RIDX].copy()
        img[_MASK] = 0
        return img

    def current_profile(self):
        return _wave_profile_1d(self.polar, MONITOR_BEARING)

    def waterfall_image(self):
        if len(self.waterfall) < 2:
            return np.zeros((N_HISTORY_REVS, N_RANGE_BINS))
        mat = np.array(self.waterfall)
        # Pad top if not full yet
        if mat.shape[0] < N_HISTORY_REVS:
            pad = np.zeros((N_HISTORY_REVS - mat.shape[0], N_RANGE_BINS))
            mat = np.vstack([pad, mat])
        return mat


# ─────────────────────────────────────────────────────────────────────────────
# Build display
# ─────────────────────────────────────────────────────────────────────────────

def build_display(sim):
    fig = plt.figure(figsize=(15, 9), facecolor='#060810')
    fig.canvas.manager.set_window_title('Wave Monitoring — Radar Backscatter Analysis')

    gs = fig.add_gridspec(
        3, 3,
        left=0.05, right=0.98, top=0.93, bottom=0.06,
        wspace=0.32, hspace=0.45,
        width_ratios=[1.1, 1.4, 1.0],
        height_ratios=[1.3, 1.0, 1.0],
    )

    ax_ppi    = fig.add_subplot(gs[0:2, 0], facecolor='black')   # PPI (tall)
    ax_prof   = fig.add_subplot(gs[0, 1:], facecolor='#060810')  # Wave profile
    ax_temp   = fig.add_subplot(gs[1, 1],  facecolor='#060810')  # Temporal trace
    ax_stats  = fig.add_subplot(gs[1, 2],  facecolor='#060810')  # Statistics
    ax_water  = fig.add_subplot(gs[2, :],  facecolor='#060810')  # Waterfall

    # ── PPI static ────────────────────────────────────────────────────────────
    ax_ppi.set_aspect('equal')
    ax_ppi.set_xlim(-MAX_RANGE_M, MAX_RANGE_M)
    ax_ppi.set_ylim(-MAX_RANGE_M, MAX_RANGE_M)
    ax_ppi.axis('off')
    ax_ppi.set_title('PPI  —  Sea Surface', color='#70e070', fontsize=8, pad=4)

    for r_nm in np.arange(0.5, RANGE_NM + 0.01, 0.5):
        ax_ppi.add_patch(plt.Circle((0, 0), r_nm * METERS_PER_NM,
                                    color='#182818', lw=0.6, fill=False, ls='--'))
    ax_ppi.add_patch(plt.Circle((0, 0), MAX_RANGE_M,
                                color='#2a4a2a', lw=1.2, fill=False))
    for lbl, bx, by in [('N', 0, 1), ('E', 1, 0), ('S', 0, -1), ('W', -1, 0)]:
        ax_ppi.text(bx * MAX_RANGE_M * 1.10, by * MAX_RANGE_M * 1.10,
                    lbl, color='#50a050', fontsize=8, fontweight='bold',
                    ha='center', va='center')

    # Wave sector wedge  (shows the monitoring bearing)
    sector_half = 20
    b_lo = np.radians(MONITOR_BEARING - sector_half)
    b_hi = np.radians(MONITOR_BEARING + sector_half)
    angles = np.linspace(b_lo, b_hi, 60)
    sx = np.concatenate([[0], MAX_RANGE_M * np.sin(angles), [0]])
    sy = np.concatenate([[0], MAX_RANGE_M * np.cos(angles), [0]])
    ax_ppi.fill(sx, sy, color='#003366', alpha=0.22, zorder=1)
    ax_ppi.plot([0, MAX_RANGE_M * np.sin(np.radians(MONITOR_BEARING))],
                [0, MAX_RANGE_M * np.cos(np.radians(MONITOR_BEARING))],
                color='#4080ff', lw=0.8, ls='--', alpha=0.6, zorder=2)
    ax_ppi.text(MAX_RANGE_M * 0.55 * np.sin(np.radians(MONITOR_BEARING + 12)),
                MAX_RANGE_M * 0.55 * np.cos(np.radians(MONITOR_BEARING + 12)),
                'WAVE\nSECTOR', color='#4080ff', fontsize=6.5, ha='center')

    # Monitoring range circle
    ax_ppi.add_patch(plt.Circle((0, 0), MONITOR_RANGE_M,
                                color='#ff8040', lw=0.8, fill=False,
                                linestyle=':', alpha=0.7))
    ax_ppi.text(MONITOR_RANGE_M * np.sin(np.radians(55)),
                MONITOR_RANGE_M * np.cos(np.radians(55)),
                f'{MONITOR_RANGE_M} m', color='#ff8040', fontsize=6)

    ppi_img = ax_ppi.imshow(
        np.zeros((PPI_SIZE, PPI_SIZE), np.float32),
        cmap=RADAR_CMAP, vmin=0, vmax=1, origin='upper',
        extent=[-MAX_RANGE_M, MAX_RANGE_M, -MAX_RANGE_M, MAX_RANGE_M],
        interpolation='bilinear', zorder=0,
    )
    sweep_ln, = ax_ppi.plot([], [], color='#80ff80', lw=1.1, alpha=0.9, zorder=3)

    # ── Wave profile ──────────────────────────────────────────────────────────
    ax_prof.set_facecolor('#04060e')
    ax_prof.set_xlim(0, MAX_RANGE_M)
    ax_prof.set_ylim(-0.05, 1.15)
    ax_prof.set_xlabel('Range (m)', color='#5080a0', fontsize=8)
    ax_prof.set_ylabel('Backscatter amplitude', color='#5080a0', fontsize=8)
    ax_prof.set_title(f'Wave Profile  —  bearing {MONITOR_BEARING:.0f}°  (wave origin direction)',
                      color='#80b0d0', fontsize=8, pad=3)
    ax_prof.tick_params(colors='#304060', labelsize=7)
    ax_prof.spines[:].set_color('#1a2a3a')
    ax_prof.axhline(0, color='#1a2a3a', lw=0.5)
    ax_prof.axhline(0.25, color='#1a2a3a', lw=0.4, ls=':')   # detection threshold

    prof_line,   = ax_prof.plot(RANGE_AXIS, np.zeros(N_RANGE_BINS),
                                color='#60b0ff', lw=1.0, zorder=3)
    prof_fill    = ax_prof.fill_between(RANGE_AXIS, 0,
                                        np.zeros(N_RANGE_BINS),
                                        color='#0030a0', alpha=0.3, zorder=2)
    # Monitoring range vertical line
    ax_prof.axvline(MONITOR_RANGE_M, color='#ff8040', lw=0.8, ls=':',
                    alpha=0.7, label=f'Monitor point ({MONITOR_RANGE_M} m)')
    ax_prof.legend(loc='upper right', fontsize=6.5,
                   facecolor='#060810', edgecolor='#304060', labelcolor='#ff8040')

    # Peak markers (max N_RANGE_BINS // 5 possible peaks)
    max_peaks = 60
    peak_lines  = [ax_prof.axvline(0, color='#ff4040', lw=0.9, ls='--',
                                   alpha=0.0, zorder=4) for _ in range(max_peaks)]
    peak_labels = [ax_prof.text(0, 1.02, '', color='#ff6060', fontsize=6,
                                ha='center', rotation=90, va='bottom', zorder=5)
                   for _ in range(max_peaks)]

    # Wavelength annotation (a double-headed arrow between peak 0 and peak 1)
    wave_annot = ax_prof.annotate(
        '', xy=(0, 0.85), xytext=(0, 0.85),
        arrowprops=dict(arrowstyle='<->', color='#ffff60', lw=1.0),
    )
    lambda_label = ax_prof.text(0, 0.90, '', color='#ffff60', fontsize=7,
                                ha='center', fontweight='bold')

    wave_count_txt = ax_prof.text(
        0.98, 0.95, '', transform=ax_prof.transAxes,
        color='#ffffa0', fontsize=9, fontweight='bold',
        ha='right', va='top',
    )

    # ── Temporal trace ────────────────────────────────────────────────────────
    ax_temp.set_facecolor('#04060e')
    ax_temp.set_xlim(0, N_TRACE_REVS)
    ax_temp.set_ylim(-0.05, 1.15)
    ax_temp.set_xlabel('Antenna revolutions', color='#5080a0', fontsize=8)
    ax_temp.set_ylabel('Amplitude', color='#5080a0', fontsize=8)
    ax_temp.set_title(f'Temporal Trace  @  {MONITOR_RANGE_M} m  ·  {MONITOR_BEARING:.0f}°',
                      color='#80b0d0', fontsize=8, pad=3)
    ax_temp.tick_params(colors='#304060', labelsize=7)
    ax_temp.spines[:].set_color('#1a2a3a')
    ax_temp.axhline(0, color='#1a2a3a', lw=0.5)

    temp_line, = ax_temp.plot([], [], color='#40e0c0', lw=1.0)
    temp_fill  = [None]   # mutable placeholder (fill_between can't be updated directly)
    period_txt = ax_temp.text(0.02, 0.92, '', transform=ax_temp.transAxes,
                              color='#ffffa0', fontsize=8, va='top')

    # ── Statistics panel ──────────────────────────────────────────────────────
    ax_stats.set_facecolor('#04060e')
    ax_stats.axis('off')
    ax_stats.set_title('Wave Statistics', color='#80b0d0', fontsize=8, pad=3)
    stats_text = ax_stats.text(
        0.05, 0.95, 'Initialising…',
        transform=ax_stats.transAxes,
        color='#c0d8f0', fontsize=8.5, va='top', family='monospace',
        linespacing=1.8,
    )

    # ── Waterfall ─────────────────────────────────────────────────────────────
    ax_water.set_facecolor('#04060e')
    ax_water.set_title(
        f'Range–Time Waterfall  —  bearing {MONITOR_BEARING:.0f}°  '
        f'(diagonal bands = wave propagation toward radar)',
        color='#80b0d0', fontsize=8, pad=3,
    )
    ax_water.set_xlabel('Range (m)', color='#5080a0', fontsize=8)
    ax_water.set_ylabel('Time  (revolutions, newest→top)',
                        color='#5080a0', fontsize=8)
    ax_water.tick_params(colors='#304060', labelsize=7)
    ax_water.spines[:].set_color('#1a2a3a')

    water_img = ax_water.imshow(
        np.zeros((N_HISTORY_REVS, N_RANGE_BINS)),
        cmap=WATER_CMAP, vmin=0, vmax=1, origin='lower',
        aspect='auto',
        extent=[0, MAX_RANGE_M, 0, N_HISTORY_REVS],
        interpolation='nearest',
    )
    ax_water.set_xlim(0, MAX_RANGE_M)
    ax_water.set_ylim(0, N_HISTORY_REVS)

    # Monitoring range line on waterfall
    ax_water.axvline(MONITOR_RANGE_M, color='#ff8040', lw=0.9, ls=':',
                     alpha=0.7, label=f'{MONITOR_RANGE_M} m monitor')
    ax_water.legend(loc='upper right', fontsize=6.5,
                    facecolor='#060810', edgecolor='#304060', labelcolor='#ff8040')

    # Propagation speed annotation on waterfall (updated each frame)
    prop_annot = ax_water.annotate(
        '', xy=(0, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color='#ffff80', lw=1.2),
        annotation_clip=False,
    )
    prop_label = ax_water.text(0, 0, '', color='#ffff80', fontsize=7)

    # ── Figure title ──────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.975,
        f'WAVE MONITORING  ·  Furuno FAR-2xx8 / PCI-9812  ·  '
        f'{SAMPLE_RATE_HZ/1e6:.0f} MS/s  ·  {RANGE_NM:.0f} NM  ·  '
        f'Primary swell λ={DOMINANT_WAVE["λ"]} m  from {DOMINANT_WAVE["dir"]:.0f}°',
        color='#90d090', fontsize=8.5, ha='center', fontweight='bold',
    )

    artists = dict(
        ppi_img=ppi_img, sweep_ln=sweep_ln,
        prof_line=prof_line, prof_fill=prof_fill,
        peak_lines=peak_lines, peak_labels=peak_labels,
        wave_annot=wave_annot, lambda_label=lambda_label,
        wave_count_txt=wave_count_txt,
        temp_line=temp_line, temp_fill=temp_fill,
        period_txt=period_txt,
        stats_text=stats_text,
        water_img=water_img,
        prop_annot=prop_annot, prop_label=prop_label,
        ax_prof=ax_prof, ax_temp=ax_temp, ax_water=ax_water,
    )
    return fig, artists


# ─────────────────────────────────────────────────────────────────────────────
# Animation update
# ─────────────────────────────────────────────────────────────────────────────

def make_updater(sim, arts):
    frame_ctr = [0]

    def update(_frame):
        sim.step(steps=6)
        fc = frame_ctr[0]
        frame_ctr[0] += 1

        # ── PPI ──────────────────────────────────────────────────────────────
        arts['ppi_img'].set_data(sim.ppi_image())
        b_rad = np.radians(sim.bearing_deg)
        arts['sweep_ln'].set_data(
            [0, MAX_RANGE_M * 0.97 * np.sin(b_rad)],
            [0, MAX_RANGE_M * 0.97 * np.cos(b_rad)],
        )

        # Heavy updates every 4 frames
        if fc % 4 != 0:
            return

        # ── Wave profile ─────────────────────────────────────────────────────
        profile = sim.current_profile()
        arts['prof_line'].set_ydata(profile)

        # Update fill (must remove old and redraw)
        if arts['prof_fill']:
            try:
                arts['prof_fill'].remove()
            except Exception:
                pass
        arts['prof_fill'] = arts['ax_prof'].fill_between(
            RANGE_AXIS, 0, profile, color='#0030a0', alpha=0.25, zorder=2
        )

        # Peak detection
        peaks = find_wave_peaks(profile)

        # Hide all peak markers first
        for ln in arts['peak_lines']:
            ln.set_alpha(0.0)
        for lb in arts['peak_labels']:
            lb.set_text('')

        # Show detected peaks
        for i, pk in enumerate(peaks[:len(arts['peak_lines'])]):
            r_m = RANGE_AXIS[pk]
            arts['peak_lines'][i].set_xdata([r_m, r_m])
            arts['peak_lines'][i].set_alpha(0.70)
            arts['peak_labels'][i].set_position((r_m, 1.03))
            arts['peak_labels'][i].set_text(f'W{i+1}\n{r_m:.0f}m')

        # Wavelength double-arrow between first two peaks
        if len(peaks) >= 2:
            r0 = RANGE_AXIS[peaks[0]]
            r1 = RANGE_AXIS[peaks[1]]
            lam_meas = r1 - r0
            arts['wave_annot'].xy     = (r1, 0.82)
            arts['wave_annot'].xytext = (r0, 0.82)
            arts['wave_annot'].get_figure()   # keep ref alive
            arts['lambda_label'].set_position(((r0 + r1) / 2, 0.87))
            arts['lambda_label'].set_text(f'λ ≈ {lam_meas:.0f} m')
        else:
            arts['lambda_label'].set_text('')

        arts['wave_count_txt'].set_text(
            f'{len(peaks)} waves\nin view'
        )

        # ── Temporal trace ────────────────────────────────────────────────────
        trace = np.array(sim.temporal)
        if len(trace) > 1:
            x_vals = np.arange(len(trace))
            arts['temp_line'].set_data(x_vals, trace)
            arts['ax_temp'].set_xlim(0, max(N_TRACE_REVS, len(trace)))

            if arts['temp_fill'][0] is not None:
                try:
                    arts['temp_fill'][0].remove()
                except Exception:
                    pass
            arts['temp_fill'][0] = arts['ax_temp'].fill_between(
                x_vals, 0, trace, color='#006050', alpha=0.35
            )

            # Estimate period from zero-crossings of mean-removed trace
            mean_val = float(trace.mean())
            dm = trace - mean_val
            zc = np.where(np.diff(np.sign(dm)))[0]
            if len(zc) >= 4:
                half_periods = np.diff(zc)
                avg_period   = float(np.mean(half_periods)) * 2
                t_sim_period = avg_period * PERIOD_S * SIM_SPEED
                arts['period_txt'].set_text(
                    f'Tₘₑₐₛ ≈ {t_sim_period:.1f} s / rev  '
                    f'({avg_period:.1f} rev cycle)'
                )
            else:
                arts['period_txt'].set_text(f'Accumulating data…  ({len(trace)} revs)')

        # ── Waterfall ─────────────────────────────────────────────────────────
        wf = sim.waterfall_image()
        arts['water_img'].set_data(wf)

        # Draw propagation speed arrow on waterfall
        # Speed: c = λ/T, in bins per revolution = c / RANGE_RES_M × PERIOD_S × SIM_SPEED
        stats = _wave_stats(peaks, sim.temporal)
        c     = stats['celerity_ms']
        bins_per_rev = c / RANGE_RES_M * PERIOD_S * SIM_SPEED
        # Arrow from (start_range, bottom) to (start_range - bins_per_rev * n_revs, top)
        r_start = MAX_RANGE_M * 0.65 / RANGE_RES_M * RANGE_RES_M  # midpoint range
        n_arrow = min(20, N_HISTORY_REVS - 2)
        r_end   = r_start - bins_per_rev * RANGE_RES_M * n_arrow
        if r_end > 0:
            arts['prop_annot'].xy     = (r_end,   N_HISTORY_REVS - 2)
            arts['prop_annot'].xytext = (r_start, N_HISTORY_REVS - 2 - n_arrow)
            arts['prop_label'].set_position(
                ((r_start + r_end) / 2, N_HISTORY_REVS - 2 - n_arrow / 2)
            )
            arts['prop_label'].set_text(f'{c:.1f} m/s')

        # ── Statistics panel ──────────────────────────────────────────────────
        lam = stats['lambda_m']
        T_t = stats['period_s_theory']
        T_m = stats['period_s_meas']
        n_w = stats['wave_count']
        c   = stats['celerity_ms']
        txt = (
            f"  Waves in view    :  {n_w}\n"
            f"\n"
            f"  Wavelength  (λ)  :  {lam:>7.1f} m\n"
            f"  Phase speed (c)  :  {c:>7.2f} m/s\n"
            f"\n"
            f"  Period — theory  :  {T_t:>7.2f} s\n"
            f"  Period — radar   :  {T_m:>7.2f} s\n"
            f"\n"
            f"  Wave origin      :  {DOMINANT_WAVE['dir']:.0f}°  (NW)\n"
            f"  Sea state        :  Bft {int(SEA_CLUTTER*27):d}\n"
            f"  SWH estimate     :  {stats['swh_estimate']}\n"
            f"\n"
            f"  Monitor range    :  {MONITOR_RANGE_M} m\n"
            f"  Range res.       :  {RANGE_RES_M:.2f} m / bin\n"
            f"  Sample rate      :  {SAMPLE_RATE_HZ/1e6:.0f} MS/s"
        )
        arts['stats_text'].set_text(txt)

    return update


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('Wave Monitoring Simulator')
    print(f'  Primary swell: λ={DOMINANT_WAVE["λ"]} m  T={DOMINANT_WAVE["T"]} s  '
          f'from {DOMINANT_WAVE["dir"]}°')
    print(f'  Range res: {RANGE_RES_M:.2f} m/bin  '
          f'({N_RANGE_BINS} bins  =  {MAX_RANGE_M:.0f} m)')
    print(f'  Sim speed: {SIM_SPEED}× real time per revolution')
    print()

    sim  = WaveSimulator()
    # Pre-fill history with a few revolutions of data
    print('  Building scan history…')
    for _ in range(N_HISTORY_REVS):
        sim._on_revolution()

    fig, arts = build_display(sim)
    updater   = make_updater(sim, arts)

    ani = animation.FuncAnimation(
        fig, updater, interval=35, blit=False, cache_frame_data=False
    )
    plt.show()


if __name__ == '__main__':
    main()
