"""
wave_monitor.py  —  Shore-based Sea Wave Monitoring (180° seaward view)

Three-panel layout:
  Left        : 180° semicircular PPI (seaward), JONSWAP waves + shadow + speckle,
                Stokes drift current vectors, coloured wave-analysis boxes
  Upper-right : 1D range profile along dominant wave direction, peak counting
  Lower-right : 2D wavenumber spectrum (k-space) from wave components
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────────────────────
# Radar / display constants
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE_HZ  = 20_000_000
RPM             = 42
RANGE_NM        = 2.0

C               = 299_792_458
METERS_PER_NM   = 1852
MAX_RANGE_M     = RANGE_NM * METERS_PER_NM          # 3 704 m
RANGE_RES_M     = C / (2 * SAMPLE_RATE_HZ)          # 7.49 m/bin
N_RANGE_BINS    = int(MAX_RANGE_M / RANGE_RES_M)     # ~494
REV_PER_S       = RPM / 60
PERIOD_S        = 1.0 / REV_PER_S
N_BEARINGS      = 512
SIM_SPEED       = 5          # simulated seconds per real antenna revolution

RADAR_HEIGHT_M  = 12         # antenna height above sea level — sets shadow length

# ── JONSWAP sea-state parameters ─────────────────────────────────────────────
PEAK_LAMBDA_M    = 110       # dominant wavelength (m)
PEAK_DIR_DEG     = 315       # waves come FROM NW
N_JONSWAP_COMP   = 52        # spectral components
JONSWAP_GAMMA    = 3.3       # peak enhancement factor (3.3 = moderate fetch)
DIR_SPREAD_KAPPA = 5.0       # von Mises concentration (higher = narrower spread)
SEA_CLUTTER      = 0.11      # broadband clutter amplitude

# ── Wave-analysis boxes overlaid on PPI ──────────────────────────────────────
WAVE_BOXES = [
    {'r': (400, 1280), 'b': 344, 'w': 15, 'color': '#ff4444', 'label': 'Box A  34°'},
    {'r': (500, 1400), 'b':   4, 'w': 15, 'color': '#44ff88', 'label': 'Box B  44°'},
    {'r': (700, 1700), 'b':  30, 'w': 15, 'color': '#4488ff', 'label': 'Box C  84°'},
    {'r': (600, 1600), 'b':  58, 'w': 15, 'color': '#ffff44', 'label': 'Box D 134°'},
]

# ─────────────────────────────────────────────────────────────────────────────
# Colour maps
# ─────────────────────────────────────────────────────────────────────────────
RADAR_CMAP = LinearSegmentedColormap.from_list('radar', [
    (0.00, (0.00, 0.00, 0.00)),
    (0.12, (0.00, 0.09, 0.03)),
    (0.38, (0.00, 0.40, 0.12)),
    (0.68, (0.07, 0.72, 0.24)),
    (0.86, (0.46, 0.93, 0.48)),
    (1.00, (0.88, 1.00, 0.78)),
])

SPEC_CMAP = LinearSegmentedColormap.from_list('kspec', [
    (0.00, (0.01, 0.02, 0.10)),
    (0.28, (0.00, 0.14, 0.44)),
    (0.58, (0.04, 0.48, 0.80)),
    (0.82, (0.55, 0.90, 1.00)),
    (1.00, (1.00, 1.00, 0.82)),
])

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute polar → Cartesian look-up for the PPI image
# ─────────────────────────────────────────────────────────────────────────────
PPI_SIZE = 480

_px  = np.linspace(-MAX_RANGE_M,  MAX_RANGE_M, PPI_SIZE)
_py  = np.linspace( MAX_RANGE_M, -MAX_RANGE_M, PPI_SIZE)   # y-decreasing → imshow upper
_PX, _PY = np.meshgrid(_px, _py)
_PR  = np.sqrt(_PX**2 + _PY**2)
_PB  = np.degrees(np.arctan2(_PX, _PY)) % 360  # 0°=N, clockwise
_PRIDX = np.clip((_PR / RANGE_RES_M).astype(np.int32), 0, N_RANGE_BINS - 1)
_PBIDX = (_PB / 360.0 * N_BEARINGS).astype(np.int32) % N_BEARINGS
# Land = southern half (y < 0) or outside range circle
_LAND  = (_PY < -MAX_RANGE_M * 0.015) | (_PR > MAX_RANGE_M)

RANGE_AXIS = np.arange(N_RANGE_BINS) * RANGE_RES_M


# ─────────────────────────────────────────────────────────────────────────────
# JONSWAP spectral component generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_jonswap(rng):
    """
    Sample N_JONSWAP_COMP wave components from a JONSWAP + von Mises distribution.
    Returns a dict of float32 arrays: k, omega, dir, amp, phase, lambda_, period.
    """
    g        = 9.81
    k_peak   = 2 * np.pi / PEAK_LAMBDA_M
    om_peak  = np.sqrt(g * k_peak)           # deep-water: ω² = gk

    # Log-spaced frequencies with small random jitter
    omegas = np.exp(np.linspace(np.log(0.48 * om_peak),
                                np.log(2.90 * om_peak), N_JONSWAP_COMP))
    omegas += rng.uniform(-0.015 * om_peak, 0.015 * om_peak, N_JONSWAP_COMP)
    omegas  = np.clip(omegas, 0.3 * om_peak, 3.5 * om_peak)

    # JONSWAP S(ω)
    sigma = np.where(omegas <= om_peak, 0.07, 0.09)
    r_exp = np.exp(-0.5 * ((omegas / om_peak - 1) / sigma) ** 2)
    S     = (0.0081 * g**2 / omegas**5
             * np.exp(-1.25 * (om_peak / omegas)**4)
             * JONSWAP_GAMMA**r_exp)

    # Directional spreading (von Mises around peak direction)
    dir_mean = np.radians(PEAK_DIR_DEG)
    if dir_mean > np.pi:
        dir_mean -= 2 * np.pi                # von Mises wants mean in (-π, π]
    dirs = rng.vonmises(dir_mean, DIR_SPREAD_KAPPA, N_JONSWAP_COMP)

    # Amplitude a_i = sqrt(2 S_i Δω)
    dw   = np.gradient(omegas)
    amps = np.sqrt(2.0 * S * np.abs(dw))
    norm = np.percentile(amps, 95) * 3.2 + 1e-12
    amps = (amps / norm).astype(np.float32)

    # Deep-water dispersion: k = ω²/g
    ks      = (omegas**2 / g).astype(np.float32)
    periods = (2 * np.pi / omegas).astype(np.float32)
    phases  = rng.uniform(0, 2 * np.pi, N_JONSWAP_COMP).astype(np.float32)

    return {
        'k':       ks,
        'omega':   omegas.astype(np.float32),
        'dir':     dirs.astype(np.float32),
        'amp':     amps,
        'phase':   phases,
        'lambda_': (2 * np.pi / ks),
        'period':  periods,
    }


def _stokes_drift(comp):
    """
    Surface Stokes drift vector (Ux, Uy) in m/s.
    U_S = Σ aᵢ² ωᵢ · (kx_i, ky_i)  where waves propagate TO (dir_i + π).
    """
    travel = comp['dir'] + np.pi
    kx = comp['k'] * np.sin(travel)
    ky = comp['k'] * np.cos(travel)
    w  = comp['amp']**2 * comp['omega']
    return float(np.sum(w * kx)), float(np.sum(w * ky))


# ─────────────────────────────────────────────────────────────────────────────
# Radar backscatter simulation
# ─────────────────────────────────────────────────────────────────────────────

def _wave_echo_2d(sim_time_s, comp, rng):
    """
    Polar array (N_BEARINGS × N_RANGE_BINS) of wave backscatter with:
      • Multi-component JONSWAP surface elevation
      • Grazing-angle shadow model (crests occlude troughs behind them)
      • Rayleigh speckle × crest²  (non-sinusoidal, patchy texture)
    """
    bearings = np.linspace(0, 2 * np.pi, N_BEARINGS, endpoint=False,
                           dtype=np.float32)
    r_m      = (np.arange(N_RANGE_BINS, dtype=np.float32)) * RANGE_RES_M

    # ── Surface elevation η(bearing, range) ────────────────────────────────
    eta = np.zeros((N_BEARINGS, N_RANGE_BINS), dtype=np.float32)
    for i in range(N_JONSWAP_COMP):
        k   = float(comp['k'][i])
        om  = float(comp['omega'][i])
        d   = float(comp['dir'][i])        # FROM direction
        amp = float(comp['amp'][i])
        phi = float(comp['phase'][i])
        # Projection: waves come FROM d, so phase increases toward radar
        cos_fac = np.cos(bearings - d)                            # (N_B,)
        phase   = (k * (-cos_fac[:, np.newaxis]) * r_m[np.newaxis, :]
                   - om * sim_time_s + phi)                        # (N_B, N_R)
        eta    += amp * np.cos(phase)

    # ── Shadow model (grazing-angle occlusion) ─────────────────────────────
    r_safe     = r_m + RANGE_RES_M                                # avoid /0
    elev_angle = (eta + RADAR_HEIGHT_M) / r_safe[np.newaxis, :]   # (N_B, N_R)
    run_max    = np.maximum.accumulate(elev_angle, axis=1)
    prev_max   = np.concatenate(
        [np.full((N_BEARINGS, 1), -np.inf, np.float32), run_max[:, :-1]], axis=1
    )
    visible = (elev_angle >= prev_max).astype(np.float32)

    # ── Crest-only backscatter ──────────────────────────────────────────────
    crest  = np.clip(eta, 0.0, None) ** 2
    signal = crest * visible

    # ── Rayleigh speckle + range-decaying clutter ──────────────────────────
    speckle = rng.rayleigh(1.0, (N_BEARINGS, N_RANGE_BINS)).astype(np.float32)
    r_decay = np.exp(-r_m / (MAX_RANGE_M * 0.75)).astype(np.float32)
    clutter = SEA_CLUTTER * r_decay[np.newaxis, :] * speckle

    # Multiplicative speckle modulates signal (irregular, patchy)
    data = signal * (0.55 + 0.45 * speckle) + clutter
    p99  = np.percentile(data, 99.3)
    return np.clip(data / (p99 + 1e-9), 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2-D wavenumber spectrum
# ─────────────────────────────────────────────────────────────────────────────

def _compute_2d_spectrum(comp):
    N      = 128
    k_lim  = 4.5 * 2 * np.pi / PEAK_LAMBDA_M
    kx_ax  = np.linspace(-k_lim, k_lim, N)
    ky_ax  = np.linspace(-k_lim, k_lim, N)
    KX, KY = np.meshgrid(kx_ax, ky_ax)
    spec   = np.zeros((N, N), dtype=np.float32)
    sigma_k = k_lim / 18.0

    travel = comp['dir'] + np.pi
    kxs    = comp['k'] * np.sin(travel)
    kys    = comp['k'] * np.cos(travel)

    for i in range(N_JONSWAP_COMP):
        power = float(comp['amp'][i]**2)
        blob  = np.exp(-((KX - kxs[i])**2 + (KY - kys[i])**2)
                       / (2 * sigma_k**2)).astype(np.float32)
        spec += power * blob

    spec /= spec.max() + 1e-9
    return spec, kx_ax, ky_ax


# ─────────────────────────────────────────────────────────────────────────────
# Peak detection (no scipy)
# ─────────────────────────────────────────────────────────────────────────────

def _smooth(x, w=9):
    return np.convolve(x, np.ones(w) / w, mode='same')


def find_wave_peaks(profile, min_height=0.18):
    s     = _smooth(profile, w=7)
    min_d = max(4, int(PEAK_LAMBDA_M * 0.50 / RANGE_RES_M))
    peaks = []
    for i in range(1, len(s) - 1):
        if s[i] > s[i - 1] and s[i] >= s[i + 1] and s[i] >= min_height:
            if not peaks or (i - peaks[-1]) >= min_d:
                peaks.append(i)
    return np.array(peaks, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# Simulator state
# ─────────────────────────────────────────────────────────────────────────────

class WaveSimulator:
    PERSIST = 0.991

    def __init__(self):
        self.rng          = np.random.default_rng(42)
        self.sim_time     = 0.0
        self.polar        = np.zeros((N_BEARINGS, N_RANGE_BINS), np.float32)
        self.components   = _generate_jonswap(self.rng)
        self.scan         = _wave_echo_2d(self.sim_time, self.components, self.rng)
        self.bearing_step = 0
        self.rev_count    = 0

    def step(self, steps=6):
        self.polar *= self.PERSIST
        for _ in range(steps):
            self.polar[self.bearing_step] = self.scan[self.bearing_step]
            self.bearing_step = (self.bearing_step + 1) % N_BEARINGS
            if self.bearing_step == 0:
                self._on_revolution()

    def _on_revolution(self):
        self.sim_time  += PERIOD_S * SIM_SPEED
        self.rev_count += 1
        # Slowly evolve the wave field (sea state changes over minutes)
        if self.rev_count % 25 == 0:
            new_comp = _generate_jonswap(self.rng)
            alpha = 0.08
            for key in ('amp', 'phase', 'dir'):
                self.components[key] = (
                    (1 - alpha) * self.components[key] + alpha * new_comp[key]
                ).astype(np.float32)
        self.scan = _wave_echo_2d(self.sim_time, self.components, self.rng)

    @property
    def bearing_deg(self):
        return self.bearing_step / N_BEARINGS * 360.0

    def ppi_image(self):
        img = self.polar[_PBIDX, _PRIDX].copy()
        img[_LAND] = 0.0
        return img

    def profile_along(self, bearing_deg):
        b    = int(bearing_deg / 360 * N_BEARINGS) % N_BEARINGS
        idxs = [(b + d) % N_BEARINGS for d in range(-2, 3)]
        return self.polar[idxs, :].mean(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_static_ppi(ax):
    """Range rings, cardinal labels, shore, wave boxes — drawn once."""
    # Range rings (top-half arcs only)
    for r_nm in np.arange(0.5, RANGE_NM + 0.01, 0.5):
        r_m   = r_nm * METERS_PER_NM
        b_arc = np.linspace(-np.pi / 2, np.pi / 2, 200)
        ax.plot(r_m * np.sin(b_arc), r_m * np.cos(b_arc),
                color='#182818', lw=0.55, ls='--', zorder=1)
        ax.text(0, r_m * 1.015, f'{r_m:.0f} m',
                color='#284028', fontsize=6, ha='center', va='bottom', zorder=8)

    # Bearing spokes every 30°
    for b_deg in range(-90, 91, 30):
        b_rad = np.radians(b_deg)
        ax.plot([0, MAX_RANGE_M * np.sin(b_rad)],
                [0, MAX_RANGE_M * np.cos(b_rad)],
                color='#182818', lw=0.4, ls=':', zorder=1)
        if b_deg != 0:
            ax.text(MAX_RANGE_M * 1.03 * np.sin(b_rad),
                    MAX_RANGE_M * 1.03 * np.cos(b_rad),
                    f'{b_deg % 360}°',
                    color='#3a5a3a', fontsize=6, ha='center', va='center')

    # Cardinal labels
    for lbl, bx, by in [('N', 0, 1.07), ('NE', 0.76, 0.76),
                         ('NW', -0.76, 0.76), ('E', 1.07, 0), ('W', -1.07, 0)]:
        ax.text(bx * MAX_RANGE_M, by * MAX_RANGE_M,
                lbl, color='#50a050', fontsize=8, fontweight='bold',
                ha='center', va='center', zorder=9)

    # Shore line + land fill
    land = mpatches.Rectangle(
        (-MAX_RANGE_M * 1.10, -MAX_RANGE_M * 1.10),
        MAX_RANGE_M * 2.20, MAX_RANGE_M * 1.10,
        fc='#18130a', ec='none', zorder=5,
    )
    ax.add_patch(land)
    ax.plot([-MAX_RANGE_M * 1.05, MAX_RANGE_M * 1.05], [0, 0],
            color='#b89050', lw=2.2, zorder=6)
    ax.text(-MAX_RANGE_M * 0.92, -MAX_RANGE_M * 0.07,
            'SHORE  ←  LAND', color='#b89050', fontsize=7.5, zorder=7)

    # Wave-analysis boxes
    for box in WAVE_BOXES:
        r1, r2 = box['r']
        b_c    = box['b']
        w_d    = box['w']
        col    = box['color']
        ang    = np.linspace(np.radians(b_c - w_d/2),
                             np.radians(b_c + w_d/2), 40)
        ox, oy = r2 * np.sin(ang), r2 * np.cos(ang)
        ix, iy = r1 * np.sin(ang[::-1]), r1 * np.cos(ang[::-1])
        px, py = np.concatenate([ox, ix]), np.concatenate([oy, iy])
        ax.fill(px, py, color=col, alpha=0.10, zorder=3)
        ax.plot(np.append(px, px[0]), np.append(py, py[0]),
                color=col, lw=1.0, alpha=0.65, zorder=4)
        bm = np.radians(b_c)
        ax.text(r2 * 1.05 * np.sin(bm), r2 * 1.05 * np.cos(bm),
                box['label'], color=col, fontsize=6.5,
                ha='center', va='center', zorder=8)


def _make_drift_arrows(ax, n_x=8, n_y=4):
    """
    Create a uniform grid of annotation arrows for Stokes drift.
    Returns list of (annotation, base_x, base_y).
    """
    xs = np.linspace(-MAX_RANGE_M * 0.78, MAX_RANGE_M * 0.78, n_x)
    ys = np.linspace(MAX_RANGE_M * 0.14,  MAX_RANGE_M * 0.85, n_y)
    arrows = []
    for ay in ys:
        for arx in xs:
            ann = ax.annotate(
                '', xy=(arx, ay), xytext=(arx, ay),
                arrowprops=dict(arrowstyle='->', color='#00ccff',
                                lw=1.2, mutation_scale=7),
                zorder=12, annotation_clip=False,
            )
            arrows.append((ann, arx, ay))
    return arrows


# ─────────────────────────────────────────────────────────────────────────────
# Build display
# ─────────────────────────────────────────────────────────────────────────────

def build_display(sim):
    fig = plt.figure(figsize=(16, 9), facecolor='#050810')
    fig.canvas.manager.set_window_title(
        'Wave Monitoring — Shore-based Radar  (180° seaward view)'
    )

    gs = fig.add_gridspec(
        2, 2,
        left=0.02, right=0.97, top=0.93, bottom=0.05,
        wspace=0.26, hspace=0.40,
        width_ratios=[1.45, 1.0],
        height_ratios=[1.0, 1.0],
    )

    ax_ppi  = fig.add_subplot(gs[:, 0], facecolor='black')    # left — full height
    ax_prof = fig.add_subplot(gs[0, 1], facecolor='#04060e')  # upper right
    ax_spec = fig.add_subplot(gs[1, 1], facecolor='#04060e')  # lower right

    # ── PPI ───────────────────────────────────────────────────────────────────
    ax_ppi.set_aspect('equal')
    ax_ppi.set_xlim(-MAX_RANGE_M * 1.10, MAX_RANGE_M * 1.10)
    ax_ppi.set_ylim(-MAX_RANGE_M * 0.17, MAX_RANGE_M * 1.10)
    ax_ppi.axis('off')
    ax_ppi.set_title('Shore-based radar  —  seaward 180°  (JONSWAP + shadow + speckle)',
                     color='#70e070', fontsize=8.5, pad=4)

    _draw_static_ppi(ax_ppi)

    ppi_img = ax_ppi.imshow(
        np.zeros((PPI_SIZE, PPI_SIZE), np.float32),
        cmap=RADAR_CMAP, vmin=0, vmax=1, origin='upper',
        extent=[-MAX_RANGE_M, MAX_RANGE_M, -MAX_RANGE_M, MAX_RANGE_M],
        interpolation='bilinear', zorder=0,
    )

    sweep_ln, = ax_ppi.plot([], [], color='#90ff90', lw=1.0, alpha=0.85, zorder=13)

    drift_arrows = _make_drift_arrows(ax_ppi)
    drift_lbl = ax_ppi.text(
        -MAX_RANGE_M * 1.03, MAX_RANGE_M * 0.95,
        'Stokes drift', color='#00ccff', fontsize=7.5,
        fontweight='bold', va='top', zorder=14,
    )

    # ── Wave profile ──────────────────────────────────────────────────────────
    ax_prof.set_xlim(0, MAX_RANGE_M)
    ax_prof.set_ylim(-0.05, 1.22)
    ax_prof.set_xlabel('Range (m)', color='#5080a0', fontsize=8)
    ax_prof.set_ylabel('Backscatter amplitude', color='#5080a0', fontsize=8)
    ax_prof.set_title(f'Wave Profile  —  bearing {PEAK_DIR_DEG:.0f}°  '
                      f'(wave origin direction)',
                      color='#80b0d0', fontsize=8, pad=3)
    ax_prof.tick_params(colors='#304060', labelsize=7)
    ax_prof.spines[:].set_color('#1a2a3a')
    ax_prof.axhline(0.18, color='#1a2a3a', lw=0.5, ls=':')   # detection threshold

    prof_line, = ax_prof.plot(RANGE_AXIS, np.zeros(N_RANGE_BINS),
                              color='#60b0ff', lw=0.9)
    prof_fill  = [None]

    max_pk = 60
    pk_lines  = [ax_prof.axvline(0, color='#ff5555', lw=0.85, ls='--',
                                 alpha=0.0) for _ in range(max_pk)]
    pk_labels = [ax_prof.text(0, 1.08, '', color='#ff7777', fontsize=5.5,
                              ha='center', rotation=90, va='bottom')
                 for _ in range(max_pk)]
    lam_txt  = ax_prof.text(0.98, 0.94, '', transform=ax_prof.transAxes,
                            color='#ffff80', fontsize=9, fontweight='bold',
                            ha='right', va='top')
    cnt_txt  = ax_prof.text(0.98, 0.82, '', transform=ax_prof.transAxes,
                            color='#ffffa0', fontsize=9, fontweight='bold',
                            ha='right', va='top')

    # ── 2-D wavenumber spectrum ───────────────────────────────────────────────
    ax_spec.set_aspect('equal')
    ax_spec.set_title('2D Wavenumber Spectrum  (k-space)',
                      color='#80b0d0', fontsize=8, pad=3)
    ax_spec.set_xlabel('kₓ  (rad/m)  →  East', color='#5080a0', fontsize=7.5)
    ax_spec.set_ylabel('k_y  (rad/m)  →  North', color='#5080a0', fontsize=7.5)
    ax_spec.tick_params(colors='#304060', labelsize=7)
    ax_spec.spines[:].set_color('#1a2a3a')
    ax_spec.axhline(0, color='#222244', lw=0.5)
    ax_spec.axvline(0, color='#222244', lw=0.5)

    spec_data, kx_ax, ky_ax = _compute_2d_spectrum(sim.components)
    spec_img = ax_spec.imshow(
        spec_data, cmap=SPEC_CMAP, vmin=0, vmax=1, origin='lower',
        aspect='auto',
        extent=[kx_ax[0], kx_ax[-1], ky_ax[0], ky_ax[-1]],
        interpolation='bilinear',
    )

    # Peak wavenumber ring annotation
    k_p    = 2 * np.pi / PEAK_LAMBDA_M
    c_ang  = np.linspace(0, 2 * np.pi, 120)
    ax_spec.plot(k_p * np.cos(c_ang), k_p * np.sin(c_ang),
                 color='#ffffff', lw=0.55, ls=':', alpha=0.45)
    ax_spec.text(k_p * 1.08, 0, f'λ={PEAK_LAMBDA_M}m',
                 color='#7788aa', fontsize=6, va='center')

    # ── Figure title ──────────────────────────────────────────────────────────
    fig.text(
        0.50, 0.975,
        f'WAVE MONITORING  ·  Furuno FAR-2xx8 / PCI-9812  ·  '
        f'{SAMPLE_RATE_HZ/1e6:.0f} MS/s  ·  {RANGE_NM:.1f} NM  ·  '
        f'JONSWAP  λ_p={PEAK_LAMBDA_M} m  from {PEAK_DIR_DEG}° (NW)  ·  '
        f'{N_JONSWAP_COMP} components',
        color='#90d090', fontsize=8.5, ha='center', fontweight='bold',
    )

    artists = dict(
        ppi_img=ppi_img, sweep_ln=sweep_ln,
        drift_arrows=drift_arrows, drift_lbl=drift_lbl,
        prof_line=prof_line, prof_fill=prof_fill,
        pk_lines=pk_lines, pk_labels=pk_labels,
        lam_txt=lam_txt, cnt_txt=cnt_txt,
        spec_img=spec_img,
        ax_prof=ax_prof, ax_spec=ax_spec,
    )
    return fig, artists


# ─────────────────────────────────────────────────────────────────────────────
# Animation updater
# ─────────────────────────────────────────────────────────────────────────────

def make_updater(sim, arts):
    frame_ctr = [0]

    def update(_frame):
        sim.step(steps=6)
        fc = frame_ctr[0]
        frame_ctr[0] += 1

        # ── PPI + sweep line (every frame) ───────────────────────────────────
        arts['ppi_img'].set_data(sim.ppi_image())
        b_deg = sim.bearing_deg
        if b_deg <= 90 or b_deg >= 270:            # sea sector only
            b_rad = np.radians(b_deg)
            arts['sweep_ln'].set_data(
                [0, MAX_RANGE_M * 0.97 * np.sin(b_rad)],
                [0, MAX_RANGE_M * 0.97 * np.cos(b_rad)],
            )
        else:
            arts['sweep_ln'].set_data([], [])

        # Heavy updates every 4 frames
        if fc % 4 != 0:
            return

        # ── Stokes drift arrows ───────────────────────────────────────────────
        Ux, Uy = _stokes_drift(sim.components)
        mag    = np.sqrt(Ux**2 + Uy**2) + 1e-12
        scale  = MAX_RANGE_M * 0.09
        dx     = Ux / mag * scale
        dy     = Uy / mag * scale
        for ann, arx, ary in arts['drift_arrows']:
            ann.xy     = (arx + dx, ary + dy)
            ann.xytext = (arx, ary)
        arts['drift_lbl'].set_text(
            f'Stokes drift  {mag:.4f} m/s\n'
            f'→  {np.degrees(np.arctan2(Ux, Uy)) % 360:.0f}°'
        )

        # ── Wave profile ──────────────────────────────────────────────────────
        profile = sim.profile_along(PEAK_DIR_DEG)
        arts['prof_line'].set_ydata(profile)

        if arts['prof_fill'][0] is not None:
            try:
                arts['prof_fill'][0].remove()
            except Exception:
                pass
        arts['prof_fill'][0] = arts['ax_prof'].fill_between(
            RANGE_AXIS, 0, profile, color='#002880', alpha=0.28
        )

        peaks = find_wave_peaks(profile)
        for ln in arts['pk_lines']:
            ln.set_alpha(0.0)
        for lb in arts['pk_labels']:
            lb.set_text('')
        for i, pk in enumerate(peaks[:len(arts['pk_lines'])]):
            rm = RANGE_AXIS[pk]
            arts['pk_lines'][i].set_xdata([rm, rm])
            arts['pk_lines'][i].set_alpha(0.60)
            arts['pk_labels'][i].set_position((rm, 1.09))
            arts['pk_labels'][i].set_text(f'{rm:.0f}m')

        if len(peaks) >= 2:
            lam_m = RANGE_AXIS[peaks[1]] - RANGE_AXIS[peaks[0]]
            arts['lam_txt'].set_text(f'λ ≈ {lam_m:.0f} m')
        else:
            arts['lam_txt'].set_text('')
        arts['cnt_txt'].set_text(f'{len(peaks)} waves')

        # ── 2-D spectrum ──────────────────────────────────────────────────────
        spec_data, _, _ = _compute_2d_spectrum(sim.components)
        arts['spec_img'].set_data(spec_data)

    return update


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('Wave Monitoring  —  shore-based 180° radar view')
    print(f'  Peak swell: λ={PEAK_LAMBDA_M} m  T≈{2*np.pi*np.sqrt(PEAK_LAMBDA_M/(2*np.pi*9.81)):.1f} s'
          f'  from {PEAK_DIR_DEG}° (NW)')
    print(f'  JONSWAP components: {N_JONSWAP_COMP}   γ={JONSWAP_GAMMA}')
    print(f'  Range: {MAX_RANGE_M:.0f} m   Res: {RANGE_RES_M:.2f} m/bin   '
          f'Bins: {N_RANGE_BINS}')
    print(f'  Shadow model: radar height {RADAR_HEIGHT_M} m')
    print()

    sim = WaveSimulator()
    print('  Pre-filling scan history…')
    for _ in range(10):
        sim._on_revolution()

    fig, arts = build_display(sim)
    updater   = make_updater(sim, arts)

    ani = animation.FuncAnimation(             # noqa: F841
        fig, updater, interval=35, blit=False, cache_frame_data=False
    )
    plt.show()


if __name__ == '__main__':
    main()
