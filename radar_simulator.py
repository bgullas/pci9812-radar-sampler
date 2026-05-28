"""
Furuno FAR-2xx8 Radar PPI Simulator
Simulates 20 MS/s ADC acquisition and renders a live PPI (Plan Position Indicator).

Signals simulated (matching J510 connector wiring):
  CH0 - HD    (Heading pulse: once per revolution)
  CH1 - BP    (Bearing pulse: ~357 pulses/revolution, one per ~1 degree)
  CH2 - TRIG  (Trigger: once per radar pulse, marks start of range sweep)
  CH3 - VIDEO (Radar echo amplitude vs range after each trigger)

Features:
  - Moving vessels and buoy with track history trails
  - Course/speed vectors per target
  - Dashed range arc + distance label in metres per target
  - Realistic sea clutter, wave swell, Rayleigh speckle
  - Raw ADC signal strip panel (HD / BP / TRIG / VIDEO)

Replace simulate_scan() with data from pci9812_sampler.py for live hardware.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')        # change to 'Qt5Agg' if TkAgg is not available
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
# Hardware / radar constants  (Furuno FAR-2xx8, Port J510)
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE_HZ  = 20_000_000    # PCI-9812 max: 20 MS/s
RPM             = 42             # antenna rotation speed
PRF_HZ          = 2000           # pulse repetition frequency
PULSE_WIDTH_US  = 0.3            # µs
RANGE_NM        = 3.0            # display range (nautical miles)
SEA_STATE       = 3              # 0 = calm … 9 = hurricane
WAVE_DIR_DEG    = 315            # dominant wave direction (NW)
SIM_SPEED       = 25             # simulated seconds per real revolution
                                 # (increase to see targets move faster)
TRACK_LEN       = 14             # number of past positions to show per target

# ─────────────────────────────────────────────────────────────────────────────
# Derived
# ─────────────────────────────────────────────────────────────────────────────
C               = 299_792_458
METERS_PER_NM   = 1852
MAX_RANGE_M     = RANGE_NM * METERS_PER_NM       # 5 556 m
RANGE_RES_M     = C / (2 * SAMPLE_RATE_HZ)       # 7.49 m / bin
N_RANGE_BINS    = int(MAX_RANGE_M / RANGE_RES_M) # ~741

REV_PER_S       = RPM / 60                       # 0.7 rev/s
PERIOD_S        = 1.0 / REV_PER_S                # 1.43 s / revolution
PULSES_PER_REV  = int(PRF_HZ / REV_PER_S)        # ~2857

N_BEARINGS      = 512            # angular steps per revolution
STEPS_PER_FRAME = 6              # bearing steps per animation frame

KN_TO_MPS       = 0.5144         # 1 knot in m/s

# ─────────────────────────────────────────────────────────────────────────────
# Moving targets  (name, initial position, speed in knots, course in degrees)
# ─────────────────────────────────────────────────────────────────────────────
# All targets start within MAX_RANGE_M and travel at realistic maritime speeds.
# The simulator re-enters a target at a new edge position if it leaves the display.
MOVING_TARGETS = [
    {'name': 'VESSEL A', 'range_m': 2100, 'bearing_deg':  42, 'speed_kn': 12, 'course_deg':  98, 'rcs': 1.00},
    {'name': 'VESSEL B', 'range_m': 3400, 'bearing_deg': 128, 'speed_kn':  8, 'course_deg': 215, 'rcs': 0.75},
    {'name': 'VESSEL C', 'range_m': 4700, 'bearing_deg': 287, 'speed_kn': 15, 'course_deg': 345, 'rcs': 0.85},
    {'name': 'FAST BOAT','range_m': 1400, 'bearing_deg': 195, 'speed_kn': 28, 'course_deg':  55, 'rcs': 0.45},
    {'name': 'BUOY',     'range_m':  900, 'bearing_deg': 235, 'speed_kn':  0, 'course_deg':   0, 'rcs': 0.35},
]

# ─────────────────────────────────────────────────────────────────────────────
# Color map: radar green
# ─────────────────────────────────────────────────────────────────────────────
RADAR_CMAP = LinearSegmentedColormap.from_list('radar_green', [
    (0.00, (0.00, 0.00, 0.00)),
    (0.15, (0.00, 0.12, 0.04)),
    (0.45, (0.00, 0.50, 0.16)),
    (0.75, (0.10, 0.82, 0.28)),
    (1.00, (0.75, 1.00, 0.65)),
])

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute polar → Cartesian index lookup  (once at startup)
# ─────────────────────────────────────────────────────────────────────────────
GRID_SIZE = 700
_x = np.linspace(-MAX_RANGE_M,  MAX_RANGE_M, GRID_SIZE)
_y = np.linspace( MAX_RANGE_M, -MAX_RANGE_M, GRID_SIZE)   # y flipped: N at top
XX, YY     = np.meshgrid(_x, _y)
RR_GRID    = np.sqrt(XX**2 + YY**2)
BB_GRID    = np.degrees(np.arctan2(XX, YY)) % 360   # 0°=N, cw
RANGE_IDX  = np.clip((RR_GRID / RANGE_RES_M).astype(np.int32), 0, N_RANGE_BINS - 1)
BEAR_IDX   = (BB_GRID / 360.0 * N_BEARINGS).astype(np.int32) % N_BEARINGS
CIRC_MASK  = RR_GRID > MAX_RANGE_M


# ─────────────────────────────────────────────────────────────────────────────
# Target state manager
# ─────────────────────────────────────────────────────────────────────────────

def _polar_to_xy(range_m, bearing_deg):
    rad = np.radians(bearing_deg)
    return range_m * np.sin(rad), range_m * np.cos(rad)


def _xy_to_polar(x, y):
    return np.sqrt(x**2 + y**2), np.degrees(np.arctan2(x, y)) % 360


class TargetState:
    """Tracks position, velocity and trail for one moving target."""

    def __init__(self, spec):
        self.name       = spec['name']
        self.speed_mps  = spec['speed_kn'] * KN_TO_MPS
        self.course_deg = spec['course_deg']
        self.rcs        = spec['rcs']
        self.track      = deque(maxlen=TRACK_LEN)   # past (x, y) positions

        x, y = _polar_to_xy(spec['range_m'], spec['bearing_deg'])
        self.x, self.y = float(x), float(y)

    # ------------------------------------------------------------------

    def advance(self, dt_s):
        """Move the target by dt_s seconds of real-world time."""
        self.track.append((self.x, self.y))
        cr = np.radians(self.course_deg)
        self.x += np.sin(cr) * self.speed_mps * dt_s
        self.y += np.cos(cr) * self.speed_mps * dt_s

        # Re-enter from opposite edge if too far out
        r, b = _xy_to_polar(self.x, self.y)
        if r > MAX_RANGE_M * 1.05 and self.speed_mps > 0:
            b_new = (b + 180) % 360
            r_new = MAX_RANGE_M * 0.92
            self.x, self.y = _polar_to_xy(r_new, b_new)
            self.track.clear()

    @property
    def range_m(self):
        return np.sqrt(self.x**2 + self.y**2)

    @property
    def bearing_deg(self):
        return np.degrees(np.arctan2(self.x, self.y)) % 360

    @property
    def range_bin(self):
        return int(self.range_m / RANGE_RES_M)

    @property
    def bearing_bin(self):
        return int(self.bearing_deg / 360 * N_BEARINGS) % N_BEARINGS

    def course_endpoint(self, lookahead_s=60):
        """End point of course vector (lookahead_s seconds ahead)."""
        cr = np.radians(self.course_deg)
        d  = self.speed_mps * lookahead_s
        return self.x + np.sin(cr) * d, self.y + np.cos(cr) * d


# ─────────────────────────────────────────────────────────────────────────────
# Signal simulation
# ─────────────────────────────────────────────────────────────────────────────

def _gauss1d(sigma, truncate=3):
    r = int(truncate * sigma + 0.5)
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _smooth_rows(arr, kernel):
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        out[i] = np.convolve(arr[i], kernel, mode='same')
    return out


def simulate_scan(rng, targets):
    """
    Generate one 360° polar scan array  (N_BEARINGS × N_RANGE_BINS), 0–1.
    Targets is a list of TargetState objects.
    """
    r        = np.arange(N_RANGE_BINS, dtype=np.float32)
    bearings = np.linspace(0, 360, N_BEARINGS, endpoint=False)

    # ── Sea clutter (Rayleigh speckle, range-dependent, directional) ─────────
    clutter_r  = int(25 + SEA_STATE * 18)
    range_env  = (SEA_STATE / 9.0) * np.exp(-r / clutter_r)
    dir_mod    = 0.4 + 0.6 * np.cos(np.radians(bearings - WAVE_DIR_DEG))
    speckle    = rng.rayleigh(1.0, (N_BEARINGS, N_RANGE_BINS)).astype(np.float32)
    sea        = _smooth_rows(range_env[np.newaxis, :] * dir_mod[:, np.newaxis] * speckle,
                              _gauss1d(1.5))

    # ── Wave swell pattern ────────────────────────────────────────────────────
    wl_bins    = int(100 / RANGE_RES_M)
    wave_pat   = (0.5 + 0.5 * np.sin(2 * np.pi * r / wl_bins)) ** 4
    wave_dmod  = np.clip(np.cos(np.radians(bearings - WAVE_DIR_DEG)), 0, 1) ** 2
    swell      = (SEA_STATE / 9.0) * 0.4 * wave_pat[np.newaxis, :] * wave_dmod[:, np.newaxis]

    # ── Moving targets ────────────────────────────────────────────────────────
    tgt_layer  = np.zeros((N_BEARINGS, N_RANGE_BINS), dtype=np.float32)
    for t in targets:
        rb = t.range_bin
        bb = t.bearing_bin
        if not (0 <= rb < N_RANGE_BINS):
            continue
        for dr in range(-4, 5):
            for db in range(-3, 4):
                rr = rb + dr
                bbn = (bb + db) % N_BEARINGS
                if 0 <= rr < N_RANGE_BINS:
                    w = np.exp(-0.5 * ((dr / 2.0) ** 2 + (db / 1.5) ** 2))
                    tgt_layer[bbn, rr] += t.rcs * w

    # ── Noise floor + TVG ─────────────────────────────────────────────────────
    noise = rng.exponential(0.015, (N_BEARINGS, N_RANGE_BINS)).astype(np.float32)
    tvg   = np.clip(r / (N_RANGE_BINS * 0.15), 0, 1) ** 0.4 + 0.3

    data = (sea + swell + tgt_layer + noise) * tvg[np.newaxis, :]
    peak = np.percentile(data, 99.5)
    return np.clip(data / (peak + 1e-9), 0, 1).astype(np.float32)


def simulate_raw_signals(n_samples=2000):
    t_samp  = np.arange(n_samples)
    noise   = np.random.normal(0, 0.01, n_samples).astype(np.float32)

    # BP: square wave at bearing-pulse rate
    bp_per  = max(1, int(PERIOD_S / N_BEARINGS * SAMPLE_RATE_HZ))
    bp      = ((t_samp % bp_per) < (bp_per // 2)).astype(np.float32)

    # TRIG: narrow pulse at PRF
    tr_per  = int(SAMPLE_RATE_HZ / PRF_HZ)
    tr_w    = max(1, int(PULSE_WIDTH_US * 1e-6 * SAMPLE_RATE_HZ))
    trig    = ((t_samp % tr_per) < tr_w).astype(np.float32)

    # VIDEO: decaying echo after each trigger
    video   = np.zeros(n_samples, dtype=np.float32)
    for ps in range(0, n_samples, tr_per):
        el = min(tr_per - tr_w, n_samples - ps - tr_w)
        if el <= 0:
            continue
        idx   = np.arange(el)
        amp   = np.random.rayleigh(0.25, el) * np.exp(-idx / (el * 0.4))
        s, e  = ps + tr_w, ps + tr_w + el
        if e <= n_samples:
            video[s:e] += amp

    hd = np.zeros(n_samples, dtype=np.float32)
    return {
        'HD':   np.clip(hd   * 0.9 + noise * 0.3, 0, 1),
        'BP':   np.clip(bp   * 0.9 + noise * 0.3, 0, 1),
        'TRIG': np.clip(trig * 0.9 + noise * 0.2, 0, 1),
        'VIDEO':np.clip(video        + noise * 0.5, -0.1, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PPI animator
# ─────────────────────────────────────────────────────────────────────────────

class PPIAnimator:
    PERSISTENCE = 0.993

    def __init__(self):
        self.rng          = np.random.default_rng(0)
        self.targets      = [TargetState(s) for s in MOVING_TARGETS]
        self.polar        = np.zeros((N_BEARINGS, N_RANGE_BINS), np.float32)
        self.scan         = simulate_scan(self.rng, self.targets)
        self.bearing_step = 0
        self._rev_dt      = PERIOD_S * SIM_SPEED   # sim seconds per revolution

    def advance(self, steps=STEPS_PER_FRAME):
        self.polar *= self.PERSISTENCE
        for _ in range(steps):
            self.polar[self.bearing_step] = self.scan[self.bearing_step]
            self.bearing_step = (self.bearing_step + 1) % N_BEARINGS
            if self.bearing_step == 0:
                for t in self.targets:
                    t.advance(self._rev_dt)
                self.scan = simulate_scan(self.rng, self.targets)

    @property
    def bearing_deg(self):
        return self.bearing_step / N_BEARINGS * 360.0

    def cartesian_image(self):
        img = self.polar[BEAR_IDX, RANGE_IDX].copy()
        img[CIRC_MASK] = 0
        return img


# ─────────────────────────────────────────────────────────────────────────────
# Build display
# ─────────────────────────────────────────────────────────────────────────────

def build_display(ppi):
    fig = plt.figure(figsize=(14, 9), facecolor='#080808')
    fig.canvas.manager.set_window_title('Furuno FAR-2xx8 — PPI Radar Simulator')

    gs = fig.add_gridspec(
        5, 2,
        left=0.02, right=0.98, top=0.95, bottom=0.04,
        wspace=0.06, hspace=0.28,
        width_ratios=[2.8, 1],
        height_ratios=[1, 1, 1, 1, 0.4],
    )

    ax_ppi = fig.add_subplot(gs[:, 0], facecolor='black')
    ax_hd  = fig.add_subplot(gs[0, 1], facecolor='#080808')
    ax_bp  = fig.add_subplot(gs[1, 1], facecolor='#080808')
    ax_trg = fig.add_subplot(gs[2, 1], facecolor='#080808')
    ax_vid = fig.add_subplot(gs[3, 1], facecolor='#080808')

    # ── PPI static decorations ────────────────────────────────────────────────
    ax_ppi.set_aspect('equal')
    ax_ppi.set_xlim(-MAX_RANGE_M, MAX_RANGE_M)
    ax_ppi.set_ylim(-MAX_RANGE_M, MAX_RANGE_M)
    ax_ppi.axis('off')

    for r_nm in np.arange(0.5, RANGE_NM + 0.01, 0.5):
        r_m = r_nm * METERS_PER_NM
        ax_ppi.add_patch(plt.Circle((0, 0), r_m, color='#193219',
                                    linewidth=0.6, fill=False, linestyle='--'))
        ax_ppi.text(0, r_m + MAX_RANGE_M * 0.014, f'{r_nm:.1f} NM',
                    color='#336633', fontsize=6.5, ha='center', va='bottom')

    for b in range(0, 360, 10):
        rad = np.radians(b)
        s, e = 0.93, 0.99
        ax_ppi.plot([MAX_RANGE_M * s * np.sin(rad), MAX_RANGE_M * e * np.sin(rad)],
                    [MAX_RANGE_M * s * np.cos(rad), MAX_RANGE_M * e * np.cos(rad)],
                    color='#254525', linewidth=0.8)
        if b % 30 == 0:
            f = 1.04
            ax_ppi.text(MAX_RANGE_M * f * np.sin(rad),
                        MAX_RANGE_M * f * np.cos(rad),
                        f'{b}°', color='#4a9a4a', fontsize=7.5,
                        ha='center', va='center')

    for lbl, bx, by in [('N', 0, 1), ('E', 1, 0), ('S', 0, -1), ('W', -1, 0)]:
        ax_ppi.text(bx * MAX_RANGE_M * 1.10, by * MAX_RANGE_M * 1.10,
                    lbl, color='#80e080', fontsize=11, fontweight='bold',
                    ha='center', va='center')

    ax_ppi.add_patch(plt.Circle((0, 0), MAX_RANGE_M,
                                color='#2a5a2a', linewidth=1.5, fill=False))

    ms = MAX_RANGE_M * 0.018
    ax_ppi.plot([-ms, ms], [0, 0], color='#40ff40', lw=1.5)
    ax_ppi.plot([0, 0], [-ms, ms], color='#40ff40', lw=1.5)
    ax_ppi.plot(0, 0, 'o', color='#40ff40', ms=3)

    # ── PPI dynamic image ─────────────────────────────────────────────────────
    img_h = ax_ppi.imshow(
        np.zeros((GRID_SIZE, GRID_SIZE), np.float32),
        cmap=RADAR_CMAP, vmin=0, vmax=1, origin='upper',
        extent=[-MAX_RANGE_M, MAX_RANGE_M, -MAX_RANGE_M, MAX_RANGE_M],
        interpolation='bilinear', zorder=1,
    )

    # Sweep line + sector glow
    sweep_line,   = ax_ppi.plot([], [], color='#70ff70', lw=1.2, alpha=0.95, zorder=3)
    sweep_sector, = ax_ppi.plot([], [], color='#30ff30', lw=7,   alpha=0.12, zorder=2)

    # ── Per-target artists ────────────────────────────────────────────────────
    target_artists = []
    for t in ppi.targets:
        # Track trail: dots at previous positions
        trail, = ax_ppi.plot([], [], 'o', color='#ffff30', ms=2.5,
                             alpha=0.55, zorder=5, markeredgewidth=0)

        # Dashed range arc at target's current range (±25° span)
        arc, = ax_ppi.plot([], [], color='#ffff50', lw=0.9,
                           linestyle='--', alpha=0.65, zorder=5)

        # Course/speed vector (solid orange line ahead of target)
        vec, = ax_ppi.plot([], [], color='#ff8020', lw=2.0, alpha=0.85,
                           solid_capstyle='round', zorder=6)

        # Distance label with dark background box
        lbl = ax_ppi.text(
            0, 0, '',
            color='#ffff60', fontsize=6.8, fontweight='bold',
            ha='left', va='bottom', zorder=7,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#080808',
                      edgecolor='#807030', linewidth=0.7, alpha=0.85),
        )

        # Small blip marker (circle around target position)
        ring, = ax_ppi.plot([], [], 'o', color='#ffff60', ms=6,
                            mfc='none', mew=1.0, alpha=0.80, zorder=6)

        target_artists.append(dict(trail=trail, arc=arc, vec=vec,
                                   lbl=lbl, ring=ring))

    # ── Title / status ────────────────────────────────────────────────────────
    title_txt = ax_ppi.text(
        0, MAX_RANGE_M * 1.17,
        f'PPI RADAR SIMULATOR  |  {SAMPLE_RATE_HZ/1e6:.0f} MS/s  |  '
        f'{RANGE_NM:.0f} NM  |  {RPM} RPM  |  {SEA_STATE} Bft',
        color='#80ff80', fontsize=8.5, ha='center', va='center', fontweight='bold',
    )
    hdg_txt = ax_ppi.text(
        -MAX_RANGE_M * 0.98, MAX_RANGE_M * 1.17,
        'ANT  000.0°', color='#60e060', fontsize=8, va='center',
    )

    # Legend
    ax_ppi.text(
        MAX_RANGE_M * 0.98, -MAX_RANGE_M * 1.06,
        '→  course vector   ●  track trail   ○  target ring   ---  range arc',
        color='#506850', fontsize=6, ha='right', va='center',
    )

    # ── Raw signal strip panel ────────────────────────────────────────────────
    sig_axes  = [ax_hd, ax_bp, ax_trg, ax_vid]
    sig_names = ['CH0  HD', 'CH1  BP', 'CH2  TRIG', 'CH3  VIDEO']
    sig_cols  = ['#ff6060', '#ffff40', '#ffffff', '#40e0ff']
    sig_lines = []
    n_sig     = 2000
    t_us      = np.arange(n_sig) / SAMPLE_RATE_HZ * 1e6

    for ax, name, col in zip(sig_axes, sig_names, sig_cols):
        ax.set_facecolor('#050505')
        ax.set_xlim(0, t_us[-1])
        ax.set_ylim(-0.15, 1.15)
        ax.tick_params(colors='#336633', labelsize=6)
        ax.spines[:].set_color('#193219')
        ax.set_ylabel(name, color=col, fontsize=7, rotation=0,
                      ha='right', va='center', labelpad=40)
        ax.yaxis.set_visible(False)
        ax.axhline(0, color='#193219', lw=0.5)
        ax.axhline(1, color='#193219', lw=0.5, linestyle='--')
        line, = ax.plot(t_us, np.zeros(n_sig), color=col, lw=0.8)
        sig_lines.append(line)

    ax_vid.set_xlabel('Time (µs)', color='#336633', fontsize=7)
    ax_vid.xaxis.set_tick_params(labelcolor='#336633')
    fig.text(0.728, 0.97,  'RAW ADC SIGNALS  (20 MS/s)',
             color='#60a060', fontsize=7.5, ha='center')
    fig.text(0.728, 0.945, 'CH0=HD   CH1=BP   CH2=TRIG   CH3=VIDEO',
             color='#336633', fontsize=6.5, ha='center')

    return (fig, img_h, sweep_line, sweep_sector,
            hdg_txt, target_artists, sig_lines)


# ─────────────────────────────────────────────────────────────────────────────
# Animation update
# ─────────────────────────────────────────────────────────────────────────────

def _range_arc_xy(range_m, bearing_deg, span_deg=30, n=50):
    """Points for a dashed arc at given range, centred on bearing_deg."""
    angles = np.radians(np.linspace(bearing_deg - span_deg / 2,
                                    bearing_deg + span_deg / 2, n))
    return range_m * np.sin(angles), range_m * np.cos(angles)


def make_updater(ppi, img_h, sweep_line, sweep_sector,
                 hdg_txt, target_artists, sig_lines):
    frame_ctr = [0]
    n_sig     = 2000

    def update(_frame):
        ppi.advance(STEPS_PER_FRAME)

        # PPI image
        img_h.set_data(ppi.cartesian_image())

        # Sweep
        b_rad = np.radians(ppi.bearing_deg)
        sweep_line.set_data([0, MAX_RANGE_M * 0.98 * np.sin(b_rad)],
                            [0, MAX_RANGE_M * 0.98 * np.cos(b_rad)])
        sect_ang = np.radians(np.linspace(ppi.bearing_deg - 15,
                                          ppi.bearing_deg, 30))
        sweep_sector.set_data(MAX_RANGE_M * 0.98 * np.sin(sect_ang),
                              MAX_RANGE_M * 0.98 * np.cos(sect_ang))

        hdg_txt.set_text(f'ANT  {ppi.bearing_deg:05.1f}°')

        # Target annotations
        for t, art in zip(ppi.targets, target_artists):
            in_range = t.range_m < MAX_RANGE_M
            vis      = in_range

            # Track trail
            if t.track:
                tx, ty = zip(*t.track)
                art['trail'].set_data(tx, ty)
            else:
                art['trail'].set_data([], [])
            art['trail'].set_visible(vis)

            # Range arc + distance label
            arc_x, arc_y = _range_arc_xy(t.range_m, t.bearing_deg)
            art['arc'].set_data(arc_x, arc_y)
            art['arc'].set_visible(vis)

            # Course vector  (60-second lookahead, scaled with SIM_SPEED)
            vx, vy = t.course_endpoint(lookahead_s=60 * SIM_SPEED)
            art['vec'].set_data([t.x, vx], [t.y, vy])
            art['vec'].set_visible(vis and t.speed_mps > 0)

            # Target ring at current position
            art['ring'].set_data([t.x], [t.y])
            art['ring'].set_visible(vis)

            # Distance label: offset slightly from target
            off = MAX_RANGE_M * 0.035
            art['lbl'].set_position((t.x + off, t.y + off))
            dist_m = int(round(t.range_m / 10) * 10)   # round to nearest 10 m
            art['lbl'].set_text(f'{t.name}\n{dist_m:,} m')
            art['lbl'].set_visible(vis)

        # Raw signal strip (refresh every 12 frames)
        if frame_ctr[0] % 12 == 0:
            sigs = simulate_raw_signals(n_sig)
            for line, key in zip(sig_lines, ['HD', 'BP', 'TRIG', 'VIDEO']):
                line.set_ydata(sigs[key])

        frame_ctr[0] += 1

    return update


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('Furuno FAR-2xx8  PPI Simulator')
    print(f'  Range res    : {RANGE_RES_M:.1f} m / bin  ({SAMPLE_RATE_HZ/1e6:.0f} MS/s)')
    print(f'  Range bins   : {N_RANGE_BINS}  ({MAX_RANGE_M:.0f} m  /  {RANGE_NM} NM)')
    print(f'  Bearings     : {N_BEARINGS} steps / revolution')
    print(f'  Sim speed    : {SIM_SPEED}×  (increase SIM_SPEED to see targets move faster)')
    print()

    ppi = PPIAnimator()
    (fig, img_h, sweep_line, sweep_sector,
     hdg_txt, target_artists, sig_lines) = build_display(ppi)

    updater = make_updater(ppi, img_h, sweep_line, sweep_sector,
                           hdg_txt, target_artists, sig_lines)

    ani = animation.FuncAnimation(
        fig, updater,
        interval=30,               # ~33 fps
        blit=False,
        cache_frame_data=False,
    )

    plt.show()


if __name__ == '__main__':
    main()
