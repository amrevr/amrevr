"""Generates walk.gif: a shaded-ASCII loop of an original (faceless) girl who faces the camera, does a little jig, spins away with a hair
whip, walks off down a night road, then turns back to face the camera again.

The scene is drawn as luminance + part maps, shaded into RGB, then box-averaged over small square cells
in the manner of ezascii.com: each cell picks a glyph by luminance ** exponent and is inked in its own colour.
Run: uv run --with numpy --with pillow assets/make_walk.py      (writes assets/walk.gif, the classic timeline)

WRITING A VARIANT (a separate script that imports this module and supplies its own timeline)
    import sys; sys.path.insert(0, "<repo>/assets"); import make_walk as mw
    tl = mw.Timeline(nf, pose_fn)      # pose_fn(f) -> mw.Pose for every integer f in 0..nf INCLUSIVE; frame nf is the frame after the last
    mw.render_gif(tl, "out.gif")       # checks the loop, renders frames 0..nf-1, saves with the shared palette / optimiser settings
A variant must provide: nf (frames, 80 ms each) and pose_fn. Everything else (girl, background, shading, glyphs) is reused. Build pose_fn from
the helpers or by hand: mw.Keys (Catmull-Rom yaw/squat/tuck/arm keys), mw.jig_at, mw.springs (hair / ribbon / skirt dynamics), mw.gait
(gait phase, walk amplitude, travel; mw.walk_amp is the amplitude ramp), mw.make_pose_fn (glues them into a pose_fn), or return Pose(...) directly.
See classic_timeline() for a full example; Timeline.travel_fn(f) is just pose_fn(f).travel.

LOOP CHECK (mw.check_loop(tl), run by render_gif): pose_fn(nf) must equal pose_fn(0): the same rest pose fields, yaw modulo 360 degrees, gait phase
`fl` modulo 2 pi, jig phase `wph`, fidget clock `t` modulo nf, and identical rendered characters. TRAVEL must also come back: pose.travel is metres
walked since frame 0, and it must END EQUAL TO ITS START MODULO 2.8 m (STEP_M * 2, the lamp spacing and road-dash period), so the road and the
lamps wrap seamlessly. In practice: walk a whole even number of strides in total (gait(..., strides=2, 4, ...)), or not at all (travel = 0).
A pose's `loop` field must equal nf (the hand fidget and the twinkling stars are periodic in it); make_pose_fn sets it for you.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLS, ROWS, SS, CELL = 173, 120, 3, 5  # square character grid (CELL px pitch), supersamples per cell side
STRIDE_FRAMES, STRIDES, FRAME_MS = 14, 2, 80  # base cadence: 14 frames per stride (two steps); she covers exactly 2 strides (= the lamp spacing) per loop
# ---- BASE segment boundaries, before the inserted walk block(s) shift WIND_FROM and later by BLOCK (frames at 80 ms; yaw 180 = facing the camera, 0 = back) ----
IDLE_N = 6                     # 0..5   front idle: hands at the chest bow, breathing
JIG_FROM, JIG_N = 6, 18        # 6..23  the jig: two knee-dips, hip sway, shoulder shimmy, head tilts, a wave
TURN_FROM = 24                 # 24..25 anticipation crouch, 26..28 spin to the back, 29..33 overshoot and settle
WALK_FROM = 32                 # left heel strike, during the settle; gait cadence starts 2 frames earlier
WIND_FROM, STOP_AT = 49, 57    # wind-down start; standstill (frame 57 stands with her back to us)
RET_FROM, BASE_NF = 58, 68     # 58..59 small crouch, 60..62 spin to the front, 63..67 settle; frame nf is frame 0 again
AMP_RAMP = 4                   # walk amplitude reaches 100% this many frames after WALK_FROM (frame 36)
INSERT_AT, BLOCK = 36, 24      # classic_timeline() inserts BLOCK full-amplitude walk frames here, per block
FONT = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 8)  # glyphs are centred, so the 4.8 px advance is harmless
CW = LH = CELL
W, H = COLS * CW, ROWS * LH
RAMP = " .:-=+*#%@"  # ezascii's default ramp
RAMP_FIG = " dbkhao*#MW&8%B@$"  # ezascii's detailed ramp: letters give the whole scene its tonal density, as in its colour mode
# ezascii-style colour pass (brightness, contrast, saturation, gamma, in that order) and glyph exponent
BRIGHTNESS, CONTRAST, SATURATION, GAMMA, EXPONENT = 0, 45, 1.65, 1.2, 1.4
FIG_FLOOR = 0.5  # shading range of figure parts is FIG_FLOOR..1 of the base colour
INK_MIN, INK_GAIN = 170, 1.6  # glyphs only cover part of a cell, so ink is scaled (hue kept) until its brightest channel reaches this
BG_CUT = 0.02  # only true blacks are left empty
BG_FLOOR, BG_GAIN = 0.12, 3.0  # scene shading: dim L still shows its colour; BG_GAIN stretches the faint L of sky/grass/road
HAIR_TIP, HAIR_SHADOW = 0.12, 0.3  # hair luminance below which glyphs are tip marks / dense letters

SKY, STAR, TREE, GRASS, ROAD, DASH, POST, GLOW, HAIR, HAIR_S, BOW, BOW_S, BLOUSE, TRIM, SKIRT, SKIRT_S, RUFFLE, SKIN, SOCK, SHOE, BLUSH = range(21)
COLOURS = np.array([
    (70, 50, 140), (255, 245, 215), (35, 95, 65), (45, 125, 65), (125, 105, 165), (250, 215, 110),
    (110, 100, 150), (255, 215, 150),
    (122, 82, 72), (210, 150, 120),            # hair, warm sheen
    (201, 168, 245), (189, 164, 180),         # bow / ribbon, shade
    (253, 250, 251), (43, 58, 107),           # blouse, navy trim
    (201, 168, 245), (112, 74, 206), (204, 198, 205),  # skirt, skirt shadow, ruffle
    (241, 207, 198), (250, 250, 252), (90, 52, 35),    # skin, sock, loafer
    (244, 128, 158),                                   # blush (the face has no eyes or mouth)
], float)
GIRL_PARTS = list(range(HAIR, BLUSH + 1))

# camera: horizon line, focal length (px), eye height (m); she walks 1.4 m per stride
HOR, VPX, F, EYE, STEP_M = 230.0, W * 0.47, 626.0, 1.3, 1.4

def hash2(a, b):
    """Deterministic 0..1 noise per integer lattice cell."""
    return (np.sin(a * 127.1 + b * 311.7) * 43758.5453) % 1.0

def ellipse(X, Y, cx, cy, rx, ry, rot=0.0):
    c, s = np.cos(rot), np.sin(rot)
    dx, dy = X - cx, Y - cy
    return ((dx * c + dy * s) / rx) ** 2 + ((-dx * s + dy * c) / ry) ** 2

def capsule(X, Y, ax, ay, bx, by, r):
    """Mask of points within r of segment a-b."""
    px, py, vx, vy = X - ax, Y - ay, bx - ax, by - ay
    h = np.clip((px * vx + py * vy) / (vx * vx + vy * vy), 0, 1)
    return (px - vx * h) ** 2 + (py - vy * h) ** 2 < r * r

def tube(X, Y, ax, ay, bx, by, lo, hi):
    """Round-limb shading along segment a-b: lit from the left, falling off toward the far edge."""
    vx, vy = bx - ax, by - ay
    n = np.hypot(vx, vy) + 1e-6
    s = ((X - ax) * vy - (Y - ay) * vx) / n / 7  # signed distance from the axis, in limb radii
    return lo + (hi - lo) * np.exp(-(((s * np.sign(vy + 1e-6)) + 0.35) / 0.8) ** 2)

def background(X, Y, t2, travel, L, P, T):
    travel = np.round(travel, 6) % (STEP_M * STRIDES)  # the speckle noise below repeats with this, so the ground wraps exactly
    """t2: phase over the whole loop (0..1, twinkle); travel: metres walked, wraps every STEP_M * STRIDES."""
    L[:] = 0.04 + 0.05 * np.clip((HOR - Y) / HOR, 0, 1)
    P[:] = SKY
    # stars that twinkle on integer loop frequencies
    sx, sy = np.floor(X / CELL), np.floor(Y / CELL)
    star = (hash2(sx, sy) > 0.99) & (Y < HOR - 40)
    tw = 0.5 + 0.5 * np.sin(2 * np.pi * (t2 * np.floor(1 + 3 * hash2(sy, sx)) + hash2(sx + 7, sy)))
    L[star] = 0.35 + 0.6 * tw[star]
    P[star] = STAR
    # distant tree line hugging the horizon
    canopy = HOR - 14 - 16 * (0.5 + 0.5 * np.sin(X / 23)) * (0.6 + 0.4 * np.sin(X / 7.3 + 1.7))
    trees = (Y > canopy) & (Y <= HOR)
    L[trees] = 0.04
    P[trees] = TREE
    # ground: back-project each pixel below the horizon to road coordinates
    g = Y > HOR + 0.5
    z = np.where(g, F * EYE / np.maximum(Y - HOR, 0.5), 1e9)
    xw = (X - VPX) * z / F
    zt = z + travel
    on_road = g & (np.abs(xw) < 3.2)
    grass = g & ~on_road
    L[grass] = np.where(hash2(np.floor(xw * 2), np.floor(zt / 0.7) % 4) > 0.93, 0.4, 0.05)[grass] / (1 + z[grass] / 40)
    P[grass] = GRASS
    L[on_road] = np.where(hash2(np.floor(xw * 4), np.floor(zt / 0.35) % 8) > 0.97, 0.35, 0.05)[on_road]
    P[on_road] = ROAD
    edge = on_road & (np.abs(np.abs(xw) - 3.0) < 0.12)
    L[edge] = 0.6 / (1 + z[edge] / 60)
    dash = on_road & (np.abs(xw) < 0.09) & (z > 8) & ((zt / STEP_M) % 1.0 < 0.5)  # one dash cycle per stride
    L[dash] = 0.95 / (1 + z[dash] / 60)
    P[dash] = DASH
        # streetlamps 2 strides apart on both sides, staggered by one stride
    spacing = STEP_M * STRIDES
    glow = np.zeros_like(L)  # T collects the warm lamp tint on sky, grass and road
    for side, off in ((-1, 0.0), (1, STEP_M)):
        for k in range(14):
            zl = 5.0 + off + k * spacing - (travel % spacing)
            if zl < 5.0:
                zl += 14 * spacing
            x0, base, top = VPX + side * F * 5.0 / zl, HOR + F * EYE / zl, HOR - F * (5.0 - EYE) / zl
            post = (np.abs(X - x0) < max(0.6, F * 0.09 / zl)) & (Y > top) & (Y < base)
            L[post] = np.maximum(L[post], 0.5 / (1 + zl / 25))
            P[post] = POST
            hr = max(3.0, F * 0.12 / zl)  # tight halo around the lamp head
            glow += 1.2 * np.exp(-(((X - x0) ** 2 + (Y - top) ** 2) / hr ** 2)) / (1 + zl / 40)
            T += 0.9 * np.exp(-(((X - x0) ** 2 + (Y - top) ** 2) / (3 * hr) ** 2)) / (1 + zl / 40)
            T += np.where(g, 0.3 * np.exp(-((xw - side * 3.5) ** 2 + (z - zl) ** 2) / 10), 0) / (1 + zl / 60)  # light pool on the ground
    L += glow
    P[glow > 0.2] = GLOW

def dilate(m, rx, ry):
    """Binary dilation by shifting; wrap-around at the borders is harmless (figure is central)."""
    out = m.copy()
    for dy in range(-ry, ry + 1):
        for dx in range(-rx, rx + 1):
            out |= np.roll(m, (dy, dx), (0, 1))
    return out

def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)

def interp_s(y, ys, vs):
    """Cosine-eased piecewise interpolation: smooth profile through knots (no blocky corners)."""
    i = np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2)
    ys, vs = np.asarray(ys), np.asarray(vs)
    t = smooth((y - ys[i]) / (ys[i + 1] - ys[i]))
    return vs[i] + (vs[i + 1] - vs[i]) * t

def limb(X, Y, a, b, ra, rb, bulge=0.0):
    """Tapered capsule a->b: returns signed distance (<0 inside), position along the limb h, and sideways coordinate s (-1..1)."""
    px, py, vx, vy = X - a[0], Y - a[1], b[0] - a[0], b[1] - a[1]
    n = np.hypot(vx, vy) + 1e-6
    h = np.clip((px * vx + py * vy) / (n * n), 0, 1)
    r = ra + (rb - ra) * h + bulge * np.sin(np.pi * h)
    d = np.hypot(px - vx * h, py - vy * h) - r
    s = np.clip((px * vy - py * vx) / n / r, -1, 1) * np.sign(vy + 1e-6)
    return d, h, s

def round_lum(s, lo, hi):
    """Round-limb shading: lit from the left, falling off toward the far edge."""
    return lo + (hi - lo) * np.exp(-((s + 0.35) / 0.9) ** 2)

# ---- timeline: a Timeline is nf poses; the classic one runs front idle -> jig -> spin away -> walk -> wind-down -> spin back to the front ----
@dataclass
class Pose:
    """Everything that changes between frames. The defaults are the front idle pose (frame 0 and frame nf)."""
    t: float = 0.0         # frame number; only drives the hands' fidget, which is periodic in `loop`
    loop: int = 68         # frames in the loop (set it to the timeline's nf)
    fl: float = 0.0        # gait phase, radians (0 = left heel strike)
    amp: float = 0.0       # walk amplitude 0..1: bob, sway, twist, arm swing, hair/ribbon/hem lag
    travel: float = 0.0    # metres walked since frame 0 (must end equal to its start modulo STEP_M * 2)
    yaw_hip: float = np.pi  # turn about the vertical axis, radians: 0 = back to us, pi = facing us, + = the face swings toward screen-left;
    yaw_sh: float = np.pi   # the shoulders and head lead the hips
    yaw_head: float = np.pi
    squat: float = 0.0     # knee bend 0..1 (the body drops 0.03 H at 1)
    tuck: float = 0.0      # front: hands pulled in to the chest (anticipation)
    near: float = 0.0      # 0..1 blend of the near arm toward "swung out" and the far hand toward the hip (front: out wide, both)
    far: float = 0.0
    sway: float = 0.0      # hips shift sideways, fractions of H, + = screen-right
    shim: float = 0.0      # shoulder roll, radians (the shimmy)
    htilt: float = 0.0     # head tilt, radians, + = crown toward screen-right
    wave: float = 0.0      # 0..1 right hand raised from the bow, waving at phase wph
    wph: float = 0.0
    bob: float = 0.0       # breathing, fractions of H
    fan: float = 0.0       # spin speed 0..1: the hair and hem fan out
    whip: float = 0.0      # hair tip angle off the spine, radians, + = streaming toward screen-left
    tail_x: float = 0.0    # ribbon tail tip offset, fractions of H, + = screen-right
    flare: float = 0.0     # extra hem flare 0..1 (0.29 H -> 0.34 H wide) with a 5 degree tilt
    hem_x: float = 0.0     # hem lag offset, fractions of H
    bounce: float = 0.0    # vertical lag of the hair tips, signed

@dataclass
class Timeline:
    """A looping animation: nf frames (FRAME_MS each). pose_fn(f) -> Pose must work for every integer f in 0..nf; pose_fn(nf) must flow into pose_fn(0)."""
    nf: int
    pose_fn: Callable[[int], Pose]
    def travel_fn(self, f):
        return self.pose_fn(f).travel

# The two spins, keyed per frame (80 ms): yaw in degrees (+ = the face swings toward screen-left), knee bend, hands tucked in at the chest,
# the near / far arm swinging out. Catmull-Rom between neighbouring frames; before the first key she faces us, and a gap between keys holds the
# earlier one (walk). The jig and the idle are parametric (jig_at, make_pose_fn).
SH_LEAD, HEAD_LEAD = 0.25, 0.5  # shoulders and head sample the yaw curve this many frames ahead of the hips
SPIN_AWAY = {  # frame  yaw  squat tuck near far
    23: (180,  0.0, 0.0, 0.0, 0.0),   # the jig ends upright
    24: (180,  0.6, 1.0, 0.0, 0.0),   # anticipation: dip, hands pull in
    25: (180,  1.0, 1.0, 0.0, 0.0),   #
    26: (136,  0.3, 0.2, 0.6, 0.6),   # fast ease-out spin, head ahead of the shoulders, shoulders ahead of the hips; arms fling out
    27: ( 62,  0.0, 0.0, 1.0, 1.0),
    28: (  8,  0.0, 0.0, 1.0, 1.0),
    29: (-14,  0.0, 0.0, 0.9, 0.9),   # overshoots past the back view ...
    30: ( -5,  0.0, 0.0, 0.6, 0.6),
    31: (  2,  0.0, 0.0, 0.3, 0.3),   # ... and settles; the first step lands at WALK_FROM
    32: (  0,  0.0, 0.0, 0.1, 0.1),
    33: (  0,  0.0, 0.0, 0.0, 0.0),
}
SPIN_BACK = {  # frame relative to RET_FROM (the walk is over, standing with her back to us at RET_FROM - 1)
    0: (  0,  0.4, 0.0, 0.0, 0.0),   # the return: a light dip ...
    1: (  0,  0.7, 0.0, 0.0, 0.0),
    2: (-58,  0.2, 0.0, 0.5, 0.5),   # ... and a quicker spin the other way round, through the right-hand profile
    3: (-134, 0.0, 0.0, 0.8, 0.8),
    4: (-188, 0.0, 0.0, 0.6, 0.6),   # a little overshoot
    5: (-173, 0.0, 0.0, 0.3, 0.3),
    6: (-182, 0.0, 0.0, 0.1, 0.1),
    7: (-180, 0.0, 0.0, 0.0, 0.0),   # landed on the front idle pose
    8: (-180, 0.0, 0.0, 0.0, 0.0),
    9: (-180, 0.0, 0.0, 0.0, 0.0),
    10: (-180, 0.0, 0.0, 0.0, 0.0),  # = frame 0 again (yaw -180 is yaw 180)
}

class Keys:
    """Catmull-Rom through a {frame: (yaw deg, squat, tuck, near, far)} table; keys(f) works at fractional frames. Before the first key she faces us
    (yaw 180 is only right if the first key says so); after the last key the last one holds; a gap between keys holds the earlier one."""
    def __init__(self, table):
        self.kf = sorted(table)
        self.kv = np.array([table[k] for k in self.kf], float)
    def __call__(self, f):
        KF, KV = self.kf, self.kv
        if f < KF[0]:
            return KV[0]
        if f >= KF[-1]:
            return KV[-1]
        i = max(j for j in range(len(KF)) if KF[j] <= f)
        if KF[i + 1] - KF[i] > 1:
            return KV[i]
        u = f - KF[i]
        p1, p2 = KV[i], KV[i + 1]
        p0 = KV[i - 1] if i > 0 and KF[i] - KF[i - 1] == 1 else p1
        p3 = KV[i + 2] if i + 2 < len(KF) and KF[i + 2] - KF[i + 1] == 1 else p2
        k = 0.5 * ((2 * p1) + (p2 - p0) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2 + (3 * p1 - p0 - 3 * p2 + p3) * u ** 3)
        return np.concatenate([k[:1], np.maximum(k[1:], 0)])  # the spline must not undershoot rest on the knees and arms

JIG_SQUAT, JIG_SWAY = 0.8, 0.022  # two dips this deep; the hips swing this far (H) from side to side, once across both dips
def jig_at(f, jig_from=JIG_FROM, jig_n=JIG_N):
    """The jig, parametric over jig_from .. jig_from + jig_n (jig_from=None: no jig): sway, shoulder roll, head tilt, knee bend, wave amount and phase.
    Everything is 0 at both ends. Two dips (u = 0.25, 0.75) with the hips swung to opposite sides on them; four shoulder shimmies;
    the head tilts left then right; the right hand leaves the bow between u = 0.3 and 0.88 to wave."""
    if jig_from is None:
        return np.zeros(6)
    u = (f - jig_from) / jig_n
    if not 0 < u < 1:
        return np.zeros(6)
    env = smooth(min(u, 1 - u) * 6)
    return np.array([JIG_SWAY * np.sin(2 * np.pi * u) * env, np.radians(5) * np.sin(2 * np.pi * 4 * u) * env,
                     -np.radians(9) * np.sin(2 * np.pi * u) * env, JIG_SQUAT * (0.5 - 0.5 * np.cos(4 * np.pi * u)),
                     smooth((u - 0.30) / 0.10) * smooth((0.88 - u) / 0.10), 2 * np.pi * 4 * u])

# hair, ribbons, skirt, vertical lag: natural frequency (rad/s), damping, gain (hair angle per rad, ribbon offset H per rad, skirt flare per rad, lag per unit)
SPRING_W, SPRING_ZETA = np.array([13.0, 8.0, 10.0, 11.0]), np.array([0.3, 0.3, 0.35, 0.35])
SPRING_GAIN = np.array([3.3, 0.08, 0.9, 2.0])
SWAY_COUPLE, RET_GAIN = 8.0, 0.6  # head sway (H) counts this much as yaw (rad) for the springs; the return spin drives them this much as the first

def head_drive(keys, jig_from=JIG_FROM, ret_from=None):
    """drive(f) -> [head yaw (rad, plus the jig's sway), squat] for springs(); the return spin (f >= ret_from - 1) drives them RET_GAIN as hard."""
    def drive(f):
        jg = jig_at(f, jig_from)
        yaw = np.radians(keys(f + HEAD_LEAD)[0]) * (RET_GAIN if ret_from is not None and f >= ret_from - 1 else 1)
        return np.array([yaw + SWAY_COUPLE * (0.5 * jg[0] + 0.07 * np.sin(jg[2])), keys(f)[1] + jg[3]])
    return drive

def springs(nf, drive):
    """Hair, ribbons and skirt as damped springs driven by the head's yaw acceleration (and its sideways sway): a'' = -drive'' - 2 zeta w a' - w^2 a.
    The turn starts, they lag; the turn stops, they swing on; the turn back overshoots; ~1.5 swings later they are still. A fourth spring is
    driven by the vertical acceleration of the knee dips. drive(f) -> [yaw, squat] (see head_drive). Returns an (nf + 1, 4) table
    [whip, tail, skirt, bounce]; outputs are soft-limited and fade out over the last 3 frames so the loop closes."""
    sub = 16
    dt = FRAME_MS / 1000 / sub
    a, v, out = np.zeros(4), np.zeros(4), [np.zeros(4)]
    for i in range(nf * sub):
        f = i / sub
        acc = (drive(f + 1 / sub) - 2 * drive(f) + drive(f - 1 / sub)) / dt ** 2
        acc = np.array([acc[0], acc[0], acc[0], acc[1]])
        v += dt * (-acc - 2 * SPRING_ZETA * SPRING_W * v - SPRING_W ** 2 * a)
        a += dt * v
        if (i + 1) % sub == 0:
            out.append(a.copy())
    out = np.array(out) * SPRING_GAIN
    out[:, 0], out[:, 1] = 1.45 * np.tanh(out[:, 0] / 1.45), 0.3 * np.tanh(out[:, 1] / 0.3)
    out *= smooth((nf - np.arange(nf + 1)) / 3)[:, None]
    out[np.abs(out) < 2e-3] = 0  # sub-pixel residue
    return out

def walk_amp(f, walk_from, ramp=AMP_RAMP):
    """Walk amplitude 0..1: 50% over the 3 frames before walk_from (the settle), then up to 100% over `ramp` frames. Full from walk_from + ramp."""
    return 0.5 * smooth((f - (walk_from - 3)) / 3) + 0.5 * smooth((f - walk_from) / ramp)

def gait(nf, walk_from, wind_from, stop_at, strides, ramp=AMP_RAMP):
    """Returns (fl, amp, travel), each of length nf + 1. Cadence eases in from walk_from - 2, runs steady, then eases out over wind_from .. stop_at
    (steps shorten to nothing, amp -> 0). Its scale is solved so she covers exactly `strides` strides (STEP_M each): use an even number so travel
    wraps with the lamps and the road dashes. fl = 0 (left heel strike) at walk_from."""
    f = np.arange(nf + 1)
    wind = smooth((stop_at - f) / (stop_at - wind_from))
    e = smooth((f - (walk_from - 2)) / 5) * wind
    c = e[:nf] * strides / e[:nf].sum()               # strides per frame
    stride = np.concatenate([[0], np.cumsum(c)])      # strides walked since frame 0
    fl = 2 * np.pi * (stride - stride[walk_from])
    return fl, walk_amp(f, walk_from, ramp) * wind, stride * STEP_M

def make_pose_fn(nf, keys, spr, gt, jig_from=None, idle_n=0):
    """Glue for a pose_fn: keys = Keys(...), spr = springs(...) table, gt = gait(...) tuple, jig_from = start of the jig (or None), idle_n = frames of
    breathing at the start. Returns pose_fn(f) -> Pose for f in 0..nf."""
    fl, amp, travel = gt
    def pose_at(f):
        k, jg = keys(f), jig_at(f, jig_from)
        yaw = lambda lead: np.radians(keys(f + lead)[0]) % (2 * np.pi)  # an angle: -180 and 180 are the same bits
        whip, tail, skirt, bounce = spr[f]
        fan = min(abs(keys(f + 0.5)[0] - keys(f - 0.5)[0]) / 75, 1)
        sk = min(abs(skirt), 1)
        return Pose(t=f, loop=nf, fl=fl[f], amp=amp[f], travel=travel[f], yaw_hip=yaw(0), yaw_sh=yaw(SH_LEAD), yaw_head=yaw(HEAD_LEAD),
                    squat=k[1] + jg[3], tuck=k[2], near=k[3], far=k[4], sway=jg[0], shim=jg[1], htilt=jg[2], wave=jg[4], wph=jg[5],
                    bob=0.005 * np.sin(np.pi * f / idle_n) if f < idle_n else 0.0, fan=fan, whip=whip, tail_x=-tail,
                    flare=min(sk + 0.8 * fan + 0.4 * min(abs(bounce), 1), 1), hem_x=-0.05 * np.sign(skirt) * sk, bounce=bounce)
    return pose_at

def classic_timeline(blocks=1):
    """The classic walk.gif timeline. `blocks` extra full-amplitude BLOCK-frame walk stretches are inserted at INSERT_AT (the first full-amplitude
    frame); everything after (wind-down, return spin) shifts along, and the solved cadence still covers exactly 2 * (1 + blocks) strides."""
    n = BLOCK * blocks
    wind_from, stop_at, ret_from, nf = WIND_FROM + n, STOP_AT + n, RET_FROM + n, BASE_NF + n
    keys = Keys({**SPIN_AWAY, **{ret_from + k: v for k, v in SPIN_BACK.items()}})
    spr = springs(nf, head_drive(keys, JIG_FROM, ret_from))
    gt = gait(nf, WALK_FROM, wind_from, stop_at, STRIDES * (1 + blocks))
    return Timeline(nf, make_pose_fn(nf, keys, spr, gt, JIG_FROM, IDLE_N))

CLASSIC = classic_timeline()
pose_at, NF = CLASSIC.pose_fn, CLASSIC.nf

def bend(X, Y, rx, ry, theta, length):
    """Straighten a hair spine that hangs from (rx, ry) and curls by theta (radians at its tip, growing as s**1.6, + = toward screen-left).
    Returns (X', Y') in which the spine is vertical, so the straight-hair code can run on a whipped lock; theta = 0 returns X, Y unchanged."""
    ell = np.linspace(-0.2 * length, 1.6 * length, 80)
    th = theta * np.clip(ell / length, 0, 1) ** 1.2
    tx, ty = -np.sin(th), np.cos(th)                  # spine tangent; the normal is (ty, -tx)
    d_ell = np.gradient(ell)
    sx = rx + np.cumsum(tx * d_ell) - np.interp(0, ell, np.cumsum(tx * d_ell))
    sy = ry + np.cumsum(ty * d_ell) - np.interp(0, ell, np.cumsum(ty * d_ell))
    best, bi = np.full(X.shape, np.inf), np.zeros(X.shape, int)
    for i in range(len(ell)):
        d2 = (X - sx[i]) ** 2 + (Y - sy[i]) ** 2
        better = d2 < best
        best, bi = np.where(better, d2, best), np.where(better, i, bi)
    px, py = X - sx[bi], Y - sy[bi]
    return rx + px * ty[bi] - py * tx[bi], ry + ell[bi] + px * tx[bi] + py * ty[bi]

def painter(L, P):
    def paint(mask, lum, part, gap=False):
        if gap:  # one-cell void around the part so neighbours keep a visible edge
            ring = dilate(mask, 3, 3) & ~mask
            L[ring], P[ring] = 0, SKY
        L[mask] = lum[mask] if isinstance(lum, np.ndarray) else lum
        P[mask] = part[mask] if isinstance(part, np.ndarray) else part
    return paint

def girl(X, Y, pz, L, P, hair_aux):
    """pz: the Pose (gait phase, walk amplitude, spin, jig and lags). Facing us (cos of the shoulder yaw < 0) she is painted back to front:
    hair, ribbon tails and bow behind her, legs, skirt, blouse, neck, head and face, arms, sleeves, side locks. From behind: legs, skirt, blouse,
    arms, sleeves, hair, ribbon tails, bow. Every length is a fraction of H."""
    # ---- proportions (fractions of H, measured from the crown unless noted) ----
    G0, H = 556.0, 480.0                   # ground row under her at rest, body height in px (about 7.3 heads)
    HIP_H, LT, LS, FOOT = 0.488, 0.235, 0.225, 0.03  # hip height above ground, thigh, shin, ankle height
    Y_SH, Y_WAIST, Y_HEM = 0.19, 0.40, 0.70           # shoulder line, skirt waist, skirt hem
    SH_HALF, HIP_SEP, FOOT_SEP = 0.10, 0.040, 0.022   # half shoulder width, hip joint and foot offset from the midline
    DEPTH = 244.0                          # px per H of forward depth at foot level (camera at 1.3 m)
    # ---- motion amplitudes ----
    A = pz.amp                             # the walk's amplitudes all scale with the gait amplitude
    SWAY, HIP_TILT = 0.025 * A, np.radians(4) * A
    HIP_YAW, SH_YAW, SH_TILT, TWIST_GAIN = np.radians(10), np.radians(15), np.radians(5) * A, 1.8 * A
    ARM_SWING, HAIR_SWAY, TAIL_SWAY, HEM_SWAY = np.radians(15) * A, 0.035 * A, 0.06 * A, 0.03 * A
    HAIR_LAG, TAIL_LAG, HEM_LAG = 0.15 * 2 * np.pi, 0.25 * 2 * np.pi, 0.1 * 2 * np.pi
    Yh, Ys, Yd = pz.yaw_hip, pz.yaw_sh, pz.yaw_head   # spin (0 = back; the walk's own twist is separate)
    front = np.cos(Ys) < 0                 # facing us: the whole figure is drawn from the front
    cos_h = abs(np.cos(Yh))
    psi_s, psi_d = np.pi - Ys, np.pi - Yd  # angle off the camera axis, 0 = facing us (+ = the face swings toward screen-left)
    TW = np.hypot(np.cos(Ys), 0.6 * np.sin(Ys))        # torso is an ellipse (depth 0.6 of width): projected width through the turn
    SK = np.hypot(np.cos(Yh), 0.8 * np.sin(Yh))        # skirt is nearly round

    fl = pz.fl                             # left leg phase: 0 = heel strike (leg furthest forward)
    wrap = lambda a: a % (2 * np.pi)
    vm = lambda a, c, k: np.exp(k * (np.cos(a - c) - 1))   # periodic bump centred on c
    paint = painter(L, P)

    # ---- gait and knee-dip kinematics: thigh swing, knee flexion, hip height from the planted leg ----
    leg = {}
    for side, ph in ((-1, fl), (1, fl + np.pi)):
        th = np.radians(19) * A * np.cos(ph) + np.radians(2) + np.radians(42) * pz.squat
        flex = np.radians(6 + A * (52 * vm(ph, 4.6, 2.5) + 8 * vm(ph, 0.35, 2.5))) + np.radians(84) * pz.squat
        leg[side] = (th, flex, LT * np.cos(th) + LS * np.cos(th - flex) + FOOT)
    k_plant = 0.5 + 0.5 * min(pz.squat, 1)  # in a dip both feet stay planted: the hips drop by the whole leg shortening
    hip_h = k_plant * min(leg[-1][2], leg[1][2]) + (1 - k_plant) * HIP_H
    dy = -(hip_h - HIP_H) * H - pz.bob * H  # the whole upper body rides the hip height
    walk_x = -SWAY * H * np.sin(fl)        # pelvis shifts over the planted foot
    jx = pz.sway * H                       # the jig's hip sway; the feet stay put
    hipx = walk_x + jx
    cx = W * 0.54
    dirn = np.cos(Yh)                      # which way "forward" points on screen: up when she faces away, down when she faces us
    pj = lambda lx, z, h, c, f=None: (c + lx * H, G0 - H * h - dirn * DEPTH * z * (max(0.35, 1 - h / 0.7) if f is None else f))
    top = G0 - H + dy                      # crown row
    yy = lambda frac: top + frac * H

    yaw_h, yaw_s = -HIP_YAW * np.cos(fl) * TWIST_GAIN, SH_YAW * np.cos(fl) * TWIST_GAIN  # + = right side turns away
    yaw_w, yaw_s = yaw_s, yaw_s + Ys       # walk twist alone vs. total shoulder yaw (sleeves, collar)
    tilt_h = HIP_TILT * np.sin(fl)         # + = right hip drops (it is the swing side)
    tilt_s = np.tan(SH_TILT * np.cos(fl) + pz.shim)  # forward-turning shoulder drops (walk); the shimmy rolls them in the jig
    shx = cx + walk_x * 0.3 + jx * 0.5 + 0.1 * H * pz.shim - 0.012 * H * np.sin(Ys)  # shoulders sway less than the hips; the chest bulges toward the turn
    hdx = cx + walk_x * 0.15 + jx * 0.7 + 0.1 * H * np.radians(3) * pz.squat  # head
    hdx += 0.07 * H * np.sin(pz.htilt)     # the head tilts about the neck
    hrx = hdx + 0.04 * H * np.sin(Yd)      # the back of the head (hair, bow) swings away from the turn
    wv = np.clip((yy(0.50) - Y) / (yy(0.50) - yy(Y_SH)), 0, 1)
    bx = (cx + hipx) * (1 - wv) + shx * wv  # torso midline, hips -> shoulders
    bowx, bowy = hrx + 0.028 * H * np.cos(Yd), yy(0.05)

    L[ellipse(X, Y, cx + walk_x * 0.5, G0 + 4, 80, 10) < 1] *= 0.45  # contact shadow

    def back_upper():
        """Seen from behind: sailor collar with its double stripe, arms swinging opposite the legs, puff sleeves."""
        # ---- blouse: shoulders -> waist, shoulder line tilts and counter-rotates against the hips ----
        Yw = Y - tilt_s * (X - bx) * smooth(wv * 1.1)   # inverse shoulder tilt
        fy = (Yw - top) / H
        half = interp_s(fy, [0.135, 0.16, Y_SH, 0.24, 0.31, Y_WAIST, 0.415], [0.040, 0.082, SH_HALF * np.cos(yaw_w), 0.088, 0.073, 0.066, 0.068]) * H * TW
        blouse = (np.abs(X - bx) < half) & (fy > 0.135) & (fy < 0.415)
        dxn = (X - bx) / np.maximum(half, 1)
        paint(blouse, 0.82 + 0.15 * (1 - dxn ** 2) - 0.2 * smooth((np.abs(dxn) - 0.8) / 0.2), BLOUSE)
        # square sailor collar on the back with a navy double stripe; its centre shifts with the shoulder yaw
        cdx = X - bx - 0.012 * H * np.sin(yaw_s) - 0.045 * H * np.sin(Ys)  # the back of the collar swings away from the turn
        cy_b = 0.285
        dcol = np.minimum(0.092 * H * np.cos(yaw_s) - np.abs(cdx), (cy_b - fy) * H)
        collar = (fy > 0.15) & (dcol > 0) & blouse
        paint(collar, 0.7 + 0.18 * np.cos(cdx / 9), BLOUSE)
        trim = collar & (((dcol > 0.010 * H) & (dcol < 0.019 * H)) | ((dcol > 0.029 * H) & (dcol < 0.038 * H)))
        paint(trim, 0.7, TRIM)

        # ---- arms: swing opposite the same-side leg, elbow bent ~15-30 deg, arms hang close to the body, swing mostly fore/aft ----
        sleeves = []
        for side in (-1, 1):
            u = side * np.cos(fl)
            ia = pz.near if side < 0 else pz.far                     # intro: near arm swings out, far hand goes to the hip
            al = ARM_SWING * u                                       # + forward
            al += ia * ((0.15 if side < 0 else 0.10) - al)
            back = A * max(0.0, -u)
            sc = 1 - side * 0.4 * np.sin(yaw_s)                      # foreshortening: far shoulder/sleeve narrows, near one widens
            sdx = side * SH_HALF * np.cos(yaw_s)
            sy_off = side * tilt_s * SH_HALF * H * 0.9
            sx, sy = shx + sdx * H, yy(0.215) + sy_off
            be = np.radians(2 + 3 * back)
            fle = np.radians(22 + 6 * A * u)
            bf = be + np.radians(3 * back + 1.5 * A * np.sin(fl * 2 + side))  # slight wrist flick
            be += ia * (np.radians(18 if side < 0 else 35) - be)
            bf += ia * (np.radians(30 if side < 0 else -10) - bf)
            h0 = 1 - 0.215
            ex_, ez_ = side * 0.17 * np.sin(be), 0.17 * np.sin(al)
            eh = h0 - 0.17 * np.cos(al) * np.cos(be)
            fa = al + fle
            wx_, wz_ = ex_ + side * 0.15 * np.sin(bf), ez_ + 0.15 * np.sin(fa)
            wh = eh - 0.15 * np.cos(fa) * np.cos(bf)
            cy_, sy_ = np.cos(Ys), np.sin(Ys)                        # turn the arm with the shoulders: sideways -> toward the viewer
            ex_, ez_, wx_, wz_ = ex_ * cy_ - ez_ * sy_, ez_ * cy_ + ex_ * sy_, wx_ * cy_ - wz_ * sy_, wz_ * cy_ + wx_ * sy_
            elx, ely = pj(sdx + ex_, ez_, eh, shx, 0.8)
            wrx, wry = pj(sdx + wx_, wz_, wh, shx, 0.8)
            ely += sy_off; wry += sy_off
            d1, h1, s1 = limb(X, Y, (sx, sy), (elx, ely), 0.031 * H * sc, 0.023 * H)
            d2, h2, s2 = limb(X, Y, (elx, ely), (wrx, wry), 0.023 * H, 0.0155 * H)
            dd = np.minimum(d1, d2)
            paint(dd < 0, np.where(d1 < d2, round_lum(s1, 0.5, 0.95), round_lum(s2, 0.5, 0.95)), SKIN, gap=True)
            hdir = np.array([wrx - elx, wry - ely]) / (np.hypot(wrx - elx, wry - ely) + 1e-6)
            hand = ellipse(X, Y, wrx + hdir[0] * 0.015 * H, wry + hdir[1] * 0.015 * H, 0.0185 * H, 0.026 * H, np.arctan2(-hdir[0], hdir[1])) < 1
            paint(hand, 0.9, SKIN)
            sleeves.append((side, sc, sx, sy))
        for side, sc, sx, sy in sleeves:  # short puffy sleeves with navy double-stripe hem
            ex0 = sx + side * 0.012 * H
            d = ellipse(X, Y, ex0, sy - 0.004 * H, 0.050 * H * sc, 0.040 * H, side * 0.35)
            bot = sy + 0.04 * H
            slv = (d < 1) & (Y < bot)
            paint(slv, 0.9 - 0.25 * np.clip(d, 0, 1) ** 2 - 0.18 * np.clip((Y - sy) / (0.04 * H), 0, 1), BLOUSE, gap=True)
            paint(slv & (((Y > bot - 0.024 * H) & (Y < bot - 0.016 * H)) | ((Y > bot - 0.011 * H) & (Y < bot - 0.003 * H))), 0.7, TRIM)


    def hair():
        """The long wavy hair: roots follow the head, tips swing late with an S-curve wave and curled ends."""
        # ---- hair: roots follow the head, tips swing late with a S-curve wave and curled ends ----
        Xw, Yw_ = bend(X, Y, hrx, top + 0.10 * H, pz.whip, 0.40 * H) if abs(pz.whip) > 1e-3 else (X, Y)  # the whip curls the lock; below, X, Y -> Xw, Yw_
        f = (Yw_ - top) / H
        kk = np.clip((f - 0.12) / 0.33, 0, 1)
        hph = fl - HAIR_LAG
        sway_h = -HAIR_SWAY * H * np.sin(hph) * kk ** 1.4
        wave = H * (0.006 + 0.022 * kk ** 1.3) * np.sin(2 * np.pi * f / 0.30 - hph) \
            + 0.013 * H * smooth((f - 0.33) / 0.08) * np.sin(2 * np.pi * (f - 0.33) / 0.10 - hph * 2)
        hcx = hrx + sway_h + wave
        hwh = H * np.where(f < 0.075, 0.066 * np.sqrt(np.clip(1 - ((0.075 - f) / 0.075) ** 2, 0, 1)),
                           0.060 + 0.034 * smooth((f - 0.13) / 0.30))
        hwh = hwh * (1 + 0.05 * np.sin(f * 40 + hph) * kk + 0.10 * smooth((f - 0.37) / 0.06) + (0.45 * min(abs(pz.whip), 1.1) + 0.35 * pz.fan) * kk ** 1.3) * (1.15 if front else 1)  # a whipped lock fans out; from the front more of it shows beside her
        uh = (Xw - hcx) / np.maximum(hwh, 1)
        tri = np.abs((uh * 1.6 + 1) % 1 * 2 - 1)
        tips = top + H * (0.435 + 0.035 * (1 - tri) - 0.012 * uh ** 2 + 0.010 * np.cos(2 * hph) + 0.004 * np.sin(uh * 23 + hph) + 0.03 * pz.bounce * kk)
        hair = (np.abs(uh) < 1) & (f > 0) & (Yw_ < tips)
        sfield = uh * 34 + 3 * np.sin(2 * np.pi * f / 0.30 - hph + uh)
        strands = 0.5 + 0.5 * np.sin(sfield)
        fine = 0.5 + 0.5 * np.sin(uh * 97 + f * 14)
        ring = np.exp(-((f - (0.062 + 0.02 * uh ** 2)) / 0.02) ** 2) + 0.4 * np.exp(-((f - (0.27 + 0.03 * uh ** 2 + 0.01 * np.sin(hph))) / 0.025) ** 2)
        lum = (0.28 + 0.3 * strands + 0.14 * fine) * (0.7 + 0.3 * (1 - uh ** 2)) + 0.4 * ring
        lum = lum * np.clip((tips - Yw_) / 18, 0.25, 1)
        paint(hair, lum, np.where(ring > 0.35, HAIR_S, HAIR), gap=True)
        gy, gx = np.gradient(sfield)
        phi2 = 2 * np.arctan2(-gy, gx)
        hair_aux[0][hair], hair_aux[1][hair], hair_aux[2][hair] = np.cos(phi2)[hair], np.sin(phi2)[hair], np.clip(ring, 0, 1)[hair]


    def ribbons():
        """The two ribbon tails: they hang from the bow, trail the motion with a big lag and flutter."""
        # ---- ribbon tails: hang from the bow, trail the motion with a big lag, sway widely and flutter ----
        for side in (-1, 1):
            ph = fl - TAIL_LAG - side * 0.5
            sv = (Y - bowy) / (0.50 * H)
            xc = bowx + side * 0.016 * H * sv - TAIL_SWAY * H * np.sin(ph) * np.clip(sv, 0, 1) ** 1.3 + pz.tail_x * H * np.clip(sv, 0, 1) ** 1.3 \
                + 0.010 * H * np.clip(sv, 0, 1) * np.sin(3 * fl - 5 * sv + side)
            w = 0.017 * H * (0.55 + 0.45 * np.abs(np.cos(sv * 8 - 2 * fl + side)))
            dn = (X - xc) / w
            end = 1 - 0.07 * (1 - np.abs(dn))
            rib = (np.abs(dn) < 1) & (sv > 0.02) & (sv < end)
            lm = 0.72 + 0.25 * np.sin(sv * 16 - 2 * fl * 1 + side) - 0.15 * np.abs(dn)
            paint(rib, np.clip(lm, 0.3, 1), np.where(lm < 0.62, BOW_S, BOW), gap=True)

    def bow():
        """The big bow on the back of the head."""
        # ---- big bow on the back of the head ----
        wob = 0.08 * A * np.sin(2 * fl)
        cd = abs(np.cos(Yd))                                         # the bow turns edge-on with the head
        for side in (-1, 1):
            d = ellipse(X, Y, bowx + side * 0.066 * H * cd, bowy, 0.064 * H * (0.4 + 0.6 * cd), 0.037 * H, -side * (0.35 + wob))
            hole = ellipse(X, Y, bowx + side * 0.070 * H * cd, bowy, 0.032 * H * (0.4 + 0.6 * cd), 0.014 * H, -side * (0.35 + wob)) < 1
            lm = 0.6 + 0.4 * np.clip(d, 0, 1)
            paint(d < 1, np.where(hole, 0.45, lm), np.where(hole | (d > 0.6), BOW_S, BOW), gap=True)
        paint(ellipse(X, Y, bowx, bowy, 0.018 * H, 0.020 * H) < 1, 0.95, BOW, gap=True)


    def front_upper():
        """Facing us: blouse with the sailor V and the lilac neckerchief, neck, head with the face, bangs, arms, puff sleeves, side locks."""
        c_s, s_s = np.cos(psi_s), np.sin(psi_s)
        # ---- blouse and collar ----
        Yw = Y - tilt_s * (X - bx) * smooth(wv * 1.1)    # inverse shoulder tilt
        fy = (Yw - top) / H
        half = interp_s(fy, [0.135, 0.16, Y_SH, 0.24, 0.31, Y_WAIST, 0.415], [0.040, 0.082, SH_HALF, 0.088, 0.073, 0.066, 0.068]) * H * TW
        blouse = (np.abs(X - bx) < half) & (fy > 0.135) & (fy < 0.415)
        ring = dilate(blouse, 3, 3) & ~blouse & np.isin(P, (HAIR, HAIR_S, BOW, BOW_S))  # a void line keeps the blouse apart from the hair behind it
        L[ring], P[ring] = 0, SKY
        dxn = (X - bx) / np.maximum(half, 1)
        paint(blouse, 0.82 + 0.15 * (1 - dxn ** 2) - 0.2 * smooth((np.abs(dxn) - 0.8) / 0.2), BLOUSE)
        dx = np.abs(X - bx) / H / max(abs(c_s), 0.2)       # across the chest, foreshortened with the turn
        xi = 0.03 * (0.225 - fy) / 0.075                    # inner V (the neckline) ...
        xo = 0.092 * (0.31 - fy) / 0.16                    # ... and outer V edge of the sailor collar
        neckline = blouse & (fy > 0.14) & (dx < xi)
        paint(neckline, 0.85 - 0.15 * np.clip((fy - 0.14) / 0.09, 0, 1), SKIN)
        collar = blouse & (fy > 0.15) & (fy < 0.31) & (dx < xo) & (dx > np.maximum(xi, 0))
        paint(collar, 0.7 + 0.2 * np.clip(1 - np.abs(xo - dx) / 0.05, 0, 1), BLOUSE)
        de = xo - dx
        trim = collar & (((de > 0.008) & (de < 0.014)) | ((de > 0.021) & (de < 0.027)))
        paint(trim, 0.7, TRIM)
        # lilac neckerchief tied at the chest: two loops, a knot and two short tails
        nx, ny = shx, yy(0.283)
        for side in (-1, 1):
            tail = capsule(X, Y, nx + side * 0.006 * H, ny, nx + side * 0.024 * H * (1 + 0.5 * pz.fan), ny + 0.07 * H, 0.0105 * H)
            paint(tail, 0.78, BOW, gap=True)
        for side in (-1, 1):
            d = ellipse(X, Y, nx + side * 0.030 * H * abs(c_s), ny, 0.032 * H * (0.4 + 0.6 * abs(c_s)), 0.0165 * H, -side * 0.30)
            hole = ellipse(X, Y, nx + side * 0.032 * H * abs(c_s), ny, 0.015 * H * (0.4 + 0.6 * abs(c_s)), 0.007 * H, -side * 0.30) < 1
            paint(d < 1, np.where(hole, 0.5, 0.6 + 0.4 * np.clip(d, 0, 1)), np.where(hole | (d > 0.6), BOW_S, BOW), gap=True)
        paint(ellipse(X, Y, nx, ny, 0.0145 * H, 0.0155 * H) < 1, 0.95, BOW, gap=True)

        # ---- neck and head (head-local coordinates u, v in H, v = 0 at the head centre; the crown is at v = -0.07) ----
        hcx, hcy = hdx, top + 0.07 * H
        tc, ts = np.cos(pz.htilt), np.sin(pz.htilt)
        U, V = ((X - hcx) * tc + (Y - hcy) * ts) / H, (-(X - hcx) * ts + (Y - hcy) * tc) / H
        cd_, sd_ = np.cos(psi_d), np.sin(psi_d)
        neck = (np.abs(X - (hcx * 0.4 + bx * 0.6)) < 0.017 * H) & (Y > hcy) & (Y < yy(0.19))
        paint(neck, 0.7, SKIN)
        rxv = 0.052 * (1 - 0.22 * np.clip(V / 0.067, 0, 1) ** 2)
        head = (U / rxv) ** 2 + (V / 0.067) ** 2 < 1
        face_side = np.where(sd_ >= 0, U < 0.052 * cd_, U > -0.052 * cd_)  # the hairline slides across the head as it turns away
        cap = head & ~face_side
        paint(cap, 0.45 + 0.25 * (0.5 + 0.5 * np.sin(U * 300)), HAIR)
        skin = head & face_side
        paint(skin, 0.92 - 0.25 * np.clip((V + 0.01) / 0.07, 0, 1) ** 2 - 0.12 * (U / 0.052) ** 2 - 0.06 * U / 0.052, SKIN)
        # faceless: no eyes, nose or mouth, only a light blush on each cheek; it foreshortens and slides with the turn (profile: a plain silhouette)
        w = max(cd_, 0.25)
        fx = lambda u0: u0 * cd_ - 0.05 * sd_
        for sg in (-1, 1):
            paint((ellipse(U, V, fx(sg * 0.036), 0.036, 0.020 * w, 0.0115) < 1) & skin, 0.95, BLUSH)
        # bangs across the forehead, parted at the centre, with a sheen ring; strands slant outward and down
        ub = (U + 0.05 * sd_) / max(cd_, 0.35)
        vt = V + 0.07
        vb = 0.049 + 0.021 * (np.abs(ub) / 0.052) ** 1.5 + 0.0035 * np.sin(ub * 350)
        bangs = (((U / (rxv + 0.004)) ** 2 + (V / 0.071) ** 2) < 1) & (vt < vb) & (face_side | (np.abs(U) < 0.004))
        sf = (ub + 0.25 * np.sign(ub) * vt) * 62
        ring_b = np.exp(-(((vt - 0.024 - 0.012 * (ub / 0.05) ** 2)) / 0.011) ** 2)
        lum_b = (0.3 + 0.3 * (0.5 + 0.5 * np.sin(sf)) + 0.12 * (0.5 + 0.5 * np.sin(ub * 190 + 3 * vt))) * (0.8 + 0.2 * (1 - (ub / 0.06) ** 2)) + 0.4 * ring_b
        paint(bangs, lum_b, np.where(ring_b > 0.35, HAIR_S, HAIR))
        gy, gx = np.gradient(sf)
        phi2 = 2 * np.arctan2(-gy, gx)
        hair_aux[0][bangs], hair_aux[1][bangs], hair_aux[2][bangs] = np.cos(phi2)[bangs], np.sin(phi2)[bangs], np.clip(ring_b, 0, 1)[bangs]

        # ---- arms: two-bone reach from the shoulder to the hand; the hands rest at the bow, tuck in, fling out in the spin, or one waves ----
        LU, LF = 0.17 * H, 0.15 * H
        sleeves = []
        for side in (-1, 1):                                  # side +1 = screen-right = her left; it is the near side when the face swings to screen-left
            sc = 1 + side * 0.4 * s_s
            sx, sy_off = shx + side * SH_HALF * abs(c_s) * H, side * tilt_s * SH_HALF * H * 0.9
            sx, sy = sx, yy(0.215) + sy_off
            fid = 2 * np.pi * 17 * pz.t / pz.loop + 1.3 * side      # the fidget: four frames per cycle, periodic over the loop
            hold = (1 - pz.tuck) * (1 - 0.8 * (pz.near if side > 0 else pz.far))
            hand = np.array([shx + side * 0.020 * H + hold * 0.006 * H * np.cos(fid), yy(0.300) + hold * 0.005 * H * np.sin(2 * fid)])
            tuck = np.array([shx + side * 0.008 * H, yy(0.310)])
            out = np.array([shx + side * 0.26 * H * abs(c_s), yy(0.42)])
            arms_out = pz.near if side > 0 else pz.far
            T = hand + pz.tuck * (tuck - hand)
            T = T + arms_out * (out - T)
            wave = pz.wave if side < 0 else 0.0
            T = T + wave * (np.array([shx - 0.2 * H + 0.035 * H * np.sin(pz.wph), yy(0.115 - 0.012 * np.cos(pz.wph))]) - T)
            S = np.array([sx, sy])
            d = np.hypot(*(T - S))
            d = np.clip(d, 0.06 * H, 0.97 * (LU + LF))
            dv = (T - S) / (np.hypot(*(T - S)) + 1e-6)
            a_ = (LU ** 2 - LF ** 2 + d ** 2) / (2 * d)
            perp = np.array([-dv[1], dv[0]])
            pole = np.array([side * (0.5 + 1.0 * wave + 0.8 * arms_out), 1.0])
            if perp @ pole < 0:
                perp = -perp
            E = S + a_ * dv + np.sqrt(max(LU ** 2 - a_ ** 2, 0)) * perp
            Wr = S + d * dv
            d1, h1, s1 = limb(X, Y, S, E, 0.031 * H * sc, 0.023 * H)
            d2, h2, s2 = limb(X, Y, E, Wr, 0.023 * H, 0.0155 * H)
            paint(np.minimum(d1, d2) < 0, np.where(d1 < d2, round_lum(s1, 0.5, 0.95), round_lum(s2, 0.5, 0.95)), SKIN, gap=True)
            hdir = (Wr - E) / (np.hypot(*(Wr - E)) + 1e-6)
            big = 1 + 0.25 * wave
            paint(ellipse(X, Y, Wr[0] + hdir[0] * 0.015 * H, Wr[1] + hdir[1] * 0.015 * H, 0.0185 * H * big, 0.026 * H * big, np.arctan2(-hdir[0], hdir[1])) < 1, 0.9, SKIN)
            sleeves.append((side, sc, sx, sy))
        for side, sc, sx, sy in sleeves:  # short puffy sleeves with navy double-stripe hem
            ex0 = sx + side * 0.012 * H
            d = ellipse(X, Y, ex0, sy - 0.004 * H, 0.050 * H * sc, 0.040 * H, side * 0.35)
            bot = sy + 0.04 * H
            slv = (d < 1) & (Y < bot)
            paint(slv, 0.9 - 0.25 * np.clip(d, 0, 1) ** 2 - 0.18 * np.clip((Y - sy) / (0.04 * H), 0, 1), BLOUSE, gap=True)
            paint(slv & (((Y > bot - 0.024 * H) & (Y < bot - 0.016 * H)) | ((Y > bot - 0.011 * H) & (Y < bot - 0.003 * H))), 0.7, TRIM)
        # ---- side locks framing the face, down past the shoulders; they swing with the hair springs ----
        vl = (Y - top) / H
        for side in (-1, 1):
            xc = hcx + H * (side * 0.057 * cd_ - 0.05 * sd_ + 0.008 * np.sin(vl * 26 + side)) - pz.whip * 0.07 * H * smooth((vl - 0.08) / 0.25) ** 1.3 \
                + 0.07 * H * np.sin(pz.htilt) * smooth((vl - 0.1) / 0.2)
            wl = H * (0.0105 + 0.002 * np.sin(vl * 50)) * (0.6 + 0.4 * max(cd_, 0.3)) * smooth((vl - 0.04) / 0.04) * (1 - 0.7 * smooth((vl - 0.22) / 0.11) ** 1.5)
            lock = (np.abs(X - xc) < wl) & (vl > 0.04) & (vl < 0.325 + 0.01 * side)
            ul = (X - xc) / np.maximum(wl, 1)
            lum_l = (0.3 + 0.35 * (0.5 + 0.5 * np.sin(ul * 6 + vl * 30))) * (0.75 + 0.25 * (1 - ul ** 2)) + 0.3 * np.exp(-((vl - 0.12) / 0.03) ** 2)
            paint(lock, lum_l, np.where(lum_l > 0.62, HAIR_S, HAIR))
            gy, gx = np.gradient(ul * 4 + vl * 70 * 0.2)
            phi2 = 2 * np.arctan2(-gy, gx)
            hair_aux[0][lock], hair_aux[1][lock], hair_aux[2][lock] = np.cos(phi2)[lock], np.sin(phi2)[lock], (lum_l > 0.62)[lock] * 0.0

    if front:  # hair, ribbons and bow sit behind her
        hair(); ribbons(); bow()

    # ---- legs ----
    for side in (-1, 1):
        th, flex, V = leg[side]
        hx, hyy = pj(side * HIP_SEP * cos_h, 0, hip_h, cx + hipx)
        hyy += side * tilt_h * HIP_SEP * H
        kh, kz = hip_h - LT * np.cos(th), LT * np.sin(th)
        ah, az = kh - LS * np.cos(th - flex), kz + LS * np.sin(th - flex)
        kx, ky = pj((side * (HIP_SEP + FOOT_SEP) / 2 + side * 0.004 * flex) * cos_h, kz, kh, cx + walk_x + 0.5 * jx)
        ax, ay = pj(side * FOOT_SEP * cos_h, az, ah, cx + walk_x)
        fwd = np.clip(az / 0.15, 0, 1)
        d1, h1, s1 = limb(X, Y, (hx, hyy), (kx, ky), 0.036 * H, 0.025 * H)
        d2, h2, s2 = limb(X, Y, (kx, ky), (ax, ay), 0.024 * H, 0.0145 * H, bulge=0.005 * H)
        d = np.minimum(d1, d2) - 0  # thigh and shin joined at the knee
        knee = ellipse(X, Y, kx, ky, 0.026 * H, 0.026 * H) < 1
        skin = (d < 0) | knee
        paint(skin, np.where(d1 < d2, round_lum(s1, 0.55, 0.95), round_lum(s2, 0.55, 0.95)), SKIN)
        sock = (d2 < 0.8) & (h2 > 0.3) & (d2 <= d1)
        paint(sock, 0.62 + 0.33 * (1 - np.abs(s2 + 0.3)) ** 2, SOCK)
        lift = np.clip(ah - FOOT, 0, 0.05) * H   # trailing heel lifts: the shoe stretches down and shows its sole
        sh_h = 0.034 * H + lift
        shoe = ellipse(X, Y, ax, ay + 0.006 * H + sh_h / 2, 0.026 * H * (1 - 0.12 * fwd), sh_h / 2 + 0.01 * H) < 1
        paint(shoe & (Y > ay - 0.004 * H), 0.3 + 0.25 * np.clip((ay - Y) / 8 + 1, 0, 1), SHOE)
        if th < 0 and lift > 0.012 * H:
            paint(shoe & (Y > ay + 0.02 * H + lift * 0.3), 0.55, SHOE)

    # ---- skirt: pleated A-line at the waist, knee length, hem sways and flares with a lag ----
    wy, hy = yy(Y_WAIST), yy(Y_HEM)
    hem_x = cx - HEM_SWAY * H * np.sin(fl - HEM_LAG) + pz.hem_x * H
    flare = A * abs(np.sin(fl - HEM_LAG)) + 2 * pz.flare  # intro flare is worth two walk flares: hem 0.29 H -> 0.34 H wide
    v = np.clip((Y - wy) / (hy - wy), 0, 1)
    scx = (cx + hipx) + (hem_x - (cx + hipx)) * v ** 1.4
    hw = H * (0.070 + (0.092 + 0.025 * flare) * v ** 0.9) * (1 + 0.04 * np.sin(np.pi * v)) * SK
    u = (X - scx) / hw
    top_line = wy + np.tan(tilt_h) * (X - cx - hipx)
    hem = hy + np.tan(A * np.radians(5) * np.sin(fl - HEM_LAG) - 100 * pz.hem_x * np.pi / 180) * (X - scx) + 0.004 * H * np.abs(np.sin((X - scx) / 14))
    skirt = (Y > top_line) & (Y < hem) & (np.abs(u) < 1)
    q = u * 5 + 5 + 1.6 * (yaw_h + 0.15 * A * np.sin(fl)) * 5 * 0.3 - 2.5 * np.sin(Yh)  # pleat lines slide with the hip turn
    fr = q % 1.0
    crease = (fr < 0.1) | (fr > 0.96)
    shaded = fr > 0.52
    sl = np.where(shaded, 0.68, 0.92) + 0.06 * (1 - u ** 2) - 0.08 * v
    sl = np.where(crease, 0.5, sl)
    paint(skirt, np.clip(sl, 0, 1), np.where(shaded | crease, SKIRT_S, SKIRT), gap=True)
    ruf = (np.abs(u) < 1.04) & (Y > hem - 2) & (Y < hem + 0.017 * H + 0.006 * H * np.abs(np.sin((X - scx) / 5)))
    paint(ruf, 0.72 + 0.2 * (hash2(np.floor(X / 4), np.floor(Y / 4)) > 0.5), RUFFLE)
    stripe = (np.abs(u) < 1) & (np.abs(Y - (hem - 0.03 * H)) < 2.6) & skirt
    paint(stripe, 0.85, RUFFLE)

    if front:
        front_upper()
    else:
        back_upper(); hair(); ribbons(); bow()

def shade_rgb(P, L, T, X, Y):
    """Part base colour x luminance, with a slow tonal drift inside each part and cool shadows / warm lights.
    The scene gets a violet sky gradient and warm lamp tint; the figure never goes fully dark."""
    drift = 0.9 + 0.2 * (0.5 + 0.25 * np.sin(X / 17 + 2 * np.sin(Y / 23)) + 0.25 * np.sin(Y / 11 - X / 29))
    fig = np.isin(P, GIRL_PARTS)
    L = np.where(fig, FIG_FLOOR + (1 - FIG_FLOOR) * L, BG_FLOOR + (1 - BG_FLOOR) * np.clip(L * BG_GAIN, 0, 1))
    lit = np.clip(L * drift, 0, 1)[..., None] ** np.array([1.25, 1.0, 0.8])
    base = COLOURS[P]
    sky = np.clip(Y / HOR, 0, 1)[..., None] ** 1.5 * np.array([60, 30, 30])
    base = np.where((P == SKY)[..., None], base + sky + np.array([-30, -20, 0]) * (1 - np.clip(Y / HOR, 0, 1))[..., None], base)
    return base * lit + T[..., None] * np.array([230, 120, 30]) * (~fig)[..., None]

def enhance(rgb):
    """ezascii's pixel pass: brightness, contrast, saturation, gamma."""
    rgb = rgb + BRIGHTNESS
    rgb = (259 * (CONTRAST + 255) / (255 * (259 - CONTRAST))) * (rgb - 128) + 128
    grey = (rgb * (.299, .587, .114)).sum(-1, keepdims=True)
    rgb = np.clip(grey + (rgb - grey) * SATURATION, 0, 255)
    return 255 * (rgb / 255) ** (1 / GAMMA)

def frame(tl, f):
    """Returns (chars, colours) grids for frame f of Timeline tl."""
    ys, xs = ((np.arange(n * SS) + 0.5) / SS for n in (ROWS, COLS))
    X, Y = np.meshgrid(xs * CW, ys * LH)
    L, P = np.zeros_like(X), np.zeros(X.shape, int)
    aux = [np.zeros_like(X) for _ in range(3)]  # hair strand tangent (cos2a, sin2a) and highlight
    T = np.zeros_like(X)
    pz = tl.pose_fn(f)
    background(X, Y, (f % tl.nf) / tl.nf, pz.travel, L, P, T)
    girl(X, Y, pz, L, P, aux)
    cell = lambda a: a.reshape(ROWS, SS, COLS, SS, *a.shape[2:]).mean(axis=(1, 3))
    col = enhance(cell(shade_rgb(P, L, T, X, Y)))
    lum = np.clip((col * (.299, .587, .114)).sum(-1) / 255, 0, 1)
    fig = cell(np.isin(P, GIRL_PARTS).astype(float)) > 0.4
    # dim scene cells keep the first visible glyph so the sky and grass stay filled with colour
    idx = np.maximum(np.floor(lum ** EXPONENT * (len(RAMP_FIG) - 1)).astype(int), 1)
    chars = np.where(lum > BG_CUT, np.array(list(RAMP_FIG))[idx], " ")
    # hair: strand glyphs follow the local strand angle; shadows get dense letters, tips and gaps stay light
    hc, hs, hr = (cell(a) for a in aux)
    hairc = fig & (cell(np.isin(P, (HAIR, HAIR_S)).astype(float)) > 0.5)
    phi = np.degrees(np.arctan2(hs, hc) / 2)  # strand angle from vertical, +ve leans down-right
    rnd = hash2(np.arange(COLS)[None, :] + 3.0, np.arange(ROWS)[:, None] + 5.0)
    vert = np.array(list("|I|l!|"))[(rnd * 6).astype(int)]
    strand = np.where(np.abs(phi) < 9, vert, np.where(phi > 0, "\\", "/"))
    shadow = np.array(list("B8M0Z"))[(rnd * 5).astype(int)]
    tip = np.array(list(",.:;|"))[(rnd * 5).astype(int)]
    hg = np.where(hr > 0.35, np.array(list("@W"))[(rnd * 2).astype(int)],
                  np.where(lum < HAIR_TIP, tip, np.where(lum < HAIR_SHADOW, shadow, strand)))
    chars = np.where(hairc, hg, chars)
    ink = col * np.clip(INK_MIN / np.maximum(col.max(-1, keepdims=True), 1), 1, INK_GAIN)
    return chars, np.clip(ink, 0, 255).astype(int)

def render(tl, f):
    img = Image.new("RGB", (round(W), round(H) + 22), (18, 14, 24))
    d = ImageDraw.Draw(img)
    chars, col = frame(tl, f)
    for r in range(ROWS):
        for c in range(COLS):
            if chars[r, c] != " ":  # centred in its cell, like ezascii's textAlign/textBaseline middle
                d.text(((c + .5) * CW, (r + .5) * LH), chars[r, c], font=FONT, fill=tuple(col[r, c]), anchor="mm")
    cap = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 13)
    d.text((W / 2, H + 11), "♪ now playing: cry — aisin-gioro & toyoki", font=cap, fill=(185, 163, 196), anchor="mm")
    return img

def check_loop(tl):
    """The frame after the last must flow into frame 0: the same rest pose (yaw compared as an angle), gait phase back on the beat, the road
    wrapped by a whole number of lamp spacings (travel ends equal to its start modulo STEP_M * 2), and identical characters."""
    a, b = tl.pose_fn(0), tl.pose_fn(tl.nf)
    fields = ("amp", "squat", "tuck", "near", "far", "sway", "shim", "htilt", "wave", "bob", "fan", "whip", "tail_x", "flare", "hem_x", "bounce")
    assert np.allclose([getattr(a, k) for k in fields], [getattr(b, k) for k in fields], atol=1e-3), "rest pose must return"
    assert abs(a.wph - b.wph) < 1e-6 and abs(a.t % tl.nf - b.t % tl.nf) < 1e-6, "jig phase and fidget clock must return"
    assert a.loop == b.loop == tl.nf, "Pose.loop must equal the timeline's nf"
    for k in ("yaw_hip", "yaw_sh", "yaw_head"):
        assert abs(np.sin(getattr(a, k) - getattr(b, k))) < 1e-6 and np.cos(getattr(a, k) - getattr(b, k)) > 0, f"{k} must come back round to the front"
    assert abs((b.fl - a.fl + np.pi) % (2 * np.pi) - np.pi) < 1e-6, "gait phase must return"
    lamp = STEP_M * STRIDES
    assert abs((b.travel - a.travel + lamp / 2) % lamp - lamp / 2) < 1e-6, "travel must end equal to its start modulo the lamp spacing"
    assert (frame(tl, 0)[0] == frame(tl, tl.nf)[0]).all(), "loop must wrap cleanly"

def render_gif(tl, out_path, check=True, palette_frames=None):
    """Renders frames 0..tl.nf-1 of Timeline tl and saves them as a looping GIF (one 256-colour palette sampled from four frames, no dither).
    Runs check_loop(tl) first unless check=False. Returns the Path."""
    if check:
        check_loop(tl)
    frames = [render(tl, f) for f in range(tl.nf)]
    sheet = Image.new("RGB", (frames[0].width * 2, frames[0].height * 2))
    for i, k in enumerate(palette_frames or [int(tl.nf * x) for x in (0, .2, .4, .65)]):
        sheet.paste(frames[k], (i % 2 * frames[0].width, i // 2 * frames[0].height))
    pal = sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    out = Path(out_path)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=FRAME_MS, loop=0, optimize=True)
    return out

if __name__ == "__main__":
    out = render_gif(CLASSIC, Path(__file__).with_name("walk.gif"), palette_frames=(0, 14, 27, 55))
    print(out, f"{out.stat().st_size / 1e6:.2f} MB", f"{CLASSIC.nf} frames", f"{round(W)}x{round(H) + 22}")
