"""Generates walk.gif: a shaded-ASCII loop of an original girl walking away down a night road.

The scene is drawn as luminance + part maps, shaded into RGB, then box-averaged over small square cells
in the manner of ezascii.com: each cell picks a glyph by luminance ** exponent and is inked in its own colour. Everything is periodic in the loop phase, so it loops cleanly.
Run: uv run --with numpy --with pillow assets/make_walk.py
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLS, ROWS, SS, CELL = 173, 120, 3, 5  # square character grid (CELL px pitch), supersamples per cell side
STRIDE_FRAMES, STRIDES, FRAME_MS = 14, 2, 80  # one 1.12 s gait cycle (two steps) per stride; 2 strides so the lamps (2 strides apart) loop too
NF = STRIDE_FRAMES * STRIDES
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

SKY, STAR, TREE, GRASS, ROAD, DASH, POST, GLOW, HAIR, HAIR_S, BOW, BOW_S, BLOUSE, TRIM, SKIRT, SKIRT_S, RUFFLE, SKIN, SOCK, SHOE = range(20)
COLOURS = np.array([
    (70, 50, 140), (255, 245, 215), (35, 95, 65), (45, 125, 65), (125, 105, 165), (250, 215, 110),
    (110, 100, 150), (255, 215, 150),
    (122, 82, 72), (210, 150, 120),            # hair, warm sheen
    (201, 168, 245), (189, 164, 180),         # bow / ribbon, shade
    (253, 250, 251), (43, 58, 107),           # blouse, navy trim
    (201, 168, 245), (112, 74, 206), (204, 198, 205),  # skirt, skirt shadow, ruffle
    (230, 186, 186), (250, 250, 252), (90, 52, 35),    # skin, sock, loafer
], float)
GIRL_PARTS = list(range(HAIR, SHOE + 1))

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

def background(X, Y, t2, L, P, T):
    """t2: phase over the whole loop (0..1, both strides)."""
    travel = t2 * STEP_M * STRIDES
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
    L[grass] = np.where(hash2(np.floor(xw * 2), np.floor(zt * 2)) > 0.93, 0.4, 0.05)[grass] / (1 + z[grass] / 40)
    P[grass] = GRASS
    L[on_road] = np.where(hash2(np.floor(xw * 4), np.floor(zt * 4)) > 0.97, 0.35, 0.05)[on_road]
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

def girl(X, Y, t, L, P, hair_aux):
    """t: phase of one full gait cycle (two steps, 0..1); frame 0 is the left heel strike.
    Painted back to front: legs, skirt, blouse, arms, sleeves, hair, ribbon tails, bow. Every length is a fraction of H."""
    # ---- proportions (fractions of H, measured from the crown unless noted) ----
    G0, H = 556.0, 480.0                   # ground row under her at rest, body height in px (about 7.3 heads)
    HIP_H, LT, LS, FOOT = 0.488, 0.235, 0.225, 0.03  # hip height above ground, thigh, shin, ankle height
    Y_SH, Y_WAIST, Y_HEM = 0.19, 0.40, 0.70           # shoulder line, skirt waist, skirt hem
    SH_HALF, HIP_SEP, FOOT_SEP = 0.10, 0.040, 0.022   # half shoulder width, hip joint and foot offset from the midline
    DEPTH = 244.0                          # px per H of forward depth at foot level (camera at 1.3 m)
    # ---- motion amplitudes ----
    SWAY, HIP_TILT = 0.025, np.radians(4)
    HIP_YAW, SH_YAW, SH_TILT, TWIST_GAIN = np.radians(10), np.radians(15), np.radians(5), 1.8
    ARM_SWING, HAIR_SWAY, TAIL_SWAY, HEM_SWAY = np.radians(25), 0.035, 0.06, 0.03
    HAIR_LAG, TAIL_LAG, HEM_LAG = 0.15 * 2 * np.pi, 0.25 * 2 * np.pi, 0.1 * 2 * np.pi

    fl = 2 * np.pi * t                     # left leg phase: 0 = heel strike (leg furthest forward)
    wrap = lambda a: a % (2 * np.pi)
    vm = lambda a, c, k: np.exp(k * (np.cos(a - c) - 1))   # periodic bump centred on c

    def paint(mask, lum, part, gap=False):
        if gap:  # one-cell void around the part so neighbours keep a visible edge
            ring = dilate(mask, 3, 3) & ~mask
            L[ring], P[ring] = 0, SKY
        L[mask] = lum[mask] if isinstance(lum, np.ndarray) else lum
        P[mask] = part[mask] if isinstance(part, np.ndarray) else part

    # ---- gait kinematics: thigh swing, knee flexion, hip height from the planted leg ----
    leg = {}
    for side, ph in ((-1, fl), (1, fl + np.pi)):
        th = np.radians(19) * np.cos(ph) + np.radians(2)
        flex = np.radians(6 + 52 * vm(ph, 4.6, 2.5) + 8 * vm(ph, 0.35, 2.5))
        leg[side] = (th, flex, LT * np.cos(th) + LS * np.cos(th - flex) + FOOT)
    hip_h = 0.5 * min(leg[-1][2], leg[1][2]) + 0.5 * HIP_H
    dy = -(hip_h - HIP_H) * H              # the whole upper body rides the hip height
    hipx = -SWAY * H * np.sin(fl)          # pelvis shifts over the planted foot
    cx = W * 0.54
    pj = lambda lx, z, h, c, f=None: (c + lx * H, G0 - H * h - DEPTH * z * (max(0.35, 1 - h / 0.7) if f is None else f))
    top = G0 - H + dy                      # crown row
    yy = lambda frac: top + frac * H

    yaw_h, yaw_s = -HIP_YAW * np.cos(fl) * TWIST_GAIN, SH_YAW * np.cos(fl) * TWIST_GAIN  # + = right side turns away
    tilt_h = HIP_TILT * np.sin(fl)         # + = right hip drops (it is the swing side)
    tilt_s = np.tan(SH_TILT * np.cos(fl))  # forward-turning shoulder drops
    shx = cx + hipx * 0.3                  # shoulders sway less than the hips
    hdx = cx + hipx * 0.15                 # head
    wv = np.clip((yy(0.50) - Y) / (yy(0.50) - yy(Y_SH)), 0, 1)
    bx = (cx + hipx) * (1 - wv) + shx * wv  # torso midline, hips -> shoulders

    L[ellipse(X, Y, cx + hipx * 0.5, G0 + 4, 80, 10) < 1] *= 0.45  # contact shadow

    # ---- legs ----
    for side in (-1, 1):
        th, flex, V = leg[side]
        hx, hyy = pj(side * HIP_SEP, 0, hip_h, cx + hipx)
        hyy += side * tilt_h * HIP_SEP * H
        kh, kz = hip_h - LT * np.cos(th), LT * np.sin(th)
        ah, az = kh - LS * np.cos(th - flex), kz + LS * np.sin(th - flex)
        kx, ky = pj(side * (HIP_SEP + FOOT_SEP) / 2 + side * 0.004 * flex, kz, kh, cx + hipx)
        ax, ay = pj(side * FOOT_SEP, az, ah, cx + hipx)
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
    hem_x = cx - HEM_SWAY * H * np.sin(fl - HEM_LAG)
    flare = abs(np.sin(fl - HEM_LAG))
    v = np.clip((Y - wy) / (hy - wy), 0, 1)
    scx = (cx + hipx) + (hem_x - (cx + hipx)) * v ** 1.4
    hw = H * (0.070 + (0.092 + 0.025 * flare) * v ** 0.9) * (1 + 0.04 * np.sin(np.pi * v))
    u = (X - scx) / hw
    top_line = wy + np.tan(tilt_h) * (X - cx - hipx)
    hem = hy + np.tan(np.radians(5) * np.sin(fl - HEM_LAG)) * (X - scx) + 0.004 * H * np.abs(np.sin((X - scx) / 14))
    skirt = (Y > top_line) & (Y < hem) & (np.abs(u) < 1)
    q = u * 5 + 5 + 1.6 * (yaw_h + 0.15 * np.sin(fl)) * 5 * 0.3  # pleat lines slide with the hip turn
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

    # ---- blouse: shoulders -> waist, shoulder line tilts and counter-rotates against the hips ----
    Yw = Y - tilt_s * (X - bx) * smooth(wv * 1.1)   # inverse shoulder tilt
    fy = (Yw - top) / H
    half = interp_s(fy, [0.135, 0.16, Y_SH, 0.24, 0.31, Y_WAIST, 0.415], [0.040, 0.082, SH_HALF * np.cos(yaw_s), 0.088, 0.073, 0.066, 0.068]) * H
    blouse = (np.abs(X - bx) < half) & (fy > 0.135) & (fy < 0.415)
    dxn = (X - bx) / np.maximum(half, 1)
    paint(blouse, 0.82 + 0.15 * (1 - dxn ** 2) - 0.2 * smooth((np.abs(dxn) - 0.8) / 0.2), BLOUSE)
    # square sailor collar on the back with a navy double stripe; its centre shifts with the shoulder yaw
    cdx = X - bx - 0.012 * H * np.sin(yaw_s)
    cy_b = 0.285
    dcol = np.minimum(0.092 * H * np.cos(yaw_s) - np.abs(cdx), (cy_b - fy) * H)
    collar = (fy > 0.15) & (dcol > 0) & blouse
    paint(collar, 0.7 + 0.18 * np.cos(cdx / 9), BLOUSE)
    trim = collar & (((dcol > 0.010 * H) & (dcol < 0.019 * H)) | ((dcol > 0.029 * H) & (dcol < 0.038 * H)))
    paint(trim, 0.7, TRIM)

    # ---- arms: swing opposite the same-side leg, elbow bent 30-50 deg, hand trails outward on the backswing ----
    sleeves = []
    for side in (-1, 1):
        al = side * ARM_SWING * np.cos(fl)                       # + forward
        back = max(0.0, -al / ARM_SWING)
        sc = 1 - side * 0.4 * np.sin(yaw_s)                      # foreshortening: far shoulder/sleeve narrows, near one widens
        sdx = side * SH_HALF * np.cos(yaw_s)
        sy_off = side * tilt_s * SH_HALF * H * 0.9
        sx, sy = shx + sdx * H, yy(0.215) + sy_off
        be = np.radians(4 + 9 * back)
        fle = np.radians(34 + 12 * (al / ARM_SWING))
        bf = be + np.radians(10 * back + 6 * np.sin(fl * 2 + side))  # loose flick
        h0 = 1 - 0.215
        ex_, ez_ = side * 0.17 * np.sin(be), 0.17 * np.sin(al)
        eh = h0 - 0.17 * np.cos(al) * np.cos(be)
        fa = al + fle
        wx_, wz_ = ex_ + side * 0.15 * np.sin(bf), ez_ + 0.15 * np.sin(fa)
        wh = eh - 0.15 * np.cos(fa) * np.cos(bf)
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

    # ---- hair: roots follow the head, tips swing late with a S-curve wave and curled ends ----
    f = (Y - top) / H
    kk = np.clip((f - 0.12) / 0.33, 0, 1)
    hph = fl - HAIR_LAG
    sway_h = -HAIR_SWAY * H * np.sin(hph) * kk ** 1.4
    wave = H * (0.006 + 0.022 * kk ** 1.3) * np.sin(2 * np.pi * f / 0.30 - hph) \
        + 0.013 * H * smooth((f - 0.33) / 0.08) * np.sin(2 * np.pi * (f - 0.33) / 0.10 - hph * 2)
    hcx = hdx + (bx - hdx) * 0 + sway_h + wave
    hwh = H * np.where(f < 0.075, 0.066 * np.sqrt(np.clip(1 - ((0.075 - f) / 0.075) ** 2, 0, 1)),
                       0.060 + 0.034 * smooth((f - 0.13) / 0.30))
    hwh = hwh * (1 + 0.05 * np.sin(f * 40 + hph) * kk + 0.10 * smooth((f - 0.37) / 0.06))
    uh = (X - hcx) / np.maximum(hwh, 1)
    tri = np.abs((uh * 1.6 + 1) % 1 * 2 - 1)
    tips = top + H * (0.435 + 0.035 * (1 - tri) - 0.012 * uh ** 2 + 0.010 * np.cos(2 * hph) + 0.004 * np.sin(uh * 23 + hph))
    hair = (np.abs(uh) < 1) & (f > 0) & (Y < tips)
    sfield = uh * 34 + 3 * np.sin(2 * np.pi * f / 0.30 - hph + uh)
    strands = 0.5 + 0.5 * np.sin(sfield)
    fine = 0.5 + 0.5 * np.sin(uh * 97 + f * 14)
    ring = np.exp(-((f - (0.062 + 0.02 * uh ** 2)) / 0.02) ** 2) + 0.4 * np.exp(-((f - (0.27 + 0.03 * uh ** 2 + 0.01 * np.sin(hph))) / 0.025) ** 2)
    lum = (0.28 + 0.3 * strands + 0.14 * fine) * (0.7 + 0.3 * (1 - uh ** 2)) + 0.4 * ring
    lum = lum * np.clip((tips - Y) / 18, 0.25, 1)
    paint(hair, lum, np.where(ring > 0.35, HAIR_S, HAIR), gap=True)
    gy, gx = np.gradient(sfield)
    phi2 = 2 * np.arctan2(-gy, gx)
    hair_aux[0][hair], hair_aux[1][hair], hair_aux[2][hair] = np.cos(phi2)[hair], np.sin(phi2)[hair], np.clip(ring, 0, 1)[hair]

    # ---- ribbon tails: hang from the bow, trail the motion with a big lag, sway widely and flutter ----
    bowx, bowy = hdx + 0.028 * H, yy(0.05)
    for side in (-1, 1):
        ph = fl - TAIL_LAG - side * 0.5
        sv = (Y - bowy) / (0.50 * H)
        xc = bowx + side * 0.016 * H * sv - TAIL_SWAY * H * np.sin(ph) * np.clip(sv, 0, 1) ** 1.3 \
            + 0.010 * H * np.clip(sv, 0, 1) * np.sin(3 * fl - 5 * sv + side)
        w = 0.017 * H * (0.55 + 0.45 * np.abs(np.cos(sv * 8 - 2 * fl + side)))
        dn = (X - xc) / w
        end = 1 - 0.07 * (1 - np.abs(dn))
        rib = (np.abs(dn) < 1) & (sv > 0.02) & (sv < end)
        lm = 0.72 + 0.25 * np.sin(sv * 16 - 2 * fl * 1 + side) - 0.15 * np.abs(dn)
        paint(rib, np.clip(lm, 0.3, 1), np.where(lm < 0.62, BOW_S, BOW), gap=True)
    # ---- big bow on the back of the head ----
    wob = 0.08 * np.sin(2 * fl)
    for side in (-1, 1):
        d = ellipse(X, Y, bowx + side * 0.066 * H, bowy - side * 0.0 * H, 0.064 * H, 0.037 * H, -side * (0.35 + wob))
        hole = ellipse(X, Y, bowx + side * 0.070 * H, bowy, 0.032 * H, 0.014 * H, -side * (0.35 + wob)) < 1
        lm = 0.6 + 0.4 * np.clip(d, 0, 1)
        paint(d < 1, np.where(hole, 0.45, lm), np.where(hole | (d > 0.6), BOW_S, BOW), gap=True)
    paint(ellipse(X, Y, bowx, bowy, 0.018 * H, 0.020 * H) < 1, 0.95, BOW, gap=True)

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

def frame(f):
    """Returns (chars, colours) grids for frame f."""
    ys, xs = ((np.arange(n * SS) + 0.5) / SS for n in (ROWS, COLS))
    X, Y = np.meshgrid(xs * CW, ys * LH)
    L, P = np.zeros_like(X), np.zeros(X.shape, int)
    aux = [np.zeros_like(X) for _ in range(3)]  # hair strand tangent (cos2a, sin2a) and highlight
    T = np.zeros_like(X)
    background(X, Y, (f % NF) / NF, L, P, T)
    girl(X, Y, (f % STRIDE_FRAMES) / STRIDE_FRAMES, L, P, aux)
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

def render(f):
    img = Image.new("RGB", (round(W), round(H) + 22), (18, 14, 24))
    d = ImageDraw.Draw(img)
    chars, col = frame(f)
    for r in range(ROWS):
        for c in range(COLS):
            if chars[r, c] != " ":  # centred in its cell, like ezascii's textAlign/textBaseline middle
                d.text(((c + .5) * CW, (r + .5) * LH), chars[r, c], font=FONT, fill=tuple(col[r, c]), anchor="mm")
    cap = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 13)
    d.text((W / 2, H + 11), "♪ now playing: cry — aisin-gioro & toyoki", font=cap, fill=(185, 163, 196), anchor="mm")
    return img

if __name__ == "__main__":
    a, b = frame(0), frame(NF)
    assert (a[0] == b[0]).all(), "loop must wrap cleanly"
    frames = [render(f) for f in range(NF)]
    sheet = Image.new("RGB", (frames[0].width * 2, frames[0].height * 2))  # one palette sampled from four frames
    for i, k in enumerate((0, 7, 14, 21)):
        sheet.paste(frames[k], (i % 2 * frames[0].width, i // 2 * frames[0].height))
    pal = sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    out = Path(__file__).with_name("walk.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=FRAME_MS, loop=0, optimize=True)
    print(out, f"{out.stat().st_size / 1e6:.2f} MB", f"{NF} frames", f"{round(W)}x{round(H) + 22}")
