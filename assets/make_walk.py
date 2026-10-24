"""Generates walk.gif: a shaded-ASCII loop of an original girl walking away down a night road.

The scene is drawn as luminance + part maps, shaded into RGB, then box-averaged over small square cells
in the manner of ezascii.com: each cell picks a glyph by luminance ** exponent and is inked in its own colour. Everything is periodic in the loop phase, so it loops cleanly.
Run: uv run --with numpy --with pillow assets/make_walk.py
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLS, ROWS, SS, CELL = 173, 120, 3, 5  # square character grid (CELL px pitch), supersamples per cell side
STRIDE_FRAMES, STRIDES, FRAME_MS = 12, 2, 80  # 2 strides so the lamps (2 strides apart) loop too
NF = STRIDE_FRAMES * STRIDES
FONT = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 8)  # glyphs are centred, so the 4.8 px advance is harmless
CW = LH = CELL
W, H = COLS * CW, ROWS * LH
RAMP = " .:-=+*#%@"  # ezascii's default ramp
RAMP_FIG = " dbkhao*#MW&8%B@$"  # ezascii's detailed ramp: letters give the whole scene its tonal density, as in its colour mode
# ezascii-style colour pass (brightness, contrast, saturation, gamma, in that order) and glyph exponent
BRIGHTNESS, CONTRAST, SATURATION, GAMMA, EXPONENT = 0, 45, 1.65, 1.2, 1.4
FIG_FLOOR = 0.4  # shading range of figure parts is FIG_FLOOR..1 of the base colour
INK_MIN, INK_GAIN = 170, 1.6  # glyphs only cover part of a cell, so ink is scaled (hue kept) until its brightest channel reaches this
BG_CUT = 0.02  # only true blacks are left empty
BG_FLOOR, BG_GAIN = 0.12, 3.0  # scene shading: dim L still shows its colour; BG_GAIN stretches the faint L of sky/grass/road
HAIR_TIP, HAIR_SHADOW = 0.12, 0.3  # hair luminance below which glyphs are tip marks / dense letters

SKY, STAR, TREE, GRASS, ROAD, DASH, POST, GLOW, HAIR, BOW, DRESS, LACE, SKIN, SOCK, SHOE = range(15)
COLOURS = np.array([
    (70, 50, 140), (255, 245, 215), (35, 95, 65), (45, 125, 65), (125, 105, 165), (250, 215, 110),
    (110, 100, 150), (255, 215, 150), (205, 175, 255), (255, 130, 190), (255, 224, 240),
    (255, 215, 232), (255, 205, 180), (255, 255, 255), (200, 170, 215),
], float)
GIRL_PARTS = [HAIR, BOW, DRESS, LACE, SKIN, SOCK, SHOE]

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

def girl(X, Y, t, L, P, hair_aux):
    """t: phase within one stride (0..1). Painted back to front: legs, petticoat, skirt, bodice, waist bow, arms, hair, bows.
    Phase is offset so frame 0 is a full stride (not the passing pose) and f0/f6 are mirror strides."""
    p = 2 * np.pi * t + np.pi / 2
    cx, sway, bob = W * 0.54 + 5 * np.sin(p), 5 * np.sin(p), -4 * (1 - np.cos(2 * p)) / 2
    G = 588.0

    def paint(mask, lum, part, gap=False):
        if gap:  # one-cell void around the part so neighbours keep a visible edge
            ring = dilate(mask, 3, 3) & ~mask
            L[ring], P[ring] = 0, SKY
        L[mask] = lum[mask] if isinstance(lum, np.ndarray) else lum
        P[mask] = part

    L[ellipse(X, Y, cx, G + 2, 70, 9) < 1] *= 0.45  # contact shadow
    wy, hy = 292 + bob, 420 + bob
    # legs: hip -> knee -> ankle. The forward foot is further away, so it sits ~5 rows higher and smaller;
    # the trailing foot keeps its toe down while the heel lifts, showing the sole.
    for side, ph in ((-1, p), (1, p + np.pi)):
        fwd = max(0.0, np.sin(ph))
        trail = max(0.0, -np.sin(ph + 0.5))
        hx, hip_y = cx + side * 17 - sway * 0.4, hy - 10
        ax, ay = hx + side * 2 - side * 6 * fwd, G - 10 - 50 * fwd - 30 * trail
        kx, ky = (hx + ax) / 2 + side * 14 * (fwd + trail), (hip_y + ay) / 2
        r = 7 - 2 * fwd
        thigh = capsule(X, Y, hx, hip_y, kx, ky, 7.5)
        paint(thigh | capsule(X, Y, kx, ky, ax, ay, r),
              np.where(thigh, tube(X, Y, hx, hip_y, kx, ky, 0.5, 0.95), tube(X, Y, kx, ky, ax, ay, 0.5, 0.95)), SKIN)
        sx0, sy0 = kx + (ax - kx) * 0.35, ky + (ay - ky) * 0.35
        paint(capsule(X, Y, sx0, sy0, ax, ay, r + 0.5), 0.95 - 0.12 * (np.abs(X - ax) / 7), SOCK)  # sock/skin split
        top, bot = ay + 3, ay + 11 + 30 * trail
        shoe = ellipse(X, Y, ax, (top + bot) / 2, 10 - 3.5 * fwd, (bot - top) / 2) < 1
        paint(shoe, 0.3 + 0.25 * np.clip((ay - Y) / 8 + 1, 0, 1), SHOE)
        if trail > 0.15:  # sole faces the viewer: a light band
            paint(ellipse(X, Y, ax, (top + bot) / 2 + 1, 7, (bot - top) / 2 - 2) < 1, 0.9, LACE)

    # skirt: flares from a nipped waist into a bell, scalloped lace hem, two ruffle tiers; hip drops toward the planted leg
    v = np.clip((Y - wy) / (hy - wy), 0, 1)
    scx = cx + sway * 1.6 * v ** 1.5 - 4 * np.sin(p - 0.8) * v
    hw = 34 + 96 * v ** 0.6 + 14 * np.sin(np.pi * v)
    u = (X - scx) / hw
    hem = hy + 7 * np.abs(np.sin((X - scx) / 11)) + 5 * np.sin(p) * np.clip(u, -1, 1)
    # petticoat lace peeks out below the hem with its own, finer scallops
    pet = (np.abs(u) < 1.04) & (Y > hem - 2) & (Y < hem + 11 + 5 * np.abs(np.sin((X - scx) / 5)))
    paint(pet, 0.55 + 0.3 * (hash2(np.floor(X / 4), np.floor(Y / 4)) > 0.5), DRESS)
    skirt = (Y > wy) & (Y < hem) & (np.abs(u) < 1)
    tiers = sum(np.exp(-((Y - (wy + d) - 5 * np.abs(np.sin((X - scx) / 9))) / 4) ** 2) for d in (42, 86))
    shade = 0.5 + 0.34 * (1 - u ** 2) + 0.08 * np.sin(u * 15 + 2 * np.sin(p)) - 0.75 * tiers
    paint(skirt, np.clip(shade, 0, 1), DRESS, gap=True)
    lace = (np.abs(u) < 1.02) & (Y > hem - 3) & (Y < hem + 6)
    paint(lace, 0.25 + 0.2 * (hash2(np.floor(X / 4), np.floor(Y / 4)) > 0.5), LACE)
    # bodice: shoulders taper to the waist
    k_b = np.clip((Y - 215 - bob) / (wy - 215 - bob), 0, 1)
    bod = (Y > 215 + bob) & (Y < wy + 2) & (np.abs(X - cx) < 52 - 27 * k_b ** 0.8)
    paint(bod, 0.75 + 0.2 * (1 - ((X - cx) / 52) ** 2), DRESS, gap=True)
    # waist bow loops reach well past the hair on both sides (knot and tails come after the hair)
    by = 304 + bob
    for side in (-1, 1):
        d = ellipse(X, Y, cx + side * 52, by, 36, 16, side * 0.25)
        paint(d < 1, 0.45 + 0.5 * np.clip(d, 0, 1) ** 2, BOW, gap=True)
    # arms: shoulder -> elbow -> hand, swinging opposite the same-side leg; the forearm folds up as the arm goes forward
    for side in (-1, 1):
        a = side * np.sin(p)  # +1 arm forward (away from us), -1 arm back
        sx, sy = cx + side * 62, 246 + bob
        ex, ey = sx + side * 14, sy + 46 - 6 * a
        hx, hy2 = ex + side * (14 + 16 * a), ey + 54 - 32 * a
        upper = capsule(X, Y, sx, sy, ex, ey, 5.5)
        paint(upper | capsule(X, Y, ex, ey, hx, hy2, 5),
              np.where(upper, tube(X, Y, sx, sy, ex, ey, 0.55, 1.0), tube(X, Y, ex, ey, hx, hy2, 0.55, 1.0)), SKIN, gap=True)
        paint(ellipse(X, Y, hx, hy2 + 3, 7, 8) < 1, 0.9, SKIN)
        puff = ellipse(X, Y, cx + side * 58, 228 + bob, 20, 17) < 1
        paint(puff, 0.62 + 0.3 * (1 - ellipse(X, Y, cx + side * 53, 222 + bob, 20, 17).clip(0, 1)), DRESS, gap=True)
        paint((ellipse(X, Y, cx + side * 58, 242 + bob, 22, 6) < 1) & (Y > 239 + bob), 0.3, LACE)

    # hair: its own narrow column; roots fixed, tips swing late (follow-through), ending around the upper ruffle tier
    k = np.clip((Y - 190 - bob) / 240, 0, 1) ** 1.5
    hcx = cx + (sway * 1.5 + 24 * np.sin(p - (Y - 200) / 120)) * k
    yy = Y - bob
    hwh = np.select(
        [yy < 165, yy < 205, yy < 260],
        [40 * np.sqrt(np.clip(1 - ((yy - 165) / 47) ** 2, 0, 1)), 40 - (yy - 165) * 0.2, 32 + (yy - 205) * 0.085],
        36.7 - (yy - 260) * 0.2,
    )
    hwh = hwh * (1 + 0.04 * np.sin(yy / 13 + p))
    uh = (X - hcx) / np.maximum(hwh, 1)
    tri = np.abs((uh * 1.6 + 1) % 1 * 2 - 1)  # triangle wave: three ragged V-shaped locks across the width
    tips = 312 + bob + 30 * (1 - tri) - 12 * uh ** 2 + 5 * np.sin(uh * 23 + p)
    hair = (np.abs(uh) < 1) & (Y > 118 + bob) & (Y < tips)
    sfield = uh * 40 + 2.5 * np.sin(yy / 47 + uh * 3)
    strands = 0.5 + 0.5 * np.sin(sfield)
    fine = 0.5 + 0.5 * np.sin(uh * 113 + yy / 31)
    ring = np.exp(-((yy - (152 + 14 * uh ** 2)) / 9) ** 2) + 0.3 * np.exp(-((yy - (270 + 18 * uh ** 2)) / 9) ** 2)
    lum = (0.22 + 0.3 * strands + 0.12 * fine) * (0.65 + 0.35 * (1 - uh ** 2)) + 0.45 * ring
    lum = lum * np.clip((tips - Y) / 18, 0.2, 1)  # tips thin out
    paint(hair, lum, HAIR, gap=True)
    # strand tangent (doubled-angle vector, so opposite directions average sensibly) for direction-aware glyphs
    gy, gx = np.gradient(sfield)
    phi2 = 2 * np.arctan2(-gy, gx)
    hair_aux[0][hair], hair_aux[1][hair], hair_aux[2][hair] = np.cos(phi2)[hair], np.sin(phi2)[hair], np.clip(ring, 0, 1)[hair]
    # waist bow knot and tails hang below the hair tips, trailing the sway
    paint(ellipse(X, Y, cx, by, 9, 9) < 1, 0.95, BOW, gap=True)
    for side in (-1, 1):
        lag = 7 * np.sin(p - 0.8)
        paint(capsule(X, Y, cx + side * 4, by + 4, cx + side * 13 + lag, by + 58, 5.5), 0.7 + 0.1 * side, BOW, gap=True)
    # bow on the back of the head
    hby = 124 + bob
    for side in (-1, 1):
        d = ellipse(X, Y, cx + side * 36, hby, 38, 21, side * 0.35)
        paint(d < 1, 0.55 + 0.4 * np.clip(d, 0, 1), BOW, gap=True)
    paint(ellipse(X, Y, cx, hby, 10, 10) < 1, 0.95, BOW, gap=True)

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
    hairc = fig & (cell((P == HAIR).astype(float)) > 0.5)
    phi = np.degrees(np.arctan2(hs, hc) / 2)  # strand angle from vertical, +ve leans down-right
    rnd = hash2(np.arange(COLS)[None, :] + 3.0, np.arange(ROWS)[:, None] + 5.0)
    vert = np.array(list("|I|l!|"))[(rnd * 6).astype(int)]
    strand = np.where(np.abs(phi) < 9, vert, np.where(phi > 0, "\\", "/"))
    shadow = np.array(list("B8M0Z"))[(rnd * 5).astype(int)]
    tip = np.array(list(",.:;|"))[(rnd * 5).astype(int)]
    hg = np.where(hr > 0.35, np.array(list("@W"))[(rnd * 2).astype(int)],
                  np.where(lum < HAIR_TIP, tip, np.where(lum < HAIR_SHADOW, shadow, strand)))
    chars = np.where(hairc, hg, chars)
    col[hairc & (hr > 0.35)] = col[hairc & (hr > 0.35)] * 0.4 + 255 * 0.6
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
    for i, k in enumerate((0, 6, 12, 18)):
        sheet.paste(frames[k], (i % 2 * frames[0].width, i // 2 * frames[0].height))
    pal = sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    out = Path(__file__).with_name("walk.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=FRAME_MS, loop=0, optimize=True)
    print(out, f"{out.stat().st_size / 1e6:.2f} MB", f"{NF} frames", f"{round(W)}x{round(H) + 22}")
