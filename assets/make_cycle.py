"""Generates the README banner: one seamless loop through all four walks (classic -> wave & skip -> twirl -> look-back -> classic ...).

Run: /path/to/python -B assets/make_cycle.py [--short] [--step N] [--out PATH] [--cache DIR]      (needs numpy + pillow; ~2-3 min)
Writes assets/walk_cycle.gif by default. --short drops the look-back (smaller file; the classic keeps its full walk because the road only wraps every 5.6 m);
--step N keeps every Nth frame and stretches each frame N times longer (same speed, lower fps, smaller file).
Each segment is rendered without its last frame (that frame is frame 0 of the next segment); an assert checks every join renders identical
characters (and colours to within 1/255 of float noise). One shared 256-colour palette (median cut, no dither) is built from frames sampled across all segments for the GIF.
Segments: variants/{twirl,waveskip,lookback}.py supply timelines (lookback_timeline(from_front=True) is the bridged version) and their own
`mw` module for rendering (waveskip patches an in-memory copy of make_walk.py). The standalone walk.gif is still made by make_walk.py.
`cache_dir` (optional) keeps the rendered RGB frames as .npy so format experiments don't re-render.
"""
import argparse, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "variants"))
import numpy as np
from PIL import Image
import make_walk as mw0, waveskip, twirl, lookback

FULL = [("classic", mw0.CLASSIC, mw0), ("waveskip", waveskip.tl, waveskip.mw), ("twirl", twirl.tl, twirl.mw),
        ("lookback", lookback.lookback_timeline(from_front=True), mw0)]
SHORT = [("classic", mw0.CLASSIC, mw0), ("waveskip", waveskip.tl, waveskip.mw), ("twirl", twirl.tl, twirl.mw)]
SEGMENTS = FULL

def check_joins():
    for i, (n, tl, m) in enumerate(SEGMENTS):
        m.check_loop(tl) if n != "lookback" else mw0.check_loop(tl)
        n2, tl2, m2 = SEGMENTS[(i + 1) % len(SEGMENTS)]
        (ca, ka), (cb, kb) = m.frame(tl, tl.nf), m2.frame(tl2, 0)
        assert (ca == cb).all() and np.abs(ka - kb).max() <= 1, f"join {n} -> {n2} differs"
        print("join ok:", n, "->", n2)

def render_all(cache=None):
    frames, ranges = [], []
    for n, tl, m in SEGMENTS:
        ranges.append((n, len(frames), len(frames) + tl.nf))
        for f in range(tl.nf):
            p = os.path.join(cache, f"{len(frames):04d}.npy") if cache else None
            if p and os.path.exists(p): frames.append(np.load(p)); continue
            a = np.asarray(m.render(tl, f))
            if p: np.save(p, a)
            frames.append(a)
    return frames, ranges

def gif_frames(frames, ranges):
    pick = [a + int((b - a) * x) for _, a, b in ranges for x in (0, .2, .4, .65)]
    w, h = frames[0].shape[1], frames[0].shape[0]
    sheet = Image.new("RGB", (w * 4, h * 4))
    for i, k in enumerate(pick): sheet.paste(Image.fromarray(frames[k]), (i % 4 * w, i // 4 * h))
    pal = sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    return [Image.fromarray(a).quantize(palette=pal, dither=Image.Dither.NONE) for a in frames]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", action="store_true")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "walk_cycle.gif"))
    ap.add_argument("--cache")
    args = ap.parse_args()
    SEGMENTS = SHORT if args.short else FULL
    if args.cache: os.makedirs(args.cache, exist_ok=True)
    check_joins()
    frames, ranges = render_all(args.cache)
    print(ranges, len(frames), "frames")
    g = gif_frames(frames, ranges)[::args.step]
    g[0].save(args.out, save_all=True, append_images=g[1:], duration=mw0.FRAME_MS * args.step, loop=0, optimize=True)
    print(args.out, os.path.getsize(args.out), len(g), "frames")
