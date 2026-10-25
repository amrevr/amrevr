"""look-back variant: walk, ease to a stop, look back over the left shoulder with a hair toss and a brushing hand, turn away, walk on."""
import os, sys; sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import make_walk as mw
import numpy as np
S = mw.smooth
NF, STRIDES = 80, 4
# timeline: 0-20 walk | 20-30 wind down | 30 stop | 30-41 turn to look | 41-46 hold (toss peaks ~38-40) | 46-57 turn back | 57-68 ramp up | 68-80 walk
STOP0, STOP1, GO0, GO1 = 18, 28, 58, 69
LOOK0, LOOK1, BACK0, BACK1 = 28, 42, 45, 56
def walkf(f):
    return S((STOP1 - f) / (STOP1 - STOP0)) + S((f - GO0) / (GO1 - GO0)) - (1 if f > STOP1 else 0) * 0 if False else \
        (S((STOP1 - f) / (STOP1 - STOP0)) if f < 43 else S((f - GO0) / (GO1 - GO0)))
def look(f):  # 0 facing away .. 1 full look, eased in and out, held 41..46
    u = lambda x: 0.5 * x + 0.5 * S(x)  # gentler than a pure ease so the turn reads
    return u(min(max((f - LOOK0) / (LOOK1 - LOOK0), 0), 1)) * S((BACK1 - f) / (BACK1 - BACK0))
def toss(f):  # a quick extra flick of the head, 0..1 bump around frame 38.5
    return np.exp(-((f - 39.5) / 1.6) ** 2)
def head_yaw(f): return np.radians(142) * look(f)
def drive(f):
    return np.array([head_yaw(f) + 0.55 * toss(f), 0.0])
spr = mw.springs(NF, drive)
f_ = np.arange(NF + 1)
e = np.array([walkf(x) for x in f_])
c = e[:NF] * STRIDES / e[:NF].sum()
stride = np.concatenate([[0], np.cumsum(c)])
fl, amp, travel = 2 * np.pi * stride, e.copy(), stride * mw.STEP_M

def pose_fn(f):
    L = look(f); wv = S((f - 37) / 3) * S((45 - f) / 3)  # hand to the hair
    whip, tail, skirt, bounce = spr[f]
    sk = min(abs(skirt), 1)
    ang = lambda d: np.radians(d) % (2 * np.pi)
    return mw.Pose(t=f, loop=NF, fl=fl[f], amp=amp[f], travel=travel[f],
                   yaw_hip=ang(46 * L), yaw_sh=ang(102 * L * S((49 - f) / 5)), yaw_head=ang(head_yaw(f) * 180 / np.pi + 0 * L),
                   htilt=np.radians(9) * L + 0.12 * toss(f) , wave=0.45 * wv, wph=0.6 * np.sin(0.5 * f) * wv, near=0.0, far=0.0,
                   whip=whip, tail_x=-tail, flare=min(sk, 1), hem_x=-0.05 * np.sign(skirt) * sk, bounce=bounce,
                   sway=-0.01 * L, squat=0.0)
if __name__ == "__main__":
    tl = mw.Timeline(NF, pose_fn)
    out = mw.render_gif(tl, os.path.join(os.path.dirname(os.path.abspath(__file__)), "walk_lookback.gif"))
    print(out, out.stat().st_size, tl.nf)
