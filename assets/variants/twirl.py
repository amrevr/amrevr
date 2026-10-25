"""twirl variant: idle, arms open, two accelerating spins (yaw unwrapped), settle with her back to us, walk 4 strides, turn back to the front."""
import os, sys; sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import make_walk as mw
from dataclasses import replace
from pathlib import Path
import numpy as np

SPIN0 = 6                                   # first spin frame; yaw 180 until then
ACC, DEC = [10, 22, 34, 44, 52, 58], [54, 44, 32, 20, 10, 4]
NC = 8                                      # cruise frames
cruise = (900 - sum(ACC) - sum(DEC)) / NC   # 2 turns + a half = 900 deg exactly
speeds = ACC + [cruise] * NC + DEC
yaw = 180 - np.concatenate([[0], np.cumsum(speeds)])   # unwrapped, decreasing like the classic spin away
SPIN_END = SPIN0 + len(speeds)              # frame she faces away (yaw -720)
assert abs(yaw[-1] + 720) < 1e-6
T = {}
for i in range(4): T[i] = (180, 0, 0, 0, 0)
T[4] = (180, 0.25, 0, 0.4, 0.4); T[5] = (180, 0.15, 0, 0.9, 0.9)
for i, y in enumerate(yaw): T[SPIN0 + i] = (y, 0, 0, 1, 1)
for k, a in enumerate([0.8, 0.45, 0.2, 0.05, 0]): T[SPIN_END + 1 + k] = (-720, 0, 0, a, a)
WALK_FROM = SPIN_END + 4
WIND_FROM, STOP_AT = WALK_FROM + 41, WALK_FROM + 49
RET = STOP_AT + 1
NF = RET + 10
for k, v in mw.SPIN_BACK.items(): T[RET + k] = (v[0] - 720,) + v[1:]
keys = mw.Keys(T)
spr = mw.springs(NF, mw.head_drive(keys, None, None))
gt = mw.gait(NF, WALK_FROM, WIND_FROM, STOP_AT, 4)
base = mw.make_pose_fn(NF, keys, spr, gt, None, 4)

def pose(f):
    p = base(f)
    # centrifugal streaming: hair, ribbons and hem follow the spin speed (deg/frame), which the yaw-acceleration springs alone would drop at steady speed
    v = min(abs(keys(f + .5)[0] - keys(f - .5)[0]) / 60, 1) ** 1.3
    if v < 0.05: return p
    lift = mw.smooth((v - .5) / .5)
    return replace(p, whip=float(np.clip(p.whip + 1.4 * v, -1.45, 1.45)), tail_x=p.tail_x - 0.22 * v, flare=1.0 if v > .6 else max(p.flare, v * 1.4),
                   fan=max(p.fan, v), bob=0.012 * lift)

tl = mw.Timeline(NF, pose)
if __name__ == "__main__":
    out = mw.render_gif(tl, os.path.join(os.path.dirname(os.path.abspath(__file__)), "walk_twirl.gif"))
    print(out, out.stat().st_size, "bytes", NF, "frames", "spin", SPIN0, SPIN_END, "walk", WALK_FROM, "ret", RET)
