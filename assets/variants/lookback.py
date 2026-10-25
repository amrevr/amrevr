"""look-back variant: walk, ease to a stop, glance back over the left shoulder (head leads, hair swings over and settles), glance front again, walk on."""
import os, sys; sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import make_walk as mw
import numpy as np
S = mw.smooth
NF, STRIDES = 72, 4
# timeline: 0-18 walk | 18-28 ease to a stop | 28-33 glance out | 33-37 hold | 37-42 glance back | 41-51 ease into the walk | 51-72 walk
STOP0, STOP1, GO0, GO1 = 18, 28, 41, 51
LOOK0, HOLD, BACK0, GLANCE = 28, 33, 37, 5
YAW_HIP, YAW_SH, YAW_HEAD = 22, 65, 105   # degrees at the peak; the shoulders stay under 90 so the torso is always the back drawing
def walkf(f): return S((STOP1 - f) / (STOP1 - STOP0)) if f < 35 else S((f - GO0) / (GO1 - GO0))
def look(f, lag=0.0):  # 0 facing away .. 1 full glance: ease out over GLANCE frames, hold, ease back; a larger lag trails the head (shoulders +1, hips +2)
    return S((f - LOOK0 - lag) / GLANCE) * S((BACK0 + GLANCE - f + lag) / GLANCE)
def head_yaw(f): return np.radians(YAW_HEAD) * look(f)
def drive(f): return np.array([head_yaw(f), 0.0])
spr = mw.springs(NF, drive)
f_ = np.arange(NF + 1)
e = np.array([walkf(x) for x in f_])
c = e[:NF] * STRIDES / e[:NF].sum()
stride = np.concatenate([[0], np.cumsum(c)])
fl, amp, travel = 2 * np.pi * stride, e.copy(), stride * mw.STEP_M
ang = lambda d: np.radians(d) % (2 * np.pi)
def glance(u):
    """The look-back part of a pose, at timeline frame u: dict of Pose fields."""
    return dict(yaw_hip=ang(YAW_HIP * look(u, 2)), yaw_sh=ang(YAW_SH * look(u, 1)), yaw_head=ang(YAW_HEAD * look(u)),
                htilt=np.radians(6) * look(u), sway=-0.01 * look(u), peek=bool(look(u) > 0.6))

def pose_fn(f):
    whip, tail, skirt, bounce = spr[f]
    sk = min(abs(skirt), 1)
    return mw.Pose(t=f, loop=NF, fl=fl[f], amp=amp[f], travel=travel[f], near=0.0, far=0.0,
                   whip=whip, tail_x=-tail, flare=min(sk, 1), hem_x=-0.05 * np.sign(skirt) * sk, bounce=bounce, squat=0.0, **glance(f))

def lookback_timeline(from_front=False):
    """from_front=False: the standalone loop (starts and ends mid-walk from behind). from_front=True: the bridged cycle segment, front idle ->
    classic spin away -> ramp into the walk -> the look-back walk above -> wind down -> classic spin back to the front idle."""
    if not from_front:
        return mw.Timeline(NF, pose_fn)
    SH = -21                                    # SPIN_AWAY key 23 (jig ends upright) lands on frame 2: two idle frames first
    WF = mw.WALK_FROM + SH                      # 11
    OFF = 15                                    # frame where the original look-back timeline's frame 0 sits
    END = OFF + NF + 8                              # walk on to here, then wind down
    WIND, STOP = END, END + 8
    RET = STOP + 1
    N = RET + 10
    keys = mw.Keys({**{k + SH: v for k, v in mw.SPIN_AWAY.items()}, **{RET + k: v for k, v in mw.SPIN_BACK.items()}})
    def fdrive(f):
        u = f - OFF
        yaw = np.radians(keys(f + mw.HEAD_LEAD)[0]) * (mw.RET_GAIN if f >= RET - 1 else 1)
        return np.array([yaw + head_yaw(u), keys(f)[1]])
    sp = mw.springs(N, fdrive)
    ff = np.arange(N + 1)
    wl = np.array([walkf(x - OFF) for x in ff])
    wind = S((STOP - ff) / (STOP - WIND))
    e = S((ff - (WF - 2)) / 5) * wl * wind
    amp = mw.walk_amp(ff, WF) * wl * wind
    c = e[:N] * STRIDES_BRIDGED / e[:N].sum()
    stride = np.concatenate([[0], np.cumsum(c)])
    fl_, tr_ = 2 * np.pi * (stride - stride[WF]), stride * mw.STEP_M
    fl_ += mw.CLASSIC.pose_fn(0).fl - fl_[0]    # same rest foot phase as the other front idles (it shows in one cell), so the joins are exact
    base = mw.make_pose_fn(N, keys, sp, (fl_, amp, tr_), None, 4)
    from dataclasses import replace
    def pose(f):
        p, g = base(f), glance(f - OFF)
        return replace(p, **{k: (getattr(p, k) + v) % (2 * np.pi) if k.startswith('yaw') else v for k, v in g.items()})
    return mw.Timeline(N, pose)
STRIDES_BRIDGED = 4

if __name__ == "__main__":
    tl = lookback_timeline()
    out = mw.render_gif(tl, os.path.join(os.path.dirname(os.path.abspath(__file__)), "walk_lookback.gif"))
    print(out, out.stat().st_size, tl.nf)
