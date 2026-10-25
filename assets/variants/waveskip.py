"""Variant "wave and skip": front idle, two-handed goodbye wave, cheery turn, skip away (4 strides), wind down, turn back to the front idle.
The shared generator is imported read-only; the few extra Pose fields (wave2, lift, kl) are patched into an in-memory copy of its source."""
import os, sys, types, dataclasses
sys.dont_write_bytecode = True
import numpy as np
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "make_walk.py")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "walk_waveskip.gif")

def load():
    s = open(SRC).read()
    def sub(a, b):
        global_n = s.count(a); assert global_n == 1, (a, global_n)
        return s.replace(a, b)
    s = sub("    bounce: float = 0.0    # vertical lag", "    wave2: float = 0.0     # 1: the left hand waves too\n    lift: float = 0.0      # whole figure off the ground, fractions of H (hop)\n    kl: float = 0.0        # skip: extra knee lift and arm swing 0..1\n    bounce: float = 0.0    # vertical lag")
    s = sub("wave = pz.wave if side < 0 else 0.0", "wave = pz.wave if (side < 0 or pz.wave2 > 0) else 0.0")
    s = sub("np.array([shx - 0.2 * H + 0.035 * H * np.sin(pz.wph), yy(0.115 - 0.012 * np.cos(pz.wph))])",
            "np.array([shx + side * 0.2 * H + 0.05 * H * np.sin(pz.wph + (0 if side < 0 else 0.9)), yy(0.095 - 0.02 * np.cos(pz.wph + (0 if side < 0 else 0.9)))])")
    s = sub("    ARM_SWING, HAIR_SWAY", "    ARM_SWING0, HAIR_SWAY")
    s = sub("    HAIR_LAG, TAIL_LAG", "    ARM_SWING = np.radians(15 + 13 * pz.kl) * A\n    HAIR_LAG, TAIL_LAG")
    s = sub("th = np.radians(19) * A * np.cos(ph) + np.radians(2) + np.radians(42) * pz.squat",
            "th = np.radians(19) * A * np.cos(ph) + np.radians(2) + np.radians(42) * pz.squat + np.radians(32) * pz.kl * vm(ph, 5.4, 2.0)")
    s = sub("pj = lambda lx, z, h, c, f=None: (c + lx * H, G0 - H * h", "pj = lambda lx, z, h, c, f=None: (c + lx * H, G0 - pz.lift * H - H * h")
    s = sub("top = G0 - H + dy", "top = G0 - H - pz.lift * H + dy")
    m = types.ModuleType("make_walk_ws"); m.__file__ = SRC
    sys.modules["make_walk_ws"] = m
    exec(compile(s, SRC, "exec"), m.__dict__)
    return m
mw = load()

# ---- timeline ----
IDLE, W0, W1 = 4, 4, 19          # idle 0..3, wave 4..19
TURN = 19                        # turn keys 19..24
WALK_FROM, WIND_FROM, STOP_AT = 25, 66, 72
RET = 72
NF = RET + 10                    # 82
HOP, SKIPK, AMPX = 0.05, 1.0, 1.35

AWAY = {0: (180, 0, 0, 0, 0), 18: (180, 0, 0, 0, 0), 19: (180, 0.5, 0, 0, 0), 20: (160, 0.9, 0, 0.2, 0.2), 21: (80, 0.1, 0, 0.9, 0.9), 22: (10, 0, 0, 1, 1), 23: (-8, 0, 0, 0.7, 0.7),
        24: (0, 0, 0, 0.3, 0.3), 25: (0, 0, 0, 0.1, 0.1), 26: (0, 0, 0, 0, 0)}
keys = mw.Keys({**AWAY, **{RET + k: v for k, v in mw.SPIN_BACK.items()}})
gt = mw.gait(NF, WALK_FROM, WIND_FROM, STOP_AT, 4)
fl, amp, trav = gt
ampn = np.minimum(amp, 1.0)
def lift_of(f):
    return HOP * ampn[f] * abs(np.sin(fl[f])) ** 0.8 if f <= NF else 0.0
LIFT = np.array([lift_of(f) for f in range(NF + 1)])

def wave_u(f):
    u = (f - W0) / (W1 - W0)
    if not 0 < u < 1: return 0.0, 0.0, 0.0, 0.0
    env = mw.smooth(u / 0.18) * mw.smooth((1 - u) / 0.18)
    ph = 2 * np.pi * 2.5 * ((u - 0.18) / 0.64) if 0.18 < u < 0.82 else (0.0 if u <= 0.18 else 2 * np.pi * 2.5)
    # each beat (cycle) the body dips once and sways
    dip = 0.3 * env * (0.5 - 0.5 * np.cos(ph * 2)) if 0.18 < u < 0.82 else 0.0
    return env, ph, 0.014 * env * np.sin(ph), dip
def wdrive(f):
    return wave_u(f)[2] * 8 + 0.0, wave_u(f)[3]
def drive(f):
    yaw = np.radians(keys(f + mw.HEAD_LEAD)[0]) * (mw.RET_GAIN if f >= RET - 1 else 1)
    w = wave_u(f)
    fi = int(np.floor(f)); a = f - fi
    lift = (1 - a) * LIFT[min(fi, NF)] + a * LIFT[min(fi + 1, NF)]
    return np.array([yaw + mw.SWAY_COUPLE * 0.5 * w[2], keys(f)[1] + w[3] - lift / 0.03])
spr = mw.springs(NF, drive)
base = mw.make_pose_fn(NF, keys, spr, (fl, amp * AMPX, trav), None, IDLE)
def pose_fn(f):
    p = base(f)
    env, ph, sway, dip = wave_u(f)
    kl = SKIPK * ampn[f]
    skf = min(abs(spr[f][2]), 1)
    return dataclasses.replace(p, wave=env, wave2=1.0 if env > 0 else 0.0, wph=ph, sway=sway, squat=p.squat + dip, kl=kl, lift=LIFT[f],
                               flare=min(p.flare + 6 * LIFT[f], 1))
tl = mw.Timeline(NF, pose_fn)

if __name__ == "__main__":
    out = mw.render_gif(tl, OUT)
    print(out, out.stat().st_size, "bytes", tl.nf, "frames")
