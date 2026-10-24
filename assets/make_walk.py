"""Generates walk.svg: an ASCII back-view walk down a night road for the profile README. Run: python3 make_walk.py"""
from html import escape
from pathlib import Path

COLS, ROWS, MID, HORIZON = 48, 23, 24, 2
N, CYCLE = 8, 1.1  # frames per stride cycle, seconds per cycle
SWAY = [0, 1, 2, 1, 0, -1, -2, -1]  # hair/hip sway, opposite to the stepping foot

HEAD = [
    ("    ._    _.    ", "bow"),
    ("   ( >-()-< )   ", "bow"),
    ("    '-'  '-'    ", "bow"),
    ("    .-" + "'" * 4 + "-.    ", "hair"),
    ("   " + "/" * 5 + "\\" * 5 + "   ", "hair"),
    ("  " + "|" * 12 + "  ", "hair"),
]

def girl(f):
    """Rows of (text, part) for frame f; 16 columns wide."""
    s = SWAY[f]
    left_step = f < N // 2
    rows = list(HEAD)
    for i in range(7):  # hair hangs past the waist; tips swing furthest
        off = round(s * i / 4)
        w = 10 - max(0, i - 4) * 2
        cells = [" "] * 16
        for j, ch in enumerate("(" + "|" * w + ")"):
            cells[2 + (10 - w) // 2 + off + j] = ch
        # arms in their own columns: the forward-swinging arm is shorter
        if i < 3:
            if i < (2 if left_step else 3):
                cells[0] = "|"
            if i < (3 if left_step else 2):
                cells[15] = "|"
        elif i == 3:
            cells[15 if left_step else 0] = "'"
        rows.append(("".join(cells), "hair"))
    rows += [
        ("  .~~~~~~~~~~.  ", "dress"),
        (" (  ~  ~  ~  ~ )", "dress"),
        (["(_,_,_,_,_,_,_,)", "(,_,_,_,_,_,_,_)"][f % 2], "dress"),
        ("     |    |     ", "legs"),
        ("     |    |     ", "legs"),
        ("    [_]   u     " if left_step else "     u   [_]    ", "legs"),
    ]
    hip = (s > 0) - (s < 0)  # whole body rocks with the hips
    return [(" " * (1 + hip) + t, p) for t, p in rows], ROWS - len(rows)

def frame(f):
    grid = [[(" ", "bg") for _ in range(COLS)] for _ in range(ROWS)]
    phase = f // 2  # road scrolls toward the viewer
    for c, ch in enumerate(("_.-^-._,.-^^-.,_" * 4)[:COLS]):
        grid[HORIZON][c] = (ch, "grass")
    for r in range(HORIZON + 1, ROWS):
        d = round((r - HORIZON) * 1.1)
        for c, ch in ((MID - d, "/"), (MID + d, "\\")):
            if 0 <= c < COLS:
                grid[r][c] = (ch, "road")
        if (r - phase) % 4 < 2:
            grid[r][MID] = ("|", "line")
        if (r - phase) % 4 == 0:
            for c in (MID - d - 3, MID + d + 3):
                if 0 <= c < COLS:
                    grid[r][c] = (",", "grass")
    rows, top = girl(f)
    x0 = MID - 4  # walks just right of the centre line
    for i, (text, part) in enumerate(rows):
        for c in range(x0, x0 + len(text)):  # no road markings peeking through gaps in her outline
            if grid[top + i][c][1] in ("line", "grass"):
                grid[top + i][c] = (" ", "bg")
        first, last = len(text) - len(text.lstrip()), len(text.rstrip())
        for j in range(first, last):  # her silhouette hides the road behind her
            if 0 <= x0 + j < COLS:
                grid[top + i][x0 + j] = (text[j], part)
    return grid

CH_W, LINE_H, PAD = 9.6, 16, 18
W, H = COLS * CH_W + PAD * 2, ROWS * LINE_H + PAD * 2 + 26

def svg_frame(f, grid):
    lines = []
    for r, row in enumerate(grid):
        runs, cur, buf = [], None, ""
        for ch, part in row + [("", None)]:
            if part != cur and buf:
                runs.append(f'<tspan class="{cur}">{escape(buf)}</tspan>')
                buf = ""
            cur, buf = part, buf + ch
        lines.append(f'<tspan x="{PAD}" y="{PAD + (r + 1) * LINE_H}">{"".join(runs)}</tspan>')
    return f'<text class="f" style="animation-delay:-{CYCLE - f * CYCLE / N:.3f}s">{"".join(lines)}</text>'

FRAMES = [frame(f) for f in range(N)]
assert all(len(g) == ROWS and all(len(r) == COLS for r in g) for g in FRAMES)

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" viewBox="0 0 {W:.0f} {H:.0f}" role="img" aria-label="ASCII girl with long hair and a bow walking away down a night road, looping">
<style>
text{{font:16px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre}}
.f{{opacity:0;animation:show {CYCLE}s steps(1) infinite}}
@keyframes show{{0%{{opacity:1}}{100 / N:.2f}%,100%{{opacity:0}}}}
.bg{{fill:none}} .road{{fill:#5d5470}} .line{{fill:#e9d48a}} .grass{{fill:#4f7a5c}}
.bow{{fill:#ff7aa8}} .hair{{fill:#cdb8ea}} .dress{{fill:#f6d3e2}} .legs{{fill:#bfaecb}}
.cap{{fill:#b9a3c4;font-size:13px}}
</style>
<rect width="100%" height="100%" rx="12" fill="#120e18"/>
{"".join(svg_frame(f, g) for f, g in enumerate(FRAMES))}
<text class="cap" x="50%" y="{H - PAD}" text-anchor="middle">&#9834; now playing: cry &#8212; aisin-gioro &amp; toyoki</text>
</svg>
'''
if __name__ == "__main__":
    Path(__file__).with_name("walk.svg").write_text(svg)
    for f in (0, 2, 4, 6):
        print(f"--- frame {f}")
        print("\n".join("".join(ch for ch, _ in r) for r in FRAMES[f]))
