"""Generates walk.svg: an ASCII walk-cycle loop for the profile README. Run: python3 make_walk.py"""
from html import escape
from pathlib import Path

HEAD = [
    r"     .~~~~~.     ",
    r"    /~~~~~~~\    ",
    r"   |~ (o o) ~|   ",
    r"   |~  \-/  ~|   ",
]
# hair swings opposite the stride, arms swing, legs alternate
BODIES = [
    [r"   ~\ `---' /~   ",
     r"   ~ /|' '|\    ",
     r"    / |   | \   ",
     r"   *  /___\  \  ",
     r"     /     \  * ",
     r"    /_______\   ",
     r"      /   |     ",
     r"     /    |     ",
     r"   _/     |_    "],
    [r"   ~\ `---' /~   ",
     r"   ~ /|' '|\ ~  ",
     r"    | |   | |   ",
     r"    * /___\ *   ",
     r"     /     \    ",
     r"    /_______\   ",
     r"       | |      ",
     r"       | |      ",
     r"      _| |_     "],
    [r"   ~\ `---' /~   ",
     r"    /|' '|\ ~   ",
     r"   / |   | \    ",
     r"  /  /___\  *   ",
     r" *  /     \     ",
     r"   /_______\    ",
     r"     |   \      ",
     r"     |    \     ",
     r"    _|     \_   "],
    [r"   ~\ `---' /~   ",
     r"  ~ /|' '|\ ~   ",
     r"   | |   | |    ",
     r"   * /___\ *    ",
     r"    /     \     ",
     r"   /_______\    ",
     r"      | |       ",
     r"      | |       ",
     r"     _| |_      "],
]
FRAMES = [HEAD + b for b in BODIES]
assert len({len(f) for f in FRAMES}) == 1, "frames must be the same height"

CH_W, LINE_H, PAD = 9.6, 18, 20
W, H = 360, PAD * 2 + LINE_H * (len(FRAMES[0]) + 3)
CYCLE = 0.9  # seconds per full walk cycle
n = len(FRAMES)

def frame(i, lines):
    x = (W - CH_W * 17) / 2
    rows = "".join(f'<tspan x="{x:.1f}" dy="{LINE_H}">{escape(l)}</tspan>' for l in lines)
    return f'<text class="f" style="animation-delay:-{CYCLE - i * CYCLE / n:.3f}s" y="{PAD}">{rows}</text>'

ground_y = PAD + LINE_H * (len(FRAMES[0]) + 1)
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="ASCII girl walking toward the camera, looping">
<style>
text{{font:16px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#f4c2d7;white-space:pre}}
.f{{opacity:0;animation:show {CYCLE}s steps(1) infinite}}
@keyframes show{{0%{{opacity:1}}{100 / n:.2f}%,100%{{opacity:0}}}}
.bob{{animation:bob {CYCLE / 2}s ease-in-out infinite alternate}}
@keyframes bob{{to{{transform:translateY(3px)}}}}
.road{{fill:#6b5b73;animation:road {CYCLE / 2}s linear infinite}}
@keyframes road{{to{{transform:translateX(-{CH_W * 4:.1f}px)}}}}
.cap{{fill:#b9a3c4;font-size:13px}}
</style>
<rect width="100%" height="100%" rx="12" fill="#17121c"/>
<g class="bob">{"".join(frame(i, f) for i, f in enumerate(FRAMES))}</g>
<text class="road" x="0" y="{ground_y}">{"=   " * 12}</text>
<text class="cap" x="50%" y="{H - PAD}" text-anchor="middle">&#9834; now playing: cry &#8212; aisin-gioro &amp; toyoki</text>
</svg>
'''
Path(__file__).with_name("walk.svg").write_text(svg)
