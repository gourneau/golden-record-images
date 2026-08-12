"""Parse Ron Barry's voyager.cpp and emit its three catalog tables as JSON.

Barry hand-tuned these in Audacity in 2017 and they remain the only published
index of where each of the 156 frames begins in the 384 kHz master. We use them
as *seeds* -- pipeline/sync.py re-detects every start from the signal itself --
and as the authoritative source for colour role and rotation.

Source: https://github.com/foodini/voyager/blob/master/voyager.cpp
"""

import json
import re
import sys
from pathlib import Path

SAMPLE_RATE = 384_000
COLOR_NAMES = {"red": "r", "grn": "g", "blu": "b", "bnw": "mono"}


def _block(src: str, decl: str) -> str:
    """Return the text between the outermost braces of a `decl = { ... };`."""
    i = src.index(decl)
    i = src.index("{", i)
    depth, j = 0, i
    while True:
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[i + 1 : j]
        j += 1


def _rows(block: str) -> list[str]:
    """Split a two-channel initialiser into its two inner brace groups."""
    out, depth, start = [], 0, None
    for j, ch in enumerate(block):
        if ch == "{":
            if depth == 0:
                start = j + 1
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                out.append(block[start:j])
    return out


def _strip_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    return re.sub(r"//[^\n]*", "", s)


def parse(cpp_path: Path) -> dict:
    src = _strip_comments(cpp_path.read_text())

    starts = []
    for row in _rows(_block(src, "start_points[2][num_images_per_channel]")):
        vals = []
        for expr in (e.strip() for e in row.split(",")):
            if not expr:
                continue
            # Each entry looks like: 27*audio_sample_rate + 296562 + 3151
            expr = expr.replace("audio_sample_rate", str(SAMPLE_RATE))
            if not re.fullmatch(r"[\d\s+\-*]+", expr):
                raise ValueError(f"unexpected start_points expression: {expr!r}")
            vals.append(int(eval(expr)))  # noqa: S307 - input is digit/operator only
        starts.append(vals)

    colors = []
    for row in _rows(_block(src, "color[2][num_images_per_channel]")):
        colors.append([COLOR_NAMES[t] for t in re.findall(r"\b(red|grn|blu|bnw)\b", row)])

    orient = []
    for row in _rows(_block(src, "orientation[2][num_images_per_channel]")):
        orient.append([int(t) for t in re.findall(r"\d+", row)])

    for name, tbl in (("start_points", starts), ("color", colors), ("orientation", orient)):
        if len(tbl) != 2 or any(len(r) != 78 for r in tbl):
            raise ValueError(f"{name}: expected 2x78, got {[len(r) for r in tbl]}")

    return {
        "source": "https://github.com/foodini/voyager/blob/master/voyager.cpp",
        "sample_rate": SAMPLE_RATE,
        "frames_per_channel": 78,
        # Index 0 == left audio channel, 1 == right, matching Barry's decode() calls.
        "start_samples": starts,
        "color_role": colors,
        "orientation": orient,
    }


if __name__ == "__main__":
    cpp = Path(sys.argv[1])
    tables = parse(cpp)
    out = Path(__file__).parent / "barry_tables.json"
    out.write_text(json.dumps(tables, indent=1))

    n_color = sum(r.count("r") for r in tables["color_role"])
    n_mono = sum(r.count("mono") for r in tables["color_role"])
    print(f"wrote {out}")
    print(f"  156 frames -> {n_mono} mono + {n_color} RGB triplets = {n_mono + n_color} images")
