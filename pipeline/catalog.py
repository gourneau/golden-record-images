"""The frame and image catalog, derived from Ron Barry's 2017 tables.

The record holds 156 frames, 78 per audio channel. Most are standalone
black-and-white pictures; the rest come in runs of three consecutive frames on
the same channel that are the red, green and blue separations of one colour
photograph. 96 mono + 20 triplets = 116 images, and that arithmetic is the
correctness check on this whole module: if the grouping logic drifts, the
counts stop landing on 156/116 and `build()` raises.

Barry's start samples are only *seeds* here -- pipeline/build.py re-detects
every frame start from the signal -- but his colour roles and orientations are
the authoritative source, since neither can be recovered from the audio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

TABLES_PATH = Path(__file__).with_name("barry_tables.json")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FRAME_MAP_PATH = DATA_DIR / "frame_map.json"
IMAGES_PATH = DATA_DIR / "images.json"

CHANNEL_LETTERS = ("L", "R")
# The order of the three separations within a colour triplet, as they occur on
# the record. Settled empirically -- see pipeline/colour.py for the measurement
# (all 20 triplets correlated against the canonical NAIC source images; 20/20
# say first frame = red). Barry's r,g,b tables are right; the b,g,r order
# asserted in amazing-rando's README is wrong.
RGB_ORDER = ("r", "g", "b")
EXPECTED_FRAMES = 156
EXPECTED_IMAGES = 116
EXPECTED_MONO = 96
EXPECTED_TRIPLETS = 20


@dataclass(frozen=True)
class Frame:
    """One scanned frame: 512 traces of one channel of the master."""

    id: str  # channel letter + zero-padded index, e.g. "L000"
    channel: int  # 0 = left, 1 = right
    index: int  # position within the channel, 0..77
    seed_sample: int  # Barry's hand-tuned start, in 384 kHz samples
    color: str  # "mono" | "r" | "g" | "b"
    triplet_id: str | None  # "T01".."T20" when color != "mono"
    orientation: int  # quarter turns, from Barry's table
    image_number: int | None = None  # canonical 1..116, from data/frame_map.json
    title: str | None = None  # canonical title, from data/frame_map.json


@dataclass(frozen=True)
class Image:
    """One picture: either a single mono frame or an r/g/b triplet."""

    frames: tuple[str, ...]
    color: bool
    n: int | None = None  # canonical 1..116
    title: str | None = None


@dataclass(frozen=True)
class Catalog:
    frames: tuple[Frame, ...]
    images: tuple[Image, ...]

    def by_id(self, frame_id: str) -> Frame:
        for f in self.frames:
            if f.id == frame_id:
                return f
        raise KeyError(frame_id)

    def ids(self) -> list[str]:
        return [f.id for f in self.frames]


def load_tables(path: str | Path = TABLES_PATH) -> dict:
    tables = json.loads(Path(path).read_text())
    n = tables["frames_per_channel"]
    for key in ("start_samples", "color_role", "orientation"):
        rows = tables[key]
        if len(rows) != len(CHANNEL_LETTERS) or any(len(r) != n for r in rows):
            raise ValueError(f"{key}: expected {len(CHANNEL_LETTERS)}x{n}, got {[len(r) for r in rows]}")
    return tables


def _identify(frames: list[Frame], images: list[Image]) -> tuple[list[Frame], list[Image]]:
    """Attach canonical imageNumber and title from data/frame_map.json.

    The frame->image mapping was established by the prior-art sweep (left
    channel carries images 1-54 in order, right channel 55-116) and verified
    visually at 30+ anchor points. Here it is cross-checked against the
    structural catalog; any disagreement means one of the two is corrupt, so
    every check raises rather than warns.
    """
    if not FRAME_MAP_PATH.exists() or not IMAGES_PATH.exists():
        raise FileNotFoundError(
            f"identification tables missing: need {FRAME_MAP_PATH} and {IMAGES_PATH}"
        )
    frame_map = {e["frame"]: e for e in json.loads(FRAME_MAP_PATH.read_text())}
    canon = {e["n"]: e for e in json.loads(IMAGES_PATH.read_text())}

    if len(frame_map) != EXPECTED_FRAMES:
        raise ValueError(f"frame_map.json has {len(frame_map)} frames, expected {EXPECTED_FRAMES}")
    if len(canon) != EXPECTED_IMAGES or set(canon) != set(range(1, EXPECTED_IMAGES + 1)):
        raise ValueError(f"images.json must cover n=1..{EXPECTED_IMAGES} exactly")

    out_frames: list[Frame] = []
    for f in frames:
        e = frame_map.get(f.id)
        if e is None:
            raise ValueError(f"{f.id}: missing from frame_map.json")
        if e["role"] != f.color:
            raise ValueError(f"{f.id}: frame_map role {e['role']!r} != catalog color {f.color!r}")
        n = int(e["imageNumber"])
        title = canon[n]["title"]
        if e.get("title") != title:
            raise ValueError(f"{f.id}: frame_map title {e['title']!r} != images.json {title!r}")
        out_frames.append(replace(f, image_number=n, title=title))

    by_id = {f.id: f for f in out_frames}
    out_images: list[Image] = []
    seen: set[int] = set()
    for im in images:
        ns = {by_id[fid].image_number for fid in im.frames}
        if len(ns) != 1:
            raise ValueError(f"image frames {im.frames} map to several imageNumbers {sorted(ns)}")
        n = ns.pop()
        if n in seen:
            raise ValueError(f"imageNumber {n} claimed by more than one image")
        seen.add(n)
        if im.color and not canon[n]["color"]:
            raise ValueError(
                f"image {n} ({canon[n]['title']!r}) is an r/g/b triplet on the record "
                f"but images.json marks it monochrome"
            )
        out_images.append(replace(im, n=n, title=canon[n]["title"]))
    if seen != set(range(1, EXPECTED_IMAGES + 1)):
        raise ValueError(f"images do not cover 1..{EXPECTED_IMAGES}: missing {sorted(set(range(1, 117)) - seen)}")

    n_colour_canon = sum(1 for e in canon.values() if e["color"])
    if n_colour_canon != EXPECTED_TRIPLETS:
        raise ValueError(f"images.json marks {n_colour_canon} colour images, expected {EXPECTED_TRIPLETS}")
    return out_frames, out_images


def build(tables: dict | None = None) -> Catalog:
    """Assemble the catalog, or raise if the 156/116 invariant fails."""
    tables = tables if tables is not None else load_tables()
    n_per_channel = tables["frames_per_channel"]

    frames: list[Frame] = []
    images: list[Image] = []
    triplet_n = 0

    for ch, letter in enumerate(CHANNEL_LETTERS):
        starts = tables["start_samples"][ch]
        roles = tables["color_role"][ch]
        orients = tables["orientation"][ch]

        # A triplet is a run of three consecutive non-mono frames. Runs longer
        # than three occur (up to nine on the left channel) and simply chunk
        # into consecutive triplets; anything that does not chunk evenly, or
        # whose roles are not r,g,b in order, means the table has been misread.
        pending: list[tuple[int, str]] = []

        for idx in range(n_per_channel):
            role = roles[idx]
            fid = f"{letter}{idx:03d}"
            record = {
                "id": fid,
                "channel": ch,
                "index": idx,
                "seed_sample": int(starts[idx]),
                "color": role,
                "triplet_id": None,
                "orientation": int(orients[idx]),
            }
            frames.append(Frame(**record))

            if role == "mono":
                if pending:
                    raise ValueError(f"{fid}: mono frame interrupts a partial triplet {pending}")
                images.append(Image(frames=(fid,), color=False))
                continue

            pending.append((len(frames) - 1, role))
            if len(pending) == 3:
                got = tuple(r for _, r in pending)
                if got != RGB_ORDER:
                    ids = [frames[i].id for i, _ in pending]
                    raise ValueError(f"triplet {ids} has roles {got}, expected {RGB_ORDER}")
                triplet_n += 1
                tid = f"T{triplet_n:02d}"
                member_ids = []
                for pos, _ in pending:
                    frames[pos] = replace(frames[pos], triplet_id=tid)
                    member_ids.append(frames[pos].id)
                images.append(Image(frames=tuple(member_ids), color=True))
                pending = []

        if pending:
            raise ValueError(f"channel {letter}: dangling colour frames {pending} at end of channel")

    n_mono = sum(1 for im in images if not im.color)
    if len(frames) != EXPECTED_FRAMES:
        raise ValueError(f"expected {EXPECTED_FRAMES} frames, built {len(frames)}")
    if len(images) != EXPECTED_IMAGES:
        raise ValueError(f"expected {EXPECTED_IMAGES} images, built {len(images)}")
    if n_mono != EXPECTED_MONO or triplet_n != EXPECTED_TRIPLETS:
        raise ValueError(
            f"expected {EXPECTED_MONO} mono + {EXPECTED_TRIPLETS} triplets, "
            f"got {n_mono} + {triplet_n}"
        )

    frames, images = _identify(frames, images)
    return Catalog(frames=tuple(frames), images=tuple(images))


def catalog_json(
    catalog: Catalog,
    *,
    sample_rate: int,
    nominal_period: float,
    traces_per_frame: int,
    frame_meta: dict[str, dict],
) -> dict:
    """Render the catalog to the web data contract.

    `frame_meta` supplies the per-frame numbers that only the asset build knows
    (file, leadIn, startSample, periodGuess, durationSamples, peak); frames
    absent from it are omitted, so a partial build produces a catalog that
    describes exactly the assets on disk. imageNumber and title come from the
    identification tables (data/frame_map.json + data/images.json).
    """
    out_frames = []
    for f in catalog.frames:
        meta = frame_meta.get(f.id)
        if meta is None:
            continue
        out_frames.append(
            {
                "id": f.id,
                "channel": f.channel,
                "index": f.index,
                "file": meta["file"],
                "leadIn": meta["leadIn"],
                "startSample": meta["startSample"],
                "periodGuess": meta["periodGuess"],
                "color": f.color,
                "tripletId": f.triplet_id,
                "orientation": f.orientation,
                "durationSamples": meta["durationSamples"],
                "peak": meta["peak"],
                "imageNumber": f.image_number,
                "title": f.title,
            }
        )

    present = {f["id"] for f in out_frames}
    out_images = [
        {"n": im.n, "title": im.title, "frames": list(im.frames), "color": im.color}
        for im in catalog.images
        if all(fid in present for fid in im.frames)
    ]

    return {
        "version": 1,
        "sampleRate": sample_rate,
        "nominalPeriod": nominal_period,
        "tracesPerFrame": traces_per_frame,
        "frames": out_frames,
        "images": out_images,
    }


if __name__ == "__main__":
    cat = build()
    n_color = sum(1 for im in cat.images if im.color)
    print(f"{len(cat.frames)} frames -> {len(cat.images) - n_color} mono + {n_color} colour images")
    for im in cat.images[:6]:
        print("  ", im.frames, "colour" if im.color else "mono")
