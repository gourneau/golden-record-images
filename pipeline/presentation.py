"""Merge the tier-2 presentation metadata back onto a freshly built catalogue.

`build.py` derives `catalog.json` from the audio, which is correct -- it is a
tier 0 product and it must not know anything else. But which way up an image is
shown, what it is called, and the evidence for each rotation are TIER 2: Earth
knowledge, added afterwards from the published slides.

So a rebuild silently drops them, and sixty portrait images come out on their
side. That is not hypothetical -- it happened, and the only reason it was caught
is that the catalogue was inspected immediately afterwards rather than trusted.

The fix is to keep the two tiers in two files. `docs/presentation.json` is the
source of truth for the tier 2 fields; this merges them in. Run it after every
build, and the tier boundary that provenance.py enforces inside the pipeline is
enforced on disk as well.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CATALOG = REPO / "web" / "public" / "data" / "catalog.json"
PRESENTATION = REPO / "docs" / "presentation.json"


def merge(catalog: Path = CATALOG, presentation: Path = PRESENTATION) -> dict:
    cat = json.loads(catalog.read_text())
    pres = json.loads(presentation.read_text())["images"]
    applied = missing = 0
    for im in cat["images"]:
        row = pres.get(str(im["n"]))
        if not row:
            missing += 1
            continue
        for k, v in row.items():
            im[k] = v
            applied += 1
    catalog.write_text(json.dumps(cat, indent=1))
    return {"fields_applied": applied, "images_without_metadata": missing,
            "rotated": [i["n"] for i in cat["images"] if i.get("displayRotate")]}


if __name__ == "__main__":  # pragma: no cover
    r = merge()
    print(f"{r['fields_applied']} presentation fields applied to the catalogue")
    print(f"rotations restored: {r['rotated']}")
    if r["images_without_metadata"]:
        print(f"  {r['images_without_metadata']} images have no presentation metadata")
