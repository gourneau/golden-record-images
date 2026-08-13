# Vendored artwork

Not drawn by this project. Both are public-domain NASA material, copied here so the pages do not
hot-link to another repository.

| file | what it is | provenance |
|---|---|---|
| `voyager_golden_plaque.svg` | The engraved cover of the record — the pulsar map, the hydrogen transition, and the picture-decoding instruction this decoder implements | NASA, public domain |
| `voyager_cover_explanation.svg` | The annotated version, explaining what each engraving means | NASA original; revised for historical accuracy by Brian Krent, 28 July 2015. Public domain |

**The transparent background is not a defect and the two files want opposite treatments.**
`voyager_golden_plaque.svg` carries its own gold plate (`#d4af37`) with **white** engraved lines,
so it is self-contained and reads on any backdrop. `voyager_cover_explanation.svg` is **black**
line art with no background, so it needs a light one — white, or gold if you want it to look like
the object rather than the diagram.

## Rendering them, and the trap in it

`qlmanage` — the obvious macOS renderer — **has no transparent output**. It flattens onto white,
so a plaque rendered that way carries a white square that is invisible on a light page and ugly on
a dark one. That is where `plaque.png`'s box came from.

`plaque.png` is therefore built by recovering the alpha geometrically rather than by adding a
system dependency: the plate is a clean circle of `#d4af37`, so fitting that circle from the gold
pixels gives the mask, with a 2.4px feathered rim so the edge is not stair-stepped. Everything
outside is fully transparent, so it sits on any background.

The spindle hole is filled **black**, not white. Flattening made it white; on the real object you
are looking *through* it, and black is both truer and stops the centre glowing on a dark page.

If you have `rsvg-convert` or a working `cairosvg`, render the SVG directly and skip all of this —
those preserve alpha properly.
