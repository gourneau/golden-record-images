"""What the decoder knew, and where it learned it.

The project has two goals that pull against each other. One is a clean-room
decode: recover the pictures from the artifact alone, the way a recipient who
has never heard of Earth would have to. The other is the best pictures we can
get, which means using the originals to check our work and, if we want to know
what perfect tuning would buy, to tune against.

Both are worth having. Mixing them silently is what makes a decode worthless,
because a reader cannot tell which claims survive without Earth knowledge. So
every shipped artifact carries a TIER, and the boundary is asserted in code
rather than promised in prose.

    TIER 0  RECORD      the audio and the engraved cover, nothing else.
                        This is what an alien has. It is the archival product.

    TIER 1  UNIVERSAL   priors any observer could hold without knowing Earth:
                        image statistics, smoothness, noise models, the fact
                        that three separations of one slide must agree.
                        Alien-available, but prior-based, so it can invent
                        detail that was never measured. Kept separate from
                        tier 0 for that reason, not because it is cheating.

    TIER 2  EARTH       the original slides, their published captions, and
                        which edge is up. Cheating, in the sense that matters:
                        an alien could not do it. Allowed for EVALUATION and
                        for PRESENTATION, never as an input to pixel recovery.

    TIER 3  ORACLE      settings chosen per image by looking at the answer.
                        Not a decoder at all -- it is the measurement of how
                        much headroom better blind tuning could still win.

The rule that makes this more than labelling: **tier 0 and tier 1 outputs must
be reproducible without any tier 2 file existing on disk.** `check()` enforces
it by walking the decode import graph and by confirming that no tier 0 artifact
depends on a reference path.

A note on tier 2 in the gallery. Rotation is tier 2 and always will be: the
record does not say which edge is the top (ALIENS.md sec. "Which way is up?").
We rotate anyway, because a viewer should not have to tilt their head -- but
the rotation is applied at DISPLAY time and never baked into a decoded pixel,
so the tier 0 product on disk stays tier 0.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PIPELINE = REPO / "pipeline"

RECORD, UNIVERSAL, EARTH, ORACLE = 0, 1, 2, 3
TIER_NAME = {RECORD: "record", UNIVERSAL: "universal", EARTH: "earth", ORACLE: "oracle"}
TIER_LABEL = {
    RECORD: "Record only — the audio and the cover. An alien could do this.",
    UNIVERSAL: "Universal priors — no Earth knowledge, but assumptions beyond the data.",
    EARTH: "Earth knowledge — the original slides or their captions. Cheating.",
    ORACLE: "Oracle — tuned per image against the answer. A headroom measurement.",
}

# Anything under these paths is Earth reference material. A tier 0 or tier 1
# module that reads one of them is a leak, whatever its docstring claims.
EARTH_PATHS = (
    "docs/reference",
    "web/public/data/catalog.json",   # carries titles and displayRotate
)

# barry_tables.json is MIXED, and treating it as wholly forbidden was wrong:
# `start_samples` is a table of frame start positions, which is a measurement of
# the audio and carries no Earth knowledge at all (it is inherited from the 2017
# decode rather than re-derived, which is a separate criticism). `orientation`
# and `color_role` are the Earth-knowledge fields -- one was made by looking at
# the pictures, the other by knowing what colour an ocean is. So the check is at
# key level: a tier 0 module may read the file, but not those two keys.
MIXED_FILES = {"barry_tables.json": ("orientation", "color_role")}


@dataclass(frozen=True)
class Artifact:
    """One shipped thing, and what it was allowed to know."""
    name: str
    tier: int
    what: str
    why: str
    module: str | None = None          # pipeline module that produces it
    consumes: tuple[str, ...] = ()     # tier names it is permitted to read
    # Set when an artifact SHOULD sit at a cleaner tier and is only where it is
    # because of a contamination we have identified but not yet removed. This is
    # the honest place to record "we know this is dirty", so the tier table
    # cannot be read as a claim of purity we have not earned.
    should_be: int | None = None
    contamination: str = ""


REGISTRY: tuple[Artifact, ...] = (
    # ---- tier 0: the archival decode -------------------------------------
    Artifact("pixels", RECORD,
             "every decoded pixel of all 156 frames",
             "decode.py imports numpy, dotclock and sync only; its sole data input is the WAV",
             module="decode.py"),
    Artifact("timebase", RECORD,
             "per-trace start times, trace period, wow and flutter",
             "matched filter on the blanking burst, which is in the audio",
             module="sync.py"),
    Artifact("dot_clock", RECORD,
             "262.519 sample-and-hold plateaus per trace",
             "measured from the audio; the cover's trace duration predicts 262.5 independently",
             module="dotclock.py"),
    Artifact("geometry", RECORD,
             "isotropy 7.4406 samples per trace-width, picture window, 4:3 span",
             "the cover states 512 traces, 8.34 ms and 4:3; the circle checks the result",
             module="geometry.py"),
    Artifact("droop_inverse", RECORD,
             "per-channel one-pole inverse, tau from a scene-free probe",
             "the probe is a content-free region of the record's own signal",
             module="droop_inverse.py"),
    Artifact("vertical_shading", RECORD,
             "the along-trace pedestal, 92.8 grey levels p-p, and its inverse",
             "measured on L000's field, which the cover states is a uniform slide with a "
             "circle on it; confirmed on the other 155 frames without using L000's pixels",
             module="dedroop.py"),
    Artifact("circle_metrics", RECORD,
             "axis ratio, radial rms, MTF50 of the calibration frame",
             "a circle is self-checking; the cover says the first image is one",
             module="circle.py"),
    Artifact("colour_grouping", RECORD,
             "which frames form a triplet (not which is red)",
             "three consecutive frames of one scene are detectable by correlation alone",
             module="colour.py"),

    # ---- tier 1: priors an alien could also hold --------------------------
    Artifact("denoise", UNIVERSAL,
             "noise suppression using image-statistics priors",
             "assumes natural-image smoothness; no Earth photograph is consulted",
             module="enhance.py", consumes=("record",)),
    Artifact("triplet_fusion", UNIVERSAL,
             "registration and fusion across the three separations of a slide",
             "cross-separation agreement is a property of the artifact, not of Earth",
             module="fuse.py", consumes=("record",)),
    Artifact("n2n_denoiser", UNIVERSAL,
             "the reconstructed tier: every frame with a denoiser applied",
             "trained ONLY on this record's own colour repeats -- twenty scenes scanned three "
             "times give independent noisy pairs, so no Earth photograph and no external corpus "
             "is involved and an alien could run it. Measured on the colour images against a "
             "held-out scan (19/19, blur control fails); on the 96 mono images there is no "
             "held-out measurement and it is an extrapolation, checked only against the "
             "calibration circle's geometry",
             module="reconstruct.py", consumes=("record",)),
    Artifact("destripe", UNIVERSAL,
             "per-trace streak removal by 2-D spectral separation",
             "the streak's SHAPE is measured on L000's ring interior, which the cover "
             "says is a circle on a uniform field, so the calibration itself is tier 0; "
             "the separation it drives -- streak smooth down the trace and rough across "
             "traces, scene smooth in both -- is a universal prior, and the correction "
             "is applied to frames the calibration never saw. Held out across traces, "
             "checked against the circle's geometry, and bounded by a transpose control "
             "on twelve other frames",
             module="destripe.py", consumes=("record",)),
    Artifact("deconvolution", UNIVERSAL,
             "deconvolution by the measured camera PSF",
             "the PSF is measured from the record; the inversion prior is generic",
             module="deconv.py", consumes=("record",)),

    # ---- tier 2: Earth knowledge -----------------------------------------
    Artifact("display_rotation", EARTH,
             "the quarter turn each image is shown at",
             "the record encodes no 'up'; measured against published reproductions",
             module="align_refs.py", consumes=("record", "earth")),
    Artifact("mirror_correction", EARTH,
             "which published reproductions are laterally inverted",
             "decided by matching against our decode; the record is authoritative",
             module="align_refs.py", consumes=("record", "earth")),
    Artifact("titles", EARTH,
             "the caption under each image",
             "from the Ozma catalogue and NASA/JPL, or from recognising the subject",
             module="catalog.py", consumes=("earth",)),
    Artifact("colour_roles", EARTH,
             "which separation is red, green and blue",
             "intended to be tier 0 via the spectrum slides, but they are too faint; "
             "currently settled by oceans being blue, which is Earth knowledge",
             module="colour.py", consumes=("record", "earth")),
    Artifact("thumbnails", EARTH,
             "the shipped PNG of every frame, as the gallery loads it",
             "the pixels are tier 0, but their LAYOUT is not: build.py applies Barry's "
             "hand-made quarter-turn table before writing the file",
             module="build.py", consumes=("record", "earth"),
             should_be=RECORD,
             contamination="60 of 156 frames get a non-zero quarter turn from "
                           "barry_tables.json['orientation'], a table made by looking at the "
                           "pictures. Fix: write the PNG in the decoder's own raster frame and "
                           "move the turn to display time, where the gallery already applies "
                           "displayRotate. Requires recomputing displayRotate against unrotated "
                           "thumbnails, since it is currently measured relative to the baked ones."),
    Artifact("evaluation", EARTH,
             "gradient NCC against the original slides, dev / held-out split",
             "scoring only; no score is fed back into a tier 0 module",
             module="evalset.py", consumes=("record", "earth")),

    # ---- tier 3: the oracle ----------------------------------------------
    Artifact("oracle_settings", ORACLE,
             "per-image decoder settings chosen using the true slide",
             "measures the headroom a better blind decoder could still win",
             module="oracle.py", consumes=("record", "earth")),
)


def by_tier() -> dict[int, list[Artifact]]:
    out: dict[int, list[Artifact]] = {t: [] for t in TIER_NAME}
    for a in REGISTRY:
        out[a.tier].append(a)
    return out


def _string_constants(path: Path) -> list[str]:
    """Every string literal in a module, so path references cannot hide."""
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return []
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _local_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return set()
    names: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                names.add(a.name.split(".")[-1])
        elif isinstance(n, ast.ImportFrom) and n.module:
            names.add(n.module.split(".")[-1])
    return {m for m in names if (PIPELINE / f"{m}.py").exists()}


def _closure(module: str) -> set[str]:
    """Every pipeline module reachable from this one."""
    seen: set[str] = set()
    stack = [module.removesuffix(".py")]
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        stack.extend(_local_imports(PIPELINE / f"{m}.py") - seen)
    return seen


def check() -> list[str]:
    """Return every violation of the tier boundary. Empty list means clean."""
    problems: list[str] = []
    for a in REGISTRY:
        if a.module is None or a.tier >= EARTH:
            continue
        if "earth" in a.consumes:
            problems.append(f"{a.name}: tier {a.tier} artifact declares it consumes earth data")
            continue
        mod = PIPELINE / a.module
        if not mod.exists():
            problems.append(f"{a.name}: module {a.module} does not exist")
            continue
        for m in sorted(_closure(a.module)):
            lits = _string_constants(PIPELINE / f"{m}.py")
            for lit in lits:
                for bad in EARTH_PATHS:
                    if bad in lit:
                        problems.append(
                            f"{a.name} (tier {a.tier}) reaches Earth data: "
                            f"{m}.py references {lit!r}")
                for mixed, earth_keys in MIXED_FILES.items():
                    if mixed in lit:
                        for key in earth_keys:
                            if key in lits:
                                problems.append(
                                    f"{a.name} (tier {a.tier}) reads the Earth-knowledge key "
                                    f"{key!r} of {mixed} in {m}.py")
    return problems


def report() -> str:
    groups = by_tier()
    lines = ["PROVENANCE", ""]
    for t in sorted(groups):
        arts = groups[t]
        lines.append(f"  TIER {t}  {TIER_NAME[t].upper()}  ({len(arts)} artifacts)")
        lines.append(f"    {TIER_LABEL[t]}")
        for a in arts:
            lines.append(f"      {a.name:<20s} {a.what}")
            lines.append(f"      {'':<20s}   why: {a.why}")
        lines.append("")

    dirty = [a for a in REGISTRY if a.should_be is not None]
    if dirty:
        lines.append("  KNOWN CONTAMINATION (artifacts sitting at a dirtier tier than they should)")
        for a in dirty:
            lines.append(f"      {a.name}: tier {a.tier} ({TIER_NAME[a.tier]}), "
                         f"should be tier {a.should_be} ({TIER_NAME[a.should_be]})")
            lines.append(f"      {'':<4s}{a.contamination}")
        lines.append("")
    return "\n".join(lines)


def to_json() -> dict:
    return {
        "tiers": [{"tier": t, "name": TIER_NAME[t], "label": TIER_LABEL[t]} for t in sorted(TIER_NAME)],
        "artifacts": [{"name": a.name, "tier": a.tier, "tier_name": TIER_NAME[a.tier],
                       "what": a.what, "why": a.why, "module": a.module,
                       "consumes": list(a.consumes),
                       "should_be": a.should_be,
                       "contamination": a.contamination} for a in REGISTRY],
    }


if __name__ == "__main__":  # pragma: no cover
    print(report())

    problems = check()
    if problems:
        print("TIER VIOLATIONS")
        for p in problems:
            print(f"  {p}")
    else:
        print("tier boundary clean: no tier 0 or tier 1 artifact reaches Earth data")

    # The independent check that already existed, kept as a second opinion.
    try:
        from pipeline.evalset import assert_decode_is_blind
        v = assert_decode_is_blind()
        print(f"decode.py blindness: {'VIOLATIONS ' + str(v) if v else 'clean'}")
    except Exception as e:
        print(f"decode.py blindness: could not run ({e})")

    out = REPO / "web" / "public" / "data" / "provenance.json"
    out.write_text(json.dumps(to_json(), indent=1))
    print(f"wrote {out.relative_to(REPO)}")
