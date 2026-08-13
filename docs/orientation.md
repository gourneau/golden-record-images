# Which way is up

The Voyager record does not say.

That is not a decoder bug and it is not a gap in our work. It is a gap in the record's design,
and it belongs on the list in [ALIENS.md](../ALIENS.md) alongside the other things a recipient
could not recover. Of the 116 images encoded as audio on the record, **not one carries any
indication of which edge is the top.**

---

## What the signal actually fixes

The encoding is precise about a great deal. Each image is 512 traces. Each trace is 8.34
milliseconds, written on the cover as a multiple of the hydrogen hyperfine period. The traces
are laid side by side to build a raster. The cover even draws you a picture of the scan lines
so you cannot mistake what you are assembling.

So the signal fixes the geometry completely: how many traces, how long each one is, which order
they arrive in, and therefore the shape of the rectangle they make. What it never fixes is the
orientation of that rectangle relative to the subject inside it. Nothing in the waveform
distinguishes a photograph of a standing person from the same photograph lying on its side. The
raster is the raster. Up is not encoded.

You can see the consequence directly: 43 of the 116 decode "sideways" — a portrait photograph
laid into a landscape raster, or the reverse. There is no signal-level reason to prefer one
reading. A recipient with perfect instruments, infinite patience and no knowledge of
Earth would decode all 116 images correctly and still have no way to tell which of four
rotations was intended for any of them. Four to the power of 116 is not a search space you can
brute-force your way out of, because there is nothing to check an answer against.

Well — almost nothing. Which is the interesting part.

## Where our orientation comes from instead

Ours comes from recognising the subject. That is human knowledge, and it is exactly the kind of
knowledge the record's own designers assumed the recipient would lack.

We know a skeleton has its skull at the top because we have skulls. We know the sky is above the
horizon in a photograph of Monument Valley because we have stood in deserts. We know
`13 x 28 = 364` is printed the right way up because we read left to right and write our numerals
that way. Every one of those is a fact about us, not a fact about the signal.

It is worth being blunt about the strongest cases, because they are the ones most likely to be
mistaken for signal evidence. About sixty of the images carry hand-lettered annotations — scale
bars like `12 cm`, elapsed times like `22982400 s`, captions like `6787 km`. Those settle the
orientation instantly and unarguably. But they settle it *only for a reader of Latin script and
Arabic numerals*. The lettering functions as an orientation mark entirely by accident. Nobody
designed it as one, and it does not appear on the images that need it most: the landscapes, the
textures, the wildlife, the photographs with no text at all.

So this repository records two separate things, and takes some care never to blur them:

| Field | What it is |
|---|---|
| `encodedOrientation` | Always `0`, for all 116. The unrotated decode — what the signal produced. A fact. |
| `displayOrientation` | Quarter turns clockwise needed to view it as intended. Our judgement, from recognising the subject. Not in the record. |

Both live in [`data/orientation_audit.json`](../data/orientation_audit.json), one row per image,
with the evidence written out.

---

## The 2017 table, and a convention that bit everybody

The orientation values this project inherited come from Ron Barry's 2017 decode. He judged them
by eye while working and, so far as we can tell, never systematically re-checked them. His column
holds `0`, `1`, or `2`.

The trap is that `1` and `2` are quarter turns in **opposite directions** — `1` is 90° clockwise,
`2` is 90° counter-clockwise — and the column cannot express a half turn at all. That is easy to
misread as "one quarter turn" and "two quarter turns", and code in this repository has misread it
in both available ways at once: the build pipeline treats `2` as 180°, while the gallery renderer
treats `1` as counter-clockwise and `2` as clockwise, which is precisely backwards.

We pinned the convention empirically rather than trusting any of that code, using frames whose
own lettering leaves no room for argument:

- **`1` = 90° clockwise.** On image 26 the words `MALE` and `FEMALE` and every leader label
  (`URETER`, `PROSTATE`, `CERVIX`, …) read left to right only after a clockwise quarter turn. On
  image 32 the same turn sets `160 cm`, `155 cm` and `20 y` upright with both figures standing.
- **`2` = 90° counter-clockwise.** All four frames carrying the value agree: Jupiter's
  `142800 km / 318 e`, Earth's `12756km / 1e`, the ichneumon wasp's `2 cm / 1 g`, and the
  Cavatina score's staves and bar lines all come upright only under a counter-clockwise turn.

Because `displayOrientation` needs to express a half turn, it does **not** use Barry's encoding.
It is a plain count of quarter turns clockwise: `0`, `1` = 90° CW, `2` = 180°, `3` = 90° CCW. The
audit file carries `currentOrientationAsQuarterTurnsCW` so the old and new columns can be compared
without arithmetic.

## How we checked

Every one of the 116 images was rendered at all four quarter turns from the as-encoded decode and
looked at. Colour images were composited from their RGB triplets first where that helped.

Those judgements were then screened against an independent source: the scanned plates in
[`docs/reference`](reference), matched to our decodes by SIFT correspondences and RANSAC. Ninety-five
images have a plate match at or above 60 inliers, which the matching notes treat as the threshold
for a real match. **Ninety-three of the 95 confirmed the 2017 table.** The two that did not were
pulled up side by side with the plate and inspected by eye.

That is a good result for Barry. Judged by eye in 2017, under no obligation to be careful about it,
his table is right 114 times out of 116.

---

## The two corrections

### Image 2 — Solar location map (`L001`): `0` → **180°**

The pulsar map. The scanned plate prints the radiant at **upper left** with the galaxy inset at
**lower right**; our as-encoded decode has both exactly reversed.

The clinching detail is asymmetric, so it cannot be explained away as a symmetric diagram read two
ways: a small bright companion dot sits to the **left** of the galaxy's bright core in the plate,
and to the **right** of it in the as-encoded frame. Rotating our decode 180° puts the dot back on
the left and brings every dashed pulsar tick-string into register. RANSAC independently recovers
180°.

This one could never have been recorded in the old table, since a half turn is not one of the
three values available. It was not a mistake by Barry so much as a limitation he inherited.

### Image 28 — Conception (`L035`): 90° CW → **none**

A micrograph of sperm around an ovum. The plate is **landscape**: the grainy ovum fills the right
of the frame, the sperm are scattered through the dark field to the left, and a hand-drawn leader
line points rightward at a single sperm on the ovum's margin. That is the as-encoded frame, detail
for detail — the second sperm just below and left of the leader tip, the column of sperm down the
left edge, all of it. The match scored 625 inliers, the strongest in the whole set.

Barry's quarter turn stands it portrait with the ovum at the bottom. Every previous pass over this
image called it genuinely undetermined — it has no text, no scale bar and no gravity cue — and kept
the quarter turn only by analogy with the diagram beside it in the sequence (image 27), which does
put the ovum below and the sperm above. The plate shows the analogy was misleading. It is a small,
honest error, and the kind that only an external reference can catch.

---

## What remains genuinely ambiguous

**One image: number 1, the calibration circle.**

It is a plain dark circle on a light field. It is rotationally symmetric, so all four turns are
identical apart from decode noise along one frame edge, and the reference scan is likewise a bare
circle with no asymmetric feature anywhere in it. There is no evidence, and there never will be.
We record `0` as the do-nothing default and label it `ambiguous` rather than pretending to a
finding.

There is something fitting about this. ALIENS.md argues the calibration circle is the single best
idea on the record — self-checking, needing no explanation, quantifying its own error. It is also
the one image whose orientation is permanently unknowable, and it does not matter in the least,
because a circle looks the same either way. The record's most universal image is the one that
never needed an up.

Two further images are marked `likely` rather than `certain`, because no plate matched them and
they rest on conventional presentation rather than a gravity cue or a piece of text:

- **31, Fetus** — head-up specimen posture, agreeing with the diagram before it.
- **56, Tree toad** — a toad sitting on an open human hand; a held hand does not fix which way
  is down.

Several images that earlier passes flagged as unresolvable turned out to be settled by an external
reference rather than by the frame itself, and are marked `certain` on that basis with the reason
recorded: **7** (the Sun montage), **8** (solar spectrum — the reference scan fixes blue-left,
red-right, which internal evidence cannot), **9** (Mercury), **53** (seashell), **55** (school of
fish) and **112** (Ed White's EVA, a free-fall photograph with no up of its own).

---

## A note on the images that looked wrong on the gallery

Six images were reported as visibly wrong on the public page: the nursing mother, the birth, the
Malaysian father and daughter, the male and female silhouettes, the continental drift diagram and
the autumn leaves. All six were checked here individually and **all six values in the 2017 table
are correct.**

What they have in common is that all six carry orientation `1`, and every `1` frame is affected:
the build pipeline bakes the quarter turn into the stored thumbnail, and the gallery renderer then
applies a further turn in the opposite direction. The two cancel, and the page displays the raw
undirected decode. That is 40 images — 60 stored frames, counting the colour triplets separately —
lying on their side for a single reason, not 40 separate misjudgements by eye in 2017.

Fixing that is a rendering question, and it is being handled separately from this audit. It is
worth separating cleanly, because the two failures look identical on screen and have nothing to do
with each other: one is a disagreement about what the pictures show, the other is a disagreement
between two pieces of our own code about what a number means.

---

## The design lesson

The record's authors solved a much harder problem than this one. They communicated a raster
format, a time base and a colour-separation scheme to a recipient with no shared language, and
they did it well enough that we can still decode it. Then they laid 116 photographs into that
raster without a single mark saying which edge was the top.

The fix would have been nearly free. One asymmetric registration mark in a corner of every frame —
a notch, a wedge, anything that cannot be confused with its own rotation — would have carried the
orientation of all 116 images at a cost of a few dozen pixels each. The calibration circle proves
they understood the value of a self-checking mark. They just did not extend it to the one property
of an image that a rectangle cannot state on its own.

We can supply the answer because we recognise our own skeletons, our own deserts and our own
handwriting. That is precisely the resource the record was designed not to depend on.
