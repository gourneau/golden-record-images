# Corrections and enhancements we could offer back

Findings from this project that bear on public sources — Wikipedia, Wikimedia Commons,
and the pages this work was built on. **Nothing here has been submitted anywhere.** It is a
staging list, written so each item can be checked before anyone acts on it.

Ground rules, because getting this wrong would be worse than saying nothing:

- Every claim names the measurement or file behind it, so a reviewer can re-run it.
- Items are graded **solid** / **probable** / **needs work** by our own confidence, not by
  how interesting they are.
- Wikipedia wants published sources, not original research. Most of what follows is
  original measurement and would need to be published somewhere citable first, or offered
  as a talk-page note rather than an edit. Items marked **verifiable-on-file** are
  different: they are checkable against the artifact itself or against an existing
  citable source, and are the ones worth raising first.

---

## A. Wikimedia Commons — the image files

### A1. File 42 is mislabelled *(solid, verifiable-on-file)*

`File:Voyager_golden_record_42_Adams_The_Tetons_and_the_Snake_River.gif` is numbered 42
and titled as the Ansel Adams photograph. It is **image 43**. Image 42 is *Cape Neddick,
Maine*, credited to Dick Smith.

- **Evidence:** the Ozma Press catalogue (*Murmurs of Earth*'s companion listing) prints
  the sequence; our decode of the frame at position 42 is the Cape Neddick lighthouse
  coastline and matches that plate at 249 RANSAC inliers. The Adams photograph matches the
  next frame.
- **Verifiable without us:** yes — compare the file against any published numbered list.
- **Suggested action:** rename or re-caption the file, and check whether the error has
  propagated to the numbering of neighbouring files.

### A2. Fifteen reproductions are laterally inverted *(solid for the set, per-file review needed)*

Aligning our decodes against published reproductions required a transform that can mirror.
Once it could, fifteen plates matched that had never matched before, several
overwhelmingly:

| Image | Subject | Inliers direct → mirrored |
|---|---|---|
| 67 | Elephant, Mae Sariang, Thailand | 3 → 799 |
| 84 | Home construction, Sangmélima, Cameroon | 3 → 503 |
| 65 | Women in Santiago Atitlán, Guatemala | 4 → 474 |
| 64 | Dancer from Bali | 4 → 434 |
| 102 | Rush hour, Bangkok | 6 → 400 |
| 76 | Grape picker, Griffith, New South Wales | 10 → 328 |
| 79 | Fishermen at Evia, Greece | 5 → 257 |
| 66 | Craftsman carving elephants, Chiang Mai | 26 → 228 |
| 36 | Children at the UN International School | 5 → 218 |
| 68 | Farmer from Cappadocia, Turkey | 3 → 159 |
| 56 | Tree toad | 0 → 131 |
| 73 | School in Eastern Hokkaido, Japan | 4 → 112 |
| 114 | Whistling swans, Back Bay, Virginia | 3 → 112 |
| 35 | Malaysian man and his daughter | 4 → 81 |
| 8 | Solar spectrum | *tie, 37 → 36 — settled on physics instead, see C2* |

- **Evidence:** `pipeline/align_refs.py`, scores in `docs/reference/mirrored.json`.
- **Important caveat:** this says the *reproduction we hold* disagrees with the record. It
  does not by itself say which is faithful to the original photograph — a slide could have
  been mounted backwards in 1977, in which case the record is the flipped one. For our
  purposes the record is authoritative because the record is the artifact. **For a
  correction to Commons, each file needs checking against the photographer's own
  publication**, not just against us.
- **Suggested action:** treat as a per-file review list, not a batch edit.

---

## B. Wikipedia — *Voyager Golden Record* and related articles

### B1. The trace count and the picture width are different numbers *(probable)*

Articles state the images are "512 lines". That is right about the **scan** and is what the
cover encodes. It is easy to read as the picture being 512 wide in a 4:3 frame, which it is
not: the 4:3 active picture spans **503 ± 4 traces**, the rest being blanking.

- **Evidence:** `pipeline/geometry.py`; isotropy 7.4406 ± 0.0033 samples per trace-width.
- **Needs:** publication before it is citable. Best offered as a talk-page note.

### B2. The alternating sync is a documented field-ID code, not a defect *(solid, verifiable-on-file)*

Descriptions of decoding this record often treat the alternating line spacing as noise or
jitter to be worked around. It is deliberate: the Colorado Video scan converter used two
sync-pulse widths so a receiver could tell which interlace field a line belonged to.

- **Evidence:** US patent 3,950,607 (field identification by sync width) and US 4,802,008
  (the scan converter). On our master, the wide pulse sits high ~155 samples before falling
  and the narrow one ~45, but **both fall through zero at the same instant**.
- **Why it matters practically:** locking to the pulse peak gives ±100 samples of timing
  error; locking to the falling edge gives ±6.
- **Verifiable without us:** the patents are citable published sources.
- **Suggested action:** this is the strongest candidate for an actual article
  contribution, since the sourcing is already public.

### B3. The scan is sampled, not swept *(probable → solid, but needs publishing)*

Each trace is not a continuous brightness sweep. The converter sampled once per NTSC scan
line, giving **262.5 sample-and-hold plateaus per trace** — the NTSC half-line, which is
why the dot phase advances half a dot every trace.

- **Evidence:** measured at 262.519 ± 0.043 against the 262.500 the cover's trace duration
  predicts independently; survives widening the search ±3 dots on 156/156 frames. A narrow
  spectral line at 31.52 kHz in the picture band.
- **Note:** we found the number in the patent and then confirmed it from the audio alone.
  The confirmation is the citable part.

### B4. Claims about the images being quantised to 16 grey levels *(needs work)*

The cover's grey-scale reference is sometimes described as meaning the images carry 16 grey
levels. Our decodes are not quantised to 16 steps. The cover mark reads more naturally as
guidance on dynamic range.

- **Confidence:** we are confident the images are not 16-level. We are **not** confident
  about what the designers intended, and would not assert intent.

---

## C. Things we got wrong, listed so they are not propagated

If any of this project's earlier text was read and reused, these are the retractions.

### C1. "60 Hz mains hum at ~70% of picture amplitude" — **retracted**

Wrong twice over. The alternation is at **60.0436 Hz**, essentially exactly half the scan
rate, and clearly not 60.000 Hz mains. The original test could never have separated the two
hypotheses: over one 512-trace image they sit 0.21 FFT bins apart. Measuring the whole
record gives 25× the resolution and settles it. The "70%" was measured inside the sync
burst through a coordinate bug; in the picture the effect is 5–30%.

### C2. Which end of our spectrum is red *(open, and the record cannot settle it)*

The solar spectrum slides are the record's intended universal key for the colour order.
They give **ordering** — our three separations peak at 84.2%, 60.4% and 39.3% across the
frame, cleanly monotonic, so the filters are ordered by wavelength. They do **not** give
**direction**: after removing the smooth envelope, residual line structure is 0.6–1.9 grey
levels against an envelope of 60–124, i.e. 0.5–1.6%. The Fraunhofer lines are not readable.

So R,G,B versus B,G,R is a coin flip on the record's own evidence, which is exactly why
published decoders disagreed. We settled it with "oceans are blue", which is Earth
knowledge. **Anyone citing a colour order should say which evidence they used.**

### C3. A geometry contradiction we claimed and withdrew

We asserted that 512 traces, 4:3 and 234 dots were mutually inconsistent. That came from a
four-point bounding-box fit; a proper ellipse fit disagreed. Withdrawn.

---

## D. Enhancements rather than corrections

- **The pulsar map is a second self-test.** The circle checks your timebase; the pulsar map
  — which is *both* image 2 and an engraving on the cover — checks the handedness of your
  raster, because it is strongly chiral. We have not seen this pointed out anywhere, and it
  belongs in any description of how the cover works. *(We have not yet run the comparison
  ourselves; it needs a rectified image of the engraving.)*
- **The record encodes no orientation at all**, and not merely no "up". We tested whether a
  slide fed in sideways left blank bands at the frame edges, which would tell a recipient
  which images want turning: it does not — the operators reframed each slide to fill the
  4:3 raster. The detector finds 1 of 66 turned frames. (`pipeline/orient_blind.py`.)
- **A resolution ceiling worth stating.** The recording chain could have carried >1000
  elements along a trace; the 1977 camera resolved 138–172. Anything sharper that modern
  processing produces is invented, not recovered.

---

## E. Sources this work is built on, and what we owe them

- **Ron Barry**, [github.com/foodini/voyager](https://github.com/foodini/voyager) and his
  essay *The Voyager Image Decoding Method* — the starting point for all of this. Where we
  correct him (the picture window, the sync landmark, the decay bias) it is because he
  published enough detail to be checked, which is the whole point.
- **David Pescovitz and Ron Barry**, for the
  [384 kHz digitisation](https://archive.org/details/voyager.decode) without which none of
  this exists.
- **Boing Boing's 2017 article**, which is how most people including us first learned the
  decode was possible.

---

## F. Decoder findings from the adversarial gap review (2026-08)

Three gaps were attacked in parallel and each finding was then handed to a separate
agent whose only instruction was to refute it. Two of the three headline claims did not
survive contact, which is the reason for doing it that way.

### F1. The per-frame picture gate is a null *(solid)*

The picture window can be measured blindly on 156/156 frames — fold each frame onto its own
grid and the gate close shows as a bump at 1.8–3.3× the blanking floor. It does **not move**:
noise-corrected frame-to-frame spread is **1.36 bins = 0.18 of one output row in 377**. The
single global constant was already right.

**Consequence for a number this project has quoted.** `oracle.py` searches picture shifts of
±9.6 and ±19.2 bins — 7 to 14× the entire measured variation — so those shifts are not
recovering a gate. They are buying alignment against the *scanned reference slides*, whose crop
and centring are properties of the scan and are tier 2. **That part of the 0.2181 → 0.3080
oracle gap is not headroom and should not be cited as such.**

### F2. Moving PICTURE_END: rejected, against the workflow's own verdict *(measured here)*

The review found, unanimously over 156 frames, that moving the window end from 3040 to 3036.75
reduces a contamination signature in the bottom row, and recommended it.

Re-measured independently through the real code path — the module constant is bound into
`Settings` at class-definition time, so patching it does nothing, which is worth knowing before
anyone repeats this:

| window | bottom-row deficit (mean, 8 frames) | circle axis ratio | circle radial rms |
|---|---|---|---|
| shipped, 232 … 3040 | −0.0334 | **1.0051** | **0.837 px** |
| shorter, 232 … 3036.75 | −0.0196 | 1.0061 | 0.847 px |
| shifted, 228.75 … 3036.75 | **−0.0145** | 1.0061 | 0.874 px |

The bottom row does get cleaner. **But the calibration circle degrades in both variants**, and
the circle is the record's own designed self-check and the one invariant that cannot be gamed.
Trading 4.4% of the project's headline geometry metric for one row in 377 is the wrong trade.
**Not shipped.**

### F3. The aperture constant is frame-set dependent *(refuted)*

`TRANSITION_1090 = 0.49` dots was re-derived as 0.479 (0.470–0.498) — until the verifier ran
the same code on eight frames the method was not developed on and got **0.464 (0.449–0.471),
which excludes the shipped value.** There is a per-channel systematic the fit does not model.
The aperture correction stays unmerged, and its blocking condition stands.

### F4. Line art needs no special handling *(confirmed, with the claim weakened)*

On 31 blindly-identified line-art frames the denoiser does the same thing it does to
photographs — the same edge cost for the same noise removal — so no gating rule is needed. The
original claim of "identical to two decimals" was overstated by roughly 20×; the honest
statement is that a frame-level bootstrap cannot distinguish the two classes.
