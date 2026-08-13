# What this decoder found

The measurements behind the [README](../README.md), in the detail a signal-processing reader
wants and a general one does not. Everything here is measured from the 384 kHz master and
reproducible with the code in `pipeline/`.

Related: [how the decoding works](../DECODING.md) for a general audience ·
[what an alien could recover](../ALIENS.md) ·
[the machine learning](ML_HANDOFF.md) ·
[corrections and retractions](corrections.md)

---

Everything below is measured from the master, and the measurements are reproducible with the
code in `pipeline/`.

### The trace is sampled, not swept

Every decoder written for this record — Barry's, the seven other public ones, and every earlier
version of this one — treats a trace as a continuous brightness sweep to be chopped into *N*
equal bins.

It isn't. Patent [US4802008](https://patents.google.com/patent/US4802008A/en) quotes the
Colorado Video Series-262 specification: the converter sampled the source television picture
**once per scan line** and held each value. *Murmurs of Earth*'s "eight seconds each" agrees. A
trace is a staircase of 262.5 sample-and-hold plateaus — one NTSC field, one dot per line — not
a ramp.

So we went looking for that clock in the audio. Averaging the picture-band spectrum over 440
traces with the smooth background divided out produces a narrow line exactly where the patent
predicts, on 16 of 22 frames sampled across both channels and the whole record:

```
dots per trace    mean 262.519    sd 0.043    range 262.47 .. 262.61
NTSC field                        262.500     ->  error 0.007%
```

The six frames that show nothing simply lack the dot-to-dot contrast needed to excite the line;
there the rate is predicted from the trace period and flagged as unmeasured rather than
asserted.

A trace being sampled means it should be *decoded* as sampled: integrate each dot over its own
plateau, which is the matched filter for sample-and-hold. Any other bin grid mixes adjacent dots
together. Native output is now **234 × 512** — the true dot count — rather than an invented 384
rows.

It also settles the geometry — though not in the way we first thought.

Because a trace is one *field* rather than a full frame, it carries half a frame's vertical
lines, so the dots are not square. Measured against the sync falling edge, the picture gate runs
from bin 232 to bin 3040 of a 3200-bin period, giving **~231 active dots**, and the isotropy —
the amount of trace-time that equals one trace of width — is

```
7.4406 ± 0.0033 bins per trace   ->   dots are 1.638x taller than a trace is wide
```

So a square-pixel render of 512 traces is **512 × 377**, an aspect of 1.357. Which means the
cover's "512" is *not* the width of the 4:3 picture: an exactly-4:3 area is **503 ± 4 traces**,
and the converter actually scans ~535 traces in total. Barry's 540 is that scan plus a 22-trace
hatch marker.

Checked on seven circles across five frames — the calibration ring, both rings of the solar
system diagram, all three Earth separations, and the limb of Mars — the residual stretch is
0.981–1.001. The competing "512 traces is exactly 4:3" hypothesis predicts 1.018 on every one of
them, and is refuted with breadth rather than on the calibration frame alone.

*Two corrections are folded in here.* An earlier revision of this README claimed 512 traces, 4:3
and the dot count were mutually **inconsistent**, citing a ring axis ratio of 1.1159 — that came
from thresholding a thin ring's bounding box in a resampled display image, which is a poor
estimator, and the contradiction was mine, not the record's. A second revision then claimed all
three were consistent, which was closer but still wrong: with 512 traces the aspect is 1.357, not
1.333. The isotropy figure is the durable number; the trace count was the loose assumption.

**It also dissolves a 2017 mystery.** Barry hardcoded "−12 samples on even traces" and called it
a brainless heuristic. Twelve samples is *one dot*: one NTSC line period at the 2× tape speed.
It was never a fudge factor. It was the scan converter's own clock showing through, and he was
compensating for it without knowing what it was.

**And an honest deflation, because this was oversold.** The dot clock is correct science — the
measurement survived the hardest test we could design, in that widening the search from ±0.6 to
±3.0 dots returns the identical peak on all 156 frames, and no frame lands within 0.02 of the
line-rate integers that would indicate a false lock. But it buys **no measurable image quality**.
Forcing a deliberately wrong rate, 0.6 dots off, changes sharpness by less than 1%, because the
phase tracker absorbs rate error. It is a true description of the hardware and a dead end for
picture quality, and both halves of that deserve saying.

### The decode is at the physical limit

The most consequential measurement in this project, and the one that ends the search for more
detail. Reconstructing the resolution actually present, in both axes:

| axis | we sample | the 1977 camera resolves | oversampling |
|---|---|---|---|
| along a trace | 230 dots | 138–172 elements | 1.4× |
| across traces | 512 traces | 260–324 | 1.6–1.9× |

The **recording chain** could have carried more than 1000 elements per trace — its own edge
response is 1.75–2.37 samples. The **camera** could not. We are already sampling 1.4–1.9× above
the camera's Nyquist frequency in both directions, which means the pictures contain every element
the 1977 television camera was able to resolve, and no deconvolution, neural network or
super-resolution scheme can recover detail that was never captured. Anything that appears to is
inventing it.

Measured independently on the calibration ring: MTF50 of 0.102 cycles/px, a line-spread function
5.1 px wide, giving ~77 resolvable elements per trace at the ring's own contrast.

One real defect remains unrecovered: a slow tilt below 120 Hz, +0.011 to +0.023 per trace, which
is 4–9% of the black-to-white range. The calibration slide is a uniform white field by design and
ours still comes out with a −0.57 tilt across it. An earlier attempt to explain this as a
low-frequency "shelf" in the chain's response turned out to be an artefact of the estimator's
normalisation band — |S(180 Hz)| moves from 0.558 to 1.089 depending purely on where that band is
put — so the tilt is real but its cause is still open.

### The alternating sync pulse — and a correction

Every other trace behaves differently. Barry saw it in 2017 as traces alternating ~3100 and
~3300 samples apart, and killed it with a hardcoded "−12 samples on even traces, changing at
trace 164". It is the most consequential feature of this signal.

**An earlier version of this README claimed this was 60 Hz mains hum phase-locked to the
120 Hz scan, at ~70% of picture amplitude. Both halves of that were wrong, and the corrected
account is more interesting.**

*It is not mains.* Decimating the whole 473 s record and taking a 0.002 Hz-resolution spectrum
puts the component at **60.0436 Hz** — 2.5 bins from half the measured line rate (60.0488 Hz)
and 20.6 bins from mains (60.000 Hz). It is locked to the scan, not to the power grid. The
original claim rested on a peak at "exactly 0.5000 cycles/trace", but over a 512-trace frame
those two hypotheses sit 0.21 FFT bins apart — that measurement could never have distinguished
them, and I should not have asserted it.

*It is not 70%.* That figure was measured inside the sync burst, which was leaking into the
picture window because of a coordinate-convention bug. The genuine parity-locked component
within the picture is 0.0016–0.0031 in signal units, **5–30% of picture RMS**.

*What it actually is:* by design. Colorado Video's Model 262 scan converter specifies **two
sync/blanking widths, used to identify which interlace field a line belongs to**. The encoder
is telling the receiver which field it is looking at. On our master the even-trace pulse goes
high ~155 samples before the falling edge and the odd-trace pulse only ~45 before; both fall
through zero at the same instant.

The practical consequence is that the alternation is primarily a **timing** effect, not an
amplitude one — which is why a single matched filter cannot fit both parities, and why
correlating against one averaged template mislocks by ~100 samples on alternate traces. Two
parity-specific templates cut one bad frame from 89 misplaced traces out of 512 down to 2.

### Lock to the falling edge, not the peak

Landmark stability, measured as the standard deviation of spacing between consecutive
detections on our master:

| landmark | spacing std |
|---|---|
| peak maximum (Barry's, and most decoders') | **100.3 samples** |
| trough after the peak | 7.1 |
| **downward zero crossing of the falling edge** | **5.5 – 6.2** |

The peak moves because the pulse *width* alternates; the falling edge does not. A ±100-sample
landmark error is exactly what produces a diagonal staircase band across a photograph, and it
explains why line-art diagrams decode cleanly while detailed photographs fail — picture content
occasionally out-peaks the sync burst.

### Sync by matched filter instead of peak-picking

Averaging several hundred traces makes picture content vanish and the sync burst emerge
sharply. Correlating each trace against that template — using every feature of the burst at
once, rather than whichever spike happens to be tallest — and refining the peak parabolically
locates each trace to a fraction of a sample.

| | 2017 method | this decoder |
|---|---|---|
| sync timing error | 397 samples RMS | **6.1 samples RMS** |

That's about 65× better, and it removes the need for the hardcoded "±12 samples on even
traces, changing at trace 164" correction.

### The picture window, measured rather than assumed

The sync burst is identical on every trace and the picture isn't, so averaging the traces
reveals exactly where one stops and the other starts. The active picture is **2873 samples,
89.9% of the trace**. The 2017 decoder used 2680 — about 7% short, which is precisely why its
calibration circle came out as an ellipse needing a manual "couple percent" correction.

The picture also **wraps around** the point the sync template locks to; roughly 310 samples of
each stripe come *after* the lock point in the file but *before* it in the image.

### Timing measured per frame, not assumed

The cover specifies 8.34 ms per trace. The recording actually runs at **8.326 ms** — about
0.16% fast. Small, but over 512 traces it accumulates into a visibly skewed image. Every frame
gets its own fitted period, and the wobble around that fit is the master tape's wow and
flutter, which is corrected per trace rather than averaged into a slant.

### A decoder that reports its own damage

This is a single digitisation of a 1977 analog chain, so it has dropouts, clicks, level drift
and speed variation. The decoder measures each one and carries a **per-trace confidence
score**, so the interface can show which parts of a picture are trustworthy instead of quietly
presenting a guess as data.

### Things I checked and found nothing

Reported because negative results are still results:

- **The images are not quantised to 16 grey levels.** The cover's "16 levels" appears to be
  guidance on dynamic range, not literal steps.
- **The ~1.7 s gap between images is not hidden data.** It's a framing signal at exactly twice
  the trace period. It is also not the sawtooth the cover's glyph might suggest (r² = 0.016
  against a linear ramp).
- **The true sharpness limit is still unknown.** Measuring it honestly is harder than it
  looks — a careless measurement measures your own timing error instead of the record's, which
  is exactly what my first attempt did before I threw it out. No number here yet.

---
