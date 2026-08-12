# How to decode the pictures on the Voyager Golden Record

There are 116 photographs bolted to the side of two spacecraft leaving the solar system.
They aren't stored as pictures. They're stored as **sound** — about eight minutes of
screeching noise pressed into a copper phonograph record. This is how you turn that noise
back into pictures, and what goes wrong along the way.

No prior signal-processing knowledge assumed.

---

## 1. The basic trick: a picture drawn one stripe at a time

Imagine describing a photograph over the telephone. You could read it out column by column:
"top of column one is dark, middle is bright, bottom is dark… now column two…" If you read
fast enough and the person on the other end paints as fast as you talk, a picture appears.

That is exactly what the record does. Each image is **512 vertical stripes**, and each
stripe is one continuous swoop of a tone whose loudness is that stripe's brightness, top to
bottom. Play it and you hear a harsh buzz. Draw it and you get a photograph.

The 1970s hardware that did this was a one-off box built by a small television-equipment
firm in Boulder, Colorado. Each slide was projected into a TV camera, and the box converted
the camera's video into a few seconds of audio.

One stripe is called a **trace**. 512 traces make one image, and one image is about 4.3
seconds of sound.

---

## 2. The cover is the instruction manual

None of that would be guessable, so the aluminium cover bolted over the record is engraved
with the instructions — written to be readable by someone who shares no language, no units,
and no mathematics beyond physics.

The keystone is a small diagram of two hydrogen atoms. It depicts the hyperfine transition
of hydrogen: the most common event in the universe, with a period of **0.704 nanoseconds**.
Anyone anywhere can measure it. That single number is the cover's unit of time, and every
other duration is written as a multiple of it, in binary.

From the cover you get, without being told in words:

| Engraved | Means |
|---|---|
| binary counting key (`\|`, `\|-`, `\|\|`) | numbers are base-2, most significant digit first |
| hydrogen hyperfine diagram | 1 time unit = 0.704 ns |
| `1000000000` beside a striped rectangle | 512 — the image is 512 traces |
| `101101001100000000000000` | 11,845,632 time units = **8.34 ms per trace** |
| a waveform glyph | brightness is carried as signal level |
| a picture-in-picture of a circle | the first image is a circle, so you can check your work |
| rectangle proportions | the picture is 4:3 |

That last one is the cleverest part. **The first image is a circle**, and a circle is
self-checking: if yours comes out as an oval, your timing is wrong, and you can tell by
exactly how much. It's a test pattern that needs no explanation.

---

## 3. The seven steps

### Step 1 — Find where each trace starts

Between traces there's a burst of signal that carries no picture: a tall plateau, a sharp
falling edge, and a flat shelf. This is the metronome. Find those and you know where every
stripe begins.

The obvious approach — look for the tallest spike — doesn't work well, because there are
several features in that burst and on any given trace a different one may happen to be
tallest. That's what the 2017 decode did, and it's why its traces came out alternately
~3100 and ~3300 samples apart.

**The better way:** average several hundred traces together. Picture content is different
every trace so it averages away to nothing; the sync burst is identical every trace so it
survives, sharply. That gives you a clean picture of what the burst looks like. Then slide
that template along the audio and find where it fits best, on *every* feature at once
instead of one fragile spike. Fit a parabola through the three best-matching positions and
you get the start of each trace to a *fraction* of a sample.

Measured result: timing error drops from **397 samples to 6.1** — about 65× better.

### Step 2 — Measure the trace length, don't assume it

The cover says 8.34 ms. Measure it and you get **8.326 ms** — the recording runs about
0.16% fast. Tiny, but over 512 traces a 0.16% error accumulates into a badly skewed image.

So: measure it, per image. Fit a straight line through all 512 measured trace positions.
The slope is the true trace length; the wobble around that line is the tape's speed
wandering (**wow and flutter**), which you correct by using each trace's own measured
position rather than the average.

### Step 3 — Find where the picture actually is

Each trace is 3197 samples long, but not all of it is picture — the sync burst takes up
some. How much?

Don't guess: the sync burst is identical on every trace, and the picture isn't. So average
all the traces and look for the stretch that's identical — that's the burst, and everything
else is picture.

Measured: **the picture is 2873 samples, 89.9% of the trace.** The 2017 decode used 2680,
about 7% short — which is precisely why its calibration circle came out as an oval that had
to be nudged "by a couple percent" by hand. With the right number the circle comes out
round on its own, with no fudge factor.

One surprise: the picture **wraps around** the point the sync template locks to. The lock
point sits about 310 samples *inside* the picture, so you have to read the last bit of the
stripe first. Miss that and the image comes out sliced and shuffled.

### Step 4 — Turn each stripe into pixels

Now chop each stripe into 384 brightness values. The naive way is to average the samples in
each slot. Slightly better is to use a proper resampling filter (Lanczos), because you have
about 7 audio samples per pixel and want all of them contributing sensibly.

Two things matter more than the filter choice:

- **Use the fractional start position.** You measured the trace start to a fraction of a
  sample in step 1; rounding it off here throws that precision away.
- **The image is a negative.** More signal means *darker*. Forget this and you get a
  photograph of the Moon lit from the wrong side — which is exactly how the 2017 decode
  discovered the inversion.

### Step 5 — Fix the brightness drift

Nothing in a 1977 recording chain passes a constant level. Feed it a steady bright patch
and the signal sags back toward the middle. The visible result: every image is bright at
the top and dark at the bottom, and bright objects trail shadows behind them.

The fix is the one television engineers have used since the 1950s. That flat shelf in the
sync burst is *known* to be a fixed level, so measure it on every trace and subtract it.
That's called clamping, and it removes most of the drift immediately.

### Step 6 — Handle the alternating stripes ⚡

**Every other stripe behaves differently, and this is the thing that breaks decoders.**

If you measure the gap between one sync burst and the next, you get ~3100 samples, then
~3300, then ~3100, then ~3300. That alternation is why the 2017 decode had a persistent
"jitter" it fixed with a hardcoded fudge.

The cause turns out to be documented. The 1977 scan converter — Colorado Video's Model 262 —
**deliberately used two different sync-pulse widths, so the receiver could tell which
interlace field a line belonged to.** It's a feature, not a fault. On our recording, the
even-trace pulse goes high about 155 samples before it falls; the odd-trace pulse only about
45 samples before. But both fall through zero at the *same* instant.

That last detail is the whole trick:

| What you lock onto | How much it wanders |
|---|---|
| the peak of the pulse *(what most decoders use)* | **±100 samples** |
| the trough after the peak | ±7 |
| **the falling edge, where it crosses zero** | **±6** |

Lock to the falling edge and your timing error drops by a factor of ~16, because the edge
doesn't care how wide the pulse was. This single change is the difference between a
photograph decoding cleanly and decoding with a diagonal staircase torn across it.

It also means **one template can't match both stripe types** — they're genuinely different
shapes. Use two, one per parity, or match only the falling edge that they share.

> **A correction.** An earlier version of this document claimed the alternation was 60 Hz
> mains hum locked to the 120 Hz scan, at about 70% of the picture signal. That was wrong on
> both counts and it's worth saying why, because the mistake is instructive.
>
> The evidence was a peak at "exactly half the stripe rate". But over one 512-stripe image,
> "mains hum at 60.000 Hz" and "something locked to the scan" sit only a fifth of a
> measurement bin apart — the test could never have separated them. Measuring the *whole*
> record instead gives 25× finer resolution, and the answer is 60.0436 Hz: essentially exactly
> half the scan rate, and clearly *not* 60.000 Hz mains. And the "70%" was measured inside the
> sync burst, which was leaking into the picture because of a coordinate bug. In the picture
> itself the effect is 5–30%.
>
> The lesson generalises: if two explanations predict nearly the same number, measuring more
> carefully beats arguing about which is more plausible.

### Step 7 — Set the levels, and check the circle

Stretch the brightness range, then decode image 1 and look at it. **If the circle is round,
you got it right.** If it's an oval, your trace length or picture window is off, and the
amount it's squashed by tells you by how much.

For the 20 colour photographs, the same scene was scanned three times through red, green
and blue filters and stored as three consecutive images. Decode all three, line them up,
and stack them into colour.

---

## 4. Expect the recording to be damaged, and say so

This is a single digitisation of an analog artefact from 1977. It has the flaws analog
recordings always have, and a decoder should measure each one rather than pretend it isn't
there:

| Defect | What it does | What to do |
|---|---|---|
| 60 Hz hum | fixed odd/even stripe pattern | subtract exactly (step 6) |
| wow and flutter | wavy, wobbling edges | time every trace separately |
| AC-coupling droop | bright top, dark bottom, trailing shadows | clamp on the sync shelf |
| dropouts | a stripe briefly loses signal | detect and flag; fill from neighbours |
| clicks and ticks | single bright/dark specks | median-filter the *audio*, before making pixels — a click is a few samples wide there but smears across a whole pixel afterwards |

The important discipline: when something can't be repaired, **report it instead of hiding
it**. Our decoder scores every trace for confidence, so the interface can show you which
parts of a picture are trustworthy and which are guesses. A restoration that quietly
invents detail is worse than one that admits uncertainty.

---

## 5. Honest limits

Things that are measured and solid:

- trace period 8.326 ms; sync timing to ~6 samples RMS, 65× better than peak-picking
- active picture window 2873 samples, which makes the calibration circle round with no fudge
- 60 Hz hum locked at exactly half the trace rate, ~70% of signal amplitude, removable
- the quiet gap between images is a framing signal at exactly twice the trace period — not
  a sawtooth, and not, as far as we can tell, hidden data

Things still open, stated plainly:

- **The true sharpness limit is not yet known.** Measuring it is harder than it sounds,
  because a careless measurement measures your own timing error instead of the record's. My
  first attempt did exactly that and produced a nonsense answer, which I threw away.
- The images are *not* quantised to the 16 grey levels the cover mentions — that appears to
  be guidance on dynamic range, not literal steps.
- A few frames still need their start position re-detected rather than inherited from the
  2017 hand-tuned table.

---

## 6. Why the circle is the best part

The engineers who cut this record in 1977 could not send a manual, a language, or a unit of
measurement. So they sent a circle — and trusted that anyone who found it would understand
that a circle come out oval means *you are doing it wrong, and here is exactly how wrong*.

Forty-nine years later, that's still the check that tells you your decoder works.

---

*Source audio: 384 kHz digitisation of the Golden Record master tapes, from David Pescovitz
via Ron Barry, [archived at the Internet Archive](https://archive.org/details/voyager.decode).
Prior art and the starting point for this work: Ron Barry's 2017 decoder,
[github.com/foodini/voyager](https://github.com/foodini/voyager).*
