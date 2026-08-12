# Golden Record: Decode

There are 116 photographs bolted to the side of two spacecraft leaving the solar system.
They aren't stored as pictures — they're stored as **sound**, about eight minutes of harsh
buzzing pressed into a gold-plated copper phonograph record.

This is a decoder that turns that sound back into pictures, in your browser, in real time.

**[Read how the decoding works →](DECODING.md)** (written for a general audience, no signal
processing assumed)

**[What an alien could not easily work out →](ALIENS.md)** — which parts of this a recipient
could actually recover, and where the record lets its reader down

<p align="center">
  <img src="docs/img/calibration-barry.png" width="45%" alt="Calibration circle decoded with the 2017 method: an ellipse">
  <img src="docs/img/calibration-restored.png" width="45%" alt="Calibration circle decoded with this decoder: round">
</p>

*Image 1 on the record is a circle — a self-checking test pattern, so recipients can tell
whether they've decoded it correctly. Left: the 2017 method, which produced an ellipse that
had to be corrected by hand. Right: this decoder, round with no fudge factor.*

---

## Standing on other people's shoulders

**This project exists because of [Ron Barry](https://github.com/foodini).** In 2017 he sat in
a talk at the Exploratorium, recognised the scan-line glyph on the record cover, and then spent
two weeks working out — deliberately, from the cover alone, refusing to look anything up —
how to get the pictures back. He wrote [`foodini/voyager`](https://github.com/foodini/voyager)
and an accompanying essay, *The Voyager Image Decoding Method*, which is the single best
document on this subject and which you should read.

He also did something rarer than solving the problem: he wrote down, precisely and without
defensiveness, everything that was still wrong with his result. His TODO list — the brightness
decay he couldn't explain, the "shadow and anti-shadow" around bright objects, the jitter he
fixed with what he called a "brainless heuristic", the colour channels off by a pixel, his
suspicion that there was more fidelity in the data than he was extracting — is the direct
roadmap for this repository. Nearly every improvement here is an answer to a question he
asked first and asked well.

His hand-tuned table of where each of the 156 frames begins, produced by zooming into
waveforms in Audacity one at a time, remains the only published index of the image track. This
decoder still uses it as its starting seed. It is in
[`pipeline/barry_tables.json`](pipeline/barry_tables.json), parsed directly from his source.

**[David Pescovitz](https://boingboing.net/author/david_pescovitz)** obtained access to the
master tapes while producing the Ozma Records 40th-anniversary vinyl edition of the Golden
Record (with Timothy Daly and Lawrence Azerrad), and gave Barry the high-fidelity audio that
makes any of this possible. Without that, everyone would still be decoding a lossy MP3. He
wrote up Barry's work at Boing Boing:
**[How to decode the images on the Voyager Golden Record](https://boingboing.net/2017/09/05/how-to-decode-the-images-on-th.html)**.

**The [Internet Archive](https://archive.org/details/voyager.decode)** hosts the 384 kHz
master digitisation, Barry's decoder, and his essay. That item is the source of every number
in this project.

**The people who made the thing in the first place**, in 1977: Carl Sagan, who chaired the
committee; Frank Drake, who devised the encoding and the cover notation; Ann Druyan, creative
director; Timothy Ferris, producer; Jon Lomberg, design director; and Linda Salzman Sagan.
And the engineers at **Colorado Video** in Boulder, who built the one-off machine that turned
each projected slide into a few seconds of audio. Their scan-conversion hardware is the thing
this repository is reverse-engineering, and it was built well.

**Others who have decoded this record**, whose work I read before starting — several of whom
solved parts of this independently and better than I would have:
[MalteGruber/voyager-record-decoder](https://github.com/MalteGruber/voyager-record-decoder)
(in-browser, and the first to note the digitisation's high-pass artefacts),
[MarcBaeuerle/Golden-record-images](https://github.com/MarcBaeuerle/Golden-record-images)
(real-time browser decoding; independently found the two-line sync structure),
[Aurélien Ginolhac's R walkthrough](https://ginolhac.github.io/posts/2024-07-26_decoding-golden-record/),
[amazing-rando/voyager-decoder](https://github.com/amazing-rando/voyager-decoder),
[aizquier/voyagerimb](https://github.com/aizquier/voyagerimb),
[tomswartz07/voyager](https://github.com/tomswartz07/voyager), and
[mmcc1/Voyager-Golden-Record-Decoder](https://github.com/mmcc1/Voyager-Golden-Record-Decoder).

---

## What this adds

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

**It also dissolves a 2017 mystery.** Barry hardcoded "−12 samples on even traces" and called it
a brainless heuristic. Twelve samples is *one dot*: one NTSC line period at the 2× tape speed.
It was never a fudge factor. It was the scan converter's own clock showing through, and he was
compensating for it without knowing what it was.

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

## How it works

Two implementations, deliberately:

- **`pipeline/`** — the Python reference decoder. Its jobs are to cut web assets from the
  master and to be the thing the browser decoder is held to pixel-parity against, so
  correctness is *demonstrated* rather than hoped for.
- **`web/`** — the real product. The browser fetches one ~540 KB FLAC slice per image and runs
  the whole decode in JavaScript: sync detection, timebase recovery, resampling, restoration.
  It is not a slideshow of pre-rendered PNGs.

Nobody downloads 1.4 GB. That's a one-time fetch for whoever builds the assets.

```
pipeline/
  wav.py                  memory-mapped reader for the 384 kHz master
  sync.py                 matched-filter timebase recovery
  decode.py               the reference decoder
  catalog.py              frame/image catalog, asserts 156 frames -> 116 images
  build.py                cuts FLAC slices, thumbnails, catalog.json
  extract_barry_tables.py parses the tables out of Barry's voyager.cpp
web/
  src/dsp/                the browser decoder (mirrors pipeline/decode.py)
  src/audio/              FLAC loading at a preserved 96 kHz, playback clock
  src/ui/, src/panels/    the instrument
```

## Building the assets

```bash
brew install flac ffmpeg
pip install numpy scipy pillow

# ~1.4 GB, one time
python -m pipeline.fetch

python -m pipeline.build --limit 6      # try a few frames first
python -m pipeline.build                # all 156

cd web && npm install && npm run dev
```

The master is at
[archive.org/details/voyager.decode](https://archive.org/details/voyager.decode) — note the
files live in a subdirectory with a space and an ampersand in its name, so the bare filename
404s.

## A note on what's in this repository

The decoded images and the audio slices are **not** committed. A majority of the 116
photographs on the record are third-party works licensed for the record itself, which is why
only a subset is publicly displayed anywhere, and the audio slices derive from a digitisation
of the master tapes. This repository ships the code and the measurements that reproduce them.

The calibration circle in `docs/img/` is committed as an exception: it is a geometric test
pattern, not a photograph.

## Status

Working: the reference decoder, the timebase, the hum removal, the geometry solve, the asset
pipeline, FLAC loading in the browser at a preserved 96 kHz.

In progress: the browser decoder port and its parity test, frame-start re-detection across all
156 frames (a few still inherit a bad seed and decode rolled), identifying which decoded frame
corresponds to which of the 116 catalogued images, and the interface.

## Licence

Code is [MIT](LICENSE).

The Voyager Golden Record and its contents are not mine to license. NASA material is generally
public domain; a significant portion of the photographs are not. The master-tape digitisation
is hosted by the Internet Archive courtesy of Ron Barry and David Pescovitz. Please respect
all of that.

---

*In 1977 a group of people had to explain how to read a photograph to someone with no shared
language, no shared units, and no shared mathematics beyond physics. So they engraved the
period of the hydrogen hyperfine transition on the cover, wrote every other number as a
multiple of it, and made the first image a circle — trusting that anyone who found it would
understand that a circle come out oval means "you are doing it wrong, and here is exactly how
wrong."*

*Forty-nine years later that's still the check that tells you your decoder works.*
