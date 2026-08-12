# pipeline

Turns the 1.4 GB, 384 kHz digitisation of the Voyager Golden Record's image
track into the small asset set the browser decoder fetches. Nothing here runs at
request time; a visitor gets a ~530 KB FLAC slice per frame and does the DSP
themselves.

```
fetch.py       download the master (once)
wav.py         memory-mapped RIFF reader -- the master is never loaded whole
sync.py        timebase recovery: where every trace begins, to a fraction of a sample
decode.py      one frame of audio -> one picture. The reference implementation.
catalog.py     Barry's tables -> the 156-frame / 116-image catalog
build.py       master -> web/public/data/{catalog.json, frames/*.flac} + thumbnails
```

`catalog.py` and `build.py` are described below. For the signal itself, read the
module docstrings in `sync.py` and `decode.py` -- they are the authority, and
this file does not restate their numbers.

## Running the build

```bash
python -m pipeline.build --limit 6 --jobs 3      # fast smoke test, 6 frames
python -m pipeline.build --frames L000,R012      # named frames
python -m pipeline.build --jobs 6                # the full 156
python -m pipeline.build --sheet-only            # re-tile existing thumbnails
```

Requires `numpy`, `scipy`, `Pillow`, and the `flac` CLI on `$PATH`.

Roughly 6 s per frame, dominated by two calls to `sync.recover()`; the full set
takes a few minutes at `--jobs 6`. Output:

| path | what |
| --- | --- |
| `web/public/data/catalog.json` | the data contract, ~60 KiB for the full set |
| `web/public/data/frames/<id>.flac` | mono 96 kHz 16-bit, ~530 KiB each, ~85 MB total |
| `data/thumbs/<id>.png` | one decoded frame each, for identification |
| `data/contact_sheet.png` | all 156 tiled and labelled, ~4.5 MB |
| `data/build_report.json` | every per-frame measurement listed below |

A partial run (`--limit`, `--frames`) writes a catalog describing **only** the
frames it built, and says so loudly. Colour images are emitted only when all
three of their separations are present.

## catalog.py

Reads `barry_tables.json` and produces the frame list: ids (`L000`..`L077`,
`R000`..`R077`), channel, colour role, orientation, and triplet grouping. A run
of consecutive non-mono frames on one channel chunks into triplets, each of
which must be exactly `r`, `g`, `b` in order; runs of three, six and nine all
occur. Every grouping decision is checked against the counts, and `build()`
raises rather than return anything that is not **156 frames / 116 images**
(96 mono + 20 colour). That invariant is the correctness check on the catalog.

`imageNumber` and `title` are `null` here. Identifying which picture is which is
a later, visual step -- that is what the contact sheet is for.

## build.py

Per frame:

1. **Read** a window of the correct channel around Barry's seed, wide enough to
   cover both the recovery window and the emitted cut.
2. **Decimate** the whole window 384 -> 96 kHz with `resample_poly(1, 4)`.
   Decimating *before* cutting means the cut is a slice, so the emitted file
   needs no second resampling pass and trace 0 lands within half a sample of
   where the catalog says it does.
3. **Recover** the timebase with `sync.recover()` on the decimated window. The
   fitted phase -- the robust line fit over all 512 traces, not a single
   measured position -- is the frame's origin.
4. **Cut** at `origin - leadIn`, normalise to int16 recording the
   pre-normalisation peak, and write FLAC through the `flac` CLI with `-V`.
5. **Verify** by decoding the file back and comparing sample for sample. A
   mismatch raises; it is never repaired quietly.
6. **Decode** the round-tripped samples -- what the browser will see, not the
   master -- to a PNG thumbnail, and tile all of them into the contact sheet.

### Why the timebase is recovered at 96 kHz

Decimation removes everything above 48 kHz, which as far as the sync edges are
concerned is noise: the correlation in `sync.py` runs on the derivative of the
signal, and differentiating amplifies exactly the band decimation throws away.
The 384 kHz recovery is available behind `--full-rate-check` and is off by
default, because `sync.estimate_period()` autocorrelates with
`np.correlate(mode="full")`, which is O(n^2) in the segment length -- so
recovering at four times the rate costs sixteen times as much for a number we do
not use. (That O(n^2) is also why a frame takes ~6 s instead of ~0.3 s. An
FFT-based autocorrelation, or one evaluated only over the lags it actually
searches, would give the whole build back an order of magnitude. `sync.py` is
not ours to change.)

### Seed versus detected start

Barry's `start_points` are seeds only. Every frame start is re-detected from the
signal, and the run reports the difference. Over the first six frames:

```
signed: mean -324.8   sd 27.0   min -377.8   max -298.5     (384 kHz samples)
after removing the median offset:  median 18.0   max 60.4   rms 28.0
```

Two separate things are in that number:

* **A constant ~325-sample offset**, about a tenth of a line. This is a
  convention difference, not error: `sync.py` calls the leading edge of the
  blanking burst the trace start, and Barry's seeds land near a different
  feature of the same sync region. A constant offset costs nothing -- it moves
  every trace of every frame by the same amount.
* **The scatter around it**, tens of samples, which is what his hand-tuning
  actually cost. His table carries manual `+3151`-style fudges precisely because
  a seed that is a full line out has to be nudged by hand; re-detection removes
  the need for any of them, and the run prints how many frames land within 8,
  50, 200, 800 and 1599 samples once the common offset is removed.

The run also reports the odd/even alternation in the fit residuals, the artefact
Barry compensated with a hardcoded "+/-12 samples on even traces, changing at
trace 164". At 96 kHz his fudge is 3 samples; we measure ~1.

### Per-frame quality

`measurement_noise96` above `NOISE_LIMIT` flags a frame whose timebase is not
trustworthy, and flagged frames are listed by id. The flag earns its keep: an
earlier build flagged 117 of 156 frames -- every photograph, no line-art diagram
-- which turned out to be `sync.py` anchoring its matched filter on the notch
*inside* the picture region instead of on the blanking burst. That is fixed in
`sync.py`; the tripwire stays.

`trace0_offset96` is the strongest end-to-end check in the build: it is the
shipped file's own answer, after decimation, int16 quantisation and a FLAC round
trip, to "where is trace 0?". It should equal `leadIn`. It currently does, to
within 0.53 samples worst case.

### Known gaps

* **Rotation direction.** `ROT_CLOCKWISE` assumes Barry's orientation counts are
  clockwise quarter turns. That is not recoverable from his tables and is
  marked `TODO` in the source; confirm it during visual identification.
* **decode.py's geometry versus sync.py's trace-start convention.** `sync.py`
  now returns the blanking burst's leading edge, while `decode.py`'s
  `PICTURE_START` is still measured from the notch. Thumbnails therefore come
  out with the picture wrapped vertically. The assets themselves are unaffected
  -- the cut, `leadIn` and `periodGuess` are all independent of where the
  picture is deemed to start within a trace -- but the thumbnails and contact
  sheet will need regenerating once those two agree.
