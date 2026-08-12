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

About 0.3 s per frame; the full 156 take 37 s at `--jobs 2`. Output:

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
default: it costs several times what the decimated recovery costs, for a
measurement the assets do not use.

### Seed versus detected start

Barry's `start_points` are seeds only. Every frame start is re-detected from the
signal, and the run reports the difference. Over all 156 frames, in 384 kHz
samples (one line = 3197.4):

```
signed:  mean -172.3   sd 229.1   median -149.4   min -1471.1   max +1282.1
after removing the median offset:  p50 31.7   p90 166.5   max 1431.5
    <=   8 samples:  27/156 (17%)      <=  200 samples: 146/156 (94%)
    <=  50 samples: 115/156 (74%)      <= 1599 samples: 156/156 (100%)
```

Three separate things are in that number:

* **A constant ~150-sample offset**, a twentieth of a line. This is a convention
  difference, not error: `sync.py` calls the leading edge of the blanking burst
  the trace start, and Barry's seeds land near a different feature of the same
  sync region. A constant offset costs nothing -- it moves every trace of every
  frame alike.
* **The scatter around it** -- 32 samples for the median frame, 167 at the 90th
  percentile -- which is what his hand-tuning actually cost. His table carries
  manual `+3151`-style fudges precisely because a seed that is a full line out
  has to be nudged by hand; re-detection removes the need for all of them.
* **Six frames** (`R061 L028 R039 R044 L023 R063`) sit 700-1500 samples out,
  a quarter to a half line. Five of those six are the frames the confidence
  check flags, so they are our uncertainty rather than his -- see below.

The run also reports the odd/even alternation in the fit residuals, the artefact
Barry compensated with a hardcoded "+/-12 samples on even traces, changing at
trace 164". At 96 kHz his fudge is 3 samples; we measure ~1.

### Per-frame quality

`measurement_noise96` above `NOISE_LIMIT` flags a frame whose timebase is not
trustworthy, and flagged frames are listed by id. Five of 156 are flagged today
(`L023 L028 R039 R044 R063`), and they are the same frames that carry the worst
seed deltas. The flag earns its keep: an earlier build flagged 117 of 156 --
every photograph, no line-art diagram -- which turned out to be `sync.py`
anchoring its matched filter on the notch *inside* the picture region instead of
on the blanking burst. That is fixed in `sync.py`; the tripwire stays.

`trace0_offset96` is the strongest end-to-end check in the build: it is the
shipped file's own answer, after decimation, int16 quantisation and a FLAC round
trip, to "where is trace 0?". It should equal `leadIn`, and for 149 of 156
frames it does to within 2 samples (median 0.45, p90 1.04). The exceptions are
`R044` (278), `R038` (64), `L023` (56), `R063` (13), `L028` (7): all but `R038`
are already flagged, so `R038` is the one frame where the confidence check is
too generous. None of these is fatal -- the error is a fraction of a line, so a
consumer that picks the trace nearest `leadIn` still picks the right one -- but
they are the frames to look at first if a picture comes out misframed.

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
