"""Build the web asset set from the 384 kHz master.

For every one of the 156 frames:

  1. read a window of the correct channel around Barry's hand-tuned seed,
  2. decimate the whole window 384 kHz -> 96 kHz (exact /4),
  3. re-detect the true trace-0 position and line period with sync.recover(),
  4. cut at the *detected* start, keeping LEAD_IN samples of run-up, and
     normalise to int16 recording the pre-normalisation peak,
  5. write mono 16-bit FLAC and verify it round-trips sample for sample,
  6. decode the emitted asset -- not the master -- to a PNG, so the thumbnail
     proves the thing we shipped is decodable.

Decimating before cutting makes the cut a slice, so the shipped file is never
resampled a second time and trace 0 lands within half a sample of `leadIn`.

Then emit catalog.json per the web data contract, plus a contact sheet of every
frame for the later visual-identification pass.

Usage:
    python -m pipeline.build --limit 6 --jobs 4
    python -m pipeline.build --frames L000,R012
    python -m pipeline.build --jobs 8            # the full 156
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import resample_poly

from . import catalog as catalog_mod
from . import decode as decode_mod
from . import quality as quality_mod
from . import sync as sync_mod
from . import wav

REPO = Path(__file__).resolve().parent.parent
MASTER = REPO / "data" / "master" / "384kHzStereo.wav"
WEB_DATA = REPO / "web" / "public" / "data"
THUMB_DIR = REPO / "data" / "thumbs"
SHEET_PATH = REPO / "data" / "contact_sheet.png"
REPORT_PATH = REPO / "data" / "build_report.json"

MASTER_RATE = 384_000
TARGET_RATE = 96_000
DECIM = MASTER_RATE // TARGET_RATE  # exact integer decimation, 4
NOMINAL_96 = sync_mod.NOMINAL_PERIOD / DECIM  # 799.35
TRACES = sync_mod.TRACES_PER_FRAME  # 512

# Samples of run-up kept in the emitted file before trace 0. It has to exceed
# one line period so the consumer's own sync.recover() has a whole line to lock
# onto before the picture starts; 2.5 lines is comfortable and costs 20 ms.
LEAD_IN = 2000  # 96 kHz samples
TAIL_TRACES = 4  # whole lines kept past the last trace start

# sync.recover() takes "trace 0" to be the first blanking burst that sits about
# one period into the array it is given. Starting the read 1.5 nominal periods
# before Barry's seed therefore makes its trace 0 the true trace start *nearest*
# the seed, so the delta we report is a genuine phase error in [-P/2, +P/2)
# rather than an arbitrary whole-line offset.
PRE_384 = int(round(1.5 * sync_mod.NOMINAL_PERIOD))

INT16_FULL = 32767.0

# Per-line measurement noise above which a frame's timebase is not to be
# trusted, so the run flags it instead of quietly shipping a soft period seed.
# Clean frames sit near 0.6 samples at 96 kHz. An earlier build flagged 117 of
# the 156 frames here -- all the photographs, none of the line-art diagrams --
# which turned out to be sync.py anchoring its template on the notch inside the
# picture region rather than on the blanking burst. That is fixed in sync.py
# now; the threshold stays as the tripwire that caught it.
NOISE_LIMIT = 5.0  # samples at 96 kHz

# Barry's orientation table is in quarter turns, but his renderer's rotation
# direction is not recoverable from the tables alone.
# TODO(unverified): direction is assumed clockwise. Confirm against Barry's
# published renders during the visual-identification pass and flip if wrong.
ROT_CLOCKWISE = True


@dataclass(frozen=True)
class Options:
    master: Path
    out_dir: Path
    thumb_dir: Path
    traces: int
    compression: int
    full_rate_check: bool
    thumbs: bool
    verify: bool


# --------------------------------------------------------------------------
# master access (one memmap per process)
# --------------------------------------------------------------------------

_MASTER_CACHE: dict[Path, tuple[wav.WavInfo, np.ndarray]] = {}


def _master(path: Path) -> tuple[wav.WavInfo, np.ndarray]:
    hit = _MASTER_CACHE.get(path)
    if hit is None:
        info = wav.probe(path)
        if info.sample_rate != MASTER_RATE:
            raise ValueError(f"{path}: expected {MASTER_RATE} Hz, got {info.sample_rate}")
        if info.channels != 2:
            raise ValueError(f"{path}: expected 2 channels, got {info.channels}")
        hit = (info, wav.memmap(info))
        _MASTER_CACHE[path] = hit
    return hit


# --------------------------------------------------------------------------
# FLAC
# --------------------------------------------------------------------------


def write_flac(path: Path, pcm: np.ndarray, rate: int, compression: int) -> None:
    """Encode int16 mono PCM through the flac CLI. -V makes it verify itself."""
    if pcm.dtype != np.int16:
        raise TypeError(f"expected int16, got {pcm.dtype}")
    cmd = [
        "flac", "--silent", "--force", "--verify", f"-{compression}",
        "--force-raw-format", "--endian=little", "--sign=signed",
        "--channels=1", "--bps=16", f"--sample-rate={rate}",
        "-o", str(path), "-",
    ]
    proc = subprocess.run(cmd, input=pcm.tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"flac failed on {path.name}: {proc.stderr.decode(errors='replace')}")


def read_flac(path: Path) -> np.ndarray:
    cmd = [
        "flac", "--silent", "--decode", "--stdout",
        "--force-raw-format", "--endian=little", "--sign=signed", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"flac -d failed on {path.name}: {proc.stderr.decode(errors='replace')}")
    return np.frombuffer(proc.stdout, dtype="<i2")


# --------------------------------------------------------------------------
# timebase alignment
# --------------------------------------------------------------------------


def alternation(tb: sync_mod.Timebase) -> float:
    """Amplitude of the odd/even line-position alternation, in samples.

    If the sync region is not identical on odd and even traces, a template
    folded over both correlates slightly early on one parity and slightly late
    on the other, and it shows up as a square wave in the fit residuals. This is
    the artefact Barry pinned down with a hardcoded "+/-12 samples on even
    traces, changing at trace 164", which is 3 samples at 96 kHz; we measure it
    per frame instead. Parity is relative to the timebase's own trace 0.
    """
    r = tb.residuals
    if len(r) < 4:
        return 0.0
    return float(np.mean(r[0::2]) - np.mean(r[1::2]))


def fitted_trace_near(tb: sync_mod.Timebase, near: float) -> float:
    """Fitted start of the trace closest to `near`.

    The straight-line fit, not the smoothed measurement: it averages over all
    512 traces, so it is immune both to the odd/even alternation and to the
    Savitzky-Golay edge extrapolation, which together move a single measured
    position by tens of samples on the noisier frames.
    """
    k = round((near - tb.phase) / tb.period)
    return tb.phase + k * tb.period


def align_timebase(tb: sync_mod.Timebase, lead_in: float, n_traces: int) -> sync_mod.Timebase:
    """Re-index a recovered timebase so its trace 0 is the trace at `lead_in`.

    sync.recover() locks onto whichever notch sits about one period into the
    array, which for our assets is a line or two before the frame's real first
    trace. The emitted file's `leadIn` is what pins the frame down, so we shift
    the measured positions to match. The browser decoder has to do exactly this.
    """
    if len(tb.smoothed) == 0:
        raise ValueError("empty timebase")
    k = int(np.argmin(np.abs(tb.smoothed - lead_in)))
    off = float(tb.smoothed[k] - lead_in)
    if abs(off) > 0.5 * tb.period:
        raise ValueError(f"no trace within half a period of leadIn={lead_in} (nearest {off:+.1f})")
    if len(tb.smoothed) - k < n_traces:
        raise ValueError(f"only {len(tb.smoothed) - k} traces after leadIn, need {n_traces}")
    extra = {}
    if getattr(tb, "located", np.zeros(0)).size:  # sync v2 carries per-trace flags
        extra["located"] = tb.located[k:]
        # parity offset is keyed to trace-index parity; an odd shift flips it
        extra["parity_offset"] = tb.parity_offset * (-1 if k % 2 else 1)
    return dataclasses.replace(
        tb,
        phase=tb.phase + k * tb.period,
        positions=tb.positions[k:],
        smoothed=tb.smoothed[k:],
        residuals=tb.residuals[k:],
        n_traces=len(tb.smoothed) - k,
        **extra,
    )


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def orient(img: np.ndarray, quarter_turns: int) -> np.ndarray:
    """Apply Barry's orientation. See ROT_CLOCKWISE for the open question."""
    k = quarter_turns % 4
    if k == 0:
        return img
    return np.rot90(img, k=-k if ROT_CLOCKWISE else k)


def to_png(img: np.ndarray, path: Path) -> None:
    """Save a float image in [0, 1] as 8-bit grayscale."""
    arr = np.clip(img, 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="L").save(path, optimize=True)


# --------------------------------------------------------------------------
# per-frame build
# --------------------------------------------------------------------------


def process_frame(frame: catalog_mod.Frame, opts: Options) -> dict:
    """Build one frame's assets. Returns its report row + catalog metadata."""
    t0 = time.time()
    info, mm = _master(opts.master)
    n_traces = opts.traces

    # One read covers both the sync-recovery window and the emitted cut, which
    # can start up to half a period before the seed plus the whole lead-in.
    head = LEAD_IN * DECIM + PRE_384
    span = int(math.ceil((n_traces + TAIL_TRACES + 2) * sync_mod.NOMINAL_PERIOD))
    wide_start = frame.seed_sample - head
    if wide_start < 0:
        raise ValueError(f"{frame.id}: seed {frame.seed_sample} is too close to the file start")
    wide = np.asarray(
        wav.read(info, frame.channel, wide_start, head + span, mm=mm), dtype=np.float64
    )
    if len(wide) < head + span:
        raise ValueError(f"{frame.id}: master is short: got {len(wide)} of {head + span} samples")

    # --- 1. optional full-rate cross-check ----------------------------------
    # Off by default. sync.estimate_period() autocorrelates a ~80-trace segment
    # with np.correlate(mode="full"), which is O(n^2) in the segment length, so
    # recovering at 384 kHz costs 16x what the decimated recovery costs for a
    # measurement we do not use: decimation removes everything above 48 kHz,
    # which is noise as far as the sync edges are concerned.
    detected384 = float("nan")
    tb384 = None
    if opts.full_rate_check:
        tb384 = sync_mod.recover(
            wide[LEAD_IN * DECIM :],  # starts PRE_384 before the seed
            period_guess=sync_mod.NOMINAL_PERIOD,
            n_traces=n_traces,
        )
        detected384 = wide_start + LEAD_IN * DECIM + tb384.phase

    # --- 2. decimate the whole window, then detect on it --------------------
    # Decimating first lets us cut by slicing, so the emitted file needs no
    # second resampling pass and trace 0 lands within half a sample of leadIn.
    w96 = resample_poly(wide, 1, DECIM)
    tb96w = sync_mod.recover(
        w96[LEAD_IN:],  # PRE_384 / DECIM = 1.5 nominal periods before the seed
        period_guess=NOMINAL_96,
        n_traces=n_traces,
    )
    pos96 = LEAD_IN + tb96w.phase  # trace 0, in w96 coordinates
    detected = wide_start + DECIM * pos96
    delta = detected - frame.seed_sample
    period96 = tb96w.period

    # --- 3. cut and normalise ----------------------------------------------
    n_out = LEAD_IN + int(math.ceil((n_traces + TAIL_TRACES) * period96))
    cut = int(round(pos96)) - LEAD_IN
    if cut < 0 or cut + n_out > len(w96):
        raise ValueError(f"{frame.id}: cut window [{cut}, {cut + n_out}) escapes the read")
    y96 = w96[cut : cut + n_out]

    peak = float(np.max(np.abs(y96)))
    if peak <= 0.0:
        raise ValueError(f"{frame.id}: window is silent")
    pcm = np.clip(np.round(y96 / peak * INT16_FULL), -INT16_FULL, INT16_FULL).astype(np.int16)

    # --- 4. write and verify the FLAC ---------------------------------------
    frames_dir = opts.out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    flac_path = frames_dir / f"{frame.id}.flac"
    write_flac(flac_path, pcm, TARGET_RATE, opts.compression)

    back = pcm
    if opts.verify:
        back = read_flac(flac_path)
        if len(back) != len(pcm) or not np.array_equal(back, pcm):
            n_bad = -1 if len(back) != len(pcm) else int(np.count_nonzero(back != pcm))
            raise RuntimeError(
                f"FLAC ROUND-TRIP FAILURE on {frame.id}: wrote {len(pcm)} samples, "
                f"read back {len(back)}, {n_bad} differing samples"
            )

    # --- 5. decode the emitted asset ---------------------------------------
    # Everything from here runs on what came back out of the FLAC, at the rate
    # and word length the browser sees, so the thumbnail proves the shipped
    # asset decodes rather than proving the master does.
    xs = back.astype(np.float32) / INT16_FULL * peak  # restore absolute level
    trace0_at = float("nan")
    thumb_bytes = 0
    tau = 0.0
    quality = float("nan")
    reread_period = float("nan")
    composite = float("nan")
    qmetrics: dict = {}
    diag: dict = {}
    if opts.thumbs or opts.verify:
        tb96 = sync_mod.recover(xs, period_guess=period96, n_traces=n_traces + TAIL_TRACES)
        reread_period = tb96.period
        # The shipped file's own answer to "where is trace 0?", which is the
        # question the browser asks. Want: LEAD_IN, to well under a sample.
        trace0_at = fitted_trace_near(tb96, LEAD_IN)
        tb96 = align_timebase(tb96, LEAD_IN, n_traces)

        if opts.thumbs:
            # frame.id[0] is the audio channel letter: it selects the decay-
            # bias time constant (decode.UNCOUPLE_TAU_384), which is per
            # channel because the two channels' poles differ by 1.8x.
            cfg = decode_mod.Settings(traces=n_traces, rotate=frame.orientation,
                                      channel=frame.id[0])
            dec = decode_mod.decode(xs, cfg, tb96)
            tau = dec.tau
            quality = dec.quality.score
            # quality.py composite on the decode the browser will reproduce,
            # scored pre-orientation (its axes are height x traces) and WITH
            # the timebase so the sync-lock factor cannot be dodged.
            fs = quality_mod.frame_report(dec.image, tb96, frame.id)
            composite = fs.composite
            qmetrics = {
                k: (round(float(v), 4) if np.isfinite(v) else None)
                for k, v in fs.metrics.items()
                if k in ("shift_rms", "drift_span", "step_frac", "stair_support",
                         "parity_db", "coh_ratio", "accept_frac", "sharpness")
            }
            d = dec.diagnostics
            diag = {
                "dot_locked": bool(d.get("dot_locked")),
                "dot_clock_strength": (round(float(d["dot_clock_strength"]), 3)
                                       if d.get("dot_clock_strength") is not None else None),
                "dot_phase_coherence": round(float(d.get("dot_phase_coherence", 0.0)), 3),
                "levels_mode": d.get("levels_mode"),
                "snr_db": round(float(d.get("snr_db", float("nan"))), 2),
                "dropouts": int(d.get("dropouts", 0)),
            }
            opts.thumb_dir.mkdir(parents=True, exist_ok=True)
            thumb_path = opts.thumb_dir / f"{frame.id}.png"
            to_png(orient(dec.image, cfg.rotate), thumb_path)
            thumb_bytes = thumb_path.stat().st_size

    return {
        "id": frame.id,
        "seed": frame.seed_sample,
        "detected": round(detected, 3),
        "delta": round(delta, 3),
        "detected384": round(detected384, 3),
        "delta384": round(detected384 - frame.seed_sample, 3),
        "period96": round(period96, 5),
        "period384_equiv": round(tb384.period, 4) if tb384 else float("nan"),
        "period96_reread": round(reread_period, 5),
        "jitter_rms96": round(tb96w.jitter_rms, 3),
        "measurement_noise96": round(tb96w.measurement_noise, 3),
        "alternation96": round(alternation(tb96w), 3),
        "templatePP": round(float(tb96w.template.max() - tb96w.template.min()), 4),
        "lowConfidence": bool(tb96w.measurement_noise > NOISE_LIMIT),
        "quality": round(quality, 4),
        "composite": round(composite, 2) if math.isfinite(composite) else None,
        "qmetrics": qmetrics,
        "decode": diag,
        "jitter_rms384": round(tb384.jitter_rms, 3) if tb384 else float("nan"),
        "measurement_noise384": round(tb384.measurement_noise, 3) if tb384 else float("nan"),
        "traces_located": tb96w.n_traces,
        "trace0_offset96": round(trace0_at, 3),  # should be LEAD_IN to within half a sample
        "peak": round(peak, 6),
        "tau_samples": round(tau, 1),
        "samples96": int(len(pcm)),
        "flac_bytes": flac_path.stat().st_size,
        "thumb_bytes": thumb_bytes,
        "seconds": round(time.time() - t0, 2),
        # catalog.json fields
        "meta": {
            "file": f"frames/{frame.id}.flac",
            "leadIn": LEAD_IN,
            "startSample": int(round(detected)),
            "periodGuess": round(period96, 5),
            "durationSamples": int(len(pcm)),
            "peak": round(peak, 6),
        },
    }


def _worker(args: tuple[catalog_mod.Frame, Options]) -> dict:
    frame, opts = args
    try:
        return process_frame(frame, opts)
    except Exception as exc:  # keep one bad frame from killing the whole run
        return {"id": frame.id, "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------
# contact sheet
# --------------------------------------------------------------------------

SHEET_COLS = 13
TILE = 200
LABEL_H = 16
PAD = 4


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def contact_sheet(cat: catalog_mod.Catalog, thumb_dir: Path, out_path: Path) -> int:
    """Tile every thumbnail on disk into one labelled PNG. Returns tile count."""
    tiles = [(f, thumb_dir / f"{f.id}.png") for f in cat.frames]
    tiles = [(f, p) for f, p in tiles if p.exists()]
    if not tiles:
        return 0

    cols = min(SHEET_COLS, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    cw, ch = TILE + 2 * PAD, TILE + LABEL_H + 2 * PAD
    sheet = Image.new("RGB", (cols * cw, rows * ch), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    font = _font(13)

    for i, (frame, path) in enumerate(tiles):
        x, y = (i % cols) * cw, (i // cols) * ch
        im = Image.open(path).convert("L")
        im.thumbnail((TILE, TILE), Image.LANCZOS)
        sheet.paste(im.convert("RGB"), (x + PAD + (TILE - im.width) // 2,
                                        y + PAD + (TILE - im.height) // 2))
        role = "" if frame.color == "mono" else f"  {frame.color.upper()}{frame.triplet_id[1:]}"
        colour = {"mono": (210, 210, 210), "r": (240, 130, 130),
                  "g": (130, 220, 140), "b": (140, 170, 245)}[frame.color]
        draw.text((x + PAD, y + PAD + TILE + 2), f"{frame.id}{role}", fill=colour, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, optimize=True)
    return len(tiles)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def print_report(rows: list[dict]) -> None:
    ok = [r for r in rows if "error" not in r]
    bad = [r for r in rows if "error" in r]

    hdr = (f"{'id':6}{'seed':>11} {'detected':>13} {'delta':>9} {'period96':>9} "
           f"{'jit96':>6} {'noise':>6} {'alt96':>6} {'trace0':>9} {'qual':>5} {'comp':>6} "
           f"{'dot':>4} {'peak':>7} {'KiB':>6} {'s':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in ok:
        flag = "!" if r["lowConfidence"] else " "
        comp = r.get("composite")
        dot = r.get("decode", {}).get("dot_locked")
        print(f"{r['id']:5}{flag}{r['seed']:>11} {r['detected']:>13.1f} {r['delta']:>+9.1f} "
              f"{r['period96']:>9.3f} {r['jitter_rms96']:>6.2f} "
              f"{r['measurement_noise96']:>6.2f} {r['alternation96']:>+6.1f} "
              f"{r['trace0_offset96']:>9.3f} {r['quality']:>5.2f} "
              f"{comp if comp is not None else float('nan'):>6.1f} "
              f"{('yes' if dot else ' no') if dot is not None else '  ?':>4} "
              f"{r['peak']:>7.4f} {r['flac_bytes'] / 1024:>6.0f} {r['seconds']:>5.1f}")
    for r in bad:
        print(f"{r['id']:5} ERROR {r['error']}")
    shaky = [r["id"] for r in ok if r["lowConfidence"]]
    if shaky:
        print(f"low-confidence timebase (measurement noise > {NOISE_LIMIT} samples at 96 kHz): "
              f"{len(shaky)} frames {' '.join(shaky)}")

    if not ok:
        return

    signed = np.array([r["delta"] for r in ok])
    d = np.abs(signed)
    per = np.array([r["period96"] for r in ok]) * DECIM
    off = np.abs(np.array([r["trace0_offset96"] for r in ok]) - LEAD_IN)
    agree = np.abs(np.array([r["delta384"] for r in ok]) - signed)
    total = sum(r["flac_bytes"] for r in ok)
    print()
    print(f"frames ok {len(ok)}   failed {len(bad)}")
    print(f"detected - Barry seed, in 384 kHz samples (1 line = {sync_mod.NOMINAL_PERIOD}):")
    print(f"    signed: mean {signed.mean():+8.1f}   sd {signed.std():7.1f}   "
          f"min {signed.min():+8.1f}   max {signed.max():+8.1f}")
    print(f"    |delta|: median {np.median(d):7.1f}   max {d.max():7.1f}   "
          f"rms {np.sqrt((d ** 2).mean()):7.1f}   ({100 * np.median(d) / sync_mod.NOMINAL_PERIOD:.1f}% of a line)")
    # The bulk of the offset is a convention difference, not error: sync.py
    # calls the leading edge of the blanking burst the trace start, and Barry's
    # seeds sit a tenth of a line from it, consistently. A constant offset moves
    # every trace of every frame alike and costs nothing. What is left after
    # removing it is his hand-tuning scatter -- what the seeds actually cost us.
    wobble = np.abs(signed - np.median(signed))
    print("    after removing the median offset (i.e. Barry's hand-tuning scatter):")
    print(f"        median {np.median(wobble):7.1f}   max {wobble.max():7.1f}   "
          f"rms {np.sqrt((wobble ** 2).mean()):7.1f}")
    for thr in (8, 50, 200, 800, 1599):
        n = int((wobble <= thr).sum())
        print(f"        <= {thr:4} samples: {n:3}/{len(ok)}  ({100 * n / len(ok):.0f}%)")
    alt = np.abs(np.array([r["alternation96"] for r in ok]))
    if np.isfinite(agree).any():
        print(f"384 kHz vs 96 kHz detection disagree by {np.nanmean(agree):.1f} samples mean, "
              f"{np.nanmax(agree):.1f} max (384 kHz samples)")
    print(f"odd/even trace alternation (Barry's hardcoded fudge): median {np.median(alt):.1f}, "
          f"max {alt.max():.1f} samples at 96 kHz")
    print(f"fitted line period x{DECIM}: mean {per.mean():.3f}  sd {per.std():.3f}  "
          f"min {per.min():.3f}  max {per.max():.3f}  (nominal {sync_mod.NOMINAL_PERIOD})")
    good = np.array([not r["lowConfidence"] for r in ok])
    print(f"trace 0 lands at leadIn +- {off[good].max() if good.any() else float('nan'):.2f} samples "
          f"(96 kHz) worst case over the {int(good.sum())} confident frames"
          + (f", {off.max():.2f} including the {int((~good).sum())} flagged" if not good.all() else ""))
    print(f"flac total {total / 1e6:.1f} MB, mean {total / len(ok) / 1024:.0f} KiB/frame")
    comps = np.array([r["composite"] for r in ok
                      if r.get("composite") is not None], dtype=np.float64)
    if len(comps):
        print(f"quality.py composite over {len(comps)} scored frames: "
              f"mean {comps.mean():.1f}  median {np.median(comps):.1f}  "
              f"min {comps.min():.1f}  max {comps.max():.1f}")
    locks = [r.get("decode", {}).get("dot_locked") for r in ok]
    known = [v for v in locks if v is not None]
    if known:
        print(f"dot clock locked on {sum(known)}/{len(known)} frames "
              f"(fallback binning on {len(known) - sum(known)})")


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", type=Path, default=MASTER)
    ap.add_argument("--out", type=Path, default=WEB_DATA, help="web/public/data")
    ap.add_argument("--thumbs-dir", type=Path, default=THUMB_DIR)
    ap.add_argument("--sheet", type=Path, default=SHEET_PATH)
    ap.add_argument("--report", type=Path, default=REPORT_PATH)
    ap.add_argument("--limit", type=int, default=0, help="build only the first N frames")
    ap.add_argument("--frames", default="", help="comma-separated frame ids, e.g. L000,R012")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--traces", type=int, default=TRACES)
    ap.add_argument("--full-rate-check", action="store_true",
                    help="also recover the timebase at 384 kHz and report the disagreement")
    ap.add_argument("--compression", type=int, default=8, help="flac -0..-8")
    ap.add_argument("--no-thumbs", action="store_true")
    ap.add_argument("--no-verify", action="store_true", help="skip the FLAC read-back compare")
    ap.add_argument("--no-sheet", action="store_true")
    ap.add_argument("--sheet-only", action="store_true", help="rebuild the contact sheet and stop")
    args = ap.parse_args(argv)

    cat = catalog_mod.build()
    print(f"catalog: {len(cat.frames)} frames, {len(cat.images)} images "
          f"({sum(1 for i in cat.images if i.color)} colour)")

    if args.sheet_only:
        n = contact_sheet(cat, args.thumbs_dir, args.sheet)
        print(f"contact sheet: {n} tiles -> {args.sheet} ({args.sheet.stat().st_size / 1e6:.1f} MB)")
        return 0

    selected = list(cat.frames)
    if args.frames:
        want = [s.strip() for s in args.frames.split(",") if s.strip()]
        index = {f.id: f for f in cat.frames}
        missing = [w for w in want if w not in index]
        if missing:
            print(f"unknown frame ids: {missing}", file=sys.stderr)
            return 2
        selected = [index[w] for w in want]
    if args.limit:
        selected = selected[: args.limit]

    opts = Options(
        master=args.master.resolve(),
        out_dir=args.out.resolve(),
        thumb_dir=args.thumbs_dir.resolve(),
        traces=args.traces,
        compression=args.compression,
        full_rate_check=args.full_rate_check,
        thumbs=not args.no_thumbs,
        verify=not args.no_verify,
    )
    (opts.out_dir / "frames").mkdir(parents=True, exist_ok=True)

    print(f"building {len(selected)} frames with {args.jobs} job(s)")
    t0 = time.time()
    work = [(f, opts) for f in selected]
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(_worker, work))
    else:
        rows = [_worker(w) for w in work]
    elapsed = time.time() - t0

    print_report(rows)
    print(f"wall clock {elapsed:.1f} s")

    ok = [r for r in rows if "error" not in r]
    frame_meta = {r["id"]: r["meta"] for r in ok}
    doc = catalog_mod.catalog_json(
        cat,
        sample_rate=TARGET_RATE,
        nominal_period=NOMINAL_96,
        traces_per_frame=args.traces,
        frame_meta=frame_meta,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    cat_path = args.out / "catalog.json"
    cat_path.write_text(json.dumps(doc, indent=1))
    partial = len(doc["frames"]) != len(cat.frames)
    print(f"{'PARTIAL ' if partial else ''}catalog.json: {len(doc['frames'])} frames, "
          f"{len(doc['images'])} images -> {cat_path} ({cat_path.stat().st_size / 1024:.0f} KiB)")
    if partial:
        print("  !! this catalog describes only the frames built in this run")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"leadIn": LEAD_IN, "rows": rows}, indent=1))
    print(f"build report -> {args.report}")

    if not args.no_sheet and opts.thumbs:
        n = contact_sheet(cat, opts.thumb_dir, args.sheet)
        if n:
            print(f"contact sheet: {n} tiles -> {args.sheet} "
                  f"({args.sheet.stat().st_size / 1e6:.1f} MB)")

    failed = [r for r in rows if "error" in r]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
