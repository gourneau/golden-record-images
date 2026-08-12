/* The Cover panel.
 *
 * A hand-authored SVG of the Voyager Golden Record cover, glyph by glyph, where
 * every glyph is a control: hover or focus it and the reader pane says what the
 * engraving tells you, shows the binary arithmetic worked out, and names the
 * exact thing our decoder does with it.
 *
 * The argument the panel is making: every parameter the decoder needs was
 * engraved on that plate in 1977, in a notation that assumes no shared language
 * and no shared units -- only that the reader knows what a hydrogen atom does.
 *
 * All geometry is computed here rather than hand-typed as path data, so the
 * drawing stays editable and the binary numbers are rendered from their actual
 * digit strings instead of being drawn as decoration.
 */

import './cover.css';

// --------------------------------------------------------------------------
// numbers
// --------------------------------------------------------------------------

/** Period of the 1420.405751768 MHz hydrogen hyperfine line, in seconds. */
const H_PERIOD = 1 / 1420405751.768; // 7.040241837624815e-10

/** The three long binary numbers engraved on the cover, as digit strings. */
const BIN_TRACE = '101101001100000000000000'; // 11845632 -> 8.3396 ms
const BIN_TRACES_PER_FRAME = '1000000000'; // 512
const BIN_ROTATION = '100110000110010000000000000000000'; // 5113380864 -> 3.5999 s
const BIN_SIDE = '1000010110000000000000000000000000000000000'; // 4587025072128 -> 3229.4 s

/** Our own measurements, from the project's established signal facts. */
const SAMPLE_RATE = 96000;
const NOMINAL_PERIOD = 799.35; // samples per trace at 96 kHz, measured
const TRACES_PER_FRAME = 512;

const binValue = (bits: string) => Number.parseInt(bits, 2);

// --------------------------------------------------------------------------
// SVG primitives
// --------------------------------------------------------------------------

const n = (v: number) => (Math.round(v * 100) / 100).toString();

function line(x1: number, y1: number, x2: number, y2: number, cls = 'ln'): string {
  return `<line class="${cls}" x1="${n(x1)}" y1="${n(y1)}" x2="${n(x2)}" y2="${n(y2)}"/>`;
}

function circle(cx: number, cy: number, r: number, cls = 'ln'): string {
  return `<circle class="${cls}" cx="${n(cx)}" cy="${n(cy)}" r="${n(r)}"/>`;
}

function rect(x: number, y: number, w: number, h: number, cls = 'ln', rx = 0): string {
  return `<rect class="${cls}" x="${n(x)}" y="${n(y)}" width="${n(w)}" height="${n(h)}" rx="${n(rx)}"/>`;
}

/** Binary digits in the cover's notation: a vertical bar is 1, a dash is 0.
 *  `dir` is the writing direction; bars are drawn across it, dashes along it,
 *  which is how the numbers are laid out beside the pulsar rays and around the
 *  rim of the record as well as in straight runs. */
function binaryRun(
  bits: string,
  x: number,
  y: number,
  opts: { pitch: number; height: number; dir?: number; cls?: string },
): string {
  const { pitch, height, dir = 0, cls = 'bit' } = opts;
  const a = (dir * Math.PI) / 180;
  const ux = Math.cos(a);
  const uy = Math.sin(a); // along the number
  const px = -uy;
  const py = ux; // across it
  const dash = pitch * 0.45;
  let out = '';
  for (let i = 0; i < bits.length; i++) {
    const cx = x + ux * i * pitch;
    const cy = y + uy * i * pitch;
    if (bits[i] === '1') {
      out += line(
        cx - (px * height) / 2,
        cy - (py * height) / 2,
        cx + (px * height) / 2,
        cy + (py * height) / 2,
        cls,
      );
    } else {
      out += line(
        cx - (ux * dash) / 2,
        cy - (uy * dash) / 2,
        cx + (ux * dash) / 2,
        cy + (uy * dash) / 2,
        cls,
      );
    }
  }
  return out;
}

/** The same notation written around an arc, as the rotation period is on the
 *  real cover: bars point radially, zeros lie tangentially. */
function binaryArc(
  bits: string,
  cx: number,
  cy: number,
  r: number,
  a0: number,
  a1: number,
  height: number,
): string {
  let out = '';
  const step = (a1 - a0) / (bits.length - 1);
  const dash = ((Math.abs(step) * Math.PI) / 180) * r * 0.45;
  for (let i = 0; i < bits.length; i++) {
    const a = ((a0 + step * i) * Math.PI) / 180;
    const ux = Math.cos(a);
    const uy = Math.sin(a); // radial
    const tx = -uy;
    const ty = ux; // tangential
    const x = cx + ux * r;
    const y = cy + uy * r;
    if (bits[i] === '1') {
      out += line(x - (ux * height) / 2, y - (uy * height) / 2, x + (ux * height) / 2, y + (uy * height) / 2, 'bit');
    } else {
      out += line(x - (tx * dash) / 2, y - (ty * dash) / 2, x + (tx * dash) / 2, y + (ty * dash) / 2, 'bit');
    }
  }
  return out;
}

function arrow(x1: number, y1: number, x2: number, y2: number, size = 9, cls = 'ln'): string {
  const a = Math.atan2(y2 - y1, x2 - x1);
  const w = 0.42;
  const p = [
    [x2, y2],
    [x2 - size * Math.cos(a - w), y2 - size * Math.sin(a - w)],
    [x2 - size * Math.cos(a + w), y2 - size * Math.sin(a + w)],
  ]
    .map(([x, y]) => `${n(x)},${n(y)}`)
    .join(' ');
  return line(x1, y1, x2, y2, cls) + `<polygon class="fill" points="${p}"/>`;
}

// --------------------------------------------------------------------------
// the drawings
// --------------------------------------------------------------------------

/** One trace of the picture signal, drawn to the layout we actually measured:
 *  a deep notch at the trace start, then the picture, then the sync plateau,
 *  its falling edge, and the flat back porch that is our DC reference. */
function traceWaveform(x0: number, w: number, index: number): string {
  const TOP = 132; // sync plateau
  const MID = 158; // picture centre
  const PORCH = 170;
  const NOTCH = 193;

  // Deterministic, so the drawing is stable across renders; three incommensurate
  // components give it the look of picture content without repeating.
  const wiggle = (t: number) => {
    const s = index * 1.7;
    return (
      5.5 * Math.sin(6.1 * t + s) +
      3 * Math.sin(15.7 * t + 2.1 * s) +
      1.6 * Math.sin(31.3 * t + 0.6 * s)
    );
  };

  const pts: string[] = [];
  const at = (f: number, y: number) => pts.push(`${n(x0 + f * w)},${n(y)}`);

  at(0.0, MID + 4);
  at(0.008, NOTCH); // the trace start: deep negative notch
  at(0.02, MID + 6);
  for (let k = 0; k <= 40; k++) {
    const f = 0.02 + (0.78 * k) / 40; // picture, 0 .. 2560/3197.4
    at(f, MID + wiggle(f));
  }
  at(0.801, MID - 2);
  at(0.812, TOP); // sync plateau, 2560 .. 2900
  at(0.905, TOP);
  at(0.911, PORCH); // falling edge, 2900 .. 2910
  at(0.988, PORCH); // back porch, 2910 .. 3160 -- the DC reference
  at(0.997, MID + 4);

  return `<polyline class="ln wave" points="${pts.join(' ')}"/>`;
}

/** Pulsar rays. The digit strings here are stylised: the real map carries each
 *  pulsar's period as a specific binary number and we have not transcribed
 *  those, so these are texture, not data. Said plainly in the reader text. */
const PULSAR_RAYS: { a: number; r: number; bits: string }[] = [
  { a: -158, r: 150, bits: '110100101101' },
  { a: -126, r: 118, bits: '1011001101' },
  { a: -101, r: 162, bits: '10011010110101' },
  { a: -74, r: 128, bits: '110101100111' },
  { a: -48, r: 96, bits: '1001101101' },
  { a: -22, r: 140, bits: '11010011010101' },
  { a: 34, r: 112, bits: '101100110101' },
  { a: 58, r: 158, bits: '1101001101101' },
  { a: 84, r: 122, bits: '10110101101' },
  { a: 110, r: 152, bits: '110011010011' },
  { a: 136, r: 104, bits: '1010110011' },
  { a: 162, r: 134, bits: '11010110101101' },
  { a: 178, r: 92, bits: '10011011' },
  { a: -177, r: 166, bits: '110101001101101' },
];

function pulsarMap(cx: number, cy: number): string {
  let out = '';
  for (const ray of PULSAR_RAYS) {
    const a = (ray.a * Math.PI) / 180;
    const ux = Math.cos(a);
    const uy = Math.sin(a);
    out += line(cx, cy, cx + ux * ray.r, cy + uy * ray.r, 'ln thin');
    const start = 0.3 * ray.r;
    out += binaryRun(ray.bits, cx + ux * start, cy + uy * start, {
      pitch: (ray.r * 0.62) / ray.bits.length,
      height: 8,
      dir: ray.a,
      cls: 'bit tiny',
    });
  }
  // The long unlabelled ray: direction and relative distance to the galactic
  // centre, closed with a bar rather than a period.
  const ga = (8 * Math.PI) / 180;
  const gx = cx + Math.cos(ga) * 190;
  const gy = cy + Math.sin(ga) * 190;
  out += line(cx, cy, gx, gy, 'ln');
  out += line(gx - Math.sin(ga) * 12, gy + Math.cos(ga) * 12, gx + Math.sin(ga) * 12, gy - Math.cos(ga) * 12, 'ln');
  out += circle(cx, cy, 5, 'fill'); // the Sun
  return out;
}

function hydrogenAtom(cx: number, cy: number, electronDown: boolean): string {
  const r = 44;
  let out = circle(cx, cy, r, 'ln');
  out += circle(cx, cy, 6.5, 'fill'); // proton
  out += arrow(cx + 17, cy + 16, cx + 17, cy - 16, 9); // proton spin, always up
  const ex = cx;
  const ey = cy - r;
  out += circle(ex, ey, 5.5, 'fill'); // electron on the orbit
  // The transition is between parallel and antiparallel spins; only the
  // electron's arrow differs between the two atoms.
  out += electronDown
    ? arrow(ex + 17, ey - 16, ex + 17, ey + 16, 9)
    : arrow(ex + 17, ey + 16, ex + 17, ey - 16, 9);
  return out;
}

// --------------------------------------------------------------------------
// glyph content
// --------------------------------------------------------------------------

interface Glyph {
  id: string;
  /** Short name, used for the aria-label and the reader heading. */
  name: string;
  /** Where it sits on the plate. */
  place: string;
  /** Hit box in viewBox units: x, y, w, h. */
  box: [number, number, number, number];
  /** The drawing. */
  art: () => string;
  /** What the engraving says. Trusted, hand-authored HTML. */
  says: string;
  /** Arithmetic, shown as a monospace block. */
  working?: string;
  /** What our decoder does with it. */
  decoder: string;
}

const traceValue = binValue(BIN_TRACE); // 11 845 632
const traceSeconds = traceValue * H_PERIOD; // 0.008339611 s
const traceSamples96 = traceSeconds * SAMPLE_RATE; // 800.6027
const measuredSeconds = NOMINAL_PERIOD / SAMPLE_RATE; // 0.008326563 s
const fastPct = (traceSeconds / measuredSeconds - 1) * 100; // 0.1567 %
const rotSeconds = binValue(BIN_ROTATION) * H_PERIOD; // 3.59994 s
const sideSeconds = binValue(BIN_SIDE) * H_PERIOD; // 3229.38 s

const GLYPHS: Glyph[] = [
  {
    id: 'hydrogen',
    name: 'The hydrogen hyperfine transition',
    place: 'Lower right — the keystone',
    box: [530, 782, 420, 168],
    art: () =>
      hydrogenAtom(648, 862, false) +
      hydrogenAtom(846, 862, true) +
      line(692, 862, 802, 862, 'ln') +
      binaryRun('1', 747, 912, { pitch: 0, height: 34 }),
    says: `Two hydrogen atoms, each drawn as a proton with one electron. On the left the
      two spins are parallel; on the right they are antiparallel. A line joins the two states and
      is marked with a single vertical bar: the digit <b>1</b>. That line is a photon, and the
      cover is saying <em>one unit of time = the period of that photon</em>.
      <p>This is the only number on the cover that is not <em>on</em> the cover. You cannot look it
      up in the engraving; you have to know it from physics. Any civilisation that has built a
      radio telescope has already measured it, because neutral hydrogen fills the galaxy and
      shouts at 21 cm. That is the whole trick: no shared language, no shared units, one shared
      fact.</p>`,
    working: `f = 1 420 405 751.768 Hz        (the 21 cm line)
T = 1 / f
  = 0.000 000 000 704 024 184 s
  = 0.704 ns

one sample at 96 kHz
  = 1 / 96 000 s
  = ${(1 / SAMPLE_RATE / H_PERIOD).toFixed(1)} hydrogen periods`,
    decoder: `Every duration below is a count of these. Our decoder never handles the number
      itself — by the time the audio reaches the browser it is already a sequence of samples at
      96 kHz — but the two engraved counts that matter, 11 845 632 per trace and 512 traces per
      frame, are what set <code>nominalPeriod</code> and <code>tracesPerFrame</code> in
      <code>catalog.json</code>. Change your belief about T and every timing on the plate moves
      with it, in lockstep, which is exactly what a good unit does.`,
  },

  {
    id: 'countKey',
    name: 'The binary counting key',
    place: 'Upper right — above the first three traces',
    box: [530, 62, 420, 56],
    art: () =>
      binaryRun('1', 566, 96, { pitch: 16, height: 26 }) +
      binaryRun('10', 691, 96, { pitch: 16, height: 26 }) +
      binaryRun('11', 816, 96, { pitch: 16, height: 26 }),
    says: `Above the first three cycles of the picture signal, the cover simply counts:
      <b>one, two, three</b>. A vertical bar is the digit 1, a horizontal dash is the digit 0.
      <p>Three symbols do an enormous amount of work here. Counting to three establishes that the
      notation is positional and base two; it names both digits; and because two is written
      <span class="gr-lit">|–</span> and not <span class="gr-lit">–|</span>, it fixes the reading
      order as most-significant digit first. Every other number on the plate — the rotation
      period, the play time, the line duration, the pulsar periods — is written in this notation
      and cannot be read until these three marks have been understood.</p>`,
    working: `|      = 1
|-     = 1x2^1 + 0x2^0 = 2
||     = 1x2^1 + 1x2^0 = 3

so the digits are  | = 1   - = 0
and the leftmost digit is the largest.`,
    decoder: `Nothing in the code reads bars and dashes, but everything in the code inherits
      from them. The two numbers we lifted off this plate and hard-wrote into the data contract —
      <code>tracesPerFrame: 512</code> and the 8.34 ms line that becomes
      <code>nominalPeriod</code> — are only knowable because of these three marks.`,
  },

  {
    id: 'traceDuration',
    name: 'How long one trace lasts',
    place: 'Upper right — beneath picture line 1',
    box: [530, 200, 420, 62],
    art: () =>
      // span bracket over the first trace
      line(560, 212, 678, 212, 'ln') +
      line(560, 206, 560, 218, 'ln') +
      line(678, 206, 678, 218, 'ln') +
      binaryRun(BIN_TRACE, 560, 242, { pitch: 14, height: 22 }),
    says: `A twenty-four digit number, written under the span of a single picture line. In
      hydrogen periods, this is how long one trace takes.
      <p>This is the number that makes the record decodable at all. Without it you would have
      512 columns of noise and no idea how wide a column is.</p>`,
    working: `${BIN_TRACE}   (24 digits)
  = 2^23 + 2^21 + 2^20 + 2^18 + 2^15 + 2^14
  = 8388608 + 2097152 + 1048576
      + 262144 + 32768 + 16384
  = 11 845 632 hydrogen periods

x 0.704 024 184 ns = ${(traceSeconds * 1000).toFixed(6)} ms per trace
x 96 000 samples/s = ${traceSamples96.toFixed(3)} samples per trace`,
    decoder: `Our catalog carries <code>nominalPeriod: ${NOMINAL_PERIOD}</code> samples, not
      ${traceSamples96.toFixed(2)}. The engraved value is ${fastPct.toFixed(3)}% longer than what
      this digitisation actually contains. So the engraved number is only ever a starting guess:
      <code>sync.recover()</code> folds several hundred lines together to build a template of the
      whole sync region, cross-correlates every line against it on the derivative, refines each
      peak parabolically to a fraction of a sample, and fits position against line index with
      iteratively reweighted least squares. <em>The slope of that fit is the real period</em>,
      measured per frame. The residuals are the master tape's wow and flutter.`,
  },

  {
    id: 'traces',
    name: '512 traces make one picture',
    place: 'Upper right — the raster rectangle',
    box: [530, 392, 420, 176],
    art: () => {
      let out = binaryRun(BIN_TRACES_PER_FRAME, 614, 410, { pitch: 24, height: 22 });
      out += rect(600, 428, 280, 132, 'ln');
      for (let i = 0; i < 33; i++) {
        const x = 608 + i * 8.3;
        out += line(x, 436, x, 552, 'ln thin');
      }
      return out;
    },
    says: `A rectangle filled with vertical lines, labelled with a ten-digit binary number.
      The picture is scanned as vertical columns — like television turned on its side — and this
      says how many of them there are.
      <p>512 is not an accident of engineering. It is a round number in the notation the cover
      just taught you: a one followed by nine zeros.</p>`,
    working: `${BIN_TRACES_PER_FRAME}     (10 digits)
  = 2^9
  = 512 vertical lines per picture

512 x 8.339 611 ms = ${(TRACES_PER_FRAME * traceSeconds).toFixed(4)} s per frame (engraved)
512 x 8.326 563 ms = ${(TRACES_PER_FRAME * measuredSeconds).toFixed(4)} s per frame (measured)`,
    decoder: `<code>tracesPerFrame: 512</code> in the catalog, <code>traces: 512</code> in the
      decoder settings, and 512 columns in the canvas. Note what is <em>not</em> engraved: how many
      pixels tall each column is. The record does not say, because along the trace the signal is
      continuous — it is an analogue brightness curve, not a row of samples. Our default
      <code>height: 384</code> is a choice about how finely to resample the picture window
      (fractions <code>pictureStart</code> to <code>pictureEnd</code> of the period), not a fact
      recovered from the plate.`,
  },

  {
    id: 'waveform',
    name: 'Brightness as a signal',
    place: 'Upper right — the top drawing',
    box: [530, 120, 420, 78],
    art: () => traceWaveform(560, 118, 0) + traceWaveform(685, 118, 1) + traceWaveform(810, 118, 2),
    says: `The signal at the start of a picture: three cycles, each one a single vertical
      column of the image. The tall flat shelf near the end of every cycle is the sync pulse that
      says <em>a new column begins now</em>; the wobbling part is the picture itself.
      <p>The drawing here is our own measurement rather than a copy of the cover's cruder sketch,
      because this glyph is where the plate and the recording actually meet.</p>`,
    working: `one trace, as fractions of the measured period:

  0.000 .. 0.801   picture (one column)
  0.801 .. 0.907   sync plateau
  0.907 .. 0.910   falling edge
  0.910 .. 0.988   back porch  <- DC reference
  0.988 .. 1.000   down into the next notch

the trace start is the deep negative notch
that follows the porch.`,
    decoder: `Those fractions are literally the constants at the top of <code>decode.py</code>:
      <code>PICTURE_END = 2560/3197.4</code>, <code>PORCH_START = -282/3197.4</code>,
      <code>PORCH_END = -37/3197.4</code>. Expressing them as fractions of a period rather than as
      sample counts is what lets the same code run on the 384 kHz master and the 96 kHz web slice.
      The porch is flat, so its median is the level that <em>should</em> be zero; subtracting it
      per trace is <code>dcRestore</code>. And because the record stores a negative — more signal
      means darker — the last thing we do is <code>invert</code>.`,
  },

  {
    id: 'interlace',
    name: 'Lines are drawn vertically',
    place: 'Upper right — the middle drawing',
    box: [530, 274, 420, 106],
    art: () => {
      let out = '';
      for (let i = 0; i < 9; i++) {
        const x = 618 + i * 30;
        const off = i % 2 ? 14 : 0;
        out += line(x, 296 + off, x, 356 + off, 'ln bold');
      }
      out += arrow(618, 284, 706, 284, 8, 'ln thin'); // order: left to right
      out += arrow(600, 296, 600, 356, 8, 'ln thin'); // each line runs downward
      return out;
    },
    says: `Under the waveform, a drawing of the lines being laid down: vertical, one after
      another, left to right, with a slight stagger between neighbours — the cover's note that the
      lines are interlaced rather than simply consecutive.
      <p>It is a small drawing carrying a large instruction: the signal is one dimensional, the
      picture is two dimensional, and this is the rule that converts between them.</p>`,
    decoder: `Our decoder writes trace <em>i</em> into column <em>i</em>, in recorded order, and
      the catalog carries no interlace field — the images in this digitisation come out coherent
      that way. <code>flipScan</code> reverses the column order when a frame was scanned the other
      way, and <code>rotate</code> applies the quarter turns from Barry's orientation table.
      <span class="gr-todo">TODO: we have not independently verified whether the recorded frames
      use the staggered interlace the cover depicts, or whether any stagger is simply absorbed by
      the per-trace timebase. Do not treat the absence of a de-interlace step as evidence either
      way.</span>`,
  },

  {
    id: 'circle',
    name: 'The first picture is a circle',
    place: 'Upper right — the bottom drawing',
    box: [530, 580, 420, 190],
    art: () => {
      let out = rect(626, 592, 228, 170, 'ln');
      for (let i = 0; i < 26; i++) {
        const x = 632 + i * 8.6;
        out += line(x, 598, x, 756, 'ln hair');
      }
      out += circle(740, 677, 72, 'ln bold');
      return out;
    },
    says: `A replica of the very first image on the record: a circle in a frame. It is a
      checksum you can see. If your reconstruction comes out as an ellipse, your horizontal scale
      is wrong. If it comes out as a slanted smear, your line period is wrong. If it comes out as
      noise, you have not found sync at all.
      <p>Nothing else on the plate closes the loop like this. The cover tells you what to do and
      then tells you what the answer looks like.</p>`,
    decoder: `Image 1 in the catalog is the calibration circle, and it is the first frame this
      page decodes for exactly the reason above. The <code>aspect</code> setting is the horizontal
      scale applied at render time, and the cover's instruction for choosing it is unambiguous:
      the correct value is the one that makes image 1 round. Sync failures announce themselves the
      same way the cover implies — a wrong period tilts the circle into a spiral, a lost trace
      tears it.`,
  },

  {
    id: 'record',
    name: 'The record, the stylus, and 16⅔ rpm',
    place: 'Upper left',
    box: [70, 60, 380, 380],
    art: () => {
      const cx = 250;
      const cy = 245;
      let out = '';
      for (const r of [132, 120, 108, 96, 84, 72, 60]) out += circle(cx, cy, r, 'ln thin');
      out += circle(cx, cy, 132, 'ln');
      out += circle(cx, cy, 48, 'ln');
      out += circle(cx, cy, 9, 'fill');
      // stylus, sitting on the outermost groove: play from the outside in
      const a = (-55 * Math.PI) / 180;
      const px = cx + Math.cos(a) * 132;
      const py = cy + Math.sin(a) * 132;
      out += line(415, 72, px + 6, py - 10, 'ln');
      out += line(px + 6, py - 10, px, py, 'ln bold');
      out += circle(415, 72, 6, 'fill');
      // direction of travel along the groove
      out += arrow(cx + Math.cos(a - 0.5) * 146, cy + Math.sin(a - 0.5) * 146, cx + Math.cos(a - 0.18) * 146, cy + Math.sin(a - 0.18) * 146, 8, 'ln thin');
      out += binaryArc(BIN_ROTATION, cx, cy, 164, 32, 148, 16);
      return out;
    },
    says: `The record seen face on, with the stylus resting where it should start: the outer
      edge, playing inwards. Written around the rim in binary is the time for one rotation.
      <p>Note what this glyph assumes — that the reader will build a machine to spin a disc at a
      speed specified to nine significant figures, using a clock derived from a hydrogen atom. The
      cover is not describing a record player. It is specifying one.</p>`,
    working: `${BIN_ROTATION}
  (33 digits)
  = 2^32 + 2^29 + 2^28 + 2^23 + 2^22 + 2^19
  = 5 113 380 864 hydrogen periods

x 0.704 024 184 ns = ${rotSeconds.toFixed(4)} s per rotation
60 / ${rotSeconds.toFixed(4)}     = ${(60 / rotSeconds).toFixed(3)} rpm  = 16 2/3 rpm

${rotSeconds.toFixed(4)} s / 8.339 611 ms = ${(rotSeconds / traceSeconds).toFixed(2)} traces
per revolution -- deliberately not a whole
number, so a frame is not locked to the disc.`,
    decoder: `This is the one instruction our decoder is spared: we start from a 384 kHz
      digitisation of the master tape, so the turntable step has already been carried out by
      somebody else. That is also precisely where the error lives. Any deviation in playback speed
      — of the mastering lathe, the tape machine, or the transfer — arrives as a uniform scale
      error on every duration downstream, which is exactly the ${fastPct.toFixed(2)}% we measure.
      An alien with a hydrogen clock would have got the period right; we have to measure ours.`,
  },

  {
    id: 'side',
    name: 'How long one side runs',
    place: 'Upper left — the side view',
    box: [70, 455, 380, 140],
    art: () => {
      let out = rect(90, 502, 330, 7, 'ln fill-plate', 3);
      out += line(90, 505.5, 420, 505.5, 'ln thin');
      out += rect(348, 470, 38, 22, 'ln', 3); // cartridge
      out += line(367, 492, 367, 501, 'ln bold'); // needle
      out += binaryRun(BIN_SIDE, 90, 556, { pitch: 7.9, height: 15 });
      return out;
    },
    says: `The record seen edge on, a wafer with the stylus riding on it, and beneath it a
      forty-three digit number: the time to play one side.
      <p>Between them, this glyph and the one above give the reader a self-check. Play time divided
      by rotation time is a count of revolutions, and a reader who arrives at a sensible number
      knows they have read both numbers correctly.</p>`,
    working: `${BIN_SIDE.slice(0, 22)}
${BIN_SIDE.slice(22)}                     (43 digits)
  = 2^42 + 2^37 + 2^35 + 2^34
  = 4 587 025 072 128 hydrogen periods

x 0.704 024 184 ns = ${sideSeconds.toFixed(1)} s
                   = ${Math.floor(sideSeconds / 60)} min ${(sideSeconds % 60).toFixed(0)} s

${sideSeconds.toFixed(1)} / ${rotSeconds.toFixed(4)} = ${(sideSeconds / rotSeconds).toFixed(1)} revolutions per side`,
    decoder: `The pictures are only one stretch of that side. Our master is 473.86 s of image
      audio; 78 frames per channel at the measured ${(TRACES_PER_FRAME * measuredSeconds).toFixed(3)} s
      per frame accounts for ${(78 * TRACES_PER_FRAME * measuredSeconds).toFixed(1)} s of it. The
      catalog slices that into one FLAC per frame, each starting <code>leadIn</code> samples before
      its first trace so the timebase fit has signal to lock onto before column zero.`,
  },

  {
    id: 'pulsar',
    name: 'The pulsar map',
    place: 'Lower left',
    box: [70, 610, 380, 335],
    art: () => pulsarMap(250, 782),
    says: `Fourteen rays from the Sun, each pointing at a pulsar, each ray's length set by that
      pulsar's distance, and each labelled in binary with that pulsar's rotation period — again in
      hydrogen periods. A fifteenth, longer ray carries no number: it points at the centre of the
      galaxy.
      <p>It is an address and a date at once. Pulsars spin down at rates we can measure, so the
      periods engraved here are only correct for a particular decade; a reader who can observe
      those same fourteen objects can work backwards to when the plate was made.</p>
      <p class="gr-caveat">The digit patterns drawn on these rays are stylised. We have not
      transcribed the fourteen real periods, and you should not read these marks as data.</p>`,
    decoder: `Same idea as the trace-duration number, applied to the sky: identify a periodic
      thing everyone can see, and write its period as a count of hydrogen periods. And the same
      caveat applies in both directions. A reader in the far future who trusted the engraved pulsar
      periods rather than measuring the ones actually in front of them would be wrong by the spin
      down, just as we would be wrong by ${fastPct.toFixed(2)}% if we trusted 8.34 ms rather than
      fitting the period each frame. The cover tells you where to look; you still have to measure.`,
  },
];

// --------------------------------------------------------------------------
// page
// --------------------------------------------------------------------------

function glyphSvg(g: Glyph, order: number): string {
  const [x, y, w, h] = g.box;
  return (
    `<g class="gr-glyph" id="gr-glyph-${g.id}" data-id="${g.id}" role="button" tabindex="0"` +
    ` aria-pressed="false" aria-label="${order + 1} of ${GLYPHS.length}. ${g.name}. ${g.place}.">` +
    `<title>${g.name}</title>` +
    rect(x, y, w, h, 'gr-glyph__frame', 12) +
    `<g class="gr-glyph__art">${g.art()}</g>` +
    `<rect class="gr-glyph__hit" x="${n(x)}" y="${n(y)}" width="${n(w)}" height="${n(h)}"/>` +
    `</g>`
  );
}

function coverSvg(): string {
  return (
    `<svg class="gr-plate" viewBox="0 0 1000 1000" role="group" ` +
    `aria-label="Diagram engraved on the Voyager Golden Record cover. ${GLYPHS.length} glyphs." ` +
    `xmlns="http://www.w3.org/2000/svg">` +
    rect(6, 6, 988, 988, 'gr-plate__edge', 20) +
    GLYPHS.map(glyphSvg).join('') +
    `</svg>`
  );
}

const INTRO = `<p class="gr-reader__lede">Everything this page's decoder needs to know was
  engraved on a gold-plated aluminium plate in 1977, in a notation designed for a reader who
  shares no language and no units with us.</p>
  <p>Hover, tap or tab through the glyphs. Each one says what it tells you, works out the
  arithmetic, and names the thing our decoder does because of it.</p>`;

function readerHtml(g: Glyph, order: number): string {
  const working = g.working
    ? `<h4 class="gr-reader__h">The arithmetic</h4><pre class="gr-reader__work">${g.working}</pre>`
    : '';
  return (
    `<p class="gr-reader__place">${g.place} &middot; ${order + 1} of ${GLYPHS.length}</p>` +
    `<h3 class="gr-reader__title">${g.name}</h3>` +
    `<div class="gr-reader__says">${g.says}</div>` +
    working +
    `<h4 class="gr-reader__h gr-reader__h--decoder">In our decoder</h4>` +
    `<div class="gr-reader__decoder">${g.decoder}</div>`
  );
}

export function init(container: HTMLElement): void {
  container.innerHTML = '';

  const root = document.createElement('section');
  root.className = 'gr-cover';
  root.innerHTML = `
    <header class="gr-cover__head">
      <p class="gr-cover__kicker">The instructions</p>
      <h2 class="gr-cover__title">The Cover</h2>
      <p class="gr-cover__sub">The decoder's entire parameter list, engraved on the outside of
        the record. Read the plate, then read the code.</p>
    </header>
    <div class="gr-cover__body">
      <figure class="gr-cover__figure">
        ${coverSvg()}
        <figcaption class="gr-cover__caption">Hand-drawn from the engraving, not traced from a
          photograph. Proportions follow the original layout; the waveform is drawn to our own
          measurements.</figcaption>
      </figure>
      <aside class="gr-reader" aria-label="Glyph explanation">
        <div class="gr-reader__dots" aria-hidden="true"></div>
        <div class="gr-reader__panel" role="status" aria-live="polite">${INTRO}</div>
        <div class="gr-reader__nav">
          <button type="button" class="gr-btn" data-step="-1">&larr; Previous glyph</button>
          <button type="button" class="gr-btn" data-step="1">Next glyph &rarr;</button>
        </div>
      </aside>
    </div>
    <p class="gr-cover__note"><b>An honest discrepancy.</b> The cover specifies
      ${(traceSeconds * 1000).toFixed(3)} ms per trace, exactly ${traceValue.toLocaleString('en-US')}
      hydrogen periods. This digitisation of the master tape runs about ${fastPct.toFixed(2)}% fast:
      ${(measuredSeconds * 1000).toFixed(3)} ms, or ${NOMINAL_PERIOD} samples at 96 kHz rather than
      the engraved ${traceSamples96.toFixed(2)}. Over a 512-trace frame that is
      ${((TRACES_PER_FRAME * (traceSeconds - measuredSeconds)) * 1000).toFixed(1)} ms of drift —
      more than three quarters of a trace, enough to visibly shear the picture. Somewhere between
      the 1977 lathe and this file, a machine ran slightly fast. That is why the decoder measures
      the period on every frame instead of trusting the engraved value: the artefact states the
      ideal, the recording is a physical object, and only one of them is in front of us.</p>
  `;
  container.appendChild(root);

  const svg = root.querySelector('.gr-plate') as SVGSVGElement;
  const panel = root.querySelector('.gr-reader__panel') as HTMLElement;
  const dots = root.querySelector('.gr-reader__dots') as HTMLElement;
  const nodes = Array.from(root.querySelectorAll<SVGGElement>('.gr-glyph'));

  dots.innerHTML = GLYPHS.map((g) => `<span class="gr-dot" data-id="${g.id}"></span>`).join('');
  const dotNodes = Array.from(dots.querySelectorAll<HTMLElement>('.gr-dot'));

  let pinned = -1; // index locked by click / Enter / Space, or -1
  let shown = -1; // index currently rendered

  const render = (i: number) => {
    if (i === shown) return;
    shown = i;
    panel.innerHTML = i < 0 ? INTRO : readerHtml(GLYPHS[i], i);
    root.classList.toggle('is-selecting', i >= 0);
    nodes.forEach((el, k) => el.classList.toggle('is-active', k === i));
    dotNodes.forEach((el, k) => el.classList.toggle('is-active', k === i));
  };

  const preview = (i: number) => render(i);
  const clearPreview = () => render(pinned);

  const pin = (i: number) => {
    pinned = pinned === i ? -1 : i;
    nodes.forEach((el, k) => el.setAttribute('aria-pressed', String(k === pinned)));
    render(pinned === -1 ? i : pinned);
  };

  const focusGlyph = (i: number) => {
    const j = ((i % nodes.length) + nodes.length) % nodes.length;
    // Safari will not always move focus to an SVG element from a click handler,
    // so render first and let focus be best effort.
    render(j);
    nodes[j].focus();
  };

  nodes.forEach((el, i) => {
    el.addEventListener('pointerenter', () => preview(i));
    el.addEventListener('focus', () => preview(i));
    el.addEventListener('blur', clearPreview);
    el.addEventListener('click', () => pin(i));
    el.addEventListener('keydown', (ev: KeyboardEvent) => {
      switch (ev.key) {
        case 'Enter':
        case ' ':
        case 'Spacebar':
          ev.preventDefault();
          pin(i);
          break;
        case 'ArrowRight':
        case 'ArrowDown':
          ev.preventDefault();
          focusGlyph(i + 1);
          break;
        case 'ArrowLeft':
        case 'ArrowUp':
          ev.preventDefault();
          focusGlyph(i - 1);
          break;
        case 'Home':
          ev.preventDefault();
          focusGlyph(0);
          break;
        case 'End':
          ev.preventDefault();
          focusGlyph(nodes.length - 1);
          break;
        case 'Escape':
          pinned = -1;
          nodes.forEach((g) => g.setAttribute('aria-pressed', 'false'));
          render(-1);
          break;
      }
    });
  });

  svg.addEventListener('pointerleave', clearPreview);

  dotNodes.forEach((el, i) => {
    el.addEventListener('pointerenter', () => preview(i));
    el.addEventListener('pointerleave', clearPreview);
  });

  // The reader-pane buttons are the guaranteed keyboard path: they are real HTML
  // buttons, so they work even where tabbing into SVG content does not.
  root.querySelectorAll<HTMLButtonElement>('.gr-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const step = Number(btn.dataset.step);
      const next = shown < 0 ? (step > 0 ? 0 : GLYPHS.length - 1) : shown + step;
      const j = ((next % GLYPHS.length) + GLYPHS.length) % GLYPHS.length;
      pinned = j;
      nodes.forEach((el, k) => el.setAttribute('aria-pressed', String(k === j)));
      render(j);
    });
  });
}
