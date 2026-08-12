# The Cover panel — sources, arithmetic, and what is stylised

Working notes for `cover.ts` / `cover.css`. Everything asserted in the panel's text is either
checked below, taken from the project's established signal facts, or explicitly flagged as
stylised.

## What the panel is for

The page's thesis is that the decoder is not clever — it is obedient. Every parameter it needs
(how many traces, how long a trace lasts, which way the picture scans, what the first image should
look like) was engraved on the record's cover in 1977 in a notation that assumes no shared language
and no shared units. The panel walks that argument glyph by glyph and, for each one, names the
line of our code that exists because of it.

The one place the argument does not close cleanly is the trace duration, and that tension is the
most interesting thing on the panel, so it is stated twice: inside the `traceDuration` glyph and
in a standing note under the diagram.

## Layout, and the source for it

JPL/NASA's own explanation gives the positions, and the panel follows them:

| Position | Glyph(s) in `cover.ts` |
|---|---|
| Upper left | `record` (plan view, stylus at the outer edge, rotation time around the rim) |
| Upper left, below | `side` (edge-on view, playing time of one side) |
| Upper right, top to bottom | `countKey`, `waveform`, `traceDuration`, `interlace`, `traces`, `circle` |
| Lower left | `pulsar` |
| Lower right | `hydrogen` |

Sources for layout and meaning:

- NASA/JPL, *Golden Record Cover* — <https://science.nasa.gov/mission/voyager/golden-record-cover/>
  ("In the upper left-hand corner is an easily recognized drawing of the phonograph record and the
  stylus…"; the picture-construction sequence; the calibration circle; the pulsar map; the hydrogen
  atom diagram; the uranium-238 plating.)
- Wikipedia, *Voyager Golden Record* — <https://en.wikipedia.org/wiki/Voyager_Golden_Record>
  (same explanation, plus "between 53 and 54 minutes" for one side.)
- Wikipedia, *Pioneer plaque* — <https://en.wikipedia.org/wiki/Pioneer_plaque>
  (binary notation: "Rather than the familiar '1' and '0', 'I' and '–' are used"; the hydrogen
  hyperfine transition supplying both the 21 cm length unit and the 0.704 ns time unit.)
- CED Magic, *The Voyager Interstellar Record* — <http://cedmagic.com/featured/voyager/voyager-record.html>
  (the actual digit strings for the three long engraved numbers, and the 7.04024183647e-10 s value
  for the hydrogen period.)
- R. W. Johnston, *Reading the Pioneer/Voyager pulsar map* —
  <https://www.johnstonsarchive.net/astro/pulsarmap.html>
  (14 pulsars; periods given "in binary notation as multiples of the hyperfine transition period of
  hydrogen"; the extra line to the galactic centre as the reference direction.)
- PBS, *How to Read a Pulsar Map* — <https://www.pbs.org/the-farthest/science/pulsar-map/>
  (ray length ∝ distance; the binary beside each ray is that pulsar's period.)

The one point where sources disagree: a NASA image caption
(<https://www.nasa.gov/image-article/instructions-for-aliens/>) places the record-and-stylus drawing
in the *bottom right*, which contradicts JPL's own "upper left". That caption is describing a
differently-oriented reproduction. We follow the JPL text.

## Arithmetic, verified

All three engraved numbers were re-derived rather than copied. `H_PERIOD` in `cover.ts` is
`1 / 1420405751.768 Hz = 7.040241837624815e-10 s`.

```
101101001100000000000000                      (24 digits, trace duration)
  = 2^23 + 2^21 + 2^20 + 2^18 + 2^15 + 2^14
  = 8388608 + 2097152 + 1048576 + 262144 + 32768 + 16384
  = 11 845 632
  x H_PERIOD = 8.33961140 ms          -> 800.6027 samples at 96 kHz
                                      -> 3202.411 samples at 384 kHz

1000000000                                    (10 digits, traces per picture)
  = 2^9 = 512

100110000110010000000000000000000             (33 digits, one rotation)
  = 2^32 + 2^29 + 2^28 + 2^23 + 2^22 + 2^19
  = 5 113 380 864
  x H_PERIOD = 3.59994 s              -> 60 / 3.59994 = 16.667 rpm = 16 2/3 rpm

1000010110000000000000000000000000000000000   (43 digits, one side)
  = 2^42 + 2^37 + 2^35 + 2^34
  = 4 587 025 072 128
  x H_PERIOD = 3229.38 s = 53 min 49 s        (matches "between 53 and 54 minutes")

cross-checks
  4 587 025 072 128 / 5 113 380 864 = 897.06 revolutions per side
  5 113 380 864 / 11 845 632        = 431.67 traces per revolution (not an integer)
  512 x 8.33961 ms                  = 4.2699 s per frame, engraved
  512 x 8.32656 ms                  = 4.2632 s per frame, measured
```

Note the panel prints `1000000000` (ten digits, 2^9) for 512. The brief wrote it with nine digits,
which is 2^8 = 256; ten digits is correct and that is what is drawn and explained.

The engraved-vs-measured discrepancy, from `pipeline/sync.py`'s `NOMINAL_PERIOD = 3197.4` at
384 kHz (799.35 at 96 kHz):

```
engraved  8.339611 ms   800.603 samples @ 96 kHz
measured  8.326563 ms   799.350 samples @ 96 kHz
ratio     1.0015671     -> the engraved value is 0.157% longer,
                           i.e. this digitisation runs ~0.16% fast
over 512 traces: 6.68 ms, or 0.80 of a whole trace
```

0.80 of a trace of accumulated drift across a frame is why trusting the engraved number would
visibly shear every picture, and why `sync.recover()` fits the period per frame instead. This is
the same figure already recorded in the comment above `NOMINAL_PERIOD` in `pipeline/sync.py`; the
panel does not introduce a new measurement.

## Deliberately stylised, i.e. not data

- **The fourteen pulsar periods.** `PULSAR_RAYS` in `cover.ts` carries invented digit strings and
  invented ray angles/lengths. The real map encodes fourteen specific periods and fourteen specific
  directions and we have not transcribed them. The panel says so in the glyph text, in a
  `.gr-caveat` paragraph, and the code comment says so too. If someone later transcribes the real
  values, only that one array needs replacing.
- **The waveform.** Drawn to *our* measured trace layout (picture 0–0.801 of the period, plateau to
  0.907, falling edge, back porch to 0.988, notch at the trace start), not to the cover's own much
  cruder three-cycle sketch. This is stated in the glyph text and in the figure caption, because
  the whole point of that glyph is where the plate and the recording meet.
- **Groove count, cartridge shape, atom orbit radii** and similar drawing details are chosen for
  legibility, not measured off the plate.
- The uranium-238 electroplating is on the real cover but is not a *diagram*, so it has no glyph.

## Open TODOs

- **Interlace.** NASA's explanation says the middle drawing shows the lines "with staggered
  'interlace'". Our established signal facts say only that there are 512 traces per frame and that
  each trace is one column; nothing in the catalog or the pipeline mentions interlace. The panel
  draws the stagger the cover depicts, says our decoder writes trace *i* into column *i*, and
  carries an explicit `TODO` in the glyph text warning the reader not to read the absence of a
  de-interlace step as evidence either way. Resolve by measuring, not by reasoning.
- Whether the cover's engraved rotation figure or the transfer chain is responsible for the 0.16%
  is not something this panel can settle; it says "somewhere between the 1977 lathe and this file",
  which is as far as our evidence goes.

## Implementation decisions

- **Everything is computed, nothing is traced.** No external image, no pasted path data. Binary
  numbers are rendered from their real digit strings by `binaryRun` / `binaryArc`, so a wrong digit
  would be a visible bug rather than decoration. A vertical stroke is 1, a horizontal stroke is 0,
  matching the plaque convention; along the pulsar rays and around the record rim the marks rotate
  with the writing direction.
- **Story order, not spatial order.** DOM order (and therefore tab order and the "n of 10"
  numbering) runs hydrogen → counting key → trace duration → 512 → waveform → interlace → circle →
  record → side → pulsar. Tabbing through the plate walks the argument rather than the layout.
- **Keyboard.** Each glyph is a `role="button"` with `tabindex="0"`; Enter/Space pins, Escape
  unpins, arrows/Home/End move focus. Because tabbing *into* SVG content is the least reliable
  thing in this panel across browsers, the reader pane also has real HTML Previous/Next buttons
  that drive the same state — that is the guaranteed keyboard path, not a decoration.
- **Focus ring is drawn, not `outline`.** `outline` on SVG elements is inconsistent in Safari, so
  each glyph has a `.gr-glyph__frame` rect that takes the hover, active and focus styling.
- **Theme.** Light is the artefact (gold plate, dark engraving); dark inverts it to gold lines on
  charcoal, which is closer to how the plate looks under raking light. Driven by
  `prefers-color-scheme` with `:root[data-theme=...]` overrides that win in both directions.
- **`aria-live="polite"` on the reader pane.** Hover-driven changes will make it chatty for screen
  reader users; the alternative (announcing nothing) is worse for the keyboard path, which is the
  one screen reader users are actually on.
- The panel is self-contained: it imports only `./cover.css`, touches no other module, and exports
  a single `init(container: HTMLElement): void`.

## Verified in-browser

Bundled with esbuild and rendered in Chrome at wide and narrow widths, in both themes: layout,
hover, focus ring, arrow-key navigation, Enter to pin, the dot strip, and the stacking of the
two-column layout under ~860 px. Type-checked with `tsc --strict`. Not yet opened in Firefox or
Safari.
