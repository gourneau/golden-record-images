# What an alien could not easily work out

Ron Barry decoded this record in 2017 under a self-imposed rule: use only knowledge any
intelligent species would have. Physics and mathematics were fair game; anything Earth-specific
was not. It is a good rule and it produced a good decode.

We have broken it comprehensively.

Our best result so far — that each trace is 262.5 sample-and-hold plateaus rather than a
continuous sweep — came from a 1977 patent for a scan converter built by a company in Boulder,
Colorado. That is not physics. That is industrial archaeology, available only to someone
standing on the same planet the record left.

So it is worth asking seriously: **which parts of this could a recipient actually recover?**
The answer is more encouraging than it first appears, and the failures are concentrated in
places the designers could have fixed.

Assume the recipient is at least as capable as us, with instruments and something like a
competent AI — able to search a signal exhaustively for structure, but with no access to human
history, conventions, or biology.

---

## Level 1 — Engraved on the cover, and genuinely universal

The cover is a small masterpiece. It communicates the decoding parameters without language,
units, or shared mathematics beyond physics.

| What | Why it works anywhere |
|---|---|
| The hydrogen hyperfine transition as the unit of time | The most common event in the universe. Anyone can measure it. Every other duration is written as a multiple of it. |
| Binary notation, taught in place with a counting key | Any species that counts can follow a worked example. |
| 512 traces per image | Written in that binary, next to a picture of scan lines. |
| 8.34 ms per trace | 11,845,632 hydrogen periods. |
| The picture is built from stripes | Drawn, not described. |
| **The first image is a circle** | Self-checking. If yours comes out oval, you are wrong, and *by exactly how much*. |

The circle is the best idea on the whole cover. It needs no explanation, it validates the entire
chain at once, and it quantifies its own error. If you take one lesson from this record for
designing a message to strangers, take that one.

## Level 2 — Not on the cover, but findable in the signal by anyone patient

This is the encouraging tier. Everything here we found empirically, or could have. A capable
recipient with a good analysis engine gets all of it without knowing anything about Earth.

**The dot clock.** Our headline finding. We got the number 262.5 from a patent — but we then
*confirmed* it by averaging the picture-band spectrum across hundreds of traces and finding a
narrow line at 31.52 kHz. That measurement needed no patent. A recipient who simply asks "is
this signal quantised in time?" finds the same line, measures 262.5 plateaus per trace, and
starts integrating each dot over its own interval. They would never know the phrase "NTSC field"
and would not need to. **The why is Earth-specific; the what is in the data.**

**The alternating sync pulse.** Two distinct pulse widths, cleanly separable by clustering.
Discoverable in an afternoon. That it encodes which television field a line came from is
unknowable and irrelevant — you only need to know there are two shapes and to lock onto the
falling edge they share.

**The 60 Hz hum.** A periodic contaminant at half the line rate. Findable, characterisable,
removable. That it originates in a planetary power grid running at sixty cycles is a charming
irrelevance.

**Tape speed drift, wow and flutter.** The sync train samples the transport's speed 120 times a
second. Measurable and correctable.

**The brightness droop.** Nothing in the chain passed a constant level. The calibration image's
uniform field makes this obvious and lets you fit and invert it.

**Where each image starts.** Every frame is preceded by a stretch of gap and then a block of
constant level. Content-free by construction, and therefore a reliable marker.

## Level 3 — Genuinely hard, and arguably design faults

Here the record lets its reader down.

There are six things a recipient needs on top of a correctly timed decode before the pictures
mean what they are supposed to mean. Only one is genuinely settled by the artifact:

| Question | Settled by the record? | How |
|---|---|---|
| **Handedness** — is the whole raster mirrored? | **Yes**, and by design | The pulsar map is *both* image 2 and an engraving on the cover. It is strongly chiral, so comparing them pins it. Global: one answer covers all 116. |
| **Polarity** — is more signal brighter or darker? | Almost | Sync pulses sit beyond black, so the picture only ever swings one way from blanking. Sound engineering inference, not a statement. |
| **Which edge is up** | **No** | Nothing encodes it. |
| **Which images were turned** | **No** — and we tested this | A sideways slide should have left blank bands. It did not: the operators reframed each slide to fill the raster. 1.5% recall. |
| **Which separation is red** | **In principle** | The spectrum slides are the intended universal key. In practice they are too faint to read — see below. |
| **Scale, place, date, subject** | **No** | Some slides carry engraved scale bars, which is a real partial answer. Nothing carries a subject. |

The pattern is worth naming. **Every ambiguity the record actually solves, it solves with a
self-checking picture** — the circle for timing, the pulsar map for handedness. Every ambiguity it
fails, it fails because nobody drew a picture for it. An orientation mark would have been a mark.
A grey wedge would have been a wedge. The technique that worked was simply not applied again.

Note on words: these levels describe *what a recipient could recover*. They are not the provenance
tiers in `pipeline/provenance.py`, which describe *what our decoder was allowed to look at*. The
two are related — provenance tier 0 is roughly "levels 1 and 2" — but they are different scales
and it is worth not confusing them.

### Mirroring, and why it is the dangerous one

Mirroring deserves separating from rotation, because it fails differently and worse.

A rotation is obvious once you look: a photograph on its side announces itself. A **mirror does
not**. A laterally inverted photograph of an unfamiliar world is exactly as plausible as the real
one — there is no internal evidence, because the recipient has never seen the subject. The error
is silent and permanent.

It is also not hypothetical among *humans who have the originals*. Aligning our decodes against
published reproductions turned up **fifteen plates printed backwards** — the Thai elephant, the
Cameroon house, the Bali dancer, the Guatemalan women, the Japanese school, the Greek fishermen,
the Cappadocia farmer, the Australian grape picker, Bangkok's traffic, the whistling swans, and
the solar spectrum. If publishers on the source planet flip one plate in eight, a recipient with
no reference at all has no chance of catching it.

Which is what makes the pulsar map matter so much. It is the one piece of chirality the record
carries that a recipient can independently verify, and it settles the question for every image at
once — provided the mirror was introduced by the *decoder*. It cannot help where a mirror was
introduced upstream, by a slide mounted backwards in 1977, which is a per-image error the record
has no way to signal.

### Polarity: is more signal brighter or darker?

The image is stored as a negative. Get this backwards and every photograph is inverted.

The circle does **not** help — a dark ring on a light field and a light ring on a dark field are
both circles. Barry resolved it by human knowledge: a photograph of the Moon only makes sense if
the shadowed limb is as dark as the surrounding space.

There is a possible universal escape. In television practice the sync pulses sit *beyond* black,
so the picture only ever occupies one side of the blanking level. A recipient who noticed that
the signal excursions are asymmetric about the blanking reference could deduce polarity from
that alone. We believe this works here, but it is inference from an engineering convention that
happens to be sensible, not something the record states.

**A single unambiguous grey wedge — black through white, in a known order — would have settled
it forever.** The cover shows a grey scale, but the images do not contain a labelled one.

### Which way is up?

Barry's decoder carries a per-frame rotation table, hand-made by looking at the pictures. Nothing
in the encoding says which edge is the top.

We tried to do better than that verdict and failed, which turned out to be the more useful result.
The hypothesis was that being *turned* should leave a trace even if being *upright* does not: the
scan converter scans a fixed 4:3 raster, so a portrait slide projected into it ought to leave
featureless bands down the left and right. That is a property of the signal, not of the scene, so
an alien could read it. It would at least tell you **which** images want a quarter turn, even if
not which way.

It is false. The detector finds 1 of the 66 turned frames — 1.5% recall — and one raw frame
explains why: the human skeleton is a portrait subject lying on its side, filling the raster edge
to edge, with no surround at all. The 1977 operators reframed every slide to fill the frame. The
content-box aspect ratio settles it: 1.44 for turned slides against 1.40 for upright ones, where
the hypothesis predicted 0.75 against 1.33. (`pipeline/orient_blind.py`.)

So the honest verdict is worse than "the record does not say which edge is up." It is: **the
record does not say which images were turned, either.** An alien recovers 116 correct photographs
and cannot tell that 66 of them are lying on their side, because turning them left no trace.

A capable AI could still guess statistically — skies are bright and above, faces have a canonical
orientation, gravity organises built structures. But every one of those priors is Earth biology
and Earth architecture. It is inference about *us*, not decoding.

**One asymmetric orientation mark per frame would have cost nothing.**

### The one orientation check the record *does* provide

There is an escape we had not credited, and it is global rather than per-image.

Four raster conventions are consistent with the cover: trace 0 on the left or the right, and the
start of a trace at the top or the bottom. Two of those four are mirror images of the other two,
so getting it wrong flips every one of the 116 pictures. Nothing in the picture data can catch
that — a mirrored photograph of a stranger's world looks exactly as plausible as the real one.

But **image 2 is the pulsar map, and the pulsar map is also engraved on the cover.** The cover is
bolted to the record; a recipient has both. The map is strongly chiral — fourteen lines at
fourteen distinct angles, each labelled with its pulsar's period in binary — so comparing the
decoded image against the engraving pins the handedness of the entire raster at a stroke, and
with it every mirror question for all 116 images.

That is a real, designed-in self-test, and it is the companion to the circle: **the circle checks
your timebase, the pulsar map checks your handedness.** Neither needs a word of language.

We have not yet run the comparison, because it needs a rectified image of the cover engraving,
which is not in `docs/reference/`. It is recorded here as a specific thing to do, not as a result.

### Which separation is red, green and blue?

Twenty images are stored as three monochrome scans through colour filters. Which is which?

The intended key is genuinely elegant: the record includes photographs of the solar spectrum,
and any competent chemist anywhere recognises hydrogen's absorption lines and can therefore
attach wavelengths to positions. From that you learn which separation is the long-wavelength one.
Universal, and rather beautiful.

In practice it is too faint. Barry found the spectrum slides nearly featureless and the
absorption lines almost impossible to see — and the ambiguity persists among *humans* today:
Barry's table orders the separations red, green, blue; another published decoder asserts the
opposite. We are currently settling it by looking at which separation makes a desert bright and
an ocean dark, which is exactly the kind of Earth-specific reasoning the exercise forbids.

**A key that its own designers' species cannot read is not yet a key.** Drawing dark bars at the
absorption wavelengths, rather than relying on a photograph of a real spectrum, would have fixed
this.

### The circle nails the shape, but not the frame

The circle does its job: measured against it, the isotropy — how much trace-time equals one trace
of width — comes out at 7.4406 ± 0.0033 bins per trace, consistent across seven circles in five
different images.

What it cannot tell you is where the frame *ends*. It turns out the cover's "512" is the
converter's nominal trace count, not the width of the 4:3 picture: an exactly-4:3 area is about
503 traces, and the hardware scans roughly 535. A recipient trusting "512 traces, 4:3" would
render every image about 1.8% too wide — a small error, but one the circle alone will not reveal,
because the circle is not centred in the frame and does not touch its edges.

**A frame marker — anything that says "the picture stops here" — would have closed this.** As it
is, you have to infer the frame edge from the scan converter's own behaviour, which is exactly
the Earth-specific knowledge a recipient does not have.

## Level 4 — Not decodable, and not meant to be

Meaning. A recipient can recover a sharp photograph of a supermarket and learn nothing about
commerce, refrigeration, or queuing. The record includes bootstrapping diagrams — number
systems, chemical notation, physical units — which are real attempts at building a dictionary,
but they get you to quantities, not to culture.

This is not a flaw. No message solves it.

## Level 5 — Things *we* still cannot explain

Honesty requires listing these too. On these points we are exactly as stuck as any recipient:

- **The ~1.7 second gap between images.** Barry flagged it in 2017: far more than framing
  requires, and it could have carried a couple of dozen more pictures. We have confirmed it is a
  framing signal at twice the line period and found no payload in it. Neither of us knows why it
  is so long.
- **Image order and channel assignment.** Which images come first, and why they are split across
  two audio channels, appears to be arbitrary. We only mapped it by recognising the pictures.
- **Why 512.** Nothing explains the choice.

---

## A newer line, drawn deliberately

Since this document was written the project acquired the original photographs — 47 slides from
NASA, and the whole set from the reissue's book. That creates a temptation and a boundary.

The boundary: **the decoder never sees them.** `decode.py` imports three numeric modules and
reads one WAV file, and an automated check walks its import graph and fails the build if any
reference path appears. The photographs are used to *score* the decoder, never to produce it.

Even scoring is split. Half the references are a development set, used freely; the other half is
held out and looked at once. Tuning against all of them would be overfitting the test set — the
same error as cheating, only slower and easier to miss.

And there is a third, deliberately dishonest tier: an **oracle** decoder allowed to pick its
settings per image by consulting the true slide. It is useless as a decoder, since it needs the
answer to produce the answer, but the gap between it and the blind decoder measures exactly what
outside knowledge is worth. Currently that gap is **+41%**, and the largest single cause is that
our picture-gate position is one global constant where it should be measured per frame — which
is recoverable from the signal alone. In other words, most of what cheating buys is not knowledge
an alien lacks. It is work we have not finished.

The calibration circle sits on the honest side of this line, which is worth saying plainly: using
it is sanctioned by the record's own cover, which draws the circle and says image 1 is one. A
recipient who read the cover can calibrate geometry, sharpness, noise and black-and-white levels
from that frame exactly as we do.

## The irony worth stating plainly

Our decode is *better* than the 2017 one and *less* alien-reproducible. We used a patent, a
television standard, a national grid frequency, a hand-made frame table, and an encyclopaedia of
image titles. None of that survives the trip.

But — and this is the part that matters — **every shortcut we took has an empirical equivalent
already demonstrated in the signal.** The patent told us to look for 262.5 dots; the spectrum
told us they were there. The standards told us why the sync alternates; clustering the pulses
would have shown us that it does.

A recipient with a good analysis engine and patience recovers levels 1 and 2 in full. They will
get sharp, correctly-framed, correctly-scaled pictures.

They may well hang them upside down, and in the wrong colours.

---

## If you are designing the next one

Cheap additions, in rough order of value:

1. **A grey wedge in every frame**, in a stated order. Settles polarity and the intensity
   transfer at a stroke.
2. **An orientation mark in every frame.** Any asymmetric glyph in a fixed corner.
3. **Make the colour key explicit** — drawn bars at known wavelengths, not a photograph of a
   spectrum and a hope.
4. **An index.** A frame number, in the same binary the cover teaches, in the corner of each
   image. Order, count, and loss detection, free.
5. **Keep the circle.** It is the best thing on the cover, and it earned its place forty-nine
   years later on this project — it is still what tells us our decoder works.

*See [DECODING.md](DECODING.md) for how the decoding actually works, and the
[README](README.md) for what this project adds.*
