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

## Tier 1 — Engraved on the cover, and genuinely universal

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

## Tier 2 — Not on the cover, but findable in the signal by anyone patient

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

## Tier 3 — Genuinely hard, and arguably design faults

Here the record lets its reader down.

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

A capable AI could guess statistically — skies are bright and above, faces have a canonical
orientation, gravity organises built structures. But every one of those priors is Earth biology
and Earth architecture. It is inference about *us*, not decoding.

**One asymmetric orientation mark per frame would have cost nothing.**

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

## Tier 4 — Not decodable, and not meant to be

Meaning. A recipient can recover a sharp photograph of a supermarket and learn nothing about
commerce, refrigeration, or queuing. The record includes bootstrapping diagrams — number
systems, chemical notation, physical units — which are real attempts at building a dictionary,
but they get you to quantities, not to culture.

This is not a flaw. No message solves it.

## Tier 5 — Things *we* still cannot explain

Honesty requires listing these too. On these points we are exactly as stuck as any recipient:

- **The ~1.7 second gap between images.** Barry flagged it in 2017: far more than framing
  requires, and it could have carried a couple of dozen more pictures. We have confirmed it is a
  framing signal at twice the line period and found no payload in it. Neither of us knows why it
  is so long.
- **Image order and channel assignment.** Which images come first, and why they are split across
  two audio channels, appears to be arbitrary. We only mapped it by recognising the pictures.
- **Why 512.** Nothing explains the choice.

---

## The irony worth stating plainly

Our decode is *better* than the 2017 one and *less* alien-reproducible. We used a patent, a
television standard, a national grid frequency, a hand-made frame table, and an encyclopaedia of
image titles. None of that survives the trip.

But — and this is the part that matters — **every shortcut we took has an empirical equivalent
already demonstrated in the signal.** The patent told us to look for 262.5 dots; the spectrum
told us they were there. The standards told us why the sync alternates; clustering the pulses
would have shown us that it does.

A recipient with a good analysis engine and patience recovers Tiers 1 and 2 in full. They will
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
