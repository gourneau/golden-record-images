"""Recall-and-ranking for the Ozma book reference scans.

Four stages, each a subcommand, in this order:

  fetch    pull book pages n20..n74 at _w1200 from archive.org (polite, sequential)
  ocr      cache the DjVu text layer and pull out the per-figure "104." labels
  detect   find every photograph rectangle on each page with OpenCV, over-detecting
           on purpose, and write the crops to docs/reference/candidates/
  rank     score every candidate against our own decoded thumbnails and emit a
           ranked shortlist per canonical image number to
           docs/reference/candidates.json

Nothing here ever feeds pixel recovery. These reference scans are used only to
label and verify decodes that were produced from the WAV alone.

House rule that shapes the whole file: a wrong crop is worse than no crop. The
ranker therefore reports scores and margins and refuses to collapse a tie; the
final call belongs to a later vision pass.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "docs" / "reference" / "book"
CANDIDATES = ROOT / "docs" / "reference" / "candidates"
THUMBS = ROOT / "data" / "thumbs"
CATALOG = ROOT / "web" / "public" / "data" / "catalog.json"

ITEM = "voyager-golden-record-book-ozma"
PAGE_URL = f"https://archive.org/download/{ITEM}/page/n{{n}}_w{{w}}.jpg"
DJVU_URL = (
    f"https://archive.org/download/{ITEM}/"
    "The%20Voyager%20Golden%20Record_djvu.xml"
)
UA = "golden-record-decode/1.0 (reference crop tooling; contact via github)"

FIRST_PAGE, LAST_PAGE = 20, 74

# The picture section is printed as spreads. Archive page index nX carries
# printed page X-7, so a spread is (odd n, even n) and the caption block sits at
# the foot of the even-n page, covering BOTH pages of that spread. Verified
# against the DjVu text: caption "3. Definitions of mathematical notation" is on
# n26 while the picture it names is on n25.
#
# Filled from the OCR at runtime, but kept here as the checked-in ground truth so
# that a bad OCR day cannot silently rewrite the prior.
SPREAD_NUMBERS: dict[int, list[int]] = {
    24: [1, 2],
    26: [3, 4, 5, 6, 7],
    28: [8, 9, 10, 11, 12, 13],
    30: [14, 15, 16, 17],
    32: [18, 19, 20, 21, 22, 23, 24, 25],
    34: [26, 27, 28, 29, 30, 31, 32],
    36: [33, 34, 35, 36, 37, 38],
    38: [39, 40, 41, 42, 43, 44, 45],
    40: [46, 47, 48, 49, 50],
    42: [51, 52, 53, 54, 55],
    44: [56, 57, 58, 59, 60, 61, 62],
    46: [63, 64, 65, 66, 67, 68, 69],
    48: [70, 71],
    50: [72, 73, 74],
    52: [75, 76, 77, 78, 79, 80, 81, 82, 83],
    54: [84, 85, 86, 87, 88, 89, 90, 91, 92],
    56: [93, 94, 95, 96, 97, 98, 99, 100, 101],
    58: [102, 103, 104, 105, 106, 107],
    60: [108, 109],
    62: [110, 111, 112],
    64: [113, 114, 115, 116],
}


def spread_of(page: int) -> int | None:
    """Even-n caption page owning this page's spread, or None if outside."""
    even = page if page % 2 == 0 else page + 1
    return even if even in SPREAD_NUMBERS else None


# --------------------------------------------------------------------------
# the decisions
# --------------------------------------------------------------------------
#
# Canonical image number -> the candidate rectangle that IS that picture.
#
# This table is not the ranker's output. The ranker only proposes; every row
# here was settled by looking at our own decode beside the shortlisted crops
# and, where the page prints it, reading the small italic figure number beside
# the plate. Rows 1-44 came from the vision judges that read the 1-24 and 25-48
# ranges; rows 45-116 were judged in the same way in a later pass. Twelve of
# the rows the ranker had flagged "uncertain" (1, 8, 27, 28, 35, 36, 56, 64,
# 65, 73, 84, 102) survived that look unchanged, and one did not: image 64 (the
# Bali dancer) is n045_2, not the ranker's top row n046_5, which is the
# photograph of an old man on the facing page.
#
# House rule: a wrong crop is worse than no crop. Anything that could not be
# settled by eye belongs in UNCERTAIN, not here.
DECISIONS: dict[int, str] = {
    1: "n023_0", 2: "n024_0", 3: "n025_1", 4: "n025_2", 5: "n026_1",
    6: "n026_3", 7: "n026_4", 8: "n027_1", 9: "n027_3", 10: "n027_4",
    11: "n027_5", 12: "n028_0", 13: "n028_2", 14: "n029_1", 15: "n029_3",
    16: "n029_4", 17: "n030_0", 18: "n031_0", 19: "n031_1", 20: "n032_7",
    21: "n032_4", 22: "n032_6", 23: "n032_10", 24: "n032_9", 25: "n032_8",
    26: "n033_1", 27: "n033_4", 28: "n033_6", 29: "n033_7", 30: "n033_3",
    31: "n034_0", 32: "n034_1", 33: "n035_3", 34: "n035_2", 35: "n035_1",
    36: "n036_0", 37: "n036_2", 38: "n036_3", 39: "n037_0", 40: "n038_1",
    41: "n038_3", 42: "n038_5", 43: "n038_4", 44: "n038_6", 45: "n038_7",
    46: "n039_1", 47: "n039_2", 48: "n040_0", 49: "n040_2", 50: "n040_1",
    51: "n041_1", 52: "n041_2", 53: "n041_3", 54: "n042_0", 55: "n042_1",
    56: "n043_1", 57: "n043_3", 58: "n043_5", 59: "n043_4", 60: "n044_0",
    61: "n044_2", 62: "n044_4", 63: "n045_1", 64: "n045_2", 65: "n045_3",
    66: "n046_1", 67: "n046_2", 68: "n046_5", 69: "n046_4", 70: "n047_0",
    71: "n048_0", 72: "n049_1", 73: "n049_2", 74: "n050_0", 75: "n051_2",
    76: "n051_5", 77: "n051_9", 78: "n051_4", 79: "n051_6", 80: "n051_10",
    81: "n051_8", 82: "n051_11", 83: "n052_0", 84: "n053_1", 85: "n053_2",
    86: "n053_3", 87: "n053_4", 88: "n054_3", 89: "n054_7", 90: "n054_6",
    91: "n054_4", 92: "n054_5", 93: "n055_1", 94: "n055_2", 95: "n056_5",
    96: "n056_8", 97: "n056_4", 98: "n056_7", 99: "n056_3", 100: "n056_9",
    101: "n056_10", 102: "n057_0", 103: "n057_1", 104: "n058_0", 105: "n058_3",
    106: "n058_4", 107: "n058_2", 108: "n059_0", 109: "n060_0", 110: "n061_2",
    111: "n061_4", 112: "n062_0", 113: "n063_1", 114: "n063_0", 115: "n064_0",
    116: "n064_1",
}

# Numbers deliberately left without a crop, with the reason. Empty is a claim in
# itself and has to be earned; if a later pass cannot settle a picture, move it
# out of DECISIONS and into here rather than shipping the best guess.
UNCERTAIN: dict[int, str] = {}


# --------------------------------------------------------------------------
# the captions
# --------------------------------------------------------------------------
#
# The Ozma book prints its captions in a small italic block at the foot of the
# even-numbered page of each spread, as "42. Cape Neddick, Maine (Dick Smith)".
# These are transcribed from those blocks, read off the page scans. They are
# NOT the DjVu text layer as it stands: that layer renders the book's italic J
# as an F throughout, which turns Jon Lomberg into "fon Lomberg", Jodi Cobb
# into "Fodi Cobb" and Doñana into "Donana". Every name here was checked
# against the printed block at 1200 px.
#
# A dash in the number means one caption covers a run of plates: the book
# prints "5-6.", "14-16.", "18-25." and "93-94." and each of those numbers
# carries the same words.
#
# Spellings are the book's own, not the world's: it prints "Joe Schershel"
# where the photographer is usually indexed as Scherschel. Quoting a source
# means quoting it.
OZMA: dict[int, tuple[str, str | None]] = {
    1: ("Calibration circle", "Jon Lomberg"),
    2: ("Pulsar map with the Andromeda galaxy as a landmark", "Frank Drake"),
    3: ("Definitions of mathematical notation", "Frank Drake"),
    4: ("Definitions of physical units of measurement", "Frank Drake"),
    5: ("Solar system with planetary measurements", "Frank Drake"),
    6: ("Solar system with planetary measurements", "Frank Drake"),
    7: ("The sun", "Hale Telescope"),
    8: ("Solar spectrum", "Valentin Boriakoff and Dan Mitler"),
    9: ("Mercury", "NASA"),
    10: ("Mars", "NASA"),
    11: ("Jupiter", "NASA"),
    12: ("Earth", "NASA"),
    13: ("Egypt, the Red Sea, the Sinai Peninsula, and the Nile with the "
         "composition of Earth's atmosphere", "NASA"),
    14: ("DNA structure and replication", "Jon Lomberg"),
    15: ("DNA structure and replication", "Jon Lomberg"),
    16: ("DNA structure and replication", "Jon Lomberg"),
    17: ("Cell division", None),
    18: ("Human anatomy", None),
    19: ("Human anatomy", None),
    20: ("Human anatomy", None),
    21: ("Human anatomy", None),
    22: ("Human anatomy", None),
    23: ("Human anatomy", None),
    24: ("Human anatomy", None),
    25: ("Human anatomy", None),
    26: ("Diagram of human sex organs", "Sarah Landry"),
    27: ("Conception silhouette", "Jon Lomberg"),
    28: ("Conception", "Lennart Nilsson"),
    29: ("Fertilized ovum", "Lennart Nilsson"),
    30: ("Fetus silhouette", "Jon Lomberg"),
    31: ("Fetus", "Frank Allan"),
    32: ("Silhouette of male and female", "Jon Lomberg"),
    33: ("Birth", "Wayne Miller"),
    34: ("Nursing mother", "United Nations"),
    35: ("Malaysian man and his daughter", "David Alan Harvey"),
    36: ("Group of children at the United Nations International School",
         "Ruby Mera"),
    37: ("Family portrait silhouette", "Jon Lomberg"),
    38: ("Family portrait", "Nina Leen"),
    39: ("Diagram of continental drift", "Jon Lomberg"),
    40: ("Structure of the Earth with abundance of elements",
         "Jon Lomberg with Steven Soter"),
    41: ("Heron Island", "Jay M. Pasachoff"),
    42: ("Cape Neddick, Maine", "Dick Smith"),
    43: ("The Tetons and the Snake River, Wyoming", "Ansel Adams"),
    44: ("A horseman and his dog cross the desert, Pisco, Peru",
         "George Mobley"),
    45: ("Monument Valley, Arizona", "Ray Manley"),
    46: ("Forest scene with mushrooms, Petersburg, Virginia", "Bruce Dale"),
    47: ("Strawberry leaf", "J. Arthur Herrick"),
    48: ("Fallen leaves, White Sulphur Springs, West Virginia", "Jodi Cobb"),
    49: ("Sequoia and snowflake", "Josef Muench/Robert Sisson"),
    50: ("Tree and daffodils", "Gottlieb Hampfler"),
    51: ("Ichneumon fly", "Stephen Dalton"),
    52: ("Diagram of vertebrate evolution", "Jon Lomberg"),
    53: ("Turbinellidae seashell", "Hermann Landshoff"),
    54: ("Dolphins", "Thomas Nebbia"),
    55: ("Sweeper fish and diver in the Red Sea", "David Doubilet"),
    56: ("Tree toad", "David Wickstrom"),
    57: ("Crocodile in Alia Bay, Lake Rudolf, Kenya", "Peter Beard"),
    58: ("Short-toed eagle, Doñana National Park, Spain", "José Ramón Pons"),
    59: ("Watering hole", "South African Tourism"),
    60: ("Jane Goodall observing chimpanzees, Gombe Stream National Park, "
         "Tanzania", "Vanne Morris-Goodall"),
    61: ("Bushmen hunters in Botswana silhouette", "Jon Lomberg"),
    62: ("Bushmen hunters in Botswana", "Nat Farbman"),
    63: ("Man from Nicaragua", "Yutaka Nagata"),
    64: ("Dancer from Bali", "Donna Grosvenor"),
    65: ("Women in Santiago Atitlan, Guatemala", "Joe Schershel"),
    66: ("Craftsman carving elephants from teak, Chiang Mai, Thailand",
         "Dean Conger"),
    67: ("Elephant, Mae Sariang, Thailand", "Peter Kunstadter"),
    68: ("Elderly farmer from Cappadocia, Turkey", "Jonathan Blair"),
    69: ("Man searching for herbs in Owen County, Indiana", "Bruce Baumann"),
    70: ("Gaston Rébuffat atop Aiguille de Roc, Mont Blanc massif, France",
         "Georges Tairraz"),
    71: ("Gymnast Cathy Rigby", "Phillip Leonian"),
    72: ("Valeriy Borzov winning gold, 1972 Olympic Games, Munich, Germany",
         "Topham Picturepoint"),
    73: ("School in Eastern Hokkaido, Japan", "Yutaka Nagata"),
    74: ("Children with globe at the United Nations International School",
         "Yutaka Nagata"),
    75: ("Cotton harvesting, New South Wales, Australia", "Howell Walker"),
    76: ("Grape picker near Griffith, New South Wales, Australia",
         "David Moore"),
    77: ("Supermarket", "Herman Eckelmann"),
    78: ("Diver and fish, Buck Island Reef National Monument, St. Croix, "
         "U.S. Virgin Islands", "Jerry Greenberg"),
    79: ("Fishermen at Evia, Greece", "Tsagris"),
    80: ("Cooking fish", "Brian Seed"),
    81: ("Chinese dinner party", "Michael Rougier"),
    82: ("Demonstration of licking, eating, and drinking", "Herman Eckelmann"),
    83: ("The Great Wall of China", "Edward Kim"),
    84: ("Home construction in Sangmélima, Republic of Cameroon", "UN"),
    85: ("Amish barn raising", "William Albert Allard"),
    86: ("Rural house in Bishoftu, Ethiopia", "Ray Witlin"),
    87: ("House in Provincetown, Massachusetts", "Robert Sisson"),
    88: ("House in Cloudcroft, New Mexico", "Frank Drake"),
    89: ("Home interior", "James L. Amos"),
    90: ("Taj Mahal", "David Carroll"),
    91: ("Oxford, England", "Douglas R. Gilbert"),
    92: ("Sailboats in the Charles River Basin, Boston, Massachusetts",
         "Ted Spiegel"),
    93: ("Headquarters of the United Nations, day and night",
         "UN/Yutaka Nagata"),
    94: ("Headquarters of the United Nations, day and night",
         "UN/Yutaka Nagata"),
    95: ("Sydney Opera House", "Michael E. Long"),
    96: ("Artisan with drill", "Frank Hewlett"),
    97: ("Factory interior", "Fred Ward"),
    98: ("Visitors at a museum", "David Cupp"),
    99: ("X-ray of a hand", "Herman Eckelmann"),
    100: ("Woman with microscope in a Mogadishu health center", "Rice"),
    101: ("Street scene in Lahore, Pakistan", "B. Wolff"),
    102: ("Rush hour in Bangkok, Thailand", "UN"),
    103: ("Route 13, Ithaca, New York", "Herman Eckelmann"),
    104: ("The Golden Gate and Bridge from Baker Beach, San Francisco, "
          "California", "Ansel Adams"),
    105: ("TurboTrain linking Boston to New York", "Gordon Gahan"),
    106: ("Airplane from runway at Syracuse Hancock International Airport",
          "Frank Drake"),
    107: ("Toronto Pearson International Airport Terminal 1", "Ray Manley"),
    108: ("Sno-Cat hanging over crevasse, Trans-Antarctic Expedition, "
          "1955-1958", "George Lowe"),
    109: ("The Westerbork Synthesis Radio Telescope, The Netherlands",
          "James P. Blair"),
    110: ("Arecibo Observatory, Puerto Rico", "Herman Eckelmann"),
    111: ("Pages from Sir Isaac Newton's Philosophiæ Naturalis Principia "
          "Mathematica, Book 3: De Mundi Systemate", "Jon Lomberg"),
    112: ("Gemini 4 astronaut Ed White, the first American to conduct a "
          "spacewalk", "James McDivitt"),
    113: ("Liftoff of Titan IIIE-Centaur rocket", "NASA"),
    114: ("Whistling swans at sunset, Back Bay National Wildlife Refuge, "
          "Virginia", "David Alan Harvey"),
    115: ("Quartetto Italiano", None),
    116: ("Beethoven's String Quartet No. 13 in B-flat Major, Opus 130: "
          "V. Cavatina, the closing piece of music on the Voyager "
          "Interstellar Record", "Jon Lomberg"),
}


# --------------------------------------------------------------------------
# stage 1: fetch
# --------------------------------------------------------------------------


def fetch_bytes(url: str, tries: int = 3) -> bytes:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:  # noqa: S310
                return r.read()
        except Exception as exc:  # noqa: BLE001 - retry anything transient
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def cmd_fetch(args: argparse.Namespace) -> None:
    BOOK.mkdir(parents=True, exist_ok=True)
    got, skipped, failed = [], [], []
    for n in range(args.first, args.last + 1):
        dest = BOOK / f"n{n:03d}.jpg"
        if dest.exists() and not args.force:
            # Existing holdings are the older _w800 render. Upgrade in place when
            # the wider render is genuinely bigger; otherwise leave it alone.
            import PIL.Image

            if PIL.Image.open(dest).size[0] >= args.width * 1.3:
                skipped.append(n)
                continue
        url = PAGE_URL.format(n=n, w=args.width)
        try:
            blob = fetch_bytes(url)
        except RuntimeError as exc:
            print(f"n{n:03d}  FAILED  {exc}", file=sys.stderr)
            failed.append(n)
            continue
        dest.write_bytes(blob)
        import PIL.Image

        size = PIL.Image.open(dest).size
        print(f"n{n:03d}  {len(blob):>8,} B  {size[0]}x{size[1]}")
        got.append(n)
        time.sleep(args.delay)

    print(f"\nfetched {len(got)}, skipped {len(skipped)}, failed {len(failed)}")
    if failed:
        print("failed pages:", failed)


def cmd_ocr(args: argparse.Namespace) -> None:
    """Cache the DjVu XML and dump the per-page numeric labels it carries.

    Several pages print a small "104." beside each photograph. Those labels,
    with their word boxes, are a much stronger positional prior than the caption
    block, so they are extracted here and consumed by `rank`.
    """
    dest = ROOT / "docs" / "reference" / "book" / "_djvu.xml"
    if not dest.exists() or args.force:
        dest.write_bytes(fetch_bytes(DJVU_URL))
    labels = extract_page_labels(dest)
    out = ROOT / "docs" / "reference" / "book" / "_labels.json"
    out.write_text(json.dumps(labels, indent=1))
    for page, items in sorted(labels.items(), key=lambda kv: int(kv[0])):
        print(page, [(i["n"], i["box"]) for i in items])


def extract_page_labels(xml_path: Path) -> dict[str, list[dict]]:
    """In-page "NN." figure labels with normalised boxes, per archive page."""
    root = ET.parse(xml_path).getroot()
    pages = root.findall(".//OBJECT")
    out: dict[str, list[dict]] = {}
    for idx, page in enumerate(pages):
        if not (FIRST_PAGE <= idx <= LAST_PAGE):
            continue
        pw = float(page.get("width") or 0) or 1.0
        ph = float(page.get("height") or 0) or 1.0
        hits = []
        for line in page.findall(".//LINE"):
            words = [w for w in line.findall("WORD") if w.text]
            # A caption line runs on into prose; a bare figure label is a short
            # line whose only content is "104." (possibly a couple of them).
            if len(words) > 3:
                continue
            for w in words:
                m = re.fullmatch(r"(\d{1,3})[.,]", (w.text or "").strip())
                if not m:
                    continue
                n = int(m.group(1))
                if not 1 <= n <= 116:
                    continue
                spread = spread_of(idx)
                if not spread or n not in SPREAD_NUMBERS[spread]:
                    continue  # a page number or an OCR fantasy, not a figure label
                c = [float(v) for v in (w.get("coords") or "").split(",")[:4]]
                if len(c) != 4:
                    continue
                x0, y1, x1, y0 = c  # djvu order: left, bottom, right, top
                cx, cy = (x0 + x1) / 2 / pw, (y0 + y1) / 2 / ph
                if not (0.02 < cx < 0.98 and 0.02 < cy < 0.96):
                    continue  # hugging a page edge: spine text or folio
                hits.append(
                    {
                        "n": n,
                        "box": [
                            round(x0 / pw, 4),
                            round(y0 / ph, 4),
                            round(x1 / pw, 4),
                            round(y1 / ph, 4),
                        ],
                    }
                )
        if hits:
            out[str(idx)] = hits
    return out


# --------------------------------------------------------------------------
# stage 2: detect photo rectangles
# --------------------------------------------------------------------------


@dataclass
class Candidate:
    id: str
    page: int
    spread: int | None
    box: list[int]  # x, y, w, h in page pixels
    norm_box: list[float]  # same, normalised to page size
    area_frac: float
    aspect: float
    fill: float
    paper_frac: float
    ink_frac: float
    text_block: bool
    file: str


def detect_page(img: np.ndarray, page: int, min_area_frac: float) -> list[Candidate]:
    """Over-detect dark photo blocks on a white page.

    The scans are halftone: a photograph is a cloud of dots, not a solid mass, so
    the close has to be aggressive enough to fuse the dots without swallowing the
    white gutter between two neighbouring pictures. Two closes at different
    scales are run and their boxes pooled, then near-duplicates are merged.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    page_area = float(h * w)

    boxes: list[tuple[int, int, int, int, float]] = []

    # Otsu on a blurred copy, plus a fixed generous threshold: some plates are
    # pale (sky, snow) and Otsu alone loses them into the paper.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    paper = np.percentile(gray, 92)
    _, loose = cv2.threshold(blur, max(60.0, paper - 26), 255, cv2.THRESH_BINARY_INV)

    # Level alone loses the pale plates -- a line diagram on near-white ground, a
    # snowfield, a bright sky. Those are still *textured* where the paper is
    # flat, so a local-standard-deviation mask picks them up.
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, -1, (15, 15))
    sq = cv2.boxFilter(g * g, -1, (15, 15))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    texture = (std > max(3.0, np.percentile(std, 60))).astype(np.uint8) * 255

    for mask in (otsu, loose, texture):
        for k in (9, 17, 29):
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
            closed = cv2.morphologyEx(
                closed,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            cnts, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in cnts:
                x, y, bw, bh = cv2.boundingRect(c)
                area = bw * bh
                if not min_area_frac <= area / page_area <= 0.95:
                    continue
                if bw < 60 or bh < 60:
                    continue
                aspect = bw / bh
                if not 0.18 < aspect < 5.5:
                    continue
                fill = float(cv2.contourArea(c)) / max(area, 1)
                if fill < 0.55:  # ragged blob, e.g. a run of caption text
                    continue
                # Caption text lines: wide, very short, low ink.
                if bh < 0.035 * h and aspect > 3.0:
                    continue
                boxes.append((x, y, bw, bh, fill))

    merged = merge_boxes(boxes)
    out: list[Candidate] = []
    for i, (x, y, bw, bh, fill) in enumerate(merged):
        sub = gray[y : y + bh, x : x + bw]
        # A printed plate -- photograph or line diagram -- covers its rectangle
        # with tinted halftone. A caption block is bare paper with thin type on
        # it, so most of its pixels are still paper. Measured on n026/n038/n044/
        # n052: caption blocks land at 0.68-0.78 paper, every plate below 0.11.
        paper_frac = float((sub > paper - 6).mean())
        ink_frac = float((sub < paper - 45).mean())
        out.append(
            Candidate(
                id=f"n{page:03d}_{i}",
                page=page,
                spread=spread_of(page),
                box=[x, y, bw, bh],
                norm_box=[
                    round(x / w, 4),
                    round(y / h, 4),
                    round(bw / w, 4),
                    round(bh / h, 4),
                ],
                area_frac=round(bw * bh / page_area, 4),
                aspect=round(bw / bh, 3),
                fill=round(fill, 3),
                paper_frac=round(paper_frac, 3),
                ink_frac=round(ink_frac, 3),
                text_block=bool(paper_frac > 0.45),
                file=f"docs/reference/candidates/n{page:03d}_{i}.jpg",
            )
        )
    return out


def iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def merge_boxes(boxes: list[tuple], thresh: float = 0.72) -> list[tuple]:
    """Pool boxes from several thresholds; keep the largest of each cluster.

    Deliberately permissive: two boxes only merge when they are nearly the same
    rectangle, so a big box that contains two smaller ones survives alongside
    them and both readings reach the ranker.
    """
    boxes = sorted(boxes, key=lambda b: -b[2] * b[3])
    keep: list[tuple] = []
    for b in boxes:
        if any(iou(b, k) > thresh for k in keep):
            continue
        keep.append(b)
    return sorted(keep, key=lambda b: (b[1], b[0]))


def cmd_detect(args: argparse.Namespace) -> None:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    all_c: list[dict] = []
    for path in sorted(BOOK.glob("n*.jpg")):
        page = int(path.stem[1:])
        if not (args.first <= page <= args.last):
            continue
        img = cv2.imread(str(path))
        if img is None:
            print(f"unreadable {path}", file=sys.stderr)
            continue
        cands = detect_page(img, page, args.min_area)
        for c in cands:
            x, y, w, h = c.box
            crop = img[y : y + h, x : x + w]
            # Capped at 1200 on the long side: the decode it will be compared
            # against is 512 px wide, so more than that is bytes in the repo
            # buying nothing. The box in this record is in full page pixels.
            s = args.crop_long_side / max(crop.shape[:2])
            if s < 1.0:
                crop = cv2.resize(
                    crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imwrite(
                str(ROOT / c.file), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88]
            )
            all_c.append(asdict(c))
        print(f"n{page:03d}  {len(cands)} candidates")
    out = ROOT / "docs" / "reference" / "candidates" / "_index.json"
    out.write_text(json.dumps(all_c, indent=1))
    print(f"\n{len(all_c)} candidates across "
          f"{len({c['page'] for c in all_c})} pages -> {out}")


# --------------------------------------------------------------------------
# stage 3: rank
# --------------------------------------------------------------------------


def load_decode_index() -> dict[int, list[str]]:
    """Canonical image number -> thumbnail paths, first frame first.

    A colour image is three consecutive frames, one per channel. Any single
    channel is a poor stand-in for the luminance of a printed colour plate -- the
    red channel of a sunset is nearly flat -- so all three are returned and the
    matcher averages them.
    """
    cat = json.loads(CATALOG.read_text())
    out: dict[int, list[str]] = {}
    for f in cat["frames"]:
        n = f.get("imageNumber")
        if n is None:
            continue
        out.setdefault(n, []).append(str(THUMBS / f"{f['id']}.png"))
    return out


def decode_image(paths: list[str]) -> np.ndarray | None:
    """Mean of a colour triplet, or the single mono frame, as 8-bit grey."""
    ims = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        if ims and im.shape != ims[0].shape:
            im = cv2.resize(im, (ims[0].shape[1], ims[0].shape[0]))
        ims.append(im.astype(np.float32))
    if not ims:
        return None
    return np.mean(ims, axis=0).astype(np.uint8)


def gradient_signature(img: np.ndarray, size: int = 96) -> np.ndarray:
    """Size-normalised, contrast-normalised gradient magnitude.

    Raw level is useless here: the decode carries residual droop and a different
    tone curve from a halftone scan. Gradient magnitude keeps the structure and
    throws away the level, and the halftone dots are killed by the blur before
    the gradient is taken.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    img = cv2.resize(img, (size * 2, size * 2), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (0, 0), size / 24.0)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.resize(mag, (size, size), interpolation=cv2.INTER_AREA)
    mag -= mag.mean()
    n = np.linalg.norm(mag)
    return mag / n if n else mag


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).sum())


def sig_variants(img: np.ndarray, size: int) -> dict[str, np.ndarray]:
    """rotation x polarity -> gradient signature.

    Kept as a second opinion for plates the keypoint matcher cannot handle:
    a near-featureless sunset or a flat silhouette produces almost no SIFT
    keypoints, but its coarse layout still correlates.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out: dict[str, np.ndarray] = {}
    for pol_name, pol in (("pos", img), ("neg", 255 - img)):
        for rot in range(4):
            r = np.ascontiguousarray(np.rot90(pol, rot))
            out[f"{rot * 90}:{pol_name}"] = gradient_signature(r, size)
    return out


# --- keypoint geometry -----------------------------------------------------
#
# Gradient correlation alone is not enough. The book plate and the record frame
# are the same photograph but not the same picture: the record image is often a
# different crop, at a different scale, sometimes letterboxed inside a black
# surround, and always through a different tone curve and a halftone screen.
# Global correlation punishes all of that. SIFT keypoints plus a RANSAC
# similarity fit do not care about crop, scale, rotation or level -- they ask
# the only question that matters, "do these two pictures share a rigid set of
# corresponding details".

_SIFT = None


def sift() -> "cv2.SIFT":
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create(nfeatures=1500)
    return _SIFT


def prep_for_sift(img: np.ndarray, long_side: int = 520) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = long_side / max(img.shape[:2])
    if s < 1.0:
        img = cv2.resize(
            img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.createCLAHE(2.0, (8, 8)).apply(img)


def keypoints(img: np.ndarray) -> tuple:
    return sift().detectAndCompute(prep_for_sift(img), None)


def geometric_match(kp_a, des_a, kp_b, des_b) -> tuple[int, int | None, float]:
    """RANSAC inlier count, recovered rotation (nearest 90 deg), scale.

    a -> b, so the rotation returned is the turn that takes the decode onto the
    book crop.
    """
    if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return 0, None, 0.0
    matches = cv2.BFMatcher().knnMatch(des_a, des_b, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2)
            if m.distance < 0.8 * n.distance]
    if len(good) < 6:
        return 0, None, 0.0
    src = np.float32([kp_a[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=6.0,
        maxIters=4000, confidence=0.995,
    )
    if M is None or mask is None:
        return 0, None, 0.0
    n = int(mask.sum())
    ang = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    rot = int(round(ang / 90.0) * 90) % 360
    # A fit that is not close to an upright quarter turn is a fit through noise:
    # the book never prints a plate askew.
    if min(abs(((ang - rot + 180) % 360) - 180), 90) > 12:
        n = 0
    if not 0.25 < scale < 4.0:
        n = 0
    return n, rot, scale


def overlap_of_smaller(a: list[int], b: list[int]) -> float:
    """Intersection as a fraction of the smaller box.

    Used to tell a genuine rival from the same photograph detected twice: the
    detector's coarser closing scales routinely return a block that contains the
    winning rectangle, and that block is not evidence of ambiguity.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return (ix * iy) / max(1, min(aw * ah, bw * bh))


def confidence_of(top: dict, rival: dict | None) -> str:
    """How much this shortlist's first row should be trusted.

    Deliberately conservative, because a wrong crop is worse than no crop. The
    thresholds come from the sixteen pairs whose identity was read off the
    printed figure label: those matched at 144-426 inliers while unrelated
    plates sat at 0-15.
    """
    inl = top["inliers"]
    r_inl = rival["inliers"] if rival else 0
    if inl < 25:
        return "uncertain"
    if inl >= 60 and r_inl < 0.4 * inl:
        return "high"
    if inl >= 25 and r_inl < 0.6 * inl:
        return "medium"
    return "uncertain"


def geom_term(inliers: int) -> float:
    """Inlier count squashed into [0,1). 60 inliers is already conclusive."""
    return inliers / (inliers + 60.0)


def load_orientations() -> dict[int, int]:
    """Canonical image number -> the catalog's quarter-turn count."""
    cat = json.loads(CATALOG.read_text())
    out: dict[int, int] = {}
    for f in cat["frames"]:
        n = f.get("imageNumber")
        if n is not None and n not in out:
            out[n] = f.get("orientation", 0)
    return out


def aspect_term(orientation: int, aspect: float, weight: float) -> float:
    """Agreement between our own orientation flag and the crop's shape.

    Every decoded frame is 512x377, so the frame's own aspect says nothing. The
    catalog's orientation does: a frame flagged as needing a quarter turn is a
    picture that stands upright in the world, and the book prints it upright
    too. Checked on twenty pairs read off the printed figure labels: all
    fourteen orientation-0 images sit in a landscape crop and all six
    quarter-turn images in a portrait crop.
    """
    portrait = aspect < 0.95
    landscape = aspect > 1.05
    if orientation == 0:
        return weight if landscape else (-weight if portrait else 0.0)
    return weight if portrait else (-weight if landscape else 0.0)


def mark_containers(cands: list[dict]) -> None:
    """Flag boxes that swallow two or more other boxes on the same page.

    The detector is run at several closing scales on purpose, so a block of
    neighbouring plates often survives as one big rectangle alongside its parts.
    Those merges are worth keeping -- they are the safety net when a plate is
    never split out -- but they should not win a slot from the tighter crop that
    actually is the photograph.
    """
    by_page: dict[int, list[dict]] = {}
    for c in cands:
        by_page.setdefault(c["page"], []).append(c)
    for group in by_page.values():
        for a in group:
            ax, ay, aw, ah = a["box"]
            n = 0
            for b in group:
                if b is a:
                    continue
                bx, by, bw, bh = b["box"]
                ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
                iy = max(0, min(ay + ah, by + bh) - max(ay, by))
                inside = (ix * iy) / float(bw * bh)
                if inside > 0.9 and bw * bh < 0.7 * aw * ah:
                    n += 1
            a["contains"] = n


def cmd_rank(args: argparse.Namespace) -> None:
    from scipy.optimize import linear_sum_assignment

    cands = json.loads((CANDIDATES / "_index.json").read_text())
    mark_containers(cands)
    decode_paths = load_decode_index()
    orientations = load_orientations()
    size = args.size

    # --- decode side: one entry per canonical image number ------------------
    dec_sig: dict[int, np.ndarray] = {}
    dec_kp: dict[int, tuple[tuple, tuple]] = {}
    for n, paths in sorted(decode_paths.items()):
        im = decode_image(paths)
        if im is None:
            print(f"no thumbnail for image {n}", file=sys.stderr)
            continue
        dec_sig[n] = gradient_signature(im, size)
        # Both polarities: nothing guarantees a decoded frame is not inverted,
        # and SIFT descriptors are built from signed gradients, so an inverted
        # copy does not match its own negative.
        dec_kp[n] = (keypoints(im), keypoints(255 - im))
    print(f"decodes: {len(dec_sig)}")

    # --- full score matrix --------------------------------------------------
    # Roughly four minutes of BFMatcher, so it is cached: the weights below can
    # then be re-tuned and the shortlists rebuilt in a second.
    cache = CANDIDATES / "_scores.json"
    scores: dict[str, dict[int, dict]] = {}
    if args.reuse and cache.exists():
        raw = json.loads(cache.read_text())
        scores = {cid: {int(k): v for k, v in row.items()}
                  for cid, row in raw["scores"].items()}
        print(f"reusing cached scores for {len(scores)} candidates")
    else:
        cand_sig: dict[str, dict[str, np.ndarray]] = {}
        cand_kp: dict[str, tuple] = {}
        for c in cands:
            im = cv2.imread(str(ROOT / c["file"]), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            cand_sig[c["id"]] = sig_variants(im, size)
            cand_kp[c["id"]] = keypoints(im)
        print(f"candidates: {len(cand_kp)}")

        t0 = time.time()
        for i, (cid, ckp) in enumerate(cand_kp.items(), 1):
            row: dict[int, dict] = {}
            vs = cand_sig[cid]
            for n in dec_sig:
                best_n, best_rot, best_pol, best_scale = 0, None, "pos", 0.0
                for pol, (kp, des) in zip(("pos", "neg"), dec_kp[n]):
                    inl, rot, scale = geometric_match(kp, des, *ckp)
                    if inl > best_n:
                        best_n, best_rot, best_pol, best_scale = inl, rot, pol, scale
                ds = dec_sig[n]
                nsig, nvar = max((ncc(v, ds), k) for k, v in vs.items())
                row[n] = {
                    "inliers": best_n,
                    "rotation": best_rot if best_rot is not None
                    else int(nvar.split(":")[0]),
                    "polarity": best_pol if best_n else nvar.split(":")[1],
                    "scale": round(best_scale, 3),
                    "ncc": round(float(nsig), 4),
                }
            scores[cid] = row
            if i % 20 == 0:
                print(f"  {i}/{len(cand_kp)} candidates scored "
                      f"({time.time() - t0:.0f}s)")
        cache.write_text(json.dumps({"size": size, "scores": scores}))

    labels = json.loads((BOOK / "_labels.json").read_text()) \
        if (BOOK / "_labels.json").exists() else {}

    def combined(cid: str, n: int, c: dict) -> tuple[float, float, float]:
        r = scores[cid][n]
        base = geom_term(r["inliers"]) + args.ncc_weight * max(r["ncc"], 0.0)
        in_spread = bool(c["spread"] and n in SPREAD_NUMBERS[c["spread"]])
        prior = (args.spread_bonus if in_spread else 0.0)
        prior += label_bonus(c, n, labels)
        prior += aspect_term(orientations.get(n, 0), c["aspect"], args.aspect_weight)
        penalty = args.text_penalty if c["text_block"] else 0.0
        penalty += args.container_penalty if c.get("contains", 0) >= 2 else 0.0
        return base + prior - penalty, base, prior

    # --- joint per-spread assignment ---------------------------------------
    # A spread's k plates map to k distinct image numbers, so the assignment is
    # solved as a whole rather than greedily: one strong plate cannot steal a
    # number that another plate needs more.
    by_id = {c["id"]: c for c in cands}
    assignment: dict[str, int] = {}
    unmatched: dict[int, list[int]] = {}
    by_spread: dict[int, list[dict]] = {}
    for c in cands:
        if c["spread"] and c["id"] in scores:
            by_spread.setdefault(c["spread"], []).append(c)

    for spread, group in sorted(by_spread.items()):
        nums = [n for n in SPREAD_NUMBERS[spread] if n in dec_sig]
        cost = np.zeros((len(group), len(nums)))
        for i, c in enumerate(group):
            for j, n in enumerate(nums):
                cost[i, j] = -combined(c["id"], n, c)[0]
        ri, ci = linear_sum_assignment(cost)
        taken = set()
        for i, j in zip(ri, ci):
            assignment[group[i]["id"]] = nums[j]
            taken.add(nums[j])
        left = [n for n in nums if n not in taken]
        if left:
            unmatched[spread] = left

    # --- shortlists ---------------------------------------------------------
    shortlists: dict[str, dict] = {}
    for n in sorted(dec_sig):
        rows = []
        for cid in scores:
            c = by_id[cid]
            rank_score, base, prior = combined(cid, n, c)
            r = scores[cid][n]
            rows.append(
                {
                    "candidate": cid,
                    "page": c["page"],
                    "spread": c["spread"],
                    "box": c["box"],
                    "norm_box": c["norm_box"],
                    "file": c["file"],
                    "rotation": r["rotation"],
                    "polarity": r["polarity"],
                    "score": round(rank_score, 4),
                    "inliers": r["inliers"],
                    "ncc": r["ncc"],
                    "scale": r["scale"],
                    "on_caption_spread": bool(
                        c["spread"] and n in SPREAD_NUMBERS[c["spread"]]
                    ),
                    "figure_label": label_bonus(c, n, labels) > 0,
                    "text_block": c["text_block"],
                    "contains": c.get("contains", 0),
                    "aspect": c["aspect"],
                    "hungarian": assignment.get(cid) == n,
                }
            )
        # ncc only breaks exact ties: on the degraded decodes it is noise,
        # and letting it into the score itself cost four cases on the
        # thirty-eight pairs whose identity is known from the page layout.
        rows.sort(key=lambda r: (-r["score"], -r["ncc"]))
        top = rows[: args.top]
        # The joint assignment is a different opinion from the per-image ranking
        # and is sometimes the right one, so it always gets a seat: it is what
        # the caption block's one-to-one constraint actually implies.
        if not any(r["hungarian"] for r in top):
            pick = next((r for r in rows if r["hungarian"]), None)
            if pick is not None:
                top = top[: args.top - 1] + [pick]

        # The runner-up is only a rival if it is a different rectangle. A big
        # merged box that contains the leader, or the leader detected again at
        # another closing scale, is the same photograph and says nothing about
        # ambiguity.
        leader = rows[0]
        rival = next(
            (r for r in rows[1:] if r["page"] != leader["page"]
             or overlap_of_smaller(r["box"], leader["box"]) < 0.5),
            None,
        )
        conf = confidence_of(leader, rival)
        margin = round(leader["score"] - rival["score"], 4) if rival else None
        for r in top:
            r["margin_over_rival"] = margin

        if conf == "uncertain":
            # No appearance evidence worth the name, so the only real evidence
            # left is the caption block's one-to-one constraint. On the ten
            # ground-truth images that land in this bucket the joint assignment
            # is right eight times against the score order's six, so it leads.
            top.sort(key=lambda r: (not r["hungarian"], -r["score"], -r["ncc"]))
        shortlists[str(n)] = {"confidence": conf, "candidates": top}

    out = ROOT / "docs" / "reference" / "candidates.json"
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "what_this_is": (
            "Ranked shortlists of book-page crops for each canonical image "
            "number. Nothing here is a decision: a vision pass judges these. A "
            "wrong crop is worse than no crop, so read 'inliers' and 'margin' "
            "before trusting a row."
        ),
        "fields": {
            "box": (
                "[x, y, w, h] in pixels on the archive page scan, which is "
                "1738x1849 for even pages and 1740x1851 for odd ones. norm_box "
                "is the same rectangle as fractions of the page. The saved crop "
                "is capped at 1200 px on its long side, so it is not always at "
                "page scale."
            ),
            "inliers": (
                "RANSAC inliers from SIFT correspondences between our decoded "
                "frame and the book crop, best over both polarities. This is "
                "the load-bearing number. On sixteen pairs verified by reading "
                "the printed figure label off the page, true pairs scored 144 "
                "to 426 and unrelated pairs 0 to 15; a plate with fewer than "
                "about 25 inliers has not really been matched."
            ),
            "ncc": (
                "normalised cross-correlation of gradient magnitude at "
                f"{size}x{size}, best over 4 rotations x 2 polarities. A weak "
                "second opinion, for plates too flat for keypoints."
            ),
            "rotation": (
                "degrees clockwise the decoded frame must be turned to sit the "
                "same way up as the book crop, recovered from the RANSAC "
                "similarity: np.rot90(decode, -rotation // 90). Meaningless "
                "when inliers is 0, where it falls back to the best gradient "
                "correlation."
            ),
            "polarity": "'neg' means the decode had to be inverted to match.",
            "score": (
                f"geom + {args.ncc_weight}*ncc + {args.spread_bonus} if the "
                "candidate is on the spread whose caption block names this "
                "image + 0.25 if the page prints that figure number beside "
                f"this very rectangle +/- {args.aspect_weight} on whether the "
                "crop's portrait/landscape shape agrees with the catalog's "
                f"orientation flag - {args.text_penalty} if the rectangle looks "
                f"like a block of caption type - {args.container_penalty} if it "
                "swallows two or more other candidate rectangles. "
                "geom = inliers/(inliers+60)."
            ),
            "hungarian": (
                "true if the joint per-spread assignment (Hungarian on the same "
                "score) gave this candidate to this image number. Where the "
                "confidence is 'uncertain' the shortlist is led by this row "
                "rather than by the score, because with no keypoint evidence "
                "the caption block's one-to-one constraint is all there is."
            ),
        },
        "confidence_levels": {
            "high": ">=60 inliers and the best rival plate under 40% of that",
            "medium": ">=25 inliers and the best rival plate under 60% of that",
            "uncertain": (
                "under 25 inliers, or a rival too close to separate. Treat the "
                "rows as leads, not as an answer; several of these top rows are "
                "known to be the wrong photograph."
            ),
        },
        "summary": {
            "images": len(shortlists),
            "candidates_detected": len(cands),
            "pages_searched": len({c["page"] for c in cands}),
            "high": sum(1 for v in shortlists.values()
                        if v["confidence"] == "high"),
            "medium": sum(1 for v in shortlists.values()
                          if v["confidence"] == "medium"),
            "uncertain": sorted(int(k) for k, v in shortlists.items()
                                if v["confidence"] == "uncertain"),
        },
        "spread_numbers": {str(k): v for k, v in SPREAD_NUMBERS.items()},
        "unassigned_numbers_by_spread": {
            str(k): v for k, v in sorted(unmatched.items())
        },
        "shortlists": shortlists,
    }
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}")



# --------------------------------------------------------------------------
# stage 5: crop
# --------------------------------------------------------------------------
#
# The detector's rectangles are generous on purpose -- recall first, so that no
# plate is missed. What ships to the gallery has to be the other way round: the
# whole photograph, and as little as possible of the page it was printed on.
#
# Two things get trimmed, both of them page rather than picture:
#
#   paper      a band of bare page along an edge, which the detector keeps
#              whenever the plate does not sit square in its rectangle.
#   caption    the "13." figure number printed under the plate. It is real
#              evidence -- it is how most of these matches were confirmed --
#              but it is type, not photograph, and it is cut off from the plate
#              by a clear band of paper, which is what finds it.
#
# What is NOT trimmed is any margin that belongs to the picture. Image 1 is a
# single ring on a plain pale field: the field reads as "empty" to every
# activity measure there is, and cutting it would be exactly the mistake the
# judge rejected when it declined the tighter rival crop. So the trimmer only
# ever eats bands that match the PAGE GROUND -- sampled from the page border,
# white on a white page and black on a black one -- and never more than
# MAX_TRIM of a side.

CROPS = ROOT / "docs" / "reference" / "crops"

MAX_TRIM = 0.22          # never eat more than this fraction of a side
PAPER_TOL = 14           # levels below the paper level that still read as paper
FLAT = 14                # p90-p10 within a row/col at or under this is featureless
CAPTION_MAX = 0.11       # a trailing block this small can be a caption strip
CAPTION_GAP = 0.012      # ... if this much clear ground separates it from the plate

# The second trim, for a plate sitting on the book's black ground rather than
# on paper. This one is NOT applied everywhere, and the reason is worth stating:
# black is usually picture. It is the sky Jupiter hangs in, the ground the
# anatomy figures are drawn on, the field around Earth that carries the "12756
# km" tick bars at its left and right edges. Turned loose on all 116 it took a
# quarter off the left of image 12 and moved the Earth off centre. So it fires
# only where a judge called the framing out, which is image 25: the detector
# handed back the whole black page and the pale muscle plate occupies the
# middle three fifths of it.
DARK_TRIM: set[int] = {25}
DARK_MAX = 70            # median at or under this is black ground
DARK_FLAT = 20           # halftone black is noisier than paper
DARK_MIN_BAND = 0.10     # only a band at least this wide is page rather than picture
DARK_KEEP = 0.04         # ... and this much of it is left as margin
DARK_SLACK = 0.02        # ignore a hairline of paper outside the black band


def page_ground(page_img: np.ndarray) -> float:
    """Bare-paper level of the page, from its outer border ring.

    A high percentile and not the median: several plates bleed into the ring,
    and the median of a ring the plate has bled into is the plate, which would
    stop the trimmer from recognising paper at all.
    """
    b = 6
    ring = np.concatenate([
        page_img[:b].ravel(), page_img[-b:].ravel(),
        page_img[:, :b].ravel(), page_img[:, -b:].ravel(),
    ])
    return float(np.percentile(ring, 80))


def _ground_runs(img: np.ndarray, ground: float, axis: int) -> np.ndarray:
    """True where a row (axis=0) or column (axis=1) is bare paper."""
    lines = img if axis == 0 else img.T
    p10 = np.percentile(lines, 10, axis=1)
    p90 = np.percentile(lines, 90, axis=1)
    med = np.median(lines, axis=1)
    return (p90 - p10 <= FLAT) & (med >= ground - PAPER_TOL)


def _dark_runs(img: np.ndarray, axis: int) -> np.ndarray:
    """True where a row or column is featureless black."""
    lines = img if axis == 0 else img.T
    p10 = np.percentile(lines, 10, axis=1)
    p90 = np.percentile(lines, 90, axis=1)
    med = np.median(lines, axis=1)
    return (p90 - p10 <= DARK_FLAT) & (med <= DARK_MAX)


def _trim_dark(mask: np.ndarray, a: int, b: int, n: int) -> tuple[int, int]:
    """Eat a wide black band at either end of [a, b), leaving a margin."""
    keep = max(1, int(n * DARK_KEEP))
    slack = max(2, int(n * DARK_SLACK))
    lo = a
    while lo < a + slack and not mask[lo]:      # step over a hairline of paper
        lo += 1
    start = lo
    while lo < b and mask[lo]:
        lo += 1
    if lo - start >= n * DARK_MIN_BAND:
        a = lo - keep
    hi = b
    while hi > b - slack and not mask[hi - 1]:
        hi -= 1
    end = hi
    while hi > a and mask[hi - 1]:
        hi -= 1
    if end - hi >= n * DARK_MIN_BAND:
        b = hi + keep
    return a, b


def tighten(img: np.ndarray, ground: float,
            dark: bool = False) -> tuple[int, int, int, int]:
    """Box (x0, y0, x1, y1) inside img holding the plate and nothing else."""
    h, w = img.shape[:2]
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rows = _ground_runs(g, ground, 0)
    cols = _ground_runs(g, ground, 1)

    def lead(mask: np.ndarray, n: int) -> tuple[int, int]:
        cap = int(n * MAX_TRIM)
        a = 0
        while a < cap and mask[a]:
            a += 1
        b = n
        while b > n - cap and mask[b - 1]:
            b -= 1
        return a, b

    y0, y1 = lead(rows, h)
    x0, x1 = lead(cols, w)

    # A caption is a short block of type below (or above) the plate, cut off
    # from it by clear ground. Only ever drop one, and only a short one.
    def drop_caption(a: int, b: int, mask: np.ndarray, n: int) -> tuple[int, int]:
        gap = max(2, int(n * CAPTION_GAP))
        lim = int(n * CAPTION_MAX)
        for _ in range(2):                      # foot first, then head
            run = 0
            i = b - 1
            while i > a and not mask[i]:        # walk back over the type
                run += 1
                i -= 1
            if 0 < run <= lim:
                clear = 0
                while i > a and mask[i]:
                    clear += 1
                    i -= 1
                if clear >= gap:
                    b = i + 1
                    continue
            break
        return a, b

    y0, y1 = drop_caption(y0, y1, rows, h)
    # and the same walking down from the top, by mirroring
    ry0, ry1 = drop_caption(h - y1, h - y0, rows[::-1], h)
    y0, y1 = h - ry1, h - ry0

    if dark:
        y0, y1 = _trim_dark(_dark_runs(g, 0), y0, y1, h)
        x0, x1 = _trim_dark(_dark_runs(g, 1), x0, x1, w)

    if (y1 - y0) < h * 0.5 or (x1 - x0) < w * 0.5:
        return 0, 0, w, h                        # implausible; keep the raw box
    return x0, y0, x1, y1


def _contained(inner: list[int], outer: list[int]) -> float:
    """Fraction of the inner rectangle that lies inside the outer one."""
    ax, ay, aw, ah = inner
    bx, by, bw, bh = outer
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy / max(1, aw * ah)


def carve(box: list[int], intruders: list[list[int]]) -> list[int]:
    """Shrink box until it holds none of the intruders, losing as little as possible.

    The detector sometimes returns a whole printed column as one rectangle: on
    page n053 the right-hand column comes back as a single box holding both the
    Amish barn raising (image 85) and the New England house below it (image 87).
    Since both plates were decided independently, each one's rectangle is the
    evidence that the other's is too big, and the container can be cut back
    against it. Cheapest of the four cuts wins, so the plate keeps whichever
    side it was already flush with.
    """
    x, y, w, h = box
    for it in intruders:
        if _contained(it, [x, y, w, h]) < 0.7:
            continue
        ix, iy, iw, ih = it
        options = [
            (ix - x, [x, y, max(1, ix - x), h]),                    # cut right
            (x + w - (ix + iw), [ix + iw, y, max(1, x + w - ix - iw), h]),
            (iy - y, [x, y, w, max(1, iy - y)]),                    # cut bottom
            (y + h - (iy + ih), [x, iy + ih, w, max(1, y + h - iy - ih)]),
        ]
        keep = max(options, key=lambda o: o[0])
        x, y, w, h = keep[1]
    return [x, y, w, h]


def cmd_crop(args: argparse.Namespace) -> None:
    """Cut every decided plate out of the page scan into docs/reference/crops."""
    index = {c["id"]: c for c in
             json.loads((CANDIDATES / "_index.json").read_text())}
    by_page: dict[int, list[list[int]]] = {}
    for cid in DECISIONS.values():
        c = index[cid]
        by_page.setdefault(c["page"], []).append(c["box"])
    CROPS.mkdir(parents=True, exist_ok=True)
    pages: dict[int, np.ndarray] = {}
    report = []

    for n in sorted(DECISIONS):
        cid = DECISIONS[n]
        c = index[cid]
        page = c["page"]
        if page not in pages:
            im = cv2.imread(str(BOOK / f"n{page:03d}.jpg"), cv2.IMREAD_COLOR)
            if im is None:
                raise SystemExit(f"missing page scan for {cid}")
            pages[page] = im
        img = pages[page]
        others = [b for b in by_page[page] if b is not c["box"]]
        x, y, w, h = carve(c["box"], others)
        raw = img[y:y + h, x:x + w]
        ground = page_ground(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if args.no_tighten:
            cut = raw
            box = (0, 0, w, h)
        else:
            box = tighten(raw, ground, dark=n in DARK_TRIM)
            cut = raw[box[1]:box[3], box[0]:box[2]]

        long_side = max(cut.shape[:2])
        if long_side > args.long_side:
            s = args.long_side / long_side
            cut = cv2.resize(cut, (max(1, round(cut.shape[1] * s)),
                                   max(1, round(cut.shape[0] * s))),
                             interpolation=cv2.INTER_AREA)
        out = CROPS / f"{n}.jpg"
        cv2.imwrite(str(out), cut,
                    [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        trimmed = 1 - ((box[2] - box[0]) * (box[3] - box[1])) / (w * h)
        report.append((n, cid, page, cut.shape[1], cut.shape[0], trimmed))
        print(f"{n:3d} {cid:9} page {page:3d}  {cut.shape[1]:4d}x{cut.shape[0]:4d}"
              f"  trimmed {trimmed * 100:4.1f}%")

    print(f"\n{len(report)} crops -> {CROPS}")
    if UNCERTAIN:
        print(f"{len(UNCERTAIN)} left without a crop: "
              f"{', '.join(str(k) for k in sorted(UNCERTAIN))}")


# --------------------------------------------------------------------------
# stage 6: map
# --------------------------------------------------------------------------

MAP = ROOT / "docs" / "reference" / "map.json"
WIKI_KEYS = ("file", "commons", "title", "credit", "licence", "mislabelled",
             "note")


def wikimedia_of(rec: dict) -> dict:
    """The Wikimedia block of an old map record, in either shape it has had.

    Written as its own function because the obvious version of it is wrong and
    silently so. This stage reads the previous map and writes the map, so it
    reads its own output on the second run; the record it writes carries an
    Ozma "credit" at the top level, and a rule that scoops loose top-level keys
    would take that credit for a Wikimedia credit and quietly rewrite the
    photographer of every picture that has no Commons scan. So: prefer the
    nested block, and accept loose keys only when a file or a Commons link is
    there to say they really are about a Wikimedia scan.
    """
    if isinstance(rec.get("wikimedia"), dict):
        return {k: v for k, v in rec["wikimedia"].items() if k in WIKI_KEYS}
    if "file" in rec or "commons" in rec:
        return {k: rec[k] for k in WIKI_KEYS if k in rec}
    return {}


def cmd_map(args: argparse.Namespace) -> None:
    """Rewrite docs/reference/map.json from the decisions, crops and captions.

    Everything the old map held about Wikimedia is carried across untouched,
    including the correction on image 42: the Commons file numbered 42 is the
    Ansel Adams Tetons photograph, which the Ozma book prints as 43. The book
    is authoritative on the numbering, so 42 keeps its "mislabelled" flag and
    the gallery goes on refusing to offer that scan as image 42.
    """
    index = {c["id"]: c for c in
             json.loads((CANDIDATES / "_index.json").read_text())}
    old = json.loads(MAP.read_text()) if MAP.exists() else {}

    out: dict[str, object] = {
        "_about": {
            "what": "Per-image reference: the plate in the Ozma Records book "
                    "that IS this picture, cut out of the page scan, plus the "
                    "printed caption and any Wikimedia scan.",
            "status": {
                "confirmed": "our decode was put beside the shortlisted book "
                             "crops and the plate was identified by eye; on "
                             "most pages the small italic figure number "
                             "printed beside the plate corroborates it",
                "uncertain": "a candidate is held but the match is not "
                             "settled; shown as a lead, never as an answer",
                "none": "no reference held",
            },
            "crops": "docs/reference/crops/<n>.jpg, cut from the page scan at "
                     "the decided rectangle, tightened to drop bare paper and "
                     "the printed figure number, long side 900 px",
            "captions": "transcribed from the caption block at the foot of "
                        "the even page of each spread; the book's own "
                        "spellings, and the italic J the DjVu layer reads as "
                        "an F put back",
            "credit": "credit is the photographer the Ozma book names for the "
                      "photograph. licence describes the Wikimedia scan only, "
                      "and is present only where there is one.",
            "reference_images_never_feed_the_decode": True,
        },
    }

    tally = {"confirmed": 0, "uncertain": 0, "none": 0}
    for n in range(1, 117):
        rec: dict[str, object] = {}
        cid = DECISIONS.get(n)
        crop = CROPS / f"{n}.jpg"
        if cid and crop.exists():
            c = index[cid]
            rec["status"] = "confirmed"
            rec["crop"] = f"docs/reference/crops/{n}.jpg"
            rec["candidate"] = cid
            rec["book"] = f"docs/reference/book/n{c['page']:03d}.jpg"
            rec["bookPage"] = c["page"]
            rec["captionPage"] = spread_of(c["page"])
        elif n in UNCERTAIN:
            rec["status"] = "uncertain"
            rec["note"] = UNCERTAIN[n]
            cid = None
        else:
            rec["status"] = "none"

        if n in OZMA:
            title, who = OZMA[n]
            rec["ozma"] = title
            if who:
                rec["credit"] = who

        w = wikimedia_of(old.get(str(n), {}))
        if n == 43 and not w:
            # The other half of the 42/43 correction. That Commons scan really
            # is the Tetons, so it belongs to 43 -- it is only its filename
            # that is wrong. Withholding it from 42 without offering it on 43
            # would throw away a good scan to punish a bad label.
            w = dict(wikimedia_of(old.get("42", {})))
            w.pop("mislabelled", None)
            w["note"] = ("filed on Commons as image 42; the Ozma book prints "
                         "this photograph as 43 and 42 as Cape Neddick")
        if w:
            rec["wikimedia"] = w
            for k in ("file", "commons", "licence", "mislabelled"):
                if k in w:
                    rec[k] = w[k]
            if "credit" not in rec:
                rec["credit"] = w.get("credit")
        tally[str(rec["status"])] += 1
        out[str(n)] = rec

    MAP.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {MAP}")
    print(f"confirmed {tally['confirmed']}  uncertain {tally['uncertain']}  "
          f"none {tally['none']}")
    for n in range(1, 117):
        if out[str(n)]["status"] != "confirmed":      # type: ignore[index]
            print(f"  not confirmed: {n}")


def label_bonus(c: dict, n: int, labels: dict) -> float:
    """Bonus when the page prints "n." right next to this rectangle."""
    items = labels.get(str(c["page"]))
    if not items:
        return 0.0
    x, y, w, h = c["norm_box"]
    best = 0.0
    for it in items:
        if it["n"] != n:
            continue
        lx0, ly0, lx1, ly1 = it["box"]
        cx, cy = (lx0 + lx1) / 2, (ly0 + ly1) / 2
        # inside the rectangle, or within a short reach of its edge
        pad = 0.03
        if x - pad <= cx <= x + w + pad and y - pad <= cy <= y + h + pad:
            best = max(best, 0.25)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch")
    f.add_argument("--first", type=int, default=FIRST_PAGE)
    f.add_argument("--last", type=int, default=LAST_PAGE)
    f.add_argument("--width", type=int, default=1200)
    f.add_argument("--delay", type=float, default=0.3)
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=cmd_fetch)

    o = sub.add_parser("ocr")
    o.add_argument("--force", action="store_true")
    o.set_defaults(func=cmd_ocr)

    d = sub.add_parser("detect")
    d.add_argument("--first", type=int, default=FIRST_PAGE)
    d.add_argument("--last", type=int, default=LAST_PAGE)
    d.add_argument("--min-area", type=float, default=0.008)
    d.add_argument("--crop-long-side", type=int, default=1200)
    d.set_defaults(func=cmd_detect)

    r = sub.add_parser("rank")
    r.add_argument("--size", type=int, default=96)
    r.add_argument("--top", type=int, default=4)
    r.add_argument("--spread-bonus", type=float, default=0.30)
    r.add_argument("--ncc-weight", type=float, default=0.0)
    r.add_argument("--text-penalty", type=float, default=0.25)
    r.add_argument("--aspect-weight", type=float, default=0.12)
    r.add_argument("--container-penalty", type=float, default=0.2)
    r.add_argument("--reuse", action="store_true",
                   help="reuse the cached candidate x decode score matrix")
    r.set_defaults(func=cmd_rank)

    c = sub.add_parser("crop", help="cut the decided plates for the gallery")
    c.add_argument("--long-side", type=int, default=900)
    c.add_argument("--quality", type=int, default=85)
    c.add_argument("--no-tighten", action="store_true",
                   help="write the detector's raw rectangle, for comparison")
    c.set_defaults(func=cmd_crop)

    m = sub.add_parser("map", help="rewrite docs/reference/map.json")
    m.set_defaults(func=cmd_map)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
