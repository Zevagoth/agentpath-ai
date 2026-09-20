"""Read the words inside each box, and beside each arrow.

Separated from the geometry because the two fail independently and are fixed independently. A
node whose text is misread is still a node in the right place with the right arrows; the
topology does not depend on the reading. Keeping them apart is what lets the evaluation say
whether a failure was OCR or structure.

Two things make this harder than "run tesseract on the image":

**Polarity varies.** These renderings use a different background colour per chart, and the node
fill and text colour follow. Dark text on a light fill and light text on a dark fill both occur,
so the crop's polarity is decided per node by comparing its border to its interior, and inverted
when needed. Tesseract expects dark text on light.

**The crops are small.** A node is around 30 px tall, which is below tesseract's comfortable
range, so each crop is upscaled before recognition. Measured on dev: upscaling by 3 with cubic
interpolation moved label recall substantially, and page-segmentation mode 7 ("one text line")
beats the default block mode on a single-line box.
"""

from __future__ import annotations

import csv
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import structlog

from agentpath_ingestion.corpora.diagram import DetectedEdge, DetectedNode

logger = structlog.get_logger(__name__)

#: Tesseract is invoked as a subprocess rather than through ``pytesseract``.
#:
#: Not a style preference. ``pytesseract`` requires Pillow to hand a numpy array to the binary,
#: Pillow is not a dependency here, and without it ``image_to_data`` hangs rather than failing -
#: which cost an hour of looking in the wrong place. The binary itself answers a crop in 0.07 s.
#:
#: Calling it directly also means the OpenMP thread limit can be set per call. Tesseract spawns
#: a thread pool by default, and running it inside a loop over a hundred small crops spends more
#: time creating threads than recognising text.
TESSERACT_BINARY = "tesseract"

#: Per-call timeout. A crop that takes longer than this is a crop something is wrong with.
TESSERACT_TIMEOUT = 20.0

#: Crops are upscaled by this before recognition. A node here is ~30 px tall and tesseract wants
#: a capital height of roughly 30-40 px on its own.
OCR_UPSCALE = 3

#: LSTM engine only, rather than tesseract's default of running the legacy engine as well and
#: combining them. Measured on 132 dev crops: exact label equality rose from 64.4% to 85.6% and
#: the call got *faster*, 65 ms against 88 ms. The legacy engine is trained on scanned documents
#: and these are 30-pixel synthetic renderings; its vote was doing harm.
TESSERACT_ENGINE = "1"

#: Pixels trimmed from each side of a node crop before reading, so the border stroke is not
#: mistaken for a glyph. Five, not three: the box comes from a threshold sweep that on
#: low-contrast charts settles at 8 or 14 and reports a box a pixel or two larger than the node,
#: which pulls the border inside the crop. Measured, with the edge-word rule below: 91.7% exact
#: at a trim of 5 against 90.9% at 3.
BORDER_TRIM = 5

#: "Uniform block of text", not "one text line". Counter-intuitive for a one-line label, but
#: measured: psm 6 reads the words right on every one of 132 dev crops where psm 7 misses 4.5%.
#: A Mermaid label wraps onto two lines when it is long, and psm 7 promises a single line.
NODE_PSM = "6"

#: Same, for the free-floating text beside an arrow, which really is one short line.
EDGE_LABEL_PSM = "7"

#: A recognised word whose box touches the left or right edge of the crop, and which is at most
#: this long, is the node's border stroke read as a glyph rather than a word.
#:
#: This is deliberately a test on the word's *position*, not on its text. Stripping short tokens
#: from the ends of the reading would score the same on this corpus and would quietly destroy a
#: real label like "Go to step 3"; dropping a word drawn on top of the border cannot.
EDGE_WORD_MAX_LENGTH = 2

#: How close to the crop edge counts as touching it, in upscaled pixels - so about 3 pixels of
#: the original, which covers a 1-2 pixel border stroke and its anti-aliasing.
EDGE_WORD_MARGIN = 10

#: Words tesseract is this unsure of are not read at all. A glyph invented from a stroke scores
#: far lower than a real word, and 50 sits in the gap. Measured: 95.5% exact with this and the
#: edge rule together, against 91.7% with the edge rule alone.
MIN_WORD_CONFIDENCE = 50.0

#: How far from an arrow's midpoint to look for its label.
EDGE_LABEL_RADIUS = 26

#: Below this mean confidence the text is reported but flagged, so a downstream consumer can
#: treat it as unread rather than as read-and-wrong.
MIN_TEXT_CONFIDENCE = 0.45

_WHITESPACE = re.compile(r"\s+")


def _normalise_polarity(crop: np.ndarray) -> np.ndarray:
    """Return a greyscale crop with dark text on a light field, whatever the original was.

    Polarity is decided from the crop's own border: the border is the node's fill, and if the
    fill is darker than the average of the interior then the text must be lighter than the fill,
    so the crop is inverted.
    """
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if grey.size == 0:
        return grey
    border = np.concatenate([grey[0, :].ravel(), grey[-1, :].ravel(), grey[:, 0].ravel(), grey[:, -1].ravel()])
    # The fill is whatever the border says; the text is the minority population inside.
    fill = float(np.median(border))
    interior = grey[1:-1, 1:-1] if grey.shape[0] > 2 and grey.shape[1] > 2 else grey
    if interior.size == 0:
        return grey
    # If the darkest content is not much darker than the fill, the text is probably lighter.
    darker = fill - float(np.percentile(interior, 5))
    lighter = float(np.percentile(interior, 95)) - fill
    return cv2.bitwise_not(grey) if lighter > darker else grey


def _read(grey: np.ndarray, psm: str) -> tuple[str, float]:
    """Run tesseract on one crop and return its text with a mean per-word confidence in 0..1.

    TSV output is requested because it carries a confidence per word. Plain text output would
    give the words with no way to tell a confident reading from a guess, and the confidence is
    what lets a downstream consumer treat a label as unread rather than as read-and-wrong.
    """
    if grey.size == 0:
        return "", 0.0
    upscaled = cv2.resize(grey, None, fx=OCR_UPSCALE, fy=OCR_UPSCALE, interpolation=cv2.INTER_CUBIC)
    # Otsu after upscaling: once polarity is normalised the crop is bimodal by construction.
    _, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    with tempfile.TemporaryDirectory(prefix="agentpath-ocr-") as directory:
        source = Path(directory) / "crop.png"
        cv2.imwrite(str(source), binary)
        try:
            completed = subprocess.run(
                [TESSERACT_BINARY, str(source), "stdout", "--psm", psm, "--oem", TESSERACT_ENGINE, "tsv"],
                capture_output=True,
                text=True,
                timeout=TESSERACT_TIMEOUT,
                # One thread. See TESSERACT_BINARY for why.
                env={"OMP_THREAD_LIMIT": "1", "PATH": "/usr/bin:/bin:/usr/local/bin"},
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("diagram_ocr.failed", error=str(exc))
            return "", 0.0

    if completed.returncode != 0:
        logger.warning("diagram_ocr.nonzero_exit", code=completed.returncode, detail=completed.stderr[:160])
        return "", 0.0

    crop_width = binary.shape[1]
    words: list[str] = []
    confidences: list[float] = []
    reader = csv.DictReader(completed.stdout.splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            confidence = float(row.get("conf") or -1)
            left = int(row.get("left") or 0)
            width = int(row.get("width") or 0)
        except ValueError:
            continue
        if confidence < MIN_WORD_CONFIDENCE:
            # Not read, rather than read badly.
            continue
        touches_edge = left <= EDGE_WORD_MARGIN or left + width >= crop_width - EDGE_WORD_MARGIN
        if touches_edge and len(text) <= EDGE_WORD_MAX_LENGTH:
            # The node's own border, recognised as a glyph. See EDGE_WORD_MARGIN.
            continue
        words.append(text)
        confidences.append(confidence / 100.0)

    joined = _WHITESPACE.sub(" ", " ".join(words)).strip()
    return _clean(joined), float(np.mean(confidences)) if confidences else 0.0


#: Characters tesseract emits when it reads a node's own border as a glyph. Measured on the
#: white-background charts, where the border sits closer to the text after the low threshold
#: widens the detected box: readings came back as '| s_ituations |' and ':voicelet yeastiest:'.
_EDGE_JUNK = re.compile(r"^[^0-9A-Za-z]+|[^0-9A-Za-z]+$")

#: A lone symbol between real words is border noise too, not a word.
_LONE_SYMBOL = re.compile(r"\s+[^0-9A-Za-z]{1,2}(?=\s|$)")


def _clean(text: str) -> str:
    """Strip border artefacts from a reading without altering the words themselves.

    Only leading, trailing and isolated non-alphanumeric runs are removed. Nothing inside a word
    is touched: a label this corpus generated is a real dictionary word, and "correcting" its
    spelling would be inventing text rather than reading it.
    """
    cleaned = _LONE_SYMBOL.sub("", text)
    cleaned = _EDGE_JUNK.sub("", cleaned)
    return _WHITESPACE.sub(" ", cleaned).strip()


def read_node_text(image: np.ndarray, nodes: list[DetectedNode]) -> list[DetectedNode]:
    """Fill in each node's text. Geometry is untouched; only the text fields change."""
    from dataclasses import replace

    updated: list[DetectedNode] = []
    for node in nodes:
        y0 = max(node.y + BORDER_TRIM, 0)
        y1 = min(node.y + node.height - BORDER_TRIM, image.shape[0])
        x0 = max(node.x + BORDER_TRIM, 0)
        x1 = min(node.x + node.width - BORDER_TRIM, image.shape[1])
        if y1 <= y0 or x1 <= x0:
            updated.append(node)
            continue
        crop = image[y0:y1, x0:x1]
        text, confidence = _read(_normalise_polarity(crop), NODE_PSM)
        updated.append(replace(node, text=text, text_confidence=confidence))
    return updated


def read_edge_labels(image: np.ndarray, edges: list[DetectedEdge], nodes: list[DetectedNode]) -> list[DetectedEdge]:
    """Look for a short label beside each arrow, e.g. ``Yes`` or ``No``.

    Searched in a window around the midpoint between the arrow's head and tail, excluding any
    node's interior so a node's own words are not read as an arrow label. An arrow with no label
    is the normal case on this corpus and produces an empty string rather than a guess.
    """
    from dataclasses import replace

    updated: list[DetectedEdge] = []
    for edge in edges:
        if edge.source_point is None or edge.target_point is None:
            updated.append(edge)
            continue
        mid_x = (edge.target_point[0] + edge.source_point[0]) // 2
        mid_y = (edge.target_point[1] + edge.source_point[1]) // 2
        if any(node.contains(mid_x, mid_y) for node in nodes):
            updated.append(edge)
            continue

        y0 = max(mid_y - EDGE_LABEL_RADIUS, 0)
        y1 = min(mid_y + EDGE_LABEL_RADIUS, image.shape[0])
        x0 = max(mid_x - EDGE_LABEL_RADIUS, 0)
        x1 = min(mid_x + EDGE_LABEL_RADIUS, image.shape[1])
        if y1 <= y0 or x1 <= x0:
            updated.append(edge)
            continue

        text, _confidence = _read(_normalise_polarity(image[y0:y1, x0:x1]), EDGE_LABEL_PSM)
        # Only short, word-like results are believed: a fragment of a crossing curve reads as
        # punctuation, and accepting that as a branch label would be worse than no label.
        cleaned = text.strip()
        if cleaned and len(cleaned) <= 12 and re.fullmatch(r"[A-Za-z][A-Za-z ]*", cleaned):
            updated.append(replace(edge, label=cleaned))
        else:
            updated.append(edge)
    return updated


__all__ = [
    "BORDER_TRIM",
    "EDGE_LABEL_RADIUS",
    "EDGE_WORD_MARGIN",
    "EDGE_WORD_MAX_LENGTH",
    "MIN_TEXT_CONFIDENCE",
    "MIN_WORD_CONFIDENCE",
    "NODE_PSM",
    "OCR_UPSCALE",
    "TESSERACT_ENGINE",
    "read_edge_labels",
    "read_node_text",
]
