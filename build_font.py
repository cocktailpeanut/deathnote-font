#!/usr/bin/env python3
"""
Generate a rough hand-drawn horror-style TrueType font inspired by the
provided reference style. Output is an original design, not a tracing.
"""

from __future__ import annotations

import math
import os
import random
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


# -------------------------------
# Geometry and glyph generation
# -------------------------------

Point = Tuple[float, float]
Contour = List[Point]

UPM = 1000
ASCENDER = 800
DESCENDER = -220
LINE_GAP = 90
CAP_HEIGHT = 700
X_HEIGHT = 500
DEFAULT_ADVANCE = 620


@dataclass
class GlyphShape:
    contours: List[Contour]
    advance: int


@dataclass(frozen=True)
class FontVariant:
    style_name: str
    weight: int
    italic: bool
    embolden: float
    slant: float
    advance_add: int = 0


def _seed_for(text: str) -> int:
    h = 2166136261
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def _add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def _mul(a: Point, s: float) -> Point:
    return (a[0] * s, a[1] * s)


def _length(v: Point) -> float:
    return math.hypot(v[0], v[1])


def _norm(v: Point) -> Point:
    d = _length(v)
    if d < 1e-9:
        return (0.0, 0.0)
    return (v[0] / d, v[1] / d)


def _perp(v: Point) -> Point:
    return (-v[1], v[0])


def _poly_area(contour: Contour) -> float:
    s = 0.0
    n = len(contour)
    for i in range(n):
        x1, y1 = contour[i]
        x2, y2 = contour[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _jitter(pt: Point, rng: random.Random, amount: float) -> Point:
    return (pt[0] + rng.uniform(-amount, amount), pt[1] + rng.uniform(-amount, amount))


def _stroke_segment(
    p0: Point,
    p1: Point,
    width: float,
    rng: random.Random,
    rough: float = 5.0,
) -> Contour:
    v = _sub(p1, p0)
    if _length(v) < 1e-3:
        return []
    t = _norm(v)
    n = _perp(t)

    # Taper and roughness create the chipped, hand-cut feel.
    w0 = width * rng.uniform(0.82, 1.18)
    w1 = width * rng.uniform(0.78, 1.22)
    ext0 = width * rng.uniform(0.08, 0.34)
    ext1 = width * rng.uniform(0.08, 0.34)

    a = _add(_add(p0, _mul(n, +w0 * 0.5)), _mul(t, +ext0))
    b = _add(_add(p1, _mul(n, +w1 * 0.5)), _mul(t, +ext1))
    c = _add(_add(p1, _mul(n, -w1 * 0.5)), _mul(t, +ext1))
    d = _add(_add(p0, _mul(n, -w0 * 0.5)), _mul(t, +ext0))

    tip_start = _add(_add(p0, _mul(t, -ext0 * rng.uniform(0.6, 1.2))), _mul(n, rng.uniform(-0.18, 0.18) * width))
    tip_end = _add(_add(p1, _mul(t, +ext1 * rng.uniform(0.6, 1.2))), _mul(n, rng.uniform(-0.18, 0.18) * width))

    contour = [
        _jitter(a, rng, rough),
        _jitter(b, rng, rough),
        _jitter(tip_end, rng, rough * 0.7),
        _jitter(c, rng, rough),
        _jitter(d, rng, rough),
        _jitter(tip_start, rng, rough * 0.7),
    ]

    # Enforce clockwise contour for consistent non-zero winding fill.
    if _poly_area(contour) > 0:
        contour.reverse()
    return contour


def _stroke_polyline(points: Sequence[Point], width: float, seed: str) -> List[Contour]:
    rng = random.Random(_seed_for(seed))
    contours: List[Contour] = []
    for i in range(len(points) - 1):
        seg = _stroke_segment(points[i], points[i + 1], width, rng)
        if len(seg) >= 3:
            contours.append(seg)
    return contours


def _circleish(cx: float, cy: float, rx: float, ry: float, seed: str, n: int = 10) -> Contour:
    rng = random.Random(_seed_for(seed))
    pts: Contour = []
    for i in range(n):
        ang = (2.0 * math.pi * i) / n
        jx = rng.uniform(-rx * 0.08, rx * 0.08)
        jy = rng.uniform(-ry * 0.08, ry * 0.08)
        pts.append((cx + math.cos(ang) * rx + jx, cy + math.sin(ang) * ry + jy))
    if _poly_area(pts) > 0:
        pts.reverse()
    return pts


def _segments_to_contours(segments: Sequence[Tuple[Point, Point]], width: float, seed: str) -> List[Contour]:
    contours: List[Contour] = []
    for idx, (a, b) in enumerate(segments):
        contours.extend(_stroke_polyline([a, b], width, f"{seed}:{idx}"))
    return contours


# Normalized stroke definitions in a 0..1 box.
# y=0 baseline, y=1 capline for uppercase.
UPPER_STROKES: Dict[str, List[Tuple[Point, Point]]] = {
    "A": [((0.10, 0.00), (0.46, 1.00)), ((0.88, 0.00), (0.46, 1.00)), ((0.26, 0.50), (0.70, 0.50))],
    "B": [
        ((0.12, 0.00), (0.12, 1.00)),
        ((0.12, 1.00), (0.64, 0.92)),
        ((0.64, 0.92), (0.72, 0.74)),
        ((0.72, 0.74), (0.12, 0.52)),
        ((0.12, 0.52), (0.66, 0.45)),
        ((0.66, 0.45), (0.76, 0.24)),
        ((0.76, 0.24), (0.64, 0.04)),
        ((0.64, 0.04), (0.12, 0.00)),
    ],
    "C": [((0.78, 0.92), (0.50, 1.00)), ((0.50, 1.00), (0.18, 0.72)), ((0.18, 0.72), (0.18, 0.24)), ((0.18, 0.24), (0.50, 0.00)), ((0.50, 0.00), (0.82, 0.08))],
    "D": [((0.12, 0.00), (0.12, 1.00)), ((0.12, 1.00), (0.58, 0.92)), ((0.58, 0.92), (0.84, 0.58)), ((0.84, 0.58), (0.78, 0.20)), ((0.78, 0.20), (0.56, 0.04)), ((0.56, 0.04), (0.12, 0.00))],
    "E": [((0.14, 0.00), (0.14, 1.00)), ((0.14, 1.00), (0.84, 0.94)), ((0.14, 0.52), (0.66, 0.52)), ((0.14, 0.00), (0.82, 0.06))],
    "F": [((0.14, 0.00), (0.14, 1.00)), ((0.14, 1.00), (0.84, 0.94)), ((0.14, 0.54), (0.62, 0.54))],
    "G": [((0.82, 0.88), (0.52, 1.00)), ((0.52, 1.00), (0.18, 0.72)), ((0.18, 0.72), (0.18, 0.24)), ((0.18, 0.24), (0.50, 0.00)), ((0.50, 0.00), (0.84, 0.14)), ((0.84, 0.14), (0.84, 0.44)), ((0.84, 0.44), (0.54, 0.44))],
    "H": [((0.14, 0.00), (0.14, 1.00)), ((0.84, 0.00), (0.84, 1.00)), ((0.14, 0.52), (0.84, 0.52))],
    "I": [((0.16, 1.00), (0.84, 1.00)), ((0.50, 0.00), (0.50, 1.00)), ((0.20, 0.00), (0.80, 0.00))],
    "J": [((0.22, 1.00), (0.86, 1.00)), ((0.60, 1.00), (0.60, 0.22)), ((0.60, 0.22), (0.42, 0.00)), ((0.42, 0.00), (0.16, 0.10))],
    "K": [((0.14, 0.00), (0.14, 1.00)), ((0.84, 1.00), (0.14, 0.48)), ((0.14, 0.48), (0.86, 0.00))],
    "L": [((0.14, 1.00), (0.14, 0.00)), ((0.14, 0.00), (0.84, 0.06))],
    "M": [((0.10, 0.00), (0.10, 1.00)), ((0.10, 1.00), (0.48, 0.50)), ((0.48, 0.50), (0.86, 1.00)), ((0.86, 1.00), (0.86, 0.00))],
    "N": [((0.12, 0.00), (0.12, 1.00)), ((0.12, 1.00), (0.84, 0.00)), ((0.84, 0.00), (0.84, 1.00))],
    "O": [((0.50, 1.00), (0.20, 0.72)), ((0.20, 0.72), (0.20, 0.24)), ((0.20, 0.24), (0.50, 0.00)), ((0.50, 0.00), (0.82, 0.24)), ((0.82, 0.24), (0.82, 0.72)), ((0.82, 0.72), (0.50, 1.00))],
    "P": [((0.14, 0.00), (0.14, 1.00)), ((0.14, 1.00), (0.66, 0.92)), ((0.66, 0.92), (0.76, 0.72)), ((0.76, 0.72), (0.64, 0.56)), ((0.64, 0.56), (0.14, 0.52))],
    "Q": [((0.50, 1.00), (0.20, 0.72)), ((0.20, 0.72), (0.20, 0.24)), ((0.20, 0.24), (0.50, 0.00)), ((0.50, 0.00), (0.82, 0.24)), ((0.82, 0.24), (0.82, 0.72)), ((0.82, 0.72), (0.50, 1.00)), ((0.58, 0.24), (0.88, -0.10))],
    "R": [((0.14, 0.00), (0.14, 1.00)), ((0.14, 1.00), (0.66, 0.92)), ((0.66, 0.92), (0.76, 0.72)), ((0.76, 0.72), (0.64, 0.56)), ((0.64, 0.56), (0.14, 0.52)), ((0.40, 0.52), (0.84, 0.00))],
    "S": [((0.80, 0.88), (0.54, 1.00)), ((0.54, 1.00), (0.22, 0.82)), ((0.22, 0.82), (0.70, 0.52)), ((0.70, 0.52), (0.28, 0.20)), ((0.28, 0.20), (0.12, 0.02)), ((0.12, 0.02), (0.76, 0.12))],
    "T": [((0.08, 1.00), (0.90, 1.00)), ((0.50, 1.00), (0.50, 0.00))],
    "U": [((0.14, 1.00), (0.14, 0.26)), ((0.14, 0.26), (0.42, 0.00)), ((0.42, 0.00), (0.74, 0.16)), ((0.74, 0.16), (0.84, 1.00))],
    "V": [((0.10, 1.00), (0.48, 0.00)), ((0.48, 0.00), (0.88, 1.00))],
    "W": [((0.08, 1.00), (0.28, 0.00)), ((0.28, 0.00), (0.50, 0.58)), ((0.50, 0.58), (0.72, 0.00)), ((0.72, 0.00), (0.92, 1.00))],
    "X": [((0.12, 1.00), (0.86, 0.00)), ((0.86, 1.00), (0.12, 0.00))],
    "Y": [((0.10, 1.00), (0.48, 0.52)), ((0.88, 1.00), (0.48, 0.52)), ((0.48, 0.52), (0.48, 0.00))],
    "Z": [((0.12, 1.00), (0.88, 1.00)), ((0.88, 1.00), (0.14, 0.00)), ((0.14, 0.00), (0.90, 0.00))],
}

DIGIT_STROKES: Dict[str, List[Tuple[Point, Point]]] = {
    "0": UPPER_STROKES["O"],
    "1": [((0.42, 0.80), (0.56, 1.00)), ((0.56, 1.00), (0.56, 0.00)), ((0.34, 0.00), (0.82, 0.00))],
    "2": [((0.22, 0.82), (0.50, 1.00)), ((0.50, 1.00), (0.80, 0.80)), ((0.80, 0.80), (0.22, 0.00)), ((0.22, 0.00), (0.86, 0.00))],
    "3": [((0.18, 0.92), (0.76, 1.00)), ((0.76, 1.00), (0.50, 0.52)), ((0.50, 0.52), (0.80, 0.12)), ((0.80, 0.12), (0.20, 0.00))],
    "4": [((0.76, 0.00), (0.76, 1.00)), ((0.14, 0.30), (0.88, 0.30)), ((0.14, 0.30), (0.58, 1.00))],
    "5": [((0.84, 1.00), (0.24, 1.00)), ((0.24, 1.00), (0.24, 0.56)), ((0.24, 0.56), (0.72, 0.56)), ((0.72, 0.56), (0.82, 0.18)), ((0.82, 0.18), (0.20, 0.00))],
    "6": [((0.78, 0.86), (0.52, 1.00)), ((0.52, 1.00), (0.24, 0.56)), ((0.24, 0.56), (0.26, 0.20)), ((0.26, 0.20), (0.52, 0.00)), ((0.52, 0.00), (0.80, 0.18)), ((0.80, 0.18), (0.68, 0.48)), ((0.68, 0.48), (0.28, 0.48))],
    "7": [((0.14, 1.00), (0.88, 1.00)), ((0.88, 1.00), (0.38, 0.00))],
    "8": [((0.50, 1.00), (0.24, 0.76)), ((0.24, 0.76), (0.50, 0.52)), ((0.50, 0.52), (0.80, 0.76)), ((0.80, 0.76), (0.50, 1.00)), ((0.50, 0.52), (0.22, 0.20)), ((0.22, 0.20), (0.52, 0.00)), ((0.52, 0.00), (0.82, 0.20)), ((0.82, 0.20), (0.50, 0.52))],
    "9": [((0.80, 0.44), (0.52, 0.52)), ((0.52, 0.52), (0.24, 0.72)), ((0.24, 0.72), (0.34, 0.94)), ((0.34, 0.94), (0.62, 1.00)), ((0.62, 1.00), (0.82, 0.80)), ((0.82, 0.80), (0.78, 0.00))],
}


def _norm_to_em(pt: Point, width_em: float, y_scale: float = 1.0, y_shift: float = 0.0) -> Point:
    # Normalize x into width_em, y into cap height with optional shift.
    x = pt[0] * width_em
    y = (pt[1] * CAP_HEIGHT * y_scale) + (y_shift * CAP_HEIGHT)
    return (x, y)


def _build_upper(ch: str, width_em: int = 600) -> GlyphShape:
    strokes = UPPER_STROKES[ch]
    segments = [(_norm_to_em(a, width_em), _norm_to_em(b, width_em)) for (a, b) in strokes]
    contours = _segments_to_contours(segments, width=76.0, seed=f"U:{ch}")
    return GlyphShape(contours=contours, advance=width_em + 70)


def _build_digit(ch: str, width_em: int = 580) -> GlyphShape:
    strokes = DIGIT_STROKES[ch]
    segments = [(_norm_to_em(a, width_em), _norm_to_em(b, width_em)) for (a, b) in strokes]
    contours = _segments_to_contours(segments, width=74.0, seed=f"D:{ch}")
    return GlyphShape(contours=contours, advance=width_em + 70)


def _transform_contours(contours: List[Contour], sx: float, sy: float, tx: float, ty: float, seed: str) -> List[Contour]:
    rng = random.Random(_seed_for(seed))
    out: List[Contour] = []
    for contour in contours:
        c2: Contour = []
        for x, y in contour:
            nx = x * sx + tx + rng.uniform(-3.0, 3.0)
            ny = y * sy + ty + rng.uniform(-3.0, 3.0)
            c2.append((nx, ny))
        if _poly_area(c2) > 0:
            c2.reverse()
        out.append(c2)
    return out


def _build_lower(ch: str) -> GlyphShape:
    # Lowercase is stylistically linked, with modified scale and descenders.
    up = _build_upper(ch.upper(), width_em=560)
    descenders = set("gjpqy")
    ascenders = set("bdfhklt")
    if ch in ascenders:
        contours = _transform_contours(up.contours, 0.90, 0.94, 18.0, 0.0, f"L:{ch}:asc")
        adv = 610
    elif ch in descenders:
        contours = _transform_contours(up.contours, 0.88, 0.72, 16.0, -160.0, f"L:{ch}:des")
        tail = _segments_to_contours([((320.0, 120.0), (420.0, -140.0))], width=58.0, seed=f"L:{ch}:tail")
        contours.extend(tail)
        adv = 610
    else:
        contours = _transform_contours(up.contours, 0.88, 0.72, 16.0, 0.0, f"L:{ch}:x")
        adv = 600
    return GlyphShape(contours=contours, advance=adv)


def _build_space() -> GlyphShape:
    return GlyphShape(contours=[], advance=280)


def _build_basic_punct(ch: str) -> GlyphShape:
    w = 540
    width = 62.0
    segs: List[Tuple[Point, Point]] = []
    extra: List[Contour] = []

    if ch == "!":
        segs = [((260, 120), (260, 700)), ((260, 0), (260, 50))]
    elif ch == '"':
        segs = [((180, 430), (180, 700)), ((360, 430), (360, 700))]
    elif ch == "#":
        segs = [((180, 0), (220, 700)), ((360, 0), (400, 700)), ((80, 230), (500, 270)), ((70, 460), (490, 500))]
    elif ch == "$":
        segs = [((280, -60), (280, 760)), ((420, 640), (260, 700)), ((260, 700), (140, 540)), ((140, 540), (360, 410)), ((360, 410), (170, 200)), ((170, 200), (110, 40)), ((110, 40), (410, 120))]
    elif ch == "%":
        segs = [((100, 0), (440, 700))]
        extra = [_circleish(130, 560, 70, 90, "pct:u"), _circleish(410, 130, 70, 90, "pct:l")]
    elif ch == "&":
        segs = [((420, 80), (280, 220)), ((280, 220), (190, 360)), ((190, 360), (250, 560)), ((250, 560), (420, 680)), ((420, 680), (350, 460)), ((350, 460), (140, 180)), ((140, 180), (250, 20)), ((250, 20), (440, 120))]
        w = 620
    elif ch == "'":
        segs = [((260, 440), (260, 700))]
        w = 280
    elif ch == "(":
        segs = [((360, 740), (250, 520)), ((250, 520), (220, 240)), ((220, 240), (360, -40))]
        w = 340
    elif ch == ")":
        segs = [((180, 740), (290, 520)), ((290, 520), (320, 240)), ((320, 240), (180, -40))]
        w = 340
    elif ch == "*":
        segs = [((260, 160), (260, 640)), ((120, 250), (410, 560)), ((410, 250), (120, 560))]
    elif ch == "+":
        segs = [((90, 350), (450, 350)), ((270, 130), (270, 570))]
    elif ch == ",":
        segs = [((250, -120), (290, 120))]
        w = 260
    elif ch == "-":
        segs = [((120, 260), (430, 290))]
        w = 450
    elif ch == ".":
        segs = [((250, 0), (250, 40))]
        w = 260
    elif ch == "/":
        segs = [((90, -60), (430, 760))]
    elif ch == ":":
        segs = [((260, 430), (260, 470)), ((260, 0), (260, 40))]
        w = 260
    elif ch == ";":
        segs = [((260, 430), (260, 470)), ((260, -120), (300, 120))]
        w = 300
    elif ch == "<":
        segs = [((420, 640), (120, 320)), ((120, 320), (420, 40))]
    elif ch == "=":
        segs = [((90, 420), (450, 420)), ((90, 240), (450, 240))]
    elif ch == ">":
        segs = [((120, 640), (420, 320)), ((420, 320), (120, 40))]
    elif ch == "?":
        segs = [((140, 520), (250, 700)), ((250, 700), (410, 610)), ((410, 610), (280, 430)), ((280, 430), (260, 260)), ((260, 40), (260, 0))]
    elif ch == "@":
        segs = [((470, 120), (470, 520)), ((470, 520), (320, 700)), ((320, 700), (150, 560)), ((150, 560), (150, 180)), ((150, 180), (320, 20)), ((320, 20), (420, 160)), ((420, 160), (330, 320)), ((330, 320), (250, 280))]
        w = 700
    elif ch == "[":
        segs = [((320, 740), (180, 740)), ((180, 740), (180, -40)), ((180, -40), (320, -40))]
        w = 320
    elif ch == "\\":
        segs = [((420, -60), (100, 760))]
    elif ch == "]":
        segs = [((180, 740), (320, 740)), ((320, 740), (320, -40)), ((320, -40), (180, -40))]
        w = 320
    elif ch == "^":
        segs = [((100, 420), (260, 700)), ((260, 700), (430, 420))]
    elif ch == "_":
        segs = [((80, -40), (460, -40))]
    elif ch == "`":
        segs = [((280, 530), (220, 700))]
        w = 280
    elif ch == "{":
        segs = [((330, 740), (220, 620)), ((220, 620), (240, 430)), ((240, 430), (160, 350)), ((160, 350), (240, 270)), ((240, 270), (220, 80)), ((220, 80), (330, -40))]
        w = 340
    elif ch == "|":
        segs = [((260, -80), (260, 760))]
        w = 280
    elif ch == "}":
        segs = [((190, 740), (300, 620)), ((300, 620), (280, 430)), ((280, 430), (360, 350)), ((360, 350), (280, 270)), ((280, 270), (300, 80)), ((300, 80), (190, -40))]
        w = 340
    elif ch == "~":
        segs = [((90, 310), (180, 390)), ((180, 390), (300, 300)), ((300, 300), (420, 380))]
    else:
        # Fallback: rough boxed X for any unsupported punctuation.
        segs = [((120, 40), (420, 40)), ((420, 40), (420, 660)), ((420, 660), (120, 660)), ((120, 660), (120, 40)), ((120, 40), (420, 660)), ((420, 40), (120, 660))]

    contours = _segments_to_contours(segs, width=width, seed=f"P:{ch}")
    for contour in extra:
        contours.append(contour)
    return GlyphShape(contours=contours, advance=w)


def build_ascii_glyphs() -> Dict[int, GlyphShape]:
    glyphs: Dict[int, GlyphShape] = {}
    for cp in range(32, 127):
        ch = chr(cp)
        if ch == " ":
            glyphs[cp] = _build_space()
        elif "A" <= ch <= "Z":
            glyphs[cp] = _build_upper(ch)
        elif "a" <= ch <= "z":
            glyphs[cp] = _build_lower(ch)
        elif "0" <= ch <= "9":
            glyphs[cp] = _build_digit(ch)
        else:
            glyphs[cp] = _build_basic_punct(ch)
    return glyphs


def _apply_variant(base: Dict[int, GlyphShape], variant: FontVariant) -> Dict[int, GlyphShape]:
    styled: Dict[int, GlyphShape] = {}
    for cp, shape in base.items():
        if not shape.contours:
            styled[cp] = GlyphShape(contours=[], advance=max(120, shape.advance + variant.advance_add))
            continue

        transformed: List[Contour] = []
        for contour in shape.contours:
            if not contour:
                continue
            cx = sum(x for x, _ in contour) / len(contour)
            cy = sum(y for _, y in contour) / len(contour)
            c2: Contour = []
            for x, y in contour:
                vx = x - cx
                vy = y - cy
                nx = cx + (vx * variant.embolden)
                ny = cy + (vy * variant.embolden)
                # Shear by y to create an italic posture.
                nx += variant.slant * (ny / CAP_HEIGHT) * 120.0
                c2.append((nx, ny))
            if _poly_area(c2) > 0:
                c2.reverse()
            transformed.append(c2)

        # Keep spacing stable; only nudge for heavier/slanted variants.
        adv = int(round(shape.advance + variant.advance_add + (variant.embolden - 1.0) * 28.0))
        styled[cp] = GlyphShape(contours=transformed, advance=max(120, adv))
    return styled


# -------------------------------
# TrueType serialization helpers
# -------------------------------


def _pad4(data: bytes) -> bytes:
    return data + (b"\0" * ((4 - (len(data) % 4)) % 4))


def _checksum(data: bytes) -> int:
    d = _pad4(data)
    s = 0
    for i in range(0, len(d), 4):
        s = (s + struct.unpack(">I", d[i : i + 4])[0]) & 0xFFFFFFFF
    return s


def _int16(v: int) -> bytes:
    return struct.pack(">h", int(v))


def _uint16(v: int) -> bytes:
    return struct.pack(">H", int(v) & 0xFFFF)


def _uint32(v: int) -> bytes:
    return struct.pack(">I", int(v) & 0xFFFFFFFF)


def _fixed_16_16(v: float) -> bytes:
    return _uint32(int(round(v * 65536.0)))


def _longdatetime(dt: datetime) -> int:
    epoch = datetime(1904, 1, 1, tzinfo=timezone.utc)
    return int((dt - epoch).total_seconds())


@dataclass
class EncodedGlyph:
    data: bytes
    xMin: int
    yMin: int
    xMax: int
    yMax: int
    points: int
    contours: int


def encode_simple_glyph(contours: List[Contour]) -> EncodedGlyph:
    if not contours:
        # Empty glyph record (numberOfContours=0 with zero bbox) is acceptable.
        header = struct.pack(">hhhhh", 0, 0, 0, 0, 0)
        return EncodedGlyph(header, 0, 0, 0, 0, 0, 0)

    pts: List[Tuple[int, int]] = []
    end_pts: List[int] = []

    xMin = 10**9
    yMin = 10**9
    xMax = -10**9
    yMax = -10**9

    for contour in contours:
        # Round and dedupe immediate duplicates.
        rounded: List[Tuple[int, int]] = []
        for x, y in contour:
            p = (int(round(x)), int(round(y)))
            if not rounded or rounded[-1] != p:
                rounded.append(p)
        if len(rounded) < 3:
            continue
        # Remove closing duplicate if present.
        if rounded[0] == rounded[-1]:
            rounded.pop()
        if len(rounded) < 3:
            continue

        for x, y in rounded:
            xMin = min(xMin, x)
            yMin = min(yMin, y)
            xMax = max(xMax, x)
            yMax = max(yMax, y)
            pts.append((x, y))
        end_pts.append(len(pts) - 1)

    if not pts:
        header = struct.pack(">hhhhh", 0, 0, 0, 0, 0)
        return EncodedGlyph(header, 0, 0, 0, 0, 0, 0)

    number_of_contours = len(end_pts)
    header = struct.pack(">hhhhh", number_of_contours, xMin, yMin, xMax, yMax)

    body = bytearray()
    for ep in end_pts:
        body += _uint16(ep)

    body += _uint16(0)  # instructionLength

    # Flags: all points are on-curve; always encode 16-bit signed deltas for simplicity.
    flags = bytes([0x01] * len(pts))
    body += flags

    # X deltas then Y deltas.
    prev_x = 0
    x_deltas = bytearray()
    for x, _ in pts:
        x_deltas += _int16(x - prev_x)
        prev_x = x
    body += x_deltas

    prev_y = 0
    y_deltas = bytearray()
    for _, y in pts:
        y_deltas += _int16(y - prev_y)
        prev_y = y
    body += y_deltas

    return EncodedGlyph(bytes(header + body), xMin, yMin, xMax, yMax, len(pts), number_of_contours)


def build_glyf_and_loca(glyph_order: List[int], glyphs: Dict[int, GlyphShape]) -> Tuple[bytes, bytes, List[EncodedGlyph]]:
    encoded: List[EncodedGlyph] = []
    chunks: List[bytes] = []
    offsets: List[int] = [0]

    # glyph_order[0] is .notdef (cp=-1)
    notdef = _segments_to_contours([((80, 0), (80, 700)), ((80, 700), (500, 700)), ((500, 700), (500, 0)), ((500, 0), (80, 0)), ((80, 0), (500, 700)), ((500, 0), (80, 700))], width=52.0, seed=".notdef")
    notdef_enc = encode_simple_glyph(notdef)
    encoded.append(notdef_enc)
    chunks.append(_pad4(notdef_enc.data))
    offsets.append(len(chunks[0]))

    running = len(chunks[0])
    for cp in glyph_order[1:]:
        shape = glyphs[cp]
        enc = encode_simple_glyph(shape.contours)
        encoded.append(enc)
        gdat = _pad4(enc.data)
        chunks.append(gdat)
        running += len(gdat)
        offsets.append(running)

    glyf = b"".join(chunks)
    loca = b"".join(_uint32(o) for o in offsets)
    return glyf, loca, encoded


def build_hmtx(glyph_order: List[int], glyphs: Dict[int, GlyphShape], encoded: List[EncodedGlyph]) -> Tuple[bytes, int, int, int, int]:
    rows = bytearray()
    adv_max = 0
    min_lsb = 10**9
    min_rsb = 10**9
    x_max_extent = -10**9

    for gid, cp in enumerate(glyph_order):
        if cp == -1:
            adv = 620
        else:
            adv = glyphs[cp].advance
        lsb = encoded[gid].xMin if encoded[gid].points else 0
        x_max = encoded[gid].xMax if encoded[gid].points else 0
        rsb = adv - lsb - x_max

        adv_max = max(adv_max, adv)
        min_lsb = min(min_lsb, lsb)
        min_rsb = min(min_rsb, rsb)
        x_max_extent = max(x_max_extent, lsb + x_max)

        rows += _uint16(adv)
        rows += _int16(lsb)

    return bytes(rows), adv_max, min_lsb, min_rsb, x_max_extent


def build_cmap(glyph_order: List[int]) -> bytes:
    gid_by_cp = {cp: gid for gid, cp in enumerate(glyph_order) if cp != -1}

    # Build a format 4 cmap with one segment per contiguous gidDelta run.
    cps = sorted(gid_by_cp)
    segments: List[Tuple[int, int, int]] = []  # (start, end, delta)
    start = cps[0]
    prev = cps[0]
    delta = (gid_by_cp[start] - start) & 0xFFFF
    for cp in cps[1:]:
        d = (gid_by_cp[cp] - cp) & 0xFFFF
        if cp == prev + 1 and d == delta:
            prev = cp
            continue
        segments.append((start, prev, delta))
        start = cp
        prev = cp
        delta = d
    segments.append((start, prev, delta))

    # Required end segment.
    segments.append((0xFFFF, 0xFFFF, 1))

    seg_count = len(segments)
    seg_count_x2 = seg_count * 2
    max_pow2 = 1
    entry_selector = 0
    while max_pow2 * 2 <= seg_count:
        max_pow2 *= 2
        entry_selector += 1
    search_range = max_pow2 * 2
    range_shift = seg_count_x2 - search_range

    sub = bytearray()
    sub += _uint16(4)  # format
    sub += _uint16(0)  # length placeholder
    sub += _uint16(0)  # language
    sub += _uint16(seg_count_x2)
    sub += _uint16(search_range)
    sub += _uint16(entry_selector)
    sub += _uint16(range_shift)

    for _, end, _ in segments:
        sub += _uint16(end)
    sub += _uint16(0)  # reservedPad
    for start, _, _ in segments:
        sub += _uint16(start)
    for _, _, d in segments:
        sub += _uint16(d)
    for _ in segments:
        sub += _uint16(0)  # idRangeOffset

    struct.pack_into(">H", sub, 2, len(sub))

    # cmap header + one encoding record (Windows Unicode BMP)
    cmap = bytearray()
    cmap += _uint16(0)  # version
    cmap += _uint16(1)  # numTables
    cmap += _uint16(3)  # platformID
    cmap += _uint16(1)  # encodingID
    cmap += _uint32(12)  # subtable offset
    cmap += sub
    return bytes(cmap)


def build_name(family: str, style: str = "Regular") -> bytes:
    records: List[Tuple[int, str]] = [
        (1, family),
        (2, style),
        (3, f"1.0;CODX;{family.replace(' ', '')}-{style}"),
        (4, f"{family} {style}"),
        (5, "Version 1.000"),
        (6, f"{family.replace(' ', '')}-{style}"),
    ]

    name_records = []
    string_data = bytearray()
    for name_id, text in records:
        raw = text.encode("utf-16-be")
        off = len(string_data)
        string_data += raw
        name_records.append((3, 1, 0x0409, name_id, len(raw), off))

    # Sort for compliance.
    name_records.sort()

    out = bytearray()
    out += _uint16(0)  # format
    out += _uint16(len(name_records))
    out += _uint16(6 + len(name_records) * 12)
    for rec in name_records:
        out += struct.pack(">HHHHHH", *rec)
    out += string_data
    return bytes(out)


def build_post(italic_angle: float = 0.0) -> bytes:
    out = bytearray()
    out += _uint32(0x00030000)  # format 3.0
    out += _fixed_16_16(italic_angle)
    out += _int16(-90)  # underlinePosition
    out += _int16(55)  # underlineThickness
    out += _uint32(0)  # isFixedPitch
    out += _uint32(0)
    out += _uint32(0)
    out += _uint32(0)
    out += _uint32(0)
    return bytes(out)


def build_os2(avg_adv: int, first_cp: int, last_cp: int, weight_class: int, fs_selection: int) -> bytes:
    panose = bytes([2, 0, 6, 3, 5, 4, 5, 2, 3, 4])
    out = bytearray()
    out += _uint16(0)  # version
    out += _int16(avg_adv)
    out += _uint16(weight_class)  # usWeightClass
    out += _uint16(5)  # usWidthClass
    out += _uint16(0)  # fsType
    out += _int16(650)  # ySubscriptXSize
    out += _int16(700)  # ySubscriptYSize
    out += _int16(0)  # ySubscriptXOffset
    out += _int16(140)  # ySubscriptYOffset
    out += _int16(650)  # ySuperscriptXSize
    out += _int16(700)  # ySuperscriptYSize
    out += _int16(0)  # ySuperscriptXOffset
    out += _int16(350)  # ySuperscriptYOffset
    out += _int16(50)  # yStrikeoutSize
    out += _int16(280)  # yStrikeoutPosition
    out += _int16(0)  # sFamilyClass
    out += panose
    out += _uint32(0x00000001)  # Basic Latin
    out += _uint32(0)
    out += _uint32(0)
    out += _uint32(0)
    out += b"CDX "  # achVendID
    out += _uint16(fs_selection)
    out += _uint16(first_cp)
    out += _uint16(last_cp)
    out += _int16(ASCENDER)
    out += _int16(DESCENDER)
    out += _int16(LINE_GAP)
    out += _uint16(max(0, ASCENDER))
    out += _uint16(max(0, -DESCENDER))
    return bytes(out)


def build_head(xMin: int, yMin: int, xMax: int, yMax: int, created: int, modified: int, mac_style: int) -> bytes:
    out = bytearray()
    out += _uint32(0x00010000)  # version
    out += _uint32(0x00010000)  # fontRevision
    out += _uint32(0)  # checkSumAdjustment (patched later)
    out += _uint32(0x5F0F3CF5)  # magicNumber
    out += _uint16(0x000B)  # flags
    out += _uint16(UPM)
    out += struct.pack(">Q", created)
    out += struct.pack(">Q", modified)
    out += _int16(xMin)
    out += _int16(yMin)
    out += _int16(xMax)
    out += _int16(yMax)
    out += _uint16(mac_style)
    out += _uint16(8)  # lowestRecPPEM
    out += _int16(2)  # fontDirectionHint
    out += _int16(1)  # indexToLocFormat (long)
    out += _int16(0)  # glyphDataFormat
    return bytes(out)


def build_hhea(adv_max: int, min_lsb: int, min_rsb: int, x_max_extent: int, num_hmetrics: int) -> bytes:
    out = bytearray()
    out += _uint32(0x00010000)
    out += _int16(ASCENDER)
    out += _int16(DESCENDER)
    out += _int16(LINE_GAP)
    out += _uint16(adv_max)
    out += _int16(min_lsb)
    out += _int16(min_rsb)
    out += _int16(x_max_extent)
    out += _int16(1)  # caretSlopeRise
    out += _int16(0)  # caretSlopeRun
    out += _int16(0)  # caretOffset
    out += _int16(0)
    out += _int16(0)
    out += _int16(0)
    out += _int16(0)
    out += _int16(0)  # metricDataFormat
    out += _uint16(num_hmetrics)
    return bytes(out)


def build_maxp(num_glyphs: int, max_points: int, max_contours: int) -> bytes:
    out = bytearray()
    out += _uint32(0x00010000)
    out += _uint16(num_glyphs)
    out += _uint16(max_points)
    out += _uint16(max_contours)
    out += _uint16(0)  # maxCompositePoints
    out += _uint16(0)  # maxCompositeContours
    out += _uint16(2)  # maxZones
    out += _uint16(0)  # maxTwilightPoints
    out += _uint16(0)  # maxStorage
    out += _uint16(0)  # maxFunctionDefs
    out += _uint16(0)  # maxInstructionDefs
    out += _uint16(0)  # maxStackElements
    out += _uint16(0)  # maxSizeOfInstructions
    out += _uint16(0)  # maxComponentElements
    out += _uint16(0)  # maxComponentDepth
    return bytes(out)


def assemble_font(tables: Dict[str, bytes]) -> bytes:
    tags = sorted(tables.keys())
    num_tables = len(tags)

    max_pow2 = 1
    entry_selector = 0
    while max_pow2 * 2 <= num_tables:
        max_pow2 *= 2
        entry_selector += 1
    search_range = max_pow2 * 16
    range_shift = num_tables * 16 - search_range

    header = bytearray()
    header += _uint32(0x00010000)
    header += _uint16(num_tables)
    header += _uint16(search_range)
    header += _uint16(entry_selector)
    header += _uint16(range_shift)

    records = bytearray()
    body = bytearray()
    offset = 12 + num_tables * 16

    offsets: Dict[str, int] = {}
    lengths: Dict[str, int] = {}

    for tag in tags:
        data = tables[tag]
        padded = _pad4(data)
        chk = _checksum(data)
        records += tag.encode("ascii")
        records += _uint32(chk)
        records += _uint32(offset)
        records += _uint32(len(data))
        offsets[tag] = offset
        lengths[tag] = len(data)
        body += padded
        offset += len(padded)

    font = bytearray(header + records + body)

    # Patch checkSumAdjustment in head table.
    head_off = offsets["head"]
    struct.pack_into(">I", font, head_off + 8, 0)
    total_sum = _checksum(bytes(font))
    adjustment = (0xB1B0AFBA - total_sum) & 0xFFFFFFFF
    struct.pack_into(">I", font, head_off + 8, adjustment)

    return bytes(font)


VARIANTS = [
    FontVariant(style_name="Regular", weight=400, italic=False, embolden=1.00, slant=0.00, advance_add=0),
    FontVariant(style_name="Bold", weight=700, italic=False, embolden=1.10, slant=0.00, advance_add=12),
    FontVariant(style_name="Italic", weight=400, italic=True, embolden=1.00, slant=0.19, advance_add=22),
    FontVariant(style_name="BoldItalic", weight=700, italic=True, embolden=1.10, slant=0.19, advance_add=30),
]


def _style_flags(variant: FontVariant) -> Tuple[int, int]:
    fs_selection = 0
    mac_style = 0
    if variant.weight >= 700:
        fs_selection |= 1 << 5  # BOLD
        mac_style |= 1 << 0
    if variant.italic:
        fs_selection |= 1 << 0  # ITALIC
        mac_style |= 1 << 1
    if fs_selection == 0:
        fs_selection = 1 << 6  # REGULAR
    return fs_selection, mac_style


def generate_font_variant(output_dir: Path, variant: FontVariant, base_glyphs: Dict[int, GlyphShape]) -> Path:
    glyphs = _apply_variant(base_glyphs, variant)
    glyph_order = [-1] + list(range(32, 127))
    glyf, loca, encoded = build_glyf_and_loca(glyph_order, glyphs)

    hmtx, adv_max, min_lsb, min_rsb, x_max_extent = build_hmtx(glyph_order, glyphs, encoded)
    avg_adv = int(round(sum((620 if cp == -1 else glyphs[cp].advance) for cp in glyph_order) / len(glyph_order)))

    xMin = min(g.xMin for g in encoded if g.points)
    yMin = min(g.yMin for g in encoded if g.points)
    xMax = max(g.xMax for g in encoded if g.points)
    yMax = max(g.yMax for g in encoded if g.points)

    now = datetime.now(timezone.utc)
    ts = _longdatetime(now)
    fs_selection, mac_style = _style_flags(variant)
    italic_angle = -11.0 if variant.italic else 0.0

    tables = {
        "cmap": build_cmap(glyph_order),
        "glyf": glyf,
        "head": build_head(xMin, yMin, xMax, yMax, ts, ts, mac_style),
        "hhea": build_hhea(adv_max, min_lsb, min_rsb, x_max_extent, len(glyph_order)),
        "hmtx": hmtx,
        "loca": loca,
        "maxp": build_maxp(len(glyph_order), max((g.points for g in encoded), default=0), max((g.contours for g in encoded), default=0)),
        "name": build_name("Death Ledger", variant.style_name),
        "OS/2": build_os2(avg_adv, 32, 126, variant.weight, fs_selection),
        "post": build_post(italic_angle=italic_angle),
    }

    font_bytes = assemble_font(tables)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_ttf = output_dir / f"DeathLedger-{variant.style_name}.ttf"
    out_ttf.write_bytes(font_bytes)
    return out_ttf


def generate_fontset(output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_glyphs = build_ascii_glyphs()
    paths = [generate_font_variant(output_dir, v, base_glyphs) for v in VARIANTS]

    css = """@font-face {
  font-family: 'Death Ledger';
  src: url('./DeathLedger-Regular.ttf') format('truetype');
  font-weight: 400;
  font-style: normal;
}

@font-face {
  font-family: 'Death Ledger';
  src: url('./DeathLedger-Bold.ttf') format('truetype');
  font-weight: 700;
  font-style: normal;
}

@font-face {
  font-family: 'Death Ledger';
  src: url('./DeathLedger-Italic.ttf') format('truetype');
  font-weight: 400;
  font-style: italic;
}

@font-face {
  font-family: 'Death Ledger';
  src: url('./DeathLedger-BoldItalic.ttf') format('truetype');
  font-weight: 700;
  font-style: italic;
}
"""
    (output_dir / "font.css").write_text(css, encoding="utf-8")

    specimen = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>Death Ledger Specimen</title>
  <link rel=\"stylesheet\" href=\"./font.css\" />
  <style>
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: radial-gradient(circle at 20% 10%, #262626 0, #111 45%, #000 100%);
      color: #f5f5f5;
      font-family: 'Death Ledger', serif;
    }
    .wrap { width: min(1000px, 92vw); padding: 2rem; }
    h1 { margin: 0 0 1rem; font-size: clamp(3rem, 10vw, 7rem); letter-spacing: 0.03em; font-weight: 700; }
    p { margin: 0.5rem 0; font-size: clamp(1.1rem, 2.4vw, 2rem); line-height: 1.35; letter-spacing: 0.02em; }
    .small { font-size: clamp(0.95rem, 1.8vw, 1.3rem); opacity: 0.9; }
    .italic { font-style: italic; }
    .bold { font-weight: 700; }
  </style>
</head>
<body>
  <main class=\"wrap\">
    <h1>DEATH LEDGER</h1>
    <p>ABCDEFGHIJKLMNOPQRSTUVWXYZ</p>
    <p class=\"italic\">abcdefghijklmnopqrstuvwxyz</p>
    <p class=\"bold\">0123456789 !\"#$%&'()*+,-./:;&lt;=&gt;?@[\\]^_`{|}~</p>
    <p class=\"bold italic\">THE NAMES WRITTEN HERE WILL DIE</p>
    <p class=\"small\">Original horror-inspired display font set generated from procedural strokes.</p>
  </main>
</body>
</html>
"""
    (output_dir / "specimen.html").write_text(specimen, encoding="utf-8")

    readme = """# Death Ledger Font Set

Generated files:
- `DeathLedger-Regular.ttf`
- `DeathLedger-Bold.ttf`
- `DeathLedger-Italic.ttf`
- `DeathLedger-BoldItalic.ttf`
- `font.css`
- `specimen.html`

Coverage:
- ASCII printable range: U+0020 to U+007E

Usage:
```css
font-family: 'Death Ledger', serif;
```

This is an original design inspired by rough horror calligraphy aesthetics.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return paths


if __name__ == "__main__":
    out = Path("fontface-set")
    generated = generate_fontset(out)
    for path in generated:
        print(f"Generated: {path}")
