"""Rasterize the client's Volume SVG exactly as drawn (no redraw).

The file only uses absolute M/H/V/L/Q/Z path commands, <rect rx> and
translate/scale group transforms, so a small parser is enough. Every shape
filled with the gold accent (#baa88a) goes to the "square" mask; everything
else is letterform and goes to the "letters" mask, so the caller can recolor
the letters (cream on dark) while the square stays gold.
"""
import re
import xml.etree.ElementTree as ET
from functools import lru_cache

import cv2
import numpy as np

from launchfilm.config import ROOT

LOGO_SVG = ROOT / "brand" / "volume-logo.svg"
GOLD_HEX = "#baa88a"
NS = "{http://www.w3.org/2000/svg}"


def _parse_transform(s):
    m = np.eye(3)
    for name, args in re.findall(r"(\w+)\(([^)]*)\)", s or ""):
        v = [float(a) for a in re.split(r"[ ,]+", args.strip()) if a]
        t = np.eye(3)
        if name == "translate":
            t[0, 2], t[1, 2] = v[0], (v[1] if len(v) > 1 else 0.0)
        elif name == "scale":
            t[0, 0], t[1, 1] = v[0], (v[1] if len(v) > 1 else v[0])
        else:
            raise ValueError(f"unsupported transform {name}")
        m = m @ t
    return m


def _path_polys(d, steps=12):
    toks = re.findall(r"[MHVLQZ]|-?\d*\.?\d+(?:e-?\d+)?", d)
    polys, cur, i = [], [], 0
    x = y = 0.0
    cmd = None
    while i < len(toks):
        tk = toks[i]
        if tk in "MHVLQZ":
            cmd = tk
            i += 1
            if cmd == "Z":
                if cur:
                    polys.append(cur)
                cur = []
            continue
        n = lambda k: float(toks[i + k])
        if cmd == "M":
            if cur:
                polys.append(cur)
            x, y = n(0), n(1)
            cur = [(x, y)]
            i += 2
            cmd = "L"
        elif cmd == "L":
            x, y = n(0), n(1)
            cur.append((x, y))
            i += 2
        elif cmd == "H":
            x = n(0)
            cur.append((x, y))
            i += 1
        elif cmd == "V":
            y = n(0)
            cur.append((x, y))
            i += 1
        elif cmd == "Q":
            cx, cy, ex, ey = n(0), n(1), n(2), n(3)
            for s in range(1, steps + 1):
                t = s / steps
                px = (1 - t) ** 2 * x + 2 * (1 - t) * t * cx + t * t * ex
                py = (1 - t) ** 2 * y + 2 * (1 - t) * t * cy + t * t * ey
                cur.append((px, py))
            x, y = ex, ey
            i += 4
        else:
            raise ValueError(f"unsupported path command {cmd}")
    if cur:
        polys.append(cur)
    return polys


def _rect_poly(x, y, w, h, rx, steps=6):
    rx = min(rx, w / 2, h / 2)
    pts = []
    for cx, cy, a0 in ((x + w - rx, y + rx, -90), (x + w - rx, y + h - rx, 0),
                       (x + rx, y + h - rx, 90), (x + rx, y + rx, 180)):
        for s in range(steps + 1):
            a = np.radians(a0 + 90 * s / steps)
            pts.append((cx + rx * np.cos(a), cy + rx * np.sin(a)))
    return [pts]


def _walk(el, m, out):
    m = m @ _parse_transform(el.get("transform"))
    tag = el.tag.replace(NS, "")
    fill = (el.get("fill") or "").lower()
    if tag == "path":
        out.append((fill, _path_polys(el.get("d")), m))
    elif tag == "rect":
        out.append((fill, _rect_poly(float(el.get("x", 0)), float(el.get("y", 0)),
                                     float(el.get("width")), float(el.get("height")),
                                     float(el.get("rx", 0))), m))
    for ch in el:
        _walk(ch, m, out)


@lru_cache(maxsize=8)
def logo_masks(width_px, ss=4):
    """Return (letters, square, (x0, y0, x1, y1) of the square) as float32 alpha
    masks of the full SVG viewBox scaled to width_px."""
    root = ET.parse(LOGO_SVG).getroot()
    vx, vy, vw, vh = [float(v) for v in root.get("viewBox").split()]
    s = width_px * ss / vw
    H = int(round(vh * s))
    Wd = int(round(vw * s))
    shapes = []
    _walk(root, np.eye(3), shapes)
    letters = np.zeros((H, Wd), np.uint8)
    square = np.zeros((H, Wd), np.uint8)
    sq_pts = []
    for fill, polys, m in shapes:
        pts = []
        for p in polys:
            a = np.array([[px, py, 1.0] for px, py in p]) @ m.T
            pts.append(np.round((a[:, :2] - [vx, vy]) * s).astype(np.int32))
        target = square if fill == GOLD_HEX else letters
        cv2.fillPoly(target, pts, 255, lineType=cv2.LINE_AA)
        if fill == GOLD_HEX:
            sq_pts.extend(pts)
    size = (width_px, int(round(vh * width_px / vw)))
    L = cv2.resize(letters, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    S = cv2.resize(square, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    allp = np.concatenate(sq_pts) / ss
    box = (allp[:, 0].min(), allp[:, 1].min(), allp[:, 0].max(), allp[:, 1].max())
    return L, S, box
