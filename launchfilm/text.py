"""Type: kinetic lowercase lines (word by word), voice captions, closing lines.

Layout is computed for the full line first, so words that appear later never
shift the ones already on screen.
"""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from launchfilm.config import FONTS, PALETTE

STYLES = {
    # size_16x9, size_9x16, font, tracking(em), max width fraction
    "kinetic": dict(s169=78, s916=80, font="sans_medium", track=-0.035, maxw=0.8, lead=1.12),
    "closing": dict(s169=74, s916=74, font="sans_medium", track=-0.035, maxw=0.82, lead=1.12),
    "caption": dict(s169=46, s916=50, font="sans_medium", track=-0.015, maxw=0.8, lead=1.2),
    "label": dict(s169=19, s916=24, font="mono", track=0.08, maxw=0.9, lead=1.2),
    "label_big": dict(s169=25, s916=30, font="mono", track=0.08, maxw=0.9, lead=1.2),
}


@lru_cache(maxsize=32)
def _font(key, size):
    return ImageFont.truetype(FONTS[key](), size)


def _layout(text, style, fmt, W):
    st = STYLES[style]
    size = st["s169"] if fmt == "16x9" else st["s916"]
    font = _font(st["font"], size)
    track = st["track"] * size
    space = font.getlength(" ") + track
    words = text.split(" ")
    widths = [sum(font.getlength(c) + track for c in w) - track for w in words]
    maxw = st["maxw"] * W
    lines, cur, curw = [], [], 0.0
    for i, w in enumerate(words):
        add = widths[i] if not cur else curw + space + widths[i]
        if cur and add > maxw:
            lines.append(cur)
            cur, curw = [i], widths[i]
        else:
            cur.append(i)
            curw = add
    lines.append(cur)
    asc, desc = font.getmetrics()
    lh = size * st["lead"]
    return font, size, track, space, words, widths, lines, lh, asc


@lru_cache(maxsize=512)
def render_text(text, style, fmt, W, n_visible=None, color="white"):
    """RGBA uint8 array tightly bounding the whole text block, plus (w, h).
    Only the first n_visible words are drawn."""
    font, size, track, space, words, widths, lines, lh, asc = _layout(text, style, fmt, W)
    n_visible = len(words) if n_visible is None else n_visible
    bw = int(max(sum(widths[i] for i in ln) + space * (len(ln) - 1) for ln in lines)) + 8
    bh = int(lh * len(lines) + size * 0.35) + 8
    img = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    col = PALETTE[color] + (255,)
    for li, ln in enumerate(lines):
        lw = sum(widths[i] for i in ln) + space * (len(ln) - 1)
        x = (bw - lw) / 2
        y = 4 + li * lh
        for i in ln:
            if i < n_visible:
                for c in words[i]:
                    d.text((x, y), c, font=font, fill=col)
                    x += font.getlength(c) + track
                x += space - track
            else:
                x += widths[i] + space
    return np.asarray(img)


def shadow(rgba, radius=10, strength=0.55):
    a = Image.fromarray(rgba[..., 3]).filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(a).astype(np.float32) / 255 * strength


def composite(frame, rgba, cx, cy, alpha=1.0, drop_shadow=True, scrim=None):
    """Alpha-composite a text block centered at (cx, cy) onto a float32 RGB frame."""
    h, w = rgba.shape[:2]
    H, W = frame.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    x1, y1 = x0 + w, y0 + h
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1c <= x0c or y1c <= y0c:
        return frame
    region = frame[y0c:y1c, x0c:x1c]
    src = rgba[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)].astype(np.float32) / 255
    if drop_shadow:
        sh = shadow(rgba)[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)] * alpha
        region *= (1 - sh[..., None])
    a = src[..., 3:4] * alpha
    region[:] = region * (1 - a) + src[..., :3] * a
    return frame


def scrim(frame, cy, height, strength=0.45, cx=None, width=None):
    """Soft elliptical darkening behind type over footage (full-width band if no width)."""
    H, W = frame.shape[:2]
    y = np.arange(H, dtype=np.float32)
    vy = np.exp(-0.5 * ((y - cy) / (height / 2.2)) ** 2)
    if width is None:
        band = vy[:, None] * strength
    else:
        x = np.arange(W, dtype=np.float32)
        cx = W / 2 if cx is None else cx
        vx = np.exp(-0.5 * (np.abs(x - cx) / (width / 2.0)) ** 4)
        band = np.outer(vy, vx) * strength
    frame *= (1 - band)[..., None]
    return frame
