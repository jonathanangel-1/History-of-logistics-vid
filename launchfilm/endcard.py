"""Closing: logo hit, then a restrained end card.

The gold square lands on the musical downbeat (picture.compose_logo) over the last shot,
which dims to black. On the track's final hit the logo rises a little and the end line
and URL fade up beneath it; the DoD non-endorsement notice sits small at the bottom.
One typeface (Inter) for every line of type in the film. Strings live in `endcard`.
"""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from launchfilm.config import FONTS, PALETTE
from launchfilm.logo import logo_masks


def rgbf(name_or_hex):
    if isinstance(name_or_hex, str) and name_or_hex.startswith("#"):
        h = name_or_hex.lstrip("#")
        return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255
    return np.array(PALETTE[name_or_hex], np.float32) / 255


def ease(x):
    """Strong ease-out."""
    x = float(np.clip(x, 0, 1))
    return 1 - (1 - x) ** 4


def ease_io(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def prog(t, t0, dur):
    return ease((t - t0) / dur)


@lru_cache(maxsize=16)
def _font(key, size):
    return ImageFont.truetype(FONTS[key](), size)


@lru_cache(maxsize=128)
def text_rgba(text, key, size, track_em=0.0, color="cream"):
    """Float RGBA (h, w, 4) tightly around one line of text."""
    font = _font(key, size)
    track = track_em * size
    w = int(sum(font.getlength(c) + track for c in text) - track) + 8
    asc, desc = font.getmetrics()
    h = asc + desc + 8
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    x = 4
    for c in text:
        d.text((x, 4), c, font=font, fill=255)
        x += font.getlength(c) + track
    out = np.zeros((h, w, 4), np.float32)
    out[..., :3] = rgbf(color)
    out[..., 3] = np.asarray(mask).astype(np.float32) / 255
    return out


def blit(frame, rgba, x, y, alpha=1.0):
    h, w = rgba.shape[:2]
    H, W = frame.shape[:2]
    x, y = int(round(x)), int(round(y))
    y0, y1, x0, x1 = max(0, y), min(H, y + h), max(0, x), min(W, x + w)
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return
    src = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = src[..., 3:4] * alpha
    reg = frame[y0:y1, x0:x1]
    reg[:] = reg * (1 - a) + src[..., :3] * a


def rise(frame, rgba, cx, y, p, dist=12):
    """Fade + rise into place, horizontally centered on cx."""
    blit(frame, rgba, cx - rgba.shape[1] / 2, y + (1 - p) * dist, p)


def logo(frame, cx, cy, width, alpha=1.0):
    L, S, _ = logo_masks(int(round(width)))
    lh, lw = L.shape
    x0, y0 = int(round(cx - lw / 2)), int(round(cy - lh / 2))
    reg = frame[y0:y0 + lh, x0:x0 + lw]
    for m, col in ((L, rgbf("cream")), (S, rgbf("gold"))):
        a = (m * alpha)[..., None]
        reg[:] = reg * (1 - a) + col * a


LAYOUT = {
    "16x9": dict(logo_cy=0.5, logo_cy_end=0.43, line_y=0.535, line_size=50, url_y=0.615, url_size=28,
                 note_size=15, note_bottom=46, note_wrap=None),
    "9x16": dict(logo_cy=0.47, logo_cy_end=0.43, line_y=0.487, line_size=52, url_y=0.528, url_size=30,
                 note_size=19, note_bottom=250, note_wrap=2),
}


def _wrap(text, n):
    if not n or n < 2:
        return [text]
    words = text.split(" ")
    k = len(words) // 2
    best = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
    return [" ".join(words[:best]), " ".join(words[best:])] if k else [text]


def compose(img_bg, cs, fmt, t, t_hit):
    """Full frame from the logo hit to the end. `img_bg` is the graded last shot."""
    from launchfilm.picture import compose_logo
    H, W = img_bg.shape[:2]
    ec, rv, lay = cs["endcard"], cs["reveal"], LAYOUT[fmt]
    T0 = cs.when(ec)
    # the last shot holds dimmed under the hit, then goes to black for the card
    dim = rv.get("bg_level", 0.3) * (1 - ease_io((t - T0 + 0.4) / 1.4))
    img = img_bg * dim
    w0 = rv["logo_w_16x9" if fmt == "16x9" else "logo_w_9x16"]
    if t < T0:
        return compose_logo(img, cs, fmt, t, W / 2, H * lay["logo_cy"], w0, t_hit)
    p = ease_io((t - T0) / 1.1)
    w1 = ec.get("logo_w_end", 0.78) * w0
    logo(img, W / 2, H * (lay["logo_cy"] + (lay["logo_cy_end"] - lay["logo_cy"]) * p), w0 + (w1 - w0) * p)
    line = text_rgba(ec["line"], "sans_medium", lay["line_size"], -0.02, "cream")
    rise(img, line, W / 2, H * lay["line_y"], prog(t, T0 + 0.45, 0.9))
    url = text_rgba(ec["url"], "sans", lay["url_size"], 0.01, "gold")
    rise(img, url, W / 2, H * lay["url_y"], prog(t, T0 + 0.85, 0.9))
    if ec.get("notice"):
        rows = _wrap(ec["notice"], lay["note_wrap"])
        q = prog(t, T0 + 1.2, 0.9) * 0.62
        for i, r in enumerate(rows):
            n = text_rgba(r, "sans", lay["note_size"], 0.0, "cream")
            y = H - lay["note_bottom"] - (len(rows) - i) * n.shape[0] * 1.05
            blit(img, n, W / 2 - n.shape[1] / 2, y, q)
    return img
