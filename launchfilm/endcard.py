"""Closing sequence after the logo hit, in the volumeba.com design language.

The logo lands centered on the hit (picture.compose_logo), then glides to the top-left
while the frame opens into a rounded photographic stage (the site's hero), and the
page's elements build in: mono eyebrow, a large Manrope headline with a gold gradient,
the service line, a gold route that draws through the gateways with the gold square
riding it, and a smoked-glass URL button. Every string comes from `endcard` in the cue
sheet and only repeats claims made on the live site.
"""
from functools import lru_cache

import cv2
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
    """Strong ease-out, close to the site's cubic-bezier(.16, 1, .3, 1)."""
    x = float(np.clip(x, 0, 1))
    return 1 - (1 - x) ** 5


def prog(t, t0, dur):
    return ease((t - t0) / dur)


# ------------------------------------------------------------------ type
@lru_cache(maxsize=16)
def _font(key, size, weight=None):
    f = ImageFont.truetype(FONTS[key](), size)
    if weight is not None:
        try:
            f.set_variation_by_axes([weight])
        except OSError:
            pass
    return f


@lru_cache(maxsize=128)
def text_rgba(text, key, size, weight=None, track_em=0.0, color="cream", gradient=None):
    """Float RGBA (h, w, 4) tightly around the text; optional left-to-right gradient fill."""
    font = _font(key, size, weight)
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
    a = np.asarray(mask).astype(np.float32) / 255
    out = np.zeros((h, w, 4), np.float32)
    if gradient:
        stops = [rgbf(c) for c in gradient]
        u = np.linspace(0, 1, w, dtype=np.float32)
        seg = np.clip(u * (len(stops) - 1), 0, len(stops) - 1 - 1e-6)
        i = seg.astype(int)
        f = (seg - i)[:, None]
        row = np.array([stops[k] for k in i]) * (1 - f) + np.array([stops[k + 1] for k in i]) * f
        out[..., :3] = row[None, :, :]
    else:
        out[..., :3] = rgbf(color)
    out[..., 3] = a
    return out


def blit(frame, rgba, x, y, alpha=1.0, clip=None):
    """Composite float RGBA at integer top-left (x, y); `clip` = (y0, y1) visible rows."""
    h, w = rgba.shape[:2]
    H, W = frame.shape[:2]
    x, y = int(round(x)), int(round(y))
    y0, y1 = max(0, y), min(H, y + h)
    if clip is not None:
        y0, y1 = max(y0, clip[0]), min(y1, clip[1])
    x0, x1 = max(0, x), min(W, x + w)
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return
    src = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = src[..., 3:4] * alpha
    reg = frame[y0:y1, x0:x1]
    reg[:] = reg * (1 - a) + src[..., :3] * a


def rise(frame, rgba, x, y, p, dist=14):
    """Fade + rise into place."""
    blit(frame, rgba, x, y + (1 - p) * dist, p)


def mask_reveal(frame, rgba, x, y, p):
    """Line slides up from behind its own baseline (clipped to its box)."""
    h = rgba.shape[0]
    blit(frame, rgba, x, y + (1 - p) * h * 0.9, min(1.0, p * 1.6), clip=(int(y), int(y + h)))


# ------------------------------------------------------------------ shapes
def rounded_mask(h, w, r, ss=3):
    m = np.zeros((h * ss, w * ss), np.uint8)
    R = int(r * ss)
    cv2.rectangle(m, (R, 0), (w * ss - R - 1, h * ss - 1), 255, -1)
    cv2.rectangle(m, (0, R), (w * ss - 1, h * ss - R - 1), 255, -1)
    for cx, cy in ((R, R), (w * ss - R - 1, R), (R, h * ss - R - 1), (w * ss - R - 1, h * ss - R - 1)):
        cv2.circle(m, (cx, cy), R, 255, -1, lineType=cv2.LINE_AA)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def glass(frame, x0, y0, x1, y1, r, alpha=1.0):
    """Smoked glass: blurred backdrop, dark tint with warm/cool reflections, fine edges."""
    x0, y0, x1, y1 = (int(round(v)) for v in (x0, y0, x1, y1))
    h, w = y1 - y0, x1 - x0
    reg = frame[y0:y1, x0:x1]
    m = rounded_mask(h, w, r)[..., None] * alpha
    blur = cv2.GaussianBlur(reg, (0, 0), 16)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    warm = np.exp(-((xx / w) ** 2 + (yy / h) ** 2) * 1.6)[..., None] * rgbf("#c68973") * 0.11
    cool = np.exp(-(((xx - w) / w) ** 2 + ((yy - h) / h) ** 2) * 1.6)[..., None] * rgbf("#7191cc") * 0.14
    tinted = blur * 0.42 + rgbf("#0c1215") * 0.5 + warm + cool
    reg[:] = reg * (1 - m) + tinted * m
    edge = rounded_mask(h, w, r) - np.pad(rounded_mask(h - 2, w - 2, max(1, r - 1)), 1)
    reg[:] = reg * (1 - edge[..., None] * 0.22 * alpha) + edge[..., None] * 0.22 * alpha
    hl = np.clip(1 - np.abs(np.linspace(-1, 1, w - 2 * r)) ** 2, 0, 1)
    reg[0, r:w - r] = reg[0, r:w - r] * (1 - hl[:, None] * 0.55 * alpha) + hl[:, None] * 0.55 * alpha


def square(frame, cx, cy, size, color, alpha=1.0, glow=0.0):
    s = size / 2
    x0, y0, x1, y1 = int(cx - s), int(cy - s), int(cx + s), int(cy + s)
    if glow > 0:
        g = int(size * 3)
        gx0, gy0 = max(0, int(cx - g)), max(0, int(cy - g))
        gx1, gy1 = min(frame.shape[1], int(cx + g)), min(frame.shape[0], int(cy + g))
        yy, xx = np.mgrid[gy0:gy1, gx0:gx1].astype(np.float32)
        k = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (size * 1.1) ** 2)))[..., None] * glow
        frame[gy0:gy1, gx0:gx1] = np.clip(frame[gy0:gy1, gx0:gx1] + color * k, 0, 1)
    reg = frame[max(0, y0):y1, max(0, x0):x1]
    reg[:] = reg * (1 - alpha) + color * alpha


def hline(frame, x0, x1, y, color, alpha, thick=2):
    if x1 <= x0:
        return
    reg = frame[int(y):int(y) + thick, int(x0):int(x1)]
    reg[:] = reg * (1 - alpha) + color * alpha


def vline(frame, x, y0, y1, color, alpha, thick=2):
    if y1 <= y0:
        return
    reg = frame[int(y0):int(y1), int(x):int(x) + thick]
    reg[:] = reg * (1 - alpha) + color * alpha


# ------------------------------------------------------------------ logo
def logo(frame, cx, cy, width, alpha=1.0):
    L, S, _ = logo_masks(int(round(width)))
    lh, lw = L.shape
    x0, y0 = int(round(cx - lw / 2)), int(round(cy - lh / 2))
    reg = frame[y0:y0 + lh, x0:x0 + lw]
    for m, col in ((L, rgbf("cream")), (S, rgbf("gold"))):
        a = (m * alpha)[..., None]
        reg[:] = reg * (1 - a) + col * a


def logo_size(width):
    L, _, _ = logo_masks(int(round(width)))
    return L.shape[1], L.shape[0]


# ------------------------------------------------------------------ layout
LAYOUT = {
    "16x9": dict(inset=28, radius=22, logo_w=250, logo_xy=(120, 104), eyebrow_right=1800, eyebrow_y=118,
                 h_size=150, h_xy=(112, 292), h_lead=1.04, sub_size=34, sub_xy=(120, 676),
                 route="h", route_y=842, route_x=(126, 1380), nodes=(126, 548, 970),
                 cta=(1432, 800, 1800, 884), cov_xy=(1800, 752), name_size=30, label_size=17),
    "9x16": dict(inset=20, radius=22, logo_w=236, logo_xy=(92, 282), eyebrow_left=94, eyebrow_y=378,
                 h_size=112, h_xy=(84, 560), h_lead=1.06, sub_size=33, sub_xy=(94, 836),
                 route="v", route_x0=104, route_y=(990, 1400), nodes=(1000, 1140, 1280),
                 cta=(92, 1520, 988, 1612), cov_xy=(94, 1446), name_size=34, label_size=19),
}


def compose(img_bg, cs, fmt, t, t_hit, charcoal):
    """Full frame from the logo hit to the end. `img_bg` is the graded background footage."""
    H, W = img_bg.shape[:2]
    ec = cs["endcard"]
    rv = cs["reveal"]
    T0 = cs.when(ec)
    lay = LAYOUT[fmt]
    p = prog(t, T0, 0.95)

    # background: faint luminance plate under the hit, opening to full color on the stage
    lum = (img_bg @ np.float32([0.299, 0.587, 0.114]))[..., None]
    plate = charcoal + (lum * 0.9 + img_bg * 0.1 - charcoal) * rv.get("bg_opacity", 0.16)
    full = img_bg * 0.92
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    xx = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
    if fmt == "16x9":
        shade = 0.78 * np.clip(1 - xx / 0.72, 0, 1) ** 1.4 + 0.5 * np.clip((yy - 0.55) / 0.45, 0, 1) ** 1.5
    else:
        shade = 0.55 + 0.25 * np.clip((yy - 0.45) / 0.55, 0, 1)
    full = full * (1 - np.clip(shade, 0, 0.86)) + rgbf("#0c1215") * np.clip(shade, 0, 0.86) * 0.5
    img = plate * (1 - p) + full * p

    # the rounded stage (site hero) closes in from full-bleed
    if p > 0:
        ins = lay["inset"] * p
        r = max(1, int(lay["radius"] * p))
        x0, y0 = int(round(ins)), int(round(ins))
        stage = rounded_mask(H - 2 * y0, W - 2 * x0, r)
        out = np.empty_like(img)
        out[:] = rgbf("charcoal")
        reg = out[y0:H - y0, x0:W - x0]
        reg[:] = reg * (1 - stage[..., None]) + img[y0:H - y0, x0:W - x0] * stage[..., None]
        img = out

    # logo: hit animation until T0, then glide to the top-left
    if t < T0:
        from launchfilm.picture import compose_logo
        cy = H / 2 if fmt == "16x9" else H * 0.47
        wl = rv["logo_w_16x9" if fmt == "16x9" else "logo_w_9x16"]
        return compose_logo(img, cs, fmt, t, W / 2, cy, wl, t_hit)
    w0 = rv["logo_w_16x9" if fmt == "16x9" else "logo_w_9x16"]
    cy0 = H / 2 if fmt == "16x9" else H * 0.47
    tw, th = logo_size(lay["logo_w"])
    cx1, cy1 = lay["logo_xy"][0] + tw / 2, lay["logo_xy"][1] + th / 2
    lw = w0 + (lay["logo_w"] - w0) * p
    logo(img, W / 2 + (cx1 - W / 2) * p, cy0 + (cy1 - cy0) * p, lw)

    cream, gold = rgbf("cream"), rgbf("gold")
    k = lambda d, dur=0.55: prog(t, T0 + d, dur)

    # eyebrow
    eb = text_rgba(ec["eyebrow"], "mono", 19 if fmt == "16x9" else 20, None, 0.08, "gold_bright")
    if fmt == "16x9":
        rise(img, eb, lay["eyebrow_right"] - eb.shape[1], lay["eyebrow_y"], k(0.55), 10)
    else:
        rise(img, eb, lay["eyebrow_left"], lay["eyebrow_y"], k(0.55), 10)

    # headline, line by line
    grad = ("#baa88a", "#e3d7bb", "#eee9de")
    hx, hy = lay["h_xy"]
    for i, line in enumerate(ec["headline"]):
        rg = text_rgba(line, "display", lay["h_size"], 450, -0.055, gradient=grad)
        mask_reveal(img, rg, hx, hy + i * lay["h_size"] * lay["h_lead"], k(0.72 + 0.16 * i, 0.7))

    # service line
    sub = text_rgba(ec["services_line"], "sans", lay["sub_size"], None, -0.02, "cream")
    rise(img, sub, lay["sub_xy"][0], lay["sub_xy"][1], k(1.25) * 0.86, 12)

    # route through the gateways, the gold square riding its head
    rp = prog(t, T0 + 1.55, 1.5)
    nodes = ec["nodes"]
    if lay["route"] == "h":
        ry = lay["route_y"]
        xa, xb = lay["route_x"]
        cta_x0 = lay["cta"][0]
        head = xa + (cta_x0 - 14 - xa) * rp
        hline(img, xa, xb, ry, cream, 0.16 * min(1, rp * 4), 1)
        hline(img, xa, head, ry - 0, gold, 0.9, 2)
        for i, nx in enumerate(lay["nodes"]):
            q = prog(t, T0 + 1.55 + 1.5 * (nx - xa) / (cta_x0 - xa), 0.45) if rp > 0 else 0
            if q <= 0:
                continue
            square(img, nx + 6, ry + 1, 12, gold, q)
            lab = text_rgba(nodes[i]["label"], "mono", lay["label_size"], None, 0.08, "gold")
            nm = text_rgba(nodes[i]["name"], "sans", lay["name_size"], None, -0.02, "cream")
            rise(img, lab, nx - 4, ry + 26, q, 8)
            rise(img, nm, nx - 4, ry + 56, q, 8)
        if 0 < rp < 1:
            square(img, head, ry + 1, 16, gold, 1.0, glow=0.55)
    else:
        rx = lay["route_x0"]
        ya, yb = lay["route_y"]
        head = ya + (yb - ya) * rp
        vline(img, rx, ya, yb, cream, 0.16 * min(1, rp * 4), 1)
        vline(img, rx, ya, head, gold, 0.9, 2)
        for i, ny in enumerate(lay["nodes"]):
            q = prog(t, T0 + 1.55 + 1.5 * (ny - ya) / (yb - ya), 0.45) if rp > 0 else 0
            if q <= 0:
                continue
            square(img, rx + 1, ny + 14, 14, gold, q)
            lab = text_rgba(nodes[i]["label"], "mono", lay["label_size"], None, 0.08, "gold")
            nm = text_rgba(nodes[i]["name"], "sans", lay["name_size"], None, -0.02, "cream")
            rise(img, lab, rx + 40, ny, q, 8)
            rise(img, nm, rx + 40, ny + 34, q, 8)
        if 0 < rp < 1:
            square(img, rx + 1, head, 18, gold, 1.0, glow=0.55)

    # coverage line + URL button (the square lands in the arrow tile)
    cq = k(2.75, 0.6)
    if cq > 0:
        cov = text_rgba(ec["coverage"], "mono", lay["label_size"], None, 0.08, "gold_bright")
        cx0, cy0_, cx1, cy1_ = lay["cta"]
        if fmt == "16x9":
            rise(img, cov, lay["cov_xy"][0] - cov.shape[1], lay["cov_xy"][1], cq, 8)
        else:
            rise(img, cov, lay["cov_xy"][0], lay["cov_xy"][1], cq, 8)
        dy = (1 - cq) * 14
        glass(img, cx0, cy0_ + dy, cx1, cy1_ + dy, 12, cq)
        bh = cy1_ - cy0_
        url = text_rgba(ec["url"], "sans_medium", 32 if fmt == "16x9" else 36, None, -0.02, "white")
        blit(img, url, cx0 + 30, cy0_ + dy + (bh - url.shape[0]) / 2, cq)
        tile = bh - 16
        tx0, ty0 = cx1 - 8 - tile, cy0_ + dy + 8
        tm = rounded_mask(tile, tile, 6)[..., None] * cq
        reg = img[int(ty0):int(ty0) + tile, int(tx0):int(tx0) + tile]
        reg[:] = reg * (1 - tm) + rgbf("#f6f6ec") * tm
        arr = text_rgba("\u2192", "sans", int(tile * 0.42), None, 0.0, "#28312b")
        blit(img, arr, tx0 + (tile - arr.shape[1]) / 2, ty0 + (tile - arr.shape[0]) / 2 - 2, cq)
        square(img, tx0 + tile - 11, ty0 + 11, 8, gold, cq)
    return img
