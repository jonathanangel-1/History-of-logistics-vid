"""Film finishing: bloom, filmic tonemap, per-scene grade, vignette, CA, grain, letterbox."""
import math

import cv2
import numpy as np

from logistics.gfx import W, H, LB, cached

GRADES = {
    #          exposure  wb (r,g,b)           sat   shadow tint            highlight tint        bloom thr  vign
    "intro":  dict(exp=1.0, wb=(1.05, 0.98, 0.9), sat=0.8, sh=(0.02, 0.01, 0.0), hi=(0.06, 0.03, -0.02), bloom=0.7, thr=0.55, vig=0.55),
    "desert": dict(exp=1.05, wb=(1.06, 0.98, 0.92), sat=1.08, sh=(-0.01, 0.0, 0.03), hi=(0.05, 0.02, -0.03), bloom=0.75, thr=0.6, vig=0.5),
    "dawn":   dict(exp=0.95, wb=(1.05, 1.0, 0.92), sat=0.95, sh=(0.0, 0.01, 0.03), hi=(0.05, 0.03, -0.02), bloom=0.6, thr=0.65, vig=0.6),
    "fire":   dict(exp=1.05, wb=(1.0, 1.0, 1.0), sat=1.05, sh=(-0.01, 0.02, 0.05), hi=(0.07, 0.02, -0.04), bloom=0.9, thr=0.5, vig=0.55),
    "storm":  dict(exp=1.0, wb=(0.92, 1.0, 1.06), sat=0.7, sh=(-0.01, 0.02, 0.04), hi=(0.0, 0.02, 0.03), bloom=0.6, thr=0.6, vig=0.6),
    "steam":  dict(exp=1.05, wb=(1.0, 0.98, 1.02), sat=0.95, sh=(-0.02, 0.01, 0.05), hi=(0.08, 0.03, -0.04), bloom=0.9, thr=0.5, vig=0.55),
    "fog":    dict(exp=1.0, wb=(0.99, 1.0, 1.03), sat=0.8, sh=(0.02, 0.03, 0.05), hi=(0.04, 0.02, 0.0), bloom=0.6, thr=0.6, vig=0.4),
    "golden": dict(exp=0.95, wb=(1.05, 1.0, 0.92), sat=1.1, sh=(-0.02, 0.0, 0.05), hi=(0.05, 0.02, -0.03), bloom=0.55, thr=0.72, vig=0.55),
    "night":  dict(exp=1.1, wb=(1.0, 1.0, 1.02), sat=1.05, sh=(-0.01, 0.02, 0.05), hi=(0.08, 0.03, -0.04), bloom=1.0, thr=0.45, vig=0.55),
    "future": dict(exp=1.1, wb=(0.96, 1.0, 1.06), sat=1.1, sh=(0.0, 0.01, 0.06), hi=(0.02, 0.04, 0.05), bloom=1.1, thr=0.45, vig=0.55),
    "end":    dict(exp=1.0, wb=(1.04, 1.0, 0.94), sat=0.9, sh=(0.0, 0.0, 0.0), hi=(0.06, 0.03, -0.02), bloom=0.9, thr=0.5, vig=0.6),
}


def _vignette():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (x - W / 2) / (W / 2)
    ny = (y - H / 2) / (H / 2 * 0.75)
    r2 = nx * nx * 0.8 + ny * ny * 0.6
    return np.clip(1 - r2, 0, 1) ** 0.9


def _grain(k):
    r = np.random.default_rng(100 + k)
    g = r.standard_normal((H // 2, W // 2)).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 0.6)
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
    return g / (g.std() + 1e-6)


def surface_to_float(surface):
    buf = np.ndarray((H, W, 4), np.uint8, surface.get_data())
    img = buf[..., 2::-1].astype(np.float32) * (1.0 / 255.0)  # BGRA -> RGB
    return img


def aces(x):
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def finish(img, glow, grade, frame_idx, flash=0.0, exposure=1.0, bloom_boost=0.0, ca=1.0,
           text=None, fade=1.0, letterbox=True):
    g = GRADES[grade]
    if glow is not None:
        img += cv2.resize(glow, (W, H), interpolation=cv2.INTER_LINEAR)

    # bloom from highlights, two radii
    small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    br = np.maximum(small - g["thr"], 0)
    b1 = cv2.GaussianBlur(br, (0, 0), 4)
    b2 = cv2.GaussianBlur(cv2.resize(br, (W // 8, H // 8), interpolation=cv2.INTER_AREA), (0, 0), 10)
    bloom = cv2.resize(b1, (W, H)) * 0.6 + cv2.resize(b2, (W, H)) * 1.0
    img += bloom * (g["bloom"] + bloom_boost)

    img *= np.array(g["wb"], np.float32) * (g["exp"] * exposure)
    if flash > 0:
        img += flash * np.array([1.0, 0.96, 0.9], np.float32)
    img = aces(img * 1.25)

    lum = img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722
    lum3 = lum[..., None]
    img = lum3 + (img - lum3) * g["sat"]
    img += np.array(g["sh"], np.float32) * (1 - lum3) ** 2 + np.array(g["hi"], np.float32) * lum3 ** 2

    vig = cached("vig", _vignette)
    img *= (1 - g["vig"]) + g["vig"] * vig[..., None]

    if ca > 0:
        s = 0.0016 * ca
        for ch, sc in ((0, 1 + s), (2, 1 - s)):
            M = np.float32([[sc, 0, (1 - sc) * W / 2], [0, sc, (1 - sc) * H / 2]])
            img[..., ch] = cv2.warpAffine(np.ascontiguousarray(img[..., ch]), M, (W, H), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REPLICATE)

    if text is not None:
        rgb, a = text
        img = img * (1 - a[..., None]) + rgb

    gr = cached(("grain", frame_idx % 12), lambda: _grain(frame_idx % 12))
    lum = img.mean(axis=2, keepdims=True)
    amp = 0.022 * (0.35 + 1.2 * lum * (1 - lum))
    img += gr[..., None] * amp

    img *= fade
    if letterbox:
        img[:LB] = 0
        img[H - LB:] = 0
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
