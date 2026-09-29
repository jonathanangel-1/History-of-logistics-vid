"""Contact sheet: one frame from the middle of each of up to 24 shots."""
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from launchfilm.config import FONTS, Cues


def _pick_times(cs, n=24):
    shots = cs["shots"]
    mids = []
    for i, s in enumerate(shots):
        t0 = cs.when(s)
        t1 = cs.when(shots[i + 1]) if i + 1 < len(shots) else cs.duration
        mids.append((t0 + t1) / 2)
    # add type moments (closing lines, reveal, end card) and thin to n evenly
    for ti in cs["titles"]:
        if ti["style"] == "closing":
            mids.append((cs.when(ti) + cs.when(ti, "end")) / 2)
    mids += [cs.when(cs["reveal"]) + 1.0, cs.when(cs["endcard"]) + 2.0]
    mids = sorted(set(round(m, 2) for m in mids))
    if len(mids) > n:
        idx = np.linspace(0, len(mids) - 1, n).round().astype(int)
        mids = [mids[i] for i in idx]
    return mids


def sheet(mp4, out_png, fmt, n=24):
    cs = Cues()
    times = _pick_times(cs, n)
    tw, th = (384, 216) if fmt == "16x9" else (216, 384)
    cols = 6 if fmt == "16x9" else 8
    rows = (len(times) + cols - 1) // cols
    pad, lab = 8, 26
    g = Image.new("RGB", (cols * (tw + pad) + pad, rows * (th + lab + pad) + pad), (16, 18, 18))
    d = ImageDraw.Draw(g)
    font = ImageFont.truetype(FONTS["mono"](), 16)
    for i, t in enumerate(times):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(mp4), "-frames:v", "1",
                              "-vf", f"scale={tw}:{th}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True).stdout
        if len(raw) != tw * th * 3:
            continue
        x = pad + (i % cols) * (tw + pad)
        y = pad + (i // cols) * (th + lab + pad)
        g.paste(Image.frombytes("RGB", (tw, th), raw), (x, y))
        d.text((x + 2, y + th + 4), f"{t:05.2f}s", fill=(186, 168, 138), font=font)
    g.save(out_png)
    print("contact sheet ->", out_png, len(times), "frames")
