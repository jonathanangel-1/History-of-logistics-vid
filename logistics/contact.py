"""Tile ~12 frames of the finished film into a contact sheet.

  python3 -m logistics.contact [video] [out.png]
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from logistics.cues import CueSheet, ROOT

CS = CueSheet()
BUILD = os.path.join(ROOT, "build")
TIMES = [4.0, 12.0, 24.0, 30.0, 40.0, 50.5, 57.0, 67.0, 81.0, 88.5, 108.5, 116.0]
TW, TH, COLS, PAD = 640, 360, 4, 12


def grab(video, t):
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1",
           "-vf", f"scale={TW}:{TH}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    buf = subprocess.check_output(cmd)
    return Image.fromarray(np.frombuffer(buf, np.uint8).reshape(TH, TW, 3))


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BUILD, "history_of_logistics.mp4")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BUILD, "contact_sheet.png")
    rows = (len(TIMES) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * TW + (COLS + 1) * PAD, rows * (TH + 30) + (rows + 1) * PAD), (12, 12, 14))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for k, t in enumerate(TIMES):
        r, c = divmod(k, COLS)
        x, y = PAD + c * (TW + PAD), PAD + r * (TH + 30 + PAD)
        sheet.paste(grab(video, t), (x, y))
        bar = t / CS.bar_s
        shot = next(s["id"] for s in CS.shots if s["start"] <= bar < s["end"])
        d.text((x + 4, y + TH + 5), f"{t:6.2f}s  bar {bar:5.2f}  {shot}", fill=(200, 200, 205), font=font)
    sheet.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
