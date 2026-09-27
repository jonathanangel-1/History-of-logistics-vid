"""Render driver: frames -> per-shot H.264 segments -> final MP4 with the score.

  python3 -m logistics.render --stills 10,25.5,40          # quick PNG stills
  python3 -m logistics.render --shots silk,sail            # re-render some shots
  python3 -m logistics.render --all                        # every shot + audio + mux
"""
import argparse
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time

import cairo
import numpy as np

from logistics.cues import CueSheet, frame_info, ROOT
from logistics.gfx import Glow, new_surface, W, H, smooth
from logistics.post import finish, surface_to_float
from logistics.text import render_titles
from logistics.scenes import SCENES

CS = CueSheet()
BUILD = os.path.join(ROOT, "build")


def render_frame(i):
    t = i / CS.fps
    F = frame_info(CS, t)
    grade = F.shot["grade"]
    if F.silent:
        return np.zeros((H, W, 3), np.uint8)
    surf, ctx = new_surface()
    ctx.set_source_rgb(0, 0, 0)
    ctx.paint()
    glow = Glow()
    extra = SCENES[F.shot["id"]](ctx, glow, F) or {}
    surf.flush()
    img = surface_to_float(surf)
    gl = glow.resolve()
    text = render_titles(CS, t)
    # short exposure punch on every impact, on top of whatever the scene does
    punch = 0.0
    for im in CS.impacts:
        dt = t - CS.t(im["at"])
        if 0 <= dt < 0.5:
            punch += 0.18 * im["strength"] * math.exp(-dt / 0.07)
    fade = smooth(t / 1.6) * smooth((CS.duration - t) / 1.2)
    return finish(img, gl, extra.get("grade", grade), i,
                  flash=extra.get("flash", 0.0) + punch,
                  exposure=extra.get("exposure", 1.0),
                  bloom_boost=extra.get("bloom", 0.0),
                  ca=extra.get("ca", 1.0) + 1.5 * F.impact,
                  text=text, fade=fade * extra.get("fade", 1.0))


def _job(i):
    return render_frame(i).tobytes()


def shot_frames(shot):
    fpb = CS.bar_s * CS.fps
    return int(round(shot["start"] * fpb)), int(round(shot["end"] * fpb))


def render_shot(shot, pool, crf):
    os.makedirs(os.path.join(BUILD, "shots"), exist_ok=True)
    out = os.path.join(BUILD, "shots", f"{shot['id']}.mp4")
    a, b = shot_frames(shot)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(CS.fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-tune", "grain", "-pix_fmt", "yuv420p", "-g", "120", "-bf", "3", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for k, buf in enumerate(pool.imap(_job, range(a, b), chunksize=4)):
        p.stdin.write(buf)
        if k % 60 == 0:
            el = time.time() - t0
            print(f"  {shot['id']}: {k}/{b - a} frames  {el:.0f}s", flush=True)
    p.stdin.close()
    p.wait()
    print(f"  {shot['id']} done in {time.time() - t0:.0f}s -> {out}", flush=True)
    return out


def mux(out_path):
    lst = os.path.join(BUILD, "shots", "list.txt")
    with open(lst, "w") as f:
        for s in CS.shots:
            f.write(f"file '{s['id']}.mp4'\n")
    wav = os.path.join(BUILD, "score.wav")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav,
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
           "-t", f"{CS.duration:.3f}", "-movflags", "+faststart", out_path]
    subprocess.check_call(cmd)
    print("wrote", out_path)


def deliver(master, out_path, vbitrate):
    """Two-pass re-encode of the master to a fixed size (the grain makes CRF output very large)."""
    log = os.path.join(BUILD, "x264pass")
    base = ["ffmpeg", "-y", "-loglevel", "error", "-i", master, "-c:v", "libx264", "-preset", "slow",
            "-tune", "grain", "-b:v", vbitrate, "-maxrate", vbitrate, "-bufsize", "16M",
            "-pix_fmt", "yuv420p", "-g", "120", "-passlogfile", log]
    subprocess.check_call(base + ["-pass", "1", "-an", "-f", "null", os.devnull])
    subprocess.check_call(base + ["-pass", "2", "-c:a", "copy", "-movflags", "+faststart", out_path])
    print("wrote", out_path)


def stills(times, outdir):
    from PIL import Image
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for t in times:
        i = int(round(t * CS.fps))
        img = render_frame(i)
        p = os.path.join(outdir, f"still_{t:07.2f}.png")
        Image.fromarray(img).save(p)
        paths.append(p)
        print("wrote", p, flush=True)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", default=None)
    ap.add_argument("--shots", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--mux", action="store_true")
    ap.add_argument("--crf", type=int, default=21)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(BUILD, "history_of_logistics.mp4"))
    ap.add_argument("--master", default=os.path.join(BUILD, "history_of_logistics_master.mp4"))
    ap.add_argument("--deliver", action="store_true")
    ap.add_argument("--vbitrate", default="6500k")
    a = ap.parse_args()

    if a.stills:
        stills([float(x) for x in a.stills.split(",")], os.path.join(BUILD, "stills"))
        return
    shots = CS.shots if a.all else [s for s in CS.shots if a.shots and s["id"] in a.shots.split(",")]
    if shots:
        with mp.get_context("fork").Pool(a.workers) as pool:
            for s in shots:
                render_shot(s, pool, a.crf)
    if a.all or a.mux:
        if a.all or not os.path.exists(os.path.join(BUILD, "score.wav")):
            from logistics import audio
            audio.main()
        mux(a.master)
    if a.all or a.mux or a.deliver:
        deliver(a.master, a.out, a.vbitrate)


if __name__ == "__main__":
    main()
