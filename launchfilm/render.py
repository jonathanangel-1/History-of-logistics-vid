"""Command line entry point for the v2 launch film.

    python3 -m launchfilm.render --all                 fetch, mix, both formats, credits, contact sheets
    python3 -m launchfilm.render --fetch               download sources.yaml into cache/sources
    python3 -m launchfilm.render --narration           (re)synthesize narration lines -> cache/work/narration
    python3 -m launchfilm.render --audio               narration + mix -> cache/work/mix.wav
    python3 -m launchfilm.render --format 16x9         one format (uses the existing mix)
    python3 -m launchfilm.render --stills 12.5,40      PNG stills (seconds) -> out/stills/
    python3 -m launchfilm.render --contact             contact sheets from the rendered MP4s
    python3 -m launchfilm.render --credits             regenerate CREDITS.csv / CREDITS.md
"""
import argparse
import shutil
from pathlib import Path

from launchfilm import audio, contact, credits, fetch, narrate, picture
from launchfilm.config import FORMATS, OUT, WORK, Cues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--narration", action="store_true")
    ap.add_argument("--audio", action="store_true")
    ap.add_argument("--format", choices=list(FORMATS), action="append")
    ap.add_argument("--stills")
    ap.add_argument("--contact", action="store_true")
    ap.add_argument("--credits", action="store_true")
    ap.add_argument("--artifacts", help="also copy deliverables into this directory")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if a.all or a.fetch:
        fetch.fetch()
    mix = WORK / "mix.wav"
    if a.all or a.audio or a.narration or ((a.format) and not mix.exists()):
        narrate.build()
    if a.all or a.audio or ((a.format) and not mix.exists()):
        audio.build(stems=True)
    fmts = list(FORMATS) if a.all else ([] if a.stills else (a.format or []))
    for f in fmts:
        out = OUT / FORMATS[f]["name"]
        print(f"render {f} -> {out}")
        picture.render(f, out, audio=mix)
    if a.stills:
        cs = Cues()
        d = OUT / "stills"
        d.mkdir(parents=True, exist_ok=True)
        frames = [int(round(float(s) * cs.fps)) for s in a.stills.split(",")]
        for f in (a.format or list(FORMATS)):
            picture.render(f, None, only_frames=frames, still_dir=d)
        print("stills ->", d)
    if a.all or a.contact:
        for f in FORMATS:
            mp4 = OUT / FORMATS[f]["name"]
            if mp4.exists():
                contact.sheet(mp4, OUT / f"contact_sheet_{f}.png", f)
    if a.all or a.credits:
        credits.write()
    if a.artifacts:
        dst = Path(a.artifacts)
        dst.mkdir(parents=True, exist_ok=True)
        for f in FORMATS:
            for p in (OUT / FORMATS[f]["name"], OUT / f"contact_sheet_{f}.png"):
                if p.exists():
                    shutil.copy2(p, dst / p.name)
        for p in ("CREDITS.csv", "CREDITS.md"):
            shutil.copy2(Path(__file__).parent / p, dst / p)
        print("artifacts ->", dst)


if __name__ == "__main__":
    main()
