"""Narration: one voice for the whole film, one WAV per cue-sheet line.

Lines are synthesized offline with Kokoro-82M (open weights, Apache-2.0) through
kokoro-onnx, using a fixed blend of two of its stock male voices. A recorded read
replaces any line automatically: drop `<line id>.wav` into `recorded_dir`.
"""
import hashlib
import json
import sys

import numpy as np

from launchfilm.config import ROOT, WORK, Cues, load_sources, src_path

OUT_DIR = WORK / "narration"


def _key(cfg, line):
    blob = json.dumps([cfg["voice_mix"], cfg["speed"], cfg["lang"], line["text"],
                       line.get("speed")], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _engine(cfg, sources):
    from kokoro_onnx import Kokoro
    k = Kokoro(str(src_path(cfg["model"], sources)), str(src_path(cfg["voices"], sources)))
    style = sum(w * k.get_voice_style(v) for v, w in cfg["voice_mix"].items())
    return k, style


LEAD_S = 0.03  # every file starts exactly this long before its first word


def trim(y, sr, tail=0.2, thresh=0.05):
    """Cut to the spoken part so a line's `at_s` is the onset of its first word."""
    fr = int(0.01 * sr)
    env = np.array([np.sqrt(np.mean(y[i:i + fr] ** 2)) for i in range(0, len(y) - fr, fr)])
    on = np.where(env > env.max() * thresh)[0]
    a = on[0] * fr - int(LEAD_S * sr)
    if a < 0:
        y, a = np.concatenate([np.zeros(-a), y]), 0
    b = min(len(y), (on[-1] + 1) * fr + int(tail * sr))
    y = y[a:b].copy()
    k = int(0.06 * sr)
    y[-k:] *= np.linspace(1, 0, k) ** 2
    return y


def line_path(line, cfg):
    rec = ROOT / cfg.get("recorded_dir", "narration_recorded") / f"{line['id']}.wav"
    if rec.exists():
        return rec
    return OUT_DIR / f"{line['id']}_{_key(cfg, line)}.wav"


def build(force=False):
    import soundfile as sf
    cs = Cues()
    cfg = cs["narration"]
    sources = load_sources()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = None
    for line in cfg["lines"]:
        p = line_path(line, cfg)
        if p.exists() and not force:
            continue
        if engine is None:
            engine = _engine(cfg, sources)
        k, style = engine
        y, sr = k.create(line["text"], voice=style, speed=line.get("speed", cfg["speed"]),
                         lang=cfg["lang"])
        y = trim(np.asarray(y, np.float64), sr)
        sf.write(p, y.astype(np.float32), sr)
        print(f"narration {line['id']}: {len(y) / sr:5.2f}s  {line['text']}")
    report()


def report():
    """Print each line's placement so overlaps with the next line are easy to spot."""
    import soundfile as sf
    cs = Cues()
    cfg = cs["narration"]
    lines = cfg["lines"]
    for i, line in enumerate(lines):
        info = sf.info(str(line_path(line, cfg)))
        t0 = line["at_s"]
        t1 = t0 + info.duration - LEAD_S
        nxt = lines[i + 1]["at_s"] if i + 1 < len(lines) else cs.duration
        flag = "  <-- overlaps next line" if t1 > nxt else ""
        print(f"{line['id']:>4} {t0:6.2f}-{t1:6.2f}s gap {nxt - t1:5.2f}  {line['text']}{flag}")


if __name__ == "__main__":
    build(force="--force" in sys.argv)
