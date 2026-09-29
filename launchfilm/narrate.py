"""Narration: one voice for the whole film, one WAV per cue-sheet line.

`narration.model` picks the engine: `tts_edge` (Microsoft neural voice, several takes per
line, best kept; see tts_edge.py) or `tts_chatterbox` (v2). The engine runs in the Python
named by $NARRATOR_PYTHON (it needs torch for the take scoring). A recorded read replaces
any line automatically: drop `<line id>.wav` into `recorded_dir`.
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

from launchfilm.config import ROOT, WORK, Cues

OUT_DIR = WORK / "narration"
LEAD_S = 0.03  # every file starts exactly this long before its first word


ENGINE_CFG = {"tts_edge": "edge", "tts_chatterbox": "chatterbox"}


def _key(cfg, line):
    blob = json.dumps([cfg[ENGINE_CFG[cfg["model"]]], line["text"], line.get("tts_text"), line.get("takes"),
                       line.get("exaggeration"), line.get("cfg_weight")], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def trim(y, sr, tail=0.2, thresh=0.05, onset_thresh=0.012):
    """Cut to the spoken part so a line's `at_s` is the onset of its first word. The onset
    threshold sits low (-38 dB re the loudest 10 ms) so a voiced B or the start of a Ch,
    which are far quieter than the vowel after them, stay in the file."""
    fr = int(0.01 * sr)
    env = np.array([np.sqrt(np.mean(y[i:i + fr] ** 2)) for i in range(0, len(y) - fr, fr)])
    a = np.where(env > env.max() * onset_thresh)[0][0] * fr - int(LEAD_S * sr)
    b = (np.where(env > env.max() * thresh)[0][-1] + 1) * fr + int(tail * sr)
    if a < 0:
        y, a, b = np.concatenate([np.zeros(-a), y]), 0, b - a
    b = min(len(y), b)
    y = y[a:b].copy()
    k = int(0.06 * sr)
    y[-k:] *= np.linspace(1, 0, k) ** 2
    return y


def split(y, sr, n, thresh=0.04, min_gap=0.06):
    """Cut one read into `n` phrases at its n-1 longest pauses (each chunk trimmed)."""
    fr = int(0.01 * sr)
    env = np.array([np.sqrt(np.mean(y[i:i + fr] ** 2)) for i in range(0, len(y) - fr, fr)])
    quiet = env < env.max() * thresh
    gaps, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if i > 0 and j < len(quiet) and (j - i) * 0.01 >= min_gap:
                gaps.append((j - i, (i + j) // 2))
            i = j
        else:
            i += 1
    cuts = sorted(c for _, c in sorted(gaps, reverse=True)[:n - 1])
    if len(cuts) != n - 1:
        raise ValueError(f"found {len(cuts) + 1} phrases, expected {n}")
    bounds = [0] + [c * fr for c in cuts] + [len(y)]
    return [trim(y[a:b], sr) for a, b in zip(bounds, bounds[1:])]


def line_path(line, cfg):
    rec = ROOT / cfg.get("recorded_dir", "narration_recorded") / f"{line['id']}.wav"
    if rec.exists():
        return rec
    return OUT_DIR / f"{line['id']}_{_key(cfg, line)}.wav"


def placements(line, cfg, sr=48000, decode=None):
    """[(at_s, mono samples)] for a line; `split_at_s` spreads one read over several beats."""
    x = decode(line_path(line, cfg), sr)
    if "split_at_s" in line:
        return list(zip(line["split_at_s"], split(x, sr, len(line["split_at_s"]))))
    return [(line["at_s"], trim(x, sr))]


def _job(cfg, todo):
    if cfg["model"] == "tts_edge":
        return {"edge": cfg["edge"], "work_dir": str(OUT_DIR / "takes"),
                "report_path": str(OUT_DIR / "takes_report.json"),
                "all_lines": [{k: l[k] for k in ("id", "text", "tts_text") if k in l} for l in cfg["lines"]],
                "lines": [dict(l, out=str(line_path(l, cfg))) for l in todo]}
    cb = cfg["chatterbox"]
    refkey = hashlib.sha1(json.dumps([cb["reference_text"], cb["reference_seed"],
                                      cb["reference_pitch"]]).encode()).hexdigest()[:12]
    return {"chatterbox": cb, "reference_path": str(OUT_DIR / f"reference_{refkey}.wav"),
            "report_path": str(OUT_DIR / "takes_report.json"),
            "lines": [dict(l, n=i, out=str(line_path(l, cfg)))
                      for i, l in enumerate(cfg["lines"]) if l in todo]}


def build(force=False):
    import soundfile as sf
    cfg = Cues()["narration"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [l for l in cfg["lines"] if force or not line_path(l, cfg).exists()]
    if todo:
        job_path = OUT_DIR / "job.json"
        job_path.write_text(json.dumps(_job(cfg, todo), indent=1))
        py = os.environ.get("NARRATOR_PYTHON", os.environ.get("CHATTERBOX_PYTHON", sys.executable))
        subprocess.run([py, "-m", f"launchfilm.{cfg['model']}", str(job_path)], cwd=ROOT, check=True)
        for l in todo:
            p = line_path(l, cfg)
            y, sr = sf.read(str(p))
            sf.write(str(p), trim(np.asarray(y, np.float64), sr).astype(np.float32), sr)
    report()


def report():
    """Print each placement so overlaps with the next one are easy to spot."""
    from launchfilm.audio import decode_file
    cfg = Cues()["narration"]
    dec = lambda p, sr: decode_file(p, sr).mean(axis=1)
    rows = []
    for line in cfg["lines"]:
        for t0, x in placements(line, cfg, decode=dec):
            rows.append((t0, t0 + len(x) / 48000 - LEAD_S, line["id"], line["text"]))
    rows.sort()
    end = Cues().duration
    for i, (t0, t1, lid, text) in enumerate(rows):
        nxt = rows[i + 1][0] if i + 1 < len(rows) else end
        flag = "  <-- overlaps next" if t1 > nxt else ""
        print(f"{lid:>4} {t0:6.2f}-{t1:6.2f}s gap {nxt - t1:5.2f}  {text}{flag}")


if __name__ == "__main__":
    build(force="--force" in sys.argv)
