"""Narration takes with Microsoft's neural TTS (the Edge "Read aloud" service, via edge-tts).

    $NARRATOR_PYTHON -m launchfilm.tts_edge job.json

Run like a voice-over session with one narrator:
  * "session" takes: the whole script read as one continuous passage (so each line has the
    prosody of a read, not of a sentence in isolation), once per speaking rate; each line
    is cut out of the passage at the service's own word boundaries.
  * "line" takes: each line read on its own, once per speaking rate.
The kept take for a line
  1. transcribes back to exactly the script (faster-whisper), and
  2. scores best on UTMOS (automatic naturalness predictor, 1-5) among takes that pass,
  tie-broken by how close its length is to the line's slot in the cue sheet.
No pitch shift, EQ or effects are applied here; the file is the service's output
(24 kHz, 96 kbit/s MP3, decoded to WAV).
"""
import asyncio
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

SR = 24000
FORMAT = "audio-24khz-96kbitrate-mono-mp3"


def _norm(s):
    s = s.lower().replace("-", " ")
    return re.sub(r"[^a-z' ]", " ", s).split()


def _communicate_cls():
    """edge-tts hard-codes a 48 kbit/s format; load a copy of its module asking for 96 kbit/s."""
    import edge_tts.communicate as c
    src = Path(c.__file__).read_text().replace("audio-24khz-48kbitrate-mono-mp3", FORMAT)
    mod = types.ModuleType("edge_tts._communicate96")
    mod.__dict__.update({"__name__": "edge_tts._communicate96", "__package__": "edge_tts", "__file__": c.__file__})
    exec(compile(src, c.__file__, "exec"), mod.__dict__)
    return mod.Communicate


async def _synth(text, voice, rate, mp3_path):
    Communicate = _communicate_cls()
    words = []
    for attempt in range(5):
        try:
            words = []
            with open(mp3_path, "wb") as f:
                async for ch in Communicate(text, voice, rate=rate, boundary="WordBoundary").stream():
                    if ch["type"] == "audio":
                        f.write(ch["data"])
                    elif ch["type"] == "WordBoundary":
                        words.append((ch["offset"] / 1e7, (ch["offset"] + ch["duration"]) / 1e7, ch["text"]))
            return words
        except Exception as e:  # service hiccups: back off 4, 8, 16, 32 s
            if attempt == 4:
                raise
            await asyncio.sleep(4 * 2 ** attempt)
            print("  retry", e, flush=True)


def synth(text, voice, rate, mp3_path):
    return asyncio.run(_synth(text, voice, rate, mp3_path))


def decode(path, sr=SR):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def write_wav(path, y, sr=SR):
    import soundfile as sf
    sf.write(str(path), y.astype(np.float32), sr, subtype="PCM_24")


def cut_lines(y, words, lines, sr=SR, pre=0.08, post=0.22):
    """Split a passage into lines using word boundaries (lines matched by word count)."""
    out, k = [], 0
    for li, line in enumerate(lines):
        n = len(re.findall(r"[A-Za-z0-9']+", line["text"].replace("-", " ")))
        seg = words[k:k + n]
        k += n
        nxt = words[k][0] if k < len(words) else len(y) / sr
        a = max(0.0, seg[0][0] - pre)
        b = min(nxt - 0.03, seg[-1][1] + post)
        out.append(y[int(a * sr):int(b * sr)].copy())
    return out


class Judge:
    def __init__(self, asr_model="medium.en"):
        from faster_whisper import WhisperModel
        self.asr = WhisperModel(asr_model, device="cpu", compute_type="int8")
        try:
            import torch
            self.mos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        except Exception as e:  # selection still works on the transcript + length
            print("UTMOS unavailable:", e)
            self.mos = None

    def __call__(self, y, sr, text):
        import librosa
        y16 = librosa.resample(y, orig_sr=sr, target_sr=16000).astype(np.float32)
        pad = np.zeros(8000, np.float32)  # the recognizer drops a first word that starts at t=0
        segs, _ = self.asr.transcribe(np.concatenate([pad, y16, pad]), beam_size=5, language="en",
                                      condition_on_previous_text=False)
        heard = " ".join(s.text for s in segs).strip()
        mos = 0.0
        if self.mos is not None:
            import torch
            with torch.no_grad():
                mos = float(self.mos(torch.from_numpy(y16).unsqueeze(0), 16000).item())
        return {"heard": heard, "match": _norm(heard) == _norm(text), "mos": round(mos, 3)}


def main(job_path):
    job = json.loads(Path(job_path).read_text())
    cfg = job["edge"]
    work = Path(job["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    judge = Judge()
    all_lines = job["all_lines"]
    todo = {l["id"]: l for l in job["lines"]}
    takes = {lid: [] for lid in todo}

    passage = " ".join(l["text"] for l in all_lines)
    for rate in cfg["session_rates"]:
        mp3 = work / f"session_{cfg['voice']}_{rate}.mp3"
        words = synth(passage, cfg["voice"], rate, mp3)
        y = decode(mp3)
        for line, seg in zip(all_lines, cut_lines(y, words, all_lines)):
            if line["id"] in todo:
                takes[line["id"]].append({"kind": "session", "rate": rate, "y": seg})
    for rate in cfg["line_rates"]:
        for lid, line in todo.items():
            mp3 = work / f"line_{lid}_{cfg['voice']}_{rate}.mp3"
            synth(line["text"], cfg["voice"], rate, mp3)
            takes[lid].append({"kind": "line", "rate": rate, "y": decode(mp3)})

    report = []
    for lid, line in todo.items():
        for t in takes[lid]:
            t.update(judge(t["y"], SR, line["text"]))
            t["dur"] = round(len(t["y"]) / SR, 2)
            print(f"{lid} {t['kind']:7s} {t['rate']:>4s} match={t['match']} mos={t['mos']:.2f} "
                  f"dur={t['dur']:.2f} | {t['heard']}", flush=True)
        pool = [t for t in takes[lid] if t["match"]] or takes[lid]
        slot = line.get("slot_s")
        best = max(pool, key=lambda t: (round(t["mos"], 1), -abs(t["dur"] - slot) if slot else 0, t["mos"]))
        write_wav(line["out"], best["y"])
        report.append({"id": lid, "text": line["text"],
                       "kept": {k: v for k, v in best.items() if k != "y"},
                       "takes": [{k: v for k, v in t.items() if k != "y"} for t in takes[lid]]})
    Path(job["report_path"]).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
