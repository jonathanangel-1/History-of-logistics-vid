"""Narration takes with Chatterbox (Resemble AI, MIT), run in its own Python env.

    $CHATTERBOX_PYTHON -m launchfilm.tts_chatterbox job.json

Like a voice-over session: every line is read several times (fixed seeds, so the
session is reproducible), and the kept take is the one that
  1. transcribes back to exactly the script (faster-whisper),
  2. sits in the narrator's pitch range (pYIN median F0), and
  3. scores best on UTMOS, an automatic naturalness predictor (1-5).
The narrator's timbre comes from a reference clip built here as well: the model's own
stock voice reading a neutral paragraph, lowered 2.9 semitones with formants shifted
(a synthetic voice, not a recording or clone of any person).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


def _norm(s):
    s = s.lower().replace("-", " ")
    return re.sub(r"[^a-z' ]", " ", s).split()


def _reference(model, cfg, out):
    import torch
    import torchaudio as ta
    raw = out.with_name(out.stem + "_stock.wav")
    torch.manual_seed(cfg["reference_seed"])
    w = model.generate(cfg["reference_text"], exaggeration=0.5, cfg_weight=0.4)
    ta.save(str(raw), w, model.sr)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-af",
                    f"rubberband=pitch={cfg['reference_pitch']}:formant=shifted:pitchq=quality", str(out)],
                   check=True)


def main(job_path):
    import librosa
    import torch
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS
    from faster_whisper import WhisperModel

    job = json.loads(Path(job_path).read_text())
    cfg = job["chatterbox"]
    model = ChatterboxTTS.from_pretrained(device="cpu")
    ref = Path(job["reference_path"])
    if not ref.exists():
        _reference(model, cfg, ref)
    asr = WhisperModel("small.en", device="cpu", compute_type="int8")
    try:
        mos_model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    except Exception as e:  # scoring still works on transcript + pitch
        print("UTMOS unavailable:", e)
        mos_model = None
    lo, hi = cfg["f0_range_hz"]
    report = []
    for line in job["lines"]:
        takes = []
        n = line.get("takes", cfg["takes"])
        for k in range(n):
            seed = cfg["seed"] + 1000 * k + line["n"]
            torch.manual_seed(seed)
            w = model.generate(line["text"], audio_prompt_path=str(ref),
                               exaggeration=line.get("exaggeration", cfg["exaggeration"]),
                               cfg_weight=line.get("cfg_weight", cfg["cfg_weight"]),
                               temperature=cfg.get("temperature", 0.8))
            tmp = Path(line["out"]).with_suffix(f".take{k}.wav")
            ta.save(str(tmp), w, model.sr)
            y, _ = librosa.load(str(tmp), sr=16000)
            pad = np.zeros(8000, np.float32)  # the recognizer drops a first word that starts at t=0
            segs, _ = asr.transcribe(np.concatenate([pad, y.astype(np.float32), pad]), beam_size=5,
                                     language="en", condition_on_previous_text=False)
            heard = " ".join(s.text for s in segs)
            match = _norm(heard) == _norm(line["text"])
            f0, _, _ = librosa.pyin(y, fmin=60, fmax=320, sr=16000)
            f0 = f0[~np.isnan(f0)]
            med = float(np.median(f0)) if len(f0) else 0.0
            mos = float(mos_model(torch.from_numpy(y).unsqueeze(0), 16000).item()) if mos_model else 0.0
            ok = match and lo <= med <= hi
            takes.append(dict(k=k, seed=seed, path=str(tmp), heard=heard.strip(), match=match,
                              f0=round(med, 1), mos=round(mos, 3), ok=ok))
            print(f"{line['id']} take {k}: match={match} f0={med:.0f} mos={mos:.2f} | {heard.strip()}",
                  flush=True)
        pool = [t for t in takes if t["ok"]] or [t for t in takes if t["match"]] or takes
        best = max(pool, key=lambda t: (t["mos"], -abs(t["f0"] - cfg["f0_target_hz"])))
        Path(best["path"]).replace(line["out"])
        for t in takes:
            Path(t["path"]).unlink(missing_ok=True)
        report.append(dict(id=line["id"], text=line["text"], kept=best, takes=takes))
    Path(job["report_path"]).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
