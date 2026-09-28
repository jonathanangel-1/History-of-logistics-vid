"""Narrator screening: render the script with each candidate engine/voice and score it.

    $NARRATOR_PYTHON -m launchfilm.voice_eval edge:en-US-AndrewNeural kokoro:am_michael orpheus:leo ...

For every line: does it transcribe back to the script (faster-whisper small.en), UTMOS
(automatic MOS predictor, 1-5), and pitch movement (st. dev. of F0 in semitones: very low
reads as monotone/robotic). Results go to cache/work/voice_eval/report.json and a table on
stdout; the rendered WAVs stay next to it for listening.
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from launchfilm.config import WORK, Cues
from launchfilm import tts_edge

OUT = WORK / "voice_eval"


def lines():
    return [(l["id"], l["text"]) for l in Cues()["narration"]["lines"]]


# ------------------------------------------------------------------ engines
def render_edge(voice, items, rate="-4%"):
    """One continuous read of the whole script, cut at word boundaries (as in the film)."""
    mp3 = OUT / f"edge_{voice}.mp3"
    words = tts_edge.synth(" ".join(t for _, t in items), voice, rate, mp3)
    y = tts_edge.decode(mp3)
    segs = tts_edge.cut_lines(y, words, [{"text": t} for _, t in items])
    return [(s, tts_edge.SR) for s in segs]


_kokoro = {}


def render_kokoro(voice, items):
    from kokoro import KPipeline
    if "p" not in _kokoro:
        _kokoro["p"] = KPipeline(lang_code="a")
    out = []
    for _, text in items:
        chunks = [a for _, _, a in _kokoro["p"](text, voice=voice, speed=0.95)]
        out.append((np.concatenate([np.asarray(c) for c in chunks]), 24000))
    return out


_orph = {}


def render_orpheus(voice, items, seed=1):
    """Orpheus 3B (Canopy Labs, Apache-2.0 on Llama 3.2) as GGUF via llama.cpp, SNAC 24 kHz decoder."""
    import torch
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    from snac import SNAC
    if "llm" not in _orph:
        path = hf_hub_download("lex-au/Orpheus-3b-FT-Q8_0.gguf", "Orpheus-3b-FT-Q8_0.gguf")
        _orph["llm"] = Llama(model_path=path, n_ctx=2048, n_threads=4, verbose=False, seed=seed)
        _orph["snac"] = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    llm, snac = _orph["llm"], _orph["snac"]
    out = []
    for _, text in items:
        ids = [128259] + llm.tokenize(f"{voice}: {text}".encode(), add_bos=True, special=False) + [128009, 128260]
        codes = []
        llm.reset()
        for tok in llm.generate(ids, temp=0.6, top_p=0.9, top_k=40, repeat_penalty=1.1):
            if tok == 128258 or len(codes) > 1400:
                break
            if tok >= 128266:
                codes.append(tok - 128266)
        n = len(codes) // 7
        l1, l2, l3 = [], [], []
        for i in range(n):
            c = codes[7 * i:7 * i + 7]
            l1.append(c[0])
            l2 += [c[1] - 4096, c[4] - 4 * 4096]
            l3 += [c[2] - 2 * 4096, c[3] - 3 * 4096, c[5] - 5 * 4096, c[6] - 6 * 4096]
        ok = all(0 <= v < 4096 for v in l1 + l2 + l3)
        if not n or not ok:
            out.append((np.zeros(2400, np.float32), 24000))
            continue
        with torch.no_grad():
            y = snac.decode([torch.tensor([l1]), torch.tensor([l2]), torch.tensor([l3])])[0, 0].numpy()
        out.append((y, 24000))
    return out


ENGINES = {"edge": render_edge, "kokoro": render_kokoro, "orpheus": render_orpheus}


def f0_semitone_sd(y, sr):
    import librosa
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
    f0 = librosa.yin(y16, fmin=60, fmax=300, sr=16000)
    rms = librosa.feature.rms(y=y16, frame_length=2048, hop_length=512)[0][:len(f0)]
    f0 = f0[:len(rms)][rms > rms.max() * 0.1]
    if len(f0) < 5:
        return 0.0, 0.0
    st = 12 * np.log2(f0 / np.median(f0))
    return float(np.std(st)), float(np.median(f0))


def main(specs, only=None):
    OUT.mkdir(parents=True, exist_ok=True)
    items = lines()
    if only:
        items = [i for i in items if i[0] in only]
    judge = tts_edge.Judge("small.en")
    rp = OUT / "report.json"
    report = json.loads(rp.read_text()) if rp.exists() else {}
    for spec in specs:
        eng, voice = spec.split(":", 1)
        t0 = time.time()
        rendered = ENGINES[eng](voice, items)
        rows = []
        for (lid, text), (y, sr) in zip(items, rendered):
            tts_edge.write_wav(OUT / f"{eng}_{voice}_{lid}.wav", y, sr)
            j = judge(np.asarray(y, np.float32), sr, text)
            sd, med = f0_semitone_sd(np.asarray(y, np.float32), sr)
            rows.append(dict(id=lid, text=text, f0_sd_st=round(sd, 2), f0_med=round(med, 1),
                             dur=round(len(y) / sr, 2), **j))
            print(f"{spec:36s} {lid} match={j['match']!s:5} mos={j['mos']:.2f} f0sd={sd:.1f}st | {j['heard']}",
                  flush=True)
        report[spec] = {"secs": round(time.time() - t0), "rows": rows,
                        "match_rate": float(np.mean([r["match"] for r in rows])),
                        "mos_mean": float(np.mean([r["mos"] for r in rows])),
                        "mos_min": float(np.min([r["mos"] for r in rows])),
                        "f0_sd_mean": float(np.mean([r["f0_sd_st"] for r in rows]))}
        rp.write_text(json.dumps(report, indent=1))
    print(f"\n{'engine:voice':38s} match  MOS mean  MOS min  F0 sd(st)  render s")
    for spec, r in sorted(report.items(), key=lambda kv: -kv[1]["mos_mean"]):
        print(f"{spec:38s} {r['match_rate']:5.2f}  {r['mos_mean']:8.2f}  {r['mos_min']:7.2f}  "
              f"{r['f0_sd_mean']:9.2f}  {r['secs']:8d}")


if __name__ == "__main__":
    args = sys.argv[1:]
    only = None
    if args and args[0].startswith("--only="):
        only = set(args[0][7:].split(","))
        args = args[1:]
    main(args, only)
