"""Soundtrack: licensed music edited on the bar grid, public-domain voices with
ducking, real/CC0 sound effects, mastered to -14 LUFS integrated / -1 dBTP."""
import json
import re
import subprocess

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt, sosfiltfilt

from launchfilm.config import WORK, Cues, load_sources, src_path

TARGET_LUFS = -14.0
CEILING_DBTP = -1.3


def db(x):
    return 10 ** (x / 20.0)


def decode(sid, t0=None, t1=None, sr=48000, sources=None, af=None):
    """Decode a window of a source (times in the ORIGINAL file's timeline) to float32 stereo."""
    sources = sources or load_sources()
    s = sources[sid]
    offset = s.get("segment", [0])[0]
    cmd = ["ffmpeg", "-v", "error"]
    if t0 is not None:
        cmd += ["-ss", f"{t0 - offset:.4f}"]
    cmd += ["-i", str(src_path(sid, sources))]
    if t1 is not None:
        cmd += ["-t", f"{t1 - t0:.4f}"]
    cmd += ["-vn", "-ac", "2", "-ar", str(sr)]
    if af:
        cmd += ["-af", af]
    cmd += ["-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)


def fades(x, sr, fin=0.0, fout=0.0):
    n = len(x)
    g = np.ones(n)
    if fin > 0:
        k = min(n, int(fin * sr))
        g[:k] = np.linspace(0, 1, k) ** 2
    if fout > 0:
        k = min(n, int(fout * sr))
        g[n - k:] *= np.linspace(1, 0, k) ** 2
    return x * g[:, None]


def place(bus, x, t, sr):
    i = int(round(t * sr))
    if i < 0:
        x, i = x[-i:], 0
    n = min(len(x), len(bus) - i)
    if n > 0:
        bus[i:i + n] += x[:n]


def sweep_lowpass(x, sr, f0, f1, blocks=64):
    """Time-varying lowpass (block-wise 4th-order Butterworth, overlap-add crossfaded)."""
    n = len(x)
    out = np.zeros_like(x)
    edges = np.linspace(0, n, blocks + 1).astype(int)
    for b in range(blocks):
        a, e = edges[b], edges[b + 1]
        pad = min(a, int(0.02 * sr))
        f = f0 * (f1 / f0) ** ((b + 0.5) / blocks)
        sos = butter(4, min(f, sr * 0.45), "low", fs=sr, output="sos")
        y = sosfilt(sos, x[a - pad:e], axis=0)
        out[a:e] = y[pad:]
    return out


def measure(path):
    """Integrated loudness, true peak, LRA via ffmpeg ebur128."""
    r = subprocess.run(["ffmpeg", "-nostats", "-i", str(path), "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True)
    tail = r.stderr[r.stderr.rfind("Summary:"):]
    I = float(re.search(r"I:\s+(-?[\d.]+) LUFS", tail).group(1))
    TP = float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", tail).group(1))
    LRA = float(re.search(r"LRA:\s+(-?[\d.]+) LU", tail).group(1))
    return {"I": I, "TP": TP, "LRA": LRA, "summary": tail.strip()}


def write_wav(path, x, sr):
    x = np.clip(x, -1, 1).astype(np.float32)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "-",
                    "-c:a", "pcm_s24le", str(path)], input=x.tobytes(), check=True)


def true_peak_limit(x, sr, ceiling_db, lookahead=0.004, release=0.08):
    """Brickwall limiter driven by a 4x-oversampled peak estimate."""
    ceil = db(ceiling_db)
    up = np.abs(resample_poly(x, 4, 1, axis=0)).max(axis=1)
    pk = up[: len(up) // 4 * 4].reshape(-1, 4).max(axis=1)
    pk = np.concatenate([pk, np.zeros(len(x) - len(pk))])
    need = np.minimum(1.0, ceil / np.maximum(pk, 1e-9))
    la = int(lookahead * sr)
    # hold the minimum over the lookahead window so gain is down before the peak arrives
    from scipy.ndimage import minimum_filter1d
    need = minimum_filter1d(need, size=2 * la + 1, origin=0)
    g = np.empty_like(need)
    rel = np.exp(-1.0 / (release * sr))
    cur = 1.0
    for i in range(len(need)):  # attack instant (already look-ahead), smooth release
        v = need[i]
        cur = v if v < cur else v + (cur - v) * rel
        g[i] = cur
    return x * g[:, None]


def build(out_path=None, stems=False):
    cs = Cues()
    sources = load_sources()
    sr = cs.sr
    n = int(round(cs.duration * sr)) + sr // 2
    music = np.zeros((n, 2))
    voice = np.zeros((n, 2))
    sfx = np.zeros((n, 2))

    # ---- music edit
    m = cs["music"]
    ms = sources[m["src"]]
    bar_src = 4 * 60.0 / ms["bpm"]
    track = decode(m["src"], sources=sources)
    for seg in m["segments"]:
        a = ms["first_beat_s"] + seg["src_bar"] * bar_src
        pre = 0.006  # start a hair early so the downbeat transient is intact
        i0 = int(round((a - pre) * sr))
        i1 = int(round((a + seg["bars"] * bar_src) * sr))
        x = track[i0:i1].copy()
        if "lp_hz" in seg:
            x = sweep_lowpass(x, sr, *seg["lp_hz"])
        if "gain_db" in seg:
            g0, g1 = seg["gain_db"] if isinstance(seg["gain_db"], list) else (seg["gain_db"],) * 2
            x *= db(np.linspace(g0, g1, len(x)))[:, None]
        x = fades(x, sr, seg.get("fade_in_s", 0.004), seg.get("fade_out_s", 0.004))
        place(music, x, cs.t(seg["at"]) - pre, sr)

    # ---- voices (cleaned, level-matched) and duck envelope
    duck = np.ones(n)
    dk = cs["duck"]
    for v in cs["voices"]:
        af = ("highpass=f=95,lowpass=f=9000,afftdn=nr=12:nf=-42,"
              "equalizer=f=2800:t=q:w=1.2:g=3,equalizer=f=220:t=q:w=1:g=-2,"
              "acompressor=threshold=-24dB:ratio=3:attack=5:release=120:makeup=4")
        x = decode(v["src"], v["in"], v["out"], sr, sources, af=af)
        x = x.mean(axis=1, keepdims=True).repeat(2, axis=1)
        rms = np.sqrt(np.mean(x[np.abs(x[:, 0]) > 1e-3] ** 2)) + 1e-9
        x *= db(-19.0) / rms * db(v.get("gain_db", 0))
        x = fades(x, sr, 0.012, 0.03)
        t0 = v["at_s"]
        place(voice, x, t0, sr)
        d = db(v.get("duck_db", dk["db"]))
        a0 = int((t0 - dk["pre_s"]) * sr)
        a1 = int((t0 + len(x) / sr + dk["post_s"]) * sr)
        att = int(dk["attack_s"] * sr)
        rel = int(dk["release_s"] * sr)
        env = np.ones(n)
        env[a0:a1] = d
        env[max(0, a0 - att):a0] = np.linspace(1, d, a0 - max(0, a0 - att))
        env[a1:a1 + rel] = np.linspace(d, 1, rel)
        duck = np.minimum(duck, env)
    music *= duck[:, None]

    # ---- sound effects
    for e in cs["sfx"]:
        af = []
        if "lp_hz" in e:
            af.append(f"lowpass=f={e['lp_hz']}")
        if "hp_hz" in e:
            af.append(f"highpass=f={e['hp_hz']}")
        x = decode(e["src"], e["in"], e["out"], sr, sources, af=",".join(af) or None)
        if e.get("reverse"):
            x = x[::-1].copy()
        x *= db(e.get("gain_db", 0))
        x = fades(x, sr, e.get("fade_in_s", 0.01), e.get("fade_out_s", 0.05))
        place(sfx, x, cs.when(e), sr)

    mix = music + voice + sfx
    mix = mix[: int(round(cs.duration * sr))]
    # tiny tail fade so the file never ends on a click
    mix = fades(mix, sr, 0.0, 0.02)

    WORK.mkdir(parents=True, exist_ok=True)
    out_path = out_path or (WORK / "mix.wav")
    scale = 1.0 / max(1e-6, np.abs(mix).max())
    for _ in range(6):
        y = true_peak_limit(mix * scale, sr, CEILING_DBTP)
        write_wav(out_path, y, sr)
        meas = measure(out_path)
        if abs(meas["I"] - TARGET_LUFS) < 0.1:
            break
        scale *= db(TARGET_LUFS - meas["I"])
    (WORK / "loudness.json").write_text(json.dumps(meas, indent=1))
    print(f"mix -> {out_path}: I={meas['I']} LUFS  TP={meas['TP']} dBTP  LRA={meas['LRA']}")
    if stems:
        for name, b in (("music", music), ("voice", voice), ("sfx", sfx)):
            write_wav(WORK / f"stem_{name}.wav", b[: len(y)] * scale, sr)
    return out_path


if __name__ == "__main__":
    build(stems=True)
