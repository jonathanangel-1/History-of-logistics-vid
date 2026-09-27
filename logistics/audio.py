"""Fully synthesized score. Oscillators, noise, envelopes, filters and reverb only.

Every event is placed from cuesheet.json (tempo, chords, motif, impacts, risers, silences),
so the score and the picture edit share one clock.
"""
import os
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logistics.cues import CueSheet, ROOT  # noqa: E402

CS = CueSheet()
SR = CS.sr
SPB = CS.spb
RNG = np.random.default_rng(1956)

PCS = {"Dm": [2, 5, 9], "C": [0, 4, 7], "Bb": [10, 2, 5], "A": [9, 1, 4], "F": [5, 9, 0]}
ROOTS = {"Dm": 38, "C": 36, "Bb": 34, "A": 33, "F": 41}


def hz(m):
    return 440.0 * 2 ** ((np.asarray(m, dtype=np.float64) - 69) / 12)


def tb(bar, beat=0.0):
    return CS.t(bar) + beat * SPB


def n_(sec):
    return int(round(sec * SR))


def tvec(dur):
    return np.arange(n_(dur)) / SR


# ---------------------------------------------------------------- oscillators

def phase_from_freq(freq, n, phase0=0.0):
    f = np.broadcast_to(np.asarray(freq, dtype=np.float64), (n,))
    ph = phase0 + np.cumsum(f) / SR
    return ph % 1.0, f / SR


def saw(freq, n, phase0=0.0):
    """PolyBLEP band-limited sawtooth."""
    p, dt = phase_from_freq(freq, n, phase0)
    y = 2.0 * p - 1.0
    m = p < dt
    x = p[m] / dt[m]
    y[m] -= x + x - x * x - 1.0
    m = p > 1.0 - dt
    x = (p[m] - 1.0) / dt[m]
    y[m] -= x * x + x + x + 1.0
    return y


def sine(freq, n, phase0=0.0):
    p, _ = phase_from_freq(freq, n, phase0)
    return np.sin(2 * np.pi * p)


def noise(n, rng=RNG):
    return rng.standard_normal(n)


def pink(n, rng=RNG):
    w = rng.standard_normal(n)
    b, a = [0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400]
    return signal.lfilter(b, a, w) * 4.0


def brown(n, rng=RNG):
    y = np.cumsum(rng.standard_normal(n))
    y = signal.lfilter([1, -1], [1, -0.999], y)
    return y / (np.std(y) + 1e-9)


# ---------------------------------------------------------------- envelopes

def adsr(n, a=0.01, d=0.1, s=0.7, r=0.2, sustain_n=None):
    """Attack/decay/sustain then release over the last `r` seconds of n samples."""
    env = np.full(n, s, dtype=np.float64)
    na, nd, nr = n_(a), n_(d), n_(r)
    na = max(1, min(na, n))
    env[:na] = np.linspace(0, 1, na) ** 1.5
    if nd > 0 and na < n:
        e = min(n, na + nd)
        env[na:e] = 1 - (1 - s) * (np.linspace(0, 1, e - na) ** 0.7)
    nr = min(nr, n)
    if nr > 0:
        env[n - nr:] *= np.linspace(1, 0, nr) ** 2
    return env


def expdec(n, tau):
    return np.exp(-np.arange(n) / (tau * SR))


# ---------------------------------------------------------------- filters

def lp(x, fc, order=2):
    fc = min(fc, SR * 0.45)
    sos = signal.butter(order, fc, "low", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def hp(x, fc, order=2):
    sos = signal.butter(order, fc, "high", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def bp(x, lo, hi, order=2):
    hi = min(hi, SR * 0.45)
    sos = signal.butter(order, [lo, hi], "band", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def peak_bp(x, fc, q):
    b, a = signal.iirpeak(min(fc, SR * 0.45), q, fs=SR)
    return signal.lfilter(b, a, x)


def tv_lowpass(x, cutoff, q=0.707, block=128):
    """Time-varying resonant lowpass (RBJ biquad, coefficients updated per block)."""
    n = len(x)
    cutoff = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), (n,))
    y = np.empty(n)
    zi = np.zeros(2)
    for s in range(0, n, block):
        e = min(n, s + block)
        fc = float(np.clip(cutoff[s], 20, SR * 0.45))
        w0 = 2 * np.pi * fc / SR
        al = np.sin(w0) / (2 * q)
        cw = np.cos(w0)
        b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2])
        a = np.array([1 + al, -2 * cw, 1 - al])
        y[s:e], zi = signal.lfilter(b / a[0], a / a[0], x[s:e], zi=zi)
    return y


def tv_bandpass(x, center, q=2.0, block=128):
    n = len(x)
    center = np.broadcast_to(np.asarray(center, dtype=np.float64), (n,))
    y = np.empty(n)
    zi = np.zeros(2)
    for s in range(0, n, block):
        e = min(n, s + block)
        fc = float(np.clip(center[s], 30, SR * 0.45))
        w0 = 2 * np.pi * fc / SR
        al = np.sin(w0) / (2 * q)
        cw = np.cos(w0)
        b = np.array([al, 0, -al])
        a = np.array([1 + al, -2 * cw, 1 - al])
        y[s:e], zi = signal.lfilter(b / a[0], a / a[0], x[s:e], zi=zi)
    return y


def pan2(x, p):
    """Constant-power pan, p in [-1, 1]."""
    a = (p + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)])


# ---------------------------------------------------------------- instruments

def karplus(freq, dur, bright=0.6, decay=0.996, rng=RNG):
    total = n_(dur)
    N = max(2, int(round(SR / freq - 0.5)))
    y = np.zeros(total + N + 1)
    exc = rng.uniform(-1, 1, N)
    exc = lp(exc, 800 + 7000 * bright, 1)
    exc -= exc.mean()
    y[1:N + 1] = exc
    s = N + 1
    while s < len(y):
        e = min(len(y), s + N)
        k = e - s
        y[s:e] = decay * 0.5 * (y[s - N:s - N + k] + y[s - N - 1:s - N - 1 + k])
        s = e
    return y[1:total + 1]


def oud(m, dur, vel=1.0):
    f = float(hz(m))
    x = karplus(f, dur + 1.2, bright=0.45 + 0.3 * vel, decay=0.9975)
    body = peak_bp(x, 220, 3) * 0.6 + peak_bp(x, 480, 4) * 0.35
    x = x + body
    x *= adsr(len(x), 0.002, 0.05, 1.0, 0.25)
    return x * vel


def ney(m, dur, vel=1.0):
    n = n_(dur + 0.6)
    t = np.arange(n) / SR
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.2 * t) * np.clip(t / 0.6, 0, 1)
    f = float(hz(m)) * vib
    tone = sine(f, n) + 0.18 * sine(2 * f, n) + 0.07 * sine(3 * f, n)
    breath = peak_bp(noise(n), float(hz(m)), 6) * 0.5 + hp(noise(n), 3000) * 0.05
    x = tone + breath
    return x * adsr(n, 0.18, 0.2, 0.8, 0.5) * vel * 0.5


def celesta(m, dur, vel=1.0):
    n = n_(dur + 2.5)
    f = float(hz(m))
    x = (sine(f, n) * expdec(n, 1.6)
         + 0.35 * sine(2 * f, n) * expdec(n, 0.7)
         + 0.12 * sine(3.01 * f, n) * expdec(n, 0.35)
         + 0.08 * sine(4.16 * f, n) * expdec(n, 0.2)
         + 0.05 * sine(5.43 * f, n) * expdec(n, 0.08))
    x *= adsr(n, 0.002, 0.0, 1.0, 0.3)
    return x * vel * 0.6


def supersaw(m, dur, vel=1.0, voices=7, detune=12.0, cutoff=3000.0, a=0.3, r=0.8,
             vib=0.0, width=0.8, rng=RNG, env=None):
    """Detuned saw ensemble -> stereo. Returns (2, n)."""
    n = n_(dur + r)
    t = np.arange(n) / SR
    f0 = float(hz(m))
    out = np.zeros((2, n))
    for v in range(voices):
        cents = (v - (voices - 1) / 2) / max(1, (voices - 1) / 2) * detune + rng.normal(0, 1.5)
        vf = 1 + vib * np.sin(2 * np.pi * (4.8 + rng.uniform(0, 1.2)) * t + rng.uniform(0, 6.28)) * np.clip(t / 0.8, 0, 1)
        x = saw(f0 * 2 ** (cents / 1200) * vf, n, rng.uniform())
        p = (v / max(1, voices - 1) * 2 - 1) * width
        out += pan2(x, p)
    out /= np.sqrt(voices)
    out = np.stack([lp(out[0], cutoff), lp(out[1], cutoff)])
    e = adsr(n, a, 0.3, 0.85, r) if env is None else env[:n]
    return out * e * vel


def brass(m, dur, vel=1.0, rng=RNG, a=0.06, bright=1.0):
    n = n_(dur + 0.5)
    t = np.arange(n) / SR
    f0 = float(hz(m))
    x = np.zeros(n)
    for c in (-6, 0, 7):
        vf = 1 + 0.004 * np.sin(2 * np.pi * 5.0 * t + rng.uniform(0, 6)) * np.clip(t / 0.5, 0, 1)
        x += saw(f0 * 2 ** (c / 1200) * vf, n, rng.uniform())
    fenv = np.clip(t / (a * 2.5), 0, 1) ** 0.5
    cutoff = f0 * (1.5 + 5.5 * vel * bright * fenv) + 200
    x = tv_lowpass(x / 3, cutoff, q=0.9)
    x = np.tanh(1.6 * x) / np.tanh(1.6)
    return x * adsr(n, a, 0.25, 0.8, 0.45) * vel


FORMANTS_AH = [(800, 1.0, 8), (1150, 0.5, 9), (2900, 0.25, 10), (3900, 0.1, 12)]


def choir(m, dur, vel=1.0, rng=RNG, voices=5, a=0.6, r=1.2):
    n = n_(dur + r)
    t = np.arange(n) / SR
    f0 = float(hz(m))
    out = np.zeros((2, n))
    for v in range(voices):
        vf = 1 + 0.007 * np.sin(2 * np.pi * rng.uniform(4.5, 6.0) * t + rng.uniform(0, 6)) \
            + rng.normal(0, 0.002)
        src = saw(f0 * vf, n, rng.uniform()) + 0.3 * noise(n) * 0.2
        y = sum(g * peak_bp(src, fc, q) for fc, g, q in FORMANTS_AH)
        out += pan2(y, rng.uniform(-0.7, 0.7))
    out /= voices ** 0.5
    return out * adsr(n, a, 0.4, 0.9, r) * vel * 1.4


def bass_string(m, dur, vel=1.0, rng=RNG, a=0.2):
    n = n_(dur + 0.6)
    x = saw(float(hz(m)), n, rng.uniform()) + saw(float(hz(m)) * 1.003, n, rng.uniform())
    x = lp(x, 380, 2)
    return x * adsr(n, a, 0.3, 0.9, 0.6) * vel * 0.8


def spiccato(m, vel=1.0, length=0.14, rng=RNG):
    n = n_(length + 0.15)
    t = np.arange(n) / SR
    f0 = float(hz(m))
    x = saw(f0, n, rng.uniform()) + saw(f0 * 1.004, n, rng.uniform()) + 0.5 * saw(f0 * 2.002, n, rng.uniform())
    cutoff = 400 + 2600 * vel * np.exp(-t / 0.05)
    x = tv_lowpass(x / 2.5, cutoff, q=1.1, block=64)
    return x * adsr(n, 0.004, 0.05, 0.6, 0.12) * vel


# ---------------------------------------------------------------- percussion + sfx

def taiko(vel=1.0, pitch=1.0, dur=1.4):
    n = n_(dur)
    t = np.arange(n) / SR
    f = (48 + 90 * np.exp(-t / 0.03)) * pitch
    body = sine(f, n) * np.exp(-t / 0.38)
    body2 = sine(f * 1.6, n) * np.exp(-t / 0.12) * 0.3
    skin = lp(noise(n), 1400) * np.exp(-t / 0.025) * 0.6
    x = body + body2 + skin
    return np.tanh(1.8 * x * vel) / np.tanh(1.8)


def snare(vel=1.0):
    n = n_(0.9)
    t = np.arange(n) / SR
    x = bp(noise(n), 900, 7000) * np.exp(-t / 0.16) * 0.9 + sine(185 * (1 + 0.3 * np.exp(-t / 0.01)), n) * np.exp(-t / 0.07)
    return x * vel


def hat(vel=1.0, open_=False):
    n = n_(0.4 if open_ else 0.12)
    t = np.arange(n) / SR
    return hp(noise(n), 7000, 3) * np.exp(-t / (0.12 if open_ else 0.025)) * vel


def chug(vel=1.0):
    n = n_(0.25)
    t = np.arange(n) / SR
    return bp(noise(n), 250, 2200) * np.exp(-t / 0.05) * vel


def crash(vel=1.0, dur=4.0):
    n = n_(dur)
    t = np.arange(n) / SR
    metal = np.zeros(n)
    for f in (205.3, 304.4, 369.6, 522.7, 540.0, 800.0):
        metal += np.sign(np.sin(2 * np.pi * f * t))
    x = hp(noise(n) + 0.25 * metal, 3500, 2) * np.exp(-t / 1.5)
    return x * vel * 0.5


def reverse_cymbal(dur):
    x = crash(1.0, dur)[::-1].copy()
    x *= np.linspace(0, 1, len(x)) ** 2
    return x


def sub_boom(vel=1.0, dur=3.0, f_hi=62, f_lo=28):
    n = n_(dur)
    t = np.arange(n) / SR
    f = f_lo + (f_hi - f_lo) * np.exp(-t / 0.4)
    return sine(f, n) * np.exp(-t / 1.1) * adsr(n, 0.004, 0, 1, 0.3) * vel


def crack(vel=1.0):
    n = n_(0.6)
    t = np.arange(n) / SR
    return lp(noise(n), 5000) * np.exp(-t / 0.06) * vel


def braam(root=26, vel=1.0, dur=4.0, rng=RNG):
    n = n_(dur)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for m in (root, root + 12, root + 19, root + 24):
        for c in (-10, 0, 9):
            x += saw(float(hz(m)) * 2 ** (c / 1200), n, rng.uniform())
    cutoff = 120 + 2600 * np.clip(t / 0.12, 0, 1) * np.exp(-np.maximum(t - 0.12, 0) / 0.9)
    x = tv_lowpass(x / 6, cutoff, q=1.4)
    x = np.tanh(3.0 * x)
    return x * adsr(n, 0.015, 0.2, 0.85, 1.5) * np.exp(-t / 2.5) * vel


def riser(dur, rng=RNG):
    n = n_(dur)
    t = np.arange(n) / SR
    u = t / dur
    center = 250 * (40 ** u)
    x = tv_bandpass(noise(n), center, q=3.0) * 3
    gl = sine(float(hz(50)) * 2 ** (2 * u ** 1.3), n) * 0.35 + sine(float(hz(57)) * 2 ** (2 * u ** 1.3), n) * 0.25
    rate = 4 + 16 * u ** 2
    trem = 0.6 + 0.4 * np.sign(np.sin(2 * np.pi * np.cumsum(rate) / SR))
    x = (x + gl) * trem * u ** 2.2
    x[-n_(0.004):] *= np.linspace(1, 0, n_(0.004))
    return x


def thunder(vel=1.0, dur=4.0):
    n = n_(dur)
    t = np.arange(n) / SR
    rumble = lp(brown(n), 180, 2) * np.exp(-t / 1.4)
    crackle = lp(noise(n) * (RNG.random(n) < 0.004) * 6, 2500) * np.exp(-t / 0.4)
    x = rumble * adsr(n, 0.05, 0, 1, 0.8) + crackle
    return x * vel


def ship_horn(dur=3.0):
    n = n_(dur + 1.5)
    t = np.arange(n) / SR
    drift = 1 + 0.002 * np.sin(2 * np.pi * 0.7 * t)
    x = saw(98 * drift, n) + saw(123.5 * drift, n) * 0.8 + saw(97.6 * drift, n)
    x = lp(x, 520, 2)
    return x * adsr(n, 0.35, 0.1, 1.0, 1.4) * 0.5


def train_whistle(dur=1.6):
    n = n_(dur + 0.3)
    t = np.arange(n) / SR
    bend = 1 - 0.03 * np.exp(-t / 0.08)
    x = sum(sine(f * bend, n) * g for f, g in ((622, 1), (740, 0.8), (932, 0.6), (1244, 0.15)))
    x += bp(noise(n), 600, 1400) * 0.25
    return x * adsr(n, 0.08, 0.1, 0.9, 0.35) * 0.35


def clang(vel=1.0):
    n = n_(3.5)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for r, g, tau in ((1, 1, 1.4), (2.76, 0.6, 0.9), (5.40, 0.4, 0.6), (8.93, 0.25, 0.35), (13.34, 0.15, 0.2)):
        x += sine(118 * r, n) * g * np.exp(-t / tau)
    x += lp(noise(n), 3000) * np.exp(-t / 0.04)
    return x * vel * 0.6


def jet(dur, vel=1.0):
    n = n_(dur)
    t = np.arange(n) / SR
    x = lp(pink(n), 700, 2) + 0.2 * hp(noise(n), 4000) * np.sin(np.pi * t / dur) ** 2
    return x * np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 2 * vel


def whoosh(dur=1.6, vel=1.0):
    n = n_(dur)
    t = np.arange(n) / SR
    u = t / dur
    c = 300 + 3000 * np.exp(-((u - 0.55) / 0.18) ** 2)
    x = tv_bandpass(noise(n), c, q=1.2) * np.exp(-((u - 0.55) / 0.25) ** 2) * 2.5
    return x * vel


def wind(dur, vel=1.0, howl=0.0):
    n = n_(dur)
    t = np.arange(n) / SR
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.11 * t) * np.sin(2 * np.pi * 0.037 * t + 1)
    c = 300 + 900 * lfo + howl * 600 * np.sin(2 * np.pi * 0.23 * t) ** 2
    x = tv_bandpass(pink(n), c, q=1.5 + howl * 3, block=256) * (0.4 + 0.6 * lfo)
    return x * vel


def heartbeat(vel=1.0):
    n = n_(0.5)
    t = np.arange(n) / SR
    return sine(55 * (1 + 0.6 * np.exp(-t / 0.02)), n) * np.exp(-t / 0.12) * vel


# ---------------------------------------------------------------- mixer

class Mixer:
    """Buses per 'layer'. Layers are choked (hard-gated) at edit points: the rail
    section is cut dead at the 1956 hush, and everything before the breath is cut
    before the final hit."""

    SENDS = {"lead": (0.32, 0.0), "pad": (0.45, 0.0), "brass": (0.3, 0.05), "low": (0.08, 0.0),
             "perc": (0.12, 0.28), "sfx": (0.3, 0.1)}

    def __init__(self, dur):
        self.n = n_(dur)
        self.layers = {}

    def layer_of(self, t):
        b = CS.bar_at(t)
        if b < CS.hush["start"] - 1e-6:
            return 0
        if b < CS.silences[0]["start"] - 1e-6:
            return 1
        return 2

    def add(self, bus, x, t, gain=1.0, pan=0.0, layer=None):
        if layer is None:
            layer = self.layer_of(t)
        if x.ndim == 1:
            x = pan2(x, pan)
        s = n_(t)
        if s >= self.n:
            return
        e = min(self.n, s + x.shape[1])
        L = self.layers.setdefault(layer, {})
        if bus not in L:
            L[bus] = np.zeros((2, self.n))
        if s < 0:
            x = x[:, -s:]
            s = 0
        L[bus][:, s:e] += x[:, : e - s] * gain


def make_ir(rt_low, rt_mid, rt_high, length, predelay=0.02, seed=7):
    rng = np.random.default_rng(seed)
    n = n_(length)
    t = np.arange(n) / SR
    ir = np.zeros((2, n))
    for ch in range(2):
        w = rng.standard_normal(n)
        lo = lp(w, 400) * np.exp(-6.9 * t / rt_low)
        mid = bp(w, 400, 3500) * np.exp(-6.9 * t / rt_mid)
        hi = hp(w, 3500) * np.exp(-6.9 * t / rt_high)
        x = lo + mid + 0.6 * hi
        x *= np.clip(t / 0.03, 0, 1)  # smooth onset of the diffuse tail
        for k in range(10):  # early reflections
            d = n_(predelay + rng.uniform(0.005, 0.07))
            x[d] += rng.uniform(-0.6, 0.6) * (1 - k / 12)
        pd = n_(predelay)
        ir[ch, pd:] = x[: n - pd]
    ir /= np.sqrt(np.sum(ir ** 2) / 2)
    return ir


def reverb(x, ir):
    return np.stack([signal.fftconvolve(x[c], ir[c])[: x.shape[1]] for c in range(2)])


# ---------------------------------------------------------------- composition

def chord_notes(ch, center=57, span=(50, 72)):
    notes = []
    for pc in PCS[ch]:
        best = min((m for m in range(span[0], span[1]) if m % 12 == pc), key=lambda m: abs(m - center))
        notes.append(best)
    return sorted(notes)


def compose():
    mx = Mixer(CS.duration)
    bar_s = CS.bar_s
    silence0 = CS.t(CS.silences[0]["start"])

    def clip_dur(start, dur):
        """Keep sustained notes from crossing the breath before the final hit."""
        if start < silence0 < start + dur:
            return silence0 - start
        return dur

    # ---- drone: D pedal under everything, swelling with the arc
    for b in range(CS.total_bars):
        if 20 <= b < 22:
            continue
        inten = CS.intensity_bars[b]
        t0 = CS.t(b)
        root = 26 if CS.chords[b] in ("Dm",) or b < 7 else ROOTS[CS.chords[b]] - 12
        g = 0.05 + 0.3 * inten
        n = n_(bar_s + 1.0)
        tt = np.arange(n) / SR
        x = sine(float(hz(root)), n) + 0.5 * sine(float(hz(root + 7)), n) + 0.3 * lp(saw(float(hz(root + 12)), n), 300)
        x *= adsr(n, 0.6 if b else 3.0, 0.0, 1.0, 1.0)
        x *= 1 + 0.08 * np.sin(2 * np.pi * 0.25 * tt)
        mx.add("low", x, t0, g)

    # ---- wind: desert, then storm
    mx.add("sfx", wind(CS.t(7.5)), 0.0, 0.12, 0.0)
    mx.add("sfx", wind(CS.t(5.2), howl=1.0), CS.t(10.8), 0.5, 0.0)
    mx.add("sfx", wind(CS.t(4.5), 0.7), CS.t(20), 0.1, 0.0)

    # ---- motif statements
    for st in CS.motif_statements:
        inst = st["instrument"]
        b0 = st["bar"]
        notes = CS.motif[: st.get("notes", len(CS.motif))]
        g = st["gain"]
        for i, (beat, m, dur) in enumerate(notes):
            m = m + st["transpose"]
            t0 = tb(b0, beat) + RNG.normal(0, 0.006)
            dsec = clip_dur(t0, dur * SPB)
            if CS.in_silence(t0) or dsec <= 0.02:
                continue
            vel = g * (0.85 + 0.15 * RNG.random())
            if inst == "ney":
                mx.add("lead", ney(m, dsec, vel), t0, 0.9, -0.1)
            elif inst == "oud":
                mx.add("lead", oud(m, dsec, vel), t0, 0.8, 0.15)
                if dur >= 1.5:  # tremolo picking on long notes
                    k = 1
                    while k * SPB / 4 < dsec - 0.05:
                        mx.add("lead", oud(m, 0.3, vel * 0.35), t0 + k * SPB / 4, 0.8, 0.15)
                        k += 1
            elif inst == "horn":
                mx.add("brass", brass(m, dsec, vel * 0.7, a=0.12, bright=0.5), t0, 0.7, -0.2)
            elif inst == "celesta":
                mx.add("lead", celesta(m, dsec, vel), t0, 0.9, 0.1)
                mx.add("lead", celesta(m + 12, dsec, vel * 0.25), t0 + 0.004, 0.9, -0.3)
            elif inst == "strings":
                mx.add("lead", supersaw(m, dsec, vel, voices=6, detune=9, cutoff=5200, a=0.12, r=0.5, vib=0.004), t0, 0.5)
                mx.add("lead", supersaw(m - 12, dsec, vel * 0.8, voices=6, detune=9, cutoff=3000, a=0.12, r=0.5, vib=0.004), t0, 0.5)
                mx.add("pad", choir(m, dsec, vel * 0.5, a=0.15, r=0.6), t0, 0.35)
            elif inst == "tutti":
                mx.add("brass", brass(m, dsec, vel, a=0.05), t0, 0.55, -0.25)
                mx.add("brass", brass(m - 12, dsec, vel, a=0.05), t0, 0.5, 0.25)
                mx.add("lead", supersaw(m + 12, dsec, vel, voices=6, detune=10, cutoff=6000, a=0.08, r=0.4, vib=0.005), t0, 0.45)
                mx.add("lead", supersaw(m + 24, dsec, vel * 0.5, voices=5, detune=10, cutoff=7000, a=0.08, r=0.4, vib=0.005), t0, 0.3)
                mx.add("pad", choir(m + 12, dsec, vel * 0.7, a=0.1, r=0.5), t0, 0.5)

    # oud low-string drones on bar downbeats through the Silk Road
    for b in range(2, 7):
        mx.add("lead", oud(38, 2.0, 0.45), tb(b), 0.7, -0.2)
        mx.add("lead", oud(45, 1.0, 0.25), tb(b, 2.5), 0.7, -0.25)

    # ---- string pads and choir, by intensity
    for b in range(4, CS.total_bars):
        if 20 <= b < 22 or b >= 37:
            continue
        ch = CS.chords[b]
        inten = CS.intensity_bars[b]
        t0 = CS.t(b)
        dur = clip_dur(t0, bar_s + 0.05)
        cutoff = 900 + 5000 * inten
        for m in chord_notes(ch, 57):
            mx.add("pad", supersaw(m, dur, 0.22 + 0.4 * inten, cutoff=cutoff, a=0.5 if inten < 0.5 else 0.2, r=1.0, vib=0.003), t0, 0.5)
        if inten > 0.4:
            for m in chord_notes(ch, 69, (62, 84)):
                mx.add("pad", supersaw(m, dur, 0.4 * inten, cutoff=cutoff * 1.2, a=0.3, r=1.0, vib=0.004), t0, 0.35)
        if b >= 7:
            mx.add("low", bass_string(ROOTS[ch], dur, 0.35 + 0.5 * inten), t0, 0.7)
        if inten >= 0.7:
            for m in chord_notes(ch, 64, (57, 76)):
                mx.add("pad", choir(m, dur, 0.55 * inten, a=0.35, r=1.0), t0, 0.5)
        if b >= 29:
            for m in chord_notes(ch, 55, (48, 64)):
                mx.add("brass", brass(m, dur, 0.55 * inten, a=0.15, bright=0.6), t0, 0.35, RNG.uniform(-0.4, 0.4))

    # high string harmonic over the 1956 hush
    mx.add("pad", supersaw(86, CS.t(4) - 0.1, 0.25, voices=4, detune=4, cutoff=7000, a=2.0, r=1.5, vib=0.002), CS.t(20), 0.4)
    for b in (22, 23):
        for m in chord_notes(CS.chords[b], 57):
            mx.add("pad", supersaw(m, bar_s, 0.18 + 0.1 * (b - 22), cutoff=1500, a=1.0, r=0.8), CS.t(b), 0.5)

    # ---- counter-melody over the ports
    for bar, beat, m, d in [(29, 0, 77, 2), (29, 2, 81, 2), (30, 0, 84, 3), (30, 3, 81, 1),
                            (31, 0, 84, 2), (31, 2, 88, 2), (32, 0, 88, 2), (32, 2, 85, 2)]:
        t0 = tb(bar, beat)
        mx.add("lead", supersaw(m, d * SPB, 0.7, voices=6, detune=8, cutoff=6500, a=0.15, r=0.6, vib=0.005), t0, 0.4)
        mx.add("brass", brass(m - 12, d * SPB, 0.6, a=0.1), t0, 0.35)

    # ---- ostinati
    def ostinato(b0, b1, sub, pattern, vel=0.8, octave=0):
        for b in range(b0, b1):
            ch = CS.chords[b]
            r = ROOTS[ch] + 12 + octave
            inten = CS.intensity_bars[b]
            for k in range(4 * sub):
                t0 = tb(b, k / sub)
                if CS.in_silence(t0):
                    continue
                iv = pattern[k % len(pattern)]
                acc = 1.0 if k % (sub * 2) == 0 else 0.7
                mx.add("pad", spiccato(r + iv, vel * acc * (0.6 + 0.4 * inten)), t0, 0.55, RNG.uniform(-0.3, 0.3))

    ostinato(11, 16, 2, [0, 0, 7, 0, 12, 0, 7, 0], 0.7)
    ostinato(16, 20, 4, [0, 0, 0, 12, 0, 0, 7, 0, 0, 12, 0, 0, 7, 0, 12, 0], 0.75)
    ostinato(25, 29, 2, [0, 7, 12, 7, 0, 7, 12, 15], 0.75)
    ostinato(29, 33, 4, [0, 0, 12, 0, 7, 0, 12, 0, 0, 0, 12, 0, 7, 0, 15, 12], 0.8)
    ostinato(33, 37, 4, [0, 12, 0, 12, 7, 12, 0, 12, 0, 12, 0, 12, 7, 12, 15, 12], 0.85)
    ostinato(33, 37, 4, [0, 0, 12, 0], 0.6, octave=12)

    # ---- percussion
    def drums(b0, b1, hits):
        for b in range(b0, b1):
            for beat, kind, v in hits:
                t0 = tb(b, beat)
                if CS.in_silence(t0):
                    continue
                v = v * (0.85 + 0.15 * RNG.random())
                if kind == "T":
                    mx.add("perc", taiko(v), t0, 0.9, RNG.uniform(-0.2, 0.2))
                elif kind == "t":
                    mx.add("perc", taiko(v * 0.6, pitch=1.6, dur=0.6), t0, 0.6, RNG.uniform(-0.5, 0.5))
                elif kind == "S":
                    mx.add("perc", snare(v), t0, 0.45, 0.05)
                elif kind == "h":
                    mx.add("perc", hat(v), t0, 0.18, 0.35)
                elif kind == "c":
                    mx.add("perc", chug(v), t0, 0.35, -0.1)

    drums(11, 16, [(0, "T", 0.8), (2.5, "T", 0.6), (3, "t", 0.5), (3.5, "t", 0.6)])
    drums(16, 20, [(0, "T", 1.0), (1.5, "T", 0.7), (2, "T", 0.9), (3.5, "t", 0.6), (1, "S", 0.6), (3, "S", 0.7)]
          + [(k / 4, "h", 0.5 + 0.3 * (k % 2 == 0)) for k in range(16)]
          + [(k / 2, "c", 0.8 if k % 2 == 0 else 0.5) for k in range(8)])
    drums(25, 29, [(0, "T", 0.9), (2, "T", 0.8), (2.75, "t", 0.5), (3.5, "t", 0.7)]
          + [(k / 2 + 0.5, "h", 0.4) for k in range(0, 8, 2)])
    drums(29, 33, [(0, "T", 1.0), (0.75, "T", 0.6), (1.5, "T", 0.8), (2, "T", 0.9), (3, "T", 0.8), (3.5, "t", 0.8),
                   (1, "S", 0.6), (3, "S", 0.7)] + [(k / 4, "h", 0.45) for k in range(16)])
    drums(33, 37, [(k / 2, "T", 1.0 if k % 2 == 0 else 0.75) for k in range(8)]
          + [(1, "S", 0.8), (3, "S", 0.9), (3.75, "t", 0.8)] + [(k / 4, "h", 0.5) for k in range(16)])
    # snare roll into the breath
    k = 0
    while True:
        t0 = tb(36) + k * SPB / 8
        if t0 >= silence0:
            break
        mx.add("perc", snare(0.2 + 0.8 * (k / 45)), t0, 0.35, 0.0)
        k += 1

    # heartbeat under the lifting box
    for b in (22, 23):
        for beat in range(4):
            v = 0.4 + 0.5 * ((b - 22) * 4 + beat) / 8
            mx.add("low", heartbeat(v), tb(b, beat), 1.0)
            mx.add("low", heartbeat(v * 0.7), tb(b, beat + 0.28), 1.0)

    # ---- risers and reverse cymbals
    for r in CS.risers:
        t0, t1 = CS.t(r["start"]), CS.t(r["end"])
        mx.add("sfx", riser(t1 - t0), t0, 0.45)
        mx.add("perc", reverse_cymbal(t1 - t0), t0, 0.5)

    # ---- impacts
    for im in CS.impacts:
        t0 = CS.t(im["at"])
        s = im["strength"]
        mx.add("low", sub_boom(min(1.0, s)), t0, 0.9)
        mx.add("perc", taiko(min(1.0, 0.5 + s * 0.5), pitch=0.85), t0, 0.8)
        mx.add("perc", crack(s * 0.6), t0, 0.35)
        if s >= 0.5:
            mx.add("perc", crash(s * 0.8), t0, 0.35, 0.2)
        if s >= 0.85:
            mx.add("brass", braam(26, min(1.2, s * 0.8)), t0, 0.5)
        if im.get("lightning"):
            mx.add("sfx", thunder(s * 1.2), t0 + 0.05, 0.7, RNG.uniform(-0.5, 0.5))
        if im.get("clang"):
            mx.add("sfx", clang(1.0), t0, 0.7, 0.1)

    # ---- the final hit and coda (layer 2)
    tF = CS.t(37)
    for p in (0.8, 1.0, 1.3):
        mx.add("perc", taiko(1.0, pitch=p, dur=2.5), tF, 0.6, RNG.uniform(-0.4, 0.4))
    mx.add("brass", braam(26, 1.3, dur=6.0), tF, 0.65)
    mx.add("low", sub_boom(1.0, dur=6.0, f_hi=70, f_lo=24), tF, 1.0)
    mx.add("perc", crash(1.2, 7.0), tF, 0.5)
    for m in (50, 57, 62, 65, 69, 74):
        mx.add("pad", choir(m, 5.0, 0.6, a=0.02, r=3.5), tF, 0.45)
        mx.add("pad", supersaw(m + 12, 5.0, 0.5, cutoff=5000, a=0.02, r=3.5, vib=0.003), tF, 0.35)
    # the caravan's motif, alone again
    for beat, m, d in [(0, 62, 1.5), (1.5, 64, 0.5), (2, 65, 1.0), (3, 62, 4.0)]:
        mx.add("lead", oud(m, d * SPB, 0.55), tb(38, beat), 0.8, 0.1, layer=2)
    mx.add("pad", supersaw(86, CS.t(2.5), 0.2, voices=4, detune=4, cutoff=6000, a=1.0, r=2.5), tb(37.5), 0.35, layer=2)

    # ---- sound design
    mx.add("sfx", ship_horn(3.0), tb(20, 1.0), 0.22, -0.3)
    mx.add("sfx", ship_horn(2.0), tb(22, 3.0), 0.15, -0.3)
    mx.add("sfx", train_whistle(1.8), tb(16, 0.25), 0.5, 0.3)
    mx.add("sfx", jet(CS.t(4)), CS.t(25), 0.25)
    mx.add("sfx", whoosh(1.8, 1.0), tb(26, 3.0), 0.5, -0.6)
    for b in (35, 36):
        mx.add("sfx", hp(pink(n_(CS.bar_s)), 180) * 0.08 * sine(np.full(n_(CS.bar_s), 36.0), n_(CS.bar_s)) ** 2,
               CS.t(b), 0.6, 0.0)
    return mx


def master(mx):
    n = mx.n
    hall = make_ir(2.6, 3.4, 1.6, 4.5, 0.03, seed=1)
    room = make_ir(0.7, 1.1, 0.5, 1.6, 0.01, seed=2)
    gates = {0: CS.t(CS.hush["start"]), 1: CS.t(CS.silences[0]["start"]), 2: None}
    pre = np.zeros((2, n))
    post = np.zeros((2, n))
    for layer, buses in mx.layers.items():
        dry = np.zeros((2, n))
        hsend = np.zeros((2, n))
        rsend = np.zeros((2, n))
        for bus, x in buses.items():
            h, r = Mixer.SENDS[bus]
            dry += x
            hsend += x * h
            rsend += x * r
        wet = reverb(hsend, hall) * 0.9 + reverb(rsend, room) * 0.7
        y = dry + wet
        g_end = gates[layer]
        if g_end is not None:
            e = n_(g_end)
            fade = n_(0.012)
            y[:, e - fade:e] *= np.linspace(1, 0, fade)
            y[:, e:] = 0
        if layer == 2:
            post += y
        else:
            pre += y

    # glue compression on the body of the cue (feed-forward, RMS detector);
    # the final hit bypasses it so it stays the loudest moment of the film
    det = np.sqrt(signal.lfilter([0.002], [1, -0.998], (pre ** 2).mean(axis=0)) + 1e-12)
    thr = np.percentile(det, 99.5) * 0.6
    over = np.maximum(det / thr, 1.0)
    gain = signal.lfilter([0.01], [1, -0.99], over ** (1 / 2.0 - 1))
    pre *= gain
    total = pre + post * 1.15
    total = np.stack([hp(total[c], 28, 2) for c in range(2)])
    total /= np.max(np.abs(total)) + 1e-9
    total = np.tanh(1.6 * total) / np.tanh(1.6)
    total *= 10 ** (-1.0 / 20) / (np.max(np.abs(total)) + 1e-9)

    fo = n_(1.5)
    total[:, -fo:] *= np.linspace(1, 0, fo) ** 2
    return total


def main(out=None):
    out = out or os.path.join(ROOT, "build", "score.wav")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    mx = compose()
    y = master(mx)
    wavfile.write(out, SR, y.T.astype(np.float32))
    print(f"wrote {out}  {y.shape[1] / SR:.2f}s  peak {np.max(np.abs(y)):.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
