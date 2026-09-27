"""Tempo map / cue sheet shared by the score and the picture edit."""
import json
import math
import os
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUE_PATH = os.path.join(ROOT, "cuesheet.json")


class CueSheet:
    def __init__(self, path=CUE_PATH):
        with open(path) as f:
            self.raw = json.load(f)
        r = self.raw
        self.bpm = r["bpm"]
        self.bpb = r["beats_per_bar"]
        self.spb = 60.0 / self.bpm
        self.bar_s = self.spb * self.bpb
        self.total_bars = r["total_bars"]
        self.duration = self.total_bars * self.bar_s
        self.fps = r["fps"]
        self.W, self.H = r["width"], r["height"]
        self.sr = r["sample_rate"]
        self.shots = r["shots"]
        self.impacts = r["impacts"]
        self.risers = r["risers"]
        self.silences = r["silences"]
        self.titles = r["titles"]
        self.chords = r["chords"]
        self.intensity_bars = r["intensity"]
        self.motif = r["motif"]["notes"]
        self.motif_statements = r["motif_statements"]
        self.hush = r["hush"]

    def t(self, bar):
        return bar * self.bar_s

    def bar_at(self, t):
        return t / self.bar_s

    def shot_at(self, t):
        b = self.bar_at(t)
        for s in self.shots:
            if s["start"] <= b < s["end"]:
                return s
        return self.shots[-1]

    def intensity(self, t):
        b = min(max(self.bar_at(t), 0.0), self.total_bars - 1e-6)
        i = int(b)
        f = b - i
        a = self.intensity_bars[i]
        n = self.intensity_bars[min(i + 1, len(self.intensity_bars) - 1)]
        # hold within a bar, glide in the last beat
        g = max(0.0, (f - 0.75) / 0.25)
        return a + (n - a) * g * g

    def impact_env(self, t, decay=0.35):
        """Sum of exponentially decaying pulses at each impact (visual flash / shake)."""
        v = 0.0
        for im in self.impacts:
            dt = t - self.t(im["at"])
            if 0 <= dt < 4 * decay * (2 if im.get("final") else 1):
                v += im["strength"] * math.exp(-dt / decay)
        return v

    def lightning_env(self, t):
        v = 0.0
        for im in self.impacts:
            if not im.get("lightning"):
                continue
            dt = t - self.t(im["at"])
            if 0 <= dt < 0.6:
                # double-strobe flicker
                v += im["strength"] * (math.exp(-dt / 0.05) + 0.7 * math.exp(-max(0, dt - 0.11) / 0.08) * (dt > 0.11))
        return v

    def beat_pulse(self, t, decay=0.12):
        beat = t / self.spb
        ph = beat - math.floor(beat)
        return math.exp(-ph * self.spb / decay)

    def in_silence(self, t):
        b = self.bar_at(t)
        return any(s["start"] <= b < s["end"] for s in self.silences)

    def riser_env(self, t):
        b = self.bar_at(t)
        for r in self.risers:
            if r["start"] <= b < r["end"]:
                return (b - r["start"]) / (r["end"] - r["start"])
        return 0.0


@dataclass
class FrameInfo:
    t: float
    bar: float
    beat: float
    shot: dict
    local: float      # seconds since shot start
    u: float          # 0..1 progress through shot
    shot_len: float
    intensity: float
    impact: float
    lightning: float
    pulse: float
    riser: float
    silent: bool


def frame_info(cs, t):
    shot = cs.shot_at(t)
    t0, t1 = cs.t(shot["start"]), cs.t(shot["end"])
    return FrameInfo(
        t=t, bar=cs.bar_at(t), beat=t / cs.spb, shot=shot,
        local=t - t0, u=(t - t0) / (t1 - t0), shot_len=t1 - t0,
        intensity=cs.intensity(t), impact=cs.impact_env(t),
        lightning=cs.lightning_env(t), pulse=cs.beat_pulse(t),
        riser=cs.riser_env(t), silent=cs.in_silence(t),
    )
