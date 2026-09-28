"""Paths, fonts and the cue sheet (single tempo map for picture and sound)."""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "launchfilm"
CACHE = ROOT / "cache"
SRC_DIR = CACHE / "sources"
WORK = CACHE / "work"
OUT = ROOT / "out"

FONT_DIRS = [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts",
             Path.home() / ".local/share/fonts", ROOT / "fonts", SRC_DIR]


def find_font(names):
    """First matching .ttf/.otf file name (case-insensitive) in the usual font dirs."""
    wanted = [n.lower() for n in names]
    found = {}
    for d in FONT_DIRS:
        if d.exists():
            for p in d.rglob("*"):
                if p.suffix.lower() in (".ttf", ".otf"):
                    found.setdefault(p.name.lower(), p)
    for n in wanted:
        if n in found:
            return str(found[n])
    raise FileNotFoundError(f"none of {names} found; install Inter (apt install fonts-inter) "
                            f"or drop the TTFs into {ROOT / 'fonts'}")


FONTS = {
    "sans": lambda: find_font(["Inter-Regular.ttf", "Inter-Regular.otf", "Inter.ttf"]),
    "sans_medium": lambda: find_font(["Inter-Medium.ttf", "Inter-Medium.otf", "Inter-Regular.ttf"]),
    "sans_semibold": lambda: find_font(["Inter-SemiBold.ttf", "Inter-SemiBold.otf", "Inter-Medium.ttf"]),
    "mono": lambda: find_font(["JetBrainsMono-Regular.ttf", "DejaVuSansMono.ttf"]),
    "display": lambda: find_font(["font_manrope.ttf", "Manrope[wght].ttf", "Manrope-VariableFont_wght.ttf"]),
}

PALETTE = {
    "charcoal": (0x18, 0x1E, 0x1F),
    "charcoal2": (0x24, 0x28, 0x27),
    "gold": (0xBA, 0xA8, 0x8A),
    "gold_bright": (0xE3, 0xD7, 0xBB),
    "cream": (0xEE, 0xE9, 0xDE),
    "paper": (0xF3, 0xF3, 0xEF),
    "white": (0xFF, 0xFF, 0xFF),
}

FORMATS = {
    "16x9": {"w": 1920, "h": 1080, "name": "volume_launch_16x9_1080p.mp4"},
    "9x16": {"w": 1080, "h": 1920, "name": "volume_launch_9x16_1080x1920.mp4"},
}


class Cues:
    """cuesheet.json with helpers. Film positions are bars (float) unless a key ends in _s."""

    def __init__(self, path=PKG / "cuesheet.json"):
        self.raw = json.loads(Path(path).read_text())
        r = self.raw
        self.bpm = r["bpm"]
        self.bar_s = r["beats_per_bar"] * 60.0 / self.bpm
        self.beat_s = 60.0 / self.bpm
        self.fps = r["fps"]
        self.bars = r["total_bars"]
        self.duration = self.bars * self.bar_s
        self.nframes = int(round(self.duration * self.fps))
        self.sr = r["sample_rate"]

    def t(self, bar):
        return bar * self.bar_s

    def when(self, item, key="at"):
        """Seconds for an item that has either `at` (bars) or `at_s` (seconds)."""
        if key + "_s" in item:
            return float(item[key + "_s"])
        return self.t(item[key])

    def frame(self, bar):
        return int(round(self.t(bar) * self.fps))

    def __getitem__(self, k):
        return self.raw[k]


def load_sources(path=PKG / "sources.yaml"):
    return yaml.safe_load(Path(path).read_text())["sources"]


def src_path(sid, sources=None):
    sources = sources or load_sources()
    s = sources[sid]
    ext = s.get("ext") or Path(s["url"].split("?")[0]).suffix or ".bin"
    return SRC_DIR / f"{sid}{ext}"
