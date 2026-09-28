"""Frame renderer: real footage cut on the cue sheet grid, one warm film grade, the
closing lines, logo reveal and end card (endcard.py). One pass per format."""
import json
import subprocess
from functools import lru_cache

import cv2
import numpy as np

from launchfilm import endcard
from launchfilm import text as T
from launchfilm.config import FORMATS, PALETTE, Cues, load_sources, src_path
from launchfilm.logo import logo_masks


def rgbf(name):
    return np.array(PALETTE[name], np.float32) / 255


@lru_cache(maxsize=128)
def probe(path):
    """Display size (square pixels), whether the stored pixels are anamorphic, and fps."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,sample_aspect_ratio,r_frame_rate", "-of", "json", str(path)],
                       capture_output=True, text=True)
    s = json.loads(r.stdout)["streams"][0]
    sar = s.get("sample_aspect_ratio", "1:1")
    num, den = (int(v) for v in sar.split(":")) if sar[:1].isdigit() and not sar.startswith("0:") else (1, 1)
    a, b = (float(v) for v in s.get("r_frame_rate", "24/1").split("/"))
    return int(round(s["width"] * num / den)) // 2 * 2, s["height"], num != den, a / b


def shot_speed(shot, fps, sources):
    """Playback speed. Unless a shot sets `speed`, every source frame is shown exactly once
    (or every k-th frame): 30 fps plays as 0.8x, 25 fps as 0.96x, 60 fps as 0.8x. No
    dropped or doubled frames, so pans never judder."""
    if "speed" in shot:
        return float(shot["speed"])
    src_fps = probe(src_path(shot["src"], sources))[3]
    k = max(1, round(src_fps * shot.get("speed_target", 1.0) / fps))
    return k * fps / src_fps


def crop_box(sw, sh, active, aspect, cx, cy, zoom):
    """Pixel crop (x, y, w, h) of the given aspect inside the active area."""
    ax0, ay0, ax1, ay1 = active
    aw, ah = (ax1 - ax0) * sw, (ay1 - ay0) * sh
    if aw / ah > aspect:
        h = ah
        w = h * aspect
    else:
        w = aw
        h = w / aspect
    w, h = w / zoom, h / zoom
    x = ax0 * sw + cx * aw - w / 2
    y = ay0 * sh + cy * ah - h / 2
    x = min(max(x, ax0 * sw), ax1 * sw - w)
    y = min(max(y, ay0 * sh), ay1 * sh - h)
    w2, h2 = int(w) // 2 * 2, int(h) // 2 * 2
    return int(x), int(y), w2, h2


class ShotReader:
    """Streams the frames of one shot, cropped and scaled, conformed to the film fps.

    `blur`: [{"box": [x0, y0, x1, y1], "to": [...], "t": [t0, t1]}] in source fractions and
    source seconds; `to` moves the box linearly over `t` (for markings on moving vehicles)."""

    def __init__(self, shot, fmt, W, H, fps, nframes, sources):
        self.W, self.H, self.n, self.fps = W, H, nframes, fps
        s = sources[shot["src"]]
        path = src_path(shot["src"], sources)
        sw, sh, anamorphic, _ = probe(path)
        cx, cy, zoom = shot.get("c16" if fmt == "16x9" else "c9", [0.5, 0.5, 1.0])
        x, y, w, h = crop_box(sw, sh, s.get("active", [0, 0, 1, 1]), W / H, cx, cy, zoom)
        self.crop = (x, y, w, h, sw, sh)
        self.push = shot.get("push", 0.0)
        self.pan = shot.get("pan", [0.0, 0.0])
        m = 1 + abs(self.push) + 2 * max(abs(self.pan[0]), abs(self.pan[1]))
        self.DW = int(round(W * m)) // 2 * 2
        self.DH = int(round(H * m)) // 2 * 2
        self.speed = shot_speed(shot, fps, sources)
        self.t_in = shot["in"]
        t_in = shot["in"] - s.get("segment", [0])[0]
        dur = nframes / fps * self.speed + 0.5
        self.blur = shot.get("blur", [])
        square = f"scale={sw}:{sh},setsar=1," if anamorphic else ""
        vf = (f"{square}crop={w}:{h}:{x}:{y},scale={self.DW}:{self.DH}:flags=lanczos,"
              f"setpts=(PTS-STARTPTS)/{self.speed},fps={fps}")
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-ss", f"{t_in:.3f}", "-i", str(path), "-t", f"{dur:.3f}",
             "-vf", vf, "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.last = None
        self.i = 0

    def _blur(self, img):
        x, y, w, h, sw, sh = self.crop
        ts = self.t_in + self.i / self.fps * self.speed
        out = img
        for b in self.blur:
            t0, t1 = b.get("t", [-1e9, 1e9])
            if not (t0 <= ts <= t1):
                continue
            box = np.array(b["box"], np.float32)
            if "to" in b and t1 > t0:
                box = box + (np.array(b["to"], np.float32) - box) * (ts - t0) / (t1 - t0)
            u0 = int((box[0] * sw - x) / w * self.DW)
            v0 = int((box[1] * sh - y) / h * self.DH)
            u1 = int((box[2] * sw - x) / w * self.DW)
            v1 = int((box[3] * sh - y) / h * self.DH)
            pad = 12
            u0, v0 = max(0, u0 - pad), max(0, v0 - pad)
            u1, v1 = min(self.DW, u1 + pad), min(self.DH, v1 + pad)
            if u1 <= u0 or v1 <= v0:
                continue
            if out is img:
                out = img.copy()
            reg = out[v0:v1, u0:u1].astype(np.float32)
            sig = max(6.0, 0.18 * min(u1 - u0, v1 - v0))
            bl = cv2.GaussianBlur(cv2.GaussianBlur(reg, (0, 0), sig), (0, 0), sig)
            mh, mw = reg.shape[:2]
            yy = np.minimum(np.arange(mh), np.arange(mh)[::-1])[:, None]
            xx = np.minimum(np.arange(mw), np.arange(mw)[::-1])[None, :]
            a = np.clip(np.minimum(yy, xx) / pad, 0, 1)[..., None]
            out[v0:v1, u0:u1] = (reg * (1 - a) + bl * a).astype(np.uint8)
        return out

    def next(self):
        size = self.DW * self.DH * 3
        buf = self.proc.stdout.read(size)
        if len(buf) == size:
            self.last = np.frombuffer(buf, np.uint8).reshape(self.DH, self.DW, 3)
        elif self.last is None:
            self.last = np.zeros((self.DH, self.DW, 3), np.uint8)
        img = self._blur(self.last) if self.blur else self.last
        if self.push or any(self.pan):
            # push > 0 eases in toward 1 + push; push < 0 starts tight and pulls out;
            # pan [x, y] drifts the window by up to that fraction of the frame
            p = self.i / max(1, self.n - 1)
            pm = max(abs(self.pan[0]), abs(self.pan[1]))
            if self.push >= 0:
                zf = (1 + 2 * pm) * (1 + self.push * (1 - (1 - p) ** 2))
            else:
                zf = (1 + 2 * pm) * (1 + abs(self.push) * (1 - p * p * (3 - 2 * p)))
            cw, ch = self.DW / zf, self.DH / zf
            ox = (self.DW - cw) / 2 + self.pan[0] * (2 * p - 1) * self.DW / (1 + 2 * pm)
            oy = (self.DH - ch) / 2 + self.pan[1] * (2 * p - 1) * self.DH / (1 + 2 * pm)
            M = np.float32([[self.W / cw, 0, -ox * self.W / cw], [0, self.H / ch, -oy * self.H / ch]])
            img = cv2.warpAffine(img, M, (self.W, self.H), flags=cv2.INTER_LINEAR)
        elif img.shape[1] != self.W:
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        self.i += 1
        return img.astype(np.float32) / 255

    def close(self):
        self.proc.stdout.close()
        self.proc.kill()
        self.proc.wait()


# ------------------------------------------------------------------ look
LUMA = np.float32([0.2126, 0.7152, 0.0722])


class Look:
    """One warm film grade for every shot, after a per-shot normalization so sources from
    different cameras (DVIDS, NASA, stock) sit together."""

    def __init__(self, W, H, seed=7):
        self.W, self.H = W, H
        self.rng = np.random.default_rng(seed)
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2) / np.sqrt(2)
        self.vig = (1 - 0.36 * np.clip(r, 0, 1) ** 2.4)[..., None]
        self.gw, self.gh = W // 2, H // 2

    def grain(self, amount):
        g = self.rng.standard_normal((self.gh, self.gw)).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 0.7)
        g = cv2.resize(g, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return g[..., None] * amount

    @staticmethod
    def normalize(img, shot):
        """Per-shot gains from the shot's first frame: partial gray-world white balance and a
        gentle exposure match (mid-grey toward the film's level). `exp` (stops) and `wb`
        (rgb gains) in the shot override or add to it; `norm: false` turns it off."""
        if shot.get("norm", True) is False:
            g = np.ones(3, np.float32)
        else:
            l = img @ LUMA
            mid = (l > 0.12) & (l < 0.88)
            m = img[mid].mean(0) if mid.sum() > 500 else img.reshape(-1, 3).mean(0)
            wb = np.clip((m.mean() / np.maximum(m, 1e-3)) ** 0.45, 0.9, 1.1)
            p50 = float(np.median(l))
            ex = np.clip((shot.get("mid", 0.4) / max(p50, 0.02)) ** 0.5, 0.8, 1.3)
            g = (wb * ex).astype(np.float32)
        g = g * np.float32(2 ** shot.get("exp", 0.0)) * np.float32(shot.get("wb", [1, 1, 1]))
        return g

    def film(self, img, gains, fi, sat=0.84):
        x = img * gains
        l = (x @ LUMA)[..., None]
        x = l + (x - l) * sat
        # soft shoulder: highlights roll off instead of clipping
        x = np.where(x > 0.7, 0.7 + 0.3 * np.tanh((x - 0.7) / 0.3), x)
        x = np.clip(x, 0, 1)
        # gentle S-curve around mid grey
        x = x + 0.22 * x * (1 - x) * (x - 0.42)
        l = (x @ LUMA)[..., None]
        # warm grade: amber highlights, slightly cool shadows
        x = x * np.float32([1.035, 1.0, 0.93]) + (1 - l) ** 3 * np.float32([-0.012, 0.0, 0.01]) \
            + l ** 2 * np.float32([0.03, 0.012, -0.02])
        # halation: warm bloom off the brightest areas
        hi = np.clip((l[..., 0] - 0.72) / 0.28, 0, 1)
        small = cv2.resize(hi, (self.W // 8, self.H // 8), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 3.0)
        h = cv2.resize(small, (self.W, self.H), interpolation=cv2.INTER_LINEAR)[..., None]
        x = x + h * np.float32([0.07, 0.03, 0.0])
        # printed blacks
        x = 0.018 + x * 0.972
        return np.clip(x, 0, 1) * self.vig + self.grain(0.024)

    def apply(self, img, gains, fi):
        return self.film(img, gains, fi)


# ------------------------------------------------------------------ overlays
def draw_titles(img, cs, fmt, t, over_footage):
    H, W = img.shape[:2]
    for ti in cs["titles"]:
        t0, t1 = cs.when(ti), cs.when(ti, "end")
        if not (t0 <= t < t1):
            continue
        fi, fo = ti.get("fade_in_s", 0.35), ti.get("fade_out_s", 0.25)
        a = min(1.0, (t - t0) / fi if fi else 1.0, (t1 - t) / fo if fo else 1.0)
        a = a * a * (3 - 2 * a)
        rgba = T.render_text(ti["text"], ti["style"], fmt, W)
        cy = H * ti.get("y", 0.5)
        if over_footage:
            T.scrim(img, cy, rgba.shape[0] * 3.0, 0.32 * a, W / 2, rgba.shape[1] * 1.6)
        T.composite(img, rgba, W / 2, cy, a, drop_shadow=over_footage)
    return img


def compose_logo(img, cs, fmt, t, cx, cy, width, t_hit, alpha=1.0):
    fps = cs.fps
    rv = cs["reveal"]
    L, S, (sx0, sy0, sx1, sy1) = logo_masks(width)
    k = (t - t_hit) * fps
    if k < 0:
        return img
    lh, lw = L.shape
    x0, y0 = int(cx - lw / 2), int(cy - lh / 2)
    # gold square lands first: pops from 1.6x to 1x over `square_frames`
    sq_p = min(1.0, (k + 1) / rv["square_frames"])
    sc = 1 + 0.6 * (1 - sq_p) ** 2
    scx, scy = (sx0 + sx1) / 2, (sy0 + sy1) / 2
    M = np.float32([[sc, 0, scx * (1 - sc)], [0, sc, scy * (1 - sc)]])
    Sm = cv2.warpAffine(S, M, (lw, lh), flags=cv2.INTER_LINEAR)
    # letters resolve outward from the square
    lp = np.clip((k - rv["letters_delay_frames"]) / rv["letters_frames"], 0, 1)
    lp = 1 - (1 - lp) ** 3
    if lp > 0:
        xs = np.arange(lw, dtype=np.float32)
        reach = lp * lw * 1.15
        dist = np.abs(xs - scx)
        wipe = np.clip((reach - dist) / 40.0, 0, 1)[None, :]
        La = L * wipe * lp ** 0.5
    else:
        La = L * 0
    reg = img[y0:y0 + lh, x0:x0 + lw]
    cream, gold = rgbf("cream"), rgbf("gold")
    a = (La * alpha)[..., None]
    reg[:] = reg * (1 - a) + cream * a
    if k < 10:  # brief glow on the landing
        g = cv2.GaussianBlur(Sm, (0, 0), 14) * (1 - k / 10) * 1.2
        reg[:] = np.clip(reg + gold * g[..., None] * alpha, 0, 1)
    a = (Sm * alpha)[..., None]
    reg[:] = reg * (1 - a) + gold * a
    return img


# ------------------------------------------------------------------ main loop
def shot_table(cs):
    shots = cs["shots"]
    out = []
    for i, s in enumerate(shots):
        f0 = int(round(cs.when(s) * cs.fps))
        f1 = int(round(cs.when(shots[i + 1]) * cs.fps)) if i + 1 < len(shots) else cs.nframes
        out.append((f0, f1, s))
    return out


def encoder_cmd(cs, W, H, out_path, audio):
    enc = cs.raw.get("encode", {})
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(cs.fps), "-i", "-"]
    if audio:
        cmd += ["-i", str(audio)]
    cmd += ["-map", "0:v"] + (["-map", "1:a"] if audio else [])
    cmd += ["-c:v", "libx264", "-profile:v", "high", "-preset", "slow", "-crf", str(enc.get("crf", 17)),
            "-maxrate", enc.get("maxrate", "10M"), "-bufsize", enc.get("bufsize", "20M"),
            "-pix_fmt", "yuv420p", "-g", "48", "-tune", "grain",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-ac", "2"]
    cmd += ["-movflags", "+faststart", "-metadata", "title=Volume", "-shortest", str(out_path)]
    return cmd


def render(fmt, out_path, audio=None, only_frames=None, still_dir=None):
    cs = Cues()
    sources = load_sources()
    W, H = FORMATS[fmt]["w"], FORMATS[fmt]["h"]
    look = Look(W, H)
    t_hit = cs.when(cs["reveal"])
    enc = None
    if only_frames is None:
        enc = subprocess.Popen(encoder_cmd(cs, W, H, out_path, audio), stdin=subprocess.PIPE)
    wanted = set(only_frames or [])
    for f0, f1, shot in shot_table(cs):
        if only_frames is not None and not any(f0 <= f < f1 for f in wanted):
            continue
        src = shot["src"]
        reader = None
        n = f1 - f0
        if src not in ("black",):
            reader = ShotReader(shot, fmt, W, H, cs.fps, n, sources)
        gains = None
        for fi in range(f0, f1):
            t = fi / cs.fps
            if reader is not None:
                img = reader.next()
                if gains is None:
                    gains = look.normalize(img, shot)
            if only_frames is not None and fi not in wanted:
                continue
            if src == "black":
                img = np.zeros((H, W, 3), np.float32)
            else:
                img = look.apply(img, gains, fi)
                if shot.get("closing"):
                    img = endcard.compose(img, cs, fmt, t, t_hit)
            img = np.clip(img, 0, 1).astype(np.float32)
            img = draw_titles(img, cs, fmt, t, reader is not None and not shot.get("closing"))
            out = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
            if enc is not None:
                enc.stdin.write(out.tobytes())
            if still_dir is not None and fi in wanted:
                cv2.imwrite(str(still_dir / f"{fmt}_{fi:05d}.png"), out[..., ::-1])
        if reader is not None:
            reader.close()
    if enc is not None:
        enc.stdin.close()
        enc.wait()
        if enc.returncode:
            raise RuntimeError("encoder failed")
