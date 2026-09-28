"""Frame renderer: real footage cut on the cue sheet grid, light grade + grain,
kinetic type, voice captions, logo reveal and end card. One pass per format."""
import json
import subprocess
from functools import lru_cache

import cv2
import numpy as np

from launchfilm import text as T
from launchfilm.config import FORMATS, PALETTE, Cues, load_sources, src_path
from launchfilm.logo import logo_masks


def rgbf(name):
    return np.array(PALETTE[name], np.float32) / 255


@lru_cache(maxsize=64)
def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height", "-of", "json", str(path)], capture_output=True, text=True)
    s = json.loads(r.stdout)["streams"][0]
    return s["width"], s["height"]


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
    """Streams the frames of one shot, cropped and scaled, conformed to the film fps."""

    def __init__(self, shot, fmt, W, H, fps, nframes, sources):
        self.W, self.H, self.n = W, H, nframes
        s = sources[shot["src"]]
        path = src_path(shot["src"], sources)
        sw, sh = probe(path)
        cx, cy, zoom = shot.get("c16" if fmt == "16x9" else "c9", [0.5, 0.5, 1.0])
        x, y, w, h = crop_box(sw, sh, s.get("active", [0, 0, 1, 1]), W / H, cx, cy, zoom)
        self.push = shot.get("push", 0.0)
        self.DW = int(round(W * (1 + self.push))) // 2 * 2
        self.DH = int(round(H * (1 + self.push))) // 2 * 2
        speed = shot.get("speed", 1.0)
        t_in = shot["in"] - s.get("segment", [0])[0]
        dur = nframes / fps * speed + 0.5
        vf = (f"crop={w}:{h}:{x}:{y},scale={self.DW}:{self.DH}:flags=lanczos,"
              f"setpts=(PTS-STARTPTS)/{speed},fps={fps}")
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-ss", f"{t_in:.3f}", "-i", str(path), "-t", f"{dur:.3f}",
             "-vf", vf, "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.last = None
        self.i = 0

    def next(self):
        size = self.DW * self.DH * 3
        buf = self.proc.stdout.read(size)
        if len(buf) == size:
            self.last = np.frombuffer(buf, np.uint8).reshape(self.DH, self.DW, 3)
        elif self.last is None:
            self.last = np.zeros((self.DH, self.DW, 3), np.uint8)
        img = self.last
        if self.push:
            p = self.i / max(1, self.n - 1)
            z = 1 + self.push * (1 - (1 - p) ** 2)
            cw, ch = self.DW / z, self.DH / z
            M = np.float32([[self.W / cw, 0, -(self.DW - cw) / 2 * self.W / cw],
                            [0, self.H / ch, -(self.DH - ch) / 2 * self.H / ch]])
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
class Look:
    def __init__(self, W, H, seed=7):
        self.W, self.H = W, H
        self.rng = np.random.default_rng(seed)
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2) / np.sqrt(2)
        self.vig = (1 - 0.32 * np.clip(r, 0, 1) ** 2.2)[..., None]
        self.gw, self.gh = W // 2, H // 2

    def grain(self, amount):
        g = self.rng.standard_normal((self.gh, self.gw)).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 0.6)
        g = cv2.resize(g, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return g[..., None] * amount

    def apply(self, img, look, fi):
        if look == "bw":
            l = img @ np.float32([0.299, 0.587, 0.114])
            l = np.clip((l - 0.06) / 0.88, 0, 1)
            l = l * l * (3 - 2 * l) * 0.35 + l * 0.65
            l *= 1 + 0.03 * np.sin(fi * 2.1) + 0.02 * self.rng.standard_normal()
            img = l[..., None] * np.float32([1.0, 0.955, 0.87]) + np.float32([0.012, 0.01, 0.006])
            img = img * self.vig + self.grain(0.045)
        elif look in ("desat", "full", "night", "launch"):
            sat = {"desat": 0.42, "full": 1.12, "night": 1.05, "launch": 1.1}[look]
            con = {"desat": 1.08, "full": 1.12, "night": 1.18, "launch": 1.1}[look]
            l = (img @ np.float32([0.299, 0.587, 0.114]))[..., None]
            img = l + (img - l) * sat
            img = (img - 0.5) * con + 0.5
            if look == "desat":
                img = img * np.float32([0.97, 1.0, 1.03]) + np.float32([0.0, 0.004, 0.012])
            elif look in ("full", "launch"):
                sh = np.clip(1 - l * 1.6, 0, 1)
                hi = np.clip(l * 1.4 - 0.4, 0, 1)
                img = img + sh * np.float32([-0.012, 0.004, 0.02]) + hi * np.float32([0.03, 0.012, -0.018])
            elif look == "night":
                img = np.clip(img - 0.02, 0, 1) ** 1.08
            img = np.clip(img, 0, 1) * self.vig + self.grain(0.022)
        return img


def flash_amount(k):
    return [0.95, 0.55, 0.25, 0.1][k] if k < 4 else 0.0


def glitch(img, k, rng):
    if k > 1:
        return img
    s = 18 if k == 0 else 8
    out = img.copy()
    out[..., 0] = np.roll(img[..., 0], s, axis=1)
    out[..., 2] = np.roll(img[..., 2], -s, axis=1)
    H = img.shape[0]
    for _ in range(6):
        y0 = int(rng.integers(0, H - 40))
        h = int(rng.integers(8, 60))
        out[y0:y0 + h] = np.roll(out[y0:y0 + h], int(rng.integers(-60, 60)), axis=1)
    return out


# ------------------------------------------------------------------ overlays
def _words_visible(t, t0, step_s, nwords):
    return min(nwords, int((t - t0) / step_s + 1e-6) + 1) if t >= t0 else 0


def draw_titles(img, cs, fmt, t, over_footage):
    H, W = img.shape[:2]
    for ti in cs["titles"]:
        t0, t1 = cs.when(ti), cs.when(ti, "end")
        if not (t0 <= t < t1):
            continue
        words = ti["text"].split(" ")
        nv = _words_visible(t, t0, ti.get("step_beats", 1) * cs.beat_s, len(words))
        rgba = T.render_text(ti["text"], ti["style"], fmt, W, nv)
        # dy stacks two-line pairs: -1 above center, +1 below
        cy = H / 2 + ti.get("dy", 0) * rgba.shape[0] * 0.55
        if over_footage:
            T.scrim(img, cy, rgba.shape[0] * 2.6, 0.4, W / 2, rgba.shape[1] * 1.5)
        T.composite(img, rgba, W / 2, cy, 1.0, drop_shadow=over_footage)
    return img


def draw_captions(img, cs, fmt, t):
    H, W = img.shape[:2]
    for v in cs["voices"]:
        base = v["at_s"] - v["in"]
        for (a, b, txt) in v["captions"]:
            ta, tb = base + a, base + b
            if not (ta - 0.03 <= t < tb + 0.12):
                continue
            rgba = T.render_text(txt, "caption", fmt, W)
            cy = H - 150 if fmt == "16x9" else 1390
            T.scrim(img, cy + 12, 160, 0.5, W / 2, rgba.shape[1] * 1.4 + 120)
            T.composite(img, rgba, W / 2, cy, 1.0)
            lab = T.render_text(v["label"], "label", fmt, W, color="gold_bright")
            T.composite(img, lab, W / 2, cy + rgba.shape[0] / 2 + 26, 0.9)
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


def draw_endcard(img, cs, fmt, t):
    H, W = img.shape[:2]
    ec = cs["endcard"]
    t0 = cs.when(ec)
    k = (t - t0) * cs.fps
    a = lambda d: float(np.clip((k - d) / 8.0, 0, 1))
    v = fmt == "9x16"
    lw = 420 if not v else 520
    L, S, _ = logo_masks(lw)
    lh = L.shape[0]
    cy_logo = H * (0.34 if not v else 0.38)
    x0, y0 = int(W / 2 - lw / 2), int(cy_logo - lh / 2)
    reg = img[y0:y0 + lh, x0:x0 + lw]
    for m, col in ((L, rgbf("cream")), (S, rgbf("gold"))):
        al = (m * a(0))[..., None]
        reg[:] = reg * (1 - al) + col * al
    rows = [(ec["tagline"], "caption", "cream", 0.52 if not v else 0.52, 4),
            (ec["services"], "caption_small", "gold_bright", 0.61 if not v else 0.585, 8),
            (ec["since"], "label_big", "gold", 0.675 if not v else 0.63, 12),
            (ec["url"], "url", "cream", 0.80 if not v else 0.72, 16)]
    for txt, style, col, fy, d in rows:
        st = {"caption_small": "caption", "url": "caption"}.get(style, style)
        rgba = T.render_text(txt, st, fmt, W, color=col)
        if style == "caption_small":
            rgba = cv2.resize(rgba, None, fx=0.72, fy=0.72, interpolation=cv2.INTER_AREA)
        if style == "url":
            rgba = cv2.resize(rgba, None, fx=0.9, fy=0.9, interpolation=cv2.INTER_AREA)
        T.composite(img, rgba, W / 2, H * fy, a(d), drop_shadow=False)
    return img


# ------------------------------------------------------------------ main loop
def shot_table(cs):
    shots = cs["shots"]
    out = []
    for i, s in enumerate(shots):
        f0 = cs.frame(s["at"])
        f1 = cs.frame(shots[i + 1]["at"]) if i + 1 < len(shots) else cs.nframes
        out.append((f0, f1, s))
    return out


def render(fmt, out_path, audio=None, only_frames=None, still_dir=None):
    cs = Cues()
    sources = load_sources()
    W, H = FORMATS[fmt]["w"], FORMATS[fmt]["h"]
    look = Look(W, H)
    rng = np.random.default_rng(3)
    charcoal = rgbf("charcoal")
    t_hit = cs.when(cs["reveal"])
    t_end = cs.when(cs["endcard"])
    enc = None
    if only_frames is None:
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(cs.fps), "-i", "-"]
        if audio:
            cmd += ["-i", str(audio)]
        cmd += ["-map", "0:v"] + (["-map", "1:a"] if audio else [])
        cmd += ["-c:v", "libx264", "-profile:v", "high", "-preset", "slow", "-crf", "17",
                "-maxrate", "16M", "-bufsize", "24M", "-pix_fmt", "yuv420p", "-g", "48",
                "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
        if audio:
            cmd += ["-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-ac", "2"]
        cmd += ["-movflags", "+faststart", "-metadata", "title=Volume", "-shortest", str(out_path)]
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    wanted = set(only_frames or [])
    for f0, f1, shot in shot_table(cs):
        if only_frames is not None and not any(f0 <= f < f1 for f in wanted):
            continue
        src = shot["src"]
        reader = None
        n = f1 - f0
        if src == "brand_bg":
            reader = ShotReader({"src": shot["bg"], "in": shot["in"], "look": "desat"}, fmt, W, H,
                                cs.fps, n, sources)
        elif src not in ("black", "charcoal"):
            reader = ShotReader(shot, fmt, W, H, cs.fps, n, sources)
        for fi in range(f0, f1):
            t = fi / cs.fps
            k = fi - f0
            if reader is not None:
                img = reader.next()
            if only_frames is not None and fi not in wanted:
                continue
            if src == "black":
                img = np.zeros((H, W, 3), np.float32)
            elif src == "charcoal":
                img = np.empty((H, W, 3), np.float32)
                img[:] = charcoal
            elif src == "brand_bg":
                l = (img @ np.float32([0.299, 0.587, 0.114]))[..., None]
                img = charcoal + (l * 0.9 + img * 0.1 - charcoal) * shot["opacity"]
                img = img.astype(np.float32)
            else:
                img = look.apply(img, shot.get("look", "full"), fi)
                if shot.get("glitch"):
                    img = glitch(img, k, rng)
                if shot.get("flash"):
                    fa = flash_amount(k)
                    img = img + (1 - img) * fa
            footage = reader is not None and src != "brand_bg"
            img = np.clip(img, 0, 1).astype(np.float32)
            img = draw_titles(img, cs, fmt, t, footage)
            if footage:
                img = draw_captions(img, cs, fmt, t)
            if t_hit <= t < t_end:
                cy = H / 2 if fmt == "16x9" else H * 0.47
                wl = cs["reveal"]["logo_w_16x9" if fmt == "16x9" else "logo_w_9x16"]
                img = compose_logo(img, cs, fmt, t, W / 2, cy, wl, t_hit)
            elif t >= t_end:
                img = draw_endcard(img, cs, fmt, t)
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
