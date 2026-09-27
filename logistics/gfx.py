"""Drawing helpers shared by all scenes (cairo for vectors, numpy for particles)."""
import math

import cairo
import numpy as np

W, H = 1920, 1080
LB = 138  # 2.39:1 letterbox bar height
CX, CY = W / 2, H / 2

_CACHE = {}
_KEEP = []


def cached(key, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def rng(seed):
    return np.random.default_rng(seed)


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_io(x):
    x = clamp(x)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def ease_out(x, p=3):
    return 1 - (1 - clamp(x)) ** p


def ease_in(x, p=3):
    return clamp(x) ** p


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t):
    return tuple(a + (b - a) * t for a, b in zip(c1, c2))


def hash1(i, seed=0):
    x = math.sin(i * 127.1 + seed * 311.7) * 43758.5453
    return x - math.floor(x)


def vnoise(x, seed=0):
    """Smooth 1-D value noise, scalar or array, range ~[-1, 1]."""
    x = np.asarray(x, dtype=np.float64)
    i = np.floor(x)
    f = x - i
    def h(k):
        v = np.sin(k * 127.1 + seed * 311.7) * 43758.5453
        return (v - np.floor(v)) * 2 - 1
    u = f * f * (3 - 2 * f)
    return h(i) * (1 - u) + h(i + 1) * u


def fbm(x, seed=0, octaves=4):
    v = 0.0
    a = 0.5
    for o in range(octaves):
        v = v + a * vnoise(x * (2 ** o), seed + o * 13)
        a *= 0.5
    return v


def noise2d_texture(w, h, seed, octaves=6, base=4):
    """Tileable fbm value-noise texture in [0,1] (float32), generated in code."""
    r = rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    import cv2
    for o in range(octaves):
        gw, gh = base * 2 ** o, max(2, int(base * 2 ** o * h / w))
        g = r.random((gh, gw)).astype(np.float32)
        g = np.concatenate([g, g[:1]], 0)
        g = np.concatenate([g, g[:, :1]], 1)
        up = cv2.resize(g, (w + w // gw, h + h // gh), interpolation=cv2.INTER_CUBIC)[:h, :w]
        out += amp * up
        tot += amp
        amp *= 0.5
    out /= tot
    ranks = np.empty(out.size, np.float32)
    ranks[np.argsort(out, axis=None)] = np.linspace(0, 1, out.size, dtype=np.float32)
    return ranks.reshape(out.shape)


# ---------------------------------------------------------------- cairo helpers

def new_surface(w=W, h=H):
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    c = cairo.Context(s)
    fo = cairo.FontOptions()
    fo.set_antialias(cairo.ANTIALIAS_GRAY)
    fo.set_hint_style(cairo.HINT_STYLE_NONE)
    c.set_font_options(fo)
    return s, c


def vgrad(ctx, y0, y1, stops, x0=0, x1=W):
    g = cairo.LinearGradient(0, y0, 0, y1)
    for off, col in stops:
        if len(col) == 3:
            g.add_color_stop_rgb(off, *col)
        else:
            g.add_color_stop_rgba(off, *col)
    ctx.rectangle(x0, y0, x1 - x0, y1 - y0)
    ctx.set_source(g)
    ctx.fill()


def radial(ctx, x, y, r, col, a=1.0, inner=0.0, op=None):
    g = cairo.RadialGradient(x, y, r * inner, x, y, r)
    g.add_color_stop_rgba(0, *col, a)
    g.add_color_stop_rgba(0.35, *col, a * 0.35)
    g.add_color_stop_rgba(1, *col, 0)
    if op is not None:
        ctx.save()
        ctx.set_operator(op)
    ctx.set_source(g)
    ctx.arc(x, y, r, 0, 2 * math.pi)
    ctx.fill()
    if op is not None:
        ctx.restore()


def poly(ctx, pts, close=True):
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    if close:
        ctx.close_path()


def fill_under(ctx, xs, ys, bottom, col, a=1.0):
    ctx.move_to(xs[0], bottom)
    for x, y in zip(xs, ys):
        ctx.line_to(x, y)
    ctx.line_to(xs[-1], bottom)
    ctx.close_path()
    ctx.set_source_rgba(*col, a)
    ctx.fill()


def src(ctx, col, a=1.0):
    ctx.set_source_rgba(col[0], col[1], col[2], a)


def ellipse(ctx, x, y, rx, ry, ang=0.0):
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(ang)
    ctx.scale(max(rx, 1e-3), max(ry, 1e-3))
    ctx.arc(0, 0, 1, 0, 2 * math.pi)
    ctx.restore()


def surface_from_array(arr):
    """float RGB(A) [0,1] HxWx{3,4} -> cairo ImageSurface (premultiplied BGRA)."""
    h, w = arr.shape[:2]
    out = np.empty((h, w, 4), np.uint8)
    if arr.shape[2] == 3:
        a = np.ones((h, w), np.float32)
        rgb = arr
    else:
        a = arr[..., 3]
        rgb = arr[..., :3] * a[..., None]
    out[..., 0] = np.clip(rgb[..., 2] * 255, 0, 255)
    out[..., 1] = np.clip(rgb[..., 1] * 255, 0, 255)
    out[..., 2] = np.clip(rgb[..., 0] * 255, 0, 255)
    out[..., 3] = np.clip(a * 255, 0, 255)
    s = cairo.ImageSurface.create_for_data(memoryview(out), cairo.FORMAT_ARGB32, w, h, w * 4)
    _KEEP.append(out)  # cairo does not own the pixel buffer
    return s


# ---------------------------------------------------------------- camera

class Cam:
    """2.5D camera: parallax by depth, zoom about screen centre, roll, shake."""

    def __init__(self, x=0.0, y=0.0, zoom=1.0, roll=0.0, sx=0.0, sy=0.0):
        self.x, self.y, self.zoom, self.roll, self.sx, self.sy = x, y, zoom, roll, sx, sy

    def push(self, ctx, depth=1.0):
        ctx.save()
        ctx.translate(CX + self.sx / depth, CY + self.sy / depth)
        ctx.rotate(self.roll)
        s = self.zoom ** (1.0 / depth)
        ctx.scale(s, s)
        ctx.translate(-CX - self.x / depth, -CY - self.y / depth)

    def pop(self, ctx):
        ctx.restore()

    def project(self, x, y, depth=1.0):
        """Same transform as push() for numpy particle arrays."""
        s = self.zoom ** (1.0 / depth)
        x = (np.asarray(x) - CX - self.x / depth) * s
        y = (np.asarray(y) - CY - self.y / depth) * s
        c, sn = math.cos(self.roll), math.sin(self.roll)
        return CX + self.sx / depth + x * c - y * sn, CY + self.sy / depth + x * sn + y * c

    def scale(self, depth=1.0):
        return self.zoom ** (1.0 / depth)


def shake(t, amount, seed=3):
    if amount <= 1e-4:
        return 0.0, 0.0
    return (float(vnoise(t * 23.0, seed)) * amount, float(vnoise(t * 21.0, seed + 5)) * amount)


# ---------------------------------------------------------------- emissive buffer

class Glow:
    """Half-resolution additive light buffer: particles, lamps, sparks, city lights."""

    def __init__(self):
        self.h, self.w = H // 2, W // 2
        self.fine = np.zeros((self.h * self.w, 3), np.float32)
        self.soft = np.zeros((self.h * self.w, 3), np.float32)
        self.streak = None

    def _splat(self, buf, x, y, col, inten):
        x = np.asarray(x, np.float64) * 0.5
        y = np.asarray(y, np.float64) * 0.5
        ix = x.astype(np.int64)
        iy = y.astype(np.int64)
        m = (ix >= 0) & (ix < self.w) & (iy >= 0) & (iy < self.h)
        if not np.any(m):
            return
        idx = iy[m] * self.w + ix[m]
        inten = np.broadcast_to(np.asarray(inten, np.float64), x.shape)[m]
        col = np.asarray(col, np.float64)
        n = self.h * self.w
        if col.ndim == 1:
            v = np.bincount(idx, weights=inten, minlength=n).astype(np.float32)
            buf += v[:, None] * col[None, :].astype(np.float32)
        else:
            col = col[m]
            for c in range(3):
                buf[:, c] += np.bincount(idx, weights=inten * col[:, c], minlength=n).astype(np.float32)

    def points(self, x, y, col, inten=1.0, soft=False):
        self._splat(self.soft if soft else self.fine, x, y, col, inten)

    def streaks(self, x, y, col, inten, angle, length):
        """Motion-blurred particles (rain, sparks): splat then convolve with a line kernel."""
        if self.streak is None:
            self.streak = []
        buf = np.zeros((self.h * self.w, 3), np.float32)
        self._splat(buf, x, y, col, inten)
        self.streak.append((buf, angle, length))

    def resolve(self):
        import cv2
        fine = self.fine.reshape(self.h, self.w, 3)
        soft = self.soft.reshape(self.h, self.w, 3)
        out = cv2.GaussianBlur(fine, (0, 0), 0.9) * 2.2 + cv2.GaussianBlur(fine, (0, 0), 5) * 0.9
        if soft.any():
            out += cv2.GaussianBlur(soft, (0, 0), 7) * 1.6 + cv2.GaussianBlur(soft, (0, 0), 20) * 0.9
        if self.streak:
            for buf, ang, L in self.streak:
                b = buf.reshape(self.h, self.w, 3)
                k = int(max(3, L / 2)) | 1
                ker = np.zeros((k, k), np.float32)
                c = k // 2
                for i in range(k):
                    d = (i - c)
                    x = int(round(c + d * math.cos(ang)))
                    y = int(round(c + d * math.sin(ang)))
                    ker[y, x] = 1.0
                ker /= ker.sum()
                out += cv2.filter2D(b, -1, ker) * 1.5
        return out
