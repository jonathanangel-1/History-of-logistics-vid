"""Air freight, the global port, the globe, autonomy, and the end card."""
import math

import cairo
import cv2
import numpy as np

from logistics.gfx import (W, H, LB, CX, CY, Cam, cached, rng, vgrad, radial, poly, fill_under, src,
                           ellipse, mix, smooth, ease_io, ease_out, ease_in, clamp, vnoise, fbm, shake,
                           lerp, noise2d_texture)
from logistics.scenes.common import world_span, stars, camel_outline_points

ADD = cairo.OPERATOR_ADD


def paint_array(ctx, rgb, a, x, y):
    """Paint a float RGB + alpha array at (x, y); the buffer lives for the duration of the call."""
    h, w = a.shape
    buf = np.empty((h, w, 4), np.uint8)
    pa = np.clip(a, 0, 1)
    buf[..., 0] = np.clip(rgb[..., 2] * pa * 255, 0, 255)
    buf[..., 1] = np.clip(rgb[..., 1] * pa * 255, 0, 255)
    buf[..., 2] = np.clip(rgb[..., 0] * pa * 255, 0, 255)
    buf[..., 3] = pa * 255
    surf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_ARGB32, w, h, w * 4)
    ctx.set_source_surface(surf, x, y)
    ctx.paint()
    ctx.set_source_rgb(0, 0, 0)
    surf.finish()
    del buf


# ------------------------------------------------------------------ air freight

def _cloud_tex():
    a = noise2d_texture(512, 512, 303, octaves=7, base=4)
    b = noise2d_texture(512, 512, 304, octaves=5, base=8)
    return (0.7 * a + 0.3 * b).astype(np.float32)


def cloud_sea(ctx, hor, fwd, side, sun_dir, hot, cool, fogc, roll=0.0):
    T = cached("cloudsea", _cloud_tex)
    y0 = int(max(LB, hor + 2))
    ys = np.arange(y0, H - LB + 40, dtype=np.float32)
    xs = np.arange(-60, W + 60, 2, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    z = 90.0 / np.maximum(Y - hor, 0.5)
    u = (X - CX) * z / 700.0 + side
    v = z + fwd
    sc = 60.0
    mx = (u * sc) % 512
    my = (v * sc) % 512
    d = cv2.remap(T, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    d2 = cv2.remap(T, ((mx + sun_dir[0] * 6) % 512).astype(np.float32), ((my + sun_dir[1] * 6) % 512).astype(np.float32),
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    lit = np.clip(0.5 + (d - d2) * 6.0, 0, 1)
    dens = np.clip((d - 0.25) / 0.5, 0, 1)
    shade = (0.15 + 0.85 * lit ** 1.4) * (0.3 + 0.7 * dens)
    col = np.array(cool, np.float32) + (np.array(hot, np.float32) - np.array(cool, np.float32)) * shade[..., None]
    fog = np.clip(1 - np.exp(-z / 11.0), 0, 1)[..., None] ** 1.5
    col = col * (1 - fog) + np.array(fogc, np.float32) * fog
    rgb = cv2.resize(col, (len(xs) * 2, len(ys)), interpolation=cv2.INTER_LINEAR)
    a = np.ones(rgb.shape[:2], np.float32)
    ctx.save()
    if roll:
        ctx.translate(CX, CY)
        ctx.rotate(roll)
        ctx.translate(-CX, -CY)
    paint_array(ctx, rgb, a, -60, y0)
    ctx.restore()


def air(ctx, glow, F):
    t, lt, u, bar = F.t, F.local, F.u, F.bar
    sxk, syk = shake(t, 12 * F.impact)
    if bar < 27:
        return _air_side(ctx, glow, F, sxk, syk)
    return _air_rear(ctx, glow, F, sxk, syk)


def _sky_golden(ctx, hor):
    vgrad(ctx, LB - 100, hor + 4, [(0, (0.03, 0.07, 0.2)), (0.55, (0.22, 0.26, 0.42)), (0.85, (0.9, 0.6, 0.4)),
                                   (1, (1.0, 0.82, 0.55))], -300, W + 300)


def _air_side(ctx, glow, F, sxk, syk):
    t, lt = F.t, F.local
    k = lt / 6.0
    roll = -0.05 + 0.04 * math.sin(t * 0.4)
    hor = 575 + 20 * math.sin(t * 0.3)
    ctx.save()
    ctx.translate(sxk, syk)
    ctx.translate(CX, CY)
    ctx.rotate(roll)
    ctx.translate(-CX, -CY)
    _sky_golden(ctx, hor)
    sun = (330.0, hor - 110)
    radial(ctx, *sun, 1100, (1.0, 0.65, 0.35), 0.5, op=ADD)
    radial(ctx, *sun, 260, (1.0, 0.85, 0.6), 0.9, op=ADD)
    ctx.set_source_rgb(1, 0.97, 0.9)
    ctx.arc(*sun, 44, 0, 2 * math.pi)
    ctx.fill()
    ctx.restore()
    ctx.save()
    ctx.translate(sxk, syk)
    cloud_sea(ctx, hor, fwd=0.0, side=-lt * 1.8, sun_dir=(-0.8, 0.3), hot=(1.0, 0.8, 0.58),
              cool=(0.16, 0.15, 0.28), fogc=(0.95, 0.72, 0.52), roll=roll)
    ctx.restore()
    ssx, ssy = sun[0] + sxk, sun[1] + syk
    glow.points([ssx], [ssy], (1.0, 0.8, 0.55), 500, soft=True)

    # the freighter, tracked by the camera: slow drift, gentle bank
    px = 1060 - 120 * ease_io(k) + 12 * math.sin(t * 0.7)
    py = 470 + 18 * math.sin(t * 0.5)
    bank = roll * 0.6 + 0.03 * math.sin(t * 0.6)
    ctx.save()
    ctx.translate(sxk, syk)
    jumbo_side(ctx, glow, px, py, 0.8 + 0.1 * ease_io(k), bank, t, F)
    ctx.restore()
    return {"bloom": 0.1}


def jumbo_side(ctx, glow, x, y, s, ang, t, F):
    """747F in profile, flying toward -x, lit from the low sun on the left."""
    c, sn = math.cos(ang), math.sin(ang)

    def P(px, py):
        return x + (px * c - py * sn) * s, y + (px * sn + py * c) * s

    # contrails (behind everything)
    ctx.save()
    ctx.set_operator(ADD)
    for (ex, ey, w) in ((60, 70, 10), (-60, 95, 14)):
        sx0, sy0 = P(ex + 60, ey)
        g = cairo.LinearGradient(sx0, sy0, sx0 + 1400, sy0)
        g.add_color_stop_rgba(0, 1, 0.95, 0.9, 0.0)
        g.add_color_stop_rgba(0.05, 1, 0.95, 0.9, 0.5)
        g.add_color_stop_rgba(1, 1, 0.95, 0.9, 0.0)
        ctx.set_source(g)
        ctx.move_to(sx0, sy0 - w * 0.3)
        for i in range(1, 30):
            xx = sx0 + i * 50
            ctx.line_to(xx, sy0 - w * (0.3 + i * 0.1) + 8 * math.sin(i * 0.7 + t * 2))
        for i in range(29, 0, -1):
            xx = sx0 + i * 50
            ctx.line_to(xx, sy0 + w * (0.3 + i * 0.1) + 8 * math.sin(i * 0.7 + t * 2 + 1))
        ctx.close_path()
        ctx.fill()
    ctx.restore()
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(ang)
    ctx.scale(s, s)
    body = cairo.LinearGradient(0, -60, 0, 60)
    body.add_color_stop_rgb(0, 0.98, 0.9, 0.8)
    body.add_color_stop_rgb(0.45, 0.78, 0.74, 0.74)
    body.add_color_stop_rgb(1, 0.3, 0.3, 0.38)
    shadow = (0.28, 0.27, 0.34)
    # far wing + far engines
    poly(ctx, [(-40, -10), (120, -20), (330, -120), (300, -118), (60, -25)])
    src(ctx, (0.35, 0.33, 0.4))
    ctx.fill()
    # tail
    poly(ctx, [(420, -30), (560, -260), (620, -262), (600, -40)])
    ctx.set_source(body)
    ctx.fill()
    poly(ctx, [(480, -10), (650, -60), (680, -58), (600, 10)])
    src(ctx, shadow)
    ctx.fill()
    # fuselage with the upper-deck hump
    ctx.move_to(-560, 10)
    ctx.curve_to(-560, -30, -520, -58, -440, -64)
    ctx.curve_to(-380, -110, -250, -112, -160, -80)
    ctx.line_to(300, -56)
    ctx.curve_to(450, -52, 560, -40, 640, -18)
    ctx.line_to(640, 0)
    ctx.curve_to(520, 30, 400, 44, 300, 50)
    ctx.line_to(-440, 56)
    ctx.curve_to(-520, 52, -560, 40, -560, 10)
    ctx.close_path()
    ctx.set_source(body)
    ctx.fill()
    ctx.set_line_width(4)
    ctx.set_source_rgba(0.1, 0.3, 0.6, 0.9)
    ctx.move_to(-520, 20)
    ctx.line_to(620, -6)
    ctx.stroke()
    for k in range(5):
        ctx.rectangle(-420 + k * 24, -86 + k * 2, 12, 9)
        ctx.set_source_rgba(0.1, 0.12, 0.18, 0.9)
        ctx.fill()
    ctx.move_to(-548, -12)
    ctx.line_to(-520, -30)
    ctx.line_to(-495, -28)
    ctx.line_to(-510, -8)
    ctx.close_path()
    src(ctx, (0.08, 0.1, 0.14))
    ctx.fill()
    # near wing, swept toward the viewer
    poly(ctx, [(-120, 30), (60, 26), (360, 190), (320, 196), (-20, 60)])
    g = cairo.LinearGradient(-120, 30, 360, 190)
    g.add_color_stop_rgb(0, 0.85, 0.8, 0.75)
    g.add_color_stop_rgb(1, 0.42, 0.4, 0.46)
    ctx.set_source(g)
    ctx.fill()
    for (ex, ey, r) in ((40, 80, 26), (200, 150, 24)):
        ellipse(ctx, ex, ey, 62, r)
        eg = cairo.LinearGradient(0, ey - r, 0, ey + r)
        eg.add_color_stop_rgb(0, 0.9, 0.85, 0.8)
        eg.add_color_stop_rgb(1, 0.25, 0.25, 0.3)
        ctx.set_source(eg)
        ctx.fill()
        ellipse(ctx, ex - 58, ey, 8, r * 0.85)
        src(ctx, (0.05, 0.05, 0.07))
        ctx.fill()
    ctx.restore()
    # nav lights on the beat
    pulse = F.pulse
    for (lx, ly, col, amp) in ((330, 196, (0.2, 1.0, 0.3), 1.0), (-160, -100, (1.0, 0.15, 0.1), 1.0),
                               (600, -262, (1, 1, 1), 1.0)):
        qx, qy = P(lx, ly)
        glow.points([qx], [qy], col, 40 * amp * pulse + 2, soft=True)
        glow.points([qx], [qy], col, 8 * pulse + 1)


def _air_rear(ctx, glow, F, sxk, syk):
    t = F.t
    lt = (F.bar - 27) * 3.0
    k = lt / 6.0
    hor = 600
    ctx.save()
    ctx.translate(sxk, syk)
    _sky_golden(ctx, hor)
    sun = (960.0, 470.0 + 20 * k)
    ctx.save()
    ctx.set_operator(ADD)
    for i in range(24):
        a0 = i * math.pi * 2 / 24 + t * 0.03
        w = 0.03 + 0.02 * math.sin(i * 1.7)
        poly(ctx, [sun, (sun[0] + 2400 * math.cos(a0 - w), sun[1] + 2400 * math.sin(a0 - w)),
                   (sun[0] + 2400 * math.cos(a0 + w), sun[1] + 2400 * math.sin(a0 + w))])
        g = cairo.RadialGradient(*sun, 0, *sun, 1200)
        g.add_color_stop_rgba(0, 1, 0.85, 0.6, 0.03)
        g.add_color_stop_rgba(1, 1, 0.85, 0.6, 0)
        ctx.set_source(g)
        ctx.fill()
    ctx.restore()
    radial(ctx, *sun, 1200, (1.0, 0.62, 0.32), 0.35, op=ADD)
    radial(ctx, *sun, 300, (1.0, 0.88, 0.62), 0.8, op=ADD)
    ctx.set_source_rgb(1, 0.97, 0.9)
    ctx.arc(*sun, 60, 0, 2 * math.pi)
    ctx.fill()
    cloud_sea(ctx, hor, fwd=lt * 0.9, side=0.0, sun_dir=(0.0, -1.0), hot=(1.0, 0.82, 0.58),
              cool=(0.14, 0.12, 0.24), fogc=(0.95, 0.72, 0.5))
    # plane from behind, flying into the sun
    s = 1.6 * (1 - 0.7 * ease_out(k, 2))
    px, py = 960 + 40 * math.sin(t * 0.5), 520 - 60 * k
    bank = 0.08 * math.sin(t * 0.6)
    ctx.save()
    ctx.translate(px, py)
    ctx.rotate(bank)
    ctx.scale(s, s)
    sil = (0.08, 0.06, 0.07)
    src(ctx, sil)
    poly(ctx, [(-360, 30), (-20, -4), (20, -4), (360, 30), (360, 40), (20, 14), (-20, 14), (-360, 40)])
    ctx.fill()
    poly(ctx, [(-120, -40), (-10, -52), (10, -52), (120, -40), (120, -34), (-120, -34)])
    ctx.fill()
    poly(ctx, [(-6, -10), (-4, -150), (4, -150), (6, -10)])
    ctx.fill()
    ellipse(ctx, 0, 0, 34, 36)
    ctx.fill()
    for ex in (-240, -120, 120, 240):
        ellipse(ctx, ex, 36 + abs(ex) * 0.02, 17, 19)
        ctx.fill()
    ctx.restore()
    # contrails streaming toward camera
    ctx.save()
    ctx.set_operator(ADD)
    for ex in (-240, -120, 120, 240):
        x0 = px + ex * s
        y0 = py + (40 + abs(ex) * 0.02) * s
        x1 = CX + ex * 7
        y1 = H + 200
        g = cairo.LinearGradient(x0, y0, x1, y1)
        g.add_color_stop_rgba(0, 1, 0.92, 0.8, 0.0)
        g.add_color_stop_rgba(0.1, 1, 0.92, 0.8, 0.22)
        g.add_color_stop_rgba(1, 1, 0.92, 0.8, 0.02)
        ctx.set_source(g)
        wn = 5 * s
        ctx.move_to(x0 - wn, y0)
        ctx.line_to(x1 - 70, y1)
        ctx.line_to(x1 + 70, y1)
        ctx.line_to(x0 + wn, y0)
        ctx.close_path()
        ctx.fill()
    ctx.restore()
    ctx.restore()
    glow.points([sun[0] + sxk], [sun[1] + syk], (1.0, 0.8, 0.55), 600, soft=True)
    # lens flare ghosts on the sun-centre axis
    for i, (f_, r, a) in enumerate(((0.4, 60, 0.12), (0.7, 26, 0.18), (1.2, 110, 0.07), (1.5, 40, 0.12))):
        gx = CX + (CX - sun[0]) * f_ + 120 * f_
        gy = CY + (CY - sun[1]) * f_ + 160 * f_
        radial(ctx, gx, gy, r, (0.6, 0.8, 1.0) if i % 2 else (1.0, 0.7, 0.4), a, inner=0.7, op=ADD)
    pulse = F.pulse
    for ex in (-360, 360):
        qx = px + ex * s
        glow.points([qx], [py + 35 * s], (1, 1, 1), 60 * pulse + 2, soft=True)
    return {"bloom": -0.15}


# ------------------------------------------------------------------ the global port at night

BOX_COLORS = [(0.25, 0.45, 0.7), (0.65, 0.15, 0.1), (0.85, 0.45, 0.1), (0.2, 0.45, 0.3), (0.6, 0.6, 0.62),
              (0.85, 0.85, 0.8), (0.8, 0.65, 0.12), (0.15, 0.5, 0.55), (0.45, 0.2, 0.35)]


class Iso:
    def __init__(self, cx, cy, ang, S, tilt=0.55, zk=0.85):
        self.cx, self.cy, self.c, self.s, self.S, self.tilt, self.zk = cx, cy, math.cos(ang), math.sin(ang), S, tilt, zk

    def __call__(self, X, Y, Z=0.0):
        X = np.asarray(X) - self.cx
        Y = np.asarray(Y) - self.cy
        sx = CX + (X * self.c - Y * self.s) * self.S
        sy = CY + (X * self.s + Y * self.c) * self.S * self.tilt - np.asarray(Z) * self.S * self.zk
        return sx, sy


def _yard():
    r = rng(808)
    stacks = []
    for band in range(9):
        y0 = 80 + band * 34
        for bay in range(-34, 34):
            if r.random() < 0.08:
                continue
            for row in range(5):
                h = int(r.choice([0, 2, 3, 3, 4, 4, 4]))
                cols = [BOX_COLORS[int(r.integers(len(BOX_COLORS)))] for _ in range(h)]
                stacks.append((bay * 13.0, y0 + row * 5.6, h, cols))
    ship = []
    for bay in range(-30, 30):
        for row in range(8):
            h = int(r.integers(2, 6))
            cols = [BOX_COLORS[int(r.integers(len(BOX_COLORS)))] for _ in range(h)]
            ship.append((bay * 13.0, -62 + row * 5.6, h, cols))
    return stacks, ship


def draw_stack(ctx, iso, X, Y, h, cols, light, base_z=0.0, L=12.0, D=5.2, Hh=5.0):
    if h == 0:
        return
    for k in range(h):
        z0 = base_z + k * Hh
        col = tuple(c * light for c in cols[k])
        # front face (+Y)
        a = iso(X, Y + D, z0)
        b = iso(X + L, Y + D, z0)
        c_ = iso(X + L, Y + D, z0 + Hh)
        d = iso(X, Y + D, z0 + Hh)
        poly(ctx, [a, b, c_, d])
        src(ctx, tuple(v * 0.7 for v in col))
        ctx.fill()
        # side face (+X)
        e = iso(X + L, Y, z0)
        f = iso(X + L, Y, z0 + Hh)
        poly(ctx, [b, e, f, c_])
        src(ctx, tuple(v * 0.45 for v in col))
        ctx.fill()
    zt = base_z + h * Hh
    top = [iso(X, Y, zt), iso(X + L, Y, zt), iso(X + L, Y + D, zt), iso(X, Y + D, zt)]
    poly(ctx, top)
    src(ctx, tuple(min(1, v * 1.05) for v in (c * light for c in cols[-1])))
    ctx.fill()


def port(ctx, glow, F):
    t, lt, u = F.t, F.local, F.u
    sxk, syk = shake(t, 10 * F.impact)
    iso = Iso(cx=-40 + 60 * u, cy=120 - 40 * u, ang=-0.42 + 0.1 * ease_io(u), S=2.25 + 0.35 * ease_io(u), tilt=0.62)
    ctx.save()
    ctx.translate(sxk, syk)
    ctx.set_source_rgb(0.006, 0.01, 0.02)
    ctx.paint()
    # far shore city lights across the water
    base = cached("farcity", lambda: rng(81).random((1500, 3)))
    fx, fy = iso(-700 + base[:, 0] * 1400, -330 - base[:, 1] * 40, base[:, 2] * 8)
    glow.points(fx + sxk, fy + syk, (1.0, 0.7, 0.4), 0.8 * base[:, 2] * (0.7 + 0.3 * np.sin(t * 3 + base[:, 0] * 90)))
    # water
    w = [iso(-900, -330), iso(900, -330), iso(900, 0), iso(-900, 0)]
    poly(ctx, w)
    src(ctx, (0.01, 0.02, 0.035))
    ctx.fill()
    # yard apron
    poly(ctx, [iso(-900, 0), iso(900, 0), iso(900, 500), iso(-900, 500)])
    src(ctx, (0.05, 0.045, 0.04))
    ctx.fill()
    ctx.set_line_width(1.0)
    for yl in (14, 40, 66):
        a, b = iso(-900, yl), iso(900, yl)
        ctx.move_to(*a)
        ctx.line_to(*b)
    ctx.set_source_rgba(0.9, 0.75, 0.3, 0.25)
    ctx.stroke()

    masts = [(-420 + i * 120, 70 + 102 * j) for i in range(8) for j in range(3)]
    mx = np.array([m[0] for m in masts])
    my = np.array([m[1] for m in masts])

    def light_at(X, Y):
        d2 = (mx - X) ** 2 + (my - Y) ** 2
        return min(0.55, 0.07 + 0.35 * float(np.sum(np.exp(-d2 / (2 * 55 ** 2)))))

    # sodium pools on the ground
    for (X, Y) in masts:
        sx_, sy_ = iso(X, Y)
        ctx.save()
        ctx.translate(sx_, sy_)
        ctx.scale(1.0, iso.tilt)
        radial(ctx, 0, 0, 90 * iso.S, (1.0, 0.6, 0.25), 0.07, op=ADD)
        ctx.restore()

    stacks, ship = cached("yard", _yard)
    # ship hull, then its deck cargo
    hull = [iso(-420, -66, 0), iso(400, -66, 0), iso(420, -12, 0), iso(-400, -12, 0)]
    poly(ctx, hull)
    src(ctx, (0.08, 0.06, 0.06))
    ctx.fill()
    poly(ctx, [iso(-400, -12, 0), iso(420, -12, 0), iso(420, -12, -14), iso(-400, -12, -14)])
    src(ctx, (0.12, 0.05, 0.04))
    ctx.fill()
    for (X, Y, h, cols) in ship:
        draw_stack(ctx, iso, X, Y, h, cols, 0.32, base_z=2.0)
    bx, by = iso(430, -40, 10)
    # container yard, far to near
    for (X, Y, h, cols) in stacks:
        sx_, sy_ = iso(X, Y)
        if sx_ < -150 or sx_ > W + 150 or sy_ < LB - 120 or sy_ > H - LB + 150:
            continue
        draw_stack(ctx, iso, X, Y, h, cols, light_at(X + 6, Y + 2))
    # ship-to-shore cranes, spreaders cycling on the beat
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for i in range(7):
        X = -330 + i * 110
        ph = (F.beat * 0.5 + i * 0.37) % 1.0
        lift = 0.5 - 0.5 * math.cos(ph * 2 * math.pi)
        legs = [(X, 4), (X + 30, 4), (X, 36), (X + 30, 36)]
        ctx.set_line_width(2.4)
        ctx.set_source_rgba(0.75, 0.72, 0.66, 0.95)
        for (lx, ly) in legs:
            ctx.move_to(*iso(lx, ly, 0))
            ctx.line_to(*iso(lx, ly, 55))
        for (ly) in (4, 36):
            ctx.move_to(*iso(X, ly, 55))
            ctx.line_to(*iso(X + 30, ly, 55))
        ctx.move_to(*iso(X + 15, 20, 55))
        ctx.line_to(*iso(X + 15, 20, 85))
        ctx.stroke()
        ctx.set_line_width(3.0)
        ctx.move_to(*iso(X + 15, 90, 58))
        ctx.line_to(*iso(X + 15, -95, 58))
        ctx.stroke()
        ctx.set_line_width(1.0)
        ctx.move_to(*iso(X + 15, 20, 85))
        ctx.line_to(*iso(X + 15, -95, 58))
        ctx.move_to(*iso(X + 15, 20, 85))
        ctx.line_to(*iso(X + 15, 90, 58))
        ctx.stroke()
        ty = lerp(-40, 18, 0.5 + 0.5 * math.sin(ph * 2 * math.pi + 1.2))
        zc = 12 + 40 * lift
        ctx.move_to(*iso(X + 15, ty, 58))
        ctx.line_to(*iso(X + 15, ty, zc + 5))
        ctx.stroke()
        draw_stack(ctx, iso, X + 9, ty - 2.6, 1, [BOX_COLORS[(i * 3) % 9]], 1.0, base_z=zc)
        tx, ty_ = iso(X + 15, 20, 86)
        glow.points([tx + sxk], [ty_ + syk], (1.0, 0.1, 0.05), 20 * (0.4 + 0.6 * F.pulse), soft=True)
        glow.points([tx + sxk], [ty_ + syk], (1.0, 0.3, 0.2), 4)
        for fy_ in (-60, 60):
            fx_, fyy = iso(X + 15, fy_ / 3, 58)
            glow.points([fx_ + sxk], [fyy + syk], (1.0, 0.95, 0.85), 3.0)
    # light masts
    for (X, Y) in masts:
        a = iso(X, Y, 0)
        b = iso(X, Y, 40)
        ctx.set_line_width(1.5)
        ctx.set_source_rgba(0.3, 0.3, 0.3, 1)
        ctx.move_to(*a)
        ctx.line_to(*b)
        ctx.stroke()
        glow.points([b[0] + sxk], [b[1] + syk], (1.0, 0.75, 0.45), 30, soft=True)
        glow.points([b[0] + sxk], [b[1] + syk], (1.0, 0.85, 0.6), 5)
    ctx.restore()
    # trucks and straddle carriers along the lanes
    base = cached("trucks", lambda: rng(82).random((70, 4)))
    lanes = np.array([14, 40, 66, 110, 146, 180, 214, 248, 282])
    lane = lanes[(base[:, 0] * len(lanes)).astype(int)]
    dirn = np.where(base[:, 1] > 0.5, 1, -1)
    X = ((base[:, 2] * 1400 + dirn * t * (40 + 30 * base[:, 3])) % 1400) - 700
    hx, hy = iso(X + dirn * 5, lane + 2, 2)
    tx, ty = iso(X - dirn * 5, lane + 2, 2)
    glow.points(hx + sxk, hy + syk, (1.0, 0.95, 0.85), 6.0)
    glow.points(hx + sxk, hy + syk, (1.0, 0.9, 0.7), 10.0, soft=True)
    glow.points(tx + sxk, ty + syk, (1.0, 0.1, 0.05), 3.0)
    # water reflections of the quay lights
    ctx.save()
    ctx.set_operator(ADD)
    for i in range(40):
        X = -420 + i * 21
        a = iso(X, -2, 0)
        b = iso(X + 2, -150, 0)
        g = cairo.LinearGradient(*a, *b)
        g.add_color_stop_rgba(0, 1.0, 0.65, 0.3, 0.15)
        g.add_color_stop_rgba(1, 1.0, 0.65, 0.3, 0)
        ctx.set_source(g)
        ctx.set_line_width(2 + 2 * math.sin(i * 1.3 + t * 2) ** 2)
        ctx.move_to(a[0] + sxk, a[1] + syk)
        ctx.line_to(b[0] + sxk, b[1] + syk)
        ctx.stroke()
    ctx.restore()
    return {"bloom": 0.1}


# ------------------------------------------------------------------ the globe

CONTINENTS = {
    "na": [(-165, 64), (-160, 70), (-140, 70), (-125, 72), (-95, 72), (-80, 73), (-65, 62), (-60, 55), (-55, 50),
           (-65, 45), (-70, 42), (-75, 38), (-76, 35), (-81, 31), (-80, 26), (-82, 25), (-84, 30), (-90, 30),
           (-97, 27), (-97, 22), (-92, 18), (-88, 15), (-83, 10), (-78, 8), (-80, 7), (-85, 11), (-92, 14),
           (-105, 20), (-110, 24), (-115, 30), (-118, 34), (-123, 39), (-124, 43), (-124, 48), (-130, 54),
           (-135, 58), (-145, 60), (-152, 58), (-158, 56), (-165, 60)],
    "sa": [(-78, 8), (-72, 12), (-62, 10), (-52, 5), (-50, 0), (-40, -3), (-35, -7), (-38, -14), (-40, -22),
           (-48, -26), (-53, -34), (-58, -38), (-65, -41), (-66, -47), (-69, -52), (-72, -54), (-75, -50),
           (-73, -40), (-72, -30), (-70, -18), (-76, -14), (-81, -6), (-80, 0), (-77, 4)],
    "eu": [(-10, 36), (-9, 43), (-2, 44), (-5, 48), (0, 50), (5, 53), (8, 57), (5, 62), (15, 69), (25, 71),
           (40, 68), (55, 70), (70, 73), (80, 73), (100, 77), (110, 74), (130, 72), (150, 71), (170, 69),
           (180, 66), (178, 62), (163, 60), (157, 51), (142, 53), (140, 48), (135, 43), (128, 39), (126, 35),
           (121, 31), (122, 25), (117, 23), (110, 20), (108, 15), (109, 11), (105, 9), (103, 1), (100, 6),
           (98, 15), (94, 17), (92, 22), (88, 22), (80, 15), (77, 8), (73, 17), (70, 22), (66, 25), (57, 25),
           (56, 27), (52, 24), (58, 22), (59, 20), (52, 15), (44, 12), (42, 15), (35, 28), (34, 31), (36, 36),
           (28, 37), (26, 40), (23, 37), (20, 40), (16, 38), (18, 40), (13, 44), (12, 45), (14, 41), (10, 44),
           (3, 43), (-1, 37), (-6, 36)],
    "af": [(-17, 21), (-16, 12), (-12, 7), (-8, 4), (0, 5), (8, 4), (9, 0), (12, -6), (13, -12), (12, -18),
           (15, -27), (18, -34), (20, -35), (26, -34), (32, -29), (35, -24), (40, -15), (40, -10), (42, -2),
           (48, 4), (51, 11), (44, 11), (43, 13), (38, 18), (34, 27), (32, 31), (25, 32), (20, 31), (11, 33),
           (10, 37), (0, 36), (-6, 36), (-10, 30), (-13, 27)],
    "au": [(114, -22), (114, -34), (118, -35), (124, -33), (131, -31), (137, -35), (140, -38), (146, -39),
           (150, -37), (153, -32), (153, -25), (146, -19), (142, -11), (141, -14), (136, -12), (131, -11),
           (126, -14), (122, -18)],
    "uk": [(-5, 50), (1, 51), (2, 53), (-2, 56), (-3, 58), (-6, 58), (-5, 55), (-3, 54), (-5, 52)],
    "jp": [(130, 31), (132, 34), (136, 35), (140, 36), (142, 40), (140, 42), (141, 45), (143, 44), (145, 43),
           (141, 41), (140, 38), (135, 33)],
    "gl": [(-55, 60), (-45, 60), (-20, 70), (-20, 80), (-60, 82), (-72, 78), (-55, 70)],
    "id": [(95, 5), (106, -6), (115, -8), (120, -9), (125, -8), (118, -4), (117, 5), (109, 2), (104, -2)],
    "mg": [(44, -25), (47, -25), (50, -15), (49, -12), (44, -17)],
}

PORTS = {
    "shanghai": (31.2, 121.5), "singapore": (1.3, 103.8), "rotterdam": (51.9, 4.5), "la": (33.7, -118.3),
    "ny": (40.6, -74.0), "hk": (22.3, 114.2), "busan": (35.1, 129.0), "dubai": (25.0, 55.1),
    "hamburg": (53.5, 9.9), "santos": (-23.9, -46.3), "durban": (-29.9, 31.0), "sydney": (-33.9, 151.2),
    "colombo": (6.9, 79.8), "aden": (12.6, 45.0), "suez": (30.0, 32.5), "gib": (36.0, -5.6),
    "panama": (9.0, -79.6), "tokyo": (35.5, 139.8), "mumbai": (18.9, 72.8), "lagos": (6.4, 3.4),
    "vancouver": (49.3, -123.1), "capetown": (-33.9, 18.4), "kingston": (17.9, -76.8), "valparaiso": (-33.0, -71.6),
}

SEA_ROUTES = [
    ["shanghai", (30, 150), (35, -160), (34, -130), "la"],
    ["busan", (42, 170), (45, -150), "vancouver"],
    ["shanghai", "hk", "singapore", "colombo", "aden", "suez", (34, 20), "gib", (45, -10), "rotterdam", "hamburg"],
    ["rotterdam", (48, -20), (42, -50), "ny"],
    ["la", (15, -100), "panama", "kingston", (30, -72), "ny"],
    ["santos", (-10, -20), (20, -25), (40, -15), "rotterdam"],
    ["singapore", (-10, 80), (-30, 50), "durban", "capetown", (-10, 0), "lagos"],
    ["singapore", (-15, 115), (-38, 135), "sydney"],
    ["dubai", (20, 62), "mumbai", "colombo"],
    ["tokyo", (20, 160), (-20, 170), "sydney"],
    ["panama", (-10, -85), "valparaiso"],
]

AIR_ROUTES = [["hk", "dubai"], ["shanghai", "la"], ["rotterdam", "ny"], ["dubai", "lagos"], ["tokyo", "vancouver"],
              ["singapore", "sydney"], ["ny", "santos"], ["hamburg", "mumbai"], ["la", "tokyo"]]


def ll2v(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.sin(lo), np.sin(la), np.cos(la) * np.cos(lo)], -1)


def slerp_path(pts, n=36):
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        dot = np.clip(np.dot(a, b), -1, 1)
        om = math.acos(dot)
        for i in range(n):
            s = i / n
            if om < 1e-6:
                out.append(a)
            else:
                out.append((math.sin((1 - s) * om) * a + math.sin(s * om) * b) / math.sin(om))
    out.append(pts[-1])
    return np.array(out)


def _pip(lon, lat, poly_):
    px = np.array([p[0] for p in poly_])
    py = np.array([p[1] for p in poly_])
    inside = np.zeros(lon.shape, bool)
    j = len(px) - 1
    for i in range(len(px)):
        c = ((py[i] > lat) != (py[j] > lat)) & (lon < (px[j] - px[i]) * (lat - py[i]) / (py[j] - py[i] + 1e-12) + px[i])
        inside ^= c
        j = i
    return inside


def _globe_data():
    r = rng(909)
    n = 160000
    z = r.uniform(-1, 1, n)
    lon = r.uniform(-180, 180, n)
    lat = np.degrees(np.arcsin(z))
    land = np.zeros(n, bool)
    for poly_ in CONTINENTS.values():
        land |= _pip(lon, lat, poly_)
    lon, lat = lon[land], lat[land]
    w = 0.15 + 0.85 * r.random(len(lon)) ** 4
    for (plat, plon) in PORTS.values():
        d = np.hypot(lat - plat, (lon - plon) * np.cos(np.radians(plat)))
        w += 2.5 * np.exp(-(d / 3.0) ** 2)
    keep = r.random(len(lon)) < np.clip(w, 0, 1) * 0.9 + 0.1
    city = ll2v(lat[keep], lon[keep])
    cw = np.clip(w[keep], 0, 3)
    coast = []
    for poly_ in CONTINENTS.values():
        pts = [ll2v(p[1], p[0]) for p in poly_] + [ll2v(poly_[0][1], poly_[0][0])]
        coast.append(slerp_path(pts, 6))
    sea = []
    for route in SEA_ROUTES:
        pts = [ll2v(*PORTS[p]) if isinstance(p, str) else ll2v(p[0], p[1]) for p in route]
        sea.append(slerp_path(pts, 30))
    air_ = []
    for a, b in AIR_ROUTES:
        path = slerp_path([ll2v(*PORTS[a]), ll2v(*PORTS[b])], 90)
        s = np.linspace(0, 1, len(path))
        air_.append(path * (1 + 0.09 * np.sin(np.pi * s))[:, None])
    return city, cw, coast, sea, air_


def globe(ctx, glow, F):
    t, lt, u = F.t, F.local, F.u
    sxk, syk = shake(t, 10 * F.impact)
    city, cw, coast, sea, air_ = cached("globe", _globe_data)
    R = 400 * (2.6 - 1.55 * ease_out(u, 2.2))
    cx, cy = CX + sxk, CY + 60 + 380 * (1 - ease_out(u, 2.2)) + syk
    lon0 = math.radians(-95 + 10 * lt)
    tilt = math.radians(-22 + 8 * u)
    cl, sl = math.cos(lon0), math.sin(lon0)
    ct, st_ = math.cos(tilt), math.sin(tilt)

    def rot(v):
        x = v[..., 0] * cl - v[..., 2] * sl
        z = v[..., 0] * sl + v[..., 2] * cl
        y = v[..., 1]
        y2 = y * ct - z * st_
        z2 = y * st_ + z * ct
        return x, y2, z2

    stars(glow, 31, 900, LB, H - LB, t, bright=0.6)
    # atmosphere and ocean
    g = cairo.RadialGradient(cx, cy, R * 0.96, cx, cy, R * 1.12)
    g.add_color_stop_rgba(0, 0.2, 0.5, 1.0, 0.55)
    g.add_color_stop_rgba(0.3, 0.1, 0.3, 0.8, 0.25)
    g.add_color_stop_rgba(1, 0.05, 0.1, 0.4, 0.0)
    ctx.set_source(g)
    ctx.arc(cx, cy, R * 1.12, 0, 2 * math.pi)
    ctx.fill()
    g = cairo.RadialGradient(cx - R * 0.35, cy - R * 0.4, R * 0.1, cx, cy, R)
    g.add_color_stop_rgb(0, 0.03, 0.06, 0.13)
    g.add_color_stop_rgb(1, 0.005, 0.012, 0.03)
    ctx.set_source(g)
    ctx.arc(cx, cy, R, 0, 2 * math.pi)
    ctx.fill()

    x, y, z = rot(city)
    vis = z > 0.02
    sx_ = cx + x[vis] * R
    sy_ = cy - y[vis] * R
    limb = np.clip(z[vis] * 3, 0, 1)
    tw = 0.8 + 0.2 * np.sin(t * 4 + x[vis] * 80)
    glow.points(sx_, sy_, (1.0, 0.72, 0.38), 0.55 * cw[vis] * limb * tw * (R / 400) ** 0.5)

    ctx.save()
    ctx.set_operator(ADD)
    ctx.set_line_width(1.2)
    for path in coast:
        px, py, pz = rot(path)
        _polyline(ctx, cx + px * R, cy - py * R, pz, (0.4, 0.6, 0.9), 0.25)
    reveal = ease_out(u * 1.6, 2)
    for i, path in enumerate(sea):
        px, py, pz = rot(path)
        n = max(2, int(len(path) * reveal))
        ctx.set_line_width(2.0)
        _polyline(ctx, cx + px[:n] * R, cy - py[:n] * R, pz[:n], (0.3, 0.85, 1.0), 0.55)
        ctx.set_line_width(7.0)
        _polyline(ctx, cx + px[:n] * R, cy - py[:n] * R, pz[:n], (0.3, 0.7, 1.0), 0.08)
        m = np.arange(0, n, 5)
        off = int(t * 12 + i * 3) % 5
        m = np.clip(m + off, 0, n - 1)
        ok = pz[m] > 0.02
        glow.points(cx + px[m][ok] * R, cy - py[m][ok] * R, (0.6, 0.95, 1.0), 5.0)
    for i, path in enumerate(air_):
        px, py, pz = rot(path)
        k = (t * 0.35 + i * 0.13) % 1.4
        n0 = int(len(path) * clamp(k - 0.4))
        n1 = max(n0 + 2, int(len(path) * clamp(k)))
        ctx.set_line_width(2.0)
        _polyline(ctx, cx + px[n0:n1] * R, cy - py[n0:n1] * R, pz[n0:n1] + 0.2, (1.0, 0.75, 0.35), 0.7)
        if n1 - 1 < len(path) and pz[n1 - 1] > -0.1:
            glow.points([cx + px[n1 - 1] * R], [cy - py[n1 - 1] * R], (1.0, 0.85, 0.5), 30, soft=True)
    ctx.restore()
    for name, (plat, plon) in PORTS.items():
        v = ll2v(plat, plon)
        px, py, pz = rot(v)
        if pz > 0.05:
            glow.points([cx + px * R], [cy - py * R], (0.7, 0.95, 1.0), 20 * (0.6 + 0.4 * F.pulse), soft=True)
            glow.points([cx + px * R], [cy - py * R], (0.9, 1.0, 1.0), 6)
    return {"bloom": 0.15}


def _polyline(ctx, xs, ys, zs, col, a):
    started = False
    for xx, yy, zz in zip(xs, ys, zs):
        if zz > 0.0:
            if not started:
                ctx.move_to(xx, yy)
                started = True
            else:
                ctx.line_to(xx, yy)
        else:
            started = False
    ctx.set_source_rgba(*col, a)
    ctx.stroke()


# ------------------------------------------------------------------ autonomy: the robot floor

def _robots():
    r = rng(1234)
    bots = []
    for i in range(760):
        horiz = i % 2 == 0
        lane = int(r.integers(-22, 22))
        start = int(r.integers(-40, 40))
        d = 1 if r.random() > 0.5 else -1
        pod = r.random() < 0.45
        bots.append((horiz, lane * 2 + (0 if horiz else 1), start, d, pod, r.random()))
    return bots


def robots(ctx, glow, F):
    t, lt, u = F.t, F.local, F.u
    sxk, syk = shake(t, 9 * F.impact)
    cell = 64.0
    zoom = 1.7 * (1 - 0.62 * ease_io(u))
    cam = Cam(x=0, y=0, zoom=zoom, roll=0.15 + 0.2 * ease_io(u), sx=sxk, sy=syk)
    ctx.set_source_rgb(0.012, 0.016, 0.028)
    ctx.paint()
    cam.push(ctx, 1.0)
    span = 60
    ctx.set_line_width(1.2)
    for k in range(-span, span + 1):
        ctx.move_to(CX + k * cell, CY - span * cell)
        ctx.line_to(CX + k * cell, CY + span * cell)
        ctx.move_to(CX - span * cell, CY + k * cell)
        ctx.line_to(CX + span * cell, CY + k * cell)
    ctx.set_source_rgba(0.2, 0.6, 0.9, 0.14)
    ctx.stroke()
    for (px, py) in ((0, 0), (-14, 8), (16, -9), (10, 12), (-18, -12)):
        radial(ctx, CX + px * cell, CY + py * cell, 900, (0.25, 0.45, 0.7), 0.1)
    beat = F.beat
    fb = beat - math.floor(beat)
    nb = math.floor(beat)
    bots = cached("bots", _robots)
    heads_x, heads_y, heads_c = [], [], []
    for (horiz, lane, start, d, pod, rr) in bots:
        if horiz:
            k = ease_io(min(1.0, fb * 2))
            pos = start + d * (nb + k)
            x = CX + (((pos + 40) % 80) - 40) * cell
            y = CY + lane * cell
            moving = fb < 0.5
            vx, vy = d, 0
        else:
            k = ease_io(max(0.0, fb * 2 - 1))
            pos = start + d * (nb + k)
            x = CX + lane * cell
            y = CY + (((pos + 30) % 60) - 30) * cell
            moving = fb >= 0.5
            vx, vy = 0, d
        if moving:
            g = cairo.LinearGradient(x - vx * cell * 1.4, y - vy * cell * 1.4, x, y)
            g.add_color_stop_rgba(0, 0.2, 0.8, 1.0, 0.0)
            g.add_color_stop_rgba(1, 0.2, 0.8, 1.0, 0.35)
            ctx.set_source(g)
            ctx.set_line_width(cell * 0.3)
            ctx.move_to(x - vx * cell * 1.4, y - vy * cell * 1.4)
            ctx.line_to(x, y)
            ctx.stroke()
        s = cell * (0.78 if pod else 0.62)
        _rrect(ctx, x - s / 2, y - s / 2, s, s, s * 0.18)
        if pod:
            src(ctx, (0.1, 0.13, 0.2))
            ctx.fill()
            ctx.set_line_width(1.5)
            ctx.set_source_rgba(0.35, 0.5, 0.7, 0.8)
            for q in (1, 2):
                ctx.move_to(x - s / 2 + q * s / 3, y - s / 2 + 3)
                ctx.line_to(x - s / 2 + q * s / 3, y + s / 2 - 3)
                ctx.move_to(x - s / 2 + 3, y - s / 2 + q * s / 3)
                ctx.line_to(x + s / 2 - 3, y - s / 2 + q * s / 3)
            ctx.stroke()
            for q in range(3):
                ctx.rectangle(x - s / 2 + 4 + q * s / 3, y - s / 2 + 4, s / 3 - 8, s / 3 - 8)
                src(ctx, BOX_COLORS[int(rr * 9 + q) % 9], 0.7)
                ctx.fill()
        else:
            g = cairo.LinearGradient(x - s / 2, y - s / 2, x + s / 2, y + s / 2)
            g.add_color_stop_rgb(0, 1.0, 0.6, 0.2)
            g.add_color_stop_rgb(1, 0.7, 0.3, 0.05)
            ctx.set_source(g)
            ctx.fill()
            ctx.arc(x, y, s * 0.22, 0, 2 * math.pi)
            src(ctx, (0.15, 0.1, 0.08))
            ctx.fill()
        heads_x.append(x + vx * s * 0.45)
        heads_y.append(y + vy * s * 0.45)
        heads_c.append(1.0 if moving else 0.35)
    cam.pop(ctx)
    hx, hy = cam.project(np.array(heads_x), np.array(heads_y), 1.0)
    glow.points(hx, hy, (0.3, 0.9, 1.0), np.array(heads_c) * 5.0)
    glow.points(hx, hy, (0.3, 0.8, 1.0), np.array(heads_c) * 5.0, soft=True)
    return {"bloom": 0.1}


def _rrect(ctx, x, y, w, h, r):
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 1.5 * math.pi)
    ctx.close_path()


# ------------------------------------------------------------------ autonomy: the swarm becomes a caravan

def swarm(ctx, glow, F):
    t, lt, u, bar = F.t, F.local, F.u, F.bar
    sxk, syk = shake(t, 8 * F.impact + 5 * F.riser)
    tilt = 120 * (1 - ease_io(clamp((bar - 35.0) / 1.2)))
    cam = Cam(y=tilt, zoom=1.0 + 0.06 * u, sx=sxk, sy=syk)
    cam.push(ctx, 3.0)
    vgrad(ctx, -200, 820, [(0, (0.01, 0.015, 0.05)), (0.55, (0.04, 0.06, 0.16)), (0.85, (0.16, 0.12, 0.26)),
                           (1, (0.35, 0.22, 0.3))], -400, W + 400)
    cam.pop(ctx)
    stars(glow, 51, 700, LB - 100, 700, t, bright=0.5, cam=cam, depth=6)
    # skyline
    cam.push(ctx, 2.5)
    r_ = rng(515)
    xx = -500
    win_x, win_y = [], []
    src(ctx, (0.015, 0.018, 0.03))
    while xx < 2500:
        w = r_.uniform(40, 120)
        h = r_.uniform(60, 280) * (1.6 if r_.random() < 0.1 else 1.0)
        ctx.rectangle(xx, 790 - h, w, h + 400)
        ctx.fill()
        for _ in range(int(h * w / 900)):
            win_x.append(xx + r_.uniform(4, w - 4))
            win_y.append(790 - r_.uniform(6, h - 4))
        xx += w + r_.uniform(2, 20)
    cam.pop(ctx)
    wx, wy = cam.project(np.array(win_x), np.array(win_y), 2.5)
    wr = cached("winr", lambda: rng(516).random(len(win_x)))
    glow.points(wx, wy, (1.0, 0.8, 0.55), 0.9 * (wr[: len(wx)] > 0.45) * (0.6 + 0.4 * wr[: len(wx)]))
    # highway and trucks
    cam.push(ctx, 1.0)
    vgrad(ctx, 840, H + 300, [(0, (0.03, 0.03, 0.04)), (1, (0.005, 0.005, 0.01))], -400, W + 400)
    ctx.rectangle(-400, 840, W + 800, 6)
    src(ctx, (0.2, 0.22, 0.28))
    ctx.fill()
    for k in range(40):
        xx = (k * 120 - t * 600) % (W + 800) - 400
        ctx.rectangle(xx, 868, 50, 3)
        ctx.set_source_rgba(0.8, 0.8, 0.9, 0.3)
        ctx.fill()
    for i in range(3):
        tx = ((i * 760 + t * 420) % (W + 1600)) - 800
        _truck(ctx, glow, cam, tx, 846, t, i)
    cam.pop(ctx)

    # drones
    n = 420
    base = cached("drones", lambda: rng(700).random((n, 6)))
    walk = F.beat * math.pi
    pts = []
    for ci in range(3):
        px, py = camel_outline_points(walk + ci * 0.9, 120, rider=(ci == 0))
        cx = 1240 - ci * 330 + 40 * (bar - 35)
        pts.append((cx + px * 112, 560 + py * 112))
    tx = np.concatenate([p[0] for p in pts])
    ty = np.concatenate([p[1] for p in pts])
    extra = n - len(tx)
    wxs = 1420 + 40 * (bar - 35) + (base[:extra, 0] - 0.5) * 30
    wys = 560 - base[:extra, 1] * 190
    tx = np.concatenate([tx, wxs])
    ty = np.concatenate([ty, wys])
    rise = ease_out(clamp((bar - 35.0) / 0.5), 2)
    form = ease_io(clamp((bar - 35.4) / 0.6))
    gx = base[:, 0] * W
    gy = 900 + base[:, 1] * 60
    sx0 = base[:, 2] * W * 1.1 - W * 0.05
    sy0 = 250 + base[:, 3] * 500 + 30 * np.sin(t * 0.8 + base[:, 4] * 9)
    x = gx + (sx0 - gx) * rise
    y = gy + (sy0 - gy) * rise
    x = x + (tx - x) * form + 3 * np.sin(t * 3 + base[:, 5] * 40)
    y = y + (ty - y) * form + 3 * np.cos(t * 2.6 + base[:, 5] * 30)
    x, y = cam.project(x, y, 1.2)
    warm = base[:, 4] > 0.15
    br = (1.2 + 1.5 * form + 2.0 * F.riser) * (0.75 + 0.25 * np.sin(t * 7 + base[:, 5] * 60))
    glow.points(x[warm], y[warm], (1.0, 0.82, 0.55), br[warm] * 2.2)
    glow.points(x[~warm], y[~warm], (0.4, 0.85, 1.0), br[~warm] * 2.2)
    glow.points(x, y, (1.0, 0.8, 0.6), br * 1.2, soft=True)
    return {"bloom": 0.15 + 0.4 * F.riser}


def _truck(ctx, glow, cam, x, y, t, i):
    L = 520
    ctx.rectangle(x - L, y - 150, L - 20, 128)
    g = cairo.LinearGradient(0, y - 150, 0, y - 22)
    g.add_color_stop_rgb(0, 0.5, 0.52, 0.58)
    g.add_color_stop_rgb(1, 0.12, 0.13, 0.16)
    ctx.set_source(g)
    ctx.fill()
    ctx.rectangle(x - L, y - 60, L - 20, 4)
    ctx.set_source_rgba(0.3, 0.85, 1.0, 0.9)
    ctx.fill()
    ctx.move_to(x - 10, y - 20)
    ctx.line_to(x - 10, y - 150)
    ctx.curve_to(x + 60, y - 150, x + 110, y - 110, x + 118, y - 40)
    ctx.line_to(x + 118, y - 20)
    ctx.close_path()
    g = cairo.LinearGradient(x - 10, 0, x + 118, 0)
    g.add_color_stop_rgb(0, 0.12, 0.13, 0.16)
    g.add_color_stop_rgb(1, 0.4, 0.42, 0.48)
    ctx.set_source(g)
    ctx.fill()
    ctx.move_to(x + 30, y - 128)
    ctx.curve_to(x + 70, y - 124, x + 98, y - 100, x + 106, y - 70)
    ctx.set_line_width(4)
    ctx.set_source_rgba(0.3, 0.9, 1.0, 1)
    ctx.stroke()
    for wx in (x - L + 60, x - L + 130, x - 90, x + 70):
        ctx.arc(wx, y - 18, 20, 0, 2 * math.pi)
        src(ctx, (0.02, 0.02, 0.02))
        ctx.fill()
    ctx.save()
    ctx.set_operator(ADD)
    g = cairo.LinearGradient(x - L - 900, 0, x - L, 0)
    g.add_color_stop_rgba(0, 1.0, 0.1, 0.05, 0.0)
    g.add_color_stop_rgba(1, 1.0, 0.1, 0.05, 0.55)
    ctx.set_source(g)
    ctx.rectangle(x - L - 900, y - 44, 900, 5)
    ctx.fill()
    poly(ctx, [(x + 118, y - 44), (x + 900, y - 90), (x + 900, y + 10), (x + 118, y - 36)])
    g = cairo.LinearGradient(x + 118, 0, x + 900, 0)
    g.add_color_stop_rgba(0, 0.8, 0.95, 1.0, 0.25)
    g.add_color_stop_rgba(1, 0.8, 0.95, 1.0, 0)
    ctx.set_source(g)
    ctx.fill()
    ctx.restore()
    hx, hy = cam.project(x + 116, y - 40, 1.0)
    glow.points([hx], [hy], (0.8, 0.95, 1.0), 60, soft=True)
    tx_, ty_ = cam.project(x - L, y - 42, 1.0)
    glow.points([tx_], [ty_], (1.0, 0.1, 0.05), 20, soft=True)


# ------------------------------------------------------------------ end card

def finale(ctx, glow, F):
    t = F.t
    dt = F.local
    ctx.set_source_rgb(0, 0, 0)
    ctx.paint()
    radial(ctx, CX, CY + 20, 900, (0.5, 0.35, 0.2), 0.18 * smooth(dt / 2.0), op=ADD)
    ctx.save()
    ctx.set_operator(ADD)
    for k, (w, a) in enumerate(((1.0, 0.1), (0.4, 0.2), (0.12, 0.45))):
        ctx.save()
        ctx.translate(CX, 528)
        ctx.scale(900 * w + 300 * smooth(dt / 3), 3 + 20 * w)
        g = cairo.RadialGradient(0, 0, 0, 0, 0, 1)
        e = math.exp(-dt / 2.5) * 0.7 + 0.3 * smooth((dt - 0.5) / 2)
        g.add_color_stop_rgba(0, 1.0, 0.8, 0.5, a * e)
        g.add_color_stop_rgba(1, 1.0, 0.8, 0.5, 0)
        ctx.set_source(g)
        ctx.arc(0, 0, 1, 0, 2 * math.pi)
        ctx.fill()
        ctx.restore()
    ctx.restore()
    base = cached("motes_end", lambda: rng(12).random((700, 5)))
    z = 0.6 + base[:, 2] * 3
    x = (base[:, 0] * W * 1.2 + 10 * t / z + 30 * np.sin(t * 0.25 + base[:, 3] * 9)) % (W * 1.2) - W * 0.1
    y = (base[:, 1] * H * 1.1 - 12 * t / z) % (H * 1.1) - H * 0.05
    glow.points(x, y, (1.0, 0.8, 0.55), 0.35 * smooth(dt / 1.5) / z * (0.6 + 0.4 * np.sin(t * 2 + base[:, 4] * 20)))
    flash = 1.4 * math.exp(-max(0.0, dt) / 0.22)
    return {"flash": flash, "bloom": 0.3 * math.exp(-dt / 0.5)}
