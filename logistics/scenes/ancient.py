"""Cold open, Silk Road, Via Appia, Alexandria."""
import math

import cairo
import numpy as np

from logistics.gfx import (W, H, LB, CX, CY, Cam, cached, rng, vgrad, radial, poly, fill_under, src,
                           ellipse, mix, smooth, ease_io, ease_out, clamp, vnoise, fbm, shake, lerp)
from logistics.scenes.common import world_span, draw_camel, draw_walker, stars

ADD = cairo.OPERATOR_ADD


# ------------------------------------------------------------------ cold open

def intro(ctx, glow, F):
    t = F.t
    cam = Cam(x=-30 * F.u, zoom=1.0 + 0.07 * ease_io(F.u))
    vgrad(ctx, 0, H, [(0, (0.02, 0.016, 0.013)), (1, (0.035, 0.026, 0.018))])
    grow = smooth(t / 3.0)
    cam.push(ctx, 2.0)
    top, bot = (640.0, LB - 80.0), (1240.0, H - LB + 80.0)
    ctx.save()
    ctx.set_operator(ADD)
    for w, a in ((1.0, 0.035), (0.72, 0.045), (0.5, 0.06), (0.3, 0.08), (0.14, 0.08)):
        wt, wb = 150 * w, 560 * w
        poly(ctx, [(top[0] - wt, top[1]), (top[0] + wt, top[1]), (bot[0] + wb, bot[1]), (bot[0] - wb, bot[1])])
        g = cairo.LinearGradient(*top, *bot)
        g.add_color_stop_rgba(0, 1.0, 0.82, 0.58, a * grow * 1.3)
        g.add_color_stop_rgba(0.6, 1.0, 0.75, 0.5, a * grow * 0.8)
        g.add_color_stop_rgba(1, 1.0, 0.7, 0.45, a * grow * 0.3)
        ctx.set_source(g)
        ctx.fill()
    ctx.restore()
    radial(ctx, bot[0], bot[1] - 90, 520, (0.9, 0.6, 0.35), 0.18 * grow, op=ADD)
    cam.pop(ctx)

    base = cached("motes", lambda: rng(11).random((1400, 5)))
    z = 0.5 + base[:, 2] * 3.5
    x = (base[:, 0] * W * 1.2 + 14 * t / z + 30 * np.sin(t * 0.25 + base[:, 3] * 9)) % (W * 1.2) - W * 0.1
    y = (base[:, 1] * (H * 1.1) - 9 * t / z + 20 * np.sin(t * 0.31 + base[:, 4] * 7)) % (H * 1.1) - H * 0.05
    # distance to beam axis (in screen space; the beam is drawn at depth 2)
    ax, ay = cam.project(np.array([top[0], bot[0]]), np.array([top[1], bot[1]]), 2.0)
    dx, dy = ax[1] - ax[0], ay[1] - ay[0]
    L = math.hypot(dx, dy)
    along = ((x - ax[0]) * dx + (y - ay[0]) * dy) / (L * L)
    perp = np.abs((x - ax[0]) * dy - (y - ay[0]) * dx) / L
    hw = (150 + 410 * np.clip(along, 0, 1)) * 0.55
    beam = np.exp(-(perp / hw) ** 2 * 1.6)
    inten = (0.03 + beam * 0.9) * grow * (0.7 + 0.3 * np.sin(t * 2 + base[:, 3] * 30))
    far = z > 1.3
    glow.points(x[far], y[far], (1.0, 0.85, 0.62), inten[far] * 0.9 / z[far])
    near = ~far
    for xi, yi, zi, ii in zip(x[near][:90], y[near][:90], z[near][:90], inten[near][:90]):
        r = 7 + 26 * (1.3 - zi)
        radial(ctx, xi, yi, r, (1.0, 0.8, 0.55), min(0.5, ii * 0.35), inner=0.6, op=ADD)
    return {}


# ------------------------------------------------------------------ Silk Road

SKY_HOR = (1.0, 0.6, 0.32)


def dune_y(x, base, amp, freq, seed):
    n = fbm(x * freq, seed, 3)
    ridged = (1 - np.abs(n * 1.6)) ** 2
    return base - amp * (0.55 * ridged + 0.45 * (0.5 + 0.5 * np.sin(x * freq * 0.7 + seed)))


def silk(ctx, glow, F):
    t, lt, u = F.t, F.local, F.u
    sx, sy = shake(t, 10 * F.impact)
    cam = Cam(x=70 * lt - 200, y=-15 * u, zoom=1.02 + 0.12 * ease_io(u), sx=sx, sy=sy)

    vgrad(ctx, 0, H, [(0, (0.03, 0.035, 0.1)), (0.3, (0.12, 0.08, 0.2)), (0.47, (0.45, 0.2, 0.27)),
                      (0.54, (0.85, 0.42, 0.26)), (0.58, SKY_HOR), (1, SKY_HOR)])
    stars(glow, 5, 500, LB, 470, t, bright=0.55, cam=cam, depth=50)
    sun_x, sun_y = 1210.0, 505.0 + 70 * u
    cam.push(ctx, 20)
    for k in range(9):
        cx_ = -200 + k * 290 + 25 * math.sin(k * 3.1) + 6 * t
        cy_ = 300 + 60 * math.sin(k * 1.7) + 30 * (k % 2)
        ctx.save()
        ctx.translate(cx_, cy_)
        ctx.scale(260 + 80 * math.sin(k), 9 + 5 * (k % 3))
        g = cairo.RadialGradient(0, 0, 0, 0, 0, 1)
        warm = math.exp(-((cx_ - sun_x) / 700) ** 2)
        g.add_color_stop_rgba(0, 0.9, 0.45 + 0.25 * warm, 0.35, 0.28 + 0.3 * warm)
        g.add_color_stop_rgba(1, 0.9, 0.45, 0.35, 0.0)
        ctx.set_source(g)
        ctx.arc(0, 0, 1, 0, 2 * math.pi)
        ctx.fill()
        ctx.restore()
    cam.pop(ctx)
    cam.push(ctx, 30)
    radial(ctx, sun_x, sun_y, 900, (1.0, 0.5, 0.25), 0.45, op=ADD)
    radial(ctx, sun_x, sun_y, 260, (1.0, 0.78, 0.5), 0.8, op=ADD)
    ctx.set_source_rgb(1.0, 0.94, 0.78)
    ctx.arc(sun_x, sun_y, 50, 0, 2 * math.pi)
    ctx.fill()
    cam.pop(ctx)
    ssx, ssy = cam.project(sun_x, sun_y, 30)
    glow.points([ssx], [ssy], (1.0, 0.8, 0.55), 260.0, soft=True)

    layers = [
        # depth, base, amp, freq, seed, color, haze
        (12.0, 606, 60, 1 / 380, 1, mix(SKY_HOR, (0.42, 0.2, 0.28), 0.6), 0.0),
        (7.0, 648, 34, 1 / 260, 2, (0.5, 0.22, 0.17), 0.3),
        (3.6, 712, 60, 1 / 900, 3, (0.13, 0.06, 0.055), 0.28),
        (1.9, 850, 90, 1 / 500, 4, (0.06, 0.03, 0.028), 0.1),
        (1.0, 960, 60, 1 / 420, 5, (0.025, 0.013, 0.012), 0.0),
    ]
    for li, (d, base, amp, fr, seed, col, haze) in enumerate(layers):
        if haze > 0:
            yb = cam.project(0, base - amp * 0.6, d)[1]
            vgrad(ctx, yb - 160, yb + 40, [(0, (*SKY_HOR, 0.0)), (0.7, (*SKY_HOR, haze)), (1, (*SKY_HOR, haze * 0.6))])
        x0, x1 = world_span(cam, d)
        xs = np.linspace(x0, x1, 300)
        ys = dune_y(xs, base, amp, fr, seed)
        cam.push(ctx, d)
        fill_under(ctx, xs, ys, H + 600, col)
        # rim light along the crest, strongest near the sun
        g = cairo.LinearGradient(x0, 0, x1, 0)
        sxw = sun_x + (cam.x / d) - cam.x / 30
        for k in range(9):
            xx = x0 + (x1 - x0) * k / 8
            a = math.exp(-((xx - sxw) / 520) ** 2) * (0.75 if li < 4 else 0.3)
            g.add_color_stop_rgba(k / 8, 1.0, 0.72, 0.42, a)
        ctx.set_source(g)
        ctx.set_line_width(2.2 / cam.scale(d))
        ctx.move_to(xs[0], ys[0])
        for xx, yy in zip(xs[1:], ys[1:]):
            ctx.line_to(xx, yy)
        ctx.stroke()
        if li == 2:
            caravan(ctx, glow, F, cam, d, base, amp, fr, seed)
        if li == 4:
            # sand ripples
            ctx.set_line_width(1.2)
            for k in range(14):
                yy0 = 980 + k * 22
                xx = np.linspace(x0, x1, 90)
                yy = yy0 + 6 * np.sin(xx / 60 + k * 1.7) + dune_y(xx, 0, 30, fr, seed) * 0.5
                ctx.move_to(xx[0], yy[0])
                for a_, b_ in zip(xx[1:], yy[1:]):
                    ctx.line_to(a_, b_)
                ctx.set_source_rgba(0.9, 0.55, 0.3, 0.05)
                ctx.stroke()
        cam.pop(ctx)

    # blowing sand, near the camera
    base = cached("sand", lambda: rng(21).random((1600, 4)))
    x = (base[:, 0] * W * 1.4 - 380 * t * (0.6 + base[:, 2])) % (W * 1.4) - W * 0.2
    y = 760 + base[:, 1] * 200 + 18 * np.sin(t * 1.3 + base[:, 3] * 20)
    glow.streaks(x, y, (1.0, 0.62, 0.35), 0.05 * (0.4 + base[:, 2]), 0.06, 26)
    return {}


def caravan(ctx, glow, F, cam, d, base, amp, fr, seed):
    lt = F.local
    walk = F.beat * math.pi  # one pacing cycle per two beats
    lead_x = 560 + 62 * lt
    col = (0.04, 0.02, 0.022)
    s = 60
    def ground(x):
        return float(dune_y(np.array([x]), base, amp, fr, seed)[0])
    for i in range(9):
        x = lead_x - 40 - i * 150
        y = ground(x) + 3
        slope = (ground(x + 20) - ground(x - 20)) / 40
        draw_camel(ctx, x, y, s * (0.95 + 0.1 * math.sin(i * 2.1)), walk + i * 0.9, col,
                   load=(i % 3 != 2), rider=(i % 4 == 0), ang=math.atan(slope) * 0.6)
        if i > 0:
            ctx.set_line_width(1.0)
            src(ctx, col, 0.8)
            ctx.move_to(x + 0.95 * s, y - 1.6 * s)
            ctx.curve_to(x + 1.3 * s, y - 1.0 * s, x + 1.8 * s, y - 1.1 * s, x + 150 - 0.55 * s, y - 1.2 * s)
            ctx.stroke()
    xw = lead_x + 40
    draw_walker(ctx, xw, ground(xw) + 2, s * 0.9, walk * 1.3, col)
    xw2 = lead_x - 9 * 150 + 30
    draw_walker(ctx, xw2, ground(xw2) + 2, s * 0.85, walk * 1.3 + 1.0, col, staff=False)


# ------------------------------------------------------------------ Via Appia

VPX, VPY, FOC, HC = 960.0, 540.0, 880.0, 1.7
DAWN_HAZE = (0.98, 0.78, 0.55)


def proj(X, Y, Z):
    Z = np.maximum(Z, 0.05)
    return VPX + FOC * X / Z, VPY + FOC * (HC - Y) / Z


def rome_road(ctx, glow, F):
    t, lt, u = F.t, F.local, F.u
    sxk, syk = shake(t, 9 * F.impact)
    camz = 3.2 * lt
    cam = Cam(zoom=1.0 + 0.05 * u, sx=sxk, sy=syk, y=10 * u)
    cam.push(ctx, 1.0)
    vgrad(ctx, LB - 40, VPY + 20, [(0, (0.14, 0.22, 0.4)), (0.45, (0.5, 0.46, 0.52)), (0.8, (1.0, 0.7, 0.42)),
                                    (1, (1.0, 0.86, 0.62))])
    sun = (1030.0, 505.0 - 20 * u)
    radial(ctx, *sun, 800, (1.0, 0.7, 0.4), 0.45, op=ADD)
    radial(ctx, *sun, 200, (1.0, 0.9, 0.7), 0.9, op=ADD)
    ctx.set_source_rgb(1, 0.97, 0.88)
    ctx.arc(*sun, 30, 0, 2 * math.pi)
    ctx.fill()
    # god rays
    ctx.save()
    ctx.set_operator(ADD)
    for k in range(16):
        a0 = -math.pi + k * (2 * math.pi / 16) + 0.05 * math.sin(t * 0.3 + k)
        w = 0.035 + 0.02 * math.sin(k * 3.3)
        poly(ctx, [sun, (sun[0] + 2200 * math.cos(a0 - w), sun[1] + 2200 * math.sin(a0 - w)),
                   (sun[0] + 2200 * math.cos(a0 + w), sun[1] + 2200 * math.sin(a0 + w))])
        g = cairo.RadialGradient(*sun, 0, *sun, 1400)
        g.add_color_stop_rgba(0, 1, 0.85, 0.6, 0.10)
        g.add_color_stop_rgba(1, 1, 0.85, 0.6, 0.0)
        ctx.set_source(g)
        ctx.fill()
    ctx.restore()
    # distant hills
    xs = np.linspace(-200, W + 200, 200)
    fill_under(ctx, xs, VPY - 18 - 30 * (0.5 + 0.5 * fbm(xs / 300, 9)), H, (0.62, 0.52, 0.5), 0.9)
    fill_under(ctx, xs, VPY - 4 - 16 * (0.5 + 0.5 * fbm(xs / 180 + 4, 10)), H, (0.4, 0.33, 0.3), 1.0)
    # fields
    vgrad(ctx, VPY, H, [(0, (0.42, 0.34, 0.26)), (0.12, (0.2, 0.17, 0.1)), (0.5, (0.07, 0.065, 0.04)), (1, (0.02, 0.02, 0.015))])

    # road surface base
    zfar = 400.0
    RW = 3.6
    pts = [proj(-RW, 0, zfar), proj(RW, 0, zfar), proj(RW, 0, 0.6), proj(-RW, 0, 0.6)]
    poly(ctx, pts)
    g = cairo.LinearGradient(0, VPY, 0, H)
    g.add_color_stop_rgb(0, 0.62, 0.52, 0.45)
    g.add_color_stop_rgb(0.2, 0.2, 0.17, 0.14)
    g.add_color_stop_rgb(1, 0.06, 0.05, 0.045)
    ctx.set_source(g)
    ctx.fill()
    # paving stones (only near rows are individually drawn)
    dz = 0.62
    k0 = int(camz / dz)
    for k in range(70, -1, -1):
        kk = k0 + k
        z0 = kk * dz - camz
        z1 = z0 + dz
        if z1 < 0.4:
            continue
        z0 = max(z0, 0.4)
        ncol = 6
        off = hash_(kk) * 0.8
        fogk = 1 - math.exp(-z0 / 55)
        for c in range(-1, ncol + 1):
            xa = -RW + (c + off) * (2 * RW / ncol)
            xb = xa + 2 * RW / ncol * (0.8 + 0.3 * hash_(kk * 7 + c))
            xa, xb = max(xa, -RW), min(xb, RW)
            if xb - xa < 0.05:
                continue
            gap = 0.04
            j1, j2, j3, j4 = (0.12 * (hash_(kk * 5 + c * 11 + m) - 0.5) for m in range(4))
            q = [proj(xa + gap + j1, 0, z0 + gap + j2 * 0.5), proj(xb - gap + j2, 0, z0 + gap + j3 * 0.5),
                 proj(xb - gap + j3, 0, z1 - gap + j4 * 0.5), proj(xa + gap + j4, 0, z1 - gap + j1 * 0.5)]
            h = hash_(kk * 13 + c * 5)
            base = (0.36 + 0.2 * h, 0.32 + 0.17 * h, 0.28 + 0.14 * h)
            lit = 0.4 + 0.5 * math.exp(-abs((xa + xb) / 2 - 0.8) / 4) * (0.6 + 0.4 * hash_(kk * 3 + c))
            colr = mix(tuple(b * lit for b in base), DAWN_HAZE, fogk * 0.85)
            poly(ctx, q)
            src(ctx, colr)
            ctx.fill()
            # sun glint on the stone's far edge
            ctx.move_to(*q[2])
            ctx.line_to(*q[3])
            ctx.set_source_rgba(1, 0.85, 0.6, 0.35 * (1 - fogk))
            ctx.set_line_width(1.0)
            ctx.stroke()
    # kerbs
    for sgn in (-1, 1):
        poly(ctx, [proj(sgn * RW, 0, zfar), proj(sgn * (RW + 0.35), 0.12, zfar), proj(sgn * (RW + 0.35), 0.12, 0.6),
                   proj(sgn * RW, 0, 0.6)])
        src(ctx, (0.3, 0.26, 0.22))
        ctx.fill()

    # aqueduct, left
    AX = -34.0
    span = 7.0
    j0 = int(camz / span)
    for j in range(60, -2, -1):
        zc = (j0 + j) * span - camz
        if zc < 1.0:
            continue
        fogk = 1 - math.exp(-zc / 220)
        col = mix((0.2, 0.15, 0.14), (0.96, 0.8, 0.62), fogk * 0.9)
        pw = 0.9
        top, mid, arch = 17.0, 13.0, 9.5
        z2 = zc + span
        ctx.new_path()
        poly(ctx, [proj(AX, 0, zc - pw), proj(AX, top, zc - pw), proj(AX, top, zc + pw), proj(AX, 0, zc + pw)])
        src(ctx, col)
        ctx.fill()
        # spandrel with arch cut
        pts = [proj(AX, top, zc + pw), proj(AX, top, z2 - pw)]
        n = 14
        for i in range(n + 1):
            th = math.pi * i / n
            zz = (zc + pw + z2 - pw) / 2 + (z2 - pw - (zc + pw)) / 2 * math.cos(th)
            yy = arch + ((z2 - zc - 2 * pw) / 2) * math.sin(th) * 0.95
            pts.append(proj(AX, min(yy, mid + 1.5), zz))
        poly(ctx, pts)
        ctx.fill()
        poly(ctx, [proj(AX, top, zc - pw), proj(AX, top + 0.8, zc - pw), proj(AX, top + 0.8, z2 + pw), proj(AX, top, z2 + pw)])
        ctx.fill()

    # cypresses both sides
    for sgn in (-1, 1):
        sp = 8.5
        i0 = int(camz / sp)
        for i in range(40, -1, -1):
            zc = (i0 + i) * sp - camz + (sp / 2 if sgn > 0 else 0)
            if zc < 0.8:
                continue
            fogk = 1 - math.exp(-zc / 70)
            col = mix((0.02, 0.025, 0.015), (0.85, 0.72, 0.56), fogk)
            X = sgn * (7.5 + 2 * hash_(i0 + i + sgn * 99))
            h = 8 + 3 * hash_(i0 + i + 5 + sgn)
            bx, by = proj(X, 0, zc)
            tx, ty = proj(X, h, zc)
            w = FOC * 0.75 / zc
            ctx.move_to(bx - w * 0.6, by)
            ctx.curve_to(bx - w * 1.2, by - (by - ty) * 0.45, tx - w * 0.5, ty + (by - ty) * 0.2, tx, ty)
            ctx.curve_to(tx + w * 0.5, ty + (by - ty) * 0.2, bx + w * 1.2, by - (by - ty) * 0.45, bx + w * 0.6, by)
            ctx.close_path()
            src(ctx, col)
            ctx.fill()
        # milestones
        for i in range(8, -1, -1):
            zc = (int(camz / 20) + i) * 20 - camz + 6
            if zc < 1:
                continue
            fogk = 1 - math.exp(-zc / 70)
            bx, by = proj(sgn * (RW + 0.9), 0, zc)
            tx, ty = proj(sgn * (RW + 0.9), 1.4, zc)
            w = FOC * 0.28 / zc
            ctx.rectangle(bx - w, ty, 2 * w, by - ty)
            src(ctx, mix((0.5, 0.44, 0.38), DAWN_HAZE, fogk))
            ctx.fill()

    # legion marching toward the sun
    for i in range(18):
        zc = 26 + i * 1.3 + 1.5 * lt - camz * 0.25 + 1.0 * (i % 3)
        if zc < 1.5:
            continue
        X = -1.4 + 1.4 * (i % 3)
        fogk = 1 - math.exp(-zc / 60)
        bx, by = proj(X, 0, zc)
        s = FOC * 1.0 / zc
        col = mix((0.08, 0.06, 0.05), DAWN_HAZE, fogk * 0.9)
        draw_walker(ctx, bx, by, s, F.beat * math.pi + i, col, staff=True)
        ctx.rectangle(bx - 0.28 * s, by - 1.2 * s, 0.22 * s, 0.6 * s)
        src(ctx, col)
        ctx.fill()
    cam.pop(ctx)

    base = cached("dust_r", lambda: rng(33).random((900, 4)))
    x = (base[:, 0] * W + 12 * t * (base[:, 2] - 0.5)) % W
    y = LB + base[:, 1] * (H - 2 * LB)
    d = np.hypot(x - sun[0], y - sun[1])
    glow.points(x, y, (1.0, 0.85, 0.6), 0.25 * np.exp(-d / 450) * base[:, 3], soft=False)
    return {}


def hash_(i):
    x = math.sin(i * 12.9898 + 78.233) * 43758.5453
    return x - math.floor(x)


# ------------------------------------------------------------------ Alexandria

def rome_port(ctx, glow, F):
    t, lt, u = F.t, F.local, F.u
    sxk, syk = shake(t, 8 * F.impact)
    cam = Cam(x=60 * ease_io(u), y=40 - 70 * ease_io(u), zoom=1.0 + 0.1 * ease_io(u), sx=sxk, sy=syk)
    HOR = 612
    vgrad(ctx, 0, H, [(0, (0.02, 0.03, 0.08)), (0.35, (0.05, 0.07, 0.15)), (0.5, (0.2, 0.14, 0.2)),
                      (0.565, (0.55, 0.3, 0.22)), (0.57, (0.05, 0.06, 0.1)), (1, (0.01, 0.015, 0.03))])
    stars(glow, 8, 700, LB, 520, t, bright=0.8, cam=cam, depth=40)

    fire_x, fire_y = 1390.0, 266.0
    flick = 0.85 + 0.15 * math.sin(t * 17) * math.sin(t * 7.3 + 1) + 0.1 * float(vnoise(t * 9, 3))

    # far city, left shore
    cam.push(ctx, 3.0)
    r_ = rng(44)
    bx = -200
    ctx.new_path()
    ctx.move_to(-300, HOR + 10)
    while bx < 980:
        w = r_.uniform(40, 110)
        h = r_.uniform(20, 90) if r_.random() > 0.15 else r_.uniform(110, 170)
        ctx.line_to(bx, HOR - h)
        if r_.random() < 0.2:
            ctx.line_to(bx + w / 2, HOR - h - 28)
        ctx.line_to(bx + w, HOR - h)
        bx += w
    ctx.line_to(1000, HOR + 10)
    ctx.close_path()
    src(ctx, (0.05, 0.045, 0.06))
    ctx.fill()
    cam.pop(ctx)
    lamps = cached("alex_lamps", lambda: rng(45).random((120, 3)))
    lx, ly = cam.project(-200 + lamps[:, 0] * 1150, HOR - 6 - lamps[:, 1] * 80, 3.0)
    glow.points(lx, ly, (1.0, 0.6, 0.25), 1.2 * (0.6 + 0.4 * np.sin(t * 5 + lamps[:, 2] * 50)) * (lamps[:, 2] > 0.35))

    # water reflections
    cam.push(ctx, 1.5)
    ctx.save()
    ctx.set_operator(ADD)
    for k in range(120):
        y = HOR + 4 + k * 3.0
        spread = 10 + k * 1.9
        for j in range(3):
            xo = float(vnoise(k * 0.7 + t * 1.6 + j * 20, 7)) * spread * 1.5
            w = spread * (0.5 + 0.5 * hash_(k * 3 + j + int(t * 8)))
            a = 0.45 * math.exp(-k / 70) * flick * (0.5 + 0.5 * math.sin(k * 1.3 + t * 5 + j))
            ctx.rectangle(fire_x + xo - w / 2, y, w, 1.6)
            ctx.set_source_rgba(1.0, 0.55, 0.22, a)
            ctx.fill()
    for k in range(60):
        y = HOR + 6 + k * 5
        xo = float(vnoise(k + t, 2)) * 400
        ctx.rectangle(300 + xo, y, 80 + 3 * k, 1.2)
        ctx.set_source_rgba(0.35, 0.4, 0.6, 0.05)
        ctx.fill()
    ctx.restore()
    cam.pop(ctx)

    # Pharos
    cam.push(ctx, 1.4)
    ctx.move_to(1180, HOR + 30)
    ctx.curve_to(1230, HOR - 30, 1560, HOR - 36, 1620, HOR + 30)
    ctx.close_path()
    src(ctx, (0.03, 0.025, 0.03))
    ctx.fill()
    tiers = [(HOR - 12, 440, 90, 76), (440, 330, 60, 52), (330, 282, 32, 30)]
    for (y0, y1, w0, w1) in tiers:
        poly(ctx, [(fire_x - w0, y0), (fire_x + w0, y0), (fire_x + w1, y1), (fire_x - w1, y1)])
        g = cairo.LinearGradient(0, y1, 0, y0)
        g.add_color_stop_rgb(0, 0.42 * flick, 0.22 * flick, 0.1 * flick)
        g.add_color_stop_rgb(1, 0.04, 0.035, 0.04)
        ctx.set_source(g)
        ctx.fill()
        ctx.rectangle(fire_x - w1 - 6, y1 - 6, 2 * w1 + 12, 8)
        src(ctx, (0.3 * flick, 0.16 * flick, 0.08 * flick))
        ctx.fill()
    for k in range(6):
        wy = 570 - k * 46
        if wy < 332:
            break
        ctx.rectangle(fire_x - 4, wy, 8, 14)
        ctx.set_source_rgba(1, 0.6, 0.25, 0.8)
        ctx.fill()
    # flame
    for k in range(7):
        ph = t * (6 + k) + k
        h = 38 + 16 * math.sin(ph) * flick
        ctx.move_to(fire_x - 26 + k * 8, 282)
        ctx.curve_to(fire_x - 30 + k * 8, 270 - h * 0.4, fire_x - 10 + k * 5 + 6 * math.sin(ph * 1.3), 270 - h,
                     fire_x - 12 + k * 4, 275 - h)
        ctx.line_to(fire_x - 18 + k * 8, 282)
        ctx.close_path()
        ctx.set_source_rgba(1.0, 0.72 - k * 0.04, 0.3, 0.85)
        ctx.fill()
    radial(ctx, fire_x, fire_y, 380, (1.0, 0.5, 0.2), 0.55 * flick, op=ADD)
    radial(ctx, fire_x, fire_y, 110, (1.0, 0.8, 0.5), 0.9 * flick, op=ADD)
    # rotating beam from the mirror
    th = t * 1.1
    c = math.cos(th)
    ctx.save()
    ctx.set_operator(ADD)
    L = 1500 * abs(c) + 200
    sg = 1 if c > 0 else -1
    spread = 0.05 + 0.25 * (1 - abs(c))
    poly(ctx, [(fire_x, fire_y), (fire_x + sg * L, fire_y - L * spread * 0.4 + 60), (fire_x + sg * L, fire_y + L * spread * 0.6 + 60)])
    g = cairo.LinearGradient(fire_x, fire_y, fire_x + sg * L, fire_y)
    g.add_color_stop_rgba(0, 1, 0.75, 0.45, 0.35 * abs(c))
    g.add_color_stop_rgba(1, 1, 0.75, 0.45, 0.0)
    ctx.set_source(g)
    ctx.fill()
    ctx.restore()
    cam.pop(ctx)
    fx, fy = cam.project(fire_x, fire_y, 1.4)
    glow.points([fx], [fy], (1.0, 0.6, 0.3), 1500 * flick * (1 + 1.5 * max(0, math.sin(th)) ** 8), soft=True)

    # ships coming home
    for i, (x0, v, y, s, d) in enumerate([(-300, 38, HOR + 70, 0.55, 1.6), (200, 30, HOR + 150, 0.9, 1.25),
                                          (-700, 46, HOR + 290, 1.5, 1.0)]):
        x = x0 + v * lt + 40 * u * (i == 2)
        bob = 4 * s * math.sin(t * 1.3 + i * 2)
        cam.push(ctx, d)
        ship_ancient(ctx, x, y + bob, s, t + i, fx=fire_x)
        cam.pop(ctx)
        lx, ly = cam.project(x - 130 * s, y + bob - 60 * s, d)
        glow.points([lx], [ly], (1.0, 0.65, 0.3), 30 * s * flick, soft=True)
        glow.points([lx], [ly], (1.0, 0.8, 0.5), 4 * s)

    # embers rising
    base = cached("embers", lambda: rng(46).random((260, 4)))
    age = (t * 0.25 + base[:, 0]) % 1.0
    ex = fire_x + (base[:, 1] - 0.5) * 60 + age * (160 + 120 * base[:, 2]) + 30 * np.sin(age * 12 + base[:, 3] * 9)
    ey = fire_y - age * 420
    ex, ey = cam.project(ex, ey, 1.4)
    glow.points(ex, ey, (1.0, 0.55, 0.2), 1.6 * (1 - age) ** 2)
    return {}


def ship_ancient(ctx, x, y, s, t, fx=1400):
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(s, s)
    ctx.rotate(0.015 * math.sin(t * 1.1))
    col = (0.035, 0.03, 0.035)
    # hull with swan-neck stern
    ctx.move_to(-190, -40)
    ctx.curve_to(-150, 10, 120, 12, 190, -30)
    ctx.line_to(205, -44)
    ctx.line_to(150, -38)
    ctx.line_to(-150, -40)
    ctx.curve_to(-190, -60, -215, -90, -190, -120)
    ctx.curve_to(-205, -90, -200, -60, -190, -40)
    ctx.close_path()
    src(ctx, col)
    ctx.fill()
    # mast + yard + sail
    ctx.set_line_width(4)
    ctx.move_to(0, -40)
    ctx.line_to(0, -250)
    ctx.stroke()
    ctx.move_to(-110, -238)
    ctx.line_to(110, -238)
    ctx.stroke()
    ctx.move_to(-105, -236)
    ctx.curve_to(-80, -170, -95, -110, -100, -70)
    ctx.line_to(100, -70)
    ctx.curve_to(95, -110, 80, -170, 105, -236)
    ctx.close_path()
    g = cairo.LinearGradient(-100, 0, 100, 0)
    lit = 0.5 if fx > x else 0.3
    g.add_color_stop_rgb(0, 0.16 * lit, 0.1 * lit, 0.07 * lit)
    g.add_color_stop_rgb(1, 0.5 * lit, 0.28 * lit, 0.14 * lit)
    ctx.set_source(g)
    ctx.fill()
    # oars
    src(ctx, col)
    ctx.set_line_width(2)
    for k in range(9):
        ox = -120 + k * 28
        ph = t * 2.2 + k * 0.2
        ctx.move_to(ox, -30)
        ctx.line_to(ox - 30 + 14 * math.sin(ph), 20 + 8 * math.cos(ph))
        ctx.stroke()
    ctx.restore()
