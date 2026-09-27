"""Age of sail, steam, and the 1956 container."""
import math

import cairo
import numpy as np

from logistics.gfx import (W, H, LB, CX, CY, Cam, cached, rng, vgrad, radial, poly, fill_under, src,
                           ellipse, mix, smooth, ease_io, ease_out, ease_in, clamp, vnoise, fbm, shake,
                           lerp, noise2d_texture, surface_from_array)
from logistics.scenes.common import world_span, lightning_bolt, stars

ADD = cairo.OPERATOR_ADD


def _cloud_surfaces():
    tex = noise2d_texture(1024, 384, 77, octaves=7, base=3)
    a = np.clip((tex - 0.35) / 0.4, 0, 1) ** 1.3
    shade = np.clip((tex[..., None] - 0.35) / 0.65, 0, 1)
    dark = np.concatenate([mix_arr((0.02, 0.025, 0.03), (0.42, 0.45, 0.48), shade ** 2.5), a[..., None]], 2)
    lit = np.concatenate([mix_arr((0.35, 0.4, 0.5), (0.95, 0.97, 1.0), shade), a[..., None]], 2)
    return surface_from_array(dark.astype(np.float32)), surface_from_array(lit.astype(np.float32))


def mix_arr(c1, c2, t):
    c1 = np.array(c1, np.float32)
    c2 = np.array(c2, np.float32)
    return c1 + (c2 - c1) * t


def paint_tiled(ctx, surf, x, y, sx, sy, alpha=1.0):
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(sx, sy)
    pat = cairo.SurfacePattern(surf)
    pat.set_extend(cairo.EXTEND_REPEAT)
    pat.set_filter(cairo.FILTER_BILINEAR)
    ctx.set_source(pat)
    ctx.paint_with_alpha(alpha)
    ctx.restore()


# ------------------------------------------------------------------ Age of sail

HOR_S = 520


def wave(x, t, r, R):
    d = r / R
    k = 0.012 / (0.18 + d)
    A = 4 + 120 * d ** 2.0
    ph = r * 1.7
    v = (0.55 * np.sin(k * x - 1.1 * t + ph) + 0.3 * np.sin(1.9 * k * x + 1.6 * t + ph * 2.3)
         + 0.15 * np.sin(3.7 * k * x - 2.3 * t + ph * 0.7))
    return A * v, v


def row_y(r, R):
    return HOR_S + 470 * (r / R) ** 1.7


def sail(ctx, glow, F):
    t, lt, u = F.t, F.local, F.u
    L = min(1.0, F.lightning)
    sxk, syk = shake(t, 12 * F.impact)
    ship_x = 820 + 30 * lt
    R = 24
    ship_row = 13
    hy, hv = wave(np.array([ship_x - 180, ship_x + 180]), t, ship_row, R)
    pitch = math.atan2(hy[1] - hy[0], 360) * 1.3
    heave = float(hy.mean())
    cam = Cam(x=ship_x - 900, y=heave * 0.5 - 30, zoom=1.0 + 0.14 * ease_io(u),
              roll=0.035 * math.sin(t * 0.8) + 0.01 * math.sin(t * 1.9), sx=sxk, sy=syk)

    cam.push(ctx, 6)
    vgrad(ctx, -400, HOR_S + 10, [(0, (0.03, 0.04, 0.05)), (0.7, (0.12, 0.14, 0.16)), (1, (0.3, 0.32, 0.33))], -800, W + 800)
    dark, lit = cached("clouds", _cloud_surfaces)
    paint_tiled(ctx, dark, -800 - (t * 25) % 2400, -380, 2.4, 2.4, 0.95)
    if L > 0.02:
        paint_tiled(ctx, lit, -800 - (t * 25) % 2400, -380, 2.4, 2.4, L * 0.6)
    cam.pop(ctx)

    if L > 0.05:
        seed = int(F.bar * 2)
        bx = 300 + (seed * 377) % 1300
        cam.push(ctx, 5)
        lightning_bolt(ctx, bx + cam.x / 5, LB - 60, bx + cam.x / 5 + 160 * math.sin(seed), HOR_S - 5, seed,
                       width=3.0, a=min(1, L))
        cam.pop(ctx)

    sky_col = mix((0.26, 0.29, 0.3), (0.8, 0.85, 0.95), min(1, L))
    for r in range(R + 1):
        if r == ship_row + 1:
            galleon(ctx, glow, cam, ship_x, row_y(ship_row, R) + heave + 30, pitch, t, L)
        d = r / R
        dep = 1 + 5 * (1 - d) ** 2
        x0, x1 = world_span(cam, dep, 200)
        x0 -= 400
        x1 += 400
        xs = np.linspace(x0, x1, 220)
        yw, v = wave(xs, t, r, R)
        ys = row_y(r, R) + yw
        cam.push(ctx, 1.0)
        base = mix(mix((0.2, 0.24, 0.26), sky_col, 0.6 * (1 - d) ** 3), (0.01, 0.025, 0.03), d ** 0.6)
        fill_under(ctx, xs, ys, H + 900, base)
        # foam on crests
        ctx.set_line_width(1 + 3 * d)
        ctx.move_to(xs[0], ys[0])
        for xx, yy in zip(xs[1:], ys[1:]):
            ctx.line_to(xx, yy)
        g = cairo.LinearGradient(x0, 0, x1, 0)
        for k in range(12):
            xx = x0 + (x1 - x0) * k / 11
            vv = float(wave(np.array([xx]), t, r, R)[1][0])
            g.add_color_stop_rgba(k / 11, 0.75, 0.82, 0.85, (0.02 + 0.8 * max(0, vv) ** 3) * (0.4 + 0.6 * d) + 0.3 * L)
        ctx.set_source(g)
        ctx.stroke()
        cam.pop(ctx)

    # rain
    base = cached("rain", lambda: rng(55).random((2200, 3)))
    fall = (base[:, 0] + t * (1.3 + base[:, 2])) % 1.0
    x = (base[:, 1] * W * 1.3 - W * 0.15 + fall * 260) % (W * 1.3) - W * 0.15
    y = LB - 40 + fall * (H - 2 * LB + 80)
    glow.streaks(x, y, (0.7, 0.78, 0.85), (0.06 + 0.25 * L) * (0.3 + base[:, 2]), 1.32, 46)
    return {"flash": 0.12 * L, "exposure": 1.0 + 0.15 * L}


def galleon(ctx, glow, cam, x, y, pitch, t, L):
    cam.push(ctx, 1.0)
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(pitch)
    s = 1.2
    ctx.scale(s, s)
    wood = mix((0.06, 0.045, 0.035), (0.3, 0.3, 0.34), L * 0.8)
    canvas_dark = mix((0.12, 0.12, 0.12), (0.55, 0.58, 0.65), L)
    canvas_lit = mix((0.34, 0.32, 0.28), (0.85, 0.88, 0.95), L)
    # masts, rigging (behind sails)
    masts = [(170, 400), (0, 470), (-175, 330)]
    ctx.set_line_width(1.2)
    src(ctx, wood, 0.9)
    for mx_, mh in masts:
        ctx.move_to(mx_, -mh)
        ctx.line_to(-300, -140)
        ctx.move_to(mx_, -mh)
        ctx.line_to(300, -100)
        for k in range(5):
            ctx.move_to(mx_, -mh * (0.45 + 0.1 * k))
            ctx.line_to(mx_ - 40 + 20 * k, -95)
    ctx.move_to(430, -170)
    ctx.line_to(170, -400)
    ctx.stroke()
    ctx.set_line_width(7)
    for mx_, mh in masts:
        ctx.move_to(mx_, -80)
        ctx.line_to(mx_, -mh)
    ctx.stroke()
    # square sails, billowing
    bill = 0.5 + 0.5 * math.sin(t * 1.7)
    for mi, (mx_, mh) in enumerate(masts[:2]):
        tiers = [(0.25, 0.52, 250), (0.55, 0.78, 205), (0.8, 0.95, 140)]
        for (a, b, w) in tiers:
            y0, y1 = -mh * b, -mh * a - 10
            w = w * (0.92 if mi == 0 else 1.0)
            bul = (18 + 10 * bill) * (w / 250)
            ctx.move_to(mx_ - w / 2, y0)
            ctx.curve_to(mx_ - w / 6, y0 - bul * 0.5, mx_ + w / 6, y0 - bul * 0.5, mx_ + w / 2, y0)
            ctx.curve_to(mx_ + w / 2 + bul, (y0 + y1) / 2, mx_ + w / 2 + bul * 0.6, y1, mx_ + w / 2 - 6, y1)
            ctx.curve_to(mx_ + w / 6, y1 + bul, mx_ - w / 6, y1 + bul, mx_ - w / 2 + 6, y1)
            ctx.curve_to(mx_ - w / 2 - bul * 0.3, y1, mx_ - w / 2 - bul * 0.5, (y0 + y1) / 2, mx_ - w / 2, y0)
            ctx.close_path()
            g = cairo.LinearGradient(mx_ - w / 2, 0, mx_ + w / 2, 0)
            g.add_color_stop_rgb(0, *canvas_dark)
            g.add_color_stop_rgb(0.65, *canvas_lit)
            g.add_color_stop_rgb(1, *mix(canvas_lit, canvas_dark, 0.4))
            ctx.set_source(g)
            ctx.fill()
            ctx.set_line_width(4)
            src(ctx, wood)
            ctx.move_to(mx_ - w / 2 - 8, y0)
            ctx.line_to(mx_ + w / 2 + 8, y0)
            ctx.stroke()
    # mizzen lateen
    ctx.move_to(-175, -330)
    ctx.line_to(-290, -120)
    ctx.curve_to(-230, -130, -150, -130, -110, -120)
    ctx.close_path()
    g = cairo.LinearGradient(-290, 0, -110, 0)
    g.add_color_stop_rgb(0, *canvas_dark)
    g.add_color_stop_rgb(1, *canvas_lit)
    ctx.set_source(g)
    ctx.fill()
    # pennant
    ctx.move_to(0, -470)
    for k in range(9):
        ctx.line_to(-k * 14, -470 - 6 * math.sin(t * 9 - k * 0.8) + k * 0.6)
    for k in range(8, -1, -1):
        ctx.line_to(-k * 14, -458 - 6 * math.sin(t * 9 - k * 0.8) - k * 0.6)
    ctx.close_path()
    ctx.set_source_rgba(0.5, 0.1, 0.08, 1)
    ctx.fill()
    # hull
    ctx.move_to(-300, -165)
    ctx.line_to(-215, -165)
    ctx.line_to(-205, -112)
    ctx.line_to(140, -98)
    ctx.line_to(160, -122)
    ctx.line_to(255, -122)
    ctx.curve_to(290, -110, 320, -90, 330, -70)
    ctx.curve_to(300, -10, 180, 40, 0, 45)
    ctx.curve_to(-160, 42, -260, 20, -285, -20)
    ctx.close_path()
    g = cairo.LinearGradient(0, -165, 0, 45)
    g.add_color_stop_rgb(0, *mix(wood, (0.3, 0.2, 0.12), 0.3))
    g.add_color_stop_rgb(1, *wood)
    ctx.set_source(g)
    ctx.fill()
    ctx.set_line_width(2.5)
    ctx.set_source_rgba(0.45, 0.32, 0.18, 0.7)
    for yy in (-80, -55):
        ctx.move_to(-280, yy - 10)
        ctx.curve_to(-100, yy + 8, 150, yy + 6, 305, yy - 18)
        ctx.stroke()
    ctx.set_line_width(5)
    src(ctx, wood)
    ctx.move_to(300, -110)
    ctx.line_to(440, -175)
    ctx.stroke()
    ctx.restore()
    cam.pop(ctx)
    # warm lights from gunports and stern lanterns
    c, sn = math.cos(pitch), math.sin(pitch)
    pts = [(-300, -175, 60), (-270, -175, 60)] + [(-220 + k * 45, -70, 6) for k in range(11)]
    for (px, py, inten) in pts:
        wx = x + (px * c - py * sn) * 1.2
        wy = y + (px * sn + py * c) * 1.2
        sx_, sy_ = cam.project(wx, wy, 1.0)
        glow.points([sx_], [sy_], (1.0, 0.62, 0.3), inten * (1 - 0.7 * min(1, L)), soft=inten > 20)
        glow.points([sx_], [sy_], (1.0, 0.75, 0.45), 3.0 if inten < 20 else 6.0)
    # spray at the bow
    base = cached("spray", lambda: rng(56).random((500, 4)))
    age = (base[:, 0] + t * 0.9) % 1.0
    bx, by = x + (330 * c + 70 * sn) * 1.2, y + (330 * sn - 70 * c) * 1.2
    sxp = bx + (base[:, 1] - 0.3) * 200 * age - 60 * age
    syp = by - 160 * age * (base[:, 2] + 0.2) + 260 * age * age
    sxp, syp = cam.project(sxp, syp, 1.0)
    glow.points(sxp, syp, (0.8, 0.85, 0.9), 0.35 * (1 - age) * (0.3 + 0.7 * max(0.0, -pitch * 12)) + 0.05, soft=False)


# ------------------------------------------------------------------ Steam

RAIL_Y = 800
DRIVER_R = 78


def train_speed(F):
    """One driver revolution per beat."""
    return 2 * math.pi * DRIVER_R / (60.0 / 80)


def _train_scene(ctx, glow, F, cam):
    t = F.t
    dist = train_speed(F) * t
    vgrad(ctx, 0, H, [(0, (0.01, 0.015, 0.04)), (0.45, (0.05, 0.07, 0.14)), (0.62, (0.12, 0.13, 0.2)),
                      (0.63, (0.03, 0.035, 0.05)), (1, (0.01, 0.01, 0.015))])
    stars(glow, 12, 600, LB, 560, t, bright=0.7, cam=cam, depth=60)
    cam.push(ctx, 40)
    mxp, myp = 430, 290
    radial(ctx, mxp, myp, 420, (0.5, 0.6, 0.85), 0.35, op=ADD)
    ctx.set_source_rgb(0.92, 0.94, 1.0)
    ctx.arc(mxp, myp, 34, 0, 2 * math.pi)
    ctx.fill()
    cam.pop(ctx)
    # far hills, scrolling
    for d, base, amp, col, seed in ((9, 660, 70, (0.05, 0.06, 0.1), 3), (4.5, 700, 50, (0.025, 0.03, 0.05), 4)):
        x0, x1 = world_span(cam, d)
        xs = np.linspace(x0, x1, 200)
        off = dist / d
        ys = base - amp * (0.5 + 0.5 * fbm((xs + off) / 500, seed, 4))
        cam.push(ctx, d)
        fill_under(ctx, xs, ys, H + 400, col)
        cam.pop(ctx)
    # tree line
    d = 2.6
    x0, x1 = world_span(cam, d)
    off = dist / d
    cam.push(ctx, d)
    src(ctx, (0.012, 0.015, 0.025))
    ctx.rectangle(x0, 735, x1 - x0, 600)
    ctx.fill()
    k0 = int((x0 + off) / 38) - 1
    for k in range(k0, k0 + int((x1 - x0) / 38) + 3):
        xx = k * 38 - off
        hh = 40 + 60 * (0.5 + 0.5 * math.sin(k * 1.37) * math.cos(k * 0.61))
        ellipse(ctx, xx, 735 - hh * 0.55, 30 + 10 * math.sin(k), hh * 0.6)
        ctx.fill()
    cam.pop(ctx)
    # embankment
    cam.push(ctx, 1.0)
    vgrad(ctx, RAIL_Y - 30, H + 300, [(0, (0.04, 0.04, 0.05)), (1, (0.01, 0.01, 0.012))], -600, W + 600)
    # sleepers with motion smear
    sp = 44
    k0 = int(dist / sp)
    for k in range(k0 - 30, k0 + 60):
        xx = k * sp - dist + 960 % sp
        ctx.rectangle(xx - 14, RAIL_Y + 6, 34, 10)
        ctx.set_source_rgba(0.08, 0.07, 0.06, 0.8)
        ctx.fill()
    ctx.rectangle(-600, RAIL_Y, W + 1200, 6)
    g = cairo.LinearGradient(0, RAIL_Y, 0, RAIL_Y + 6)
    g.add_color_stop_rgb(0, 0.55, 0.6, 0.75)
    g.add_color_stop_rgb(1, 0.1, 0.1, 0.12)
    ctx.set_source(g)
    ctx.fill()
    cam.pop(ctx)

    # telegraph poles: one passes the centre of frame on every beat
    d = 0.8
    spacing = train_speed(F) * (60.0 / 80) / d
    beat = F.beat
    cam.push(ctx, d)
    tops = []
    for k in range(int(beat) - 4, int(beat) + 5):
        xx = 960 + (k - beat) * spacing
        tops.append(xx)
        ctx.rectangle(xx - 11, 250, 22, 900)
        src(ctx, (0.012, 0.01, 0.01))
        ctx.fill()
        ctx.rectangle(xx - 70, 290, 140, 10)
        ctx.rectangle(xx - 55, 330, 110, 9)
        ctx.fill()
    ctx.set_line_width(2.0)
    for a, b in zip(tops[:-1], tops[1:]):
        for wy, sag in ((292, 40), (292, 44), (332, 36)):
            for xo in (-60, 60):
                ctx.move_to(a + xo, wy)
                ctx.curve_to(a + xo + (b - a) / 3, wy + sag, b + xo - (b - a) / 3, wy + sag, b + xo, wy)
    ctx.set_source_rgba(0.02, 0.02, 0.02, 0.9)
    ctx.stroke()
    cam.pop(ctx)
    return dist


def locomotive(ctx, glow, F, cam, x, dist):
    t = F.t
    theta = dist / DRIVER_R
    iron = (0.035, 0.035, 0.045)
    rim = (0.45, 0.52, 0.7)
    fire = 0.8 + 0.2 * math.sin(t * 13) * math.sin(t * 5.1) + 0.4 * F.riser
    cam.push(ctx, 1.0)
    ctx.save()
    ctx.translate(x, RAIL_Y)
    # firebox glow on the ground under the cab
    radial(ctx, -330, 0, 380, (1.0, 0.45, 0.15), 0.45 * fire, op=ADD)

    def body(fn):
        ctx.new_path()
        fn()
        src(ctx, iron)
        ctx.fill()

    # freight cars and tender
    for cx0, cw, ch in ((-1720, 470, 250), (-1225, 470, 250)):
        ctx.rectangle(cx0, -ch - 30, cw, ch)
        src(ctx, (0.05, 0.035, 0.03))
        ctx.fill()
        for k in range(9):
            ctx.rectangle(cx0 + 10 + k * 51, -ch - 22, 3, ch - 16)
            ctx.set_source_rgba(0.12, 0.08, 0.06, 1)
            ctx.fill()
        ctx.rectangle(cx0, -ch - 32, cw, 4)
        src(ctx, rim, 0.5)
        ctx.fill()
        for wx in (cx0 + 80, cx0 + 160, cx0 + cw - 160, cx0 + cw - 80):
            wheel(ctx, wx, -34, 34, theta * DRIVER_R / 34, iron, rim, spokes=8)
    ctx.rectangle(-735, -230, 300, 200)
    src(ctx, iron)
    ctx.fill()
    ellipse(ctx, -585, -232, 140, 26)
    src(ctx, (0.02, 0.02, 0.02))
    ctx.fill()
    ctx.rectangle(-735, -232, 300, 4)
    src(ctx, rim, 0.55)
    ctx.fill()
    for wx in (-680, -600, -560, -480):
        wheel(ctx, wx, -36, 36, theta * DRIVER_R / 36, iron, rim, spokes=8)
    # cab
    ctx.rectangle(-430, -370, 180, 260)
    src(ctx, iron)
    ctx.fill()
    ctx.move_to(-450, -370)
    ctx.line_to(-230, -370)
    ctx.line_to(-240, -392)
    ctx.line_to(-440, -392)
    ctx.close_path()
    ctx.fill()
    ctx.rectangle(-400, -340, 110, 90)
    ctx.set_source_rgba(1.0 * fire, 0.5 * fire, 0.15 * fire, 1)
    ctx.fill()
    ellipse(ctx, -350, -290, 22, 26)
    src(ctx, (0.02, 0.01, 0.01))
    ctx.fill()
    ctx.rectangle(-360, -268, 30, 30)
    ctx.fill()
    # boiler
    ctx.rectangle(-260, -260, 560, 130)
    g = cairo.LinearGradient(0, -260, 0, -130)
    g.add_color_stop_rgb(0, 0.25, 0.28, 0.38)
    g.add_color_stop_rgb(0.12, 0.05, 0.05, 0.07)
    g.add_color_stop_rgb(0.8, 0.03, 0.03, 0.04)
    g.add_color_stop_rgb(1, 0.25 * fire, 0.1 * fire, 0.04)
    ctx.set_source(g)
    ctx.fill()
    for bx in (-160, -40, 80, 200):
        ctx.rectangle(bx, -262, 8, 134)
        ctx.set_source_rgba(0.5, 0.38, 0.18, 0.8)
        ctx.fill()
    ctx.rectangle(300, -268, 50, 146)
    src(ctx, (0.02, 0.02, 0.025))
    ctx.fill()
    # domes and stack
    for dx, dw, dh in ((-120, 44, 44), (60, 50, 56)):
        ellipse(ctx, dx, -260, dw, dh)
        src(ctx, iron)
        ctx.fill()
        ctx.rectangle(dx - dw, -262, 2 * dw, 4)
        ctx.fill()
    poly(ctx, [(265, -262), (305, -262), (330, -390), (240, -390)])
    src(ctx, iron)
    ctx.fill()
    poly(ctx, [(228, -390), (342, -390), (322, -412), (248, -412)])
    ctx.fill()
    ctx.set_line_width(2)
    src(ctx, rim, 0.6)
    ctx.move_to(240, -392)
    ctx.line_to(330, -392)
    ctx.stroke()
    # headlamp
    ctx.rectangle(292, -330, 70, 62)
    src(ctx, iron)
    ctx.fill()
    radial(ctx, 362, -300, 70, (1.0, 0.9, 0.7), 1.0, op=ADD)
    ctx.save()
    ctx.set_operator(ADD)
    poly(ctx, [(362, -318), (1500, -520), (1500, -40), (362, -282)])
    g = cairo.LinearGradient(362, 0, 1500, 0)
    g.add_color_stop_rgba(0, 1.0, 0.9, 0.7, 0.16)
    g.add_color_stop_rgba(1, 1.0, 0.9, 0.7, 0.0)
    ctx.set_source(g)
    ctx.fill()
    ctx.restore()
    # running board, cylinder, pilot
    ctx.rectangle(-260, -130, 620, 12)
    src(ctx, iron)
    ctx.fill()
    ctx.rectangle(-260, -131, 620, 2)
    src(ctx, rim, 0.6)
    ctx.fill()
    ctx.rectangle(200, -118, 120, 64)
    src(ctx, iron)
    ctx.fill()
    poly(ctx, [(340, -60), (470, -4), (340, -4)])
    ctx.fill()
    ctx.set_line_width(3)
    src(ctx, rim, 0.35)
    for k in range(6):
        ctx.move_to(345 + k * 20, -54 + k * 9)
        ctx.line_to(345 + k * 20, -4)
    ctx.stroke()
    # frame plate between the wheels
    ctx.rectangle(-270, -125, 560, 70)
    src(ctx, (0.02, 0.02, 0.025))
    ctx.fill()
    # leading truck
    for wx in (170, 275):
        wheel(ctx, wx, -40, 40, theta * DRIVER_R / 40, iron, rim, spokes=10)
    # drivers + rods
    drivers = (-200, -20)
    for wx in drivers:
        wheel(ctx, wx, -DRIVER_R, DRIVER_R, theta, iron, rim, spokes=14, counter=True)
    pins = [(wx + 0.55 * DRIVER_R * math.cos(theta), -DRIVER_R + 0.55 * DRIVER_R * math.sin(theta)) for wx in drivers]
    ctx.set_line_width(12)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    src(ctx, (0.3, 0.3, 0.34))
    ctx.move_to(*pins[0])
    ctx.line_to(*pins[1])
    ctx.stroke()
    Lr = 230
    px, py = pins[1]
    chx = px + math.sqrt(max(1, Lr * Lr - (py + 86) ** 2))
    ctx.set_line_width(10)
    ctx.move_to(px, py)
    ctx.line_to(chx, -86)
    ctx.stroke()
    ctx.rectangle(chx - 14, -98, 28, 24)
    ctx.fill()
    ctx.set_line_width(5)
    ctx.move_to(chx, -86)
    ctx.line_to(210, -86)
    ctx.stroke()
    ctx.set_line_width(2)
    src(ctx, rim, 0.8)
    ctx.move_to(pins[0][0], pins[0][1] - 6)
    ctx.line_to(pins[1][0], pins[1][1] - 6)
    ctx.stroke()
    ctx.restore()
    cam.pop(ctx)

    # steam and smoke: continuous plume from the stack, lit by moon above and fire below
    stack = (x + 285, RAIL_Y - 412)
    ctx.save()
    n = 56
    for i in range(n, -1, -1):
        dt_ = (i + ((t * 20) % 1.0)) / 20.0
        age = dt_
        born = t - age
        px_ = stack[0] - 330 * age - 60 * age * age + 40 * float(vnoise(born * 3, 8))
        py_ = stack[1] - 120 * age + 30 * age * age + 30 * float(vnoise(born * 2.3, 9))
        r = 34 + 150 * age ** 0.8 + 16 * math.sin(born * 11)
        a = 0.6 * max(0.0, 1 - age / 2.8) ** 1.2
        warm = math.exp(-age * 2.5) * fire
        sxp, syp = cam.project(px_, py_, 1.0)
        rr = r * cam.scale(1.0)
        g = cairo.RadialGradient(sxp - rr * 0.2, syp - rr * 0.3, rr * 0.1, sxp, syp, rr)
        top = mix((0.32, 0.34, 0.42), (1.0, 0.55, 0.25), warm * 0.6)
        g.add_color_stop_rgba(0, *top, a)
        g.add_color_stop_rgba(0.6, *mix(top, (0.08, 0.08, 0.1), 0.5), a * 0.7)
        g.add_color_stop_rgba(1, 0.05, 0.05, 0.06, 0)
        ctx.set_source(g)
        ctx.arc(sxp, syp, rr, 0, 2 * math.pi)
        ctx.fill()
    ctx.restore()
    # cylinder exhaust chuffs on every eighth note
    eighth = F.beat * 2
    for k in range(3):
        e = int(eighth) - k
        age = (eighth - e) * (60.0 / 80 / 2) + k * (60.0 / 80 / 2)
        if age > 1.0:
            continue
        px_ = x + 320 - 380 * age
        py_ = RAIL_Y - 40 - 40 * age
        sxp, syp = cam.project(px_, py_, 1.0)
        rr = (20 + 120 * age) * cam.scale(1.0)
        radial(ctx, sxp, syp, rr, (0.75, 0.78, 0.85), 0.5 * (1 - age) ** 2)
    # sparks
    base = cached("sparks", lambda: rng(66).random((400, 4)))
    age = (base[:, 0] + t * 1.4) % 1.0
    sx_ = stack[0] - age * (500 + 300 * base[:, 1]) + (base[:, 2] - 0.5) * 60
    sy_ = stack[1] - age * (200 + 200 * base[:, 3]) + 180 * age * age
    sx_, sy_ = cam.project(sx_, sy_, 1.0)
    glow.streaks(sx_, sy_, (1.0, 0.6, 0.2), 1.2 * (1 - age) ** 2 * (base[:, 1] > 0.4) * fire, 0.2, 18)
    hx, hy = cam.project(x + 362, RAIL_Y - 300, 1.0)
    glow.points([hx], [hy], (1.0, 0.9, 0.75), 900, soft=True)
    fx, fy = cam.project(x - 345, RAIL_Y - 295, 1.0)
    glow.points([fx], [fy], (1.0, 0.5, 0.2), 300 * fire, soft=True)


def wheel(ctx, x, y, r, theta, iron, rim, spokes=12, counter=False):
    ctx.save()
    ctx.translate(x, y)
    ctx.arc(0, 0, r, 0, 2 * math.pi)
    src(ctx, iron)
    ctx.fill()
    ctx.set_line_width(max(2, r * 0.1))
    ctx.arc(0, 0, r * 0.93, 0, 2 * math.pi)
    src(ctx, (0.1, 0.1, 0.12))
    ctx.stroke()
    ctx.rotate(theta)
    ctx.set_line_width(max(1.5, r * 0.05))
    src(ctx, (0.16, 0.16, 0.19))
    for k in range(spokes):
        a = 2 * math.pi * k / spokes
        ctx.move_to(0, 0)
        ctx.line_to(r * 0.9 * math.cos(a), r * 0.9 * math.sin(a))
    ctx.stroke()
    if counter:
        ctx.arc(0, 0, r * 0.85, math.pi - 0.7, math.pi + 0.7)
        ctx.arc_negative(0, 0, r * 0.35, math.pi + 0.7, math.pi - 0.7)
        ctx.close_path()
        src(ctx, (0.05, 0.05, 0.06))
        ctx.fill()
    ctx.arc(0, 0, r * 0.14, 0, 2 * math.pi)
    src(ctx, (0.25, 0.25, 0.28))
    ctx.fill()
    ctx.restore()
    ctx.save()
    ctx.set_line_width(max(1.5, r * 0.04))
    ctx.arc(x, y, r * 0.98, math.pi * 1.05, math.pi * 1.6)
    src(ctx, rim, 0.8)
    ctx.stroke()
    ctx.restore()


def rail_side(ctx, glow, F):
    t, u = F.t, F.u
    sxk, syk = shake(t, 10 * F.impact + 2.0)
    bounce = 2.0 * math.sin(F.beat * 2 * math.pi * 2)
    cam = Cam(x=-60 + 120 * ease_io(u), y=-20 + bounce, zoom=0.86 + 0.08 * ease_io(u), sx=sxk, sy=syk)
    dist = _train_scene(ctx, glow, F, cam)
    locomotive(ctx, glow, F, cam, 1000, dist)
    return {}


def rail_wheels(ctx, glow, F):
    t, u = F.t, F.u
    sxk, syk = shake(t, 10 * F.impact + 3.0 + 10 * F.riser)
    bounce = 3.0 * math.sin(F.beat * 2 * math.pi * 2)
    cam = Cam(x=-130 + 60 * u, y=190 + bounce, zoom=1.9 + 0.25 * ease_in(u, 2), roll=-0.03, sx=sxk, sy=syk)
    dist = _train_scene(ctx, glow, F, cam)
    locomotive(ctx, glow, F, cam, 1000, dist)
    # rail sparks at the drivers
    base = cached("rsparks", lambda: rng(67).random((300, 4)))
    age = (base[:, 0] + t * 3.0) % 1.0
    for wx in (-200, -20):
        sx_ = 1000 + wx - age * (300 + 400 * base[:, 1])
        sy_ = RAIL_Y - age * 90 * base[:, 2] + 300 * age * age
        sx_, sy_ = cam.project(sx_, sy_, 1.0)
        glow.streaks(sx_, sy_, (1.0, 0.7, 0.3), 1.6 * (1 - age) ** 3 * (base[:, 3] > 0.5), 0.05, 26)
    return {"exposure": 1.0 + 0.25 * F.riser}


# ------------------------------------------------------------------ 1956

FOG = (0.62, 0.66, 0.7)
QUAY = 790
DECK = 600


def _fog_tex():
    tex = noise2d_texture(1024, 256, 91, octaves=6, base=4)
    a = np.clip((tex - 0.25) / 0.6, 0, 1)
    rgba = np.concatenate([np.ones((256, 1024, 3), np.float32) * np.array(FOG, np.float32) * 1.05,
                           a[..., None]], 2)
    return surface_from_array(rgba)


def box_path(bar):
    """Crane choreography for the first box, locked to bars 21-24."""
    rest = (470.0, QUAY - 100 - 50)
    lift_y = 330.0
    target = (1110.0, DECK - 68)
    if bar < 21:
        return rest[0], rest[1], 0.0, 0.0
    if bar < 22:
        k = ease_io(bar - 21)
        return rest[0], lerp(rest[1], lift_y, k), 0.0, k
    if bar < 23.25:
        k = ease_io((bar - 22) / 1.25)
        return lerp(rest[0], target[0], k), lift_y, (bar - 22) / 1.25, 1.0
    if bar < 24:
        k = (bar - 23.25) / 0.75
        k = 1 - (1 - k) ** 1.6
        return target[0], lerp(lift_y, target[1], k), 1.0, 1.0
    return target[0], target[1], 1.0, 1.0


def container(ctx, glow, F):
    t, lt, u, bar = F.t, F.local, F.u, F.bar
    sxk, syk = shake(t, 18 * F.impact)
    bx, by, travel, lifted = box_path(bar)
    swing = 0.0
    if 22 <= bar < 24.5:
        swing = 0.05 * math.sin((bar - 22) * 5.0) * math.exp(-max(0, bar - 23.3) * 2.0)
    after = smooth((bar - 24) / 0.9)
    zoom = 1.0 + 0.3 * ease_io((bar - 20.5) / 3.5)
    focus_x = lerp(960, bx + 60, 0.55 * ease_io((bar - 20.5) / 3.5))
    focus_y = lerp(540, by + 40, 0.45 * ease_io((bar - 20.5) / 3.5))
    cam = Cam(x=focus_x - 960, y=focus_y - 540, zoom=zoom, sx=sxk, sy=syk)

    sun = (820.0, 430.0 - 50 * after)
    cam.push(ctx, 8)
    vgrad(ctx, -300, 720, [(0, (0.2, 0.26, 0.34)), (0.6, (0.5, 0.54, 0.58)), (1, (0.8, 0.72, 0.64))], -800, W + 800)
    radial(ctx, *sun, 1000, (1.0, 0.66, 0.42), 0.4 + 0.35 * after, op=ADD)
    radial(ctx, *sun, 200, (1.0, 0.85, 0.65), 0.6 + 0.4 * after, op=ADD)
    ctx.set_source_rgba(1.0, 0.93, 0.82, 0.55 + 0.4 * after)
    ctx.arc(*sun, 38, 0, 2 * math.pi)
    ctx.fill()
    cam.pop(ctx)
    # far industrial shore
    cam.push(ctx, 5)
    r_ = rng(12)
    xx = -600
    src(ctx, mix((0.3, 0.33, 0.38), FOG, 0.55))
    while xx < 2600:
        kind = r_.random()
        w = r_.uniform(30, 140)
        if kind < 0.25:
            ctx.rectangle(xx, 640 - r_.uniform(60, 150), 14, 200)
        elif kind < 0.45:
            ellipse(ctx, xx + w / 2, 650, w / 2, 26)
            ctx.rectangle(xx, 650, w, 80)
        else:
            ctx.rectangle(xx, 700 - r_.uniform(20, 80), w, 100)
        ctx.fill()
        xx += w + r_.uniform(0, 40)
    cam.pop(ctx)
    fog_s = cached("fog", _fog_tex)
    cam.push(ctx, 4)
    paint_tiled(ctx, fog_s, -1200 - (t * 12) % 2048, 520, 2.0, 0.9, 0.4 * (1 - 0.4 * after))
    cam.pop(ctx)

    cam.push(ctx, 1.0)
    # water
    vgrad(ctx, 690, QUAY, [(0, (0.46, 0.5, 0.54)), (1, (0.2, 0.23, 0.27))], -800, W + 1200)
    ctx.save()
    ctx.set_operator(ADD)
    for k in range(40):
        yy = 694 + k * 2.4
        xo = float(vnoise(k * 0.9 + t * 0.6, 5)) * 60
        ctx.rectangle(sun[0] - 60 - k * 2 + xo, yy, 120 + k * 4, 1.2)
        ctx.set_source_rgba(1.0, 0.75, 0.55, 0.12 * (1 - k / 40) * (0.5 + after))
        ctx.fill()
    ctx.restore()

    # Ideal X: converted T2 tanker with a spar deck
    hull_col = (0.1, 0.1, 0.11)
    ctx.move_to(760, DECK - 20)
    ctx.line_to(2600, DECK - 6)
    ctx.line_to(2600, QUAY + 40)
    ctx.line_to(840, QUAY + 40)
    ctx.curve_to(800, 720, 770, 650, 760, DECK - 20)
    ctx.close_path()
    g = cairo.LinearGradient(0, DECK, 0, QUAY)
    g.add_color_stop_rgb(0, 0.13, 0.13, 0.15)
    g.add_color_stop_rgb(1, 0.04, 0.04, 0.05)
    ctx.set_source(g)
    ctx.fill()
    ctx.rectangle(800, QUAY - 22, 1800, 22)
    ctx.set_source_rgb(0.3, 0.08, 0.06)
    ctx.fill()
    ctx.rectangle(840, DECK - 18, 1500, 14)
    src(ctx, (0.08, 0.08, 0.09))
    ctx.fill()
    ctx.rectangle(780, DECK - 4, 1820, 4)
    ctx.set_source_rgba(0.9, 0.85, 0.8, 0.5)
    ctx.fill()
    # spar deck frame
    ctx.set_line_width(3)
    src(ctx, (0.18, 0.18, 0.2))
    for k in range(18):
        x0 = 860 + k * 70
        ctx.move_to(x0, DECK)
        ctx.line_to(x0, DECK - 16)
    ctx.stroke()
    # bridge house
    ctx.rectangle(1560, DECK - 150, 190, 150)
    src(ctx, (0.72, 0.72, 0.7))
    ctx.fill()
    ctx.rectangle(1540, DECK - 175, 230, 26)
    src(ctx, (0.62, 0.62, 0.6))
    ctx.fill()
    for k in range(6):
        ctx.rectangle(1575 + k * 29, DECK - 138, 18, 14)
        ctx.set_source_rgba(0.2, 0.22, 0.25, 1)
        ctx.fill()
    ctx.rectangle(1640, DECK - 330, 7, 160)
    src(ctx, (0.2, 0.2, 0.22))
    ctx.fill()
    ctx.set_line_width(1.2)
    ctx.move_to(1643, DECK - 330)
    ctx.line_to(1400, DECK - 10)
    ctx.move_to(1643, DECK - 330)
    ctx.line_to(1880, DECK - 10)
    ctx.stroke()
    # the boxes already aboard (58 in total by sailing day)
    slots = [880, 995, 1225, 1340, 1455]
    for k, sxb in enumerate(slots):
        draw_box(ctx, sxb, DECK - 18 - 50, 110, 50, k)
    # quay
    vgrad(ctx, QUAY, H + 400, [(0, (0.3, 0.3, 0.3)), (0.03, (0.13, 0.13, 0.14)), (1, (0.03, 0.03, 0.035))], -800, W + 1200)
    ctx.rectangle(-800, QUAY, W + 2000, 5)
    src(ctx, (0.6, 0.6, 0.58))
    ctx.fill()
    for k in range(9):
        ellipse(ctx, 120 + k * 260, QUAY + 4, 14, 8)
        src(ctx, (0.05, 0.05, 0.05))
        ctx.fill()
    # truck + trailer chassis
    ctx.rectangle(400, QUAY - 100, 270, 14)
    src(ctx, (0.08, 0.08, 0.09))
    ctx.fill()
    for wx in (440, 490, 640):
        ctx.arc(wx, QUAY - 20, 20, 0, 2 * math.pi)
        ctx.fill()
    ctx.move_to(680, QUAY - 40)
    ctx.line_to(680, QUAY - 150)
    ctx.curve_to(710, QUAY - 175, 770, QUAY - 175, 790, QUAY - 120)
    ctx.line_to(820, QUAY - 110)
    ctx.line_to(820, QUAY - 40)
    ctx.close_path()
    src(ctx, (0.3, 0.12, 0.1))
    ctx.fill()
    ctx.rectangle(702, QUAY - 158, 50, 34)
    ctx.set_source_rgba(0.55, 0.6, 0.65, 1)
    ctx.fill()
    for wx in (710, 795):
        ctx.arc(wx, QUAY - 20, 20, 0, 2 * math.pi)
        src(ctx, (0.05, 0.05, 0.05))
        ctx.fill()
    # workers watching
    from logistics.scenes.common import draw_walker
    for wx, fc in ((330, 1), (372, 1), (940, -1)):
        draw_walker(ctx, wx, QUAY + 2, 36, 0.0, (0.07, 0.07, 0.08), staff=False, facing=fc)

    # hammerhead crane
    steel = (0.14, 0.14, 0.16)
    ctx.set_line_width(4)
    src(ctx, steel)
    tx0, tx1 = 250, 320
    for xx in (tx0, tx1):
        ctx.move_to(xx, QUAY)
        ctx.line_to(xx, 300)
    for k in range(12):
        y0 = QUAY - k * 41
        ctx.move_to(tx0, y0)
        ctx.line_to(tx1, y0 - 41)
        ctx.move_to(tx1, y0)
        ctx.line_to(tx0, y0 - 41)
    ctx.stroke()
    jy0, jy1 = 270, 300
    ctx.set_line_width(4)
    ctx.move_to(40, jy0)
    ctx.line_to(1250, jy0)
    ctx.move_to(40, jy1)
    ctx.line_to(1250, jy1)
    ctx.stroke()
    ctx.set_line_width(2.5)
    for k in range(40):
        x0 = 40 + k * 30
        ctx.move_to(x0, jy1)
        ctx.line_to(x0 + 30, jy0)
    ctx.stroke()
    ctx.rectangle(40, 240, 120, 90)
    ctx.fill()
    ctx.rectangle(tx0 - 10, 300, 90, 60)
    ctx.fill()
    ctx.rectangle(tx0 + 10, 312, 30, 20)
    ctx.set_source_rgba(1.0, 0.8, 0.55, 0.9)
    ctx.fill()
    # trolley, cables, box
    trolley_x = bx + 55
    ctx.rectangle(trolley_x - 26, jy1, 52, 18)
    src(ctx, steel)
    ctx.fill()
    hook_x = trolley_x + (by - jy1) * math.sin(swing)
    ctx.set_line_width(1.6)
    src(ctx, (0.1, 0.1, 0.1))
    if lifted > 0 or bar >= 20.6:
        slack = 1 - clamp((bar - 20.6) / 0.4)
        top_y = jy1 + 18
        ctx.move_to(trolley_x - 8, top_y)
        ctx.curve_to(trolley_x - 8 + 20 * slack, (top_y + by) / 2, hook_x - 8, by - 60, hook_x - 8, by - 34)
        ctx.move_to(trolley_x + 8, top_y)
        ctx.curve_to(trolley_x + 8 + 20 * slack, (top_y + by) / 2, hook_x + 8, by - 60, hook_x + 8, by - 34)
        ctx.stroke()
        ctx.move_to(hook_x, by - 34)
        ctx.line_to(bx + 4, by)
        ctx.move_to(hook_x, by - 34)
        ctx.line_to(bx + 106, by)
        ctx.stroke()
    ctx.save()
    ctx.translate(bx + 55, by)
    ctx.rotate(swing * 0.6)
    draw_box(ctx, -55, 0, 110, 50, 99, hero=True, sun_x=sun[0] - bx)
    ctx.restore()
    cam.pop(ctx)

    # fog in front of everything, thinning after the landing
    cam.push(ctx, 1.4)
    paint_tiled(ctx, fog_s, -1600 + (t * 20) % 2048, 720, 2.6, 1.0, 0.22 * (1 - 0.6 * after))
    cam.pop(ctx)

    # gulls
    for k in range(6):
        gx = (200 + k * 330 + 40 * t * (1 + 0.2 * k)) % 2400 - 200
        gy = 330 + 60 * math.sin(k * 2.3) + 10 * math.sin(t * 0.7 + k)
        flap = math.sin(t * 7 + k * 1.3)
        sxg, syg = cam.project(gx, gy, 2.5)
        s = 9 * cam.scale(2.5)
        ctx.set_line_width(2)
        ctx.set_source_rgba(0.15, 0.16, 0.18, 0.9)
        ctx.move_to(sxg - s * 1.6, syg - s * flap * 0.8)
        ctx.curve_to(sxg - s * 0.7, syg - s * 0.8, sxg - s * 0.2, syg, sxg, syg + s * 0.2)
        ctx.curve_to(sxg + s * 0.2, syg, sxg + s * 0.7, syg - s * 0.8, sxg + s * 1.6, syg - s * flap * 0.8)
        ctx.stroke()

    # dust kicked up on landing
    if bar >= 24:
        age = (bar - 24) * 3.0
        base = cached("dust56", lambda: rng(57).random((500, 4)))
        ang = base[:, 0] * math.pi
        sp = 60 + 260 * base[:, 1]
        px = 1110 + 55 + np.cos(ang) * sp * age * np.sign(base[:, 2] - 0.5)
        py = DECK - 16 - np.abs(np.sin(ang)) * sp * age * 0.3 + 12 * age * age
        px, py = cam.project(px, py, 1.0)
        glow.points(px, py, (0.95, 0.9, 0.85), 0.45 * max(0.0, 1 - age / 1.6) ** 2, soft=True)
    return {"exposure": 1.0 + 0.12 * after}


def draw_box(ctx, x, y, w, h, k, hero=False, sun_x=300):
    shades = [(0.56, 0.58, 0.6), (0.5, 0.52, 0.55), (0.6, 0.6, 0.6), (0.52, 0.55, 0.58)]
    base = (0.66, 0.68, 0.7) if hero else shades[k % 4]
    ctx.rectangle(x, y, w, h)
    g = cairo.LinearGradient(x, 0, x + w, 0)
    g.add_color_stop_rgb(0, *[c * 0.7 for c in base])
    g.add_color_stop_rgb(1, *[min(1, c * (1.15 if sun_x > 0 else 0.9)) for c in base])
    ctx.set_source(g)
    ctx.fill()
    ctx.set_line_width(1.0)
    ctx.set_source_rgba(0.2, 0.2, 0.22, 0.55)
    for i in range(1, int(w / 6)):
        ctx.move_to(x + i * 6, y + 3)
        ctx.line_to(x + i * 6, y + h - 3)
    ctx.stroke()
    ctx.set_line_width(2.5)
    ctx.set_source_rgba(0.18, 0.18, 0.2, 1)
    ctx.rectangle(x, y, w, h)
    ctx.stroke()
    ctx.rectangle(x, y, w, 2)
    ctx.set_source_rgba(1, 0.92, 0.8, 0.6)
    ctx.fill()
