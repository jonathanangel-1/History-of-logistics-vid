"""Figures and props reused across scenes (all drawn procedurally)."""
import math

import cairo
import numpy as np

from logistics.gfx import W, H, CX, CY, ellipse, src, cached, rng


def world_span(cam, depth, pad=120):
    s = cam.scale(depth)
    cx = CX + cam.x / depth
    half = (W / 2 + pad) / s / max(0.2, math.cos(cam.roll))
    return cx - half - abs(math.sin(cam.roll)) * H, cx + half + abs(math.sin(cam.roll)) * H


def camel_path(ctx, phase, load=True, rider=False, outline=False):
    """Camel in profile facing +x, ground at y=0, height ~1.9 units. Pacing gait."""
    bob = 0.03 * math.sin(2 * phase)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    legs = []
    for hx, side, bend in ((0.36, 0.0, 1), (0.30, math.pi, 1), (-0.34, 0.0, -1), (-0.40, math.pi, -1)):
        ph = phase + side
        hy = -1.05 + bob
        fx = hx + 0.2 * math.sin(ph)
        fy = -0.1 * max(0.0, math.cos(ph)) ** 2
        mx, my = (hx + fx) / 2, (hy + fy) / 2
        kx = mx + bend * (0.07 + 0.05 * max(0.0, math.cos(ph)))
        legs.append(((hx, hy), (kx, my + 0.04), (fx, fy)))
    body = []
    # body + hump
    ctx.new_sub_path()
    ellipse(ctx, 0.0, -1.2 + bob, 0.55, 0.23)
    body.append("e")
    ctx.move_to(-0.42, -1.25 + bob)
    ctx.curve_to(-0.3, -1.62 + bob, 0.02, -1.8 + bob, 0.18, -1.42 + bob)
    ctx.line_to(0.3, -1.25 + bob)
    ctx.close_path()
    # head
    hb = 0.02 * math.sin(2 * phase + 0.6)
    ellipse(ctx, 0.95, -1.63 + bob + hb, 0.14, 0.065, 0.25)
    ellipse(ctx, 0.84, -1.66 + bob + hb, 0.06, 0.06)
    if load:
        ellipse(ctx, -0.16, -1.38 + bob, 0.3, 0.2)
        ctx.rectangle(-0.42, -1.4 + bob, 0.18, 0.32)
        ctx.rectangle(0.06, -1.42 + bob, 0.16, 0.28)
    if rider:
        poly_pts = [(-0.24, -1.62 + bob), (0.1, -1.62 + bob), (0.02, -2.02 + bob), (-0.1, -2.02 + bob)]
        ctx.move_to(*poly_pts[0])
        for p in poly_pts[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ellipse(ctx, -0.04, -2.1 + bob, 0.075, 0.08)
        ellipse(ctx, -0.04, -2.16 + bob, 0.09, 0.04)
    if outline:
        return legs, bob, hb
    return legs, bob, hb


def draw_camel(ctx, x, y, s, phase, col, a=1.0, load=True, rider=False, ang=0.0):
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(ang)
    ctx.scale(s, s)
    ctx.new_path()
    legs, bob, hb = camel_path(ctx, phase, load, rider)
    src(ctx, col, a)
    ctx.fill()
    ctx.set_line_width(0.075)
    for (h, k, f) in legs:
        ctx.move_to(*h)
        ctx.line_to(*k)
        ctx.line_to(*f)
    ctx.stroke()
    ctx.set_line_width(0.15)
    ctx.move_to(0.4, -1.22 + bob)
    ctx.curve_to(0.68, -1.12 + bob, 0.7, -1.55 + bob + hb, 0.86, -1.64 + bob + hb)
    ctx.stroke()
    ctx.set_line_width(0.03)
    ctx.move_to(-0.53, -1.24 + bob)
    ctx.curve_to(-0.6, -1.15 + bob, -0.58, -1.05 + bob, -0.6, -0.98 + bob)
    ctx.stroke()
    ctx.restore()


def camel_outline_points(phase, n=260, rider=False):
    """Points along the camel silhouette, for the drone-swarm callback."""
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 4, 4)
    ctx = cairo.Context(surf)
    ctx.set_tolerance(0.002)
    legs, bob, hb = camel_path(ctx, phase, True, rider)
    pts = []
    path = ctx.copy_path_flat()
    cur = None
    start = None
    for kind, p in path:
        if kind == cairo.PATH_MOVE_TO:
            cur = p
            start = p
        elif kind == cairo.PATH_LINE_TO and cur is not None:
            pts.append((cur, p))
            cur = p
        elif kind == cairo.PATH_CLOSE_PATH and cur is not None and start is not None:
            pts.append((cur, start))
            cur = start
    for (h, k, f) in legs:
        pts.append((h, k))
        pts.append((k, f))
    neck = [(0.4, -1.22 + bob), (0.68, -1.12 + bob), (0.7, -1.55 + bob + hb), (0.86, -1.64 + bob + hb)]
    tt = np.linspace(0, 1, 12)
    bez = [((1 - u) ** 3 * np.array(neck[0]) + 3 * (1 - u) ** 2 * u * np.array(neck[1])
            + 3 * (1 - u) * u * u * np.array(neck[2]) + u ** 3 * np.array(neck[3])) for u in tt]
    for i in range(len(bez) - 1):
        pts.append((tuple(bez[i]), tuple(bez[i + 1])))
    seg = np.array([[a[0], a[1], b[0], b[1]] for a, b in pts])
    L = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    cum = np.concatenate([[0], np.cumsum(L)])
    d = np.linspace(0, cum[-1], n, endpoint=False) + cum[-1] / n * 0.5
    idx = np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(L) - 1)
    u = (d - cum[idx]) / np.maximum(L[idx], 1e-9)
    x = seg[idx, 0] + (seg[idx, 2] - seg[idx, 0]) * u
    y = seg[idx, 1] + (seg[idx, 3] - seg[idx, 1]) * u
    return x, y


def draw_walker(ctx, x, y, s, phase, col, a=1.0, staff=True, facing=1):
    ctx.save()
    ctx.translate(x, y)
    ctx.scale(s * facing, s)
    src(ctx, col, a)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_width(0.09)
    for side in (0, math.pi):
        ph = phase + side
        ctx.move_to(0, -0.9)
        ctx.line_to(0.18 * math.sin(ph), -0.02 * max(0, math.cos(ph)))
    ctx.stroke()
    ctx.move_to(-0.16, -0.5)
    ctx.line_to(0.14, -0.5)
    ctx.line_to(0.08, -1.45)
    ctx.line_to(-0.1, -1.45)
    ctx.close_path()
    ctx.fill()
    ellipse(ctx, 0.0, -1.58, 0.1, 0.12)
    ctx.fill()
    if staff:
        ctx.set_line_width(0.035)
        ctx.move_to(0.25 + 0.03 * math.sin(phase), -1.75)
        ctx.line_to(0.32, 0.0)
        ctx.stroke()
    ctx.restore()


def stars(glow, seed, n, y0, y1, t, bright=0.6, cam=None, depth=40.0, twinkle=3.0):
    base = cached(("stars", seed, n), lambda: rng(seed).random((n, 4)))
    x = base[:, 0] * W * 1.3 - W * 0.15
    y = y0 + base[:, 1] * (y1 - y0)
    if cam is not None:
        x, y = cam.project(x, y, depth)
    tw = 0.6 + 0.4 * np.sin(t * twinkle * (0.5 + base[:, 2]) + base[:, 3] * 40)
    inten = bright * (base[:, 2] ** 3) * tw * (1 - (y - y0) / (y1 - y0 + 1e-6)).clip(0, 1) ** 0.7
    glow.points(x, y, (0.85, 0.9, 1.0), inten)


def lightning_bolt(ctx, x0, y0, x1, y1, seed, width=3.0, a=1.0, depth_levels=6):
    r = rng(seed)
    pts = [(x0, y0), (x1, y1)]
    disp = abs(y1 - y0) * 0.25
    for lv in range(depth_levels):
        new = [pts[0]]
        for i in range(len(pts) - 1):
            (ax, ay), (bx, by) = pts[i], pts[i + 1]
            mx = (ax + bx) / 2 + r.normal(0, disp)
            my = (ay + by) / 2 + r.normal(0, disp * 0.2)
            new += [(mx, my), (bx, by)]
        pts = new
        disp *= 0.55
    branches = []
    for _ in range(3):
        k = r.integers(len(pts) // 4, len(pts) * 3 // 4)
        bx, by = pts[k]
        ex = bx + r.normal(0, 140)
        ey = by + r.uniform(80, 260)
        branches.append((bx, by, ex, ey))
    for lw, al in ((width * 8, 0.06), (width * 3, 0.25), (width, 1.0)):
        ctx.set_line_width(lw)
        ctx.set_source_rgba(0.85, 0.9, 1.0, al * a)
        ctx.move_to(*pts[0])
        for p in pts[1:]:
            ctx.line_to(*p)
        ctx.stroke()
    return branches
