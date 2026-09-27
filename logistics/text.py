"""Dated title cards, timed from the cue sheet."""
import cairo
import cv2
import numpy as np

from logistics.gfx import W, H, new_surface, clamp, smooth, ease_out

SERIF = "Noto Serif Display"
SERIF_BODY = "Noto Serif"
SANS = "Inter"
WHITE = (1.0, 0.97, 0.92)
GOLD = (0.96, 0.8, 0.52)


def _font(ctx, family, size, italic=False, bold=False):
    ctx.select_font_face(family, cairo.FONT_SLANT_ITALIC if italic else cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(size)


def tracked(ctx, text, cx, y, tracking=0.0, col=WHITE, a=1.0, align="center"):
    adv = [ctx.text_extents(ch).x_advance for ch in text]
    total = sum(adv) + tracking * (len(text) - 1)
    x = cx - total / 2 if align == "center" else cx
    ctx.set_source_rgba(*col, a)
    for ch, ad in zip(text, adv):
        ctx.move_to(x, y)
        ctx.show_text(ch)
        x += ad + tracking
    return total


def _env(t, t0, dur, fin=0.7, fout=0.6):
    return smooth((t - t0) / fin) * smooth((t0 + dur - t) / fout)


def render_titles(cs, t):
    active = [ti for ti in cs.titles if cs.t(ti["at"]) <= t < cs.t(ti["at"] + ti["dur"])]
    if not active:
        return None
    surf, ctx = new_surface()
    blur = 0.0
    for ti in active:
        t0 = cs.t(ti["at"])
        dur = ti["dur"] * cs.bar_s
        p = (t - t0) / dur
        e = _env(t, t0, dur)
        blur = max(blur, 7.0 * (1 - smooth((t - t0) / 0.8)) ** 2 + 5.0 * (1 - smooth((t0 + dur - t) / 0.6)) ** 2)
        st = ti["style"]
        scrim_y = {"card": 540, "line": 860, "quote": 555}.get(st)
        if scrim_y:
            ctx.save()
            ctx.translate(W / 2, scrim_y)
            ctx.scale(760, 150 if st != "line" else 70)
            g = cairo.RadialGradient(0, 0, 0, 0, 0, 1)
            g.add_color_stop_rgba(0, 0, 0, 0, 0.42 * e)
            g.add_color_stop_rgba(1, 0, 0, 0, 0)
            ctx.set_source(g)
            ctx.arc(0, 0, 1, 0, 2 * 3.14159265)
            ctx.fill()
            ctx.restore()
        if st == "quote":
            for i, line in enumerate(ti["lines"]):
                ts = t0 + i * dur * 0.38
                ei = _env(t, ts, t0 + dur - ts, 0.9, 0.7)
                _font(ctx, SERIF, 50, italic=True)
                tracked(ctx, line, W / 2, 520 + i * 72 - 6 * ease_out((t - ts) / 2.0), 1.5, WHITE, ei)
        elif st == "card":
            drift = -10 * ease_out(p * 1.4)
            _font(ctx, SANS, 22)
            date = ti["date"].upper()
            wdate = tracked(ctx, date, W / 2, 452 + drift, 9.0, GOLD, e)
            rule = 120 * ease_out((t - t0) / 1.2)
            ctx.set_source_rgba(*GOLD, 0.7 * e)
            ctx.set_line_width(1.2)
            for sgn in (-1, 1):
                x0 = W / 2 + sgn * (wdate / 2 + 28)
                ctx.move_to(x0, 444 + drift)
                ctx.line_to(x0 + sgn * rule, 444 + drift)
                ctx.stroke()
            _font(ctx, SERIF, 104)
            tracked(ctx, ti["title"], W / 2, 572 + drift, 14 + 16 * p, WHITE, e)
            if ti.get("sub"):
                es = _env(t, t0 + 0.45, dur - 0.45)
                _font(ctx, SERIF_BODY, 30, italic=True)
                tracked(ctx, ti["sub"], W / 2, 634 + drift, 0.5, (0.9, 0.88, 0.84), es * 0.95)
        elif st == "line":
            _font(ctx, SERIF, 42, italic=True)
            for i, line in enumerate(ti["lines"]):
                tracked(ctx, line, W / 2, 872 + i * 54 - 5 * ease_out(p * 2), 1.0, WHITE, e)
        elif st == "final":
            ef = smooth((t - t0) / 1.6) * smooth((t0 + dur - t) / 1.2)
            _font(ctx, SERIF, 92)
            tracked(ctx, ti["title"], W / 2, 560, 10 + 10 * p, WHITE, ef)
            es = smooth((t - t0 - 1.2) / 1.2) * smooth((t0 + dur - t) / 1.2)
            ctx.set_source_rgba(*GOLD, 0.8 * es)
            ctx.set_line_width(1.2)
            wr = 180 * ease_out((t - t0 - 1.0) / 2.0)
            ctx.move_to(W / 2 - wr, 606)
            ctx.line_to(W / 2 + wr, 606)
            ctx.stroke()
            _font(ctx, SANS, 24)
            tracked(ctx, ti["sub"], W / 2, 652, 12.0, GOLD, es)
    surf.flush()
    buf = np.ndarray((H, W, 4), np.uint8, surf.get_data())
    rgba = buf[..., [2, 1, 0, 3]].astype(np.float32) / 255.0
    if blur > 0.3:
        rgba = cv2.GaussianBlur(rgba, (0, 0), blur)
    glow = cv2.GaussianBlur(cv2.resize(rgba, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 5)
    glow = cv2.resize(glow, (W, H))
    rgb = rgba[..., :3] + glow[..., :3] * 0.35
    a = np.clip(rgba[..., 3] + glow[..., 3] * 0.15, 0, 1)
    return rgb, a
