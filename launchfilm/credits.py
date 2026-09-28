"""CREDITS.csv / CREDITS.md generated from sources.yaml + the uses in cuesheet.json."""
import csv
import re

from launchfilm import audio, narrate
from launchfilm.config import PKG, Cues, load_sources


def _tc(s):
    m, sec = divmod(max(0.0, s), 60)
    return f"{int(m):02d}:{sec:06.3f}"


def rows():
    cs = Cues()
    src = load_sources()
    out = []

    def add(kind, sid, t_in, t_out, film0, film1, note):
        s = src[sid]
        out.append({
            "use": kind,
            "clip_id": sid,
            "title": s["title"],
            "date": s.get("date", ""),
            "creator_collection": s["creator"],
            "source_url": s["page"],
            "file_url": s["url"],
            "license": s["license"],
            "rights_statement_quoted": re.sub(r"\s+", " ", s["rights"]).strip(),
            "date_accessed": s["accessed"],
            "source_in": _tc(t_in),
            "source_out": _tc(t_out),
            "film_in": _tc(film0),
            "film_out": _tc(film1),
            "where_in_film": note,
        })

    from launchfilm.picture import shot_speed
    shots = cs["shots"]
    for i, s in enumerate(shots):
        t0 = cs.when(s)
        t1 = cs.when(shots[i + 1]) if i + 1 < len(shots) else cs.duration
        sid = s["src"]
        if sid in (None, "black"):
            continue
        speed = shot_speed(s, cs.fps, src)
        note = "picture, dimmed under the logo and end card" if s.get("closing") else \
            f"picture: {s.get('note', '')}".rstrip(": ")
        if s.get("blur"):
            note += " (markings blurred)"
        add("footage", sid, s["in"], s["in"] + (t1 - t0) * speed, t0, t1, note)
    nc = cs["narration"]
    dec = lambda p, sr: audio.decode_file(p, sr).mean(axis=1)
    for line in nc["lines"]:
        for t0, x in narrate.placements(line, nc, decode=dec):
            d = len(x) / 48000 - narrate.LEAD_S
            add("narration", nc["model"], 0.0, d, t0, t0 + d,
                f"narration line {line['id']} (synthetic voice): \u201c{line['text']}\u201d")
    m = cs["music"]
    ms = src[m["src"]]
    for seg in m["segments"]:
        a, b = audio.segment_src(seg, ms)
        t0 = cs.t(seg["at"])
        t1 = min(cs.duration, t0 + b - a)
        add("music", m["src"], a, a + t1 - t0, t0, t1, f"music: {seg.get('note', 'score')}")
    for e in cs["sfx"]:
        t = cs.when(e)
        add("sfx", e["src"], e["in"], e["out"], t, t + e["out"] - e["in"], "sound effect / natural sound")
    return out


def write():
    rs = rows()
    csv_path = PKG / "CREDITS.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
        w.writeheader()
        w.writerows(rs)
    src = load_sources()
    lines = ["# Credits: Volume launch film (v3)", "",
             "Every clip, voice line, music cue and sound effect in the film, generated from "
             "`sources.yaml` and the uses in `cuesheet.json` by `python3 -m launchfilm.render --credits`. "
             "Times are mm:ss.sss; `film` columns are positions in the 16:9 cut (the vertical cut "
             "uses the same timeline).", ""]
    ms = src[cs["music"]["src"]]
    lines += ["## Required attribution (music, CC BY 4.0)", "",
              f"> {ms['attribution']}", "",
              "Include this line in the post text or description wherever the film is published.", ""]
    stock = sorted({src[r["clip_id"]]["creator"] for r in rs if src[r["clip_id"]]["license"].startswith("Pexels")})
    if stock:
        lines += ["## Stock footage (Pexels License: attribution not required, given as courtesy)", "",
                  ", ".join(stock), ""]
    lines += ["## Sources", ""]
    used = {r["clip_id"] for r in rs}
    for sid, s in src.items():
        if sid not in used:
            continue
        lines += [f"### {s['title']} ({s.get('date', '')})".replace(" ()", ""), "",
                  f"- id: `{sid}`",
                  f"- creator / collection: {s['creator']}",
                  f"- item page: {s['page']}",
                  f"- license: {s['license']}",
                  f"- rights statement: {re.sub(chr(10), ' ', s['rights']).strip()}",
                  f"- accessed: {s['accessed']}", ""]
    lines += ["## Every use, in film order", "",
              "| # | use | clip | source in-out | film in-out | where |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(sorted(rs, key=lambda r: r["film_in"]), 1):
        lines.append(f"| {i} | {r['use']} | `{r['clip_id']}` | {r['source_in']}-{r['source_out']} | "
                     f"{r['film_in']}-{r['film_out']} | {r['where_in_film'].replace('|', '/')} |")
    (PKG / "CREDITS.md").write_text("\n".join(lines) + "\n")
    print("credits ->", csv_path, PKG / "CREDITS.md", len(rs), "rows")


if __name__ == "__main__":
    write()
