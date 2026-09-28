"""Footage scouting: search public-domain / free-license libraries and preview clips
without downloading them.

    python3 -m launchfilm.scout dvids "shipyard hull" [pages]    search DVIDS videos
    python3 -m launchfilm.scout dvinfo 891556                    item page -> original MP4, rights, credit
    python3 -m launchfilm.scout nasa "clean room"                 NASA Image and Video Library videos
    python3 -m launchfilm.scout commons "container ship"          Wikimedia Commons videos + license
    python3 -m launchfilm.scout pexels "power substation"         Pexels video search (ids)
    python3 -m launchfilm.scout pxinfo 10008320                   Pexels title, creator, file URL, size
    python3 -m launchfilm.scout sheet URL out.jpg [n] [t0] [t1]   n frames over a remote/local file
"""
import html
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def probe(url):
    r = subprocess.run(["ffprobe", "-v", "error", "-user_agent", UA, "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,r_frame_rate:format=duration", "-of", "json", url],
                       capture_output=True, text=True, timeout=90)
    try:
        j = json.loads(r.stdout)
        s = j["streams"][0]
        return {"w": s["width"], "h": s["height"], "fps": s["r_frame_rate"],
                "dur": round(float(j["format"]["duration"]), 1)}
    except Exception:
        return {}


def dvids(q, pages=1):
    out = []
    for p in range(1, pages + 1):
        u = ("https://www.dvidshub.net/search/?q=" + urllib.parse.quote(q) +
             "&filter%5Btype%5D=video&sort=relevance&page=" + str(p))
        s = get(u)
        for vid, slug in re.findall(r'href="/video/(\d+)/([^"]+)"', s):
            if vid not in [o[0] for o in out]:
                out.append((vid, slug))
    return out


def dvinfo(vid):
    s = get(f"https://www.dvidshub.net/video/{vid}")
    title = html.unescape((re.search(r'og:title" content="([^"]*)"', s) or [None, ""])[1])
    desc = html.unescape((re.search(r'og:description" content="([^"]*)"', s) or [None, ""])[1])
    mp4s = sorted(set(re.findall(r'https://d34w7g4gy10iej\.cloudfront\.net/video/[^"\s]+?\.mp4', s)))
    orig = None
    for m in mp4s:
        base = re.sub(r"-\d+x\d+-\d+k\.mp4$", ".mp4", m)
        orig = base
    text = re.sub(r"<[^>]+>", " ", s)
    text = re.sub(r"\s+", " ", html.unescape(text))
    date = (re.search(r"Date Taken:\s*([0-9.]+)", text) or [None, ""])[1]
    posted = (re.search(r"Date Posted:\s*([0-9.]+)", text) or [None, ""])[1]
    unit = (re.search(r"Photographer's Name:\s*([^|]+?)\s+Location", text) or [None, ""])[1]
    loc = (re.search(r"Location:\s*(.+?)\s+(?:Web Views|Downloads|Podcast)", text) or [None, ""])[1]
    pd = "PUBLIC DOMAIN" in s
    info = {"id": vid, "title": title, "date": date or posted, "credit": unit.strip(), "location": loc.strip(),
            "public_domain": pd, "url": orig, "page": f"https://www.dvidshub.net/video/{vid}", "desc": desc}
    if orig:
        info.update(probe(orig))
    return info


def nasa(q, n=40):
    j = json.loads(get("https://images-api.nasa.gov/search?media_type=video&q=" + urllib.parse.quote(q)))
    out = []
    for it in j["collection"]["items"][:n]:
        d = it["data"][0]
        out.append({"nasa_id": d["nasa_id"], "title": d.get("title"), "date": d.get("date_created", "")[:10],
                    "center": d.get("center"), "desc": (d.get("description") or "")[:240],
                    "assets": it["href"]})
    return out


def nasa_files(nasa_id):
    j = json.loads(get("https://images-api.nasa.gov/asset/" + urllib.parse.quote(nasa_id)))
    return [i["href"] for i in j["collection"]["items"]]


def commons(q, n=50):
    api = "https://commons.wikimedia.org/w/api.php?"
    params = {"action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
              "gsrsearch": q + " filetype:video", "gsrlimit": n, "prop": "imageinfo",
              "iiprop": "url|size|extmetadata|mediatype"}
    j = json.loads(get(api + urllib.parse.urlencode(params)))
    out = []
    for p in (j.get("query", {}).get("pages", {}) or {}).values():
        ii = p["imageinfo"][0]
        md = ii.get("extmetadata", {})
        g = lambda k: re.sub(r"<[^>]+>", "", md.get(k, {}).get("value", ""))[:160]
        out.append({"title": p["title"], "w": ii.get("width"), "h": ii.get("height"),
                    "dur": ii.get("duration"), "license": g("LicenseShortName"), "artist": g("Artist"),
                    "credit": g("Credit"), "url": ii["url"], "page": ii["descriptionurl"]})
    return sorted(out, key=lambda o: -(o["h"] or 0))


def pexels(q):
    """Search results via the r.jina.ai reader (pexels.com itself answers 403 to scripts)."""
    s = get("https://r.jina.ai/https://www.pexels.com/search/videos/" + urllib.parse.quote(q) + "/", 90)
    return sorted(set(re.findall(r"pexels\.com/video/([a-z0-9-]+)-(\d+)/", s)), key=lambda t: t[1])


def pxfile(pid):
    """Direct file behind Pexels' own download endpoint (a 302 to videos.pexels.com)."""
    req = urllib.request.Request(f"https://www.pexels.com/download/video/{pid}/", headers={"User-Agent": UA},
                                 method="HEAD")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    try:
        urllib.request.build_opener(NoRedirect).open(req, timeout=30)
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            return e.headers["Location"]
        raise
    return None


def pxinfo(slug_id):
    """`slug_id` is the page path, e.g. transformers-at-an-electrical-station-10008320."""
    pid = slug_id.rsplit("-", 1)[-1]
    s = get(f"https://r.jina.ai/https://www.pexels.com/video/{slug_id}/", 90)
    title = (re.search(r"^Title:\s*(.+)$", s, re.M) or [None, ""])[1]
    url = pxfile(pid)
    info = {"id": pid, "title": title.replace(" Free Stock Video Footage, Royalty-Free 4K & HD Video Clip", ""),
            "page": f"https://www.pexels.com/video/{slug_id}/", "url": url}
    m = re.search(r"#####\s*([^\n]+)\n", s)
    info["creator"] = m[1].strip() if m else ""
    if url:
        info.update(probe(url))
    return info


def sheet(url, out, n=12, t0=None, t1=None, tw=480):
    dur = probe(url).get("dur") or 60
    t0 = 0.5 if t0 is None else float(t0)
    t1 = dur - 0.5 if t1 is None else float(t1)
    times = np.linspace(t0, t1, int(n))
    th = tw * 9 // 16
    cols = 4
    rows = (len(times) + cols - 1) // cols
    g = Image.new("RGB", (cols * tw, rows * (th + 18)), (20, 20, 20))
    d = ImageDraw.Draw(g)
    for i, t in enumerate(times):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-user_agent", UA, "-ss", f"{t:.2f}", "-i", url,
                              "-frames:v", "1", "-vf", f"scale={tw}:{th}:force_original_aspect_ratio=decrease,"
                              f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, timeout=120).stdout
        x, y = (i % cols) * tw, (i // cols) * (th + 18)
        if len(raw) == tw * th * 3:
            g.paste(Image.frombytes("RGB", (tw, th), raw), (x, y))
        d.text((x + 4, y + th + 2), f"{t:.1f}s", fill=(230, 200, 140))
    g.save(out, quality=85)
    print(out)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "dvids":
        for vid, slug in dvids(args[0], int(args[1]) if len(args) > 1 else 1):
            print(vid, slug)
    elif cmd == "dvinfo":
        for v in args:
            print(json.dumps(dvinfo(v), indent=1))
    elif cmd == "nasa":
        for o in nasa(args[0]):
            print(json.dumps(o))
    elif cmd == "nasafiles":
        print("\n".join(nasa_files(args[0])))
    elif cmd == "commons":
        for o in commons(args[0]):
            print(json.dumps(o))
    elif cmd == "pexels":
        for slug, pid in pexels(args[0]):
            print(pid, slug)
    elif cmd == "pxinfo":
        for v in args:
            print(json.dumps(pxinfo(v), indent=1))
    elif cmd == "sheet":
        sheet(*args)
