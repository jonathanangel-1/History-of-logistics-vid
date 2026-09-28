"""Download every source in sources.yaml into cache/sources/ (idempotent)."""
import subprocess
import sys
import time
import urllib.request

from launchfilm.config import SRC_DIR, load_sources, src_path

UA = "Mozilla/5.0 (X11; Linux x86_64) volume-launch-film-fetch/2.0"


def _download(url, dest):
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.rename(dest)
            return
        except Exception as e:  # network hiccups: back off 4, 8, 16, 32 s
            if attempt == 4:
                raise
            wait = 4 * 2 ** attempt
            print(f"  retry in {wait}s ({e})")
            time.sleep(wait)


def _segment(url, dest, t0, t1):
    tmp = dest.with_name(dest.stem + ".part" + dest.suffix)
    cmd = ["ffmpeg", "-v", "error", "-y", "-user_agent", UA, "-ss", str(t0), "-i", url,
           "-t", str(t1 - t0), "-c:v", "libx264", "-crf", "14", "-preset", "veryfast",
           "-c:a", "flac", str(tmp)]
    subprocess.run(cmd, check=True)
    tmp.rename(dest)


def fetch(only=None):
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    sources = load_sources()
    for sid, s in sources.items():
        if only and sid not in only:
            continue
        dest = src_path(sid, sources)
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"fetch {sid} <- {s['url'][:90]}")
        if "segment" in s:
            _segment(s["url"], dest, *s["segment"])
        else:
            _download(s["url"], dest)
    print("sources ready in", SRC_DIR)


if __name__ == "__main__":
    fetch(set(sys.argv[1:]) or None)
