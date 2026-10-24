import hashlib
import io
import time
from collections import deque

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from matcher import ROOT, Matcher

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/seeds", StaticFiles(directory=ROOT / "seeds"), name="seeds")

matcher = Matcher()
cache: dict[str, dict] = {}
log: deque = deque(maxlen=100)
stats = {"checked": 0, "flagged": 0}
http = httpx.AsyncClient(timeout=10, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0 (Macintosh) AppleWebKit/537.36 Chrome/130"})


async def read_image(request: Request) -> tuple[bytes | None, str]:
    if request.headers.get("content-type", "").startswith("multipart/"):
        form = await request.form()
        return await form["image"].read(), form.get("url", "upload")
    url = (await request.json())["url"]
    try:
        r = await http.get(url)
        r.raise_for_status()
        return r.content, url
    except httpx.HTTPError:
        return None, url


@app.post("/match")
async def match(request: Request):
    t = time.time()
    data, url = await read_image(request)
    if data is None:
        return {"error": "fetch_failed"}
    key = hashlib.sha256(data).hexdigest()
    result = cache.get(key)
    if result is None:
        try:
            result = matcher.match(Image.open(io.BytesIO(data)))
        except Exception:
            return {"error": "bad_image"}
        cache[key] = result
        stats["checked"] += 1
        stats["flagged"] += result["match"]
    ms = round((time.time() - t) * 1000)
    log.appendleft({"url": url, "match": result["match"], "id": result.get("id"),
                    "score": result["score"], "ms": ms})
    return result


@app.get("/log")
def get_log():
    return list(log)


@app.get("/stats")
def get_stats():
    return stats


@app.get("/health")
def health():
    return {"ok": True, "entries": len(matcher.entries)}
