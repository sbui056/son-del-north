"""Precision / recall / latency on reposted seeds vs normal images.

Positives are built from seeds with transforms that differ from the index-time
augmentations (random crops, rotation, color shifts, overlaid text, UI chrome).
Negatives are real photos in test/negatives/, many of them the same subject as the fakes.
"""
import io
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

import matcher as M

ROOT = Path(__file__).resolve().parent.parent
rng = random.Random(7)


def repost(img):
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    f = rng.uniform(0.6, 0.9)
    x, y = rng.uniform(0, 1 - f) * w, rng.uniform(0, 1 - f) * h
    img = img.crop((int(x), int(y), int(x + f * w), int(y + f * h)))
    img = img.rotate(rng.uniform(-3, 3), expand=False, fillcolor=(255, 255, 255))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.6, 1.4))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.8, 1.2))
    w, h = img.size
    d = ImageDraw.Draw(img)
    d.rectangle((0, int(h * 0.85), w, h), fill=(0, 0, 0))
    d.text((10, int(h * 0.88)), "you won't believe this", fill=(255, 255, 255))
    if rng.random() < 0.5:
        frame = Image.new("RGB", (w + 40, int(h * 1.6)), (250, 250, 250))
        frame.paste(img, (20, int(h * 0.25)))
        img = frame
    img = img.resize((max(1, img.width * 400 // img.height), 400))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(30, 70))
    return Image.open(buf)


def scores(m, images):
    out, ms = [], []
    for img in images:
        t = time.time()
        r = m.match(img)
        ms.append((time.time() - t) * 1000)
        out.append((r["match"], r["score"], r.get("method")))
    return out, ms


def main():
    m = M.Matcher()
    seeds = [e for e in m.entries if e["kind"] != "original"]
    positives = [repost(Image.open(ROOT / "seeds" / e["image"])) for e in seeds for _ in range(3)]
    negatives = [Image.open(p) for p in sorted((ROOT / "test" / "negatives").glob("*.jpg"))]

    pos, pos_ms = scores(m, positives)
    neg, neg_ms = scores(m, negatives)

    print(f"{len(seeds)} seeds, {len(positives)} reposted positives, {len(negatives)} negatives")
    print("threshold  recall  false_flags  (embedding score; phash hits always count)")
    for t in np.arange(0.40, 0.86, 0.05):
        hit = lambda r: r[2] == "phash" or (r[0] or r[2] is None) and r[1] >= t
        tp = sum(hit(r) for r in pos)
        fp = sum(hit(r) for r in neg)
        print(f"  {t:.2f}     {tp}/{len(pos)}    {fp}/{len(neg)}")
    tp = sum(r[0] for r in pos)
    fp = sum(r[0] for r in neg)
    neg_top = sorted((r[1] for r in neg), reverse=True)[:5]
    print(f"current threshold {M.EMBED_MIN}: recall {tp}/{len(pos)}, false flags {fp}/{len(neg)}")
    print(f"highest negative scores: {neg_top}")
    print(f"latency median {np.median(pos_ms + neg_ms):.0f} ms, p95 {np.percentile(pos_ms + neg_ms, 95):.0f} ms")


if __name__ == "__main__":
    main()
