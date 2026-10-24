"""Precision / recall / latency on reposted seeds vs normal images.

Positives are built from seeds with transforms that differ from the index-time
augmentations (random crops, rotation, color shifts, overlaid text, UI chrome).
Negatives are real photos in test/negatives/, many of them the same subject as the fakes.
Wild copies in test/wild/<seed-id>/ are real reposts found online (screenshots, crops,
fact-check frames). test/wild_originals/<seed-id>/ are real copies of the untouched photo.
test/coco/ is COCO val2017 as a large set of everyday negatives. --quick uses 500 of them.
"""
import io
import json
import random
import sys
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


def load(folder):
    return [(p, Image.open(p)) for p in sorted(folder.glob("*/*.jpg"))]


def wild(m):
    copies = load(ROOT / "test" / "wild")
    hits = 0
    for p, img in copies:
        r = m.match(img)
        if r["match"] and r.get("id") == p.parent.name:
            hits += 1
        else:
            print(f"  miss {p.parent.name}/{p.name}: score {r['score']} {r.get('method')} {r.get('id', '')}")
    print(f"wild recall: {hits}/{len(copies)} matched to the right seed")
    reals = load(ROOT / "test" / "wild_originals")
    flagged = [f"{p.parent.name}/{p.name}" for p, img in reals if m.match(img)["match"]]
    print(f"wild originals flagged: {len(flagged)}/{len(reals)} {flagged}")


def coco(m, neg):
    paths = sorted((ROOT / "test" / "coco").glob("*.jpg"))
    if "--quick" in sys.argv:
        paths = random.Random(7).sample(paths, min(500, len(paths)))
    res, _ = scores(m, (Image.open(p) for p in paths))
    flagged = [p.name for p, r in zip(paths, res) if r[0]]
    print(f"coco false flags: {len(flagged)}/{len(paths)} {flagged[:10]}")
    top = sorted(((r[1], n) for r, n in zip(res + neg[0], [p.name for p in paths] + neg[1])), reverse=True)[:5]
    print(f"highest scores across negatives + coco: {top}")


def main():
    m = M.Matcher()
    seeds = [e for e in m.entries if e["kind"] != "original"]
    positives = [repost(Image.open(ROOT / "seeds" / e["image"])) for e in seeds for _ in range(3)]
    neg_paths = sorted((ROOT / "test" / "negatives").glob("*.jpg"))
    negatives = [Image.open(p) for p in neg_paths]
    neg_names = [p.name for p in neg_paths]

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
    pairs = [e for e in seeds if e.get("has_original")]
    fakes = [m.match(repost(Image.open(ROOT / "seeds" / e["image"])))["match"] for e in pairs for _ in range(10)]
    reals = [m.match(repost(Image.open(ROOT / "seeds" / e["id"] / "original.jpg")))["match"] for e in pairs for _ in range(10)]
    print(f"edited pairs: fake reposts flagged {sum(fakes)}/{len(fakes)}, real-photo reposts flagged {sum(reals)}/{len(reals)}")
    print(f"latency median {np.median(pos_ms + neg_ms):.0f} ms, p95 {np.percentile(pos_ms + neg_ms, 95):.0f} ms")
    wild(m)
    coco(m, (neg, neg_names))


if __name__ == "__main__":
    main()
