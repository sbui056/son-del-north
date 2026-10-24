"""seeds/ -> index/ (SSCD vectors + pHashes for every variant, plus entry metadata)."""
import json
import time
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from augment import variants
from embed import embed

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ROOT / "seeds"
INDEX = ROOT / "index"


def load_entries():
    entries = []
    for d in sorted(p for p in SEEDS.iterdir() if (p / "meta.json").exists()):
        meta = json.loads((d / "meta.json").read_text())
        if not meta.get("source_url") or not (d / "image.jpg").exists():
            print(f"skip {d.name}: missing source or image")
            continue
        has_original = (d / "original.jpg").exists()
        entries.append({**meta, "id": d.name, "image": f"{d.name}/image.jpg", "has_original": has_original})
        if has_original:
            entries.append({"id": f"{d.name}/original", "kind": "original", "parent": d.name,
                            "image": f"{d.name}/original.jpg"})
    return entries


def main():
    t = time.time()
    entries = load_entries()
    vecs, rows = [], []
    for i, e in enumerate(entries):
        img = Image.open(SEEDS / e["image"])
        for v in variants(img):
            vecs.append(embed(v))
            rows.append({"entry": i, "phash": str(imagehash.phash(v))})
    INDEX.mkdir(exist_ok=True)
    np.save(INDEX / "vectors.npy", np.stack(vecs))
    (INDEX / "entries.json").write_text(json.dumps(entries, indent=1))
    (INDEX / "rows.json").write_text(json.dumps(rows))
    print(f"{len(entries)} entries, {len(rows)} vectors, {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
