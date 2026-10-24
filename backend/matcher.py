import json
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from embed import embed

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "index"

PHASH_MAX = 8
EMBED_MIN = 0.55  # placeholder until eval.py calibrates it
NEAR_IDENTICAL = 0.95


class Matcher:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.vectors = np.load(INDEX / "vectors.npy")
        self.entries = json.loads((INDEX / "entries.json").read_text())
        rows = json.loads((INDEX / "rows.json").read_text())
        self.row_entry = np.array([r["entry"] for r in rows])
        self.row_hash = [imagehash.hex_to_hash(r["phash"]) for r in rows]

    def match(self, img: Image.Image) -> dict:
        img = img.convert("RGB")
        h = imagehash.phash(img)
        dists = [h - rh for rh in self.row_hash]
        best_row = int(np.argmin(dists))
        if dists[best_row] <= PHASH_MAX:
            entry = self.entries[self.row_entry[best_row]]
            return self._result(entry, 1 - dists[best_row] / 64, "phash")

        # Exact cosine search; vectors are L2-normalized. Swap for a FAISS index past ~100k rows.
        sims = self.vectors @ embed(img)
        row = int(np.argmax(sims))
        score = float(sims[row])
        if score < EMBED_MIN:
            return {"match": False, "score": round(score, 3)}
        return self._result(self.entries[self.row_entry[row]], score, "embedding")

    def _result(self, entry, score, method):
        score = round(float(score), 3)
        if entry["kind"] == "original":
            # Real photo. Edited-version detection against originals is handled separately.
            return {"match": False, "score": score, "original_of": entry["parent"]}
        note = {
            "headline": entry["truth"],
            "context": f"Shared as: {entry['claim']}",
            "source_name": entry["source_name"],
            "source_url": entry["source_url"],
        }
        if entry.get("has_original"):
            note["original_url"] = f"{self.base_url}/seeds/{entry['id']}/original.jpg"
        return {"match": True, "kind": entry["kind"], "score": score, "method": method,
                "id": entry["id"], "note": note}
