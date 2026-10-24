import json
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageChops

import edit_diff
from embed import embed

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "index"
SEEDS = ROOT / "seeds"

PHASH_MAX = 8
PAIR_MARGIN = 0.03  # unsure between a fake and its original -> stay silent
EMBED_MIN = 0.45  # unrelated photos top out around 0.25 in eval.py


def trim_borders(img: Image.Image) -> Image.Image:
    """Strip solid-color frames (screenshots, letterboxing) around the photo."""
    bg = Image.new("RGB", img.size, img.getpixel((0, 0)))
    box = ImageChops.difference(img, bg).convert("L").point(lambda p: p > 24 and 255).getbbox()
    if box and (box[2] - box[0]) * (box[3] - box[1]) > 0.3 * img.width * img.height:
        return img.crop(box)
    return img


class Matcher:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.vectors = np.load(INDEX / "vectors.npy")
        self.entries = json.loads((INDEX / "entries.json").read_text())
        rows = json.loads((INDEX / "rows.json").read_text())
        self.row_entry = np.array([r["entry"] for r in rows])
        self.row_hash = [imagehash.hex_to_hash(r["phash"]) for r in rows]
        self.ids = {e["id"]: i for i, e in enumerate(self.entries)}
        self.known_edits = {
            e["id"]: edit_diff.KnownEdit(Image.open(SEEDS / e["image"]), Image.open(SEEDS / e["id"] / "original.jpg"))
            for e in self.entries if e.get("has_original")
        }

    def match(self, img: Image.Image) -> dict:
        img = img.convert("RGB")
        # Exact cosine search; vectors are L2-normalized. Swap for a FAISS index past ~100k rows.
        sims = self.vectors @ embed(trim_borders(img))
        h = imagehash.phash(img)
        dists = [h - rh for rh in self.row_hash]
        best_row = int(np.argmin(dists))
        if dists[best_row] <= PHASH_MAX:
            return self._result(img, sims, self.row_entry[best_row], 1 - dists[best_row] / 64, "phash")

        row = int(np.argmax(sims))
        score = float(sims[row])
        if score < EMBED_MIN:
            return {"match": False, "score": round(score, 3)}
        return self._result(img, sims, self.row_entry[row], score, "embedding")

    def _entry_sim(self, sims, i):
        return float(sims[self.row_entry == i].max())

    def _result(self, img, sims, i, score, method):
        score = round(float(score), 3)
        entry = self.entries[i]
        overlay = None
        if entry["kind"] == "original" or entry.get("has_original"):
            # Known edit with its real original indexed: work out which of the two the query copies.
            fake_i = i if entry.get("has_original") else self.ids[entry["parent"]]
            orig_i = self.ids[f"{self.entries[fake_i]['id']}/original"]
            entry = self.entries[fake_i]
            known = self.known_edits[entry["id"]]
            verdict = known.verdict(img)
            if verdict is None:
                lead = self._entry_sim(sims, fake_i) - self._entry_sim(sims, orig_i)
                verdict = "fake" if lead > PAIR_MARGIN else "original"
            if verdict == "original":
                return {"match": False, "score": score, "original_of": entry["id"]}
            overlay = known.compare if known.f is not None else None
        note = {
            "headline": entry["truth"],
            "context": f"Shared as: {entry['claim']}",
            "source_name": entry["source_name"],
            "source_url": entry["source_url"],
        }
        if entry.get("has_original"):
            note["original_url"] = f"{self.base_url}/seeds/{entry['id']}/original.jpg"
        result = {"match": True, "kind": entry["kind"], "score": score, "method": method,
                  "id": entry["id"], "note": note}
        if overlay:
            result["kind"] = "edited"
            result["compare"] = overlay
        return result
