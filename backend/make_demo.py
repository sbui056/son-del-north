"""Build the offline demo feed: normal photos with a few reposted fakes mixed in."""
import json
import random
import shutil
from pathlib import Path

from PIL import Image

from eval import repost
from matcher import Matcher

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "demo" / "feed"
NORMAL_PER_FAKE = 4
# Neutral stories lead the feed; the rest follow in index order.
FIRST = ["hurricane-shark-highway", "pope-puffer-jacket", "helicopter-shark", "obama-rouhani-handshake",
         "pentagon-explosion-ai", "tourist-guy-911", "disney-world-milton-ai", "trump-hair-ap"]

CAPTIONS = [
    "can't believe this is real", "my uncle just sent this", "wow", "sharing so people see it",
    "this is insane", "nature is wild", "look at this", "unreal", "saw this on my timeline",
]


def main():
    rng = random.Random(3)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    entries = json.loads((ROOT / "index" / "entries.json").read_text())
    fakes = [e for e in entries if e["kind"] != "original"]
    fakes.sort(key=lambda e: FIRST.index(e["id"]) if e["id"] in FIRST else len(FIRST))
    matcher = Matcher()
    normals = sorted((ROOT / "test" / "negatives").glob("*.jpg"))
    rng.shuffle(normals)

    posts = []
    for i, e in enumerate(fakes):
        for n in normals[i * NORMAL_PER_FAKE:(i + 1) * NORMAL_PER_FAKE]:
            name = f"n_{n.name}"
            shutil.copy(n, OUT / name)
            posts.append({"img": name, "caption": rng.choice(CAPTIONS)})
        name = f"r_{e['id']}.jpg"
        seed = Image.open(ROOT / "seeds" / e["image"])
        # A random crop can cut out the edit itself; keep one that still shows it.
        for _ in range(5):
            img = repost(seed).convert("RGB")
            if matcher.match(img)["match"]:
                break
        img.save(OUT / name, quality=80)
        posts.append({"img": name, "caption": e["claim"]})
    (OUT / "feed.json").write_text(json.dumps(posts, indent=1))
    print(f"{len(posts)} posts, {len(fakes)} fakes")


if __name__ == "__main__":
    main()
