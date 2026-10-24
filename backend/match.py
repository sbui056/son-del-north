"""Usage: python match.py <image> [<image> ...]"""
import json
import sys
import time

from PIL import Image

from matcher import Matcher

if __name__ == "__main__":
    m = Matcher()
    for path in sys.argv[1:]:
        t = time.time()
        result = m.match(Image.open(path))
        print(path, f"{(time.time() - t) * 1000:.0f}ms", json.dumps(result))
