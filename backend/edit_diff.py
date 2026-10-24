"""Align a query image to a known original and highlight what changed."""
import base64

import cv2
import numpy as np
from PIL import Image

WORK_WIDTH = 640
VERDICT_MARGIN = 3.0  # Lab units; closer calls fall back to embedding similarity
CLOSE = 50  # Lab units; a pixel this close to one version counts as a vote for it
KEEP_REGION = 0.2  # drop changed blobs smaller than this fraction of the biggest one


def _cv(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    scale = WORK_WIDTH / img.width
    img = img.resize((WORK_WIDTH, max(1, round(img.height * scale))))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def _align(query, original):
    """Warp query onto original's frame. Returns (warped, valid_mask) or None."""
    orb = cv2.ORB_create(3000)
    gq, go = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (query, original))
    kq, dq = orb.detectAndCompute(gq, None)
    ko, do = orb.detectAndCompute(go, None)
    if dq is None or do is None:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dq, do, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 15:
        return None
    src = np.float32([kq[m.queryIdx].pt for m in good])
    dst = np.float32([ko[m.trainIdx].pt for m in good])
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None or inliers.sum() < 12:
        return None
    h, w = original.shape[:2]
    warped = cv2.warpPerspective(query, H, (w, h))
    mask = cv2.warpPerspective(np.full(gq.shape, 255, np.uint8), H, (w, h))
    return warped, cv2.erode(mask, np.ones((15, 15), np.uint8))


def _lab(x):
    return cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2LAB), (7, 7), 0).astype(np.float32)


def _match_colors(wq, wo, inside):
    """Undo filters and brightness changes so only real content changes remain."""
    for c in range(3):
        q, r = wq[..., c][inside], wo[..., c][inside]
        wq[..., c] = (wq[..., c] - q.mean()) / (q.std() + 1e-6) * r.std() + r.mean()
    return wq


def _changes(warped, mask, o):
    """Binary mask of regions that differ between an aligned copy and the original."""
    wo = _lab(o)
    wq = _match_colors(_lab(warped), wo, mask > 0)
    delta = np.abs(wq - wo).max(axis=2).astype(np.uint8)
    _, changed = cv2.threshold(delta, 40, 255, cv2.THRESH_BINARY)
    changed = cv2.bitwise_and(changed, mask)
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    changed = cv2.dilate(changed, np.ones((9, 9), np.uint8))

    # Caption bars, frames and rotation corners hug the edge of the overlap; edits sit inside it.
    edge = cv2.subtract(mask, cv2.erode(mask, np.ones((25, 25), np.uint8)))
    n, labels = cv2.connectedComponents(changed)
    for i in range(1, n):
        if (edge[labels == i] > 0).any():
            changed[labels == i] = 0
    return changed


def _outline(img, changed):
    out = img.copy()
    contours, _ = cv2.findContours(changed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 6)
    cv2.drawContours(out, contours, -1, (40, 40, 230), 3)
    ok, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode() if ok else None


def _compare(o, shared, changed):
    """The original and the shared copy, both in the original's frame with the edit outlined."""
    return {"original": _outline(o, changed), "shared": _outline(shared, changed)}


def _main_regions(changed):
    """Keep the edit itself, not the specks that JPEG noise and blending leave around it."""
    n, labels, sizes, _ = cv2.connectedComponentsWithStats(changed)
    if n <= 2:
        return changed
    areas = sizes[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.flatnonzero(areas >= KEEP_REGION * areas.max())
    return np.where(np.isin(labels, keep), 255, 0).astype(np.uint8)


class KnownEdit:
    """A fake and its authentic original. Tells which of the two a query copies by
    looking only at the region the fake changed."""

    def __init__(self, fake: Image.Image, original: Image.Image):
        self.o = _cv(original)
        aligned = _align(_cv(fake), self.o)
        self.f = None
        if aligned is not None:
            self.f, mask = aligned
            self.region = _changes(self.f, mask, self.o)
            self.compare = _compare(self.o, self.f, _main_regions(self.region))

    def verdict(self, query: Image.Image) -> str | None:
        """"fake", "original", or None when it can't tell (alignment failed or edit cropped out)."""
        if self.f is None:
            return None
        aligned = _align(_cv(query), self.o)
        if aligned is None:
            return None
        warped, mask = aligned
        area = (self.region > 0) & (cv2.erode(mask, np.ones((15, 15), np.uint8)) > 0)
        if area.sum() < 0.5 * (self.region > 0).sum() or area.sum() < 200:
            return None
        o, f = _lab(self.o), _lab(self.f)
        q = _match_colors(_lab(warped), o, cv2.erode(mask, np.ones((15, 15), np.uint8)) > 0)
        to_orig = np.abs(q - o).max(axis=2)[area]
        to_fake = np.abs(q - f).max(axis=2)[area]
        # Pixels that look like neither version (captions, stickers over the edit) don't get a vote.
        decided = (np.minimum(to_orig, to_fake) < CLOSE) & (np.abs(to_orig - to_fake) > VERDICT_MARGIN)
        if decided.sum() < max(200, 0.3 * area.sum()):
            return None
        fake_share = (to_fake < to_orig)[decided].mean()
        if fake_share > 0.7:
            return "fake"
        if fake_share < 0.3:
            return "original"
        return None
