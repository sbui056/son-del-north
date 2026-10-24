"""Align a query image to a known original and highlight what changed."""
import base64

import cv2
import numpy as np
from PIL import Image

WORK_WIDTH = 640
MIN_CHANGED = 0.01  # fraction of the aligned area that must differ


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


def diff(query: Image.Image, original: Image.Image) -> dict | None:
    """{"changed": fraction, "png": base64 overlay on the original} or None if alignment fails."""
    q, o = _cv(query), _cv(original)
    aligned = _align(q, o)
    if aligned is None:
        return None
    warped, mask = aligned
    blur = lambda x: cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2LAB), (7, 7), 0)
    delta = cv2.absdiff(blur(warped), blur(o)).max(axis=2)
    _, changed = cv2.threshold(delta, 40, 255, cv2.THRESH_BINARY)
    changed = cv2.bitwise_and(changed, mask)
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    changed = cv2.dilate(changed, np.ones((9, 9), np.uint8))
    fraction = float((changed > 0).sum()) / max(1, int((mask > 0).sum()))

    overlay = o.copy()
    overlay[changed > 0] = (0.45 * overlay[changed > 0] + 0.55 * np.array([60, 60, 255])).astype(np.uint8)
    contours, _ = cv2.findContours(changed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (40, 40, 230), 2)
    ok, png = cv2.imencode(".png", overlay)
    return {"changed": round(fraction, 4), "png": base64.b64encode(png.tobytes()).decode() if ok else None}


def is_edited(result: dict | None) -> bool:
    return result is not None and result["changed"] >= MIN_CHANGED
