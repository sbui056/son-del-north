"""Variants of a seed image that mimic how it gets reposted."""
import io

from PIL import Image, ImageDraw, ImageOps


def _crop(img, left, top, right, bottom):
    w, h = img.size
    return img.crop((int(w * left), int(h * top), int(w * right), int(h * bottom)))


def _jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def _caption_bar(img):
    w, h = img.size
    bar = int(h * 0.18)
    out = Image.new("RGB", (w, h + bar), "white")
    out.paste(img, (0, bar))
    d = ImageDraw.Draw(out)
    for i in range(3):
        y = int(bar * (0.2 + 0.25 * i))
        d.rectangle((int(w * 0.05), y, int(w * (0.9 - 0.2 * i)), y + int(bar * 0.12)), fill="black")
    return out


def _screenshot(img):
    w, h = img.size
    sw = w
    sh = max(h, int(w * 2.1))
    out = Image.new("RGB", (sw, sh), (18, 18, 18))
    out.paste(img, (0, (sh - h) // 2))
    return out


def variants(img: Image.Image) -> list[Image.Image]:
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    return [
        img,
        _crop(img, 0.1, 0.1, 0.9, 0.9),
        _crop(img, 0.0, 0.0, 0.7, 1.0),
        _crop(img, 0.3, 0.0, 1.0, 1.0),
        _jpeg(img.resize((max(1, w // 2), max(1, h // 2))), 40),
        _caption_bar(img),
        _screenshot(img),
        ImageOps.mirror(img),
    ]
