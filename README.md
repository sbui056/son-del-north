# Sidenote

**Grammarly for images.** Sidenote is a Chrome extension that quietly checks every image you scroll past and only speaks up when one is a known fake or an edited version of a real photo. When it does, it puts a short note under the image with what's actually going on and a link to the fact-check.

No popup, no clicking, no account. It stays silent on everything else.

## Why

Most viral fakes have already been fact-checked by someone. The problem is that the fact-check stays stuck on one post. A Community Note lives on X. A Snopes article lives on Snopes. Screenshot the image, repost it on Reddit or in the family group chat, and the correction is gone.

Sidenote makes the note travel with the image. It doesn't decide what's true. People already did. It recognizes the same image after it's been cropped, screenshotted, recompressed or captioned, and brings their verdict along.

## What it looks like

- **Known fake:** a note under the image saying what it really is and where it came from, with the source.
- **Edited real photo:** the same note, plus the real original next to the shared version with the edited area outlined.
- **Everything else:** nothing. The real photo of the same event, the same person or the same flood stays silent.

It works on any site: Reddit, X, news articles, blogs.

## How it works

```
on-screen <img>  ->  content.js  ->  background.js  ->  /match (FastAPI, localhost)
                                                          |
                                         pHash, then SSCD copy-detection embedding
                                         vs. an index of fact-checked images
                                                          |
                                         edited pair? compare the edited region
                                         to the fake and to the real original
                                                          |
                     note card  <-  match + fact-check  <-+
```

- Only images on screen (or about to be) get checked. The page is never scanned as a whole.
- Matching uses [SSCD](https://github.com/facebookresearch/sscd-copy-detection), a model trained to recognize an image after edits like crops, overlays and recompression. General image models match on topic and would flag every photo of a shark. SSCD matches the actual picture.
- Each fact-checked image is indexed along with 8 variants that copy how people repost (crops, screenshot frames, caption bars, low-quality JPEG), so the index is robust at no cost at query time.
- For edited photos where we have the real original, Sidenote works out which pixels the edit changed, then judges each new copy by those pixels only. A repost of the real photo stays silent even though it's nearly identical to the fake.

The full design, with diagrams, is in [docs/DESIGN.md](docs/DESIGN.md).

## Numbers

From `backend/eval.py`, 53 fact-checked images, everything on CPU:

| | |
|---|---|
| Reposts caught (random crops, rotation, filters, captions, screenshots, heavy JPEG) | 152/159 |
| Real reposts collected from the web | 21/26 |
| Edited fakes caught / real originals wrongly flagged | 83/90 / 1/90 |
| Normal photos wrongly flagged (236 lookalikes + 5,000 COCO) | 0/5,236 |
| Time per image | ~60 ms median |
| Cost | $0, runs locally |

## Run it

Requires Python 3.12 and Chrome.

```bash
python -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
mkdir -p weights
curl -L -o weights/sscd_disc_mixup.torchscript.pt \
  https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt

cd backend
../.venv/bin/python build_index.py          # seeds/ -> index/
../.venv/bin/python -m uvicorn app:app --port 8000
```

Then open `chrome://extensions`, turn on Developer mode, click **Load unpacked**, and pick the `extension/` folder.

Seed images aren't in the repo (they belong to their photographers and publishers). Each `seeds/<id>/meta.json` has the claim, the verdict and the fact-check link. Put the image next to it as `image.jpg`, and for edited photos the real one as `original.jpg`.

**Demo feed:** `../.venv/bin/python make_demo.py`, then open `http://localhost:8000/demo/?demo`. It's a fake social feed with reposted fakes mixed in among normal photos. The counter in the corner shows how many images were checked and how many got a note (Option+Shift+F toggles it on other sites).

## Layout

```
extension/   content.js (find on-screen images, place the card), background.js (queue, cache, backend calls), card.css
backend/     app.py (API), matcher.py, embed.py (SSCD), edit_diff.py (edited-photo check),
             augment.py, build_index.py, eval.py, make_demo.py
seeds/       one folder per fact-checked image: meta.json (+ image.jpg, original.jpg locally)
demo/        offline demo feed
docs/        DESIGN.md
```

## Limits and what's next

- **Coverage.** 53 hand-picked fact-checks today. Next is pulling them in automatically from ClaimReview, the markup fact-checkers already attach to their articles, and from Community Notes public data.
- **Privacy.** Right now every image you scroll past gets sent to the local backend. A hosted version would keep a compact index inside the extension and only ask the server when there's a likely match, the way Safe Browsing checks URLs.
- **Brand-new fakes.** Sidenote only knows fakes someone has already checked. That covers repeat spread, which is where most of the damage is.
- **Collages.** When the fake is one small tile in a grid of images, the match can be missed.
- **Video.** Not yet. Keyframes would go through the same index.
