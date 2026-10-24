# Sidenote: system design

Sidenote is a browser extension that recognizes images people have already fact-checked and puts a short note under them, wherever they show up. It never decides on its own whether an image is fake. Every note comes from a published fact-check.

## 1. Components

```mermaid
flowchart LR
  subgraph Browser
    CS["content.js<br/>(every tab)"]
    BG["background.js<br/>(service worker)"]
    CARD["note card<br/>(shadow DOM)"]
  end
  subgraph Backend["Backend (FastAPI, localhost:8000)"]
    API["/match"]
    M["Matcher"]
    ED["KnownEdit<br/>(edit_diff.py)"]
    IDX[("index/<br/>vectors.npy<br/>rows.json<br/>entries.json")]
  end
  subgraph Offline["Offline build"]
    SEEDS[("seeds/<br/>image.jpg<br/>original.jpg<br/>meta.json")]
    AUG["augment.py<br/>8 repost variants"]
    BUILD["build_index.py<br/>SSCD + pHash"]
  end
  CDN[("image host<br/>pbs.twimg.com, i.redd.it, ...")]

  CS -- "src of on-screen img" --> BG
  BG -- "POST {url} or bytes" --> API
  API -- fetch --> CDN
  API --> M
  M --> IDX
  M --> ED
  BG -- "match or null" --> CS
  CS --> CARD
  SEEDS --> AUG --> BUILD --> IDX
```

| Part | Job | Why it's built this way |
|---|---|---|
| `content.js` | Watches `<img>` tags with an IntersectionObserver, checks the ones on or about to come on screen, and places the card | Never scans the whole page. Each image costs one request, and only when you can actually see it |
| `background.js` | Queues requests (3 at a time), caches by URL, falls back to sending the image bytes when the backend can't download it | Content scripts can't make cross-origin requests to localhost. The queue keeps fast scrolling from flooding the backend |
| note card | Shadow DOM element inserted after the post's media block | The site's CSS can't break it and it can't break the site |
| `/match` | Gets the image, caches by SHA-256 of the bytes, runs the matcher | The same viral image shows up under many URLs. The byte hash dedupes them |
| `Matcher` | pHash first, then SSCD embedding search, then the edited-pair check | Cheap exact check first, then the robust one |
| `/report` | "Wrong match?" on a card hides it and logs the image URL and seed id | Someone reviews every report. The matcher never learns from clicks, so nobody can game it into silencing a real fact-check |
| `KnownEdit` | For an edited fake that has its real original: finds the region that was changed and judges which version a new copy came from | Without this, a copy of the real photo would get flagged as the fake |

## 2. Life of one image

```mermaid
sequenceDiagram
  participant P as Page
  participant C as content.js
  participant B as background.js
  participant S as /match
  participant M as Matcher

  P->>C: img scrolls within 1 screen of view
  C->>C: skip if < 150px, avatar, emoji, svg
  C->>B: check(src)
  alt seen this URL before
    B-->>C: cached result
  else new
    B->>S: POST {url}
    S->>S: download image
    opt host blocks the download
      S-->>B: fetch_failed
      B->>B: fetch image in the browser
      B->>S: POST multipart bytes
    end
    S->>S: sha256 cache lookup
    S->>M: match(image)
    M-->>S: {match, kind, note, diff_png?}
    S-->>B: result
    B-->>C: result or null
  end
  C->>P: insert note card under the media (only on a match)
```

## 3. Matching decision

```mermaid
flowchart TD
  Q[query image] --> T[trim solid borders<br/>screenshots, letterboxing]
  T --> H{pHash within 8 bits<br/>of any indexed variant?}
  H -- yes --> R[entry]
  H -- no --> E[SSCD 512-d embedding<br/>cosine vs all variants]
  E --> TH{best score >= 0.45?}
  TH -- no --> SILENT[silent]
  TH -- yes --> R
  R --> PAIR{entry has a real original?}
  PAIR -- no --> NOTE[note card]
  PAIR -- yes --> KE[KnownEdit: compare the edited region<br/>to the fake and to the original]
  KE --> V{closer to the fake?}
  V -- fake --> NOTE2[note card + highlight of the edit]
  V -- original --> SILENT2[silent: it's the real photo]
```

- **SSCD** (Meta's self-supervised copy detection model, ResNet-50, 512-d) is trained to match an image to edited copies of itself: crops, overlays, recompression, color shifts. That's the exact problem here. General-purpose embeddings like CLIP match on *topic*, so they'd flag every photo of a shark on a highway.
- **Threshold 0.45.** The highest-scoring normal photo out of 5,236 is 0.31. The band between 0.40 and 0.45 holds other images from the same AI generation batch as a known fake (0.42 to 0.43 on the Midjourney pope set). They were never fact-checked, so they stay silent.
- **Index-time augmentation**: every seed is stored as 8 variants (crops, half-size JPEG, caption bar, phone screenshot frame, mirror). Making the index robust costs nothing at query time.
- **Exact search** with numpy over L2-normalized vectors. That's the same math as FAISS `IndexFlatIP` and it's fine at this scale. We dropped FAISS because faiss and torch each bundle libomp on macOS and crash together. Past ~100k images, we'd move to FAISS IVF-PQ in its own process.

## 4. Edited photos

```mermaid
flowchart LR
  F[fake] & O[original] --> A[ORB features + RANSAC homography<br/>align fake onto original]
  A --> D[color-normalized Lab difference<br/>threshold, open, dilate]
  D --> REG[edited region mask]
  Q[new copy] --> A2[align to original] --> CMP{"in the edited region:<br/>closer to fake or original?"}
  REG --> CMP
```

The edited region is computed once, when the index loads. A new copy is aligned to the original, and each pixel inside that region votes for whichever version it's close to. Crops, filters and captions change the rest of the image, so they don't matter. Pixels that look like neither version, or that are covered by a solid caption bar or sticker, get no vote. A verdict needs a clear majority. If there's no clear majority, the matcher falls back to comparing embedding similarity with a margin, and when it's still unsure it stays silent. Calling a real photo fake is the worst mistake this product can make, so every tie goes to silence.

## 5. Offline build

```mermaid
flowchart LR
  FC["published fact-check<br/>(Snopes, AFP, Reuters, PolitiFact...)"] --> S["seeds/&lt;id&gt;/<br/>image.jpg, original.jpg?, meta.json"]
  S --> V["augment.py<br/>x8 variants"]
  V --> EMB["SSCD embed + pHash"]
  EMB --> I[("index/")]
  I --> EV["eval.py<br/>recall, false flags, latency"]
```

## 6. Numbers

`eval.py` measures three things:

- **Synthetic reposts** of every seed: random crops, rotation, color shifts, caption text, screenshot frames, JPEG quality 30 to 70. None of these are the transforms used at index time.
- **Real reposts found online**: tweet screenshots, fact-check frames with stamps, captioned thumbnails, news article crops.
- **Normal photos**: 236 hand-picked ones (148 show the same subject as a fake), plus the 5,000 COCO val2017 images.

| | |
|---|---|
| Seeds | 53 (9 edited pairs with the real original) |
| Synthetic reposts caught | 152/159 (96%) |
| Real reposts from the web caught | 21/26 |
| Edited fakes caught / real originals flagged | 83/90 / 1/90 |
| Normal photos flagged | 0/5,236 |
| Highest normal-photo score | 0.31 vs threshold 0.45 |
| Latency | ~60 ms median, ~100 ms p95, CPU only |
| Cost | $0, everything runs locally |

The real reposts it misses are collages where the fake is a small tile, and different images from the same AI series as a seed (they're separate images, so they *should* need their own entry).

## 7. Going to production

What's built today runs on localhost with a hand-curated seed set. These are the two things that would have to change before real users:

```mermaid
flowchart LR
  subgraph Client["Extension"]
    LOC["local prefilter<br/>pHash + compressed vectors"]
  end
  subgraph Server
    API2[match API]
    BIG[("FAISS IVF-PQ<br/>millions of entries")]
  end
  subgraph Ingest
    CR["ClaimReview feeds<br/>(fact-checker markup)"]
    CN["Community Notes<br/>public data"]
    PIPE["pull image, dedupe,<br/>embed, add"]
  end
  LOC -- "only on a likely hit" --> API2 --> BIG
  CR & CN --> PIPE --> BIG
  BIG -- "periodic compact snapshot" --> LOC
```

- **Privacy**: right now every image you scroll past gets sent to the backend. In production the extension would hold a compact local index and only ask the server on a likely hit, the same way Safe Browsing checks URLs.
- **Coverage**: seeds come from fact-checks already published in machine-readable form, mainly ClaimReview markup that fact-checkers attach to their articles and Community Notes public data, instead of being hand-picked.
