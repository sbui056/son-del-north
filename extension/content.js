const NAME = chrome.runtime.getManifest().name;
const MIN_SIZE = 150;
const LABELS = {
  debunked: "Fact-checked",
  out_of_context: "Out of context",
  ai_generated: "AI-generated",
  edited: "Edited photo",
};
const ICONS = {
  debunked: "M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13Zm3 4.6-3.6 4.2a.8.8 0 0 1-1.2 0L4.9 8.8a.8.8 0 1 1 1.2-1l.7.8 3-3.5a.8.8 0 0 1 1.2 1Z",
  ai_generated: "M8 1c.4 2.9 1.7 4.6 5 5-3.3.4-4.6 2.1-5 5-.4-2.9-1.7-4.6-5-5 3.3-.4 4.6-2.1 5-5Zm4.5 8.5c.2 1.4.8 2 2 2.2-1.2.2-1.8.8-2 2.3-.2-1.5-.8-2.1-2-2.3 1.2-.2 1.8-.8 2-2.2Z",
  edited: "M2 3.5A1.5 1.5 0 0 1 3.5 2h9A1.5 1.5 0 0 1 14 3.5v9a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 2 12.5v-9Zm1.5 0v6.2l2.4-2.4a.8.8 0 0 1 1.1 0l4.2 4.2h1.3v-8h-9Zm7 3.5a1.3 1.3 0 1 0 0-2.6 1.3 1.3 0 0 0 0 2.6Z",
};
const BACKEND = "http://localhost:8000";
const SKIP_SRC = /emoji|profile_images|avatar|styles\.redditmedia|\/styles\/|\.svg(\?|$)/i;

const results = new Map(); // normalized src -> Promise<match or null>
const checkedSrc = new WeakMap(); // img -> last src we looked at
const placed = new WeakMap(); // post -> Set of srcs that already have a card
const stats = { checked: 0, flagged: 0 };

const sheet = new CSSStyleSheet();
fetch(chrome.runtime.getURL("card.css"))
  .then((res) => res.text())
  .then((css) => sheet.replaceSync(css));

const viewport = new IntersectionObserver(onVisible, { rootMargin: "100% 0px" });

function track(node) {
  if (node.tagName !== "IMG") return;
  if (checkedSrc.get(node) === (node.currentSrc || node.src)) return;
  viewport.observe(node);
}

function onVisible(entries) {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    viewport.unobserve(entry.target);
    whenLoaded(entry.target).then(inspect);
  }
}

function whenLoaded(img) {
  if (img.complete) return Promise.resolve(img);
  return new Promise((resolve) => img.addEventListener("load", () => resolve(img), { once: true }));
}

async function inspect(img) {
  const raw = img.currentSrc || img.src;
  if (checkedSrc.get(img) === raw) return;
  checkedSrc.set(img, raw);
  if (shouldSkip(img, raw)) return;

  const src = normalize(raw);
  if (!results.has(src)) {
    results.set(src, ask(src));
    stats.checked++;
    updatePill();
  }
  const hit = await results.get(src);
  if (hit) showCard(img, src, hit);
}

function shouldSkip(img, src) {
  if (!/^https?:/i.test(src) || SKIP_SRC.test(src) || src.startsWith(`${BACKEND}/seeds/`)) return true;
  const box = img.getBoundingClientRect();
  return (
    Math.min(box.width, box.height) < MIN_SIZE ||
    Math.min(img.naturalWidth, img.naturalHeight) < MIN_SIZE
  );
}

// X serves small previews in the feed; ask for the large one so matching has more to work with.
function normalize(src) {
  const url = new URL(src);
  if (url.hostname === "pbs.twimg.com" && url.pathname.startsWith("/media/")) {
    url.searchParams.set("name", "large");
  }
  return url.href;
}

async function ask(src) {
  try {
    return await chrome.runtime.sendMessage({ type: "check", src });
  } catch {
    return null; // extension was reloaded under this page
  }
}

function showCard(img, src, hit) {
  if (!isWebUrl(hit.note?.source_url)) return;
  const post = img.closest("article, shreddit-post") || img.parentElement;
  const done = placed.get(post) || new Set();
  if (done.has(src)) return;
  done.add(src);
  placed.set(post, done);

  const anchor = anchorFor(img, post);
  const card = renderCard(hit, isDark(anchor || img));
  if (anchor) {
    if (anchor.slot) card.slot = anchor.slot; // stay in the same slot of Reddit's post element
    anchor.after(card);
  } else {
    placeGeneric(img, card);
  }
  stats.flagged++;
  updatePill();
}

function anchorFor(img, post) {
  const photo = img.closest('[data-testid="tweetPhoto"]');
  if (photo) {
    // Climb to the whole media block, stopping before the level that holds the tweet text or action bar.
    let node = photo;
    while (
      node.parentElement &&
      node.parentElement !== post &&
      !node.parentElement.querySelector('[data-testid="tweetText"], [role="group"]')
    ) {
      node = node.parentElement;
    }
    return node;
  }
  return img.closest('[slot="post-media-container"], shreddit-aspect-ratio, gallery-carousel');
}

// Any other site: right under the image or its figure/link/picture, kept to the image's width.
function placeGeneric(img, card) {
  let node = img.closest("figure") || img.closest("picture, a") || img;
  while (node.parentElement !== document.body && /^inline/.test(getComputedStyle(node.parentElement).display)) {
    node = node.parentElement;
  }
  card.style.maxWidth = `${Math.max(img.getBoundingClientRect().width, 260)}px`;
  const parent = getComputedStyle(node.parentElement).display;
  if (!/flex|grid/.test(parent)) {
    node.after(card);
    // Centered images: line the card up with the image, not the text column.
    const offset = img.getBoundingClientRect().left - card.getBoundingClientRect().left;
    if (offset > 1) card.style.marginLeft = `${offset}px`;
    return;
  }
  // Inside a grid or flex row, keep the card in the same cell instead of adding a new one.
  if (!/^(A|IMG|PICTURE)$/.test(node.tagName)) return node.append(card);
  if (parent.includes("flex")) return node.parentElement.after(card);
  card.style.gridColumn = "1 / -1";
  node.after(card);
}

function renderCard({ kind, note, compare }, dark) {
  const host = document.createElement("div");
  host.dataset.sidenote = "";
  // Keep clicks on the card from opening the post underneath.
  host.addEventListener("click", (e) => e.stopPropagation());
  const root = host.attachShadow({ mode: "closed" });
  root.adoptedStyleSheets = [sheet];

  const card = el("section", "card");
  card.dataset.kind = kind;
  card.dataset.theme = dark ? "dark" : "light";

  const top = el("div", "top");
  const label = el("span", "label");
  label.append(icon(ICONS[kind] || ICONS.debunked), LABELS[kind] || "Fact-checked");
  top.append(label, el("span", "brand", NAME));
  card.append(top);
  if (note.headline) card.append(el("p", "headline", note.headline));
  if (note.context) card.append(el("p", "context", note.context));

  if (kind === "edited" && compare?.original && compare?.shared) {
    const pair = el("div", "compare");
    pair.append(shot(compare.original, "Original"), shot(compare.shared, "Shared here"));
    // Sites whose CSP blocks data: images still get the note and the "See the original" link.
    pair.addEventListener("error", () => pair.remove(), true);
    card.append(pair);
  }

  const links = el("div", "links");
  links.append(link(note.source_url, `Source: ${note.source_name || new URL(note.source_url).hostname}`));
  if (kind === "edited" && isWebUrl(note.original_url)) links.append(link(note.original_url, "See the original"));
  card.append(links);

  root.append(card);
  return host;
}

function shot(src, caption) {
  const figure = el("figure");
  const img = el("img");
  img.src = src;
  img.alt = `${caption}, with the edited area outlined`;
  figure.append(img, el("figcaption", "", caption));
  return figure;
}

function icon(path) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("aria-hidden", "true");
  const shape = document.createElementNS(ns, "path");
  shape.setAttribute("d", path);
  svg.append(shape);
  return svg;
}

function el(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function link(href, text) {
  const a = el("a", "", text);
  a.href = href;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  return a;
}

function isWebUrl(value) {
  return typeof value === "string" && /^https?:\/\//i.test(value);
}

// Use the first solid background behind the card to pick a light or dark style.
function isDark(node) {
  for (let n = node; n; n = n.parentElement) {
    const parts = getComputedStyle(n).backgroundColor.match(/[\d.]+/g);
    if (!parts) continue;
    const [r, g, b, a = 1] = parts.map(Number);
    if (a > 0.5) return 0.299 * r + 0.587 * g + 0.114 * b < 128;
  }
  return matchMedia("(prefers-color-scheme: dark)").matches;
}

// Small counter for demos: Alt+Shift+F toggles it, or add ?demo to the URL.
let pill;

function updatePill() {
  if (pill) pill.textContent = `checked ${stats.checked} · flagged ${stats.flagged}`;
}

function togglePill() {
  if (!pill) {
    const host = el("div");
    const root = host.attachShadow({ mode: "closed" });
    root.adoptedStyleSheets = [sheet];
    pill = el("div", "pill");
    pill.hidden = true;
    root.append(pill);
    document.body.append(host);
  }
  pill.hidden = !pill.hidden;
  updatePill();
}

window.addEventListener(
  "keydown",
  (e) => {
    if (e.altKey && e.shiftKey && e.code === "KeyF") togglePill();
  },
  true
);
if (new URLSearchParams(location.search).has("demo")) togglePill();

new MutationObserver((mutations) => {
  for (const m of mutations) {
    if (m.type === "attributes") {
      track(m.target);
      continue;
    }
    for (const node of m.addedNodes) {
      if (node.nodeType !== Node.ELEMENT_NODE) continue;
      track(node);
      node.querySelectorAll("img").forEach(track);
    }
  }
}).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["src"] });

document.querySelectorAll("img").forEach(track);
