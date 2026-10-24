const NAME = chrome.runtime.getManifest().name;
const MIN_SIZE = 150;
const LABELS = {
  debunked: "Fact-checked",
  out_of_context: "Fact-checked",
  ai_generated: "AI-generated",
  edited: "Edited",
};
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
  if (!/^https?:/i.test(src) || SKIP_SRC.test(src)) return true;
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
  const card = renderCard(hit, isDark(anchor));
  if (anchor.slot) card.slot = anchor.slot; // stay in the same slot of Reddit's post element
  anchor.after(card);
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
  return (
    img.closest('[slot="post-media-container"]') ||
    img.closest("shreddit-aspect-ratio, gallery-carousel") ||
    img.parentElement
  );
}

function renderCard({ kind, note, diff_png }, dark) {
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
  top.append(el("span", "label", LABELS[kind] || "Fact-checked"), el("span", "brand", NAME));
  card.append(top);
  if (note.headline) card.append(el("p", "headline", note.headline));
  if (note.context) card.append(el("p", "context", note.context));

  const links = el("div", "links");
  links.append(link(note.source_url, `Source: ${note.source_name || new URL(note.source_url).hostname}`));
  if (kind === "edited" && isWebUrl(note.original_url)) links.append(link(note.original_url, "See the original"));
  card.append(links);

  if (kind === "edited" && diff_png) {
    const details = el("details", "diff");
    const img = el("img");
    img.src = diff_png.startsWith("data:") ? diff_png : `data:image/png;base64,${diff_png}`;
    img.alt = "Highlighted differences from the original";
    details.append(el("summary", "", "What changed"), img);
    card.append(details);
  }

  root.append(card);
  return host;
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
