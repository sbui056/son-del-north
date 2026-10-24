const API = "http://localhost:8000/match";
const MAX_ACTIVE = 3;

const cache = new Map(); // image url -> Promise<match or null>
const queue = [];
let active = 0;
let warned = false;

chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  if (msg?.type !== "check") return;
  check(msg.src).then(reply);
  return true;
});

function check(src) {
  if (!cache.has(src)) cache.set(src, schedule(() => match(src)));
  return cache.get(src);
}

function schedule(task) {
  return new Promise((resolve) => {
    queue.push({ task, resolve });
    next();
  });
}

function next() {
  if (active >= MAX_ACTIVE || !queue.length) return;
  const { task, resolve } = queue.shift();
  active++;
  task()
    .then(resolve, () => resolve(null))
    .finally(() => {
      active--;
      next();
    });
}

async function match(src) {
  try {
    let result = await post(JSON.stringify({ url: src }), { "Content-Type": "application/json" });
    // The backend couldn't download it (hotlink rules, auth), so send the bytes ourselves.
    if (result.error === "fetch_failed") result = await post(await imageForm(src));
    return result.match ? result : null;
  } catch (err) {
    if (!warned) {
      console.warn(`${chrome.runtime.getManifest().name}: backend unreachable (${err.message})`);
      warned = true;
    }
    cache.delete(src); // try again once the backend is back
    return null;
  }
}

async function post(body, headers = {}) {
  const res = await fetch(API, { method: "POST", headers, body });
  return res.json();
}

async function imageForm(src) {
  const res = await fetch(src);
  if (!res.ok) throw new Error(`image fetch ${res.status}`);
  const form = new FormData();
  form.append("image", await res.blob(), "image");
  return form;
}
