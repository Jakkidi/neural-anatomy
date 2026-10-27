// Where weight bytes come from: a URL (kept in the browser's private file
// system after the first download), or a file the person picked.
// Every source resolves to { stream, total, from }.

const FOLDER = "neural-anatomy-weights";

const keyOf = (url) => url.replace(/[^a-zA-Z0-9._-]+/g, "_").slice(-180);

async function folder(create) {
  if (typeof navigator === "undefined" || !navigator.storage?.getDirectory) return null;
  try {
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle(FOLDER, { create });
  } catch {
    return null;
  }
}

async function savedCopy(url) {
  const dir = await folder(false);
  if (!dir) return null;
  try {
    const key = keyOf(url);
    const marker = await (await dir.getFileHandle(`${key}.ok`)).getFile();
    const size = Number(await marker.text());
    const file = await (await dir.getFileHandle(key)).getFile();
    if (!size || file.size !== size) return null;
    return { stream: file.stream(), total: size, from: "saved copy" };
  } catch {
    return null;
  }
}

/** Passes a stream through unchanged while writing it to the private file system. */
async function keepCopy(url, stream) {
  const dir = await folder(true);
  if (!dir) return { stream, saved: Promise.resolve(false) };
  let access, key;
  try {
    key = keyOf(url);
    await dir.removeEntry(`${key}.ok`).catch(() => {});
    const handle = await dir.getFileHandle(key, { create: true });
    if (!handle.createSyncAccessHandle) return { stream, saved: Promise.resolve(false) };
    access = await handle.createSyncAccessHandle();
    access.truncate(0);
  } catch {
    return { stream, saved: Promise.resolve(false) };
  }
  let at = 0, failed = false, finish;
  const saved = new Promise((resolve) => (finish = resolve));
  const close = async (complete) => {
    try { access.flush(); access.close(); } catch { failed = true; }
    if (complete && !failed) {
      try {
        const marker = await dir.getFileHandle(`${key}.ok`, { create: true });
        const writer = await marker.createSyncAccessHandle();
        writer.truncate(0);
        writer.write(new TextEncoder().encode(String(at)), { at: 0 });
        writer.flush();
        writer.close();
        finish(true);
        return;
      } catch { /* fall through */ }
    }
    await dir.removeEntry(key).catch(() => {});
    finish(false);
  };
  const reader = stream.getReader();
  const through = new ReadableStream({
    async pull(controller) {
      const { done, value } = await reader.read();
      if (done) {
        await close(true);
        controller.close();
        return;
      }
      if (!failed) {
        try { access.write(value, { at }); } catch { failed = true; }
      }
      at += value.length;
      controller.enqueue(value);
    },
    async cancel(reason) {
      await reader.cancel(reason).catch(() => {});
      await close(false);
    },
  });
  return { stream: through, saved };
}

export async function openUrl(url, { expectedBytes = 0, signal } = {}) {
  const hit = await savedCopy(url);
  if (hit) return { ...hit, saved: Promise.resolve(true) };
  let response;
  try {
    response = await fetch(url, { signal, mode: "cors", credentials: "omit" });
  } catch (error) {
    const blocked = new Error(`Could not reach ${new URL(url, "http://x").host}. The network, or this page's host, blocked the download.`);
    blocked.code = "blocked";
    blocked.cause = error;
    throw blocked;
  }
  if (!response.ok || !response.body) throw new Error(`The download failed: ${response.status} ${response.statusText}`);
  const total = Number(response.headers.get("content-length")) || expectedBytes;
  const kept = await keepCopy(url, response.body);
  return { stream: kept.stream, total, from: "download", saved: kept.saved };
}

export function openFile(file) {
  return { stream: file.stream(), total: file.size, from: "your file", saved: Promise.resolve(false) };
}

export async function savedList() {
  const dir = await folder(false);
  if (!dir) return [];
  const out = [];
  try {
    for await (const [name, handle] of dir.entries()) {
      if (name.endsWith(".ok")) continue;
      const file = await handle.getFile();
      out.push({ name, bytes: file.size });
    }
  } catch { /* nothing saved */ }
  return out;
}

export async function hasSaved(url) {
  return Boolean(await savedCopy(url).then((hit) => { hit?.stream.cancel().catch(() => {}); return hit; }));
}

export async function forgetSaved() {
  if (typeof navigator === "undefined" || !navigator.storage?.getDirectory) return;
  try {
    const root = await navigator.storage.getDirectory();
    await root.removeEntry(FOLDER, { recursive: true });
  } catch { /* nothing to forget */ }
}
