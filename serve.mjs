#!/usr/bin/env node
// A static file server for this folder, with no dependencies.
//   node serve.mjs [port]
// The app is plain files; any static server works. This one exists so that
// nothing needs installing.

import { createReadStream, statSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, normalize, sep } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.argv[2] ?? process.env.PORT ?? 4173);
const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8", ".txt": "text/plain; charset=utf-8",
  ".svg": "image/svg+xml", ".png": "image/png", ".wasm": "application/wasm", ".md": "text/markdown; charset=utf-8",
  ".safetensors": "application/octet-stream",
};

createServer((request, response) => {
  let path;
  try {
    path = normalize(decodeURIComponent(new URL(request.url, "http://x").pathname));
  } catch {
    response.writeHead(400).end();
    return;
  }
  let file = join(ROOT, path);
  if (file !== ROOT && !file.startsWith(ROOT + sep)) { response.writeHead(403).end(); return; }
  let stat;
  try {
    stat = statSync(file);
    if (stat.isDirectory()) { file = join(file, "index.html"); stat = statSync(file); }
  } catch {
    response.writeHead(404, { "content-type": "text/plain" }).end("not found");
    return;
  }
  response.writeHead(200, {
    "content-type": TYPES[extname(file)] ?? "application/octet-stream",
    "content-length": stat.size,
    "cache-control": "no-cache",
    "access-control-allow-origin": "*",
  });
  if (request.method === "HEAD") { response.end(); return; }
  createReadStream(file).pipe(response);
}).listen(PORT, "127.0.0.1", () => {
  console.log(`Neural Anatomy is at http://localhost:${PORT}`);
});
