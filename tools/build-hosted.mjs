// Builds the page for a host that wraps it in its own document and forbids
// downloads from other sites (a sandboxed preview). The result is the body of
// index.html with the page marked data-host="sandbox", which makes the bench
// take weights from a file instead of from Hugging Face.
//
//   node tools/build-hosted.mjs            writes dist/neural-anatomy.html and dist/files.json
//   node tools/build-hosted.mjs --preview  also writes hosted-preview.html, a full document for a local look
//
// dist/files.json lists every file the page needs beside it, with its type.

import { mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, extname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const html = readFileSync(join(ROOT, "index.html"), "utf8");
const between = (from, to) => html.slice(html.indexOf(from) + from.length, html.indexOf(to));
const head = between("<!--PAGE-->", "<!--/HEAD-->").trim();
const body = between("<!--BODY-->", "<!--/BODY-->").trim().replace('<div class="na-app" id="app">', '<div class="na-app" id="app" data-host="sandbox">');
if (!body.includes('data-host="sandbox"')) throw new Error("index.html no longer has the #app element this script marks");
const page = `${head}\n${body}\n`;

mkdirSync(join(ROOT, "dist"), { recursive: true });
writeFileSync(join(ROOT, "dist/neural-anatomy.html"), page);

const TYPES = { ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".json": "application/json", ".txt": "text/plain" };
const files = {};
const walk = (dir) => {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) walk(path);
    else if (TYPES[extname(name)]) files[relative(ROOT, path).split("\\").join("/")] = { from: relative(ROOT, path).split("\\").join("/"), contentType: TYPES[extname(name)] };
  }
};
for (const dir of ["src", "assets", "vendor"]) walk(join(ROOT, dir));
writeFileSync(join(ROOT, "dist/files.json"), JSON.stringify(files, null, 2));

if (process.argv.includes("--preview")) {
  writeFileSync(join(ROOT, "hosted-preview.html"), `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>${page}</body></html>`);
}
const bytes = Object.keys(files).reduce((sum, file) => sum + statSync(join(ROOT, file)).size, 0);
console.log(`dist/neural-anatomy.html (${page.length} bytes) and ${Object.keys(files).length} files beside it (${(bytes / 1e6).toFixed(2)} MB)`);
