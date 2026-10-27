# Neural Anatomy in a native window (optional)

**This folder was written but never compiled.** The machine the project was
built on could not download crates, and Tauri is a crate. Everything else in
`desktop/` was built and tested; this was not. If it does not build, use
`desktop/app`, which does not depend on it.

It is the same app as `desktop/app` with one difference: the page is shown in
the operating system's own web view (WebView2 on Windows, WKWebView on macOS)
rather than in an Edge or Chrome app window, and `cargo tauri build` produces an
installer (`.msi`/`.exe` on Windows, `.app`/`.dmg` on macOS).

```
cargo install tauri-cli --version "^2"     # once
cd desktop/tauri
cargo tauri dev                            # run it
cargo tauri build                          # make an installer
```

Plain `cargo run --release` in this folder also starts it.

Things to know:

- On macOS the page runs in WKWebView, Safari's engine. The figures were only
  ever tested in Chromium, so this is the less certain of the two platforms.
  On Windows the web view is Chromium (WebView2), the engine they were tested in.
- Weights are still fetched by the `curl` that comes with the operating system
  and kept in the same folder as `desktop/app` uses, so the two share one copy.
- `icons/` was drawn by a script from the mark in the page header.
