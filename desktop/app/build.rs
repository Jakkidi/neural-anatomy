//! Packs the interface into the program.
//!
//! The interface is the web app one folder up: `index.html`, `src/`,
//! `assets/` and `vendor/`. This script lists those files in a table of
//! `include_bytes!` entries, so the built program carries its own page and
//! needs nothing beside it.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const KINDS: [&str; 8] = ["html", "js", "mjs", "css", "json", "txt", "svg", "png"];

fn walk(dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<PathBuf> = entries.flatten().map(|entry| entry.path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            walk(&path, found);
        } else if path.extension().and_then(|e| e.to_str()).map(|e| KINDS.contains(&e)).unwrap_or(false) {
            found.push(path);
        }
    }
}

fn main() {
    let here = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    // Two folders up, found by name rather than by canonicalize(): on Windows
    // that would give a \\?\ path, which is awkward inside include_bytes!.
    let root = here.parent().and_then(Path::parent).expect("the web app should be two folders above desktop/app").to_path_buf();
    let mut files = vec![root.join("index.html")];
    for dir in ["src", "assets", "vendor"] {
        println!("cargo:rerun-if-changed={}", root.join(dir).display());
        walk(&root.join(dir), &mut files);
    }
    println!("cargo:rerun-if-changed={}", root.join("index.html").display());
    println!("cargo:rerun-if-changed=build.rs");
    assert!(root.join("index.html").is_file(), "index.html was not found in {}", root.display());

    let mut table = String::from("pub static EMBEDDED: &[(&str, &[u8])] = &[\n");
    for file in &files {
        let name = file.strip_prefix(&root).unwrap().components().map(|part| part.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>().join("/");
        // {:?} quotes and escapes the path, which matters on Windows
        writeln!(table, "    ({:?}, include_bytes!({:?})),", name, file.display().to_string()).unwrap();
        println!("cargo:rerun-if-changed={}", file.display());
    }
    table.push_str("];\n");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("embedded.rs");
    std::fs::write(out, table).unwrap();
}
