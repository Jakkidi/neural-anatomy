//! Showing the interface in a window of its own.
//!
//! The interface is a page served by this program. Chromium-based browsers
//! can show a page as an app: one window, no tabs, no address bar. Microsoft
//! Edge is part of Windows 10 and 11, so that is always available there; on
//! macOS it is used when Chrome, Edge, Brave, Chromium or Vivaldi is
//! installed, and otherwise the page opens in the default browser.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub enum Opened {
    /// in a window of its own, by the named browser
    AppWindow(String),
    /// as a tab in the default browser
    DefaultBrowser,
    Nothing(String),
}

fn candidates() -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    if cfg!(target_os = "windows") {
        let roots: Vec<PathBuf> = ["ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"].iter().filter_map(|name| std::env::var_os(name)).map(PathBuf::from).collect();
        for (name, tail) in [
            ("Microsoft Edge", r"Microsoft\Edge\Application\msedge.exe"),
            ("Google Chrome", r"Google\Chrome\Application\chrome.exe"),
            ("Brave", r"BraveSoftware\Brave-Browser\Application\brave.exe"),
            ("Chromium", r"Chromium\Application\chrome.exe"),
            ("Vivaldi", r"Vivaldi\Application\vivaldi.exe"),
        ] {
            for root in &roots {
                out.push((name.to_string(), root.join(tail)));
            }
        }
    } else if cfg!(target_os = "macos") {
        let mut roots = vec![PathBuf::from("/Applications")];
        if let Some(home) = std::env::var_os("HOME") {
            roots.push(PathBuf::from(home).join("Applications"));
        }
        for (name, tail) in [
            ("Google Chrome", "Google Chrome.app/Contents/MacOS/Google Chrome"),
            ("Microsoft Edge", "Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
            ("Brave", "Brave Browser.app/Contents/MacOS/Brave Browser"),
            ("Chromium", "Chromium.app/Contents/MacOS/Chromium"),
            ("Vivaldi", "Vivaldi.app/Contents/MacOS/Vivaldi"),
        ] {
            for root in &roots {
                out.push((name.to_string(), root.join(tail)));
            }
        }
    } else {
        let path = std::env::var_os("PATH").unwrap_or_default();
        for name in ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "brave-browser"] {
            for dir in std::env::split_paths(&path) {
                out.push((name.to_string(), dir.join(name)));
            }
        }
    }
    out
}

fn quiet(command: &mut Command) -> &mut Command {
    command.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null())
}

/// Opens `url` in an app window, keeping that browser's own profile for this
/// app under `profile` so it does not touch the person's everyday one.
pub fn open_app_window(url: &str, profile: &Path) -> Option<String> {
    // NA_BROWSER names a browser program to use instead of the ones looked for
    let chosen = std::env::var_os("NA_BROWSER").map(|path| ("the chosen browser".to_string(), PathBuf::from(path)));
    for (name, program) in chosen.into_iter().chain(candidates()) {
        if !program.is_file() {
            continue;
        }
        let mut command = Command::new(&program);
        command
            .arg(format!("--app={}", url))
            .arg(format!("--user-data-dir={}", profile.display()))
            .args(["--no-first-run", "--no-default-browser-check", "--window-size=1500,1000", "--disable-features=Translate"]);
        if quiet(&mut command).spawn().is_ok() {
            return Some(name);
        }
    }
    None
}

pub fn open_default_browser(url: &str) -> bool {
    let mut command = if cfg!(target_os = "windows") {
        let mut command = Command::new("rundll32");
        command.args(["url.dll,FileProtocolHandler", url]);
        command
    } else if cfg!(target_os = "macos") {
        let mut command = Command::new("open");
        command.arg(url);
        command
    } else {
        let mut command = Command::new("xdg-open");
        command.arg(url);
        command
    };
    quiet(&mut command).spawn().is_ok()
}

pub fn open(url: &str, profile: &Path, app_window: bool) -> Opened {
    if app_window {
        if let Some(name) = open_app_window(url, profile) {
            return Opened::AppWindow(name);
        }
    }
    if open_default_browser(url) {
        Opened::DefaultBrowser
    } else {
        Opened::Nothing(format!("No browser could be started. Open {} yourself.", url))
    }
}
