//! Downloading a checkpoint to disk.
//!
//! The download is done by `curl`, which ships with Windows 10 and later and
//! with macOS. That gives resumable HTTPS downloads through the system's own
//! certificate store without the app carrying a TLS stack of its own.

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

fn curl() -> Command {
    // on Windows, ask for the one in System32 by name so a stray curl.cmd is not picked up
    let mut command = Command::new(if cfg!(target_os = "windows") { "curl.exe" } else { "curl" });
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    command.stdin(Stdio::null());
    command
}

/// Downloads `url` to `target`, resuming a part-finished download if one is
/// there. `progress(bytes_on_disk, total)` is called a few times a second;
/// `total` is 0 until the size is known.
pub fn download(url: &str, target: &Path, expected: u64, progress: &mut dyn FnMut(u64, u64)) -> Result<(), String> {
    let insecure = std::env::var_os("NA_ALLOW_HTTP_DOWNLOADS").is_some();
    if !(url.starts_with("https://") || (insecure && url.starts_with("http://"))) {
        return Err("Weights are only downloaded over https.".to_string());
    }
    let part = target.with_extension("part");
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent).map_err(|error| format!("Could not create {}: {}.", parent.display(), error))?;
    }
    let mut command = curl();
    command
        .args(["--location", "--fail", "--silent", "--show-error", "--retry", "3", "--connect-timeout", "30", "--continue-at", "-"])
        .args(["--proto", if insecure { "=https,http" } else { "=https" }, "--proto-redir", if insecure { "=https,http" } else { "=https" }])
        .arg("--output")
        .arg(&part)
        .arg(url)
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let mut child = command.spawn().map_err(|error| {
        format!(
            "This app downloads with the curl program that comes with Windows 10 and macOS, and it could not be started ({}). Download the file in your browser instead and choose it with \u{201c}Use a file\u{201d}.",
            error
        )
    })?;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                let have = std::fs::metadata(&part).map(|meta| meta.len()).unwrap_or(0);
                progress(have, expected.max(have));
                std::thread::sleep(Duration::from_millis(150));
            }
            Err(error) => return Err(format!("The download could not be followed: {}.", error)),
        }
    };
    let mut message = String::new();
    if let Some(mut stderr) = child.stderr.take() {
        use std::io::Read;
        let _ = stderr.read_to_string(&mut message);
    }
    if !status.success() {
        let detail = message.trim().trim_start_matches("curl: ").to_string();
        // 33: the server would not resume; start again next time
        if status.code() == Some(33) || status.code() == Some(36) {
            let _ = std::fs::remove_file(&part);
        }
        return Err(format!(
            "The download failed{}. Check the connection and try again; what has arrived so far is kept and the download carries on from there. Or download the file in your browser and choose it with \u{201c}Use a file\u{201d}.",
            if detail.is_empty() { String::new() } else { format!(" ({})", detail) }
        ));
    }
    let size = std::fs::metadata(&part).map(|meta| meta.len()).unwrap_or(0);
    if size == 0 {
        let _ = std::fs::remove_file(&part);
        return Err("The download finished but nothing arrived.".to_string());
    }
    progress(size, size);
    std::fs::rename(&part, target).map_err(|error| format!("The downloaded file could not be put in place: {}.", error))
}
