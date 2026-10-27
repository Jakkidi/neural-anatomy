//! Where the app keeps its files.

use std::path::PathBuf;

/// The per-user folder for downloaded weights:
/// `%LOCALAPPDATA%\NeuralAnatomy` on Windows,
/// `~/Library/Application Support/NeuralAnatomy` on macOS,
/// `$XDG_DATA_HOME/neural-anatomy` or `~/.local/share/neural-anatomy` elsewhere.
pub fn data_dir() -> PathBuf {
    let var = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty()).map(PathBuf::from);
    if cfg!(target_os = "windows") {
        if let Some(base) = var("LOCALAPPDATA").or_else(|| var("APPDATA")) {
            return base.join("NeuralAnatomy");
        }
    } else if cfg!(target_os = "macos") {
        if let Some(home) = var("HOME") {
            return home.join("Library").join("Application Support").join("NeuralAnatomy");
        }
    } else if let Some(base) = var("XDG_DATA_HOME") {
        return base.join("neural-anatomy");
    } else if let Some(home) = var("HOME") {
        return home.join(".local").join("share").join("neural-anatomy");
    }
    std::env::temp_dir().join("neural-anatomy")
}

/// A name that is safe as one path component, whatever a model card says.
pub fn safe_name(text: &str) -> String {
    let cleaned: String = text.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '_' }).collect();
    let trimmed = cleaned.trim_matches('.');
    if trimmed.is_empty() { "model".to_string() } else { trimmed.chars().take(80).collect() }
}
