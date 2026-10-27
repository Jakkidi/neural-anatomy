//! Neural Anatomy in a native window.
//!
//! The engine and the page are the ones in ../app: this program starts that
//! local server and shows its page in the system's web view. The page talks
//! to the engine over the loopback address exactly as it does in ../app, so
//! it is given none of Tauri's own commands.

// no console window behind the app on Windows
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use neural_anatomy::{start, Options};
use tauri::{WebviewUrl, WebviewWindowBuilder};

fn main() {
    // Tauri ends the program when the window closes, so the server need not watch for that
    let options = Options { quit_when_closed: false, ..Options::default() };
    let running = match start(options) {
        Ok(running) => running,
        Err(message) => {
            eprintln!("{}", message);
            std::process::exit(1);
        }
    };
    let url: tauri::Url = running.url.parse().expect("the app's own address is a valid URL");
    tauri::Builder::default()
        .setup(move |app| {
            WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url))
                .title("Neural Anatomy")
                .inner_size(1500.0, 1000.0)
                .min_inner_size(900.0, 620.0)
                .build()?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("the window could not be opened");
}
