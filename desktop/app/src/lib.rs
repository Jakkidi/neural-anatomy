//! Neural Anatomy for the desktop.
//!
//! This program is the engine and the host in one: it carries the interface
//! (the same figures as the web app), serves it to a window on this computer
//! only, and answers the interface's questions with a native engine. Weights
//! are downloaded to disk once and read from there afterwards.

pub mod api;
pub mod check;
pub mod download;
pub mod http;
pub mod paths;
pub mod window;

use std::borrow::Cow;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;

include!(concat!(env!("OUT_DIR"), "/embedded.rs"));

/// Where the interface's files come from.
pub enum Assets {
    /// packed into the program when it was built
    Embedded,
    /// read from a folder each time, for working on the interface
    Folder(PathBuf),
}

impl Assets {
    pub fn get(&self, path: &str) -> Option<Cow<'static, [u8]>> {
        match self {
            Assets::Embedded => EMBEDDED.iter().find(|(name, _)| *name == path).map(|(_, bytes)| Cow::Borrowed(*bytes)),
            Assets::Folder(root) => std::fs::read(root.join(path)).ok().map(Cow::Owned),
        }
    }
}

pub struct Options {
    /// 0 picks a free port
    pub port: u16,
    pub data_dir: PathBuf,
    pub ui: Option<PathBuf>,
    /// quit when the last window showing the page has closed
    pub quit_when_closed: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options { port: 0, data_dir: paths::data_dir(), ui: None, quit_when_closed: true }
    }
}

pub struct Running {
    pub url: String,
    pub app: Arc<api::App>,
}

/// Starts serving the interface and returns at once with its address.
pub fn start(options: Options) -> Result<Running, String> {
    let listener = TcpListener::bind(("127.0.0.1", options.port)).map_err(|error| format!("Could not open a local port: {}.", error))?;
    let port = listener.local_addr().map_err(|error| error.to_string())?.port();
    let origin = format!("http://127.0.0.1:{}", port);
    std::fs::create_dir_all(&options.data_dir).map_err(|error| format!("Could not create {}: {}.", options.data_dir.display(), error))?;
    let assets = match options.ui {
        Some(folder) => Assets::Folder(folder),
        None => Assets::Embedded,
    };
    let app = Arc::new(api::App::new(assets, options.data_dir, origin.clone()));
    let handler = app.clone();
    std::thread::Builder::new().name("server".into()).spawn(move || http::serve(listener, move |request, response| handler.handle(request, response))).map_err(|error| error.to_string())?;
    if options.quit_when_closed {
        let watched = app.clone();
        let _ = std::thread::Builder::new().name("watch".into()).spawn(move || loop {
            std::thread::sleep(std::time::Duration::from_millis(500));
            if watched.abandoned() {
                eprintln!("The Neural Anatomy window was closed, so the app is quitting.");
                std::process::exit(0);
            }
        });
    }
    Ok(Running { url: format!("{}/", origin), app })
}
