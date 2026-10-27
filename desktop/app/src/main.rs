//! `neural-anatomy`: start the app and show its window.

use neural_anatomy::{paths, start, window, Options};
use std::path::PathBuf;

const HELP: &str = "\
Neural Anatomy: a language model drawn as a machine you can open.

Usage: neural-anatomy [options]

  --browser         show the interface as a tab in the default browser
                    instead of in a window of its own
  --no-window       open nothing; print the address and wait
  --port <n>        serve on this port (default: any free port)
  --data-dir <dir>  keep downloaded weights here
                    (default: the app's folder in your user profile)
  --ui <dir>        read the interface from this folder instead of the copy
                    built into the program (for working on the interface)
  --stay            keep running after the window is closed
  --where           print the folder weights are kept in, and exit
  --check [file]    without opening a window: load the weights (downloading
                    them first if needed, or reading the given .safetensors
                    file), show what the model predicts, and say whether they
                    are trained weights run correctly
  -h, --help        this text
  -V, --version     the version

Environment:
  NA_THREADS        how many threads the engine uses (default: all cores, up to 16)
  NA_BROWSER        a Chromium-based browser program to open the window with
";

fn main() {
    let mut options = Options::default();
    let (mut app_window, mut show) = (true, true);
    let mut check: Option<Option<PathBuf>> = None;
    let mut args = std::env::args().skip(1).peekable();
    let need = |value: Option<String>, flag: &str| -> String {
        value.unwrap_or_else(|| {
            eprintln!("{} needs a value. Try --help.", flag);
            std::process::exit(2);
        })
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return print!("{}", HELP),
            "-V" | "--version" => return println!("neural-anatomy {}", env!("CARGO_PKG_VERSION")),
            "--where" => return println!("{}", paths::data_dir().display()),
            "--check" => {
                // an optional file follows, unless the next word is another option
                let file = match args.peek() {
                    Some(next) if !next.starts_with("--") => args.next().map(PathBuf::from),
                    _ => None,
                };
                check = Some(file);
            }
            "--browser" => app_window = false,
            "--no-window" => show = false,
            "--stay" => options.quit_when_closed = false,
            "--port" => {
                options.port = need(args.next(), "--port").parse().unwrap_or_else(|_| {
                    eprintln!("--port needs a number.");
                    std::process::exit(2);
                })
            }
            "--data-dir" => options.data_dir = PathBuf::from(need(args.next(), "--data-dir")),
            "--ui" => options.ui = Some(PathBuf::from(need(args.next(), "--ui"))),
            other => {
                eprintln!("Unknown option {}. Try --help.", other);
                std::process::exit(2);
            }
        }
    }
    if let Some(file) = check {
        match neural_anatomy::check::run(&options.data_dir, file) {
            Ok(true) => return,
            Ok(false) => std::process::exit(1),
            Err(message) => {
                eprintln!("{}", message);
                std::process::exit(1);
            }
        }
    }
    if !show {
        options.quit_when_closed = false;
    }
    let profile = options.data_dir.join("window");
    let data_dir = options.data_dir.clone();
    let running = match start(options) {
        Ok(running) => running,
        Err(message) => {
            eprintln!("{}", message);
            std::process::exit(1);
        }
    };
    println!("Neural Anatomy is running at {}", running.url);
    println!("Weights are kept in {}", data_dir.display());
    if show {
        match window::open(&running.url, &profile, app_window) {
            window::Opened::AppWindow(name) => println!("The window is shown by {}. Close it to quit.", name),
            window::Opened::DefaultBrowser => println!("Opened in your default browser. Close the tab to quit."),
            window::Opened::Nothing(message) => println!("{}", message),
        }
    } else {
        println!("Open that address in a browser. Press Ctrl+C here to quit.");
    }
    // the server and the watcher run on their own threads; this one only waits
    loop {
        std::thread::park();
    }
}
