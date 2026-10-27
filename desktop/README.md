# Neural Anatomy, the desktop app

A Rust program for Windows and macOS. It runs GPT-2 small with a native engine,
keeps the real weights on your disk, and shows the same figures as the web app,
with a layer walk: step from the embedding through the 12 blocks to the output
and watch the machine light up one layer at a time.

The engine and the app depend on nothing outside Rust's standard library, so
building them downloads nothing.

## Run it

### Windows 10 or 11

Double-click **Start Neural Anatomy** in the project folder. The first time, it
installs Rust for your user only (about 300 MB, no admin rights, no Visual
Studio needed) and builds the app, which takes a few minutes in all. After that
it starts in a few seconds. A console window opens first: keep it open while
you use the app, because closing it closes the app. Everything the build does is
written to `desktop\build-log.txt`.

If Windows says "Windows protected your PC", choose **More info**, then
**Run anyway**: the script is not signed.

Then press **Download weights · 548 MB**. See "The weights" below.

To build by hand instead: install Rust from <https://rustup.rs>, then
`cd neural-anatomy\desktop` and `cargo run --release`.

### macOS (and Linux)

Double-click **Start Neural Anatomy.command** in the project folder (on Linux,
run `./"Start Neural Anatomy.command"` in a terminal). If macOS says it cannot
be opened because it is from an unidentified developer, right-click it and
choose **Open**. The first time it:

1. asks to install Apple's command line tools if they are missing (they bring
   the linker); open the launcher again once they are installed;
2. installs Rust for your user only, with rustup (about 300 MB);
3. builds the app, a minute or two.

After that it starts in a few seconds. Keep the Terminal window open while you
use the app. The build's output is in `desktop/build-log.txt`.

If Chrome, Edge, Brave, Chromium or Vivaldi is installed, the app opens in a
window of its own. Otherwise it opens as a tab in your default browser.

By hand instead:

```
xcode-select --install                                          # macOS only, once
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh  # once
cd neural-anatomy/desktop
cargo run --release
```

To quit, close the window. The program in the terminal stops a few seconds
later. `Ctrl+C` in the terminal also stops it.

After the first build, the program itself is
`desktop/target/release/neural-anatomy` (`neural-anatomy.exe` on Windows). It
carries the whole interface inside it, so you can copy that one file anywhere
and start it by double-click. Started that way on Windows, a console window
stays open behind the app; that is the program, and closing it quits.

## The weights

The app uses OpenAI's published GPT-2 small weights: `model.safetensors`,
548,105,171 bytes, from
<https://huggingface.co/openai-community/gpt2>. No account or token is needed.

- **Download weights** fetches the file once, with the `curl` that comes with
  Windows 10/11 and macOS, and keeps it on disk. An interrupted download carries
  on from where it stopped.
- Every later start reads the file from disk, which takes about a second.
- **Use a file** takes a `model.safetensors` you already have (or drop it on the
  window). It is copied into the app's folder so later starts find it.
- **Remove saved copy** deletes it again. Once weights are loaded these buttons
  are folded away under **Change weights**.

Where the file is kept:

| | |
|---|---|
| Windows | `%LOCALAPPDATA%\NeuralAnatomy\models\gpt2\model.safetensors` |
| macOS | `~/Library/Application Support/NeuralAnatomy/models/gpt2/model.safetensors` |

`neural-anatomy --where` prints the folder, and the Weights panel shows the full
path of the file in use.

### Knowing the weights are real

Each time non-random weights are loaded, the app runs a paragraph of plain
English through the model and measures how well it predicts each next word.
The Weights panel then says one of:

- **Self-check passed: 3.x nats per token.** These are trained weights and the
  engine is running them correctly.
- **Self-check failed**, in red, with a number near 10.8. That is the score of
  random numbers. Either the file is not a trained model or something is wrong.

The same check from the terminal, without the window:

```
cargo run --release -- --check
```

It downloads the weights if they are not on disk yet, prints the model's five
most likely continuations of "The Eiffel Tower is located in the city of" with
their probabilities, prints the score on plain English, and ends with `PASS` or
`FAIL`. With the real weights the first continuation is " Paris".
`--check path\to\model.safetensors` checks a file of your own.

## The cabinet and the bench

The app shows the model as a cabinet: one pull-out module per unit, input at
the bottom, the next token printed on top. The plates on each module light up
with what the model computes for your text, and pulling a module out (click it,
or arrow up and down) shows all of its weights.

Click any plate and the camera flies inside, onto the **bench**: that unit's
work as the matrix multiplications it really does, on these numbers. A block is
six steps you can walk along (query/key/value, scores for any head, mixing the
values, joining the heads, MLP up with GELU, MLP down); the embedding is a
one-hot vector × the token table, and the output is the stream × the token
table, 50,257 scores. Click any output to watch its dot product: the products
appear row by row and the running sum climbs to the value, which is checked
against the engine. Zoom all the way in to read every weight and every value.
**All tokens** shows the unit for every token at once, one row per token, with
attention as the only place rows meet; replay it as generation with or without
the KV cache and watch the work each needs. **On a GPU** runs one attention
head as a graphics card does, standard attention against FlashAttention, tile
by tile, with the kernel's code and the memory traffic. **← Cabinet** (or Escape) goes back.

Type anything in the Prompt box: one word, a sentence, or a few paragraphs (the
first 240 tokens are used). Once weights are loaded the model runs by itself
when you stop typing.

## Options

```
neural-anatomy [options]

  --check [file]    check the weights from the terminal (see above)
  --browser         show the interface as a tab in the default browser
  --no-window       open nothing; print the address and wait
  --port <n>        serve on this port (default: any free port)
  --data-dir <dir>  keep downloaded weights here instead
  --where           print the folder weights are kept in
  --stay            keep running after the window is closed
  --ui <dir>        read the interface from a folder (for working on it)
```

With `cargo run`, options go after `--`, as in `cargo run --release -- --browser`.

| Environment variable | |
|---|---|
| `NA_THREADS` | how many threads the engine uses (default: every core, up to 16) |
| `NA_BROWSER` | path of a Chromium-based browser to open the window with |

## How it is put together

```
desktop/
  engine/                 the engine, as a library (crate na-engine)
    src/
      engine.rs           one loaded model and the questions asked of it
      arch.rs             what an architecture must provide; the registry
      gpt2.rs             GPT-2: checkpoint plan and the forward pass, traced
      kernels.rs          the hot loops: threads, AVX2 or NEON
      safetensors.rs      the checkpoint format, read as a stream
      store.rs            weights in memory; pooled regions for the plates
      bpe.rs              GPT-2's byte-level BPE tokenizer
      json.rs  sampling.rs  unicode_tables.rs
    tests/                against numpy, and against published token ids
  app/                    the program (crate neural-anatomy)
    build.rs              packs ../index.html, ../src, ../assets, ../vendor into the binary
    src/
      main.rs             command line
      lib.rs              start(): serve the interface on 127.0.0.1
      api.rs              the requests the page makes of the engine
      http.rs             a small HTTP/1.1 server
      download.rs         weights to disk, with curl
      check.rs            --check
      window.rs           open the app window
      paths.rs
  tauri/                  optional native-window wrapper. Never compiled; see its README
```

The interface is the web app in the folder above, unchanged: the same
`index.html` and `src/`. On the web its engine is a Web Worker
(`src/core/client.js`); here the page finds itself inside the desktop app and
uses `src/core/native-client.js` instead, which makes the same calls to the
Rust engine over HTTP on the loopback address. Activations come back as a
binary stream, a unit at a time, so the machine lights up as the forward pass
moves through it.

The server listens on `127.0.0.1` only, and refuses requests whose `Host` or
`Origin` is not its own address, so another site open in a browser cannot talk
to it.

Speed, measured on a two-core cloud machine with AVX2: about 0.13 s for a
16-token prompt and 25 ms for each token after that, roughly two and a half
times the web app's WebAssembly engine on the same machine.

## Tests

```
cargo test --release
```

- `engine/tests/reference.rs` runs a GPT-2-shaped checkpoint of random numbers
  and compares every traced activation and every logit with an independent
  numpy implementation (`../tools/reference_gpt2.py`). They agree to 2e-6.
  It needs `python3` with numpy to make the reference, and skips if there is none.
- `engine/tests/tokenizer.rs` checks the tokenizer against token ids published
  for GPT-2.
- `node ../tools/compare-tokenizers.mjs` feeds the same 9,242 strings to this
  tokenizer and the web app's and requires identical ids.

## Known limits

- **Checked:** the engine against the numpy reference; the tokenizer; the real
  GPT-2 small weights (`--check` passes: " Paris" first, 2.92 nats per token on
  plain English); the whole app end to end in a Chromium window; building and
  running on a Windows laptop (GNU toolchain). Every push is also built and tested on
  Windows, macOS and Linux by the CI workflow in `.github/workflows/ci.yml`.
- **Safari.** On a Mac with no Chromium-based browser the page opens in Safari.
  The figures were developed in Chromium. If something looks wrong there,
  install Chrome or Edge and start the app again.
- The download needs `curl`. It is part of Windows 10 version 1803 and later
  and of macOS. Without it, download the file in a browser and use **Use a file**.
- One model is on offer, GPT-2 small. Other sizes of GPT-2 (DistilGPT-2, medium,
  large) load through **Use a file**; the machine is rebuilt to their shape.
- The Windows launcher builds with Rust's GNU toolchain, so no Visual Studio is
  needed. If you already have Rust set up for Visual Studio without its C++
  tools, the launcher notices and switches to the GNU toolchain by itself.
