//! What the interface can ask of the engine, over HTTP on the loopback
//! address. The page's `NativeClient` (src/core/native-client.js) is the only
//! caller; the method names are the ones the web version's worker answers.
//!
//!   GET  /api/session              held open by the page; the app quits when the last one closes
//!   POST /api/saved      {card}            is this model's checkpoint on disk?
//!   POST /api/load       {card, source}    progress lines, then the model's description
//!   POST /api/load-file  ?card=  <bytes>   a checkpoint handed over by the page
//!   POST /api/forget     {card}            delete the checkpoint on disk
//!   POST /api/tokenize   {card, text}
//!   POST /api/pieces     {card, ids}
//!   POST /api/run        {ids, reset}      a stream of frames: activations, then predictions
//!   POST /api/pick       {temperature, topK}
//!   POST /api/readout    ?k=&target=  <f32 vector>
//!   GET  /api/region     ?tensor=&row0=&col0=&rows=&cols=&transpose=&outRows=&outCols=
//!   GET  /api/stats      ?tensor=&row0=&col0=&rows=&cols=&transpose=
//!   GET  /api/logits     ?position=          every output score at a position, as f32

use crate::http::{Request, Response};
use crate::{download, paths, Assets};
use na_engine::{Engine, Json, ModelInfo, Prediction, RunEvent, View, WeightStore};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct App {
    pub assets: Assets,
    pub data_dir: PathBuf,
    pub origin: String,
    engine: Mutex<Engine>,
    /// the weights of the loaded model, for questions that need not wait for a run
    store: Mutex<Option<Arc<WeightStore>>>,
    /// the newest run number handed out; a run that is not the newest stops
    latest_run: AtomicU64,
    /// extra facts about the loaded model that only this host knows
    extra: Mutex<Vec<(String, Json)>>,
    sessions: AtomicUsize,
    had_session: AtomicBool,
    last_session_end: Mutex<Instant>,
}

type Outcome = Result<Json, String>;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn numbers(values: &[f32]) -> Json {
    Json::Arr(values.iter().map(|&v| Json::Num(v as f64)).collect())
}

fn prediction_json(p: &Prediction) -> Vec<(&'static str, Json)> {
    vec![
        ("ids", Json::Arr(p.summary.ids.iter().map(|&id| Json::uint(id as usize)).collect())),
        ("probs", numbers(&p.summary.probs)),
        ("pieces", Json::Arr(p.pieces.iter().map(|piece| Json::str(piece.clone())).collect())),
        ("entropy", Json::Num(p.summary.entropy as f64)),
        ("targetProb", p.summary.target_prob.map(|v| Json::Num(v as f64)).unwrap_or(Json::Null)),
        ("targetRank", p.summary.target_rank.map(Json::uint).unwrap_or(Json::Num(-1.0))),
    ]
}

fn ids_of(value: Option<&Json>) -> Result<Vec<u32>, String> {
    value
        .and_then(Json::as_array)
        .ok_or("The request has no list of token ids.")?
        .iter()
        .map(|id| id.as_usize().map(|id| id as u32).ok_or_else(|| "A token id is not a whole number.".to_string()))
        .collect()
}

impl App {
    pub fn new(assets: Assets, data_dir: PathBuf, origin: String) -> App {
        App {
            assets,
            data_dir,
            origin,
            engine: Mutex::new(Engine::new()),
            store: Mutex::new(None),
            latest_run: AtomicU64::new(0),
            extra: Mutex::new(Vec::new()),
            sessions: AtomicUsize::new(0),
            had_session: AtomicBool::new(false),
            last_session_end: Mutex::new(Instant::now()),
        }
    }

    /// True once every window that was showing the page has been closed for a moment.
    pub fn abandoned(&self) -> bool {
        self.had_session.load(Ordering::SeqCst) && self.sessions.load(Ordering::SeqCst) == 0 && self.last_session_end.lock().unwrap().elapsed() > Duration::from_millis(2500)
    }

    fn read_asset(&self) -> impl Fn(&str) -> Result<String, String> + '_ {
        move |path: &str| {
            let bytes = self.assets.get(path).ok_or_else(|| format!("The app is missing its file {}.", path))?;
            String::from_utf8(bytes.into_owned()).map_err(|_| format!("The app's file {} is not text.", path))
        }
    }

    /// Where a model's checkpoint is kept on disk.
    fn checkpoint_path(&self, card: &Json) -> PathBuf {
        let id = card.get("id").and_then(Json::as_str).unwrap_or("model");
        let file = card.path(&["weights", "file"]).and_then(Json::as_str).unwrap_or("model.safetensors");
        self.data_dir.join("models").join(paths::safe_name(id)).join(paths::safe_name(file))
    }

    fn describe(&self, info: &ModelInfo) -> Json {
        let mut json = info.to_json();
        if let Json::Obj(pairs) = &mut json {
            pairs.extend(self.extra.lock().unwrap().iter().cloned());
        }
        json
    }

    /// Once a trained model is loaded, measure how surprised it is by plain
    /// English. A correctly wired GPT-2 small scores between 3 and 4 nats per
    /// token; one with a mistake in it, or with random weights, scores near
    /// the logarithm of the vocabulary size (10.8). Shown to the person so
    /// they can see the weights are real and the arithmetic is right.
    fn self_check(&self, engine: &mut Engine, card: &Json) -> Json {
        const PROSE: &str = "The river runs through the middle of the town, and on warm evenings people walk along its banks. There are two bridges, an old stone one near the market and a newer one by the railway station. In winter the water rises, and the path under the old bridge is closed for a few weeks.";
        let read = self.read_asset();
        let result = (|| -> Result<Json, String> {
            let (ids, _) = engine.tokenize(card, PROSE, &read)?;
            let capacity = engine.info().map(|info| info.capacity).unwrap_or(0);
            let ids = &ids[..ids.len().min(capacity)];
            let run = self.latest_run.load(Ordering::SeqCst);
            let (mut nats, mut count) = (0f64, 0usize);
            engine.run(ids, true, run, &self.latest_run, &mut |event| {
                if let RunEvent::Prediction(p) = event {
                    if let Some(prob) = p.summary.target_prob {
                        nats -= (prob.max(1e-30) as f64).ln();
                        count += 1;
                    }
                }
            })?;
            let vocab = engine.info().and_then(|info| info.config.get("vocab_size").and_then(Json::as_usize)).unwrap_or(2) as f64;
            if count == 0 {
                return Err("nothing to measure".to_string());
            }
            let mean = nats / count as f64;
            Ok(Json::obj(vec![("nats", Json::Num(mean)), ("tokens", Json::uint(count)), ("chance", Json::Num(vocab.ln())), ("ok", Json::Bool(mean < vocab.ln() * 0.6))]))
        })();
        // leave the model as a fresh one for the page's own first run
        let _ = engine.run(&[], true, self.latest_run.load(Ordering::SeqCst), &self.latest_run, &mut |_| {});
        result.unwrap_or(Json::Null)
    }

    fn finish_load(&self, engine: &mut Engine, card: &Json, extra: Vec<(&str, Json)>) -> Json {
        *self.store.lock().unwrap() = engine.store();
        let mut extra: Vec<(String, Json)> = extra.into_iter().map(|(key, value)| (key.to_string(), value)).collect();
        if !engine.info().map(|info| info.random).unwrap_or(true) {
            extra.push(("check".to_string(), self.self_check(engine, card)));
        }
        *self.extra.lock().unwrap() = extra;
        self.describe(engine.info().expect("a model was just loaded"))
    }

    fn load(&self, request: &mut Request, out: &mut dyn Write) -> Outcome {
        let body = request.body_json()?;
        let card = body.get("card").ok_or("The request names no model.")?.clone();
        let kind = body.path(&["source", "kind"]).and_then(Json::as_str).unwrap_or("url").to_string();
        let mut say = |phase: &str, loaded: u64, total: u64| {
            let line = Json::obj(vec![("event", Json::str("load")), ("phase", Json::str(phase)), ("loaded", Json::Num(loaded as f64)), ("total", Json::Num(total as f64))]);
            let _ = out.write_all(format!("{}\n", line).as_bytes());
            let _ = out.flush();
        };
        self.latest_run.fetch_add(1, Ordering::SeqCst); // stop a run that is in flight
        let mut engine = self.engine.lock().unwrap();
        *self.store.lock().unwrap() = None;
        engine.unload();
        let read = self.read_asset();
        if kind == "random" {
            say("random", 0, 1);
            let seed = body.path(&["source", "seed"]).and_then(Json::as_usize).or_else(|| card.path(&["weights", "seed"]).and_then(Json::as_usize)).unwrap_or(1) as u32;
            engine.load_random(&card, seed, &read)?;
            return Ok(self.finish_load(&mut engine, &card, vec![("saved", Json::Bool(false))]));
        }
        let path = self.checkpoint_path(&card);
        let mut from = "saved copy";
        if !path.is_file() {
            let url = card.path(&["weights", "url"]).and_then(Json::as_str).ok_or("This model has no download address.")?;
            let expected = card.path(&["weights", "bytes"]).and_then(Json::as_usize).unwrap_or(0) as u64;
            say("download", 0, expected);
            download::download(url, &path, expected, &mut |loaded, total| say("download", loaded, total))?;
            from = "download";
        }
        let size = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
        say("read", 0, size);
        let mut file = std::io::BufReader::with_capacity(4 << 20, std::fs::File::open(&path).map_err(|error| format!("Could not open {}: {}.", path.display(), error))?);
        match engine.load(&card, &mut file, from, &read, &mut |loaded, total| say("read", loaded, total)) {
            Ok(_) => {}
            Err(error) => {
                // a file that cannot be read is no use as a saved copy
                drop(file);
                let _ = std::fs::remove_file(&path);
                return Err(format!("{} The copy on disk was removed; load again to fetch a fresh one.", error));
            }
        }
        Ok(self.finish_load(&mut engine, &card, vec![("saved", Json::Bool(true)), ("path", Json::str(path.display().to_string()))]))
    }

    /// A checkpoint the page hands over as the request body. It is read as it
    /// arrives and, unless asked not to, kept on disk as this model's copy.
    fn load_file(&self, request: &mut Request) -> Outcome {
        let card = Json::parse(request.param("card").ok_or("The request names no model.")?).map_err(|_| "The model card in the request cannot be read.".to_string())?;
        let keep = request.param("keep") != Some("0");
        let path = self.checkpoint_path(&card);
        let part = path.with_extension("upload");
        let mut copy = None;
        if keep {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            copy = std::fs::File::create(&part).ok().map(|file| std::io::BufWriter::with_capacity(4 << 20, file));
        }
        self.latest_run.fetch_add(1, Ordering::SeqCst);
        let mut engine = self.engine.lock().unwrap();
        *self.store.lock().unwrap() = None;
        engine.unload();
        let read = self.read_asset();
        let length = request.content_length;
        // everything read for the engine is also written to the copy
        struct Tee<'a, R: Read> {
            inner: R,
            copy: &'a mut Option<std::io::BufWriter<std::fs::File>>,
        }
        impl<R: Read> Read for Tee<'_, R> {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                let n = self.inner.read(buffer)?;
                if let Some(copy) = self.copy.as_mut() {
                    if copy.write_all(&buffer[..n]).is_err() {
                        *self.copy = None; // the disk is full or gone: go on without a copy
                    }
                }
                Ok(n)
            }
        }
        let outcome = {
            let mut tee = Tee { inner: request.body(), copy: &mut copy };
            let outcome = engine.load(&card, &mut tee, "your file", &read, &mut |_, _| {}).map(|_| ());
            // the engine stops at the last tensor it wants; the copy needs the rest too
            if outcome.is_ok() {
                let _ = std::io::copy(&mut tee, &mut std::io::sink());
            }
            outcome
        };
        let mut saved = false;
        if let Some(mut writer) = copy.take() {
            let complete = outcome.is_ok() && writer.flush().is_ok() && std::fs::metadata(&part).map(|meta| meta.len() == length).unwrap_or(false);
            drop(writer);
            saved = complete && std::fs::rename(&part, &path).is_ok();
        }
        if !saved {
            let _ = std::fs::remove_file(&part);
        }
        outcome?;
        let mut extra = vec![("saved", Json::Bool(saved))];
        if saved {
            extra.push(("path", Json::str(path.display().to_string())));
        }
        Ok(self.finish_load(&mut engine, &card, extra))
    }

    fn run(&self, request: &mut Request, response: Response) {
        let parsed = (|| -> Result<(Vec<u32>, bool), String> {
            let body = request.body_json()?;
            Ok((ids_of(body.get("ids"))?, body.get("reset").and_then(Json::as_bool).unwrap_or(true)))
        })();
        let (ids, reset) = match parsed {
            Ok(parsed) => parsed,
            Err(message) => return response.fail(400, &message, "bad_request"),
        };
        let run = self.latest_run.fetch_add(1, Ordering::SeqCst) + 1;
        let mut engine = self.engine.lock().unwrap();
        let Ok(mut out) = response.stream("application/octet-stream") else { return };
        // a frame: header length, payload length, a JSON header, then float32 payload
        let mut frame = |header: Json, payload: &[u8]| {
            let header = header.to_string().into_bytes();
            let mut bytes = Vec::with_capacity(8 + header.len() + payload.len());
            bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&header);
            bytes.extend_from_slice(payload);
            let _ = out.write_all(&bytes);
        };
        let result = engine.run(&ids, reset, run, &self.latest_run, &mut |event| match event {
            RunEvent::Trace { unit, units, items } => {
                let mut payload = Vec::new();
                let mut list = Vec::new();
                for item in &items {
                    list.push(Json::obj(vec![
                        ("key", Json::str(item.key.clone())),
                        ("rowSize", Json::uint(item.row_size)),
                        ("p0", Json::uint(item.first)),
                        ("n", Json::uint(item.count)),
                        ("offset", Json::uint(payload.len())),
                        ("length", Json::uint(item.data.len())),
                    ]));
                    payload.extend_from_slice(&f32_bytes(&item.data));
                }
                frame(Json::obj(vec![("event", Json::str("trace")), ("run", Json::Num(run as f64)), ("unit", Json::uint(unit)), ("units", Json::uint(units)), ("items", Json::Arr(list))]), &payload);
            }
            RunEvent::Prediction(p) => {
                let mut pairs = vec![("event", Json::str("prediction")), ("run", Json::Num(run as f64)), ("position", Json::uint(p.position))];
                pairs.extend(prediction_json(&p));
                frame(Json::obj(pairs), &[]);
            }
        });
        let last = match result {
            Ok(result) => Json::obj(vec![(
                "result",
                Json::obj(vec![
                    ("run", Json::Num(run as f64)),
                    ("cancelled", Json::Bool(result.cancelled)),
                    ("length", Json::uint(result.length)),
                    ("forwardMs", Json::Num(result.forward_ms)),
                    ("totalMs", Json::Num(result.total_ms)),
                ]),
            )]),
            Err(message) => Json::obj(vec![("error", Json::obj(vec![("message", Json::str(message))]))]),
        };
        frame(last, &[]);
        let _ = out.flush();
    }

    fn view(request: &Request) -> Result<View, String> {
        Ok(View {
            tensor: request.param("tensor").ok_or("The request names no tensor.")?.to_string(),
            row0: request.number("row0").unwrap_or(0),
            col0: request.number("col0").unwrap_or(0),
            rows: request.number("rows")?,
            cols: request.number("cols")?,
            transpose: request.param("transpose") == Some("1"),
        })
    }

    fn weights(&self) -> Result<Arc<WeightStore>, String> {
        self.store.lock().unwrap().clone().ok_or_else(|| "No model is loaded yet.".to_string())
    }

    fn session(&self, response: Response) {
        let Ok(mut out) = response.stream("text/plain") else { return };
        if !self.had_session.swap(true, Ordering::SeqCst) {
            eprintln!("The window has connected to the engine.");
        }
        self.sessions.fetch_add(1, Ordering::SeqCst);
        // one byte a second; the write fails when the window has gone
        while out.write_all(b"\n").and_then(|_| out.flush()).is_ok() {
            std::thread::sleep(Duration::from_millis(1000));
        }
        *self.last_session_end.lock().unwrap() = Instant::now();
        self.sessions.fetch_sub(1, Ordering::SeqCst);
    }

    fn simple(&self, request: &mut Request) -> Outcome {
        let read = self.read_asset();
        match request.path.as_str() {
            "/api/saved" => {
                let body = request.body_json()?;
                let card = body.get("card").ok_or("The request names no model.")?;
                let path = self.checkpoint_path(card);
                Ok(Json::obj(vec![("saved", Json::Bool(path.is_file())), ("path", Json::str(path.display().to_string()))]))
            }
            "/api/forget" => {
                let body = request.body_json()?;
                let card = body.get("card").ok_or("The request names no model.")?;
                let path = self.checkpoint_path(card);
                for file in [path.clone(), path.with_extension("part"), path.with_extension("upload")] {
                    if file.is_file() {
                        std::fs::remove_file(&file).map_err(|error| format!("Could not remove {}: {}.", file.display(), error))?;
                    }
                }
                Ok(Json::Bool(true))
            }
            "/api/tokenize" => {
                let body = request.body_json()?;
                let card = body.get("card").ok_or("The request names no model.")?;
                let text = body.get("text").and_then(Json::as_str).unwrap_or("");
                let (ids, pieces) = self.engine.lock().unwrap().tokenize(card, text, &read)?;
                Ok(Json::obj(vec![
                    ("ids", Json::Arr(ids.iter().map(|&id| Json::uint(id as usize)).collect())),
                    ("pieces", Json::Arr(pieces.into_iter().map(Json::Str).collect())),
                ]))
            }
            "/api/pieces" => {
                let body = request.body_json()?;
                let card = body.get("card").ok_or("The request names no model.")?;
                let pieces = self.engine.lock().unwrap().pieces(card, &ids_of(body.get("ids"))?, &read)?;
                Ok(Json::Arr(pieces.into_iter().map(Json::Str).collect()))
            }
            "/api/pick" => {
                let body = request.body_json()?;
                let temperature = body.get("temperature").and_then(Json::as_f64).unwrap_or(0.0) as f32;
                let top_k = body.get("topK").and_then(Json::as_usize).unwrap_or(40);
                let seed = body.get("seed").and_then(Json::as_usize).map(|seed| seed as u32);
                let (id, piece) = self.engine.lock().unwrap().pick(temperature, top_k, seed)?;
                Ok(Json::obj(vec![("id", Json::uint(id as usize)), ("piece", Json::Str(piece))]))
            }
            "/api/readout" => {
                let bytes = request.body_bytes(1 << 20)?;
                let vector: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                let k = request.number("k").unwrap_or(5).clamp(1, 50);
                let target = request.param("target").and_then(|text| text.parse::<u32>().ok());
                let prediction = self.engine.lock().unwrap().readout(&vector, k, target)?;
                Ok(Json::obj(prediction_json(&prediction)))
            }
            "/api/stats" => {
                let stats = self.weights()?.stats(&Self::view(request)?)?;
                Ok(Json::obj(vec![
                    ("min", Json::Num(stats.min as f64)),
                    ("max", Json::Num(stats.max as f64)),
                    ("mean", Json::Num(stats.mean)),
                    ("std", Json::Num(stats.std)),
                    ("count", Json::uint(stats.count)),
                ]))
            }
            "/api/tensors" => Ok(Json::Arr(
                self.weights()?.names().into_iter().map(|(name, shape)| Json::obj(vec![("name", Json::Str(name)), ("shape", Json::Arr(shape.into_iter().map(Json::uint).collect()))])).collect(),
            )),
            _ => Err("not found".to_string()),
        }
    }

    /// Requests that change something must come from the page this app served.
    /// A page on some other site that guesses the port is turned away.
    fn from_our_page(&self, request: &Request) -> bool {
        let host_ok = request.header("host").map(|host| self.origin.ends_with(&format!("//{}", host))).unwrap_or(false);
        let origin_ok = request.header("origin").map(|origin| origin == self.origin).unwrap_or(true);
        host_ok && origin_ok
    }

    pub fn handle(&self, mut request: Request, response: Response) {
        if !self.from_our_page(&request) {
            return response.fail(403, "This app only answers its own window.", "forbidden");
        }
        let path = request.path.clone();
        if !path.starts_with("/api/") {
            return self.page(&request, response);
        }
        match path.as_str() {
            "/api/session" => self.session(response),
            "/api/run" => self.run(&mut request, response),
            "/api/load" => {
                // progress lines, then one line with the result or the error
                let Ok(mut out) = response.stream("application/x-ndjson") else { return };
                let line = match self.load(&mut request, &mut out) {
                    Ok(info) => Json::obj(vec![("result", info)]),
                    Err(message) => Json::obj(vec![("error", Json::obj(vec![("message", Json::str(message))]))]),
                };
                let _ = out.write_all(format!("{}\n", line).as_bytes());
                let _ = out.flush();
            }
            "/api/load-file" => match self.load_file(&mut request) {
                Ok(info) => response.json(&Json::obj(vec![("result", info)])),
                Err(message) => response.fail(400, &message, "load"),
            },
            "/api/logits" => {
                let result = (|| -> Result<Vec<f32>, String> { self.engine.lock().unwrap().logits_at(request.number("position")?) })();
                match result {
                    Ok(values) => response.send(200, "application/octet-stream", &f32_bytes(&values)),
                    Err(message) => response.fail(400, &message, "logits"),
                }
            }
            "/api/region" => {
                let result = (|| -> Result<Vec<f32>, String> { self.weights()?.region(&Self::view(&request)?, request.number("outRows")?, request.number("outCols")?) })();
                match result {
                    Ok(values) => response.send(200, "application/octet-stream", &f32_bytes(&values)),
                    Err(message) => response.fail(400, &message, "region"),
                }
            }
            _ => match self.simple(&mut request) {
                Ok(value) => response.json(&Json::obj(vec![("result", value)])),
                Err(message) if message == "not found" => response.fail(404, "This app has no such request.", "not_found"),
                Err(message) => response.fail(400, &message, "request"),
            },
        }
    }

    fn page(&self, request: &Request, response: Response) {
        if request.method != "GET" && request.method != "HEAD" {
            return response.fail(405, "Pages are read with GET.", "method");
        }
        let mut path = request.path.trim_start_matches('/').to_string();
        if path.is_empty() {
            path = "index.html".to_string();
        }
        // one plain relative path: no climbing out, and nothing Windows would read as a drive or a separator
        if path.contains(['\\', ':']) || path.split('/').any(|part| part == ".." || part == "." || part.is_empty()) {
            return response.fail(404, "No such file.", "not_found");
        }
        let Some(bytes) = self.assets.get(&path) else {
            return response.fail(404, "No such file.", "not_found");
        };
        let kind = match path.rsplit('.').next().unwrap_or("") {
            "html" => "text/html; charset=utf-8",
            "js" | "mjs" => "text/javascript; charset=utf-8",
            "css" => "text/css; charset=utf-8",
            "json" => "application/json; charset=utf-8",
            "txt" | "md" => "text/plain; charset=utf-8",
            "svg" => "image/svg+xml",
            "png" => "image/png",
            "ico" => "image/x-icon",
            _ => "application/octet-stream",
        };
        if path == "index.html" {
            // tell the page it is running inside the desktop app
            let html = String::from_utf8_lossy(&bytes).replace("<div class=\"na-app\" id=\"app\">", "<div class=\"na-app\" id=\"app\" data-host=\"native\">");
            return response.send(200, kind, html.as_bytes());
        }
        response.send(200, kind, &bytes);
    }
}
