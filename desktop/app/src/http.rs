//! A small HTTP/1.1 server for one local page.
//!
//! It listens on the loopback address only, answers one request per
//! connection, and can stream a response of unknown length (the connection's
//! end is the end of the body). That is all the interface needs, and it keeps
//! the app free of dependencies.

use na_engine::Json;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

const HEAD_LIMIT: usize = 64 * 1024;

pub struct Request {
    pub method: String,
    pub path: String,
    pub query: Vec<(String, String)>,
    pub headers: Vec<(String, String)>,
    pub content_length: u64,
    reader: BufReader<TcpStream>,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(key, _)| key.eq_ignore_ascii_case(name)).map(|(_, value)| value.as_str())
    }

    pub fn param(&self, name: &str) -> Option<&str> {
        self.query.iter().find(|(key, _)| key == name).map(|(_, value)| value.as_str())
    }

    pub fn number(&self, name: &str) -> Result<usize, String> {
        self.param(name).and_then(|text| text.parse().ok()).ok_or_else(|| format!("The request is missing the number \"{}\".", name))
    }

    /// The body as a stream of exactly `content_length` bytes.
    pub fn body(&mut self) -> impl Read + '_ {
        let length = self.content_length;
        (&mut self.reader).take(length)
    }

    pub fn body_bytes(&mut self, limit: u64) -> Result<Vec<u8>, String> {
        if self.content_length > limit {
            return Err("The request is too large.".to_string());
        }
        let mut bytes = Vec::with_capacity(self.content_length as usize);
        self.body().read_to_end(&mut bytes).map_err(|error| format!("The request was cut short: {}.", error))?;
        if bytes.len() as u64 != self.content_length {
            return Err("The request was cut short.".to_string());
        }
        Ok(bytes)
    }

    pub fn body_json(&mut self) -> Result<Json, String> {
        let bytes = self.body_bytes(8 << 20)?;
        let text = String::from_utf8(bytes).map_err(|_| "The request is not text.".to_string())?;
        Json::parse(&text).map_err(|error| format!("The request is not valid JSON: {}.", error))
    }
}

pub struct Response {
    stream: TcpStream,
}

impl Response {
    fn head(&mut self, status: u16, content_type: &str, length: Option<usize>) -> std::io::Result<()> {
        let reason = match status {
            200 => "OK",
            400 => "Bad Request",
            403 => "Forbidden",
            404 => "Not Found",
            405 => "Method Not Allowed",
            _ => "Error",
        };
        let mut head = format!("HTTP/1.1 {} {}\r\nContent-Type: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n", status, reason, content_type);
        match length {
            Some(length) => head.push_str(&format!("Content-Length: {}\r\n", length)),
            None => head.push_str("Transfer-Encoding: chunked\r\n"),
        }
        head.push_str("\r\n");
        self.stream.write_all(head.as_bytes())
    }

    pub fn send(mut self, status: u16, content_type: &str, body: &[u8]) {
        if self.head(status, content_type, Some(body.len())).is_ok() {
            let _ = self.stream.write_all(body);
        }
        let _ = self.stream.flush();
    }

    pub fn json(self, value: &Json) {
        self.send(200, "application/json", value.to_string().as_bytes());
    }

    /// An error the page can show: `{ "error": { "message", "code" } }`.
    pub fn fail(self, status: u16, message: &str, code: &str) {
        let body = Json::obj(vec![("error", Json::obj(vec![("message", Json::str(message)), ("code", Json::str(code))]))]);
        self.send(status, "application/json", body.to_string().as_bytes());
    }

    /// Starts a body of unknown length, sent in chunks. It ends, with the mark
    /// that says it is complete, when the returned writer is dropped.
    pub fn stream(mut self, content_type: &str) -> std::io::Result<Body> {
        self.head(200, content_type, None)?;
        self.stream.flush()?;
        Ok(Body { stream: self.stream })
    }
}

/// The writer of a chunked body: each `write` goes out as one chunk.
pub struct Body {
    stream: TcpStream,
}

impl Write for Body {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.is_empty() {
            return Ok(0);
        }
        // one buffer, so a small chunk leaves in a single packet
        let mut chunk = Vec::with_capacity(bytes.len() + 20);
        chunk.extend_from_slice(format!("{:x}\r\n", bytes.len()).as_bytes());
        chunk.extend_from_slice(bytes);
        chunk.extend_from_slice(b"\r\n");
        self.stream.write_all(&chunk)?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.stream.flush()
    }
}

impl Drop for Body {
    fn drop(&mut self) {
        let _ = self.stream.write_all(b"0\r\n\r\n");
        let _ = self.stream.flush();
    }
}

fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let hex = |byte: u8| (byte as char).to_digit(16).map(|digit| digit as u8);
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => match (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                (Some(high), Some(low)) => {
                    out.push(high * 16 + low);
                    i += 3;
                }
                _ => {
                    out.push(b'%');
                    i += 1;
                }
            },
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            byte => {
                out.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn read_request(stream: TcpStream) -> Result<Request, String> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    let mut total = 0usize;
    let mut next = |reader: &mut BufReader<TcpStream>, line: &mut String| -> Result<bool, String> {
        line.clear();
        let n = reader.read_line(line).map_err(|error| error.to_string())?;
        total += n;
        if n == 0 || total > HEAD_LIMIT {
            return Err("bad request".to_string());
        }
        Ok(!line.trim_end_matches(['\r', '\n']).is_empty())
    };
    if !next(&mut reader, &mut line)? {
        return Err("empty request".to_string());
    }
    let mut parts = line.split_whitespace();
    let (method, target) = (parts.next().unwrap_or("").to_string(), parts.next().unwrap_or("").to_string());
    let (path, query_text) = target.split_once('?').unwrap_or((target.as_str(), ""));
    let query = query_text.split('&').filter(|pair| !pair.is_empty()).map(|pair| {
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
        (percent_decode(key), percent_decode(value))
    }).collect();
    let path = percent_decode(path);
    let mut headers = Vec::new();
    while next(&mut reader, &mut line)? {
        if let Some((key, value)) = line.split_once(':') {
            headers.push((key.trim().to_string(), value.trim().to_string()));
        }
    }
    let content_length = headers.iter().find(|(key, _)| key.eq_ignore_ascii_case("content-length")).and_then(|(_, value)| value.parse().ok()).unwrap_or(0);
    Ok(Request { method, path, query, headers, content_length, reader })
}

/// Serves until the process ends. Each connection gets its own thread.
pub fn serve<F>(listener: TcpListener, handler: F)
where
    F: Fn(Request, Response) + Send + Sync + 'static,
{
    let handler = Arc::new(handler);
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let handler = handler.clone();
        let _ = std::thread::Builder::new().name("request".into()).spawn(move || {
            let _ = stream.set_nodelay(true);
            let (Ok(writer), Ok(mut closer)) = (stream.try_clone(), stream.try_clone()) else { return };
            match read_request(stream) {
                Ok(request) => handler(request, Response { stream: writer }),
                Err(_) => Response { stream: writer }.fail(400, "That was not a request this app understands.", "bad_request"),
            }
            // Close politely: say we are done writing, then wait a moment for the
            // other side to close. Closing at once can cut off the last bytes.
            let _ = closer.shutdown(std::net::Shutdown::Write);
            let _ = closer.set_read_timeout(Some(std::time::Duration::from_millis(400)));
            let mut rest = [0u8; 256];
            while matches!(closer.read(&mut rest), Ok(n) if n > 0) {}
        });
    }
}
