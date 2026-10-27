//! A small JSON reader and writer.
//!
//! The engine reads three JSON documents (a checkpoint header, a vocabulary,
//! and requests from the interface) and writes a few. That does not need a
//! dependency, and having none keeps the build to a single `cargo build`.

use std::fmt::Write as _;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    /// Key order is kept, so a document can be written back as it was read.
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(text: &str) -> Result<Json, String> {
        let mut parser = Parser { bytes: text.as_bytes(), at: 0 };
        parser.space();
        let value = parser.value()?;
        parser.space();
        if parser.at != parser.bytes.len() {
            return Err(format!("unexpected text after the JSON value at byte {}", parser.at));
        }
        Ok(value)
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Json::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Some(*n as usize),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Obj(pairs) => Some(pairs),
            _ => None,
        }
    }

    /// `value["a"]["b"]` for a path of keys; `None` if any step is missing.
    pub fn path(&self, keys: &[&str]) -> Option<&Json> {
        let mut here = self;
        for key in keys {
            here = here.get(key)?;
        }
        Some(here)
    }

    pub fn obj(pairs: Vec<(&str, Json)>) -> Json {
        Json::Obj(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    pub fn str(text: impl Into<String>) -> Json {
        Json::Str(text.into())
    }

    pub fn num(value: impl Into<f64>) -> Json {
        Json::Num(value.into())
    }

    pub fn uint(value: usize) -> Json {
        Json::Num(value as f64)
    }

    pub fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(n) => write_number(*n, out),
            Json::Str(s) => write_string(s, out),
            Json::Arr(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Json::Obj(pairs) => {
                out.push('{');
                for (i, (key, value)) in pairs.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_string(key, out);
                    out.push(':');
                    value.write(out);
                }
                out.push('}');
            }
        }
    }
}

impl std::fmt::Display for Json {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = String::new();
        self.write(&mut out);
        f.write_str(&out)
    }
}

fn write_number(n: f64, out: &mut String) {
    if !n.is_finite() {
        // JSON has no NaN or infinity; null is what JavaScript writes too
        out.push_str("null");
    } else if n.fract() == 0.0 && n.abs() < 9.0e15 {
        let _ = write!(out, "{}", n as i64);
    } else {
        let _ = write!(out, "{}", n);
    }
}

fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Parser<'a> {
    fn space(&mut self) {
        while let Some(b) = self.bytes.get(self.at) {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.at += 1;
            } else {
                break;
            }
        }
    }

    fn fail<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("JSON: {} at byte {}", what, self.at))
    }

    fn value(&mut self) -> Result<Json, String> {
        match self.bytes.get(self.at) {
            None => self.fail("unexpected end"),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.word("true", Json::Bool(true)),
            Some(b'f') => self.word("false", Json::Bool(false)),
            Some(b'n') => self.word("null", Json::Null),
            Some(_) => self.number(),
        }
    }

    fn word(&mut self, word: &str, value: Json) -> Result<Json, String> {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(value)
        } else {
            self.fail("unknown word")
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.at;
        while let Some(b) = self.bytes.get(self.at) {
            if matches!(b, b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
                self.at += 1;
            } else {
                break;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.at]).map_err(|_| "JSON: bad number".to_string())?;
        match text.parse::<f64>() {
            Ok(n) => Ok(Json::Num(n)),
            Err(_) => {
                self.at = start;
                self.fail("bad number")
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        if self.at + 4 > self.bytes.len() {
            return self.fail("short \\u escape");
        }
        let text = std::str::from_utf8(&self.bytes[self.at..self.at + 4]).map_err(|_| "JSON: bad \\u escape".to_string())?;
        let value = u32::from_str_radix(text, 16).map_err(|_| format!("JSON: bad \\u escape at byte {}", self.at))?;
        self.at += 4;
        Ok(value)
    }

    fn string(&mut self) -> Result<String, String> {
        self.at += 1; // opening quote
        let mut out: Vec<u8> = Vec::new();
        loop {
            let start = self.at;
            while self.at < self.bytes.len() && self.bytes[self.at] != b'"' && self.bytes[self.at] != b'\\' {
                self.at += 1;
            }
            out.extend_from_slice(&self.bytes[start..self.at]);
            match self.bytes.get(self.at) {
                None => return self.fail("unterminated string"),
                Some(b'"') => {
                    self.at += 1;
                    return String::from_utf8(out).map_err(|_| "JSON: string is not UTF-8".to_string());
                }
                Some(_) => {
                    self.at += 1;
                    let escape = match self.bytes.get(self.at) {
                        Some(b) => *b,
                        None => return self.fail("unterminated escape"),
                    };
                    self.at += 1;
                    let c = match escape {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let mut code = self.hex4()?;
                            if (0xD800..0xDC00).contains(&code) && self.bytes[self.at..].starts_with(b"\\u") {
                                // a surrogate pair is one character
                                self.at += 2;
                                let low = self.hex4()?;
                                if (0xDC00..0xE000).contains(&low) {
                                    code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                                } else {
                                    return self.fail("unpaired surrogate");
                                }
                            }
                            char::from_u32(code).unwrap_or('\u{FFFD}')
                        }
                        _ => return self.fail("unknown escape"),
                    };
                    let mut buffer = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut buffer).as_bytes());
                }
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.at += 1;
        let mut items = Vec::new();
        self.space();
        if self.bytes.get(self.at) == Some(&b']') {
            self.at += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            self.space();
            items.push(self.value()?);
            self.space();
            match self.bytes.get(self.at) {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return self.fail("expected , or ]"),
            }
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.at += 1;
        let mut pairs = Vec::new();
        self.space();
        if self.bytes.get(self.at) == Some(&b'}') {
            self.at += 1;
            return Ok(Json::Obj(pairs));
        }
        loop {
            self.space();
            if self.bytes.get(self.at) != Some(&b'"') {
                return self.fail("expected a key");
            }
            let key = self.string()?;
            self.space();
            if self.bytes.get(self.at) != Some(&b':') {
                return self.fail("expected :");
            }
            self.at += 1;
            self.space();
            let value = self.value()?;
            pairs.push((key, value));
            self.space();
            match self.bytes.get(self.at) {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Json::Obj(pairs));
                }
                _ => return self.fail("expected , or }"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let text = r#"{"a":[1,2.5,-3e2,true,null],"b":{"c":"x\ny Ġ 👋 \"q\""},"d":""}"#;
        let value = Json::parse(text).unwrap();
        assert_eq!(value.path(&["b", "c"]).unwrap().as_str().unwrap(), "x\ny \u{120} \u{1F44B} \"q\"");
        assert_eq!(value.get("a").unwrap().as_array().unwrap()[2].as_f64().unwrap(), -300.0);
        let again = Json::parse(&value.to_string()).unwrap();
        assert_eq!(value, again);
    }

    #[test]
    fn rejects_garbage() {
        assert!(Json::parse("{").is_err());
        assert!(Json::parse("[1,]").is_err());
        assert!(Json::parse("{} x").is_err());
    }
}
