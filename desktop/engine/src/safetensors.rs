//! The safetensors container: 8 bytes of little-endian header length, a JSON
//! header naming every tensor, then one block of raw tensor bytes.
//! <https://github.com/huggingface/safetensors>

use crate::json::Json;
use crate::store::WeightStore;
use std::io::{Read, Write};

#[derive(Debug, Clone)]
pub struct Entry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// absolute byte offsets in the file
    pub begin: u64,
    pub end: u64,
}

pub struct Header {
    pub data_start: u64,
    pub tensors: Vec<Entry>,
}

/// One tensor to keep, and the name to keep it under.
pub struct Want {
    pub file: String,
    pub name: String,
}

const HEADER_LIMIT: u64 = 100 * 1024 * 1024;

fn bytes_per(dtype: &str) -> Result<usize, String> {
    Ok(match dtype {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        "I8" | "U8" | "BOOL" => 1,
        other => return Err(format!("This file stores a tensor as {}, which is not a type this app reads.", other)),
    })
}

fn parse_header(json: &str, data_start: u64) -> Result<Header, String> {
    let header = Json::parse(json).map_err(|_| "This does not look like a safetensors file: its header cannot be read.".to_string())?;
    let pairs = header.as_object().ok_or("This does not look like a safetensors file: its header is not an object.")?;
    let mut tensors = Vec::new();
    for (name, entry) in pairs {
        if name == "__metadata__" {
            continue;
        }
        let bad = || format!("This safetensors file has a damaged entry for \"{}\".", name);
        let dtype = entry.get("dtype").and_then(Json::as_str).ok_or_else(bad)?.to_string();
        let shape: Vec<usize> = entry.get("shape").and_then(Json::as_array).ok_or_else(bad)?.iter().map(|d| d.as_usize().ok_or_else(bad)).collect::<Result<_, _>>()?;
        let offsets = entry.get("data_offsets").and_then(Json::as_array).ok_or_else(bad)?;
        if offsets.len() != 2 {
            return Err(bad());
        }
        let begin = offsets[0].as_usize().ok_or_else(bad)? as u64;
        let end = offsets[1].as_usize().ok_or_else(bad)? as u64;
        if end < begin || (end - begin) as usize != shape.iter().product::<usize>() * bytes_per(&dtype)? {
            return Err(bad());
        }
        tensors.push(Entry { name: name.clone(), dtype, shape, begin: data_start + begin, end: data_start + end });
    }
    tensors.sort_by_key(|entry| entry.begin);
    Ok(Header { data_start, tensors })
}

fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exponent = ((h >> 10) & 0x1f) as i32;
    let fraction = (h & 0x3ff) as f32;
    match exponent {
        0 => sign * 2f32.powi(-14) * (fraction / 1024.0),
        31 => if fraction == 0.0 { sign * f32::INFINITY } else { f32::NAN },
        e => sign * 2f32.powi(e - 15) * (1.0 + fraction / 1024.0),
    }
}

fn decode(dtype: &str, raw: &[u8], out: &mut [f32]) -> Result<(), String> {
    match dtype {
        "F32" => {
            for (value, bytes) in out.iter_mut().zip(raw.chunks_exact(4)) {
                *value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            }
        }
        "F64" => {
            for (value, bytes) in out.iter_mut().zip(raw.chunks_exact(8)) {
                *value = f64::from_le_bytes(bytes.try_into().unwrap()) as f32;
            }
        }
        "F16" => {
            for (value, bytes) in out.iter_mut().zip(raw.chunks_exact(2)) {
                *value = half_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
            }
        }
        "BF16" => {
            for (value, bytes) in out.iter_mut().zip(raw.chunks_exact(2)) {
                *value = f32::from_bits((u16::from_le_bytes([bytes[0], bytes[1]]) as u32) << 16);
            }
        }
        other => return Err(format!("A weight is stored as {}, which cannot be read as numbers.", other)),
    }
    Ok(())
}

fn read_fully(reader: &mut dyn Read, buffer: &mut [u8]) -> Result<bool, String> {
    let mut filled = 0;
    while filled < buffer.len() {
        match reader.read(&mut buffer[filled..]) {
            Ok(0) => return Ok(false),
            Ok(n) => filled += n,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(format!("Reading the file failed: {}.", error)),
        }
    }
    Ok(true)
}

/// Reads a safetensors stream into a weight store.
///
/// `plan` sees the header and names the tensors to keep. The rest are read
/// past without being kept, so memory use is the kept weights and no more.
/// `progress(bytes_read, total_bytes)` is called as the stream is consumed.
pub fn read(
    reader: &mut dyn Read,
    plan: &mut dyn FnMut(&Header) -> Result<Vec<Want>, String>,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<WeightStore, String> {
    let mut length = [0u8; 8];
    if !read_fully(reader, &mut length)? {
        return Err("This is not a complete safetensors file.".to_string());
    }
    let header_length = u64::from_le_bytes(length);
    if header_length == 0 || header_length > HEADER_LIMIT {
        return Err("This does not look like a safetensors file.".to_string());
    }
    let mut json = vec![0u8; header_length as usize];
    if !read_fully(reader, &mut json)? {
        return Err("This is not a complete safetensors file.".to_string());
    }
    let json = String::from_utf8(json).map_err(|_| "This does not look like a safetensors file: its header is not text.".to_string())?;
    let header = parse_header(&json, 8 + header_length)?;
    let total = header.tensors.iter().map(|entry| entry.end).max().unwrap_or(header.data_start);

    let wanted = plan(&header)?;
    let mut jobs: Vec<(&Entry, &Want)> = Vec::new();
    for want in &wanted {
        let entry = header.tensors.iter().find(|entry| entry.name == want.file).ok_or_else(|| format!("The checkpoint is missing \"{}\".", want.file))?;
        jobs.push((entry, want));
    }
    jobs.sort_by_key(|(entry, _)| entry.begin);

    let mut store = WeightStore::new();
    let mut position = header.data_start;
    let mut skip = vec![0u8; 1 << 20];
    let mut last_report = 0u64;
    let mut report = |position: u64, force: bool, progress: &mut dyn FnMut(u64, u64)| {
        if force || position - last_report >= 4 << 20 {
            last_report = position;
            progress(position.min(total), total);
        }
    };
    for (entry, want) in jobs {
        // read past whatever lies before this tensor
        while position < entry.begin {
            let step = ((entry.begin - position) as usize).min(skip.len());
            if !read_fully(reader, &mut skip[..step])? {
                return Err(format!("The file ended early: \"{}\" is missing.", entry.name));
            }
            position += step as u64;
            report(position, false, progress);
        }
        if position != entry.begin {
            return Err(format!("This safetensors file has overlapping tensors at \"{}\".", entry.name));
        }
        let count: usize = entry.shape.iter().product();
        let mut data = vec![0f32; count];
        let size = bytes_per(&entry.dtype)?;
        // in pieces, so progress keeps moving through a large tensor
        let mut raw = vec![0u8; (count * size).min(8 << 20) / size.max(1) * size];
        let mut done = 0usize;
        while done < count {
            let n = (count - done).min(raw.len() / size);
            if !read_fully(reader, &mut raw[..n * size])? {
                return Err(format!("The file ended early: \"{}\" is incomplete.", entry.name));
            }
            decode(&entry.dtype, &raw[..n * size], &mut data[done..done + n])?;
            done += n;
            position += (n * size) as u64;
            report(position, false, progress);
        }
        store.insert(&want.name, entry.shape.clone(), data);
    }
    report(total, true, progress);
    Ok(store)
}

/// Writes tensors as a safetensors file (float32). Used by the tests and tools.
pub fn write(out: &mut dyn Write, tensors: &[(&str, &[usize], &[f32])]) -> std::io::Result<()> {
    let mut entries = Vec::new();
    let mut offset = 0usize;
    for (name, shape, data) in tensors {
        let bytes = data.len() * 4;
        entries.push((
            name.to_string(),
            Json::obj(vec![
                ("dtype", Json::str("F32")),
                ("shape", Json::Arr(shape.iter().map(|d| Json::uint(*d)).collect())),
                ("data_offsets", Json::Arr(vec![Json::uint(offset), Json::uint(offset + bytes)])),
            ]),
        ));
        offset += bytes;
    }
    let mut json = Json::Obj(entries).to_string().into_bytes();
    while json.len() % 8 != 0 {
        json.push(b' ');
    }
    out.write_all(&(json.len() as u64).to_le_bytes())?;
    out.write_all(&json)?;
    for (_, _, data) in tensors {
        let mut raw = Vec::with_capacity(data.len() * 4);
        for value in data.iter() {
            raw.extend_from_slice(&value.to_le_bytes());
        }
        out.write_all(&raw)?;
    }
    Ok(())
}
