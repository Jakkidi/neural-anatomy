// Reads the safetensors container: 8 bytes of little-endian header length,
// a JSON header naming every tensor, then one block of raw tensor bytes.
// https://github.com/huggingface/safetensors

const BYTES = { F64: 8, F32: 4, F16: 2, BF16: 2, I64: 8, I32: 4, I16: 2, I8: 1, U8: 1, BOOL: 1, U16: 2, U32: 4, U64: 8 };

export function bytesPerElement(dtype) {
  const size = BYTES[dtype];
  if (!size) throw new Error(`unsupported safetensors dtype ${dtype}`);
  return size;
}

/**
 * @param {Uint8Array} bytes at least the first 8 + headerLength bytes of the file
 * @returns {null | {dataStart:number, metadata:object, tensors:{name:string,dtype:string,shape:number[],begin:number,end:number}[]}}
 *   null when more bytes are needed. Offsets are absolute file offsets.
 */
export function parseHeader(bytes) {
  if (bytes.length < 8) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const length = Number(view.getBigUint64(0, true));
  if (length > 100 * 1024 * 1024) throw new Error("This does not look like a safetensors file.");
  if (bytes.length < 8 + length) return null;
  let header;
  try {
    header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + length)));
  } catch {
    throw new Error("This does not look like a safetensors file: its header cannot be read.");
  }
  const dataStart = 8 + length;
  const tensors = [];
  for (const [name, entry] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const [begin, end] = entry.data_offsets;
    tensors.push({ name, dtype: entry.dtype, shape: entry.shape, begin: dataStart + begin, end: dataStart + end });
  }
  tensors.sort((a, b) => a.begin - b.begin);
  return { dataStart, metadata: header.__metadata__ ?? {}, tensors };
}

/** How many bytes of the file must be read before parseHeader can succeed. */
export function headerBytesNeeded(bytes) {
  if (bytes.length < 8) return 8;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return 8 + Number(view.getBigUint64(0, true));
}

function halfToFloat(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exponent = (h >> 10) & 0x1f;
  const fraction = h & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

/** Converts raw little-endian tensor bytes of any float dtype into `out`. */
export function decodeInto(dtype, raw, out) {
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const n = out.length;
  if (dtype === "F32") for (let i = 0; i < n; i++) out[i] = view.getFloat32(i * 4, true);
  else if (dtype === "F64") for (let i = 0; i < n; i++) out[i] = view.getFloat64(i * 8, true);
  else if (dtype === "F16") for (let i = 0; i < n; i++) out[i] = halfToFloat(view.getUint16(i * 2, true));
  else if (dtype === "BF16") {
    const scratch = new DataView(new ArrayBuffer(4));
    for (let i = 0; i < n; i++) {
      scratch.setUint32(0, view.getUint16(i * 2, true) << 16, true);
      out[i] = scratch.getFloat32(0, true);
    }
  } else throw new Error(`cannot read ${dtype} weights as floats`);
}

/** Builds a safetensors file in memory. Used by the tools and tests. */
export function writeSafetensors(tensors, metadata = { format: "pt" }) {
  const header = { __metadata__: metadata };
  let offset = 0;
  for (const tensor of tensors) {
    const bytes = tensor.data.byteLength;
    header[tensor.name] = { dtype: tensor.dtype ?? "F32", shape: tensor.shape, data_offsets: [offset, offset + bytes] };
    offset += bytes;
  }
  let json = new TextEncoder().encode(JSON.stringify(header));
  const pad = (8 - (json.length % 8)) % 8;
  if (pad) {
    const padded = new Uint8Array(json.length + pad).fill(0x20);
    padded.set(json);
    json = padded;
  }
  const out = new Uint8Array(8 + json.length + offset);
  new DataView(out.buffer).setBigUint64(0, BigInt(json.length), true);
  out.set(json, 8);
  let at = 8 + json.length;
  for (const tensor of tensors) {
    out.set(new Uint8Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength), at);
    at += tensor.data.byteLength;
  }
  return out;
}
