// Byte-level BPE, the tokenizer of GPT-2 (and RoBERTa, BART, ...).
// A port of OpenAI's encoder.py: text is split into word-ish chunks, each
// chunk becomes UTF-8 bytes mapped to printable characters, and adjacent
// pairs are merged in the order the merges file ranks them.

const CHUNKS = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function byteAlphabet() {
  const bytes = [];
  for (let b = 33; b <= 126; b++) bytes.push(b);
  for (let b = 161; b <= 172; b++) bytes.push(b);
  for (let b = 174; b <= 255; b++) bytes.push(b);
  const chars = bytes.slice();
  const printable = new Set(bytes);
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (printable.has(b)) continue;
    bytes.push(b);
    chars.push(256 + n++);
  }
  const encode = new Array(256), decode = new Map();
  bytes.forEach((b, i) => {
    const ch = String.fromCodePoint(chars[i]);
    encode[b] = ch;
    decode.set(ch, b);
  });
  return { encode, decode };
}

export class ByteBPE {
  /**
   * @param {Record<string, number>} vocab token text -> id (vocab.json / encoder.json)
   * @param {string} merges the merges file (merges.txt / vocab.bpe)
   * @param {{special?: Record<string, number>}} [options] tokens matched literally in the text
   */
  constructor(vocab, merges, { special = {} } = {}) {
    // Maps, not plain objects: "toString" and "constructor" are ordinary tokens.
    this.vocab = new Map(Object.entries(vocab));
    this.size = 0;
    this.byId = [];
    for (const [text, id] of this.vocab) {
      this.byId[id] = text;
      if (id + 1 > this.size) this.size = id + 1;
    }
    this.ranks = new Map();
    let rank = 0;
    for (const line of merges.split("\n")) {
      if (!line || line.startsWith("#version")) continue;
      const space = line.indexOf(" ");
      if (space < 0) continue;
      this.ranks.set(line.slice(0, space) + "\u0000" + line.slice(space + 1), rank++);
    }
    this.special = new Map(Object.entries(special));
    this.specialIds = new Set(Object.values(special));
    const names = Object.keys(special).map((name) => name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    this.splitter = names.length ? new RegExp(`(${names.join("|")})`, "g") : null;
    this.alphabet = byteAlphabet();
    this.cache = new Map();
    this.utf8 = new TextEncoder();
    this.text = new TextDecoder("utf-8");
  }

  #bpe(chunk) {
    const cached = this.cache.get(chunk);
    if (cached) return cached;
    let word = Array.from(this.utf8.encode(chunk), (b) => this.alphabet.encode[b]);
    while (word.length > 1) {
      let best = Infinity, at = -1;
      for (let i = 0; i < word.length - 1; i++) {
        const r = this.ranks.get(word[i] + "\u0000" + word[i + 1]);
        if (r !== undefined && r < best) { best = r; at = i; }
      }
      if (at < 0) break;
      const first = word[at], second = word[at + 1];
      const merged = [];
      for (let i = 0; i < word.length; i++) {
        if (i < word.length - 1 && word[i] === first && word[i + 1] === second) {
          merged.push(first + second);
          i++;
        } else merged.push(word[i]);
      }
      word = merged;
    }
    const ids = word.map((piece) => {
      const id = this.vocab.get(piece);
      if (id === undefined) throw new Error(`tokenizer: no id for piece ${JSON.stringify(piece)}`);
      return id;
    });
    if (this.cache.size < 50000) this.cache.set(chunk, ids);
    return ids;
  }

  /** @returns {number[]} */
  encode(text) {
    const ids = [];
    const parts = this.splitter ? text.split(this.splitter) : [text];
    for (const part of parts) {
      if (!part) continue;
      if (this.special.has(part)) { ids.push(this.special.get(part)); continue; }
      for (const match of part.matchAll(CHUNKS)) for (const id of this.#bpe(match[0])) ids.push(id);
    }
    return ids;
  }

  #bytesOf(id) {
    const piece = this.byId[id];
    if (piece === undefined) return new Uint8Array(0);
    if (this.specialIds.has(id)) return this.utf8.encode(piece);
    const out = new Uint8Array([...piece].length);
    let n = 0;
    for (const ch of piece) out[n++] = this.alphabet.decode.get(ch) ?? 63;
    return out;
  }

  decode(ids) {
    const chunks = ids.map((id) => this.#bytesOf(id));
    const all = new Uint8Array(chunks.reduce((sum, chunk) => sum + chunk.length, 0));
    let at = 0;
    for (const chunk of chunks) { all.set(chunk, at); at += chunk.length; }
    return this.text.decode(all);
  }

  /** The text of one token on its own. A token that is half a character shows as the replacement character. */
  piece(id) {
    return this.text.decode(this.#bytesOf(id));
  }
}
