const BYTE_PATTERN = /'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function byteMaps() {
  const bytes = [];
  for (let value = 33; value <= 126; value += 1) bytes.push(value);
  for (let value = 161; value <= 172; value += 1) bytes.push(value);
  for (let value = 174; value <= 255; value += 1) bytes.push(value);
  const codePoints = [...bytes];
  let extra = 0;
  for (let value = 0; value <= 255; value += 1) {
    if (bytes.includes(value)) continue;
    bytes.push(value);
    codePoints.push(256 + extra);
    extra += 1;
  }
  return {
    encode: new Map(bytes.map((value, index) => [value, String.fromCodePoint(codePoints[index])])),
    decode: new Map(codePoints.map((value, index) => [String.fromCodePoint(value), bytes[index]])),
  };
}

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export class TinyTokenizer {
  constructor(config) {
    this.vocab = config.model.vocab;
    this.tokens = [];
    for (const [token, id] of Object.entries(this.vocab)) this.tokens[id] = token;
    this.mergeRanks = new Map(
      config.model.merges.map((pair, rank) => [`${pair[0]}\u0000${pair[1]}`, rank]),
    );
    this.specials = new Map(config.added_tokens.map((token) => [token.content, token.id]));
    this.specialIds = new Set(config.added_tokens.map((token) => token.id));
    this.specialPattern = new RegExp(
      `(${[...this.specials.keys()].map(escapeRegex).join("|")})`,
      "g",
    );
    const maps = byteMaps();
    this.byteEncode = maps.encode;
    this.byteDecode = maps.decode;
    this.encoder = new TextEncoder();
    this.decoder = new TextDecoder("utf-8");
    this.cache = new Map();
  }

  static async load(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`tokenizer download failed (${response.status})`);
    return new TinyTokenizer(await response.json());
  }

  encode(text) {
    const ids = [];
    for (const part of text.split(this.specialPattern)) {
      if (!part) continue;
      if (this.specials.has(part)) {
        ids.push(this.specials.get(part));
        continue;
      }
      for (const match of part.matchAll(BYTE_PATTERN)) {
        const encoded = [...this.encoder.encode(match[0])].map((byte) => this.byteEncode.get(byte)).join("");
        for (const token of this.bpe(encoded)) ids.push(this.vocab[token]);
      }
    }
    return ids;
  }

  bpe(value) {
    if (this.cache.has(value)) return this.cache.get(value);
    let pieces = [...value];
    while (pieces.length > 1) {
      let bestRank = Infinity;
      let bestPair = "";
      for (let index = 0; index < pieces.length - 1; index += 1) {
        const pair = `${pieces[index]}\u0000${pieces[index + 1]}`;
        const rank = this.mergeRanks.get(pair);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestPair = pair;
        }
      }
      if (!Number.isFinite(bestRank)) break;
      const [left, right] = bestPair.split("\u0000");
      const merged = [];
      for (let index = 0; index < pieces.length; index += 1) {
        if (pieces[index] === left && pieces[index + 1] === right) {
          merged.push(left + right);
          index += 1;
        } else {
          merged.push(pieces[index]);
        }
      }
      pieces = merged;
    }
    this.cache.set(value, pieces);
    return pieces;
  }

  decode(ids, skipSpecialTokens = true) {
    const bytes = [];
    let text = "";
    const flush = () => {
      if (!bytes.length) return;
      text += this.decoder.decode(new Uint8Array(bytes));
      bytes.length = 0;
    };
    for (const id of ids) {
      if (this.specialIds.has(id)) {
        if (skipSpecialTokens) continue;
        flush();
        text += this.tokens[id];
        continue;
      }
      for (const character of [...this.tokens[id]]) bytes.push(this.byteDecode.get(character));
    }
    flush();
    return text;
  }

  tokenId(token) {
    const id = this.vocab[token];
    if (id === undefined) throw new Error(`tokenizer is missing ${token}`);
    return id;
  }

  tokenText(id) {
    return this.specialIds.has(id) ? this.tokens[id] : this.decode([id], false);
  }
}
