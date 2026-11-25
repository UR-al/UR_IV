// Anima studio: reading a PNG's text chunks (the Forge `parameters` of a gallery image) and extending that infotext.
// Pure ES module without side effects (ComfyUI also loads every .js under web/ as an extension).

/** Forge's infotext quoting: a value with a comma, colon or newline is JSON-quoted. */
function quoteParam(text) {
  return /[,:\n]/.test(text) ? JSON.stringify(text) : text;
}

/** `parameters` text with `extra` keys appended to its `Steps: …` line (Forge's grid infotext = first cell + Script keys). */
export function withExtraParams(text, extra) {
  const base = typeof text === "string" ? text.replace(/\s+$/, "") : "";
  const pairs = Object.entries(extra || {}).map(([key, value]) => `${key}: ${quoteParam(String(value))}`);
  if (!pairs.length) return base;
  if (!base) return pairs.join(", ");
  const lines = base.split("\n");
  if (/^Steps: /.test(lines[lines.length - 1])) lines[lines.length - 1] += `, ${pairs.join(", ")}`;
  else lines.push(pairs.join(", "));
  return lines.join("\n");
}

const PNG_SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];
const LATIN1 = new TextDecoder("latin1");
const MAX_INFLATED = 32 * 1024 * 1024; // all compressed text of one file: a few KB can unfold into gigabytes

/** The inflated bytes; throws when `budget.left` runs out, which skips the chunk. */
async function inflate(bytes, budget) {
  const reader = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate")).getReader();
  const parts = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return new Uint8Array(await new Blob(parts).arrayBuffer());
    budget.left -= value.length;
    if (budget.left < 0) {
      await reader.cancel();
      throw new Error("compressed text too long");
    }
    parts.push(value);
  }
}

async function textChunk(type, data, budget) {
  const keyEnd = data.indexOf(0);
  if (keyEnd < 1) return null;
  const key = LATIN1.decode(data.subarray(0, keyEnd));
  if (type === "tEXt") return [key, LATIN1.decode(data.subarray(keyEnd + 1))];
  if (type === "zTXt") return [key, LATIN1.decode(await inflate(data.subarray(keyEnd + 2), budget))];
  const compressed = data[keyEnd + 1] === 1; // iTXt: flag, method, language\0, translated keyword\0, UTF-8 text
  const languageEnd = data.indexOf(0, keyEnd + 3);
  const translatedEnd = languageEnd < 0 ? -1 : data.indexOf(0, languageEnd + 1);
  if (translatedEnd < 0) return null;
  const body = data.subarray(translatedEnd + 1);
  return [key, new TextDecoder("utf-8").decode(compressed ? await inflate(body, budget) : body)];
}

/** `[type, data]` of each PNG chunk up to IEND; nothing for anything that is not a PNG. */
export function* pngChunks(bytes) {
  if (bytes.length < 8 || PNG_SIGNATURE.some((value, i) => bytes[i] !== value)) return;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let at = 8; at + 8 <= bytes.length; ) {
    const length = view.getUint32(at);
    const type = LATIN1.decode(bytes.subarray(at + 4, at + 8));
    const start = at + 8;
    if (start + length > bytes.length || type === "IEND") return;
    yield [type, bytes.subarray(start, start + length)];
    at = start + length + 4; // + CRC
  }
}

/** `{keyword: text}` of a PNG's tEXt / zTXt / iTXt chunks (first one wins); `{}` for anything that is not a PNG. */
export async function readPngText(buffer) {
  const out = {};
  const budget = { left: MAX_INFLATED };
  for (const [type, data] of pngChunks(new Uint8Array(buffer))) {
    if (type !== "tEXt" && type !== "zTXt" && type !== "iTXt") continue;
    try {
      const pair = await textChunk(type, data, budget);
      if (pair && !(pair[0] in out)) out[pair[0]] = pair[1];
    } catch {
      // a broken chunk is skipped
    }
  }
  return out;
}
