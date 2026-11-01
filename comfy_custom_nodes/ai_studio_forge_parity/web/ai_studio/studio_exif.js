// Anima studio: what an image file says about itself, for the EXIF panel (Forge's PNG Info): PNG text chunks and the EXIF
// block of JPEG, WebP and PNG files. Forge keeps `parameters` in EXIF UserComment when it saves JPEG or WebP.
// Pure ES module without side effects (ComfyUI also loads every .js under web/ as an extension).
import { pngChunks, readPngText } from "./studio_pnginfo.js";

// The infotext of a file, in the order it should show; everything else follows in file order.
export const INFOTEXT_KEYS = Object.freeze(["parameters", "UserComment", "Comment"]);

const TEXT_TAGS = new Map([
  [0x010e, "ImageDescription"],
  [0x010f, "Make"], // ComfyUI's WebP: "workflow:{...}"
  [0x0110, "Model"], // ComfyUI's WebP: "prompt:{...}"
  [0x0131, "Software"],
  [0x0132, "DateTime"],
  [0x013b, "Artist"],
  [0x8298, "Copyright"],
  [0x9003, "DateTimeOriginal"],
]);
const EXIF_IFD = 0x8769;
const USER_COMMENT = 0x9286;
const BYTE_TYPES = new Set([1, 2, 7]); // BYTE, ASCII, UNDEFINED: `count` is the length in bytes
const EXIF_HEADER = "Exif\0\0";
const SOS = 0xda; // the image data starts
const EOI = 0xd9;
const APP1 = 0xe1;
const COM = 0xfe;
// A dropped file can be anything, and the reading runs on the page's own thread. The limits of one file:
const MAX_ENTRIES = 256; // text items
const MAX_TEXT_BYTES = 32 * 1024 * 1024; // their sum: many entries can share one value, so the file's size is no bound
const MAX_IFD_STEPS = 100000; // IFD entries looked at: IFDs can point into each other's bytes

const LATIN1 = new TextDecoder("latin1");
const UTF8 = new TextDecoder("utf-8");
const newBudget = () => ({ entries: MAX_ENTRIES, bytes: MAX_TEXT_BYTES, steps: MAX_IFD_STEPS });

/** Pays for one value before it is decoded; false when it no longer fits. */
function pay(budget, bytes) {
  if (budget.entries <= 0 || budget.bytes < bytes) return false;
  budget.entries -= 1;
  budget.bytes -= bytes;
  return true;
}

/** `text` without trailing NUL and whitespace (no regex: `[\0\s]+$` is quadratic on a long run inside the text). */
function clean(text) {
  let end = text.length;
  while (end > 0 && (text[end - 1] === "\0" || text[end - 1].trim() === "")) end -= 1;
  return text.slice(0, end);
}

const hasExifHeader = (bytes) => LATIN1.decode(bytes.subarray(0, 6)) === EXIF_HEADER;
const dataView = (bytes) => new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

/**
 * A byte order mark decides, TextDecoder drops it. Without one the TIFF's order holds (Forge: piexif, big-endian both), unless
 * zero bytes on one side fill over half the code units: that is mostly ASCII text from a writer of the other order. CJK text
 * has stray zero bytes too (가 is U+AC00), so a plain majority would misread it.
 */
function utf16(bytes, little) {
  if (bytes[0] === 0xff && bytes[1] === 0xfe) return new TextDecoder("utf-16le").decode(bytes);
  if (bytes[0] === 0xfe && bytes[1] === 0xff) return new TextDecoder("utf-16be").decode(bytes);
  const sample = Math.min(bytes.length, 512);
  const zeros = [0, 0];
  for (let i = 0; i < sample; i++) if (bytes[i] === 0) zeros[i % 2] += 1;
  const mostly = sample / 4; // half of sample / 2 code units
  const useLittle = zeros[1] > mostly ? true : zeros[0] > mostly ? false : little;
  return new TextDecoder(useLittle ? "utf-16le" : "utf-16be").decode(bytes);
}

/** UserComment: an 8-byte charset code, then the text; some tools leave the code out. */
function userComment(bytes, little) {
  const code = LATIN1.decode(bytes.subarray(0, 8));
  if (code === "UNICODE\0") return utf16(bytes.subarray(8), little);
  if (code === "JIS\0\0\0\0\0") return new TextDecoder("shift_jis").decode(bytes.subarray(8)); // as piexif writes it
  return UTF8.decode(/^(ASCII\0{3}|\0{8})$/.test(code) ? bytes.subarray(8) : bytes);
}

/** `{key, text}` of a TIFF block's text tags (IFD 0 and its Exif IFD); a broken block gives what was read before the break. */
function tiffEntries(tiff, budget) {
  const entries = [];
  const order = LATIN1.decode(tiff.subarray(0, 2));
  if (order !== "II" && order !== "MM") return entries;
  const little = order === "II";
  const view = dataView(tiff);
  const seen = new Set();
  const walk = (ifd) => {
    if (seen.has(ifd)) return;
    seen.add(ifd);
    const count = view.getUint16(ifd, little);
    for (let i = 0; i < count && budget.steps > 0; i++) {
      budget.steps -= 1;
      const at = ifd + 2 + i * 12;
      const tag = view.getUint16(at, little);
      if (tag === EXIF_IFD) {
        walk(view.getUint32(at + 8, little));
        continue;
      }
      if ((!TEXT_TAGS.has(tag) && tag !== USER_COMMENT) || !BYTE_TYPES.has(view.getUint16(at + 2, little))) continue;
      const length = view.getUint32(at + 4, little);
      const start = length <= 4 ? at + 8 : view.getUint32(at + 8, little);
      if (start + length > tiff.length || !pay(budget, length)) continue;
      const bytes = tiff.subarray(start, start + length);
      const text = clean(tag === USER_COMMENT ? userComment(bytes, little) : UTF8.decode(bytes));
      if (text) entries.push({ key: TEXT_TAGS.get(tag) || "UserComment", text });
    }
  };
  try {
    walk(view.getUint32(4, little));
  } catch {
    // a block cut short
  }
  return entries;
}

function jpegEntries(bytes, budget) {
  const entries = [];
  const view = dataView(bytes);
  for (let at = 2; at + 4 <= bytes.length && bytes[at] === 0xff; ) {
    const marker = bytes[at + 1];
    if (marker === 0xff) { // fill byte
      at += 1;
      continue;
    }
    if (marker === SOS || marker === EOI) break;
    const end = at + 2 + view.getUint16(at + 2);
    const data = bytes.subarray(at + 4, end);
    if (marker === APP1 && hasExifHeader(data)) entries.push(...tiffEntries(data.subarray(EXIF_HEADER.length), budget));
    else if (marker === COM && pay(budget, data.length)) entries.push({ key: "Comment", text: clean(UTF8.decode(data)) });
    at = end;
  }
  return entries;
}

function webpEntries(bytes, budget) {
  const entries = [];
  const view = dataView(bytes);
  for (let at = 12; at + 8 <= bytes.length; ) {
    const size = view.getUint32(at + 4, true);
    const data = bytes.subarray(at + 8, at + 8 + size);
    if (LATIN1.decode(bytes.subarray(at, at + 4)) === "EXIF") {
      entries.push(...tiffEntries(hasExifHeader(data) ? data.subarray(EXIF_HEADER.length) : data, budget));
    }
    at += 8 + size + (size & 1);
  }
  return entries;
}

async function pngEntries(buffer, bytes, budget) {
  const entries = Object.entries(await readPngText(buffer)).map(([key, text]) => ({ key, text }));
  for (const [type, data] of pngChunks(bytes)) if (type === "eXIf") entries.push(...tiffEntries(data, budget));
  return entries;
}

function formatOf(bytes) {
  const head = LATIN1.decode(bytes.subarray(0, 12));
  if (bytes[0] === 0x89 && head.slice(1, 4) === "PNG") return "PNG";
  if (bytes[0] === 0xff && bytes[1] === 0xd8) return "JPEG";
  if (head.slice(0, 4) === "RIFF" && head.slice(8) === "WEBP") return "WebP";
  return "";
}

const rank = (entry) => (INFOTEXT_KEYS.includes(entry.key) ? INFOTEXT_KEYS.indexOf(entry.key) : INFOTEXT_KEYS.length);

/** `{format, entries: [{key, text}]}` of an image file, the infotext first; `format` is "" for anything but PNG, JPEG, WebP. */
export async function readImageInfo(buffer) {
  const bytes = new Uint8Array(buffer);
  const format = formatOf(bytes);
  const budget = newBudget();
  const entries = format === "PNG" ? await pngEntries(buffer, bytes, budget)
    : format === "JPEG" ? jpegEntries(bytes, budget)
      : format === "WebP" ? webpEntries(bytes, budget)
        : [];
  return { format, entries: entries.filter((entry) => entry.text).sort((a, b) => rank(a) - rank(b)).slice(0, MAX_ENTRIES) };
}
