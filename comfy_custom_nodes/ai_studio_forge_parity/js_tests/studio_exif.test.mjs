// node:test for reading what an image file says about itself (web/ai_studio/studio_exif.js). The files are built byte by
// byte here, the way Forge (piexif: big-endian TIFF, "UNICODE\0" + UTF-16BE) and other writers lay them out.
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";
import { deflateSync } from "node:zlib";

import { readImageInfo } from "../web/ai_studio/studio_exif.js";

const ascii = (text) => [...new TextEncoder().encode(text)];
const cString = (text) => [...ascii(text), 0];
const utf16 = (text, little) => [...text].flatMap((char) => {
  const code = char.charCodeAt(0);
  return little ? [code & 255, code >> 8] : [code >> 8, code & 255];
});
const buffer = (...parts) => Uint8Array.from(parts.flat()).buffer;

const IMAGE_DESCRIPTION = 0x010e;
const MAKE = 0x010f;
const SOFTWARE = 0x0131;
const EXIF_IFD = 0x8769;
const USER_COMMENT = 0x9286;
const ASCII_TYPE = 2;
const UNDEFINED_TYPE = 7;
const LONG_TYPE = 4;

/** A TIFF block: IFD 0 with `ifd0` `[tag, type, bytes]` entries and, when `exif` has entries, an Exif sub-IFD. */
function tiff(ifd0, exif = [], little = false) {
  const u16 = (value) => (little ? [value & 255, value >> 8] : [value >> 8, value & 255]);
  const u32 = (value) => {
    const big = [value >>> 24, (value >>> 16) & 255, (value >>> 8) & 255, value & 255];
    return little ? big.reverse() : big;
  };
  const first = exif.length ? [...ifd0, [EXIF_IFD, LONG_TYPE, null]] : ifd0;
  const size = (list) => 2 + list.length * 12 + 4;
  const exifAt = 8 + size(first);
  let dataAt = exifAt + (exif.length ? size(exif) : 0);
  const data = [];
  const ifd = (list) => [
    ...u16(list.length),
    ...list.flatMap(([tag, type, bytes]) => {
      if (bytes === null) return [...u16(tag), ...u16(type), ...u32(1), ...u32(exifAt)];
      const head = [...u16(tag), ...u16(type), ...u32(bytes.length)];
      if (bytes.length <= 4) return [...head, ...bytes, ...new Array(4 - bytes.length).fill(0)];
      const entry = [...head, ...u32(dataAt)];
      for (const byte of bytes) data.push(byte);
      dataAt += bytes.length;
      return entry;
    }),
    ...u32(0),
  ];
  const head = [...(little ? ascii("II") : ascii("MM")), ...u16(42), ...u32(8)];
  const blocks = [...ifd(first), ...(exif.length ? ifd(exif) : [])];
  return [...head, ...blocks, ...data];
}

const EXIF_HEADER = [...ascii("Exif"), 0, 0];

function jpeg(...segments) {
  return buffer([0xff, 0xd8], ...segments, [0xff, 0xda, 0, 2], [1, 2, 3], [0xff, 0xd9]);
}
const segment = (marker, bytes) => [0xff, marker, (bytes.length + 2) >> 8, (bytes.length + 2) & 255, ...bytes];

function riff(...chunks) {
  const body = [...ascii("WEBP"), ...chunks.flat()];
  return buffer(ascii("RIFF"), [body.length & 255, (body.length >> 8) & 255, 0, 0], body);
}
const riffChunk = (type, bytes) => [...ascii(type), bytes.length & 255, (bytes.length >>> 8) & 255, (bytes.length >>> 16) & 255, bytes.length >>> 24, ...bytes, ...(bytes.length & 1 ? [0] : [])];

const pngChunk = (type, bytes) => [bytes.length >>> 24, (bytes.length >>> 16) & 255, (bytes.length >>> 8) & 255, bytes.length & 255, ...ascii(type), ...bytes, 0, 0, 0, 0];
const png = (...chunks) => buffer([137, 80, 78, 71, 13, 10, 26, 10], ...chunks, pngChunk("IEND", []));

const entry = (key, text) => ({ key, text });
const INFOTEXT = "1girl, 소녀\nSteps: 32, Seed: 7";
const UNICODE = [...ascii("UNICODE"), 0];

test("a PNG's parameters come first, ComfyUI's prompt and workflow after, whatever the file's order", async () => {
  const file = png(
    pngChunk("tEXt", [...cString("workflow"), ...ascii("{}")]),
    pngChunk("tEXt", [...cString("parameters"), ...ascii("Steps: 32")]),
    pngChunk("tEXt", [...cString("prompt"), ...ascii("{\"1\":{}}")]),
  );
  assert.deepEqual(await readImageInfo(file), {
    format: "PNG",
    entries: [entry("parameters", "Steps: 32"), entry("workflow", "{}"), entry("prompt", "{\"1\":{}}")],
  });
});

test("a PNG's eXIf chunk is read too", async () => {
  const file = png(pngChunk("eXIf", tiff([[SOFTWARE, ASCII_TYPE, cString("GIMP 3.0")]])));
  assert.deepEqual(await readImageInfo(file), { format: "PNG", entries: [entry("Software", "GIMP 3.0")] });
});

test("Forge's JPEG: big-endian TIFF, UserComment in the Exif IFD as UNICODE + UTF-16BE", async () => {
  const exif = tiff([], [[USER_COMMENT, UNDEFINED_TYPE, [...UNICODE, ...utf16(INFOTEXT, false)]]]);
  const file = jpeg(segment(0xe1, [...EXIF_HEADER, ...exif]));
  assert.deepEqual(await readImageInfo(file), { format: "JPEG", entries: [entry("UserComment", INFOTEXT)] });
});

test("a little-endian TIFF with little-endian UTF-16", async () => {
  const exif = tiff([], [[USER_COMMENT, UNDEFINED_TYPE, [...UNICODE, ...utf16(INFOTEXT, true)]]], true);
  const file = jpeg(segment(0xe1, [...EXIF_HEADER, ...exif]));
  assert.deepEqual((await readImageInfo(file)).entries, [entry("UserComment", INFOTEXT)]);
});

test("UTF-16BE text in a little-endian TIFF: the text's own bytes decide", async () => {
  const exif = tiff([], [[USER_COMMENT, UNDEFINED_TYPE, [...UNICODE, ...utf16("Steps: 32", false)]]], true);
  const file = jpeg(segment(0xe1, [...EXIF_HEADER, ...exif]));
  assert.deepEqual((await readImageInfo(file)).entries, [entry("UserComment", "Steps: 32")]);
});

test("UserComment with an ASCII prefix, an empty prefix or none at all", async () => {
  const cases = [
    [[...ascii("ASCII"), 0, 0, 0, ...ascii("Steps: 1")], "Steps: 1"],
    [[0, 0, 0, 0, 0, 0, 0, 0, ...ascii("Steps: 2")], "Steps: 2"],
    [ascii("Steps: 3 has no prefix"), "Steps: 3 has no prefix"],
  ];
  for (const [bytes, want] of cases) {
    const file = jpeg(segment(0xe1, [...EXIF_HEADER, ...tiff([], [[USER_COMMENT, UNDEFINED_TYPE, bytes]])]));
    assert.deepEqual((await readImageInfo(file)).entries, [entry("UserComment", want)]);
  }
});

test("text tags of IFD 0: long ones behind an offset, short ones inside the entry", async () => {
  const exif = tiff([[IMAGE_DESCRIPTION, ASCII_TYPE, cString("a long description")], [MAKE, ASCII_TYPE, cString("HP")]]);
  const file = jpeg(segment(0xe1, [...EXIF_HEADER, ...exif]));
  assert.deepEqual((await readImageInfo(file)).entries, [entry("ImageDescription", "a long description"), entry("Make", "HP")]);
});

test("a JPEG comment segment, which some tools use for the infotext, sorts after UserComment", async () => {
  const exif = tiff([], [[USER_COMMENT, UNDEFINED_TYPE, [...ascii("ASCII"), 0, 0, 0, ...ascii("from exif")]]]);
  const file = jpeg(segment(0xfe, ascii("from comment")), segment(0xe1, [...EXIF_HEADER, ...exif]));
  assert.deepEqual((await readImageInfo(file)).entries, [entry("UserComment", "from exif"), entry("Comment", "from comment")]);
});

test("segments in front of the EXIF one are stepped over, an XMP APP1 is not read as EXIF", async () => {
  const exif = tiff([[SOFTWARE, ASCII_TYPE, cString("Forge Neo")]]);
  const file = jpeg(segment(0xe0, ascii("JFIF")), segment(0xe1, cString("http://ns.adobe.com/xap/1.0/")), segment(0xe1, [...EXIF_HEADER, ...exif]));
  assert.deepEqual((await readImageInfo(file)).entries, [entry("Software", "Forge Neo")]);
});

test("WebP: the EXIF chunk with and without the Exif header, past an odd-sized chunk", async () => {
  const exif = tiff([], [[USER_COMMENT, UNDEFINED_TYPE, [...UNICODE, ...utf16(INFOTEXT, false)]]]);
  for (const body of [exif, [...EXIF_HEADER, ...exif]]) {
    const file = riff(riffChunk("VP8X", [1, 2, 3]), riffChunk("EXIF", body));
    assert.deepEqual(await readImageInfo(file), { format: "WebP", entries: [entry("UserComment", INFOTEXT)] });
  }
});

test("a file of another kind has no format and no entries", async () => {
  assert.deepEqual(await readImageInfo(buffer(ascii("GIF89a"), new Array(20).fill(0))), { format: "", entries: [] });
  assert.deepEqual(await readImageInfo(new ArrayBuffer(0)), { format: "", entries: [] });
});

test("an image without any text says nothing", async () => {
  assert.deepEqual(await readImageInfo(png()), { format: "PNG", entries: [] });
  assert.deepEqual(await readImageInfo(jpeg()), { format: "JPEG", entries: [] });
  assert.deepEqual(await readImageInfo(riff(riffChunk("VP8 ", [1, 2]))), { format: "WebP", entries: [] });
});

test("broken files give what could be read and never throw", async () => {
  const good = tiff([[SOFTWARE, ASCII_TYPE, cString("Forge Neo")], [IMAGE_DESCRIPTION, ASCII_TYPE, cString("cut off by the end of the file")]]);
  const cut = good.slice(0, good.length - 12); // the second value runs past the end
  assert.deepEqual((await readImageInfo(jpeg(segment(0xe1, [...EXIF_HEADER, ...cut])))).format, "JPEG");

  const noValue = tiff([[IMAGE_DESCRIPTION, ASCII_TYPE, cString("a value past the end")], [SOFTWARE, ASCII_TYPE, cString("Forge Neo")]]);
  noValue[8 + 2 + 8 + 3] = 200; // IFD 0, first entry, offset (big-endian, low byte)
  assert.deepEqual((await readImageInfo(jpeg(segment(0xe1, [...EXIF_HEADER, ...noValue])))).entries, [entry("Software", "Forge Neo")]);

  const manyEntries = [...ascii("MM"), 0, 42, 0, 0, 0, 8, 0xff, 0xff, 0, 0];
  assert.deepEqual((await readImageInfo(jpeg(segment(0xe1, [...EXIF_HEADER, ...manyEntries])))).entries, []);

  assert.deepEqual((await readImageInfo(buffer([0xff, 0xd8, 0xff, 0xe1, 0xff, 0xff, 1, 2]))).entries, []); // a segment past the end
  assert.deepEqual((await readImageInfo(buffer(ascii("RIFF"), [0, 0, 0, 0], ascii("WEBP"), ascii("EXIF"), [255, 255, 0, 0], [1]))).entries, []);
});

test("Forge's text without ASCII keeps the TIFF's byte order: CJK characters end in 00 too", async () => {
  for (const text of ["대한민국", "가방", "夕暮れの海辺　少女", "一个女孩站在海边，夕阳映照着她的长发。".repeat(20)]) {
    const exif = tiff([], [[USER_COMMENT, UNDEFINED_TYPE, [...UNICODE, ...utf16(text, false)]]]);
    const file = jpeg(segment(0xe1, [...EXIF_HEADER, ...exif]));
    assert.deepEqual((await readImageInfo(file)).entries, [entry("UserComment", text)]);
  }
});

test("a byte order mark decides over the TIFF's order and leaves the text", async () => {
  const cases = [
    [[0xff, 0xfe, ...utf16("소녀", true)], false],
    [[0xfe, 0xff, ...utf16("소녀", false)], true],
  ];
  for (const [body, little] of cases) {
    const exif = tiff([], [[USER_COMMENT, UNDEFINED_TYPE, [...UNICODE, ...body]]], little);
    const file = jpeg(segment(0xe1, [...EXIF_HEADER, ...exif]));
    assert.deepEqual((await readImageInfo(file)).entries, [entry("UserComment", "소녀")]);
  }
});

test("UserComment with the JIS prefix is Shift-JIS, as piexif writes it", async () => {
  const exif = tiff([], [[USER_COMMENT, UNDEFINED_TYPE, [...ascii("JIS"), 0, 0, 0, 0, 0, 0x93, 0xfa, 0x96, 0x7b, 0x8c, 0xea]]]);
  const file = jpeg(segment(0xe1, [...EXIF_HEADER, ...exif]));
  assert.deepEqual((await readImageInfo(file)).entries, [entry("UserComment", "日本語")]);
});

const quickly = async (work) => {
  const started = performance.now();
  const result = await work();
  assert.ok(performance.now() - started < 3000, "took too long");
  return result;
};

test("a long run of spaces inside a value costs no quadratic time", async () => {
  const body = [...ascii("ASCII"), 0, 0, 0, ...new Array(400000).fill(32), ...ascii("x")];
  const file = riff(riffChunk("EXIF", tiff([], [[USER_COMMENT, UNDEFINED_TYPE, body]])));
  const { entries } = await quickly(() => readImageInfo(file));
  assert.equal(entries[0].text.length, 400001);
});

test("many text tags stop at a limit instead of filling the panel", async () => {
  const many = new Array(5000).fill([MAKE, ASCII_TYPE, cString("HP")]);
  const { entries } = await readImageInfo(riff(riffChunk("EXIF", tiff(many))));
  assert.ok(entries.length > 0 && entries.length <= 256, String(entries.length));
});

test("many entries sharing one big value cannot multiply it", async () => {
  const value = new Array(1 << 20).fill(65);
  const exif = tiff([[IMAGE_DESCRIPTION, ASCII_TYPE, value]]);
  const count = 1000; // every entry repeats the first, offset included
  const first = exif.slice(10, 22);
  const out = [...exif.slice(0, 8), count >> 8, count & 255, ...new Array(count).fill(first).flat()];
  const shift = out.length + 4 - (10 + 12 + 4); // the value moved back by the added entries
  const offset = (first[8] << 24) + (first[9] << 16) + (first[10] << 8) + first[11] + shift;
  for (let i = 0; i < count; i++) out.splice(10 + i * 12 + 8, 4, offset >>> 24, (offset >>> 16) & 255, (offset >>> 8) & 255, offset & 255);
  const file = riff(riffChunk("EXIF", [...out, 0, 0, 0, 0, ...value]));
  const { entries } = await quickly(() => readImageInfo(file));
  const total = entries.reduce((sum, item) => sum + item.text.length, 0);
  assert.ok(total > 0 && total <= 32 * 1024 * 1024, String(total));
});

test("a wide tree of Exif IFDs over junk bytes ends quickly", async () => {
  const count = 65535;
  const entries = new Array(count).fill(0).flatMap((_, i) => {
    const target = 8 + 2 + count * 12 + 4 + i * 2;
    return [0x87, 0x69, 0, 4, 0, 0, 0, 1, target >>> 24, (target >>> 16) & 255, (target >>> 8) & 255, target & 255];
  });
  const junk = new Array(count * 2 + 70000).fill(0xff);
  const block = [...ascii("MM"), 0, 42, 0, 0, 0, 8, 0xff, 0xff, ...entries, 0, 0, 0, 0, ...junk];
  const { format } = await quickly(() => readImageInfo(riff(riffChunk("EXIF", block))));
  assert.equal(format, "WebP");
});

test("thousands of JPEG comment segments stop at the limit", async () => {
  const comments = new Array(20000).fill(segment(0xfe, ascii("hi")));
  const { entries } = await readImageInfo(jpeg(...comments));
  assert.ok(entries.length > 0 && entries.length <= 256, String(entries.length));
});

test("thousands of PNG text chunks stop at the limit", async () => {
  const chunks = new Array(3000).fill(0).map((_, i) => pngChunk("tEXt", [...cString(`k${i}`), ...ascii("v")]));
  const { entries } = await readImageInfo(png(...chunks));
  assert.ok(entries.length > 0 && entries.length <= 256, String(entries.length));
});

test("a compressed text chunk that unfolds too far is skipped, the other chunks stay", async () => {
  const bomb = [...deflateSync(Buffer.alloc(64 * 1024 * 1024, 65))];
  const ztxt = pngChunk("zTXt", [...cString("parameters"), 0, ...bomb]);
  const small = pngChunk("zTXt", [...cString("Comment"), 0, ...deflateSync(Buffer.from("small"))]);
  const file = png(pngChunk("tEXt", [...cString("Software"), ...ascii("x")]), small, ztxt);
  const { entries } = await quickly(() => readImageInfo(file));
  assert.deepEqual(entries, [entry("Comment", "small"), entry("Software", "x")]);
});

test("an Exif IFD pointing at itself ends", async () => {
  const looped = tiff([], [[USER_COMMENT, UNDEFINED_TYPE, ascii("not reached first")]]);
  const pointer = 8 + 2 + 8; // IFD 0, its only entry, the value (big-endian)
  looped[pointer + 3] = 8; // the Exif IFD is IFD 0 again
  assert.deepEqual((await readImageInfo(jpeg(segment(0xe1, [...EXIF_HEADER, ...looped])))).entries, []);
});
