// node:test for the EXIF panel next to XYZ 비교 (web/ai_studio/exif_view.js) with a tiny fake document (no DOM in node).
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { ExifView } from "../web/ai_studio/exif_view.js";

class FakeElement {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.style = {};
    this.attrs = {};
    this.listeners = {};
    this.className = "";
    this.value = "";
    this.clicks = 0;
  }
  append(...nodes) {
    this.children.push(...nodes);
  }
  replaceChildren(...nodes) {
    this.children = [];
    this.append(...nodes);
  }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  click() {
    this.clicks += 1;
    this.fire("click");
  }
  fire(type, event = { target: this }) {
    for (const fn of this.listeners[type] || []) fn(event);
  }
  get text() {
    return this.children.map((child) => (typeof child === "string" ? child : child.text)).join("");
  }
  /** Every descendant (itself included) the test function accepts, in document order. */
  all(accept) {
    const own = accept(this) ? [this] : [];
    return own.concat(this.children.flatMap((child) => (typeof child === "string" ? [] : child.all(accept))));
  }
  byClass(name) {
    return this.all((element) => element.className.split(" ").includes(name));
  }
  byTag(tag) {
    return this.all((element) => element.tag === tag);
  }
}

globalThis.document = { createElement: (tag) => new FakeElement(tag) };
globalThis.Node = FakeElement;

const made = [];
const revoked = [];
URL.createObjectURL = (file) => {
  made.push(file.name);
  return `blob:${file.name}`;
};
URL.revokeObjectURL = (url) => revoked.push(url);

const ascii = (text) => [...new TextEncoder().encode(text)];
const pngChunk = (type, bytes) => [bytes.length >>> 24, (bytes.length >>> 16) & 255, (bytes.length >>> 8) & 255, bytes.length & 255, ...ascii(type), ...bytes, 0, 0, 0, 0];
const png = (texts = {}) => Uint8Array.from([
  137, 80, 78, 71, 13, 10, 26, 10,
  ...Object.entries(texts).flatMap(([key, text]) => pngChunk("tEXt", [...ascii(key), 0, ...ascii(text)])),
  ...pngChunk("IEND", []),
]).buffer;

const ref = (filename) => ({ filename, subfolder: "", type: "output" });
const fakeFile = (name, buffer) => ({ name, arrayBuffer: async () => buffer });

function setup(readImage, pullSettings) {
  return new ExifView(null, { imageUrl: (image) => `/view/${image.filename}`, readImage, pullSettings });
}

/** Lets the view's pending reads finish. */
const settled = async () => {
  for (let i = 0; i < 20; i++) await new Promise((done) => setImmediate(done));
};

const pick = async (view, name) => {
  view.onStudioEvent("selected", { image: name ? ref(name) : null });
  await settled();
};

const texts = (view) => view.element.byClass("ais-exif-text").map((box) => box.value);
const picture = (view) => view.element.byClass("ais-exif-image")[0]?.attrs.src;
const label = (view) => view.element.byClass("ais-exif-name")[0]?.text;
const notes = (view) => view.element.byClass("ais-note").map((note) => note.text);

function dropEvent(files) {
  const calls = [];
  return {
    calls,
    event: {
      dataTransfer: { files },
      preventDefault: () => calls.push("prevent"),
      stopPropagation: () => calls.push("stop"),
    },
  };
}

test("the panel starts empty and asks for an image", () => {
  const view = setup(async () => null);
  assert.deepEqual(texts(view), []);
  assert.equal(picture(view), undefined);
  assert.equal(notes(view).length, 1);
});

test("the gallery's picked image shows with its parameters", async () => {
  const view = setup(async () => png({ parameters: "1girl\nSteps: 32" }));
  await pick(view, "a.png");
  assert.deepEqual(texts(view), ["1girl\nSteps: 32"]);
  assert.equal(picture(view), "/view/a.png");
  assert.equal(label(view), "a.png · PNG");
});

test("picking nothing empties the panel again", async () => {
  const view = setup(async () => png({ parameters: "Steps: 32" }));
  await pick(view, "a.png");
  await pick(view, null);
  assert.deepEqual(texts(view), []);
  assert.equal(picture(view), undefined);
});

test("an image without text says so and still shows", async () => {
  const view = setup(async () => png());
  await pick(view, "a.png");
  assert.deepEqual(texts(view), []);
  assert.equal(picture(view), "/view/a.png");
  assert.deepEqual(notes(view), ["이 이미지엔 생성 정보가 없음"]);
});

test("a file of another kind says what can be read", async () => {
  const view = setup(async () => new ArrayBuffer(16));
  await pick(view, "a.gif");
  assert.deepEqual(notes(view), ["PNG · JPEG · WebP 만 읽음"]);
});

test("an image that cannot be read says so instead of keeping the last text", async () => {
  const reads = [async () => png({ parameters: "old" }), async () => null, async () => { throw new Error("gone"); }];
  const view = setup(() => reads.shift()());
  await pick(view, "a.png");
  await pick(view, "gone.png");
  assert.deepEqual(texts(view), []);
  assert.deepEqual(notes(view), ["이미지를 읽지 못함"]);
  await pick(view, "gone2.png");
  assert.deepEqual(notes(view), ["이미지를 읽지 못함"]);
});

test("a slow read never replaces a later pick", async () => {
  let finishSlow;
  const reads = [() => new Promise((done) => { finishSlow = done; }), async () => png({ parameters: "second" })];
  const view = setup(() => reads.shift()());
  view.onStudioEvent("selected", { image: ref("first.png") });
  await pick(view, "second.png");
  finishSlow(png({ parameters: "first" }));
  await settled();
  assert.deepEqual(texts(view), ["second"]);
  assert.equal(picture(view), "/view/second.png");
});

test("a dropped file shows without ComfyUI seeing the drop, and the next pick replaces it", async () => {
  const view = setup(async () => png({ parameters: "from gallery" }));
  const { event, calls } = dropEvent([fakeFile("dropped.png", png({ parameters: "from file" }))]);
  view.element.fire("drop", event);
  await settled();
  assert.deepEqual(calls, ["prevent", "stop"]); // else ComfyUI opens the file as a workflow
  assert.deepEqual(texts(view), ["from file"]);
  assert.equal(picture(view), "blob:dropped.png");
  assert.equal(label(view), "dropped.png · PNG");
  await pick(view, "a.png");
  assert.deepEqual(texts(view), ["from gallery"]);
  assert.ok(revoked.includes("blob:dropped.png"));
});

test("a drop without a file, such as dragged text, changes nothing but is still kept from ComfyUI", async () => {
  const view = setup(async () => png({ parameters: "picked" }));
  await pick(view, "a.png");
  const { event, calls } = dropEvent([]);
  view.element.fire("drop", event);
  await settled();
  assert.deepEqual(calls, ["prevent", "stop"]);
  assert.deepEqual(texts(view), ["picked"]);
});

test("dragging over the panel asks the browser to allow the drop", () => {
  const view = setup(async () => null);
  const { event, calls } = dropEvent([]);
  view.element.fire("dragover", event);
  assert.deepEqual(calls, ["prevent", "stop"]);
});

test("파일 열기 opens the file box and shows the chosen file, the same file can be chosen again", async () => {
  const view = setup(async () => null);
  const input = view.element.byTag("input")[0];
  const button = view.element.byClass("ais-mini")[0];
  button.fire("click");
  assert.equal(input.clicks, 1);
  input.files = [fakeFile("chosen.png", png({ parameters: "chosen" }))];
  input.value = "C:\\fakepath\\chosen.png";
  input.fire("change");
  await settled();
  assert.deepEqual(texts(view), ["chosen"]);
  assert.equal(view.element.byTag("input")[0].value, "");
});

test("leaving the panel gives the dropped file's URL back", async () => {
  const view = setup(async () => null);
  const { event } = dropEvent([fakeFile("leaving.png", png())]);
  view.element.fire("drop", event);
  await settled();
  view.dispose();
  assert.ok(revoked.includes("blob:leaving.png"));
});

test("a very long text shows its beginning and says so", async () => {
  const long = "x".repeat(1100000);
  const view = setup(async () => png({ parameters: long }));
  await pick(view, "a.png");
  assert.equal(texts(view)[0].length, 1024 * 1024);
  assert.match(view.element.byClass("ais-label")[0].text, /parameters .*앞부분/);
});

const pullButton = (view) => view.element.byClass("ais-mini").find((button) => button.text === "설정 가져오기");

test("설정 가져오기 hands the shown image's settings to the studio", async () => {
  const pulled = [];
  const view = setup(async () => png({ parameters: "1girl\nSteps: 32" }), (settings) => pulled.push(settings));
  await pick(view, "a.png");
  const button = pullButton(view);
  assert.equal(button.attrs["aria-disabled"], undefined);
  button.fire("click");
  assert.deepEqual(pulled, [{ infotext: "1girl\nSteps: 32" }]);
});

test("a studio image pulls its panel, not the text", async () => {
  const pulled = [];
  const workflow = JSON.stringify({ nodes: [{ type: "AIStudioPanel", widgets_values: ['{"prompt":"x"}'] }] });
  const view = setup(async () => png({ parameters: "1girl\nSteps: 32", workflow }), (settings) => pulled.push(settings));
  await pick(view, "a.png");
  pullButton(view).fire("click");
  assert.deepEqual(pulled, [{ panel: { prompt: "x" }, xyz: null }]);
});

test("with nothing to pull the button is grey and does nothing", async () => {
  const pulled = [];
  const view = setup(async () => png({ Software: "paint" }), (settings) => pulled.push(settings));
  assert.equal(pullButton(view).attrs["aria-disabled"], "true"); // no image yet
  await pick(view, "a.png");
  const button = pullButton(view);
  assert.equal(button.attrs["aria-disabled"], "true");
  button.fire("click");
  assert.deepEqual(pulled, []);
});

test("picking nothing takes the button away again", async () => {
  const view = setup(async () => png({ parameters: "1girl\nSteps: 32" }), () => {});
  await pick(view, "a.png");
  await pick(view, null);
  assert.equal(pullButton(view).attrs["aria-disabled"], "true");
});

test("long machine text starts folded, a long infotext never", async () => {
  const long = "x".repeat(5000);
  const view = setup(async () => png({ parameters: long, workflow: long, short: "s" }));
  await pick(view, "a.png");
  assert.deepEqual(texts(view).map((text) => text.length), [5000, 5000, 1]);
  const folded = view.element.byTag("details");
  assert.equal(folded.length, 1);
  assert.equal(folded[0].byTag("summary")[0].text, "workflow");
});
