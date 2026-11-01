// node:test for the result column's gallery (web/ai_studio/result_view.js) with a tiny fake document (no DOM in node).
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { ResultView } from "../web/ai_studio/result_view.js";

class FakeElement {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.style = {};
    this.attrs = {};
    this.listeners = {};
    this.className = "";
    this.textContent = "";
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
  fire(type) {
    for (const fn of this.listeners[type] || []) fn({ target: this });
  }
  get content() { // icon(): a <template>'s parsed markup
    return { firstElementChild: new FakeElement("svg") };
  }
  get text() {
    return this.children.map((child) => (typeof child === "string" ? child : child.text)).join("");
  }
  /** Every descendant (itself included) the test function accepts, in document order. */
  all(accept = () => true) {
    const own = accept(this) ? [this] : [];
    return own.concat(this.children.flatMap((child) => (typeof child === "string" ? [] : child.all(accept))));
  }
  byClass(name) {
    return this.all((element) => element.className.split(" ").includes(name));
  }
}

globalThis.document = { createElement: (tag) => new FakeElement(tag) };
globalThis.Node = FakeElement;

const AUTO_CLEAR = "갤러리 자동 비우기";
const ref = (filename) => ({ filename, subfolder: "", type: "output" });
const item = (filename) => ({ image: ref(filename), record: null });

function setup(saved = {}) {
  const widget = { value: JSON.stringify(saved) };
  const picks = [];
  const view = new ResultView(widget, {
    imageUrl: (image) => `/view/${image.filename}`,
    action: () => {},
    selected: (image) => picks.push(image && image.filename),
  });
  return { view, widget, picks };
}

/** What the gallery shows, left to right. */
const thumbs = (view) => view.element.byClass("ais-thumb").map((thumb) => thumb.attrs.title);
const picked = (view) => view.element.byClass("sel").map((thumb) => thumb.attrs.title);
const stageImage = (view) => view.element.byClass("ais-stage-img")[0]?.attrs.src;
const autoClearPill = (view) => view.element.byClass("ais-pill").find((pill) => pill.text === AUTO_CLEAR);
const pillNamed = (view, text) => view.element.byClass("ais-pill").find((pill) => pill.text === text);

const start = (view, kind = "single") => view.onStudioEvent("running", { running: true, kind });
const finish = (view, kind = "single") => view.onStudioEvent("running", { running: false, kind, elapsed: null });
/** `replace`: RunController's flag, on for the first images of a generation. */
const images = (view, names, replace = true) => view.onStudioEvent("images", { items: names.map(item), replace });
const grid = (view, name, replace = true) => view.onStudioEvent("grid", { item: item(name), subgrids: [], replace });

const FULL = { gallery: [ref("a.png"), ref("b.png")], selected: 1 };
const FULL_ON = { ...FULL, auto_clear: true };

test("with 자동 비우기 off new images go in front of the older ones", () => {
  const { view } = setup(FULL);
  start(view);
  images(view, ["new.png"]);
  assert.deepEqual(thumbs(view), ["new.png", "a.png", "b.png"]);
});

test("with 자동 비우기 on a generation's images replace the gallery, the first one picked", () => {
  const { view } = setup(FULL_ON);
  start(view);
  assert.deepEqual(thumbs(view), ["a.png", "b.png"]); // like Forge: the last images stay until the new ones arrive
  images(view, ["n1.png", "n2.png"]);
  assert.deepEqual(thumbs(view), ["n1.png", "n2.png"]);
  assert.deepEqual(picked(view), ["n1.png"]);
});

test("each generation clears once: the second run replaces the first run's images", () => {
  const { view } = setup(FULL_ON);
  start(view);
  images(view, ["first.png"]);
  finish(view);
  start(view);
  images(view, ["second.png"]);
  assert.deepEqual(thumbs(view), ["second.png"]);
});

test("a run clears only once: images arriving later in the same run stay together", () => {
  const { view } = setup(FULL_ON);
  start(view, "xyz");
  images(view, ["cell1.png"]);
  images(view, ["cell2.png"], false);
  grid(view, "grid.png", false);
  assert.deepEqual(thumbs(view), ["grid.png", "cell2.png", "cell1.png"]);
});

test("a view rebuilt in the middle of a run (undo, another workflow tab) still clears", () => {
  const { widget } = setup(FULL_ON);
  const { view: rebuilt } = setup(JSON.parse(widget.value));
  images(rebuilt, ["new.png"]);
  assert.deepEqual(thumbs(rebuilt), ["new.png"]);
});

test("an X/Y/Z run without cell images leaves its grid in the gallery and on the grid tab", () => {
  const { view } = setup(FULL_ON);
  start(view, "xyz");
  grid(view, "grid.png");
  assert.deepEqual(thumbs(view), ["grid.png"]);
  assert.equal(stageImage(view), "/view/grid.png");
});

test("a run that made no image (stopped, failed) leaves the gallery alone", () => {
  const { view } = setup(FULL_ON);
  start(view);
  finish(view);
  assert.deepEqual(thumbs(view), ["a.png", "b.png"]);
});

test("images that do not replace, a gallery 업스케일 / Hires / SAM3 run's, keep the others", () => {
  const { view } = setup(FULL_ON);
  start(view, "post");
  images(view, ["post.png"], false);
  assert.deepEqual(thumbs(view), ["post.png", "a.png", "b.png"]);
});

test("the XYZ 격자 tab keeps its grid when a later generation clears the gallery", () => {
  const { view } = setup({ ...FULL_ON, grid: ref("grid.png"), tab: "grid" });
  start(view);
  images(view, ["new.png"]);
  finish(view);
  pillNamed(view, "XYZ 격자").fire("click");
  assert.equal(stageImage(view), "/view/grid.png");
});

test("the 자동 비우기 pill sits under the gallery buttons, starts off and is saved with the workflow", () => {
  const { view, widget } = setup(FULL);
  const all = view.element.all();
  assert.ok(all.indexOf(autoClearPill(view)) > all.indexOf(view.element.byClass("ais-buttons")[0]));
  assert.equal(autoClearPill(view).attrs["aria-pressed"], "false");
  autoClearPill(view).fire("click");
  assert.equal(autoClearPill(view).attrs["aria-pressed"], "true");
  view.load(); // what ComfyUI does when the workflow is opened again
  assert.equal(autoClearPill(view).attrs["aria-pressed"], "true");
  assert.equal(JSON.parse(widget.value).auto_clear, true);
  autoClearPill(view).fire("click");
  assert.equal(JSON.parse(widget.value).auto_clear, false);
});

test("the pill takes effect on the next generation", () => {
  const { view } = setup(FULL);
  autoClearPill(view).fire("click");
  start(view);
  images(view, ["new.png"]);
  assert.deepEqual(thumbs(view), ["new.png"]);
});

test("an old workflow or a broken saved value reads as off", () => {
  assert.equal(autoClearPill(setup({ auto_clear: "yes" }).view).attrs["aria-pressed"], "false");
  const { view } = setup({ gallery: [ref("a.png")], auto_clear: 1 });
  start(view);
  images(view, ["new.png"]);
  assert.deepEqual(thumbs(view), ["new.png", "a.png"]);
});

const header = (view) => view.element.byClass("ais-nt")[0].text;
const took = (view, ms, kind = "single") => view.onStudioEvent("running", { running: false, kind, elapsed: ms });

test("the result header shows how long the last run took", () => {
  const cases = [[12345, "걸린 시간 12.3초"], [900, "걸린 시간 0.9초"], [72000, "걸린 시간 1분 12초"]];
  for (const [ms, want] of cases) {
    const { view } = setup();
    start(view);
    assert.ok(!header(view).includes("걸린 시간"));
    took(view, ms);
    assert.ok(header(view).includes(want), header(view));
  }
});

test("the time goes away while the next run works, a run without images brings back the last one", () => {
  const { view } = setup();
  start(view);
  took(view, 5000);
  start(view);
  assert.ok(!header(view).includes("걸린 시간"));
  took(view, null); // that run's images are still in the gallery
  assert.ok(header(view).includes("걸린 시간 5.0초"), header(view));
});

test("new images and a clicked thumbnail tell the EXIF panel what is picked", () => {
  const { view, picks } = setup(FULL);
  start(view);
  images(view, ["n1.png", "n2.png"]);
  assert.deepEqual(picks, ["n1.png"]);
  view.element.byClass("ais-thumb")[2].fire("click");
  assert.deepEqual(picks, ["n1.png", "a.png"]);
});

test("announce() repeats the picked image, null with an empty gallery", () => {
  const full = setup(FULL);
  full.view.announce();
  assert.deepEqual(full.picks, ["b.png"]);
  const empty = setup();
  empty.view.announce();
  assert.deepEqual(empty.picks, [null]);
});
