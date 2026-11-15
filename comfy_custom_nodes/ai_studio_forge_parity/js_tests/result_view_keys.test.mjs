// node:test for the result gallery's keys (←/→/Home/End) and for keeping the picked thumbnail in the middle of the strip
// (web/ai_studio/result_view.js). The fake document is result_view_fakes.mjs: it bubbles events, moves the focus and lays
// the strip out.
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { centredScrollLeft, galleryKeyIndex, ResultView } from "../web/ai_studio/result_view.js";
import { FakeElement, images, layout, observers, picked, pillNamed, press, ref, saved, setup, stageImage, start, strip, stripWidth, THUMB, thumbLeft, thumbsOf } from "./result_view_fakes.mjs";

// ---------------------------------------------------------------------------------------------- the two pure helpers

test("galleryKeyIndex: ←/→ step, Home/End jump, the ends stop (no wrap), other keys are not the gallery's", () => {
  assert.equal(galleryKeyIndex("ArrowRight", 2, 5), 3);
  assert.equal(galleryKeyIndex("ArrowLeft", 2, 5), 1);
  assert.equal(galleryKeyIndex("ArrowLeft", 0, 5), 0);
  assert.equal(galleryKeyIndex("ArrowRight", 4, 5), 4);
  assert.equal(galleryKeyIndex("Home", 3, 5), 0);
  assert.equal(galleryKeyIndex("End", 1, 5), 4);
  for (const key of ["ArrowUp", "ArrowDown", "Enter", " ", "a", "Escape", "Delete"]) assert.equal(galleryKeyIndex(key, 2, 5), null, key);
  assert.equal(galleryKeyIndex("ArrowRight", 0, 0), null); // an empty gallery takes no key
});

test("centredScrollLeft: the thumbnail's middle goes to the strip's middle, clamped at both ends", () => {
  const total = stripWidth(10); // 716
  assert.equal(centredScrollLeft({ left: thumbLeft(5), width: THUMB, view: 300, total }), 243); // 362 + 31 - 150
  assert.equal(centredScrollLeft({ left: thumbLeft(0), width: THUMB, view: 300, total }), 0); // would be -107
  assert.equal(centredScrollLeft({ left: thumbLeft(9), width: THUMB, view: 300, total }), 416); // would be 523; 716 - 300
  assert.equal(centredScrollLeft({ left: thumbLeft(1), width: THUMB, view: 300, total: 300 }), 0); // everything fits
});

test("centredScrollLeft: a strip that is not laid out yet has no answer", () => {
  assert.equal(centredScrollLeft({ left: 0, width: 0, view: 0, total: 0 }), null);
  assert.equal(centredScrollLeft({ left: undefined, width: 62, view: 300, total: 716 }), null);
  assert.equal(centredScrollLeft({ left: NaN, width: 62, view: 300, total: 716 }), null);
});

// ---------------------------------------------------------------------------------------------- keys

test("→ and ← pick the next / previous image exactly as a click does", () => {
  const { view, widget, picks } = setup(saved(5, 1));
  const event = press(view, "ArrowRight");
  assert.deepEqual(picked(view), ["2.png"]);
  assert.equal(stageImage(view), "/view/2.png");
  assert.equal(view.selectedItem().image.filename, "2.png"); // what 같은 시드 / → i2i / → 인페인트 use
  assert.equal(JSON.parse(widget.value).selected, 2);
  assert.deepEqual(picks, ["2.png"]); // the EXIF panel
  assert.equal(event.defaultPrevented, true); // the browser does not scroll the strip too
  assert.equal(event.stopped, true); // nor does the canvas see the key
  press(view, "ArrowLeft");
  press(view, "ArrowLeft");
  assert.deepEqual(picked(view), ["0.png"]);
  assert.deepEqual(picks, ["2.png", "1.png", "0.png"]);
});

test("a key pick matches a click pick: same saved view, same announcement", () => {
  const byKey = setup(saved(5, 1));
  press(byKey.view, "ArrowRight");
  const byClick = setup(saved(5, 1));
  thumbsOf(byClick.view)[2].fire("click");
  assert.equal(byKey.widget.value, byClick.widget.value);
  assert.deepEqual(byKey.picks, byClick.picks);
  assert.deepEqual(picked(byKey.view), picked(byClick.view));
  assert.equal(stageImage(byKey.view), stageImage(byClick.view));
});

test("the ends stop: no wrap, nothing is saved or announced again, the key is still taken", () => {
  const { view, widget, picks } = setup(saved(3, 0));
  const before = widget.value;
  const left = press(view, "ArrowLeft");
  assert.deepEqual(picked(view), ["0.png"]);
  assert.equal(left.defaultPrevented, true);
  assert.equal(widget.value, before);
  assert.deepEqual(picks, []);
  press(view, "End");
  assert.deepEqual(picked(view), ["2.png"]);
  const right = press(view, "ArrowRight");
  assert.deepEqual(picked(view), ["2.png"]);
  assert.equal(right.defaultPrevented, true);
  assert.deepEqual(picks, ["2.png"]);
  press(view, "Home");
  assert.deepEqual(picked(view), ["0.png"]);
});

test("on the XYZ 격자 tab a key shows the picked image, like a click on its thumbnail", () => {
  const { view } = setup(saved(3, 0, { grid: ref("grid.png"), tab: "grid" }));
  assert.equal(stageImage(view), "/view/grid.png");
  press(view, "ArrowRight");
  assert.equal(stageImage(view), "/view/1.png");
  assert.equal(pillNamed(view, "미리보기").attrs["aria-pressed"], "true");
});

test("keys work from anywhere in the result node: the node itself, the image, a gallery button", () => {
  const { view } = setup(saved(5, 0));
  press(view, "ArrowRight", view.element);
  press(view, "ArrowRight", view.element.byClass("ais-stage")[0]);
  press(view, "ArrowRight", view.element.byClass("ais-ib")[0]);
  press(view, "ArrowRight", thumbsOf(view)[0]);
  assert.deepEqual(picked(view), ["4.png"]);
});

test("keys are left alone while typing: input, textarea, select, contenteditable", () => {
  const { view, picks } = setup(saved(5, 1));
  for (const tag of ["input", "textarea", "select"]) {
    const field = new FakeElement(tag);
    view.element.append(field);
    const event = press(view, "ArrowRight", field);
    assert.equal(event.defaultPrevented, false, tag);
    assert.equal(event.stopped, false, tag);
  }
  const editable = new FakeElement("div");
  editable.isContentEditable = true;
  view.element.append(editable);
  assert.equal(press(view, "ArrowRight", editable).defaultPrevented, false);
  assert.deepEqual(picked(view), ["1.png"]);
  assert.deepEqual(picks, []);
});

test("keys with Ctrl / Alt / Meta / Shift are not the gallery's (Alt+← is the browser's back)", () => {
  const { view } = setup(saved(5, 1));
  for (const mod of ["ctrlKey", "altKey", "metaKey", "shiftKey"]) {
    const event = press(view, "ArrowLeft", strip(view), { [mod]: true });
    assert.equal(event.defaultPrevented, false, mod);
  }
  assert.deepEqual(picked(view), ["1.png"]);
});

test("a key somebody already handled is left alone", () => {
  const { view } = setup(saved(5, 1));
  press(view, "ArrowRight", strip(view), { defaultPrevented: true });
  assert.deepEqual(picked(view), ["1.png"]);
});

test("keys are left alone while the enlarged view is open, and come back when it closes", () => {
  // The XYZ grid opens alone in the viewer: with one image the viewer takes no arrow key itself.
  const { view } = setup(saved(3, 0, { grid: ref("grid.png"), tab: "grid" }));
  view.element.byClass("ais-stage-img")[0].fire("click");
  const box = globalThis.document.body.byClass("ais-lightbox")[0];
  assert.ok(box, "the viewer opened");
  const event = press(view, "ArrowRight", view.element);
  assert.equal(event.defaultPrevented, false);
  assert.deepEqual(picked(view), ["0.png"]);
  assert.equal(stageImage(view), "/view/grid.png");
  box.fire("click"); // closes
  press(view, "ArrowRight", view.element);
  assert.deepEqual(picked(view), ["1.png"]);
});

test("a key pressed elsewhere (the canvas, the panel) never reaches the gallery: nothing listens on document or window", () => {
  const { view, picks } = setup(saved(5, 1));
  const elsewhere = new FakeElement("canvas");
  globalThis.document.body.append(elsewhere);
  const event = press(view, "ArrowRight", elsewhere);
  assert.equal(event.defaultPrevented, false);
  assert.deepEqual(picked(view), ["1.png"]);
  assert.deepEqual(picks, []);
  assert.deepEqual(globalThis.document.listeners.filter((l) => l.type === "keydown"), []);
  assert.deepEqual(globalThis.window.listeners, []);
});

test("an empty gallery takes no key", () => {
  const { view } = setup();
  assert.equal(press(view, "ArrowRight").defaultPrevented, false);
});

// ---------------------------------------------------------------------------------------------- the canvas's shortcuts

// ComfyUI's defaults that run only when the pressed element is inside the canvas container, where a DOM widget is.
const CANVAS_SHORTCUTS = [
  ["Delete", {}], ["Backspace", {}], [".", {}], ["p", {}],
  ["c", { altKey: true }], ["b", { ctrlKey: true }], ["m", { ctrlKey: true }], ["a", { ctrlKey: true }],
  ["A", { ctrlKey: true }], ["b", { metaKey: true }], // Caps Lock; ⌘ counts as Ctrl
  ["V", { ctrlKey: true, shiftKey: true }], ["=", { altKey: true }], ["+", { altKey: true, shiftKey: true }],
  ["+", { altKey: true }], ["-", { altKey: true }],
];

test("Delete / Backspace and the canvas's other shortcuts pressed in the node stay in it: the node is not deleted", () => {
  const { view, widget, picks } = setup(saved(5, 1));
  const before = widget.value;
  const targets = () => [strip(view), thumbsOf(view)[2], view.element, view.element.byClass("ais-ib")[0]];
  for (const [key, mods] of CANVAS_SHORTCUTS) {
    for (const target of targets()) {
      const event = press(view, key, target, mods);
      assert.equal(event.stopped, true, `${key} ${JSON.stringify(mods)} on ${target.className}`);
      assert.equal(event.defaultPrevented, false, key); // only kept from the canvas: the browser's own part stays
    }
  }
  assert.deepEqual(picked(view), ["1.png"]);
  assert.equal(widget.value, before);
  assert.deepEqual(picks, []);
});

test("Delete stays in the node while the enlarged view is open, and when somebody already handled it", () => {
  const { view } = setup(saved(3, 0));
  view.element.byClass("ais-stage-img")[0].fire("click");
  assert.ok(globalThis.document.body.byClass("ais-lightbox")[0], "the viewer opened");
  assert.equal(press(view, "Delete", view.element).stopped, true);
  assert.equal(press(view, "Backspace", view.element).stopped, true);
  globalThis.document.body.byClass("ais-lightbox")[0].fire("click"); // closes
  assert.equal(press(view, "Delete", strip(view), { defaultPrevented: true }).stopped, true);
});

test("an empty gallery keeps Delete from the canvas too", () => {
  const { view } = setup();
  assert.equal(press(view, "Delete", view.element).stopped, true);
});

test("ComfyUI's shortcuts that work anywhere still pass through the node", () => {
  const { view } = setup(saved(5, 1));
  const anywhere = [
    ["Enter", { ctrlKey: true }], ["Enter", { ctrlKey: true, shiftKey: true }], ["Enter", { ctrlKey: true, altKey: true }],
    ["s", { ctrlKey: true }], ["z", { ctrlKey: true }], ["Backspace", { ctrlKey: true }], ["Delete", { shiftKey: true }],
    ["r", {}], ["w", {}], ["P", { shiftKey: true }], ["Escape", {}], ["Tab", {}], ["Enter", {}], [" ", {}], ["ArrowUp", {}],
  ];
  for (const [key, mods] of anywhere) {
    const event = press(view, key, strip(view), mods);
    assert.equal(event.stopped, false, `${key} ${JSON.stringify(mods)}`);
    assert.equal(event.defaultPrevented, false, key);
  }
});

test("a field in the node keeps its own Delete / Backspace (ComfyUI leaves text fields alone itself)", () => {
  const { view } = setup(saved(5, 1));
  const field = new FakeElement("input");
  view.element.append(field);
  assert.equal(press(view, "Backspace", field).stopped, false);
  assert.equal(press(view, "Delete", field).stopped, false);
});

// ---------------------------------------------------------------------------------------------- focus

test("the strip is one focusable stop; a click anywhere in the node gives the node the focus", () => {
  const { view } = setup(saved(3));
  assert.equal(strip(view).attrs.tabindex, "0");
  assert.equal(view.element.attrs.tabindex, "-1");
  assert.ok(strip(view).attrs["aria-label"]);
});

test("a clicked thumbnail keeps the keys in the gallery even where a click does not focus a button", () => {
  const { view } = setup(saved(5, 0));
  thumbsOf(view)[2].fire("click"); // nothing in the node has the focus (Safari: a click does not focus a button)
  assert.ok(view.element.contains(globalThis.document.activeElement));
  assert.equal(globalThis.document.activeElement, strip(view));
  assert.deepEqual(strip(view).focused, [{ preventScroll: true }]); // the page must not jump to it
});

test("a clicked thumbnail that has the focus keeps it: it is not drawn anew", () => {
  const { view } = setup(saved(5, 0));
  const thumb = thumbsOf(view)[2];
  thumb.focus();
  thumb.fire("click");
  assert.equal(globalThis.document.activeElement, thumb);
  assert.equal(thumbsOf(view)[2], thumb);
  assert.deepEqual(picked(view), ["2.png"]);
});

test("the focus follows the pick when a thumbnail holds it: Enter / Space do not jump back to the clicked one", () => {
  const { view, picks } = setup(saved(6, 0));
  const clicked = thumbsOf(view)[1];
  clicked.focus(); // a click focuses the button
  clicked.fire("click");
  for (let i = 0; i < 3; i++) press(view, "ArrowRight", globalThis.document.activeElement);
  assert.deepEqual(picked(view), ["4.png"]);
  assert.equal(globalThis.document.activeElement, thumbsOf(view)[4]);
  assert.deepEqual(thumbsOf(view)[4].focused, [{ preventScroll: true }]); // the strip is scrolled by centre() alone
  globalThis.document.activeElement.fire("click"); // what Enter / Space do on the focused button
  assert.deepEqual(picked(view), ["4.png"]);
  assert.deepEqual(picks, ["1.png", "2.png", "3.png", "4.png", "4.png"]);
  press(view, "Home", globalThis.document.activeElement);
  assert.equal(globalThis.document.activeElement, thumbsOf(view)[0]);
});

test("the focus stays where it is when no thumbnail holds it: the strip, the node, a gallery button", () => {
  const { view } = setup(saved(5, 0));
  for (const holder of [strip(view), view.element, view.element.byClass("ais-ib")[0]]) {
    holder.focus();
    press(view, "ArrowRight", holder);
    assert.equal(globalThis.document.activeElement, holder, holder.className);
  }
  assert.deepEqual(picked(view), ["3.png"]);
  assert.deepEqual(thumbsOf(view).flatMap((thumb) => thumb.focused), []);
});

test("on the XYZ 격자 tab the focus goes to the picked thumbnail of the strip drawn anew", () => {
  const { view } = setup(saved(4, 0, { grid: ref("grid.png"), tab: "grid" }));
  const clicked = thumbsOf(view)[2];
  clicked.focus();
  clicked.fire("click");
  assert.notEqual(thumbsOf(view)[2], clicked); // the tab pills changed: all of it was drawn anew
  assert.equal(globalThis.document.activeElement, thumbsOf(view)[2]);
  press(view, "ArrowRight", globalThis.document.activeElement);
  assert.equal(globalThis.document.activeElement, thumbsOf(view)[3]);
});

test("a pick never takes the focus from outside the node", () => {
  const { view } = setup(saved(5, 0));
  const prompt = new FakeElement("textarea");
  globalThis.document.body.append(prompt);
  prompt.focus();
  view.pick(3);
  assert.equal(globalThis.document.activeElement, prompt);
});

test("a redraw that removes the focused thumbnail hands the focus to the new strip", () => {
  const { view } = setup(saved(5, 0));
  thumbsOf(view)[2].focus();
  start(view); // the whole node is drawn anew
  assert.equal(globalThis.document.activeElement, strip(view));
  assert.deepEqual(strip(view).focused, [{ preventScroll: true }]);
  press(view, "ArrowRight", globalThis.document.activeElement);
  assert.deepEqual(picked(view), ["1.png"]);
});

test("a redraw never takes the focus from somewhere else (typing in the panel while images arrive)", () => {
  const { view } = setup(saved(5, 0));
  const prompt = new FakeElement("textarea");
  globalThis.document.body.append(prompt);
  prompt.focus();
  start(view);
  images(view, ["new.png"]);
  view.load();
  assert.equal(globalThis.document.activeElement, prompt);
});

// ---------------------------------------------------------------------------------------------- centring

test("a clicked thumbnail glides to the middle of the strip", () => {
  const { view } = setup(saved(10, 0));
  thumbsOf(view)[5].fire("click");
  assert.deepEqual(strip(view).scrolls, [{ left: 243, behavior: "smooth", from: 0 }]);
});

test("a key pick glides too, and stops at the strip's ends", () => {
  const { view } = setup(saved(10, 4));
  strip(view).scrolls.length = 0;
  press(view, "ArrowRight");
  assert.deepEqual(strip(view).scrolls.at(-1), { left: 243, behavior: "smooth", from: 173 });
  press(view, "End");
  assert.equal(strip(view).scrolls.at(-1).left, 416); // 716 - 300
  press(view, "Home");
  assert.equal(strip(view).scrolls.at(-1).left, 0);
});

test("clicking the picked thumbnail again brings it back to the middle", () => {
  const { view } = setup(saved(10, 5));
  strip(view).scrollLeft = 0; // scrolled away by hand
  thumbsOf(view)[5].fire("click");
  assert.equal(strip(view).scrollLeft, 243);
});

test("a thumbnail measured from the node instead of the strip lands in the same place", () => {
  const { view } = setup(saved(10, 0), { parent: "root" });
  thumbsOf(view)[5].fire("click");
  assert.equal(strip(view).scrollLeft, 243);
});

test("a view built from a saved workflow (opened, undo, another tab) shows its pick in the middle at once", () => {
  const { view, widget } = setup(saved(10, 5));
  assert.equal(strip(view).scrollLeft, 243);
  assert.deepEqual(strip(view).scrolls, []); // set, not animated
  widget.value = JSON.stringify(saved(10, 9));
  view.load();
  assert.equal(strip(view).scrollLeft, 416);
  assert.deepEqual(strip(view).scrolls, []);
});

test("new images are picked and shown at once: the strip is back at its start", () => {
  const { view } = setup(saved(10, 5));
  images(view, ["new.png"], false);
  assert.deepEqual(picked(view), ["new.png"]);
  assert.equal(strip(view).scrollLeft, 0);
  assert.deepEqual(strip(view).scrolls, []);
});

test("a redraw without a new pick keeps where the user scrolled", () => {
  const { view } = setup(saved(10, 5));
  strip(view).scrollLeft = 100; // by hand, away from the pick
  pillNamed(view, "XYZ 격자").fire("click");
  assert.equal(strip(view).scrollLeft, 100);
  pillNamed(view, "갤러리 자동 비우기").fire("click");
  assert.equal(strip(view).scrollLeft, 100);
  start(view);
  assert.equal(strip(view).scrollLeft, 100);
  view.onStudioEvent("running", { running: false, kind: "single", elapsed: 1000 });
  assert.equal(strip(view).scrollLeft, 100);
  assert.deepEqual(strip(view).scrolls, []);
});

test("a view drawn before it is on the page centres its pick when it gets a size, once", () => {
  const { view } = setup(saved(10, 5), { laidOut: false });
  assert.equal(strip(view).scrollLeft, 0);
  assert.equal(observers.length, 1);
  assert.deepEqual(observers[0].targets, [view.element]);
  layout.on = true;
  observers[0].callback();
  assert.equal(strip(view).scrollLeft, 243);
  strip(view).scrollLeft = 100; // by hand
  observers[0].callback(); // the node was resized, hidden and shown: not a new pick
  assert.equal(strip(view).scrollLeft, 100);
});

test("with reduced motion asked for, a pick jumps instead of gliding", () => {
  const { view } = setup(saved(10, 0));
  globalThis.matchMedia = (query) => ({ matches: query.includes("prefers-reduced-motion") });
  thumbsOf(view)[5].fire("click");
  assert.equal(strip(view).scrollLeft, 243);
  assert.deepEqual(strip(view).scrolls, []);
});

test("dispose() stops watching the node's size", () => {
  const { view } = setup(saved(3));
  view.dispose();
  assert.deepEqual(observers[0].targets, []);
});

// ---------------------------------------------------------------------------------------------- its stylesheet

test("the gallery's own stylesheet is added to the page once, whatever the number of result nodes", () => {
  setup(saved(3));
  const first = globalThis.document;
  const links = () => first.head.children.filter((child) => child.rel === "stylesheet");
  assert.equal(links().length, 1);
  assert.ok(links()[0].href.endsWith("/result_gallery.css"), links()[0].href);
  const widget = { value: "{}" };
  new ResultView(widget, { imageUrl: () => "", action: () => {} });
  assert.equal(links().length, 1);
});
