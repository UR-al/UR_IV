// node:test for the canvas-shortcut guard of the studio's panels (web/ai_studio/studio_dom.js guardCanvasKeys, put on the root
// of the control panel, the XYZ panel and the EXIF panel). ComfyUI runs its canvas shortcuts (Delete, Backspace, P, ...) on
// the selected node when the pressed element is inside the canvas container and is not a text box; a node's DOM widget is
// inside it, and a click in it selects the node. So Delete on a focused button of a panel would delete that panel.
// The fake document does not bubble: a key is fired on the panel's root with the pressed element as `target`.
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { FakeElement, flush, installFakeDom, mount } from "./fake_dom.mjs";
import { ExifView } from "../web/ai_studio/exif_view.js";
import { PanelView } from "../web/ai_studio/panel_view.js";
import { CANVAS_KEYS, guardCanvasKeys, isCanvasKey } from "../web/ai_studio/studio_dom.js";
import { XyzView } from "../web/ai_studio/xyz_view.js";

installFakeDom();

const el = (tag, extra = {}) => Object.assign(new FakeElement(tag), extra);
const key = (name, target, extra = {}) => ({ key: name, target, ctrlKey: false, altKey: false, shiftKey: false, metaKey: false, ...extra });
/** A key pressed on `target` as it reaches `root`; returns the event (`stopped`, `defaultPrevented`). */
const press = (root, name, target, extra = {}) => root.fire("keydown", key(name, target, extra));

// frontend 1.58.5: the default keybindings whose targetElementId is graph-canvas(-container)
const COMBOS = [
  ["Delete"], ["Backspace"], ["."], ["p"], ["P"], ["c", { altKey: true }], ["b", { ctrlKey: true }], ["m", { ctrlKey: true }],
  ["a", { ctrlKey: true }], ["a", { metaKey: true }], ["V", { ctrlKey: true, shiftKey: true }], ["=", { altKey: true }],
  ["+", { altKey: true }], ["+", { altKey: true, shiftKey: true }], ["-", { altKey: true }],
];
const OTHER = [
  ["Enter"], ["Enter", { ctrlKey: true }], ["Escape"], ["ArrowLeft"], ["a"], [" "], ["z", { ctrlKey: true }], ["c", { ctrlKey: true }],
  ["v", { ctrlKey: true }], ["s", { ctrlKey: true }], ["P", { shiftKey: true }], ["Tab"], ["r"], ["q"],
];

test("isCanvasKey: ComfyUI's canvas shortcuts pressed on a button, a switch or a list are the canvas's", () => {
  assert.equal(CANVAS_KEYS.size, 13);
  for (const tag of ["button", "select", "div", "span", "a"]) {
    for (const [name, extra] of COMBOS) assert.equal(isCanvasKey(key(name, el(tag), extra)), true, `${tag} ${name} ${JSON.stringify(extra || {})}`);
    for (const [name, extra] of OTHER) assert.equal(isCanvasKey(key(name, el(tag), extra)), false, `${tag} ${name} ${JSON.stringify(extra || {})}`);
  }
});

test("isCanvasKey: a text box keeps every key (ComfyUI leaves the plain ones to it too)", () => {
  const boxes = [el("textarea"), el("input"), el("div", { isContentEditable: true })];
  for (const box of boxes) for (const [name, extra] of [...COMBOS, ...OTHER]) assert.equal(isCanvasKey(key(name, box, extra)), false, `${box.tag} ${name}`);
  assert.equal(isCanvasKey(key("Delete", null)), true, "no element: nothing is being typed");
  assert.equal(isCanvasKey({ target: el("button") }), false, "an event without a key");
});

test("guardCanvasKeys: the key stops at the element, and the browser's own part stays", () => {
  const root = guardCanvasKeys(el("div"));
  const button = el("button");
  const stopped = press(root, "Delete", button);
  assert.equal(stopped.stopped, true);
  assert.equal(stopped.defaultPrevented, false, "p in a list still jumps to an option");
  assert.equal(press(root, "Enter", button).stopped, false);
  assert.equal(press(root, "Enter", button, { ctrlKey: true }).stopped, false, "ComfyUI's run goes through");
  assert.equal(press(root, "Backspace", el("textarea")).stopped, false);
});

// ---------------------------------------------------------------------------------------------- the three panels

const PANEL_CTX = {
  choices: async () => ({ items: [] }),
  engineRoom: { collapsed: () => true, toggle() {} },
  i2i: { get: () => "", set() {}, url: () => "", upload: async () => ({}), maskEditor() {} },
  hasRole: () => true,
  fetchChoices: async () => ({}),
};

test("control panel: Delete / ⌫ on a prompt box's name (where the preset list leaves the focus) do not reach the canvas", () => {
  const view = new PanelView({ value: JSON.stringify({ prompt: "1girl" }) }, { ...PANEL_CTX, boxLabel() {}, boxPreset: () => null });
  mount(view.element);
  const names = view.element.byClass("ais-box-label");
  assert.equal(names.length, 8);
  for (const name of names) {
    assert.equal(name.tag, "button");
    name.focus(); // preset_quick.js close(): the focus goes back to the name
    for (const pressed of ["Delete", "Backspace"]) {
      const event = press(view.element, pressed, name);
      assert.equal(event.stopped, true, `${name.text} ${pressed}`);
      assert.equal(event.defaultPrevented, false);
    }
  }
  for (const [pressed, extra] of COMBOS) assert.equal(press(view.element, pressed, names[5], extra).stopped, true, pressed);
});

test("control panel: every button and list is guarded, the text boxes keep their keys", () => {
  const view = new PanelView({ value: "{}" }, PANEL_CTX);
  mount(view.element);
  const buttons = view.element.byTag("button");
  const lists = view.element.byTag("select");
  assert.ok(buttons.length > 10 && lists.length > 3, `${buttons.length} buttons, ${lists.length} lists`);
  for (const control of [...buttons, ...lists]) assert.equal(press(view.element, "Delete", control).stopped, true, control.className);
  const boxes = [...view.element.byTag("textarea"), ...view.element.byTag("input")];
  assert.ok(boxes.length > 8);
  for (const box of boxes) {
    assert.equal(press(view.element, "Backspace", box).stopped, false, box.className);
    assert.equal(press(view.element, "p", box).stopped, false, box.className);
  }
  assert.equal(press(view.element, "Enter", buttons[0], { ctrlKey: true }).stopped, false, "Ctrl+Enter still runs");
  view.render();
  assert.equal(press(view.element, "Delete", view.element.byTag("button")[0]).stopped, true, "the guard is on the root: a redraw keeps it");
});

test("XYZ panel: ⌫ on 직접 입력 (the focus stays on it after a toggle) does not reach the canvas", async () => {
  const text = '"blue hair, long hair"\n"red hair, short hair"';
  const widget = { value: JSON.stringify({ on: true, axes: [{ axis: "prompt_sr", text }, { axis: "sampler", text: "euler" }] }) };
  const view = new XyzView(widget, { choices: async () => ({ items: ["euler", "lcm"] }), plan: async () => ({ labels: { x: ["a", "b"] } }), getState: () => ({}) });
  mount(view.element);
  await flush();
  for (const index of [0, 1]) {
    view.element.querySelector(`[data-k="typed-${index}"]`).fire("click");
    const pill = view.element.querySelector(`[data-k="typed-${index}"]`);
    assert.equal(document.activeElement, pill, "toggleTyped leaves the focus on the switch");
    for (const pressed of ["Backspace", "Delete", "p"]) assert.equal(press(view.element, pressed, pill).stopped, true, `${index} ${pressed}`);
  }
  for (const control of [...view.element.byTag("button"), ...view.element.byTag("select")]) {
    assert.equal(press(view.element, "Delete", control).stopped, true, control.className);
  }
  const box = view.element.byClass("ais-values")[0]; // the value box of 직접 입력
  assert.equal(press(view.element, "Backspace", box).stopped, false);
  assert.equal(JSON.parse(widget.value).axes[0].text, text);
  view.dispose();
});

test("EXIF panel: its buttons are guarded too", () => {
  const view = new ExifView(null, { imageUrl: () => "", readImage: async () => null, pullSettings() {} });
  mount(view.element);
  const buttons = view.element.byTag("button");
  assert.ok(buttons.length >= 1);
  for (const button of buttons) assert.equal(press(view.element, "Delete", button).stopped, true, button.text);
});
