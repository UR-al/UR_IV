// node:test with a tiny fake document (no DOM in node): the LoRA picker window's action bar and the options an XYZ row's
// button opens the window with. Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { actionBar } from "../web/ai_studio/lora_browser_detail.js";
import { loraAxisBody } from "../web/ai_studio/xyz_lora_view.js";

class FakeElement {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.attrs = {};
    this.listeners = {};
    this.className = "";
    this.style = {};
    this.value = "";
  }
  get ownerDocument() { return globalThis.document; }
  append(...nodes) { this.children.push(...nodes); }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  getAttribute(name) { return this.attrs[name] ?? null; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  fire(type) { for (const fn of this.listeners[type] || []) fn({ target: this, stopPropagation() {} }); }
  all(accept) {
    const own = accept(this) ? [this] : [];
    return own.concat(this.children.flatMap((child) => (child instanceof FakeElement ? child.all(accept) : [])));
  }
  byClass(name) { return this.all((el) => el.className.split(" ").includes(name)); }
}
globalThis.document = { createElement: (tag) => new FakeElement(tag), activeElement: null };
globalThis.Node = FakeElement;

const on = { strength() {}, step() {}, commit() {} };

test("the action bar leaves the strength controls out for xyz and when asked (a replace from an XYZ row)", () => {
  assert.equal(actionBar("add", on).el.byClass("ais-lb-strength").length, 1);
  assert.equal(actionBar("replace", on, {}).el.byClass("ais-lb-strength").length, 1);
  assert.equal(actionBar("xyz", on).el.byClass("ais-lb-strength").length, 0);
  const bar = actionBar("replace", on, { strength: false });
  assert.equal(bar.el.byClass("ais-lb-strength").length, 0);
  assert.equal(bar.el.all((el) => el.tag === "input").length, 0, "no slider, no number box");
  bar.update({ strength: 1, text: "이걸로 바꾸기" }); // still updates the main button
  assert.equal(bar.el.byClass("ais-lb-main")[0].textContent, "이걸로 바꾸기");
});

test("an XYZ row's file button opens the window in replace mode without strength controls (the row keeps its 강도)", () => {
  const block = { find: "", after: false, rows: [
    { lora: "", strength: null, trigger: "", epoch_mode: "off", epoch: null },
    { lora: "a_2.9b-000002", strength: 0.7, trigger: "", epoch_mode: "off", epoch: null },
  ] };
  const opened = [];
  const view = {
    block, baseItems: [], loras: ["LoKr\\A\\a_2.9b\\a_2.9b-000002.safetensors"],
    change: (mutate) => mutate(block), refreshFiles() {}, loraBrowser: (opts) => opened.push(opts),
  };
  const roots = loraAxisBody(view).flat(Infinity);
  const button = roots.flatMap((root) => root.byClass("ais-lb-pick"))[0];
  assert.ok(button, "the row has its file button");
  button.fire("click");
  assert.equal(opened.length, 1);
  assert.equal(opened[0].mode, "replace");
  assert.equal(opened[0].current, "a_2.9b-000002");
  assert.ok("strength" in opened[0]);
  assert.equal(opened[0].strength, null);
});

test("the card grid switches the browser's scroll anchoring off (a click on a card must not scroll the list)", async () => {
  const { cardGrid } = await import("../web/ai_studio/lora_browser_grid.js");
  const grid = cardGrid({ mode: "xyz", thumb: () => "", on: {} });
  assert.match(grid.scroller.style.cssText, /overflow-anchor:\s*none/);
});
