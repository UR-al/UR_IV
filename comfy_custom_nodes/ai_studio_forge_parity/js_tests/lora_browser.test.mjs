// node:test for the LoRA picker window's keyboard table (web/ai_studio/lora_browser.js). node has no DOM, so only the pure
// key -> action mapping is tested here; the window itself is checked in the browser (Task 12 smoke).
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { keyAction } from "../web/ai_studio/lora_browser.js";
import { initialState, reduce, visibleRuns } from "../web/ai_studio/lora_browser_state.js";
import { keepFocus, listKey } from "../web/ai_studio/lora_browser_grid.js";

const key = (k, mods = {}) => ({ key: k, shiftKey: false, ctrlKey: false, ...mods });

test("Enter adds and closes, Shift+Enter adds and keeps the window, in both focus places", () => {
  for (const focus of ["search", "grid"]) {
    assert.equal(keyAction(key("Enter"), focus), "add", focus);
    assert.equal(keyAction(key("Enter", { shiftKey: true }), focus), "addKeep", focus);
  }
});

test("Esc closes and Ctrl+F goes back to the search box, in both focus places", () => {
  for (const focus of ["search", "grid"]) {
    assert.equal(keyAction(key("Escape"), focus), "close", focus);
    assert.equal(keyAction(key("f", { ctrlKey: true }), focus), "focusSearch", focus);
    assert.equal(keyAction(key("F", { ctrlKey: true, shiftKey: true }), focus), "focusSearch", focus);
    assert.equal(keyAction(key("f", { metaKey: true }), focus), "focusSearch", focus); // Cmd+F on a Mac
  }
});

test("arrows move the card selection on the grid", () => {
  assert.deepEqual(keyAction(key("ArrowLeft"), "grid"), { type: "move", dx: -1, dy: 0 });
  assert.deepEqual(keyAction(key("ArrowRight"), "grid"), { type: "move", dx: 1, dy: 0 });
  assert.deepEqual(keyAction(key("ArrowUp"), "grid"), { type: "move", dx: 0, dy: -1 });
  assert.deepEqual(keyAction(key("ArrowDown"), "grid"), { type: "move", dx: 0, dy: 1 });
});

test("in the search box up/down move the cards but left/right stay with the text caret", () => {
  assert.deepEqual(keyAction(key("ArrowUp"), "search"), { type: "move", dx: 0, dy: -1 });
  assert.deepEqual(keyAction(key("ArrowDown"), "search"), { type: "move", dx: 0, dy: 1 });
  assert.equal(keyAction(key("ArrowLeft"), "search"), null);
  assert.equal(keyAction(key("ArrowRight"), "search"), null);
});

test("[ and ] step the epoch and f toggles the favorite on the grid", () => {
  assert.deepEqual(keyAction(key("["), "grid"), { type: "epoch", delta: -1 });
  assert.deepEqual(keyAction(key("]"), "grid"), { type: "epoch", delta: 1 });
  assert.deepEqual(keyAction(key("f"), "grid"), { type: "favorite" });
  assert.deepEqual(keyAction(key("F", { shiftKey: true }), "grid"), { type: "favorite" });
});

test("letter keys typed in the search box are text, not commands", () => {
  for (const k of ["f", "F", "[", "]", "a", "1", " ", "Backspace", "Delete", "Home", "End"]) {
    assert.equal(keyAction(key(k), "search"), null, k);
  }
});

test("other keys and other Ctrl combinations do nothing", () => {
  for (const focus of ["search", "grid"]) {
    assert.equal(keyAction(key("a"), focus === "grid" ? "grid" : "search"), null);
    assert.equal(keyAction(key("Tab"), focus), null, focus);
    assert.equal(keyAction(key("Enter", { ctrlKey: true }), focus), null, focus); // ComfyUI's queue shortcut
    assert.equal(keyAction(key("ArrowDown", { ctrlKey: true }), focus), null, focus);
    assert.equal(keyAction(key("a", { ctrlKey: true }), focus), null, focus);
  }
  assert.equal(keyAction(key("x"), "grid"), null);
  assert.equal(keyAction({}, "grid"), null);
});

// ---- the grid goes back to the top when what it lists changes (listKey), not on selection / epoch / strength changes
const item = (group, run, epoch) => {
  const stem = epoch == null ? run : `${run}-${String(epoch).padStart(6, "0")}`;
  return {
    name: `LoKr/${group}/${run}/${stem}.safetensors`, stem, run_key: `LoKr/${group}/${run}`, epoch, final: epoch == null, num_epochs: 2,
    family: "2.9b", algo: "lokr", group, group_label: group, run, run_short: run, trigger: "", date: "2026-11-01", has_meta: true, downloaded: false,
  };
};
const ITEMS = [item("Alpha", "alpha_2.9b_lokr", 1), item("Alpha", "alpha_2.9b_lokr", null), item("Beta", "beta_2.9b_lokr", 1), item("Gamma", "gamma_2.9b_lokr", 1)];
const start = () => initialState({ mode: "add", family: "2.9b", index: { items: ITEMS, prefs: {} } });
const keyOf = (s) => listKey(s, visibleRuns(s));

test("listKey stays the same for selection, epoch, strength, tray and favourites outside ★", () => {
  const s0 = start();
  let s = reduce(s0, { type: "select", runKey: "LoKr/Alpha/alpha_2.9b_lokr" });
  s = reduce(s, { type: "epoch", value: 1 });
  s = reduce(s, { type: "strengthStep", delta: 0.05 });
  s = reduce(s, { type: "added", pick: { key: "k", name: "n", strength: 1, label: "l" } });
  s = reduce(s, { type: "favorite", runKey: "LoKr/Beta/beta_2.9b_lokr" });
  assert.equal(keyOf(s), keyOf(s0));
});

test("listKey changes with search, folder, sort, 모든 모델 and a changed run list", () => {
  const s0 = start();
  for (const action of [
    { type: "query", text: "beta" }, { type: "folder", key: "lokr/Beta" }, { type: "sort", by: "date" }, { type: "showAll", on: true },
  ]) assert.notEqual(keyOf(reduce(s0, action)), keyOf(s0), action.type);
  const fav = reduce(reduce(s0, { type: "favorite", runKey: "LoKr/Beta/beta_2.9b_lokr" }), { type: "folder", key: "*fav" });
  assert.notEqual(keyOf(reduce(fav, { type: "favorite", runKey: "LoKr/Beta/beta_2.9b_lokr" })), keyOf(fav), "un-starring inside ★ empties the list");
});

// ---- keepFocus with a tiny fake DOM: focus never drops to <body> when the restored control cannot take it
function fakeDom() {
  const doc = { body: { name: "body" }, activeElement: null };
  doc.activeElement = doc.body;
  const el = (name, k, focusable = true) => ({
    name, k, focusable, parent: null,
    get isConnected() { return this.parent != null; },
    getAttribute(attr) { return attr === "data-k" ? this.k : null; },
    focus() { if (this.focusable && this.isConnected) doc.activeElement = this; },
  });
  const box = (name) => ({
    name, ownerDocument: doc, kids: [],
    contains(node) { return node === this || this.kids.includes(node); },
    querySelectorAll() { return this.kids.filter((n) => n.k != null); },
    put(...nodes) {
      for (const n of this.kids) n.parent = null;
      if (this.kids.includes(doc.activeElement)) doc.activeElement = doc.body; // a removed focused node blurs, like a browser
      this.kids = nodes;
      for (const n of nodes) n.parent = this;
    },
  });
  const outside = box("outside");
  return { doc, el, box, outside };
}

test("keepFocus moves the focus to the redrawn control with the same data-k", () => {
  const { doc, el, box } = fakeDom();
  const tree = box("tree");
  const old = el("old", "t:a");
  tree.put(old, el("other", "t:b"));
  old.focus();
  const fresh = el("fresh", "t:a");
  keepFocus(tree, () => tree.put(el("x", "t:b"), fresh));
  assert.equal(doc.activeElement, fresh);
});

test("keepFocus falls back when the redrawn control cannot take the focus (folded tree), never leaving it on body", () => {
  const { doc, el, box, outside } = fakeDom();
  const tree = box("tree");
  const old = el("old", "t:a");
  tree.put(old);
  old.focus();
  const search = el("search", null);
  const hiddenFallback = el("hidden", null, false);
  outside.put(search, hiddenFallback);
  keepFocus(tree, () => tree.put(el("hidden twin", "t:a", false)), hiddenFallback, search);
  assert.equal(doc.activeElement, search);
  // the control is gone altogether: same fallback
  const old2 = el("old2", "t:z");
  tree.put(old2);
  old2.focus();
  keepFocus(tree, () => tree.put(), search);
  assert.equal(doc.activeElement, search);
});

test("keepFocus leaves the focus alone when it was not inside the redrawn part", () => {
  const { doc, el, box, outside } = fakeDom();
  const tree = box("tree");
  tree.put(el("a", "t:a"));
  const search = el("search", null);
  outside.put(search);
  search.focus();
  keepFocus(tree, () => tree.put(el("b", "t:a")), tree.kids[0]);
  assert.equal(doc.activeElement, search);
});
