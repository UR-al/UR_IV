// node:test for the XYZ text blocks' list box following the block that was acted on (web/ai_studio/xyz_blocks.js): the list
// scrolls inside from 300px, and Enter, Alt+↑/↓, ↑ ↓ and ✕ put the focus on a block that may be outside what is shown. The
// focus is given without a scroll (the panel's column must not move), so the list box's own scrollTop brings the block in.
// The fake document has no layout: this file gives every block a height and a place (24px a block, 4px between, 5px
// padding, a 298px high list), the numbers of xyz_blocks.css.
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { FakeElement, installFakeDom, mount } from "./fake_dom.mjs";
import { BLOCK_AXES, blockField, revealScrollTop } from "../web/ai_studio/xyz_blocks.js";
import { joinValues, splitValues } from "../web/ai_studio/xyz_chips.js";

const ROW = 24;
const GAP = 4;
const PAD = 5;
const VIEW = 298;
const topOf = (index) => PAD + index * (ROW + GAP);
const isBlock = (element) => element.classList.contains("ais-blk");

Object.defineProperties(FakeElement.prototype, {
  offsetParent: { configurable: true, get() { return this.parentElement; } },
  offsetTop: { configurable: true, get() { return isBlock(this) ? topOf(this.parentElement.children.indexOf(this)) : 0; } },
  offsetHeight: { configurable: true, get() { return isBlock(this) ? ROW : 0; } },
});

/** A field of `count` one-letter values in a laid-out list box scrolled to `scroll`. */
function field(count, scroll = 0) {
  installFakeDom();
  const store = { text: Array.from({ length: count }, (_, index) => String.fromCharCode(97 + index)).join("\n") };
  const made = blockField({
    name: "X", scrollKey: "blocks-0-prompt_sr", texts: BLOCK_AXES.prompt_sr,
    read: () => splitValues(store.text),
    write: (values) => { store.text = joinValues(values); },
  });
  mount(made.el);
  const box = made.el.byClass("ais-blks")[0];
  box.clientHeight = VIEW;
  box.scrollTop = scroll;
  return {
    store, box,
    areas: () => made.el.byClass("ais-blk-text"),
    values: () => made.el.byClass("ais-blk-text").map((area) => area.value),
    button: (index, name) => made.el.byClass("ais-blk")[index].byClass(`ais-blk-${name}`)[0],
    add: made.el.byClass("ais-blk-add")[0],
    key: (target, key, extra = {}) => target.fire("keydown", { key, altKey: false, shiftKey: false, ctrlKey: false, metaKey: false, ...extra }),
    /** Whether block `index` is wholly inside what the list shows. */
    shown: (index) => topOf(index) >= box.scrollTop && topOf(index) + ROW <= box.scrollTop + VIEW,
  };
}

test("revealScrollTop: a block in view stays, a cut one comes just in (with the padding), a block higher than the list shows its top", () => {
  assert.equal(revealScrollTop({ top: 100, height: 24, view: 298, scroll: 0, pad: 5 }), null);
  assert.equal(revealScrollTop({ top: 285, height: 24, view: 298, scroll: 0, pad: 5 }), 16); // 285 + 24 + 5 - 298
  assert.equal(revealScrollTop({ top: 285, height: 24, view: 298, scroll: 16, pad: 5 }), null);
  assert.equal(revealScrollTop({ top: 89, height: 24, view: 298, scroll: 100, pad: 5 }), 84); // 89 - 5
  assert.equal(revealScrollTop({ top: 5, height: 24, view: 298, scroll: 3, pad: 5 }), 0);
  assert.equal(revealScrollTop({ top: 2, height: 24, view: 298, scroll: 0, pad: 5 }), null, "never below 0");
  assert.equal(revealScrollTop({ top: 200, height: 400, view: 298, scroll: 0, pad: 5 }), 195, "too high to fit: its top");
  assert.equal(revealScrollTop({ top: 200, height: 400, view: 298, scroll: 195, pad: 5 }), null);
  assert.equal(revealScrollTop({ top: 200, height: 400, view: 298, scroll: 250, pad: 5 }), null, "inside a block higher than the list: left where it is");
  assert.equal(revealScrollTop({ top: 100, height: 24, view: 298, scroll: 0 }), null, "pad is optional");
});

test("revealScrollTop: a list that is not laid out has no answer", () => {
  assert.equal(revealScrollTop({ top: 0, height: 0, view: 0, scroll: 0, pad: 5 }), null);
  assert.equal(revealScrollTop({ top: undefined, height: 24, view: 298, scroll: 0, pad: 5 }), null);
  assert.equal(revealScrollTop({ top: NaN, height: 24, view: 298, scroll: 0, pad: 5 }), null);
});

test("Enter in the last block that shows: the new block under the fold is brought into the list", () => {
  const f = field(12); // 12 blocks are 342px of content in a 298px list
  assert.equal(f.shown(9), true);
  assert.equal(f.shown(10), false, "block 10 is cut by the list's bottom");
  const area = f.areas()[9];
  area.focus();
  f.key(area, "Enter");
  assert.equal(f.values()[10], "");
  assert.equal(document.activeElement, f.areas()[10]);
  assert.equal(f.box.scrollTop, 16);
  assert.equal(f.shown(10), true);
});

test("Enter where the new block shows anyway leaves the list where it is", () => {
  const f = field(12, 10);
  const area = f.areas()[3];
  area.focus();
  f.key(area, "Enter");
  assert.equal(document.activeElement, f.areas()[4]);
  assert.equal(f.box.scrollTop, 10);
});

test("Alt+↓ and the ↓ button move a block under the fold and the list follows it; Alt+↑ and ↑ over the top", () => {
  const down = field(12);
  const area = down.areas()[9];
  area.focus();
  down.key(area, "ArrowDown", { altKey: true });
  assert.equal(down.values()[10], "j");
  assert.equal(document.activeElement, down.areas()[10]);
  assert.equal(down.shown(10), true);
  assert.equal(down.box.scrollTop, 16);
  down.button(10, "down").fire("click");
  assert.equal(down.values()[11], "j");
  assert.equal(document.activeElement, down.button(11, "up"), "↓ is greyed at the end: the block's other arrow");
  assert.equal(down.box.scrollTop, 44); // 313 + 24 + 5 - 298: the end of the list
  assert.equal(down.shown(11), true);

  const up = field(12, 44); // the end of the list: blocks 0 and 1 are over the top
  assert.equal(up.shown(1), false);
  const second = up.areas()[2];
  second.focus();
  up.key(second, "ArrowUp", { altKey: true });
  assert.equal(up.values()[1], "c");
  assert.equal(up.box.scrollTop, 28); // 33 - 5
  assert.equal(up.shown(1), true);
  up.button(1, "up").fire("click");
  assert.equal(up.values()[0], "c");
  assert.equal(up.box.scrollTop, 0);
});

test("+ 블록 shows the new last block; ✕ shows the block the focus went to", () => {
  const f = field(12);
  f.add.fire("click");
  assert.equal(document.activeElement, f.areas()[12]);
  assert.equal(f.box.scrollTop, 72); // 341 + 24 + 5 - 298: the list's end
  assert.equal(f.shown(12), true);

  const last = field(12, 44);
  last.button(11, "x").fire("click"); // the focus goes to the block before, now the last
  assert.equal(document.activeElement, last.button(10, "x"));
  assert.equal(last.shown(10), true);
  assert.equal(last.box.scrollTop, 44, "it shows: nothing moves");

  const top = field(12, 20); // block 0 is cut by the top
  top.button(1, "up").focus();
  top.button(1, "up").fire("click");
  assert.equal(document.activeElement, top.button(0, "down"));
  assert.equal(top.box.scrollTop, 0);
});

test("typing in a block the fold cuts brings it in (a text that grows by a line would go under it); a block that shows moves nothing", () => {
  const f = field(12, 30); // the last block is 313..337, the list shows 30..328
  const type = (index, value) => {
    const area = f.areas()[index];
    area.focus();
    area.value = value;
    area.fire("input");
    return area;
  };
  type(5, "red hair");
  assert.equal(f.box.scrollTop, 30);
  const last = type(11, "blue hair, long hair");
  assert.equal(f.box.scrollTop, 44);
  assert.equal(f.shown(11), true);
  assert.equal(f.areas()[11], last, "typing draws nothing anew");
  assert.equal(document.activeElement, last);
  assert.equal(splitValues(f.store.text)[11], "blue hair, long hair");
});
