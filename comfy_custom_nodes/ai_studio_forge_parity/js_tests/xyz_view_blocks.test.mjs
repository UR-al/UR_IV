// node:test for the XYZ panel's Prompt S/R / Prompt order axes drawn as text blocks (web/ai_studio/xyz_view.js with
// xyz_blocks.js), in a tiny fake document: the axis' value text stays what is saved and planned, an edit of a block does not
// redraw the panel, and a redraw keeps what was scrolled.
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { flush, installFakeDom, mount } from "./fake_dom.mjs";
import { XyzView } from "../web/ai_studio/xyz_view.js";

const SR = '"blue hair, long hair"\n"red hair, short hair"';

async function xyzView(axes, { on = false, plans = [] } = {}) {
  installFakeDom();
  const widget = { value: JSON.stringify({ on, axes }) };
  const view = new XyzView(widget, {
    choices: async () => ({ items: [] }),
    plan: async (body) => { plans.push(body); return { labels: { x: ["a", "b"] } }; },
    getState: () => ({}),
  });
  mount(view.element);
  await flush();
  view.widgetOf = widget;
  return view;
}

const one = (view, name) => view.element.byClass(name)[0];
const axisEl = (view, index) => view.element.byClass("ais-ax")[index];
const blockTexts = (view, index) => axisEl(view, index).byClass("ais-blk-text").map((area) => area.value);
const tags = (view, index) => axisEl(view, index).byClass("ais-blk-tag").map((tag) => tag.textContent);
const typedPill = (view, index) => axisEl(view, index).byClass("ais-pill")[0]; // 직접 입력
const type = (area, value) => {
  area.focus();
  area.value = value;
  area.fire("input");
};

test("Prompt S/R and Prompt order show their value text as text blocks; other text axes keep their text box", async () => {
  const view = await xyzView([{ axis: "prompt_sr", text: SR }, { axis: "prompt_order", text: '"a, b", c' }, { axis: "sam3_prompt", text: "face, hand" }]);
  assert.deepEqual(blockTexts(view, 0), ["blue hair, long hair", "red hair, short hair"]);
  assert.deepEqual(tags(view, 0), ["찾을 글", "바꿀 글"]);
  assert.equal(axisEl(view, 0).byClass("ais-values").length, 0);
  assert.equal(axisEl(view, 0).byClass("ais-blks")[0].classList.contains("tall"), true, "X is the tall one, as its text box was");
  assert.equal(axisEl(view, 0).byClass("ais-blks")[0].attrs["data-scroll"], "blocks-0-prompt_sr");
  assert.deepEqual(blockTexts(view, 1), ["a, b", "c"]);
  assert.deepEqual(tags(view, 1), ["조각 1", "조각 2"]);
  assert.equal(axisEl(view, 1).byClass("ais-blks")[0].classList.contains("tall"), false);
  assert.equal(axisEl(view, 2).byClass("ais-blks").length, 0);
  assert.equal(axisEl(view, 2).byClass("ais-values")[0].value, "face, hand");
  assert.equal(typedPill(view, 2), undefined, "no 직접 입력 switch on a plain text axis");
  assert.equal(view.settings.axes[0].text, SR, "drawing changes nothing");
  assert.equal(view.widgetOf.value, JSON.stringify({ on: false, axes: [{ axis: "prompt_sr", text: SR }, { axis: "prompt_order", text: '"a, b", c' }, { axis: "sam3_prompt", text: "face, hand" }] }));
});

test("typing in a block saves the value text (CSV, one value a line) without a redraw: scroll and focus stay", async () => {
  const view = await xyzView([{ axis: "prompt_sr", text: SR }, { axis: "steps", text: "20" }]);
  const column = one(view, "ais-xb");
  const box = one(view, "ais-blks");
  column.scrollTop = 140;
  box.scrollTop = 25;
  const area = axisEl(view, 0).byClass("ais-blk-text")[1];
  type(area, "green hair, twintails, smile");
  const text = '"blue hair, long hair"\n"green hair, twintails, smile"';
  assert.equal(view.settings.axes[0].text, text);
  assert.equal(JSON.parse(view.widgetOf.value).axes[0].text, text);
  assert.equal(one(view, "ais-xb"), column, "nothing was drawn anew");
  assert.equal(axisEl(view, 0).byClass("ais-blk-text")[1], area);
  assert.deepEqual([column.scrollTop, box.scrollTop], [140, 25]);
  assert.equal(document.activeElement, area);
});

test("+ 블록, ✕ and ↑ change the value text and redraw only the blocks", async () => {
  const view = await xyzView([{ axis: "prompt_sr", text: SR }]);
  const column = one(view, "ais-xb");
  column.scrollTop = 90;
  one(view, "ais-blk-add").fire("click");
  assert.equal(view.settings.axes[0].text, `${SR}\n""`);
  type(axisEl(view, 0).byClass("ais-blk-text")[2], "bob cut");
  axisEl(view, 0).byClass("ais-blk-up")[2].fire("click");
  assert.equal(view.settings.axes[0].text, '"blue hair, long hair"\nbob cut\n"red hair, short hair"');
  axisEl(view, 0).byClass("ais-blk-x")[0].fire("click");
  assert.equal(view.settings.axes[0].text, 'bob cut\n"red hair, short hair"');
  assert.deepEqual(tags(view, 0), ["찾을 글", "바꿀 글"]);
  assert.equal(one(view, "ais-xb"), column);
  assert.equal(column.scrollTop, 90);
});

test("a switch redraws the panel: the block list keeps its place, and the blocks their values", async () => {
  const view = await xyzView([{ axis: "prompt_sr", text: "a\nb\nc\nd\ne\nf\ng" }]);
  one(view, "ais-xb").scrollTop = 200;
  one(view, "ais-blks").scrollTop = 44;
  const before = one(view, "ais-blks");
  view.element.byClass("ais-xopts")[0].byClass("ais-pill")[0].fire("click"); // 범례 그리기
  assert.notEqual(one(view, "ais-blks"), before, "the panel was drawn anew");
  assert.deepEqual([one(view, "ais-xb").scrollTop, one(view, "ais-blks").scrollTop], [200, 44]);
  assert.deepEqual(blockTexts(view, 0), ["a", "b", "c", "d", "e", "f", "g"]);
});

test("직접 입력 swaps the blocks for the value text and back; quotes written by hand come back as the right blocks", async () => {
  const view = await xyzView([{ axis: "prompt_sr", text: "" }]);
  assert.deepEqual(blockTexts(view, 0), [""], "an empty axis shows where to type the text to find");
  assert.match(typedPill(view, 0).attrs.title, /따옴표/);
  typedPill(view, 0).focus();
  typedPill(view, 0).fire("click");
  const box = axisEl(view, 0).byClass("ais-values")[0];
  assert.equal(axisEl(view, 0).byClass("ais-blks").length, 0);
  assert.match(box.attrs.placeholder, /"/, "the text box says how to keep a comma in one value");
  assert.equal(document.activeElement, typedPill(view, 0), "the focus stays on the switch");
  box.value = '"1girl, solo", "2girls, yuri", plain';
  box.fire("input");
  assert.equal(view.settings.axes[0].text, '"1girl, solo", "2girls, yuri", plain');
  typedPill(view, 0).fire("click");
  assert.deepEqual(blockTexts(view, 0), ["1girl, solo", "2girls, yuri", "plain"]);
  assert.equal(view.settings.axes[0].text, '"1girl, solo", "2girls, yuri", plain', "showing the blocks rewrites nothing");
});

test("picking Prompt S/R for an axis shows its blocks; a loaded workflow shows blocks again", async () => {
  const view = await xyzView([{ axis: "seed", text: "1, 2" }]);
  const select = axisEl(view, 0).byTag("select")[0];
  select.value = "prompt_sr";
  select.fire("change");
  assert.deepEqual(blockTexts(view, 0), ["1", "2"], "the value text is kept, as with every axis type");
  typedPill(view, 0).fire("click");
  assert.equal(axisEl(view, 0).byClass("ais-blks").length, 0);
  view.widgetOf.value = JSON.stringify({ on: false, axes: [{ axis: "prompt_sr", text: SR }] });
  view.load();
  assert.deepEqual(blockTexts(view, 0), ["blue hair, long hair", "red hair, short hair"]);
});

test("an edit of a block asks the planner again with the new value text", async () => {
  const plans = [];
  const view = await xyzView([{ axis: "prompt_sr", text: SR }], { on: true, plans });
  type(axisEl(view, 0).byClass("ais-blk-text")[0], "blue hair");
  await new Promise((done) => setTimeout(done, 480));
  assert.equal(plans.at(-1).axes[0].text, 'blue hair\n"red hair, short hair"');
  view.dispose();
});

test("the panel watches its own size to fit the text blocks (it is drawn before it is on the page), until it is disposed", async () => {
  const made = [];
  globalThis.ResizeObserver = class {
    constructor(run) { this.run = run; this.watched = []; this.off = false; made.push(this); }
    observe(element) { this.watched.push(element); }
    disconnect() { this.off = true; }
  };
  try {
    const view = await xyzView([{ axis: "prompt_sr", text: SR }]);
    assert.equal(made.length, 1);
    assert.deepEqual(made[0].watched, [view.element]);
    let grown = 0;
    const grow = view.fields[0].grow;
    view.fields[0].grow = () => { grown += 1; grow(); };
    made[0].run();
    assert.equal(grown, 1);
    view.dispose();
    assert.equal(made[0].off, true);
  } finally {
    delete globalThis.ResizeObserver;
  }
});
