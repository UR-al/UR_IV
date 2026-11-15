// node:test for the XYZ panel drawn into a tiny fake document (web/ai_studio/xyz_view.js): the value blocks of a list axis
// and what a redraw keeps. A redraw builds every element anew, and a new scroll box starts at the top as in a browser, so
// what was scrolled must be put back; an edit of the blocks must not redraw at all (the focus would drop to the page).
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { FakeElement, flush, installFakeDom, mount } from "./fake_dom.mjs";
import { keepScroll } from "../web/ai_studio/studio_dom.js";
import { XyzView } from "../web/ai_studio/xyz_view.js";

const SAMPLERS = Array.from({ length: 40 }, (_, index) => `sampler_${index}`);
const LISTS = { samplers: SAMPLERS, schedulers: ["normal", "karras", "beta"], unets: ["anima\\a-3.8b.safetensors", "anima\\b-2.9b.safetensors"] };

async function xyzView(axes = [{ axis: "sampler", text: "sampler_3\nsampler_7" }, { axis: "steps", text: "20\n30" }], lists = LISTS) {
  installFakeDom();
  const widget = { value: JSON.stringify({ on: false, axes }) };
  const asked = [];
  const view = new XyzView(widget, {
    choices: async (kind) => { asked.push(kind); return { items: lists[kind] || [] }; },
    plan: async () => ({}),
    getState: () => ({}),
  });
  mount(view.element);
  await flush();
  view.asked = asked;
  view.widgetOf = widget;
  return view;
}

const one = (view, name) => view.element.byClass(name)[0];
const axisEl = (view, index) => view.element.byClass("ais-ax")[index];
const chipTexts = (view, index) => axisEl(view, index).byClass("ais-chip-text").map((chip) => chip.textContent);
const typeBox = (view, index) => axisEl(view, index).byTag("select")[0];
const typedPill = (view, index) => axisEl(view, index).byClass("ais-pill")[0]; // 직접 입력
const pick = (view, index, axis) => {
  const box = typeBox(view, index);
  box.value = axis;
  box.fire("change");
};

test("a list axis shows its value text as blocks; a number axis keeps its text box", async () => {
  const view = await xyzView([{ axis: "sampler", text: "sampler_3, sampler_7\nnot_a_sampler" }, { axis: "steps", text: "20\n30" }, { axis: "sam3_enable", text: "True" }]);
  assert.deepEqual(chipTexts(view, 0), ["sampler_3", "sampler_7", "not_a_sampler"]);
  assert.deepEqual(axisEl(view, 0).byClass("ais-chip").map((chip) => chip.classList.contains("warn")), [false, false, true]);
  assert.equal(axisEl(view, 0).byTag("textarea").length, 0);
  assert.equal(axisEl(view, 0).byClass("ais-chips")[0].classList.contains("tall"), true, "X is the tall one, as its text box was");
  assert.equal(axisEl(view, 1).byTag("textarea")[0].value, "20\n30");
  assert.equal(axisEl(view, 1).byClass("ais-chips").length, 0);
  assert.equal(typedPill(view, 1), undefined, "no 직접 입력 switch where there is no list");
  assert.deepEqual(chipTexts(view, 2), ["True"], "a switch axis lists True / False");
  assert.deepEqual(view.asked.filter((kind) => kind !== "loras"), ["samplers"], "only the list a drawn axis needs is read");
  assert.equal(view.settings.axes[0].text, "sampler_3, sampler_7\nnot_a_sampler", "drawing changes nothing");
});

test("an edit of the blocks writes the value text (one value a line) without a redraw: scroll and focus stay", async () => {
  const view = await xyzView();
  const column = one(view, "ais-xb");
  const box = one(view, "ais-chips");
  const values = one(view, "ais-values");
  column.scrollTop = 150;
  box.scrollTop = 30;
  values.scrollTop = 60;
  const search = one(view, "ais-chip-search");
  search.focus();
  search.fire("click");
  axisEl(view, 0).byClass("ais-chip-opt").find((option) => option.textContent === "sampler_20").fire("click");
  assert.equal(view.settings.axes[0].text, "sampler_3\nsampler_7\nsampler_20");
  assert.equal(JSON.parse(view.widgetOf.value).axes[0].text, "sampler_3\nsampler_7\nsampler_20");
  assert.deepEqual(chipTexts(view, 0), ["sampler_3", "sampler_7", "sampler_20"]);
  assert.equal(one(view, "ais-xb"), column, "nothing was drawn anew");
  assert.equal(one(view, "ais-chips"), box);
  assert.equal(one(view, "ais-values"), values);
  assert.deepEqual([column.scrollTop, values.scrollTop], [150, 60]);
  assert.equal(document.activeElement, search);

  const cross = axisEl(view, 0).byClass("ais-chip-x")[0];
  cross.focus();
  cross.fire("click");
  assert.equal(view.settings.axes[0].text, "sampler_7\nsampler_20");
  assert.equal(one(view, "ais-xb"), column);
  assert.equal(column.scrollTop, 150);
  assert.equal(document.activeElement.classList.contains("ais-chip-x"), true, "the focus stays on a block, not on the page");

  // both buttons grey themselves with their own click: the focus must go to the add box, not to the page
  const press = (text) => {
    const button = axisEl(view, 0).byTag("button").find((item) => item.textContent === text);
    button.focus();
    button.fire("click");
    assert.equal(button.disabled, true, text);
    assert.ok(document.activeElement === search, `${text}: the focus is in the add box`);
  };
  press("전부 넣기");
  assert.equal(view.settings.axes[0].text, SAMPLERS.join("\n"));
  press("모두 지우기");
  assert.equal(view.settings.axes[0].text, "");
  assert.equal(one(view, "ais-xb"), column);
  assert.equal(column.scrollTop, 150);
  view.dispose();
});

test("a value with a comma is written quoted, so the planner reads it as one value", async () => {
  const view = await xyzView([{ axis: "checkpoint", text: "" }], { unets: ["sub\\a, b.safetensors", "plain.safetensors"] });
  const search = one(view, "ais-chip-search");
  search.fire("click");
  axisEl(view, 0).byClass("ais-chip-opt")[0].fire("click");
  axisEl(view, 0).byClass("ais-chip-opt")[0].fire("click"); // the list was redrawn: the next value is now its first
  assert.equal(view.settings.axes[0].text, '"sub\\a, b.safetensors"\nplain.safetensors');
  assert.deepEqual(chipTexts(view, 0), ["sub\\a, b.safetensors", "plain.safetensors"]);
});

test("a list that arrives later marks the unknown blocks in place", async () => {
  const lists = { samplers: [] };
  const view = await xyzView([{ axis: "sampler", text: "sampler_1\ngone" }], lists);
  const column = one(view, "ais-xb");
  assert.deepEqual(axisEl(view, 0).byClass("ais-chip").map((chip) => chip.classList.contains("warn")), [false, false], "an empty list marks nothing");
  lists.samplers = SAMPLERS;
  one(view, "ais-chip-search").fire("click"); // opening the list reads it again
  await flush();
  assert.deepEqual(axisEl(view, 0).byClass("ais-chip").map((chip) => chip.classList.contains("warn")), [false, true]);
  assert.equal(axisEl(view, 0).byClass("ais-chip-opt").length, SAMPLERS.length - 1, "the open list follows");
  assert.equal(one(view, "ais-xb"), column);
});

test("a switch and the XYZ pill redraw without moving what was scrolled", async () => {
  const view = await xyzView();
  const scrollAll = () => {
    one(view, "ais-chips").scrollTop = 30;
    one(view, "ais-xb").scrollTop = 90;
    one(view, "ais-values").scrollTop = 45;
  };
  const kept = () => [one(view, "ais-chips").scrollTop, one(view, "ais-xb").scrollTop, one(view, "ais-values").scrollTop];

  scrollAll();
  const before = one(view, "ais-xb");
  one(view, "ais-xopts").byClass("ais-pill")[0].fire("click"); // 범례 그리기
  assert.notEqual(one(view, "ais-xb"), before); // drawn anew
  assert.deepEqual(kept(), [30, 90, 45]);

  one(view, "ais-xh").byClass("ais-pill")[0].fire("click"); // XYZ 켬
  assert.equal(view.settings.on, true);
  assert.deepEqual(kept(), [30, 90, 45]);
  view.dispose(); // the estimate timer
});

test("직접 입력 swaps the blocks for the text box and back; the focus stays on the switch", async () => {
  const view = await xyzView();
  typedPill(view, 0).focus();
  typedPill(view, 0).fire("click");
  assert.equal(axisEl(view, 0).byClass("ais-chips").length, 0);
  const box = axisEl(view, 0).byTag("textarea")[0];
  assert.equal(box.value, "sampler_3\nsampler_7");
  assert.equal(typedPill(view, 0).classList.contains("on"), true);
  assert.equal(document.activeElement, typedPill(view, 0));
  box.value = "sampler_1, sampler_2";
  box.fire("input");
  assert.equal(view.settings.axes[0].text, "sampler_1, sampler_2");
  typedPill(view, 0).fire("click");
  assert.deepEqual(chipTexts(view, 0), ["sampler_1", "sampler_2"]);
  assert.equal(view.settings.axes[0].text, "sampler_1, sampler_2", "the text is only rewritten by an edit of the blocks");
  view.dispose();
});

test("picking an axis' type keeps the other axes as they are; its own boxes start anew", async () => {
  const view = await xyzView([{ axis: "sampler", text: "sampler_3" }, { axis: "scheduler", text: "karras" }]);
  typedPill(view, 0).fire("click"); // X: 직접 입력
  axisEl(view, 0).byTag("textarea")[0].scrollTop = 70;
  one(view, "ais-xb").scrollTop = 90;
  axisEl(view, 1).byClass("ais-chips")[0].scrollTop = 20;

  pick(view, 1, "checkpoint");
  assert.equal(view.settings.axes[1].axis, "checkpoint");
  assert.equal(axisEl(view, 0).byTag("textarea").length, 1, "X stays in 직접 입력");
  assert.equal(axisEl(view, 0).byTag("textarea")[0].scrollTop, 70);
  assert.equal(one(view, "ais-xb").scrollTop, 90);
  assert.equal(axisEl(view, 1).byClass("ais-chips")[0].scrollTop, 0, "another axis' values: from the top");

  pick(view, 0, "scheduler");
  assert.equal(axisEl(view, 0).byTag("textarea").length, 0, "its own 직접 입력 ends");
  assert.equal(axisEl(view, 0).byClass("ais-chips").length, 1);
  view.dispose();
});

test("축 바꾸기 takes 직접 입력 along with its axis", async () => {
  const view = await xyzView([{ axis: "sampler", text: "sampler_3" }, { axis: "scheduler", text: "karras" }]);
  typedPill(view, 0).fire("click");
  one(view, "ais-xb").scrollTop = 90;
  one(view, "ais-swaps").byClass("ais-swap")[0].fire("click"); // X↔Y
  assert.deepEqual(view.settings.axes.map((entry) => entry.axis), ["scheduler", "sampler", "nothing"]);
  assert.equal(axisEl(view, 0).byClass("ais-chips").length, 1);
  assert.equal(axisEl(view, 1).byTag("textarea")[0].value, "sampler_3");
  assert.equal(one(view, "ais-xb").scrollTop, 90);
  view.dispose();
});

test("a loaded workflow or 설정 가져오기 keeps the column's place only", async () => {
  const view = await xyzView();
  const scrollAll = () => {
    one(view, "ais-chips").scrollTop = 30;
    one(view, "ais-xb").scrollTop = 90;
    one(view, "ais-values").scrollTop = 45;
  };
  const kept = () => [one(view, "ais-chips").scrollTop, one(view, "ais-xb").scrollTop, one(view, "ais-values").scrollTop];
  scrollAll();
  view.load();
  assert.deepEqual(kept(), [0, 90, 0]);
  scrollAll();
  typedPill(view, 0).fire("click");
  view.pullSettings({ on: false, axes: [{ axis: "sampler", text: "sampler_9" }, { axis: "steps", text: "5" }] });
  assert.deepEqual(chipTexts(view, 0), ["sampler_9"], "직접 입력 ends with the old content");
  assert.deepEqual(kept(), [0, 90, 0]);
  view.dispose();
});

test("keepScroll puts back the offsets by key, starts an unknown key at 0 and keeps only the keys asked for", () => {
  installFakeDom();
  const root = new FakeElement("div");
  const box = (key) => {
    const element = new FakeElement("div");
    element.setAttribute("data-scroll", key);
    return element;
  };
  const scrolled = () => {
    const [a, b] = [box("a"), box("b")];
    a.scrollTop = 70;
    b.scrollLeft = 30;
    root.replaceChildren(a, b);
  };
  let drawn;
  const draw = () => {
    drawn = [box("a"), box("b"), box("c")];
    root.replaceChildren(...drawn);
  };
  scrolled();
  keepScroll(root, draw);
  assert.deepEqual(drawn.map((element) => [element.scrollTop, element.scrollLeft]), [[70, 0], [0, 30], [0, 0]]);
  scrolled();
  keepScroll(root, draw, { only: ["b"] });
  assert.deepEqual(drawn.map((element) => [element.scrollTop, element.scrollLeft]), [[0, 0], [0, 30], [0, 0]]);
});
