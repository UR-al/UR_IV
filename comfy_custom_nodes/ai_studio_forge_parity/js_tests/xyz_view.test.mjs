// node:test for the XYZ panel's pure helpers (web/ai_studio/xyz_view.js); the DOM part is checked in a browser (Task 14).
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { DEFAULT_XYZ, XYZ_AXES, XyzView, csvField, estimateText, toggleLine, xyzWithDefaults } from "../web/ai_studio/xyz_view.js";

/** `[{id, label, kind, choices}]` read from the `_axis(...)` calls of xyz_axes.py. */
function pythonAxes() {
  const source = readFileSync(new URL("../xyz_axes.py", import.meta.url), "utf-8");
  return [...source.matchAll(/^\s*_axis\((.*)\),\s*$/gm)].map(([, body]) => {
    const keyword = /choices="([^"]+)"/.exec(body);
    const positional = [...body.replace(/\w+="[^"]*"/g, "").matchAll(/"([^"]*)"/g)].map((match) => match[1]);
    return { id: positional[0], label: positional[1], kind: positional[2], choices: keyword ? keyword[1] : positional[5] ?? null };
  });
}

test("XYZ_AXES mirrors xyz_axes.py (ids, Forge labels, kinds, choices, order)", () => {
  const python = pythonAxes();
  assert.ok(python.length >= 20);
  const forge = XYZ_AXES.filter((axis) => !axis.group).map(({ id, label, kind, choices }) => ({ id, label, kind, choices }));
  assert.deepEqual(forge, python);
});

test("xyzWithDefaults fills an old or broken saved setting", () => {
  assert.deepEqual(xyzWithDefaults(null), { ...structuredClone(DEFAULT_XYZ) });
  const fixed = xyzWithDefaults({
    on: true,
    axes: [{ axis: "prompt_sr", text: "<lora:a:1>\n<lora:b:1>" }, { axis: "no_such_axis", text: 5 }],
    options: { draw_legend: false, margin: 9999, row_count: 2.5, include_images: "yes" },
  });
  assert.equal(fixed.on, true);
  assert.deepEqual(fixed.axes, [
    { axis: "prompt_sr", text: "<lora:a:1>\n<lora:b:1>" },
    { axis: "nothing", text: "" },
    { axis: "nothing", text: "" },
  ]);
  assert.equal(fixed.options.draw_legend, false);
  assert.equal(fixed.options.margin, 500);
  assert.equal(fixed.options.row_count, 0);
  assert.equal(fixed.options.include_images, false);
});

test("estimateText multiplies the used axes' value counts", () => {
  const axes = [{ axis: "prompt_sr" }, { axis: "steps" }, { axis: "nothing" }];
  assert.equal(estimateText(axes, { x: ["a", "b", "c", "d"], y: ["1", "2", "3"], z: [""] }), "4 × 3 = 12장");
  assert.equal(estimateText([{ axis: "seed" }, { axis: "nothing" }, { axis: "nothing" }], { x: ["1", "2"] }), "2장");
  assert.equal(estimateText([{ axis: "nothing" }, { axis: "nothing" }, { axis: "nothing" }], {}), "1장");
});

test("toggleLine adds or removes one value line, CSV-quoted when needed", () => {
  assert.equal(csvField("plain.safetensors"), "plain.safetensors");
  assert.equal(csvField("a, b"), '"a, b"');
  assert.equal(csvField('say "hi"'), '"say ""hi"""');
  assert.equal(toggleLine("", "a.safetensors"), "a.safetensors");
  assert.equal(toggleLine("a.safetensors\nb.safetensors", "a.safetensors"), "b.safetensors");
  assert.equal(toggleLine("x\n", "a, b"), 'x\n"a, b"');
  assert.equal(toggleLine('x\n"a, b"', "a, b"), "x");
});

// The estimate logic without the DOM: a view object whose element was never drawn (renderEstimate is then a no-op).
function bareView(settings, answer) {
  const view = Object.create(XyzView.prototype);
  view.settings = xyzWithDefaults(settings);
  view.estimate = "";
  view.error = "";
  view.requests = [];
  view.ctx = { getState: () => ({}), plan: async (body) => { view.requests.push(body); return answer; } };
  return view;
}

test("the planner is not asked, and shows no error, while XYZ is off (default Seed axis is empty)", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const answer = { cells: [], labels: {}, errors: ["Seed: no values"] };
  const off = bareView({}, answer);
  off.error = "old error";
  off.refreshEstimate();
  t.mock.timers.tick(1000);
  await Promise.resolve();
  assert.deepEqual(off.requests, []);
  assert.equal(off.error, "");
  assert.equal(DEFAULT_XYZ.axes[0].axis, "seed"); // Forge's default X axis stays

  const on = bareView({ on: true }, answer);
  on.refreshEstimate();
  t.mock.timers.tick(1000);
  for (let i = 0; i < 3; i++) await Promise.resolve();
  assert.equal(on.requests.length, 1);
  assert.equal(on.error, "Seed: no values");

  const late = bareView({ on: true }, answer); // switched off while its request ran
  late.refreshEstimate();
  t.mock.timers.tick(1000);
  late.settings.on = false;
  for (let i = 0; i < 3; i++) await Promise.resolve();
  assert.equal(late.error, "");
});
