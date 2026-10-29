// Anima studio: the X/Y/Z plot panel (C4 right column) drawn into AIStudioXYZView's DOM widget. Its settings live as
// JSON in the node's hidden `xyz` widget. The grid itself is shown in the result column's "XYZ 격자" tab.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { h, numberBox, optionItems, pill, select } from "./studio_dom.js";

// Mirror of xyz_axes.py AXES (id, Forge label, value kind, /choices kind); js_tests/xyz_view.test.mjs pins it.
export const XYZ_AXES = Object.freeze([
  ["nothing", "Nothing", "nothing", null],
  ["seed", "Seed", "int", null],
  ["steps", "Steps", "int", null],
  ["hires_steps", "Hires steps", "int", null],
  ["size", "Size", "size", null],
  ["cfg", "CFG Scale", "float", null],
  ["shift", "Shift", "float", null],
  ["prompt_sr", "Prompt S/R", "str", null],
  ["prompt_order", "Prompt order", "permutations", null],
  ["sampler", "Sampler", "choice", "samplers"],
  ["hires_sampler", "Hires sampler", "choice", "samplers"],
  ["scheduler", "Schedule type", "choice", "schedulers"],
  ["checkpoint", "Checkpoint name", "choice", "unets"],
  ["vae", "VAE", "choice", "vaes"],
  ["denoising", "Denoising", "float", null],
  ["hires_upscaler", "Hires upscaler", "choice", "upscalers"],
  ["text_encoder", "[Modules] Text Encoder", "choice", "tes"],
  ["dora_mode", "[DoRA] Inference mode", "choice", "dora_modes"],
  ["sam3_enable", "[SAM3] Enable", "bool", null],
  ["sam3_checkpoint", "[SAM3] Checkpoint", "choice", "sam3"],
  ["sam3_prompt", "[SAM3] Detect Prompt", "str", null],
  ["sam3_denoise", "[SAM3] Denoising Strength", "float", null],
  ["sam3_steps", "[SAM3] Steps", "int", null],
  ["sam3_cfg", "[SAM3] CFG Scale", "float", null],
].map(([id, label, kind, choices]) => Object.freeze({ id, label, kind, choices })));

const AXIS_BY_ID = new Map(XYZ_AXES.map((axis) => [axis.id, axis]));
const LETTERS = ["X", "Y", "Z"];
const BOOL_ITEMS = ["True", "False"];
const PLACEHOLDERS = {
  int: "1-5 (+2), 10",
  float: "0.5, 1-2 [3]",
  size: "832x1216, 1216x832",
  str: "첫 값 = 찾을 글\n바꿀 글",
  permutations: "a, b, c",
  choice: "한 줄에 하나 (고르기)",
  bool: "True, False",
  nothing: "-",
};
const ESTIMATE_DELAY_MS = 400;

export const DEFAULT_XYZ = Object.freeze({
  on: false,
  axes: [{ axis: "seed", text: "" }, { axis: "nothing", text: "" }, { axis: "nothing", text: "" }],
  options: {
    draw_legend: true,
    keep_minus1: false,
    vary_x: false,
    vary_y: false,
    vary_z: false,
    include_images: false,
    include_sub_grids: false,
    margin: 0,
    row_count: 0,
  },
});

const clampInt = (value, low, high, fallback) => (Number.isInteger(value) ? Math.max(low, Math.min(high, value)) : fallback);

/** A complete XYZ settings object from a saved (maybe older or broken) one. */
export function xyzWithDefaults(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  const axes = DEFAULT_XYZ.axes.map((fallback, index) => {
    const entry = Array.isArray(source.axes) ? source.axes[index] : null;
    const axis = AXIS_BY_ID.has(entry?.axis) ? entry.axis : fallback.axis;
    return { axis, text: typeof entry?.text === "string" ? entry.text : "" };
  });
  const options = { ...DEFAULT_XYZ.options };
  const given = source.options && typeof source.options === "object" ? source.options : {};
  for (const key of Object.keys(options)) {
    if (typeof options[key] === "boolean" && typeof given[key] === "boolean") options[key] = given[key];
  }
  options.margin = clampInt(given.margin, 0, 500, 0);
  options.row_count = clampInt(given.row_count, 0, 8, 0);
  return { on: source.on === true, axes, options };
}

/** "4 × 3 = 12장" from a plan's labels (Nothing axes left out). */
export function estimateText(axes, labels) {
  const counts = axes
    .map((entry, index) => (entry.axis === "nothing" ? null : (labels?.[["x", "y", "z"][index]] || []).length))
    .filter((count) => count !== null);
  const total = counts.reduce((product, count) => product * count, 1);
  return counts.length > 1 ? `${counts.join(" × ")} = ${total}장` : `${total}장`;
}

/** A value as one line of the CSV value box (quoted when it has a comma, quote or outer spaces). */
export function csvField(value) {
  return /[",]/.test(value) || value !== value.trim() ? `"${value.replace(/"/g, '""')}"` : value;
}

const unquote = (line) => {
  const text = line.trim();
  return /^".*"$/.test(text) ? text.slice(1, -1).replace(/""/g, '"') : text;
};

/** The value box with `value`'s line removed if present, else added at the end. */
export function toggleLine(text, value) {
  const lines = text.split("\n").filter((line) => line.trim() !== "");
  const kept = lines.filter((line) => unquote(line) !== value);
  if (kept.length === lines.length) kept.push(csvField(value));
  return kept.join("\n");
}

export class XyzView {
  /** `ctx`: `choices(kind)` -> Promise<{items}>, `plan(body)` -> Promise<plan JSON>, `getState()`, `changed()`. */
  constructor(widget, ctx) {
    this.widget = widget;
    this.ctx = ctx;
    this.choices = {};
    this.picker = -1;
    this.estimate = "";
    this.error = "";
    this.settings = this.read();
    this.element = h("div", { class: "ais-root ais-xyz" });
    this.render();
    this.refreshEstimate();
  }

  read() {
    try {
      return xyzWithDefaults(JSON.parse(this.widget?.value || "{}"));
    } catch {
      return xyzWithDefaults({});
    }
  }

  load() {
    this.settings = this.read();
    this.render();
    this.refreshEstimate();
  }

  getXyz() {
    return structuredClone(this.settings);
  }

  set(mutate, { rerender = false, estimate = true } = {}) {
    mutate(this.settings);
    if (this.widget) this.widget.value = JSON.stringify(this.settings);
    this.ctx.changed?.();
    if (rerender) this.render();
    if (estimate) this.refreshEstimate();
  }

  onStudioEvent(type) {
    if (type === "state") this.refreshEstimate();
  }

  dispose() {
    clearTimeout(this.estimateTimer);
  }

  /**
   * Asks /xyz/plan (debounced) for the cell count or the planner's error (Forge-worded). Not while XYZ is off: the
   * default X axis (Seed, empty, as in Forge) would show "Seed: no values" in a panel the user has not touched.
   */
  refreshEstimate() {
    clearTimeout(this.estimateTimer);
    if (!this.settings.on) {
      this.estimate = "";
      this.error = "";
      this.renderEstimate();
      return;
    }
    this.estimateTimer = setTimeout(async () => {
      const { axes, options } = this.settings;
      let error = "";
      let estimate = "";
      try {
        const plan = await this.ctx.plan({ state: this.ctx.getState(), axes, options });
        error = plan?.errors?.length ? plan.errors.join("\n") : "";
        estimate = error ? "" : estimateText(axes, plan?.labels);
      } catch (failure) {
        error = String(failure?.message || failure);
      }
      if (!this.settings.on) return; // switched off while the request ran
      this.error = error;
      this.estimate = estimate;
      this.renderEstimate();
    }, ESTIMATE_DELAY_MS);
  }

  async openPicker(index) {
    this.picker = this.picker === index ? -1 : index;
    const kind = AXIS_BY_ID.get(this.settings.axes[index].axis).choices;
    if (this.picker >= 0 && kind) this.choices[kind] = (await this.ctx.choices(kind))?.items || [];
    this.render();
  }

  // ------------------------------------------------------------------------------------------- drawing

  render() {
    const { settings } = this;
    const option = (key, text) =>
      h("div", { class: "ais-row" }, h("span", { class: "ais-label" }, text),
        pill(settings.options[key], () => this.set((s) => { s.options[key] = !s.options[key]; }, { rerender: true, estimate: key === "keep_minus1" })));
    const vary = h("div", { class: "ais-row" }, h("span", { class: "ais-label" }, "시드 바꾸기"),
      h("span", { class: "ais-ctl" }, ["x", "y", "z"].map((letter) =>
        pill(settings.options[`vary_${letter}`], () => this.set((s) => { s.options[`vary_${letter}`] = !s.options[`vary_${letter}`]; }, { rerender: true, estimate: false }),
          { on: letter.toUpperCase(), off: letter.toUpperCase(), title: `${letter.toUpperCase()} 축을 따라 다른 시드` }))));
    const number = (key, text, min, max, step) =>
      h("div", { class: "ais-row" }, h("span", { class: "ais-label" }, text),
        numberBox(settings.options[key], (value) => this.set((s) => { s.options[key] = Math.round(Math.max(min, Math.min(max, value))); }, { estimate: false }), { min, max, step, width: 60 }));
    const swap = (a, b) =>
      h("button", { type: "button", class: "ais-mini ais-swap", "aria-label": `${LETTERS[a]} ${LETTERS[b]} 축 바꾸기`,
        onClick: () => this.set((s) => { [s.axes[a], s.axes[b]] = [s.axes[b], s.axes[a]]; }, { rerender: true }) }, `${LETTERS[a]}↔${LETTERS[b]}`);
    this.estimateEl = h("span", { class: "ais-f ais-estimate" });
    this.errorEl = h("div", { class: "ais-xyz-error", role: "status" });
    this.element.replaceChildren(
      h("div", { class: "ais-xh" }, h("span", {}, "XYZ 비교"),
        pill(settings.on, () => this.set((s) => { s.on = !s.on; }, { rerender: true }), { title: "켜면 생성 버튼이 XYZ로 돈다" })),
      h("div", { class: "ais-xb" },
        settings.axes.map((_, index) => this.axisBlock(index)),
        h("div", { class: "ais-row ais-swaps" }, h("span", { class: "ais-label" }, "축 바꾸기"), h("span", { class: "ais-ctl" }, swap(0, 1), swap(1, 2), swap(0, 2))),
        h("div", { class: "ais-xopts" },
          option("draw_legend", "범례 그리기"),
          option("keep_minus1", "-1 시드 유지"),
          vary,
          option("include_images", "개별 이미지 포함"),
          option("include_sub_grids", "하위 격자 포함"),
          number("margin", "여백 (px)", 0, 500, 2),
          number("row_count", "Row Count (0 = 자동)", 0, 8, 1),
          h("div", { class: "ais-row" }, h("span", { class: "ais-label" }, "예상 장수"), this.estimateEl),
          this.errorEl)),
    );
    this.renderEstimate();
  }

  renderEstimate() {
    if (!this.estimateEl) return;
    this.estimateEl.textContent = this.estimate || "-";
    this.errorEl.hidden = !this.error;
    this.errorEl.textContent = this.error;
  }

  axisBlock(index) {
    const entry = this.settings.axes[index];
    const spec = AXIS_BY_ID.get(entry.axis);
    const pickable = Boolean(spec.choices) || spec.kind === "bool";
    const types = XYZ_AXES.map((axis) => ({ value: axis.id, label: axis.label }));
    const box = h("textarea", {
      class: `ais-f ais-values ${index === 0 ? "tall" : ""}`,
      value: entry.text,
      placeholder: PLACEHOLDERS[spec.kind],
      disabled: spec.kind === "nothing",
      "aria-label": `${LETTERS[index]} 값`,
      onInput: (event) => this.set((s) => { s.axes[index].text = event.target.value; }),
    });
    return h("div", { class: "ais-ax" },
      h("div", { class: "ais-axh" }, h("span", {}, LETTERS[index]),
        select(types, entry.axis, (value) => { this.picker = -1; this.set((s) => { s.axes[index].axis = value; }, { rerender: true }); },
          { class: "ais-f ais-grow", "aria-label": `${LETTERS[index]} 종류` }),
        pickable ? h("button", { type: "button", class: `ais-pill ${this.picker === index ? "on" : "off"}`, onClick: () => this.openPicker(index) }, "고르기") : null),
      box,
      this.picker === index ? this.pickerList(index, spec) : null);
  }

  pickerList(index, spec) {
    const text = this.settings.axes[index].text;
    const items = optionItems(spec.kind === "bool" ? BOOL_ITEMS : this.choices[spec.choices] || []);
    const picked = new Set(text.split("\n").map(unquote));
    const toggle = (value) => this.set((s) => { s.axes[index].text = toggleLine(s.axes[index].text, value); }, { rerender: true });
    const all = () => this.set((s) => { s.axes[index].text = items.map((item) => csvField(item.value)).join("\n"); }, { rerender: true });
    return h("div", { class: "ais-picker" },
      h("button", { type: "button", class: "ais-mini", onClick: all }, "전부"),
      items.length ? items.map((item) =>
        h("label", { class: "ais-pick", title: item.value },
          h("input", { type: "checkbox", checked: picked.has(item.value), onChange: () => toggle(item.value) }), item.label))
        : h("span", { class: "ais-label" }, "목록 없음"));
  }
}
