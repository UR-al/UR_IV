// Anima studio: the X/Y/Z plot panel (C4 right column) drawn into AIStudioXYZView's DOM widget. Its settings live as
// JSON in the node's hidden `xyz` widget. The grid itself is shown in the result column's "XYZ 격자" tab.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { guardCanvasKeys, h, keepScroll, numberBox, optionItems, pill, select } from "./studio_dom.js";
import { extAxes } from "./studio_ext.js";
import { CROSS_ATTN_MODES } from "./studio_logic.js";
import { BLOCK_AXES, addBlockStyles, blockField } from "./xyz_blocks.js";
import { addChipStyles, chipField, joinValues, knownValue, splitValues } from "./xyz_chips.js";
import { baseItems, columnList, fileItems, loraAxisBody, loraWithDefaults, newLoraBlock, refaceFiles, refillFiles } from "./xyz_lora_view.js";

// Mirror of xyz_axes.py AXES (id, Forge label, value kind, /choices kind); js_tests/xyz_view.test.mjs pins it. The extension
// table's axes follow (the golden `ext_axes` pins them).
const FORGE_AXES = Object.freeze([
  ["nothing", "Nothing", "nothing", null],
  ["seed", "Seed", "int", null],
  ["steps", "Steps", "int", null],
  ["hires_steps", "Hires steps", "int", null],
  ["size", "Size", "size", null],
  ["cfg", "CFG Scale", "float", null],
  ["shift", "Shift", "float", null],
  ["prompt_sr", "Prompt S/R", "str", null],
  ["prompt_order", "Prompt order", "permutations", null],
  ["lora", "LoRA", "lora", null],
  ["sampler", "Sampler", "choice", "samplers"],
  ["hires_sampler", "Hires sampler", "choice", "samplers"],
  ["scheduler", "Schedule type", "choice", "schedulers"],
  ["checkpoint", "Checkpoint name", "choice", "unets"],
  ["vae", "VAE", "choice", "vaes"],
  ["denoising", "Denoising", "float", null],
  ["hires_upscaler", "Hires upscaler", "choice", "upscalers"],
  ["text_encoder", "[Modules] Text Encoder", "choice", "tes"],
  ["dora_mode", "[DoRA] Inference mode", "choice", "dora_modes"],
  ["attention", "[Attention] Backend", "choice", "attentions"],
  ["sam3_enable", "[SAM3] Enable", "bool", null],
  ["sam3_checkpoint", "[SAM3] Checkpoint", "choice", "sam3"],
  ["sam3_prompt", "[SAM3] Detect Prompt", "str", null],
  ["sam3_denoise", "[SAM3] Denoising Strength", "float", null],
  ["sam3_steps", "[SAM3] Steps", "int", null],
  ["sam3_cfg", "[SAM3] CFG Scale", "float", null],
  // not Forge's (pack 1.17.0): the LoRA card's adherence settings
  ["lora_base", "[LoRA] Base 먼저 %", "int", null],
  ["lora_cross_attn", "[LoRA] 교차 어텐션 제외", "choice", null, CROSS_ATTN_MODES],
].map(([id, label, kind, choices, values = []]) => Object.freeze({ id, label, kind, choices, values: [...values], group: "" })));

export const XYZ_AXES = Object.freeze([...FORGE_AXES, ...extAxes().map(Object.freeze)]);

const AXIS_BY_ID = new Map(XYZ_AXES.map((axis) => [axis.id, axis]));
const LETTERS = ["X", "Y", "Z"];
const BOOL_ITEMS = ["True", "False"];
const PLACEHOLDERS = {
  int: "1-5 (+2), 10",
  float: "0.5, 1-2 [3]",
  size: "832x1216, 1216x832",
  str: '한 줄에 한 값 (쉼표로도 나눔)\n쉼표가 든 값은 "a, b"',
  lines: "한 줄에 한 값 (쉼표는 값의 일부)",
  permutations: "a, b, c",
  choice: "한 줄에 하나 (쉼표로도 나눔)",
  bool: "True, False",
  nothing: "-",
};
// The text box of a block axis (직접 입력), by axis id: how its values are written by hand.
const TYPED_PLACEHOLDERS = {
  prompt_sr: '첫 값 = 찾을 글\n바꿀 글\n쉼표가 든 값은 "a, b"',
  prompt_order: 'a, b, c\n쉼표가 든 조각은 "a, b"',
};
const TYPED_TITLE = "블록 대신 글자로 고치기 (한 줄에 하나, 쉼표로도 나눔)";
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
    const block = axis === "lora" || entry?.lora ? { lora: loraWithDefaults(entry?.lora) } : {};
    return { axis, text: typeof entry?.text === "string" ? entry.text : "", ...block };
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

/** An axis whose values come from a list (a studio list, the axis' own values, a switch): its value box is drawn as blocks. */
const isListed = (spec) => Boolean(spec.choices) || spec.values.length > 0 || spec.kind === "bool";

export class XyzView {
  /** `ctx`: `choices(kind)` -> Promise<{items}>, `plan(body)` -> Promise<plan JSON>, `getState()`, `changed()`. */
  constructor(widget, ctx) {
    this.widget = widget;
    this.ctx = ctx;
    this.choices = {};
    this.typed = new Set(); // the list and text-block axes (by index) shown as their text box: 직접 입력
    this.fields = []; // the drawn block fields, by axis index
    this.estimate = "";
    this.error = "";
    this.loraColumns = {};
    this.settings = this.read();
    // Delete / Backspace on a focused switch (직접 입력 keeps the focus after a toggle) must not delete this node.
    this.element = guardCanvasKeys(h("div", { class: "ais-root ais-xyz" }));
    addChipStyles();
    addBlockStyles();
    this.render();
    this.refreshEstimate();
    this.loadLoraFiles();
    // The first draw has no size yet (the DOM widget is attached later): the text blocks take their height from their lines.
    this.observer = globalThis.ResizeObserver ? new ResizeObserver(() => this.growBlocks()) : null;
    this.observer?.observe(this.element);
  }

  /** The text blocks as high as their lines (xyz_blocks.js): after a draw, and when the panel gets its size. */
  growBlocks() {
    for (const field of this.fields) field?.grow?.();
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
    this.typed = new Set();
    this.render({ fresh: true });
    this.refreshEstimate();
    this.loadLoraFiles();
  }

  getXyz() {
    return structuredClone(this.settings);
  }

  /** The EXIF panel's 설정 가져오기: the plot as the image's run had it. */
  pullSettings(saved) {
    this.typed = new Set();
    this.set((settings) => { Object.assign(settings, xyzWithDefaults(saved)); }, { rerender: true, fresh: true });
    this.loadLoraFiles();
  }

  /** `rerender`: draw the panel anew (an edit inside a box must not: its focus would go); `fresh`: the content is another one. */
  set(mutate, { rerender = false, estimate = true, fresh = false } = {}) {
    mutate(this.settings);
    if (this.widget) this.widget.value = JSON.stringify(this.settings);
    this.ctx.changed?.();
    if (rerender) this.render({ fresh });
    if (estimate) this.refreshEstimate();
  }

  onStudioEvent(type) {
    if (type !== "state") return;
    this.syncBaseLoras();
    this.refreshEstimate();
  }

  dispose() {
    clearTimeout(this.estimateTimer);
    this.observer?.disconnect();
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
      this.loraColumns = {};
      this.renderEstimate();
      return;
    }
    this.estimateTimer = setTimeout(async () => {
      this.estimateRequest = (this.estimateRequest || 0) + 1;
      const request = this.estimateRequest;
      const { axes, options } = this.settings;
      let error = "";
      let estimate = "";
      let loraColumns = {};
      try {
        const plan = await this.ctx.plan({ state: this.ctx.getState(), axes, options });
        error = plan?.errors?.length ? plan.errors.join("\n") : "";
        estimate = error ? "" : estimateText(axes, plan?.labels);
        loraColumns = error ? {} : plan?.lora_preview || {};
      } catch (failure) {
        error = String(failure?.message || failure);
      }
      if (!this.settings.on || request !== this.estimateRequest) return; // switched off, or a newer request ran, meanwhile
      this.error = error;
      this.estimate = estimate;
      this.loraColumns = loraColumns;
      this.renderEstimate();
    }, ESTIMATE_DELAY_MS);
  }

  hasLoraAxis() {
    return this.settings.axes.some((entry) => entry.axis === "lora");
  }

  /** A LoRA axis' drop-downs list the LoRA files; `fresh` skips the cache (a new epoch may have arrived). */
  async loadLoraFiles(fresh = false) {
    if (!this.hasLoraAxis()) return;
    const items = (await this.ctx.choices("loras", fresh))?.items || [];
    if (JSON.stringify(items) !== JSON.stringify(this.choices.loras)) {
      this.choices.loras = items;
      refillFiles(this.element, fileItems(items));
      refaceFiles(this.element, items);
    }
    this.syncBaseLoras();
  }

  /** The base rows list the panel's LoRAs, which change while this view stays drawn. */
  syncBaseLoras() {
    if (this.hasLoraAxis() && JSON.stringify(baseItems(this.ctx.getState())) !== JSON.stringify(this.baseLoras)) this.render();
  }

  /** What a list axis can take: a switch's words, the axis' own values, or the studio's list (`loaded`: it has been read). */
  listOf(spec) {
    const items = spec.kind === "bool" ? BOOL_ITEMS : spec.values.length ? spec.values : this.choices[spec.choices];
    return { items: optionItems(items || []), loaded: Boolean(items) };
  }

  /** Reads a studio list for the block fields (the route caches it for a while); a changed list redraws them in place. */
  async loadChoices(kind) {
    const items = (await this.ctx.choices(kind))?.items || [];
    if (JSON.stringify(items) === JSON.stringify(this.choices[kind])) return;
    this.choices[kind] = items;
    this.settings.axes.forEach((entry, index) => {
      if (AXIS_BY_ID.get(entry.axis).choices === kind) this.fields[index]?.refresh();
    });
  }

  pickAxis(index, value) {
    this.typed.delete(index); // its own 직접 입력 only: the other axes keep their boxes, and the column its length
    this.set((s) => {
      s.axes[index].axis = value;
      if (value === "lora") s.axes[index].lora ??= newLoraBlock(baseItems(this.ctx.getState())[0]?.value);
    }, { rerender: true });
    this.loadLoraFiles(true);
  }

  swapAxes(a, b) {
    const typed = [a, b].map((index) => this.typed.has(index)); // 직접 입력 goes with its axis
    for (const [index, on] of [[a, typed[1]], [b, typed[0]]]) {
      if (on) this.typed.add(index);
      else this.typed.delete(index);
    }
    this.set((s) => { [s.axes[a], s.axes[b]] = [s.axes[b], s.axes[a]]; }, { rerender: true });
  }

  /** 직접 입력: a list axis' values as blocks or as their text; the focus stays on the switch. */
  toggleTyped(index) {
    if (!this.typed.delete(index)) this.typed.add(index);
    this.render();
    this.element.querySelector(`[data-k="typed-${index}"]`)?.focus({ preventScroll: true });
  }

  // ------------------------------------------------------------------------------------------- drawing

  /**
   * Draws the panel anew; its scroll boxes (the column, each value box) keep their place. `fresh`: the content was replaced
   * (a loaded workflow, 설정 가져오기), so only the column keeps its place.
   */
  render({ fresh = false } = {}) {
    keepScroll(this.element, () => this.draw(), fresh ? { only: ["body"] } : undefined);
    if (globalThis.requestAnimationFrame) requestAnimationFrame(() => this.growBlocks());
  }

  draw() {
    const { settings } = this;
    this.fields = [];
    this.baseLoras = baseItems(this.ctx.getState());
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
        onClick: () => this.swapAxes(a, b) }, `${LETTERS[a]}↔${LETTERS[b]}`);
    this.estimateEl = h("span", { class: "ais-f ais-estimate" });
    this.columnsEl = h("div", { class: "ais-lx-lists" });
    this.errorEl = h("div", { class: "ais-xyz-error", role: "status" });
    const body = h("div", { class: "ais-xb", "data-scroll": "body" },
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
        this.columnsEl,
        this.errorEl));
    this.element.replaceChildren(
      h("div", { class: "ais-xh" }, h("span", {}, "XYZ 비교"),
        pill(settings.on, () => this.set((s) => { s.on = !s.on; }, { rerender: true }), { title: "켜면 생성 버튼이 XYZ로 돈다" })),
      body,
    );
    this.renderEstimate();
  }

  renderEstimate() {
    if (!this.estimateEl) return;
    this.estimateEl.textContent = this.estimate || "-";
    this.columnsEl.replaceChildren(...LETTERS.flatMap((letter) => {
      const columns = this.loraColumns[letter.toLowerCase()];
      return columns ? [columnList(letter, columns)] : [];
    }));
    this.errorEl.hidden = !this.error;
    this.errorEl.textContent = this.error;
  }

  loraBody(index) {
    return loraAxisBody({
      block: this.settings.axes[index].lora,
      baseItems: this.baseLoras,
      loras: this.choices.loras || [],
      loraBrowser: this.ctx.loraBrowser ? (opts) => this.ctx.loraBrowser(opts) : undefined,
      change: (mutate, rerender = false) => this.set((s) => mutate(s.axes[index].lora), { rerender }),
      refreshFiles: () => this.loadLoraFiles(true),
    });
  }

  axisBlock(index) {
    const entry = this.settings.axes[index];
    const spec = AXIS_BY_ID.get(entry.axis);
    const listed = isListed(spec);
    const blocks = BLOCK_AXES[spec.id]; // free text values drawn as text blocks (Prompt S/R, Prompt order)
    const typed = Boolean(listed || blocks) && this.typed.has(index);
    const types = XYZ_AXES.map((axis) => ({ value: axis.id, label: axis.label, group: axis.group }));
    const typedPill = listed || blocks
      ? pill(typed, () => this.toggleTyped(index), { on: "직접 입력", off: "직접 입력", title: blocks?.typed || TYPED_TITLE })
      : null;
    typedPill?.setAttribute("data-k", `typed-${index}`);
    return h("div", { class: "ais-ax" },
      h("div", { class: "ais-axh" }, h("span", {}, LETTERS[index]),
        select(types, entry.axis, (value) => this.pickAxis(index, value), { class: "ais-f ais-grow", "aria-label": `${LETTERS[index]} 종류` }),
        typedPill),
      spec.kind === "lora" ? this.loraBody(index)
        : listed && !typed ? this.chipBody(index, spec)
          : blocks && !typed ? this.blockBody(index, spec, blocks)
            : this.textBody(index, spec));
  }

  /** The value box as text: an axis that has no list, and a list axis in 직접 입력. The key has the axis: other values start at the top. */
  textBody(index, spec) {
    return h("textarea", {
      class: `ais-f ais-values ${index === 0 ? "tall" : ""}`,
      value: this.settings.axes[index].text,
      placeholder: TYPED_PLACEHOLDERS[spec.id] || PLACEHOLDERS[spec.kind],
      disabled: spec.kind === "nothing",
      "aria-label": `${LETTERS[index]} 값`,
      "data-scroll": `values-${index}-${spec.id}`,
      onInput: (event) => this.set((s) => { s.axes[index].text = event.target.value; }),
    });
  }

  /**
   * The value box of a free-text axis as text blocks (xyz_blocks.js): one editable block a value, so a value may hold commas.
   * As with the list blocks, the axis' value text stays what the planner reads: typing writes it back (one value a line, CSV
   * quoted) with no redraw at all, and adding, taking out or moving a block redraws only the field itself.
   */
  blockBody(index, spec, texts) {
    const field = blockField({
      name: LETTERS[index],
      tall: index === 0,
      scrollKey: `blocks-${index}-${spec.id}`,
      texts,
      read: () => splitValues(this.settings.axes[index].text),
      write: (values) => this.set((s) => { s.axes[index].text = joinValues(values); }),
    });
    this.fields[index] = field;
    return field.el;
  }

  /**
   * The value box of a list axis as blocks (xyz_chips.js). The blocks are a view of the axis' value text, which stays what
   * the planner reads: an edit writes it back one value a line, and redraws only the field itself.
   */
  chipBody(index, spec) {
    const readList = () => { if (spec.choices) this.loadChoices(spec.choices); };
    const field = chipField({
      name: LETTERS[index],
      tall: index === 0,
      scrollKey: `chips-${index}-${spec.id}`,
      read: () => splitValues(this.settings.axes[index].text),
      write: (values) => this.set((s) => { s.axes[index].text = joinValues(values); }),
      list: () => this.listOf(spec),
      known: (value) => {
        const { items } = this.listOf(spec);
        return knownValue(spec, value, items.length ? items : null); // an empty list (not read, or the route failed) marks nothing
      },
      opened: readList, // a new file may have arrived since the list was read
    });
    this.fields[index] = field;
    readList();
    return field.el;
  }
}
