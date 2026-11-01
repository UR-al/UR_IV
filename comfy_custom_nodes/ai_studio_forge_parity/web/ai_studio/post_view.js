// Anima studio: the panel's 후처리 and Hires cards, drawn from the field table (post_schema.js).
// Hires shows all its fields; upscale and SAM3 unfold with a caret, in the shape of the extension rows.
// The gallery buttons use these settings too, so an unfolded section shows them with its switch off.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { fieldControl, fieldsBody, flip } from "./ext_view.js";
import { POST_SCHEMA } from "./post_schema.js";
import { autoGrow, card, h, numberBox, optionItems, pill, row, select } from "./studio_dom.js";
import { generationSize, modelFamily, sizeStep } from "./studio_logic.js";
import { hiresSize, scaleFromName } from "./studio_post.js";

const ITEMS = Object.fromEntries(Object.keys(POST_SCHEMA).map((key) => [key, { key, stateKey: key, fields: POST_SCHEMA[key] }]));
const SCALE_MAX = 8;
// sam-extra's Detect Prompt grammar (mask_ops.split_prompt_groups): `;`, `|` and a new line also merge.
const SAM3_PROMPT_NOTE = "쉼표(,)는 한 마스크로 합치고 슬래시(/)는 따로 인페인트. 예: eyes, hair / hand";

const fieldOf = (section, key) => ITEMS[section].fields.find((entry) => entry.key === key);

const control = (section, key, s, view) => fieldControl(ITEMS[section], fieldOf(section, key), s[section], view);

/** One row: each field's label before its control. */
const fieldRow = (section, keys, s, view) => {
  const [first, ...rest] = keys.flatMap((key) => [fieldOf(section, key).label, control(section, key, s, view)]);
  return row(first, ...rest);
};

function caret(key, label, view) {
  const open = view.memory.open.has(key);
  return h("button", { type: "button", class: "ais-caret", "aria-label": `${label} 세부 설정`, "aria-expanded": open ? "true" : "false", title: "세부 설정", onClick: () => flip(view, key) },
    open ? "▴" : "▾");
}

function upscaleRows(s, view) {
  const pickModel = (value) => view.set((st) => {
    st.upscale.model = value;
    const scale = scaleFromName(value); // Forge's "Automatically set the Scale by factor"
    if (scale >= 1 && scale <= SCALE_MAX) st.upscale.scale = scale;
  }, true);
  const model = select(optionItems(view.lists("upscalers"), s.upscale.model, { empty: "[업스케일 모델]" }), s.upscale.model, pickModel);
  const scale = s.upscale.mode === "Scale to"
    ? null
    : row("배율", numberBox(s.upscale.scale, (value) => view.set((st) => { st.upscale.scale = value; }), { min: 1, max: SCALE_MAX, step: 0.05, width: 50 }));
  return [model, scale];
}

function upscaleBody(s, view) {
  const swap = h("button", { type: "button", class: "ais-mini", title: "폭 높이 바꾸기", "aria-label": "폭 높이 바꾸기",
    onClick: () => view.set((st) => { [st.upscale.width, st.upscale.height] = [st.upscale.height, st.upscale.width]; }, true) }, "⇅");
  const size = s.upscale.mode === "Scale to"
    ? [row("크기", control("upscale", "width", s, view), "×", control("upscale", "height", s, view), swap), fieldRow("upscale", ["crop"], s, view)]
    : fieldRow("upscale", ["max_side"], s, view);
  return h("div", { class: "ais-ext-body" },
    fieldRow("upscale", ["mode"], s, view),
    size,
    fieldRow("upscale", ["model_2"], s, view),
    fieldRow("upscale", ["visibility"], s, view),
    fieldRow("upscale", ["color_correction"], s, view),
    fieldRow("upscale", ["tile_size", "tile_overlap"], s, view));
}

function sam3Rows(s, view) {
  const write = (mutate) => view.set(mutate);
  return [
    select(optionItems(view.lists("sam3"), s.sam3.checkpoint, { empty: "[SAM3 모델]" }), s.sam3.checkpoint, (value) => write((st) => { st.sam3.checkpoint = value; })),
    row("찾을 것", h("textarea", { class: "ais-f ais-negative", value: s.sam3.prompt, onInput: (e) => { autoGrow(e.target); write((st) => { st.sam3.prompt = e.target.value || "face"; }); } })),
    h("div", { class: "ais-note" }, SAM3_PROMPT_NOTE),
    row("디노이즈", numberBox(s.sam3.denoise, (value) => write((st) => { st.sam3.denoise = value; }), { min: 0, max: 1, step: 0.05, width: 50 })),
  ];
}

/**
 * The 후처리 card. `view`: `set(mutate, rerender)`, `memory` (extMemory()), `rerender()`, `lists(kind)` (a `/choices` list;
 * `upscalers` without the latent methods), optional `attachText(box)` (prompt autocomplete).
 */
export function postCard(s, view) {
  const toggle = (section) => pill(s[section].on, () => view.set((st) => { st[section].on = !st[section].on; }, true));
  const upscaleOpen = view.memory.open.has("upscale");
  const sam3Open = view.memory.open.has("sam3");
  return card("후처리 (켜면 바로 적용)", null, [
    row("Hires fix", toggle("hires")),
    row("업스케일", toggle("upscale"), caret("upscale", "업스케일", view)),
    s.upscale.on || upscaleOpen ? upscaleRows(s, view) : null,
    upscaleOpen ? upscaleBody(s, view) : null,
    row("SAM3 디테일러", toggle("sam3"), caret("sam3", "SAM3", view)),
    s.sam3.on || sam3Open ? sam3Rows(s, view) : null,
    sam3Open ? fieldsBody(ITEMS.sam3, s.sam3, view) : null,
  ]);
}

/** The picture size a Hires run of `s` ends at, as the node computes it. */
export function hiresSizeText(s) {
  const { width, height } = generationSize(s);
  const size = hiresSize(width, height, s.hires, sizeStep(modelFamily(s.unet)));
  return `${size.width} × ${size.height}`;
}

/** The Hires card; `view` as in postCard, plus `lists("upscaler_methods")` and `keepSizeText(element)` (it shows hiresSizeText). */
export function hiresCard(s, view) {
  const { hires } = s;
  const num = (key, opts) => numberBox(hires[key], (value) => view.set((st) => { st.hires[key] = value; }), opts);
  const samplers = optionItems(view.lists("samplers"), hires.sampler, { empty: "메인과 같음" });
  const sizeText = h("span", { class: "ais-hires-size", title: "Hires 결과 크기" }, hiresSizeText(s));
  view.keepSizeText(sizeText);
  return card("Hires fix", null, [
    row("업스케일러", select(optionItems(view.lists("upscaler_methods"), hires.upscaler), hires.upscaler, (value) => view.set((st) => { st.hires.upscaler = value; }))),
    row("배율", num("scale", { min: 1, max: 8, step: 0.05, width: 46 }), "디노이즈", num("denoise", { min: 0, max: 1, step: 0.05, width: 46 }),
      "스텝", num("steps", { min: 0, max: 150, step: 1, width: 42, title: "0 = 메인과 같음" })),
    row("샘플러", select(samplers, hires.sampler, (value) => view.set((st) => { st.hires.sampler = value; }))),
    fieldRow("hires", ["scheduler"], s, view),
    fieldRow("hires", ["cfg", "shift"], s, view),
    fieldRow("hires", ["resize_width", "resize_height"], s, view),
    row("결과 크기", sizeText, "단위", control("hires", "size_step", s, view)),
    fieldRow("hires", ["tile_size", "tile_overlap"], s, view),
    fieldRow("hires", ["vae"], s, view),
    fieldRow("hires", ["prompt"], s, view),
    fieldRow("hires", ["negative"], s, view),
  ]);
}
