// Anima studio: the LoRA axis of the XYZ panel: which `<lora:...>` tag to swap for which LoRAs, with their triggers and
// epochs, and the list of columns that makes. xyz_lora.py reads the block this view edits and makes the columns.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { h, optionItems, optionTags, pill, select } from "./studio_dom.js";
import { fileStem, loraTagNames } from "./studio_logic.js";

const EPOCH_MODES = [["off", "없음"], ["direct", "직접 입력"], ["last", "last epoch"]];
const EPOCH_SUFFIX = /-(\d{6})(?:\.[^.]+)?$/;
const NO_PICK = "[LoRA]";
const MAX_STRENGTH = 10;

const emptyRow = (lora = "") => ({ lora, strength: null, trigger: "", epoch_mode: "off", epoch: null });

export const newLoraBlock = (lora = "") => ({ find: "", after: false, rows: [emptyRow(lora)] });

const finiteOrNull = (value) => (Number.isFinite(value) ? value : null);

function cleanRow(raw) {
  const row = raw && typeof raw === "object" ? raw : {};
  return {
    lora: typeof row.lora === "string" ? row.lora : "",
    strength: finiteOrNull(row.strength),
    trigger: typeof row.trigger === "string" ? row.trigger : "",
    epoch_mode: EPOCH_MODES.some(([mode]) => mode === row.epoch_mode) ? row.epoch_mode : "off",
    epoch: finiteOrNull(row.epoch),
  };
}

/** A complete block from a saved (maybe older or broken) one; it always has the base row. */
export function loraWithDefaults(raw) {
  const block = raw && typeof raw === "object" ? raw : {};
  const rows = Array.isArray(block.rows) ? block.rows.map(cleanRow) : [];
  return { find: typeof block.find === "string" ? block.find : "", after: block.after === true, rows: rows.length ? rows : [emptyRow()] };
}

/** What the rows after the base can pick: the `/choices?kind=loras` entries, named like the panel's LoRA tags, with the folder when two files share a name. */
export function fileItems(entries) {
  const list = entries || [];
  const key = (entry) => fileStem(entry).toLowerCase(); // a tag finds its file ignoring case
  const counts = new Map();
  for (const entry of list) counts.set(key(entry), (counts.get(key(entry)) || 0) + 1);
  return list.map((entry) => {
    const name = counts.get(key(entry)) > 1 ? entry.replace(/\\/g, "/").replace(/\.[^.]+$/, "") : fileStem(entry);
    return { value: name, label: name };
  });
}

/** A new file list for the drawn drop-downs; a redraw would take the focus from the box being typed in. */
export function refillFiles(root, items) {
  for (const box of root.querySelectorAll(".ais-lx-files")) {
    const { value } = box;
    box.replaceChildren(...optionTags(optionItems(items, value, { empty: NO_PICK })));
    box.value = value;
  }
}

/** What the base row can pick: the LoRAs a run of `state` loads. */
export const baseItems = (state) => loraTagNames(state).map((name) => ({ value: name, label: name }));

/** The epoch line's title: a narrow drop-down cuts off the number at the end of a file name. */
export function epochText(name) {
  const match = EPOCH_SUFFIX.exec(name);
  return match ? `에폭 ${Number(match[1])}` : "에폭";
}

const field = (label, control) => h("label", { class: "ais-lx-field" }, label, control);

const numberField = (value, onValue, props) =>
  h("input", { type: "number", class: "ais-f ais-num", value: value ?? "", ...props, onInput: (event) => onValue(event.target.value === "" ? null : Number(event.target.value)) });

const textField = (value, onValue, placeholder) =>
  h("input", { type: "text", class: "ais-f", value, placeholder, onInput: (event) => onValue(event.target.value) });

function placeLine(view) {
  const choice = (after, text) => pill(view.block.after === after, () => view.change((block) => { block.after = after; }, true), { on: text, off: text });
  return h("div", { class: "ais-lx-line" }, h("span", { class: "ais-lx-tag" }, "새 트리거는"), choice(false, "바꾸기"), choice(true, "뒤에 넣기"));
}

function epochLine(row, change) {
  const choice = ([mode, text]) => pill(row.epoch_mode === mode, () => change((r) => { r.epoch_mode = mode; }, true), { on: text, off: text });
  return h("div", { class: "ais-lx-line" }, h("span", { class: "ais-lx-tag" }, epochText(row.lora)), EPOCH_MODES.map(choice),
    row.epoch_mode === "direct"
      ? numberField(row.epoch, (value) => change((r) => { r.epoch = value; }), { min: 1, step: 1, placeholder: "번호", "aria-label": "어디까지 비교할지 에폭 번호" })
      : null);
}

function rowBlock(row, index, view) {
  const isBase = index === 0;
  const change = (mutate, rerender) => view.change((block) => mutate(block.rows[index]), rerender);
  const items = optionItems(isBase ? view.baseItems : view.fileItems, row.lora, { empty: NO_PICK, loaded: isBase });
  const strength = numberField(row.strength, (value) => change((r) => { r.strength = value; }),
    { min: -MAX_STRENGTH, max: MAX_STRENGTH, step: 0.05, placeholder: "그대로" });
  return h("div", { class: `ais-lx-row ${isBase ? "base" : ""}` },
    h("div", { class: "ais-lx-head" },
      h("span", { class: "ais-lx-badge" }, isBase ? "기준" : String(index + 1)),
      select(items, row.lora, (value) => change((r) => { r.lora = value; }, true), { class: `ais-f ais-grow ${isBase ? "" : "ais-lx-files"}`, title: row.lora, "aria-label": isBase ? "프롬프트에서 찾을 LoRA" : "바꿀 LoRA" }),
      isBase ? null : h("button", { type: "button", class: "ais-mini", title: "빼기", "aria-label": "이 LoRA 빼기", onClick: () => view.change((block) => { block.rows.splice(index, 1); }, true) }, "✕")),
    h("div", { class: "ais-lx-fields" },
      field("강도", strength),
      isBase
        ? field("찾을 말", textField(view.block.find, (value) => view.change((block) => { block.find = value; }), "트리거나 프롬프트 단어"))
        : field("새 트리거", textField(row.trigger, (value) => change((r) => { r.trigger = value; }), "비우면 그대로, - 는 지움"))),
    isBase ? placeLine(view) : null,
    epochLine(row, change));
}

/**
 * The inside of a LoRA axis. `view`: `block`, `baseItems`, `fileItems`, `change(mutate(block), rerender)`, `refreshFiles()`.
 */
export function loraAxisBody(view) {
  const add = () => {
    view.change((block) => { block.rows.push(emptyRow()); }, true);
    view.refreshFiles();
  };
  return [view.block.rows.map((row, index) => rowBlock(row, index, view)), h("button", { type: "button", class: "ais-add", onClick: add }, "+ 비교할 LoRA")];
}

/** The columns one LoRA axis makes (`plan.lora_preview`). A file name used as the title is not repeated. */
export function columnList(letter, columns) {
  return h("div", { class: "ais-lx-columns" },
    h("div", { class: "ais-row" }, h("span", { class: "ais-label" }, `만들어질 열 (${letter})`), h("span", { class: "ais-f ais-estimate" }, `${columns.length}열`)),
    columns.map((column) => {
      const named = column.label.startsWith(column.name);
      return h("div", { class: "ais-lx-column" },
        named ? null : h("span", { class: "ais-lx-label", title: column.label }, column.label),
        h("span", { class: `ais-lx-name ${named ? "wide" : ""}`, title: column.name }, column.name),
        h("span", { class: "ais-lx-strength" }, column.strength),
        h("span", { class: `ais-lx-trigger ${column.changed ? "changed" : ""}`, title: column.trigger }, column.trigger));
    }));
}
