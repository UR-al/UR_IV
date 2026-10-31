// Anima studio: the control panel's "확장기능" group, drawn from the extension table (ext_schema.js).
// One switch-name-caret line per extension; a caret unfolds its fields, and the suite's fields fold again by section.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { EXT_SCHEMA } from "./ext_schema.js";
import { h, numberBox, optionItems, pill, select } from "./studio_dom.js";
import { comboItems, extFamilyBlock, extFieldValue } from "./studio_ext.js";

const EXTENSIONS = EXT_SCHEMA.extensions;
const GROUP_COLOR = "#6b5f8a";
const missingNode = (label) => `불가: 이 워크플로에는 ${label} 노드가 없음. AI Studio Anima.json 을 새로 여세요`;

/** What the panel remembers between redraws, never saved: unfolded rows and sections, the file lists read so far. */
export const extMemory = () => ({ open: new Set(), lists: new Map(), asked: new Set() });

const blockReason = (item, hasRole, family = "anima") =>
  (hasRole && !hasRole(item.role) ? missingNode(item.label) : extFamilyBlock(item.key, family));

/** Turns off, in a run's state, the extensions this graph lacks or `family` cannot run (their switch is greyed). */
export function dropBlockedExt(ext, hasRole, family = "anima") {
  for (const item of EXTENSIONS) if (blockReason(item, hasRole, family)) ext[item.key].on = false;
}

const hasFileList = (item) => item.fields.some((field) => field.kind === "choice" && !field.choices.length);

/** Reads the class's COMBO lists once; a failure leaves the fields on their current value until the next try. */
function askLists(item, { memory, fetchChoices, rerender }) {
  const comfyClass = item.class;
  if (!fetchChoices || !hasFileList(item) || memory.asked.has(comfyClass)) return;
  memory.asked.add(comfyClass);
  new Promise((resolve) => resolve(fetchChoices(comfyClass)))
    .then((inputs) => {
      memory.lists.set(comfyClass, inputs);
      rerender();
    })
    .catch((error) => {
      memory.asked.delete(comfyClass);
      console.warn(`[AI Studio] object_info ${comfyClass}:`, error);
    });
}

function numberField(field, value, write) {
  const box = numberBox(value, (number) => {
    const clamped = extFieldValue(field, number);
    box.value = String(clamped);
    write(clamped);
  }, { min: field.min, max: field.max, step: field.kind === "int" ? 1 : "any", width: 72 });
  return box;
}

function choiceField(item, field, value, write, memory) {
  const items = field.choices.length ? field.choices : comboItems(memory.lists.get(item.class), field.input);
  const empty = field.default === "" ? `[${field.label}]` : undefined; // a file pick may stay empty
  return select(optionItems(items, value, { empty }), value, write);
}

function fieldControl(item, field, values, view) {
  const value = values[field.key];
  const write = (next, rerender = false) => view.set((s) => { s.ext[item.key][field.key] = next; }, rerender);
  if (field.kind === "bool") return pill(value, () => write(!value, true));
  if (field.kind === "int" || field.kind === "float") return numberField(field, value, write);
  if (field.kind === "choice") return choiceField(item, field, value, write, view.memory);
  return h("input", { class: "ais-f ais-grow", value, onChange: (event) => write(event.target.value) });
}

function fieldGrid(item, fields, values, view) {
  return h("div", { class: "ais-ext-fields" }, fields.map((field) => {
    const control = fieldControl(item, field, values, view);
    control.setAttribute("data-field", field.key);
    return h("div", { class: "ais-row" }, h("span", { class: "ais-label" }, field.label), h("span", { class: "ais-ctl" }, control));
  }));
}

const flip = (view, key) => {
  const { open } = view.memory;
  if (!open.delete(key)) open.add(key);
  view.rerender();
};

function section(item, name, values, view) {
  const key = `${item.key}/${name}`;
  const open = view.memory.open.has(key);
  const fields = item.fields.filter((field) => field.section === name);
  const lit = fields.some((field) => item.active_keys?.includes(field.key) && values[field.key]);
  const head = h("button", { type: "button", class: `ais-ext-sec ${lit ? "lit" : ""}`, "aria-expanded": open ? "true" : "false", onClick: () => flip(view, key) },
    `${open ? "▴" : "▾"} ${name}`);
  return [head, open ? fieldGrid(item, fields, values, view) : null];
}

function fieldsBody(item, values, view) {
  const plain = item.fields.filter((field) => !field.section);
  const sections = [...new Set(item.fields.map((field) => field.section).filter(Boolean))];
  return h("div", { class: "ais-ext-body" }, plain.length ? fieldGrid(item, plain, values, view) : null,
    sections.map((name) => section(item, name, values, view)));
}

function extRow(item, state, view) {
  const values = state.ext[item.key];
  const reason = blockReason(item, view.hasRole, view.family);
  const open = view.memory.open.has(item.key);
  if (open || values.on) askLists(item, view); // a switched-on one too: the run notices read its list
  return extLine({
    key: item.key,
    label: item.label,
    on: values.on,
    reason,
    open,
    toggle: () => view.set((s) => { s.ext[item.key].on = !s.ext[item.key].on; }, true),
    flipOpen: () => flip(view, item.key),
    body: () => fieldsBody(item, values, view),
  });
}

/** One switch-name-caret line with its fields below it, the shape of every extension row (`reference_view.js` too). */
export function extLine({ key, label, on, reason, open, toggle, flipOpen, body }) {
  const line = h("div", { class: "ais-ext-line" },
    pill(on && !reason, toggle, { title: reason || undefined, disabled: Boolean(reason) }),
    h("span", { class: "ais-ext-name" }, label),
    h("button", { type: "button", class: "ais-caret", "aria-label": `${label} 설정`, "aria-expanded": open ? "true" : "false", title: "설정", onClick: flipOpen },
      open ? "▴" : "▾"));
  return h("div", { class: "ais-ext", "data-ext": key }, line, open ? body() : null);
}

function groupCard(name, items, state, view) {
  const wide = items.some((item) => view.memory.open.has(item.key));
  return h("div", { class: `ais-node ais-ext-group ${wide ? "ais-ext-wide" : ""}` },
    h("div", { class: "ais-nt" }, h("span", {}, name)),
    h("div", { class: "ais-nb" }, items.map((item) => extRow(item, state, view))));
}

/**
 * The "확장기능" group. `ctx`: `set(mutate, rerender)`, `hasRole(role)` (optional), `fetchChoices(class)` -> Promise of the
 * `/object_info/<class>` input (optional), `memory` (extMemory()), `rerender()` (a redraw without a state change),
 * `family` (`modelFamily`, optional), `leading` (optional card shown before the groups).
 */
export function renderExtensions(state, ctx) {
  const view = { memory: extMemory(), rerender: () => {}, ...ctx };
  const count = EXTENSIONS.filter((item) => state.ext[item.key].on && !blockReason(item, view.hasRole, view.family)).length;
  const groups = [...new Set(EXTENSIONS.map((item) => item.group))];
  return h("div", { class: "ais-grp ais-ext-root", style: `--c:${GROUP_COLOR}` },
    h("div", { class: "ais-gh ais-ext-head" }, h("span", {}, "확장기능"), count ? h("span", { class: "ais-ext-count" }, `${count}개 켬`) : null),
    h("div", { class: "ais-grid2" }, view.leading || null, groups.map((name) => groupCard(name, EXTENSIONS.filter((item) => item.group === name), state, view))));
}
