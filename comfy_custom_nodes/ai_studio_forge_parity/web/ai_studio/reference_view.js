// Anima studio: the "레퍼런스" card, first in the 확장기능 group, one extension-shaped row.
// Anima picks SnapRef or Anima Ref Adapter; Krea2 uses comfyui-krea2edit with an edit LoRA; MOSS has none.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { extLine } from "./ext_view.js";
import { h, numberBox, optionItems, pill, select } from "./studio_dom.js";
import { FAMILY_LABELS } from "./studio_ext.js";
import { KREA2_POSITIONS, REFERENCE_CORRECTIONS, REFERENCE_LAYOUTS, REFERENCE_METHODS, addPromptTrigger, referenceFile } from "./studio_logic.js";

const METHOD_LABELS = Object.freeze({ snapref: "SnapRef", ref_adapter: "Ref Adapter" });
const MODE_ITEMS = [{ value: "cs", label: "캐릭터+그림체" }, { value: "char", label: "캐릭터만" }];
const LAYOUT_LABELS = Object.freeze({ separate: "따로 (선명, 느림)", stitch: "나란히 붙이기 (빠름)" });
const LAYOUT_ITEMS = REFERENCE_LAYOUTS.map((value) => ({ value, label: LAYOUT_LABELS[value] }));
const POSITION_LABELS = Object.freeze({ "aligned (trained)": "맞춤 (학습 방식)", "beside (right of target)": "옆에 (자세 자유)" });
const NODE_OF_METHOD = Object.freeze({ snapref: "SnapRef", ref_adapter: "AnimaRefAdapter" });
const FILE_PATTERNS = Object.freeze({
  snapref: /snapref|anima_ref_kv/i,
  ref_adapter: /anima_ref_adapter/i,
  krea2: /krea2_(identity|general)_edit|(quadview|dynamiccharactersheet)_krea2/i,
});
const SHEET_FILE = /(quadview|dynamiccharactersheet)_krea2/i;
const QUADVIEW_FILE = /quadview_krea2/i;
const QUADVIEW_CAPTION = "Convert the character in the image to a Character Sheet showing a face close-up, front full body, side full body and back full body views";
const DEFAULT_GROUNDING = 768;
const SHEET_GROUNDING = 1024; // the CharacterSheet workflows feed the text encoder a 1024 px reference
const CARD_KEY = "reference";
const MOSS_NONE = "불가: MOSS 에는 레퍼런스가 없음";
const OLD_GRAPH = "불가: 이 워크플로에는 레퍼런스 노드가 없음. AI Studio Anima.json 을 새로 여세요";

const fileName = (entry) => entry.replace(/\\/g, "/").split("/").pop();
const REMAPPED_COPY = /__\d+b\.[^.]+$/; // reference_layout.py's copies for 2.9B / 3.8B: picked for the model at run time

/** Absent LiteGraph (tests) counts as installed; a missing provider greys the switch and runs without it. */
export function nodeInstalled(name) {
  const types = globalThis.LiteGraph?.registered_node_types;
  return !types || Boolean(types[name]);
}

const packMissing = (name) => `불가: ${name} 노드 팩이 설치돼 있지 않음`;

/** Why a reference cannot run for `family` with Anima's `method` ("" = it can). */
export function referenceBlock(family, method, hasRole) {
  if (family === "moss") return MOSS_NONE;
  const node = family === "krea2" ? "Krea2EditModelPatch" : NODE_OF_METHOD[method];
  if (!nodeInstalled(node)) return packMissing(node);
  const role = family === "krea2" ? "krea2_ref" : "anima_ref";
  return hasRole && !hasRole(role) ? OLD_GRAPH : "";
}

/** A picture box of `key` ("image" | "image_2" | "image_3"), read and uploaded through `node`; `clear` adds a remove button. */
function imageBox(reference, view, { key, node, label, clear = false }) {
  const value = reference[key]; // not the node's value: a fresh LoadImage holds the first file of input/
  const file = h("input", { type: "file", accept: "image/*", hidden: true, onChange: async (event) => {
    const chosen = event.target.files?.[0];
    if (!chosen) return;
    const uploaded = await node?.upload(chosen);
    if (uploaded?.name) view.set((s) => { s.reference[key] = uploaded.name; }, true);
  } });
  const picture = value
    ? h("img", { class: "ais-i2i-img", src: node?.url(value), alt: label })
    : h("span", {}, label);
  const remove = clear && value
    ? h("button", { type: "button", class: "ais-mini ais-ref-clear", title: "빼기", onClick: (event) => {
      event.stopPropagation();
      view.set((s) => { s.reference[key] = ""; }, true);
    } }, "×")
    : null;
  return h("div", { class: "ais-i2i-box", title: "눌러서 이미지 고르기", onClick: () => file.click() }, picture, remove, file);
}

/** SnapRef reads a second picture of the same character; a workflow from before 1.12.0 has no node for it. */
const hasSecondImage = (reference, view) =>
  view.family === "anima" && reference.method === "snapref" && (!view.hasRole || view.hasRole("reference_image_2"));

/** SnapRef v2a reads a third one (1.18.0's node); its box opens once the second picture is there. */
const hasThirdImage = (reference, view) =>
  hasSecondImage(reference, view) && (!view.hasRole || view.hasRole("reference_image_3")) && Boolean(reference.image_2 || reference.image_3);

const field = (label, control) =>
  h("div", { class: "ais-row" }, h("span", { class: "ais-label" }, label), h("span", { class: "ais-ctl" }, control));

// Anima references come as base originals (28 blocks) and as files someone already converted to 2.9B's 40 blocks
// (pamparamm's `*_29B`): two groups, so either can be picked. SnapRef's v2a node fits a base original to 2.9B itself and
// reference_layout.py writes the 3.8B copy; Ref Adapter's node fits nothing, so both of its copies come from us.
const CONVERTED_29B = /(?:^|[^0-9a-z])2\.?9b(?:[^0-9a-z]|$)/i;
const BASE_GROUPS = Object.freeze({ snapref: "Base 원본 (2.9B 는 노드가 맞춤 · 3.8B 는 자동 변환)", ref_adapter: "Base 원본 (2.9B·3.8B 에선 자동 변환)" });
const CONVERTED_GROUP = "2.9B 변환본";

function animaFileItems(entries, key) {
  const item = (entry) => ({ value: entry, label: entry, group: CONVERTED_29B.test(entry) ? CONVERTED_GROUP : BASE_GROUPS[key] });
  return [...entries.filter((entry) => !CONVERTED_29B.test(entry)), ...entries.filter((entry) => CONVERTED_29B.test(entry))].map(item);
}

/** The picks of `key`'s adapter files in the LoRA list; the current pick stays listed even when its file is gone. */
function filePick(key, current, view, pick) {
  const entries = view.loras.filter((entry) => FILE_PATTERNS[key].test(fileName(entry)) && !REMAPPED_COPY.test(entry));
  const items = key === "krea2" ? entries : animaFileItems(entries, key);
  return select(optionItems(items, current, { empty: entries.length ? "[파일]" : "[파일 없음: LoRA 폴더에 받기]" }), current, pick);
}

function number(value, write, options) {
  return numberBox(value, write, { width: 62, ...options });
}

const corrections = REFERENCE_CORRECTIONS.map((value) => ({ value, label: value }));

function snaprefFields(reference, view) {
  const values = reference.snapref;
  const write = (key) => (value) => view.set((s) => { s.reference.snapref[key] = value; });
  const weight = (key, label, title) => field(label, number(values[key], write(key), { min: 0, max: 2, step: 0.05, title }));
  const third = Boolean(reference.image_3) && hasThirdImage(reference, view);
  const several = Boolean(reference.image_2) || third;
  const separate = values.multi_layout !== "stitch"; // stitched pictures are one reference: no per-picture weights
  return [
    field("파일", filePick("snapref", values.file, view, (value) => view.set((s) => { s.reference.snapref.file = value; }, true))),
    field("모드", select(MODE_ITEMS, values.mode, write("mode"))),
    field("세기", number(values.strength, write("strength"), { min: -3, max: 3, step: 0.1, title: "로그 눈금. 0 = 학습값, +1 = 약 2.7배" })),
    several ? field("여러 장", select(LAYOUT_ITEMS, values.multi_layout, (value) => view.set((s) => { s.reference.snapref.multi_layout = value; }, true))) : null,
    several && separate ? weight("weight_1", "비중 1", "첫 그림이 얼마나 반영되는지. 1 = 학습값") : null,
    reference.image_2 && separate ? weight("weight_2", "비중 2", "둘째 그림이 얼마나 반영되는지. 0 = 안 씀") : null,
    third && separate ? weight("weight_3", "비중 3", "셋째 그림이 얼마나 반영되는지. 0 = 안 씀") : null,
    field("가이던스", [pill(values.guidance_on, () => view.set((s) => { s.reference.snapref.guidance_on = !values.guidance_on; }, true)),
      values.guidance_on ? number(values.guidance, write("guidance"), { min: 0, max: 20, step: 0.1, title: "스텝마다 모델 호출 한 번 더" }) : null]),
    values.guidance_on
      ? field("보정", select(corrections, values.correction, write("correction")))
      : field("충실도", number(values.fidelity, write("fidelity"), { min: 0, max: 1, step: 0.05, title: "1 = 그림의 디테일을 살림, 낮출수록 프롬프트를 더 따름. 1 미만이면 느려짐" })),
    field("캡션", h("input", { class: "ais-f ais-grow", value: values.caption, placeholder: "외형 태그 (비워도 됨)", onChange: (event) => write("caption")(event.target.value) })),
  ];
}

function adapterFields(values, view) {
  const write = (key) => (value) => view.set((s) => { s.reference.ref_adapter[key] = value; });
  return [
    field("파일", filePick("ref_adapter", values.file, view, (value) => view.set((s) => { s.reference.ref_adapter.file = value; }, true))),
    field("세기", number(values.strength, write("strength"), { min: 0, max: 2, step: 0.05 })),
    field("가이던스", number(values.guidance, write("guidance"), { min: 0, max: 20, step: 0.1, title: "1 이면 추가 호출 없음" })),
    field("보정", select(corrections, values.correction, write("correction"))),
    field("가이던스 끝", number(values.ref_end, write("ref_end"), { min: 0, max: 1, step: 0.05, title: "0.6 이면 더 빠르고 그림은 거의 같음" })),
  ];
}

function krea2Fields(values, view) {
  const write = (key) => (value) => view.set((s) => { s.reference.krea2[key] = value; });
  const pickFile = (value) => view.set((s) => {
    s.reference.krea2.file = value;
    s.reference.krea2.grounding_px = SHEET_FILE.test(value) ? SHEET_GROUNDING : DEFAULT_GROUNDING;
  }, true);
  const positions = KREA2_POSITIONS.map((value) => ({ value, label: POSITION_LABELS[value] }));
  return [
    field("LoRA", filePick("krea2", values.file, view, pickFile)),
    field("LoRA 세기", number(values.strength, write("strength"), { min: -10, max: 10, step: 0.05 })),
    field("참조 세기", number(values.boost, write("boost"), { min: 0, max: 1000, step: 0.05 })),
    field("참조 크기", number(values.grounding_px, write("grounding_px"), { min: 0, max: 4096, step: 64, title: "TE 에 넣는 이미지의 긴 변" })),
    field("위치", select(positions, values.position, write("position"))),
    QUADVIEW_FILE.test(values.file)
      ? h("button", { type: "button", class: "ais-add", title: "프롬프트 칸 맨 앞에 넣음. 프롬프트 칸이 꺼져 있으면 켬",
        onClick: () => view.set((s) => addPromptTrigger(s, QUADVIEW_CAPTION), true) }, "트리거 넣기") // typing does not redraw this card
      : null,
  ];
}

function animaFields(reference, view) {
  const methods = REFERENCE_METHODS.map((name) =>
    pill(reference.method === name, () => view.set((s) => { s.reference.method = name; }, true), { on: METHOD_LABELS[name], off: METHOD_LABELS[name] }));
  const blocked = referenceBlock("anima", reference.method, view.hasRole);
  return [
    field("방식", methods),
    blocked ? h("div", { class: "ais-note" }, blocked) : null,
    ...(reference.method === "snapref" ? snaprefFields(reference, view) : adapterFields(reference.ref_adapter, view)),
    field("참조 픽셀", number(reference.megapixels, (value) => view.set((s) => { s.reference.megapixels = value; }), { min: 0.25, max: 2, step: 0.05, title: "메가픽셀" })),
  ];
}

function fields(state, view) {
  const { reference } = state;
  const pictures = [imageBox(reference, view, { key: "image", node: view.image, label: "레퍼런스 이미지" })];
  if (hasSecondImage(reference, view)) {
    pictures.push(imageBox(reference, view, { key: "image_2", node: view.image2, label: "같은 캐릭터 2장째 (선택)", clear: true }));
  }
  if (hasThirdImage(reference, view)) {
    pictures.push(imageBox(reference, view, { key: "image_3", node: view.image3, label: "같은 캐릭터 3장째 (선택)", clear: true }));
  }
  return h("div", { class: "ais-ext-body ais-ref-body" }, h("div", { class: "ais-ref-images" }, pictures), h("div", { class: "ais-ext-fields" },
    view.family === "krea2" ? krea2Fields(reference.krea2, view) : animaFields(reference, view)));
}

/**
 * The card. `view`: `set(mutate, rerender)`, `hasRole(role)` (optional), `family`, `memory` (extMemory()),
 * `rerender()`, `loras` (file names), `refreshLoras()` (optional), `image`, `image2` and `image3` {url(value), upload(file) -> {name}}.
 */
export function renderReference(state, view) {
  const { reference } = state;
  const reason = referenceBlock(view.family, reference.method, view.hasRole); // the caret still opens: the other method may run
  const open = view.memory.open.has(CARD_KEY) && view.family !== "moss";
  const flipOpen = () => {
    if (!view.memory.open.delete(CARD_KEY)) {
      view.memory.open.add(CARD_KEY);
      view.refreshLoras?.();
    }
    view.rerender();
  };
  const toggle = () => {
    view.set((s) => {
      s.reference.on = !s.reference.on;
      if (s.reference.on) view.memory.open.add(CARD_KEY); // what it needs is below the switch
    }, true);
    view.refreshLoras?.();
  };
  const ready = reference.image && referenceFile(state, view.family);
  const title = `레퍼런스 (${FAMILY_LABELS[view.family]})`;
  return h("div", { class: "ais-node ais-ext-group" },
    h("div", { class: "ais-nt" }, h("span", {}, "레퍼런스")),
    h("div", { class: "ais-nb" }, extLine({
      key: CARD_KEY,
      label: reference.on && !ready && !reason ? `${title} · 고를 것 남음` : title,
      on: reference.on,
      reason,
      open,
      toggle,
      flipOpen,
      body: () => fields(state, view),
    })));
}
