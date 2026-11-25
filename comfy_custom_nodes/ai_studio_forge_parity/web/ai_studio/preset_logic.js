// Prompt-box presets: the pure rules shared by the quick list (preset_quick.js) and the preset window (preset_window.js).
// A preset = {id, name, text, created, used} (Unix seconds; used 0 = never put into a box), kept per box by studio_presets.py.
// matchPreset / cleanName / textProblem are the twins of studio_presets.match_preset / clean_name / clean_text
// (js_tests/fixtures/preset_logic_golden.json drives both sides).
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { BOX_LABELS, PROMPT_SECTIONS } from "./studio_logic.js";

/** The boxes that have presets: the prompt card's boxes in their order, then the negative (studio_presets.SECTIONS). */
export const PRESET_SECTIONS = Object.freeze([...PROMPT_SECTIONS, "negative"]);
/** Box key -> its name on the panel: the panel's own table (studio_logic.js BOX_LABELS), not a copy. */
export const PRESET_LABELS = BOX_LABELS;
/** studio_presets NAME_MAX / TEXT_MAX / MAX_PER_SECTION (the route's `limits` answer wins when it is there). */
export const PRESET_LIMITS = Object.freeze({ name: 60, text: 20000, count: 200 });
export const PRESET_FILE_NOTE = "저장: ComfyUI user/default/ai_studio/prompt_presets.json";

const DAY_MS = 86400000;
const length = (text) => Array.from(text).length; // code points, as Python's len

/** `text` without what a joined prompt box loses at its ends (studio_logic promptEdge / studio_state._PROMPT_EDGE). */
export const presetEdge = (text) => (typeof text === "string" ? text.replace(/^[ \t\r\n,]+|[ \t\r\n,]+$/g, "") : "");

/** A document's list for one box (never null). */
export function sectionList(doc, key) {
  const list = doc?.sections?.[key];
  return Array.isArray(list) ? list : [];
}

/** The first preset whose text equals the box's once both lose their edges; an empty box matches nothing. */
export function matchPreset(list, text) {
  const want = presetEdge(text);
  if (!want || !Array.isArray(list)) return null;
  return list.find((preset) => typeof preset?.text === "string" && presetEdge(preset.text) === want) || null;
}

/** "+ 붙이기": the box without its trailing commas / spaces, then ", " and the preset (nothing to add = the box as written). */
export function appendText(current, addition) {
  const box = typeof current === "string" ? current : "";
  const add = typeof addition === "string" ? addition.replace(/^[ \t\r\n,]+/, "") : "";
  if (!add) return box;
  const base = box.replace(/[ \t\r\n,]+$/, "");
  return base ? `${base}, ${add}` : add;
}

/** The name as the store keeps it (white space runs -> one space, ends cut), or null when it is empty or too long. */
export function cleanName(name, limits = PRESET_LIMITS) {
  if (typeof name !== "string") return null;
  const clean = name.replace(/\s+/g, " ").trim();
  return clean && length(clean) <= limits.name ? clean : null;
}

/** "" when `text` can be a preset, else why not (the store refuses the same). */
export function textProblem(text, limits = PRESET_LIMITS) {
  if (typeof text !== "string" || !presetEdge(text)) return "내용이 비어 있어요";
  return length(text) > limits.text ? `내용은 ${limits.text}자까지예요` : "";
}

/** "내 <칸> N": N = the list's size + 1, raised while that name is taken. */
export function autoName(list, label) {
  const all = Array.isArray(list) ? list : [];
  const names = new Set(all.map((preset) => preset?.name));
  let number = all.length + 1;
  while (names.has(`내 ${label} ${number}`)) number += 1;
  return `내 ${label} ${number}`;
}

/** How many comma-separated entries the text has. */
export const itemCount = (text) => String(text || "").split(",").filter((part) => part.trim()).length;

/** The text on one line (for a row's preview). */
export const previewLine = (text) => (typeof text === "string" ? text.replace(/\s+/g, " ").trim() : "");

/** "오늘" / "어제" / "N일 전" / "N주 전" / "N달 전" / "N년 전" by calendar days; "" for a preset never used. */
export function usedText(usedSeconds, nowMs = Date.now()) {
  if (!Number.isFinite(usedSeconds) || usedSeconds <= 0) return "";
  const midnight = (ms) => { const d = new Date(ms); return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime(); };
  const days = Math.round((midnight(nowMs) - midnight(usedSeconds * 1000)) / DAY_MS);
  if (days <= 0) return "오늘";
  if (days === 1) return "어제";
  if (days < 7) return `${days}일 전`;
  if (days < 30) return `${Math.floor(days / 7)}주 전`;
  if (days < 365) return `${Math.floor(days / 30)}달 전`;
  return `${Math.floor(days / 365)}년 전`;
}

/** A card's small line: "항목 N개 · N자", then " · <언제> 사용" once the preset was put into a box. */
export function presetMeta(preset, nowMs = Date.now()) {
  const text = String(preset?.text || "");
  const used = usedText(preset?.used, nowMs);
  return `항목 ${itemCount(text)}개 · ${length(text)}자${used ? ` · ${used} 사용` : ""}`;
}

/** The presets whose name or text has `query` (case ignored), in the list's order. */
export function filterPresets(list, query) {
  const all = Array.isArray(list) ? list : [];
  const q = String(query || "").trim().toLowerCase();
  if (!q) return all;
  return all.filter((preset) => String(preset.name).toLowerCase().includes(q) || String(preset.text).toLowerCase().includes(q));
}

/** What the box holds: {kind: "match" | "unsaved" | "empty", preset, text} (the badge of "지금 <칸> 칸"). */
export function boxStatus(list, text) {
  const preset = matchPreset(list, text);
  if (preset) return { kind: "match", preset, text: `프리셋 ‘${preset.name}’과 같음` };
  return presetEdge(text) ? { kind: "unsaved", preset: null, text: "저장 안 된 내용" } : { kind: "empty", preset: null, text: "비어 있음" };
}

/**
 * The small line under a box's name on the panel: {kind: "match" | "unsaved" | "empty", text, title}. `preset` = the preset
 * the box holds, null when no preset has its text, undefined while the presets are not read (nothing is said then).
 */
export function labelMark(preset, text) {
  if (preset) return { kind: "match", text: preset.name, title: `프리셋 ‘${preset.name}’과 같음` };
  if (preset === null && presetEdge(text)) return { kind: "unsaved", text: "저장 안 됨", title: "같은 내용의 프리셋이 없음 (이름을 눌러 저장)" };
  return { kind: "empty", text: "", title: "" };
}

/** The preset the window shows on the right: `previous` while it is still listed, else the box's preset, else the first. */
export function pickSelection(list, text, previous) {
  const all = Array.isArray(list) ? list : [];
  if (previous && all.some((preset) => preset.id === previous)) return previous;
  return (matchPreset(all, text) || all[0])?.id ?? null;
}
