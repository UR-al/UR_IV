// node:test for the prompt-box presets' pure rules (web/ai_studio/preset_logic.js). The match / name / text rows are the
// golden table tests/test_studio_presets.py runs against studio_presets.py. Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  PRESET_LABELS, PRESET_LIMITS, PRESET_SECTIONS, appendText, autoName, boxStatus, cleanName, filterPresets, itemCount,
  matchPreset, pickSelection, presetEdge, presetMeta, previewLine, sectionList, textProblem, usedText,
} from "../web/ai_studio/preset_logic.js";
import { PROMPT_SECTIONS, assembledPrompt } from "../web/ai_studio/studio_logic.js";

const golden = JSON.parse(readFileSync(new URL("./fixtures/preset_logic_golden.json", import.meta.url), "utf-8"));
const P = (id, name, text, used = 0) => ({ id, name, text, created: 1, used });

test("the preset boxes are the prompt card's boxes plus the negative, each with its label", () => {
  assert.deepEqual([...PRESET_SECTIONS], [...PROMPT_SECTIONS, "negative"]);
  assert.deepEqual(PRESET_SECTIONS.map((key) => PRESET_LABELS[key]), ["선행 고정", "인물 수", "캐릭터", "작품", "작가", "프롬프트", "후행 고정", "네거티브"]);
  assert.deepEqual({ ...PRESET_LIMITS }, { name: 60, text: 20000, count: 200 }); // studio_presets NAME_MAX / TEXT_MAX / MAX_PER_SECTION
});

test("presetEdge cuts what a joined prompt box loses at its ends", () => {
  for (const ch of golden.edge) assert.equal(presetEdge(`${ch}a${ch}b${ch}`), `a${ch}b`, JSON.stringify(ch));
  assert.equal(presetEdge(" ,1boy, solo,\r\n"), "1boy, solo");
  assert.equal(presetEdge(undefined), "");
  assert.equal(presetEdge(5), "");
  // the same cut as the panel's own join
  assert.equal(assembledPrompt({ prompt: "x", prompt_prefix: " ,a, b,\n" }), `${presetEdge(" ,a, b,\n")}, x`);
});

test("matchPreset: golden rows (shared with studio_presets.match_preset)", () => {
  assert.ok(golden.match.length >= 8);
  for (const row of golden.match) assert.equal(matchPreset(row.list, row.text)?.id ?? null, row.id, row.case);
});

test("matchPreset ignores what is not a preset list", () => {
  assert.equal(matchPreset(null, "x"), null);
  assert.equal(matchPreset([{ text: 5 }, "x", null], "x"), null);
  assert.equal(matchPreset([P("a", "A", "x")], undefined), null);
});

test("cleanName / textProblem: golden rows (shared with studio_presets.clean_name / clean_text)", () => {
  for (const row of golden.names) assert.equal(cleanName(row.in), row.out, JSON.stringify(row.in).slice(0, 40));
  assert.equal(cleanName(5), null);
  for (const row of golden.texts) assert.equal(textProblem(row.in) === "", row.ok, JSON.stringify(row.in));
  assert.equal(textProblem("x".repeat(PRESET_LIMITS.text)), "");
  assert.notEqual(textProblem("x".repeat(PRESET_LIMITS.text + 1)), "");
  assert.notEqual(textProblem(null), "");
});

test("appendText joins with ', ' after trimming the box's trailing commas and spaces", () => {
  assert.equal(appendText("1boy, solo", "smile"), "1boy, solo, smile");
  assert.equal(appendText("1boy, solo, \n", "smile"), "1boy, solo, smile");
  assert.equal(appendText("1boy,,,", "smile, "), "1boy, smile, ");
  assert.equal(appendText("", "smile"), "smile");
  assert.equal(appendText(" ,\n", "smile"), "smile");
  assert.equal(appendText(undefined, "smile"), "smile");
  assert.equal(appendText("a,\nb", "c"), "a,\nb, c"); // inner line breaks stay
  assert.equal(appendText("  a", "b"), "  a, b"); // only the end is trimmed
  assert.equal(appendText("a", ", \nb"), "a, b"); // the preset's own leading commas would double the join
  assert.equal(appendText("a", " ,\n"), "a"); // nothing to add: the box stays as written
  assert.equal(appendText("a, ", ""), "a, ");
});

test("autoName is '내 <칸> N' with the next free number", () => {
  assert.equal(autoName([], "네거티브"), "내 네거티브 1");
  assert.equal(autoName([P("a", "Anima 기본", "x"), P("b", "가볍게", "y")], "네거티브"), "내 네거티브 3");
  assert.equal(autoName([P("a", "내 작가 2", "x")], "작가"), "내 작가 3"); // "내 작가 2" is taken
  assert.equal(autoName([P("a", "내 작가 3", "x"), P("b", "내 작가 4", "y")], "작가"), "내 작가 5");
  assert.equal(autoName(null, "작가"), "내 작가 1");
});

test("itemCount counts the comma-separated entries; previewLine is one line", () => {
  assert.equal(itemCount("1boy, solo"), 2);
  assert.equal(itemCount(" a,, b ,\n c, "), 3);
  assert.equal(itemCount("A high-quality anime illustration."), 1);
  assert.equal(itemCount(" , "), 0);
  assert.equal(previewLine(" a,\n  b\t c \n"), "a, b c");
  assert.equal(previewLine(undefined), "");
});

test("usedText says how long ago in calendar days", () => {
  const at = (y, m, d, h = 12) => new Date(y, m - 1, d, h).getTime();
  const now = at(2026, 11, 15, 9);
  const sec = (ms) => Math.floor(ms / 1000);
  assert.equal(usedText(0, now), "");
  assert.equal(usedText(undefined, now), "");
  assert.equal(usedText(sec(at(2026, 11, 15, 1)), now), "오늘");
  assert.equal(usedText(sec(at(2026, 11, 14, 23)), now), "어제"); // ten hours ago, but yesterday
  assert.equal(usedText(sec(at(2026, 11, 12)), now), "3일 전");
  assert.equal(usedText(sec(at(2026, 11, 9)), now), "6일 전");
  assert.equal(usedText(sec(at(2026, 11, 8)), now), "1주 전");
  assert.equal(usedText(sec(at(2026, 10, 20)), now), "3주 전");
  assert.equal(usedText(sec(at(2026, 10, 10)), now), "1달 전");
  assert.equal(usedText(sec(at(2025, 12, 1)), now), "11달 전");
  assert.equal(usedText(sec(at(2024, 11, 1)), now), "2년 전");
  assert.equal(usedText(sec(at(2026, 11, 16)), now), "오늘"); // a clock that ran ahead
});

test("presetMeta: 항목 N개 · N자, plus the last use when there was one", () => {
  const now = new Date(2026, 10, 15, 9).getTime();
  assert.equal(presetMeta(P("a", "A", "1boy, solo"), now), "항목 2개 · 10자");
  assert.equal(presetMeta(P("a", "A", "한글, 🙂", Math.floor(now / 1000)), now), "항목 2개 · 5자 · 오늘 사용");
});

test("filterPresets searches the name and the text, ignoring case; an empty search keeps all in order", () => {
  const list = [P("a", "Anima 기본", "worst quality, lowres"), P("b", "가볍게", "Blurry, JPEG artifacts"), P("c", "글자 막기", "text, logo")];
  assert.deepEqual(filterPresets(list, "").map((p) => p.id), ["a", "b", "c"]);
  assert.deepEqual(filterPresets(list, "  ").map((p) => p.id), ["a", "b", "c"]);
  assert.deepEqual(filterPresets(list, "anima").map((p) => p.id), ["a"]);
  assert.deepEqual(filterPresets(list, "jpeg").map((p) => p.id), ["b"]);
  assert.deepEqual(filterPresets(list, " 글자 ").map((p) => p.id), ["c"]);
  assert.deepEqual(filterPresets(list, "l").map((p) => p.id), ["a", "b", "c"]);
  assert.deepEqual(filterPresets(list, "zzz"), []);
  assert.deepEqual(filterPresets(null, "x"), []);
});

test("boxStatus: the same as a preset, not saved, or empty", () => {
  const list = [P("a", "Anima 기본", "worst quality")];
  assert.deepEqual(boxStatus(list, " worst quality, "), { kind: "match", preset: list[0], text: "프리셋 ‘Anima 기본’과 같음" });
  assert.deepEqual(boxStatus(list, "other"), { kind: "unsaved", preset: null, text: "저장 안 된 내용" });
  assert.deepEqual(boxStatus(list, " ,\n"), { kind: "empty", preset: null, text: "비어 있음" });
  assert.deepEqual(boxStatus([], undefined), { kind: "empty", preset: null, text: "비어 있음" });
});

test("pickSelection keeps the selected preset while it is there, else the box's preset, else the first", () => {
  const list = [P("a", "A", "x"), P("b", "B", "y")];
  assert.equal(pickSelection(list, "y", "a"), "a");
  assert.equal(pickSelection(list, "y", "gone"), "b");
  assert.equal(pickSelection(list, "y", null), "b");
  assert.equal(pickSelection(list, "zzz", null), "a");
  assert.equal(pickSelection([], "x", "a"), null);
});

test("sectionList reads a document's box list and never throws on a missing one", () => {
  const doc = { sections: { negative: [P("a", "A", "x")] } };
  assert.deepEqual(sectionList(doc, "negative").map((p) => p.id), ["a"]);
  assert.deepEqual(sectionList(doc, "prompt"), []);
  assert.deepEqual(sectionList(null, "negative"), []);
  assert.deepEqual(sectionList({ sections: { negative: "x" } }, "negative"), []);
});
