// node:test with a fake document (preset_fakes.mjs): the prompt-box preset window (web/ai_studio/preset_window.js and its
// parts in preset_window_view.js). Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { createPresetStore, presetApi } from "../web/ai_studio/preset_api.js";
import { openPresetWindow } from "../web/ai_studio/preset_window.js";
import { fakeServer, installFakeDom, settle } from "./preset_fakes.mjs";

const NEG = "worst quality, low quality, lowres";
const LIGHT = "worst quality,\nblurry";
const BAR = (n) => `프리셋 ${n}개 · 내 순서 · 카드를 누르면 오른쪽에서 고칠 수 있어요`;

let last = null; // the window a test left open (the module shows one at a time)

async function setup({ text = "", key = "negative", seed = true, load = true, opts = {} } = {}) {
  last?.close();
  const doc = installFakeDom();
  const server = fakeServer();
  const ids = seed ? [server.seed("negative", "Anima 기본", NEG), server.seed("negative", "가볍게", LIGHT), server.seed("negative", "글자 막기", "text, logo")] : [];
  if (seed) server.seed("prompt_artist", "Ena", "@enaa97");
  const store = createPresetStore(presetApi(server.route));
  const opener = doc.createElement("button");
  doc.body.append(opener);
  opener.focus();
  const texts = { negative: text };
  const env = { doc, server, store, ids, texts, sets: [], closedCount: 0, opener };
  env.handle = openPresetWindow(key, {
    store,
    getText: (k) => texts[k] ?? "",
    setText: (k, value) => { env.sets.push([k, value]); texts[k] = value; },
    onClose: () => { env.closedCount += 1; },
    ...opts,
  });
  last = env.handle;
  env.backdrop = () => doc.body.querySelector(".ais-pp-backdrop");
  env.win = () => doc.body.querySelector(".ais-pp");
  env.one = (name) => env.win().one(name);
  env.cards = () => env.win().querySelectorAll(".ais-pp-card");
  env.cardNames = () => env.cards().map((card) => card.one("ais-pp-card-name").textContent);
  env.card = (name) => env.cards().find((card) => card.one("ais-pp-card-name").textContent === name);
  env.msg = () => env.one("ais-pp-msg").textContent;
  env.tab = (label) => env.win().querySelectorAll(".ais-pp-tab").find((tab) => tab.textContent.startsWith(label));
  env.ops = () => server.ops().filter((op) => op.op !== "use");
  if (load) await settle();
  return env;
}

test("the window opens on the box's tab over the page, in the LoRA window's shape", async () => {
  const env = await setup({ text: `${NEG}, ` });
  assert.equal(env.backdrop().parent, env.doc.body);
  assert.equal(env.win().getAttribute("role"), "dialog");
  assert.equal(env.win().getAttribute("aria-modal"), "true");
  assert.equal(env.doc.head.all((el) => el.id === "aistudio-preset-css").length, 1);
  assert.equal(env.win().querySelector("h1").textContent, "네거티브 프리셋");
  assert.equal(env.one("ais-pp-hint").textContent, "칸 이름을 눌러 열림");
  assert.equal(env.doc.activeElement, env.one("ais-pp-search"));
  const tabs = env.win().querySelectorAll(".ais-pp-tab");
  assert.deepEqual(tabs.map((tab) => tab.textContent), ["선행 고정0", "인물 수0", "캐릭터0", "작품0", "작가1", "프롬프트0", "후행 고정0", "네거티브3"]);
  assert.deepEqual(tabs.map((tab) => tab.getAttribute("aria-selected")), ["false", "false", "false", "false", "false", "false", "false", "true"]);
  assert.equal(env.one("ais-pp-toggle").textContent, "넣으면 닫기");
  assert.equal(env.one("ais-pp-toggle").getAttribute("aria-pressed"), "false");
  assert.equal(env.one("ais-pp-path").textContent, "저장: ComfyUI user/default/ai_studio/prompt_presets.json");
});

test("a card per preset: name, text, 항목 · 자, with '지금 칸' on the one the box holds", async () => {
  const env = await setup({ text: `${NEG}, ` });
  assert.deepEqual(env.cardNames(), ["Anima 기본", "가볍게", "글자 막기"]);
  assert.equal(env.one("ais-pp-summary").textContent, BAR(3));
  const first = env.cards()[0];
  assert.equal(first.one("ais-pp-card-text").textContent, NEG);
  assert.equal(first.one("ais-pp-card-meta").textContent, "항목 3개 · 34자");
  assert.equal(first.one("ais-pp-badge").textContent, "지금 칸");
  assert.equal(first.classList.contains("cur"), true);
  assert.equal(first.classList.contains("sel"), true, "the box's preset is the one shown on the right");
  assert.equal(env.win().querySelectorAll(".ais-pp-badge").length, 1);
  assert.equal(first.one("ais-pp-replace").textContent, "바꾸기");
  assert.equal(first.one("ais-pp-append").textContent, "+ 붙이기");
  assert.equal(env.msg(), "네거티브 칸이 프리셋 ‘Anima 기본’과 같아요.");
  assert.equal(env.one("ais-pp-undo").hidden, true);
});

test("the right pane shows the box as it is and the selected preset's editors", async () => {
  const env = await setup({ text: "my own text" });
  assert.equal(env.win().querySelectorAll(".ais-pp-cap")[0].textContent, "지금 네거티브 칸");
  assert.equal(env.one("ais-pp-current").textContent, "my own text");
  assert.equal(env.one("ais-pp-status").textContent, "저장 안 된 내용");
  assert.equal(env.one("ais-pp-status").classList.contains("unsaved"), true);
  assert.equal(env.one("ais-pp-name").value, "Anima 기본", "no preset equals the box: the first one is shown");
  assert.equal(env.one("ais-pp-text").value, NEG);
  assert.equal(env.one("ais-pp-savedraft").disabled, true);
  assert.equal(env.one("ais-pp-edit").hidden, false);
  assert.equal(env.one("ais-pp-none").hidden, true);
  assert.equal(env.msg(), "네거티브 칸 프리셋 3개");

  const same = await setup({ text: `\n${LIGHT},` });
  assert.equal(same.one("ais-pp-status").textContent, "프리셋 ‘가볍게’과 같음");
  assert.equal(same.one("ais-pp-status").classList.contains("match"), true);
  assert.equal(same.one("ais-pp-name").value, "가볍게");

  const empty = await setup({ text: "" });
  assert.equal(empty.one("ais-pp-status").textContent, "비어 있음");
  assert.equal(empty.one("ais-pp-current").textContent, "(비어 있음)");
  assert.equal(empty.one("ais-pp-clear").disabled, true);
});

test("바꾸기 on a card puts the preset into the box at once; 되돌리기 takes it back", async () => {
  const env = await setup({ text: "my own text" });
  env.card("가볍게").one("ais-pp-replace").click();
  await settle();
  assert.deepEqual(env.sets, [["negative", LIGHT]]);
  assert.equal(env.msg(), "네거티브 칸을 ‘가볍게’(으)로 바꿨어요.");
  assert.equal(env.card("가볍게").querySelectorAll(".ais-pp-badge").length, 1);
  assert.equal(env.one("ais-pp-status").textContent, "프리셋 ‘가볍게’과 같음");
  assert.deepEqual(env.server.ops(), [{ op: "use", section: "negative", id: env.ids[1] }]);
  assert.ok(env.win(), "the window stays open");
  const undo = env.one("ais-pp-undo");
  assert.equal(undo.hidden, false);
  undo.click();
  await settle();
  assert.equal(env.texts.negative, "my own text");
  assert.equal(env.msg(), "되돌렸어요.");
  assert.equal(env.one("ais-pp-undo").hidden, true);
  assert.equal(env.one("ais-pp-status").textContent, "저장 안 된 내용");
});

test("+ 붙이기 on a card joins the preset after the box's text", async () => {
  const env = await setup({ text: "bad hands, \n" });
  env.card("글자 막기").one("ais-pp-append").click();
  await settle();
  assert.deepEqual(env.sets, [["negative", "bad hands, text, logo"]]);
  assert.equal(env.msg(), "‘글자 막기’을(를) 네거티브 칸 뒤에 붙였어요.");
  assert.equal(env.one("ais-pp-current").textContent, "bad hands, text, logo");
});

test("'넣으면 닫기' is remembered and closes the window after one 바꾸기", async () => {
  const env = await setup({ text: "x" });
  env.one("ais-pp-toggle").click();
  await settle();
  assert.equal(env.one("ais-pp-toggle").getAttribute("aria-pressed"), "true");
  assert.equal(env.one("ais-pp-toggle").classList.contains("on"), true);
  assert.equal(env.msg(), "넣으면 닫기 켬: 바꾸기 · 붙이기 한 번에 창이 닫혀요.");
  assert.deepEqual(env.ops(), [{ op: "prefs", prefs: { close_after_insert: true } }]);
  env.card("가볍게").one("ais-pp-replace").click();
  assert.equal(env.win(), null);
  assert.equal(env.texts.negative, LIGHT);
  assert.equal(env.closedCount, 1);

  const reopened = await setup({ text: "x" });
  reopened.server.doc.prefs.close_after_insert = true; // switched on in another tab meanwhile
  await reopened.store.load(true);
  assert.equal(reopened.one("ais-pp-toggle").getAttribute("aria-pressed"), "false", "an open window keeps its own switch");
  reopened.handle.close();
  last = openPresetWindow("negative", { store: reopened.store, getText: () => "x", setText() {} });
  assert.equal(reopened.doc.body.querySelector(".ais-pp-toggle").getAttribute("aria-pressed"), "true", "a new window starts from the saved switch");
  reopened.doc.body.querySelector(".ais-pp-toggle").click();
  assert.equal(reopened.doc.body.querySelector(".ais-pp-msg").textContent, "넣으면 닫기 끔");
});

test("a tab shows that box's presets, its count and a fresh search", async () => {
  const env = await setup({ text: "x" });
  env.one("ais-pp-search").typeIn("가볍");
  assert.deepEqual(env.cardNames(), ["가볍게"]);
  env.tab("작가").click();
  assert.equal(env.win().querySelector("h1").textContent, "작가 프리셋");
  assert.deepEqual(env.cardNames(), ["Ena"]);
  assert.equal(env.one("ais-pp-search").value, "");
  assert.equal(env.msg(), "작가 칸 프리셋 1개");
  assert.equal(env.tab("작가").getAttribute("aria-selected"), "true");
  assert.equal(env.tab("네거티브").getAttribute("aria-selected"), "false");
  assert.equal(env.win().querySelectorAll(".ais-pp-cap")[0].textContent, "지금 작가 칸");
  assert.equal(env.one("ais-pp-name").value, "Ena");
  env.card("Ena").one("ais-pp-replace").click();
  assert.deepEqual(env.sets, [["prompt_artist", "@enaa97"]]);
  env.tab("캐릭터").click();
  assert.equal(env.one("ais-pp-empty").textContent, "캐릭터 칸 프리셋이 아직 없어요. 위의 ‘+ 지금 칸 내용 저장’으로 만들 수 있어요.");
  assert.equal(env.one("ais-pp-edit").hidden, true);
  assert.equal(env.one("ais-pp-none").hidden, false);
});

test("the search looks in names and texts; Ctrl+F goes to it", async () => {
  const env = await setup({ text: "x" });
  env.one("ais-pp-search").typeIn("LOGO");
  assert.deepEqual(env.cardNames(), ["글자 막기"]);
  assert.equal(env.one("ais-pp-summary").textContent, BAR(1));
  env.one("ais-pp-search").typeIn("zzz");
  assert.deepEqual(env.cardNames(), []);
  assert.equal(env.one("ais-pp-empty").textContent, "맞는 프리셋이 없어요. 위의 ‘+ 지금 칸 내용 저장’으로 만들 수 있어요.");
  assert.equal(env.one("ais-pp-empty").hidden, false);
  env.one("ais-pp-search").typeIn("");
  assert.equal(env.cardNames().length, 3);
  assert.equal(env.one("ais-pp-empty").hidden, true);
  env.one("ais-pp-clear").focus();
  const event = env.one("ais-pp-clear").fire("keydown", { key: "f", ctrlKey: true });
  assert.equal(env.doc.activeElement, env.one("ais-pp-search"));
  assert.equal(event.defaultPrevented, true);
});

test("'+ 지금 칸 내용 저장' asks for a name in the bar and saves the box as a new preset", async () => {
  const env = await setup({ text: "bad anatomy, bad hands" });
  assert.equal(env.one("ais-pp-bar-form").hidden, true);
  env.one("ais-pp-savenew").click();
  assert.equal(env.one("ais-pp-bar-form").hidden, false);
  assert.equal(env.one("ais-pp-bar-info").hidden, true);
  const input = env.one("ais-pp-newname");
  assert.equal(env.doc.activeElement, input);
  assert.equal(input.placeholder, "비워 두면 ‘내 네거티브 4’");
  input.typeIn("  인체   강화 ");
  env.one("ais-pp-commit").click();
  await settle();
  assert.deepEqual(env.ops(), [{ op: "add", section: "negative", name: "인체 강화", text: "bad anatomy, bad hands" }]);
  assert.deepEqual(env.cardNames(), ["Anima 기본", "가볍게", "글자 막기", "인체 강화"]);
  assert.equal(env.card("인체 강화").classList.contains("sel"), true);
  assert.equal(env.card("인체 강화").querySelectorAll(".ais-pp-badge").length, 1);
  assert.equal(env.one("ais-pp-name").value, "인체 강화");
  assert.equal(env.one("ais-pp-bar-form").hidden, true);
  assert.equal(env.msg(), "지금 네거티브 칸 내용을 ‘인체 강화’(으)로 저장했어요.");
  assert.equal(env.tab("네거티브").textContent, "네거티브4");
  assert.deepEqual(env.sets, []);

  env.one("ais-pp-undo").click();
  await settle();
  assert.deepEqual(env.cardNames(), ["Anima 기본", "가볍게", "글자 막기"]);
  assert.deepEqual(env.ops()[1], { op: "delete", section: "negative", id: "id5" }, "only the preset it added");
  assert.equal(env.msg(), "되돌렸어요.");
});

test("the new preset's name: empty = '내 <칸> N', Enter saves, Esc and 취소 only close the name box", async () => {
  const env = await setup({ text: "new text" });
  env.one("ais-pp-savenew").click();
  env.one("ais-pp-newname").fire("keydown", { key: "Escape" });
  assert.ok(env.win(), "Esc closed the name box, not the window");
  assert.equal(env.one("ais-pp-bar-form").hidden, true);
  env.one("ais-pp-savenew").click();
  env.one("ais-pp-cancel").click();
  assert.equal(env.one("ais-pp-bar-form").hidden, true);
  assert.deepEqual(env.ops(), []);
  env.one("ais-pp-savenew").click();
  const enter = env.one("ais-pp-newname").fire("keydown", { key: "Enter" });
  assert.equal(enter.defaultPrevented, true);
  await settle();
  assert.deepEqual(env.ops(), [{ op: "add", section: "negative", name: "내 네거티브 4", text: "new text" }]);

  const long = await setup({ text: "new text" });
  long.one("ais-pp-savenew").click();
  long.one("ais-pp-newname").typeIn("x".repeat(61));
  long.one("ais-pp-commit").click();
  await settle();
  assert.equal(long.msg(), "이름은 60자까지예요.");
  assert.deepEqual(long.ops(), []);
  assert.equal(long.one("ais-pp-bar-form").hidden, false, "the name can be shortened and saved again");
});

test("'+ 지금 칸 내용 저장' says why when the box is empty or already a preset", async () => {
  const env = await setup({ text: " ,\n" });
  env.one("ais-pp-savenew").click();
  assert.equal(env.msg(), "네거티브 칸이 비어 있어서 저장할 게 없어요.");
  assert.equal(env.one("ais-pp-bar-form").hidden, true);
  env.texts.negative = `${LIGHT}, `;
  env.one("ais-pp-savenew").click();
  assert.equal(env.msg(), "이미 프리셋 ‘가볍게’과 같은 내용이에요.");
  assert.equal(env.card("가볍게").classList.contains("sel"), true);
  assert.equal(env.one("ais-pp-bar-form").hidden, true);
  assert.deepEqual(env.ops(), []);
});

test("a change the server refuses is said in the foot line and changes nothing", async () => {
  const env = await setup({ text: "new text" });
  env.one("ais-pp-savenew").click();
  env.server.fail = "한 칸에 프리셋은 200개까지예요";
  env.one("ais-pp-commit").click();
  await settle();
  assert.equal(env.msg(), "저장하지 못했어요: 한 칸에 프리셋은 200개까지예요");
  assert.equal(env.cardNames().length, 3);
  assert.equal(env.one("ais-pp-undo").hidden, true);
  assert.equal(env.one("ais-pp-bar-form").hidden, false);
});

test("editing: '고친 내용 저장' lights up only for a change and saves the tidied name and the text", async () => {
  const env = await setup({ text: "x" });
  const save = env.one("ais-pp-savedraft");
  assert.equal(save.disabled, true);
  env.one("ais-pp-name").typeIn("Anima 기본 ");
  assert.equal(save.disabled, false);
  env.one("ais-pp-name").typeIn("Anima 기본");
  assert.equal(save.disabled, true, "back to what is saved");
  env.one("ais-pp-name").typeIn("  Anima   v2 ");
  env.one("ais-pp-text").typeIn(`${NEG}, blurry`);
  save.click();
  await settle();
  assert.deepEqual(env.ops(), [{ op: "update", section: "negative", id: env.ids[0], name: "Anima v2", text: `${NEG}, blurry` }]);
  assert.equal(env.msg(), "‘Anima v2’을(를) 고쳐서 저장했어요.");
  assert.deepEqual(env.cardNames(), ["Anima v2", "가볍게", "글자 막기"]);
  assert.equal(env.one("ais-pp-name").value, "Anima v2");
  assert.equal(env.one("ais-pp-savedraft").disabled, true);
  env.one("ais-pp-name").typeIn("typed after saving");
  env.one("ais-pp-undo").click();
  await settle();
  assert.deepEqual(env.ops()[1], { op: "update", section: "negative", id: env.ids[0], name: "Anima 기본", text: NEG });
  assert.deepEqual(env.cardNames(), ["Anima 기본", "가볍게", "글자 막기"]);
  assert.equal(env.card("Anima 기본").one("ais-pp-card-text").textContent, NEG);
  assert.equal(env.one("ais-pp-name").value, "Anima 기본", "the editors go back with the preset");
  assert.equal(env.one("ais-pp-text").value, NEG);
  assert.equal(env.one("ais-pp-savedraft").disabled, true);
  assert.equal(env.msg(), "되돌렸어요.");
});

test("editing refuses an empty name or text before asking the server; Ctrl+S saves from the editors", async () => {
  const env = await setup({ text: "x" });
  env.one("ais-pp-name").typeIn("   ");
  env.one("ais-pp-savedraft").click();
  await settle();
  assert.equal(env.msg(), "이름을 적어 주세요.");
  env.one("ais-pp-name").typeIn("ok");
  env.one("ais-pp-text").typeIn(" ,\n");
  env.one("ais-pp-savedraft").click();
  await settle();
  assert.equal(env.msg(), "내용이 비어 있어요");
  assert.deepEqual(env.ops(), []);
  env.one("ais-pp-text").typeIn("fine");
  const event = env.one("ais-pp-text").fire("keydown", { key: "s", ctrlKey: true });
  await settle();
  assert.equal(event.defaultPrevented, true);
  assert.deepEqual(env.ops(), [{ op: "update", section: "negative", id: env.ids[0], name: "ok", text: "fine" }]);
});

test("the pane's '이걸로 바꾸기' and '+ 붙이기' put in what the editors show", async () => {
  const env = await setup({ text: "base" });
  env.one("ais-pp-text").typeIn("edited, text");
  env.one("ais-pp-use").click();
  assert.deepEqual(env.sets, [["negative", "edited, text"]]);
  assert.equal(env.msg(), "네거티브 칸을 ‘Anima 기본’(으)로 바꿨어요.");
  env.one("ais-pp-add").click();
  assert.deepEqual(env.sets[1], ["negative", "edited, text, edited, text"]);
  assert.equal(env.one("ais-pp-text").value, "edited, text", "the edit is still there, unsaved");
  assert.equal(env.one("ais-pp-savedraft").disabled, false);
  env.one("ais-pp-text").typeIn(" , ");
  env.one("ais-pp-use").click();
  assert.equal(env.msg(), "내용이 비어 있어요");
  assert.equal(env.sets.length, 2);
});

test("choosing another card with unsaved edits says they were not saved", async () => {
  const env = await setup({ text: "x" });
  env.one("ais-pp-name").typeIn("changed");
  env.card("가볍게").one("ais-pp-card-main").click();
  assert.equal(env.msg(), "‘Anima 기본’을(를) 고친 내용은 저장하지 않았어요.");
  assert.equal(env.one("ais-pp-name").value, "가볍게");
  assert.equal(env.card("가볍게").classList.contains("sel"), true);
  assert.equal(env.card("가볍게").one("ais-pp-card-main").getAttribute("aria-pressed"), "true");
  env.card("글자 막기").one("ais-pp-card-main").click();
  assert.equal(env.msg(), "‘Anima 기본’을(를) 고친 내용은 저장하지 않았어요.", "a plain choice leaves the line alone");
  assert.equal(env.one("ais-pp-text").value, "text, logo");
});

test("↑ 위로 / ↓ 아래로 move the selected preset in the user's order and stop at the ends", async () => {
  const env = await setup({ text: "x" });
  assert.equal(env.one("ais-pp-up").disabled, true);
  assert.equal(env.one("ais-pp-down").disabled, false);
  env.one("ais-pp-down").click();
  await settle();
  assert.deepEqual(env.cardNames(), ["가볍게", "Anima 기본", "글자 막기"]);
  assert.equal(env.msg(), "아래로 옮겼어요.");
  env.one("ais-pp-down").click();
  await settle();
  assert.deepEqual(env.cardNames(), ["가볍게", "글자 막기", "Anima 기본"]);
  assert.equal(env.one("ais-pp-down").disabled, true);
  env.one("ais-pp-up").click();
  await settle();
  assert.equal(env.msg(), "위로 옮겼어요.");
  assert.deepEqual(env.ops().map((op) => [op.op, op.id, op.delta]), [["move", env.ids[0], 1], ["move", env.ids[0], 1], ["move", env.ids[0], -1]]);
  assert.equal(env.card("Anima 기본").classList.contains("sel"), true, "the moved preset stays selected");
  env.one("ais-pp-undo").click();
  await settle();
  assert.deepEqual(env.cardNames(), ["가볍게", "글자 막기", "Anima 기본"]);
  assert.deepEqual(env.ops().at(-1), { op: "move", section: "negative", id: env.ids[0], delta: 1 }, "the one step back");
});

test("삭제 removes the preset, shows the next one, and 되돌리기 brings it back", async () => {
  const env = await setup({ text: "x" });
  env.card("가볍게").one("ais-pp-card-main").click();
  env.one("ais-pp-delete").click();
  await settle();
  assert.deepEqual(env.ops(), [{ op: "delete", section: "negative", id: env.ids[1] }]);
  assert.deepEqual(env.cardNames(), ["Anima 기본", "글자 막기"]);
  assert.equal(env.msg(), "‘가볍게’을(를) 지웠어요.");
  assert.equal(env.one("ais-pp-name").value, "글자 막기", "the one after it");
  assert.equal(env.tab("네거티브").textContent, "네거티브2");
  env.one("ais-pp-undo").click();
  await settle();
  assert.deepEqual(env.cardNames(), ["Anima 기본", "가볍게", "글자 막기"]);
  assert.equal(env.one("ais-pp-name").value, "가볍게");
  assert.equal(env.store.list("negative")[1].id, env.ids[1], "the same preset, not a copy");
});

test("'칸 비우기' empties the box and can be undone", async () => {
  const env = await setup({ text: "some text" });
  env.one("ais-pp-clear").focus(); // a real click leaves the focus on the button
  env.one("ais-pp-clear").click();
  assert.deepEqual(env.sets, [["negative", ""]]);
  assert.equal(env.msg(), "네거티브 칸을 비웠어요.");
  assert.equal(env.one("ais-pp-status").textContent, "비어 있음");
  assert.equal(env.one("ais-pp-clear").disabled, true);
  assert.equal(env.doc.activeElement, env.win(), "the button just went grey: the focus moves to the window, not to <body>");
  env.one("ais-pp-undo").click();
  await settle();
  assert.equal(env.texts.negative, "some text");
  assert.deepEqual(env.server.ops(), [], "emptying is not a preset use");
});

test("되돌리기 takes back only its own change: what another tab saved meanwhile stays", async () => {
  const env = await setup({ text: "x" });
  env.card("가볍게").one("ais-pp-card-main").click();
  env.server.seed("negative", "다른 탭", "from another tab"); // saved elsewhere while this window is open
  env.one("ais-pp-delete").click();
  await settle();
  assert.deepEqual(env.cardNames(), ["Anima 기본", "글자 막기", "다른 탭"]);
  env.one("ais-pp-undo").click();
  await settle();
  const back = env.ops()[1];
  assert.deepEqual([back.op, back.section, back.item.id, back.item.name, back.item.text, back.index], ["insert", "negative", env.ids[1], "가볍게", LIGHT, 1]);
  assert.deepEqual(env.cardNames(), ["Anima 기본", "가볍게", "글자 막기", "다른 탭"]);
  assert.equal(env.one("ais-pp-name").value, "가볍게");

  env.texts.negative = "brand new";
  env.one("ais-pp-savenew").click();
  env.one("ais-pp-commit").click();
  await settle();
  env.server.seed("negative", "또 다른 탭", "again");
  env.one("ais-pp-undo").click();
  await settle();
  assert.deepEqual(env.cardNames(), ["Anima 기본", "가볍게", "글자 막기", "다른 탭", "또 다른 탭"]);
  assert.deepEqual(env.server.ops().map((op) => op.op), ["delete", "insert", "add", "delete"], "no change sends a whole list");
});

test("'+ 지금 칸 내용 저장' waits for the presets, so a save and its 되돌리기 never start from an unread list", async () => {
  const slow = await setup({ text: `${LIGHT}, `, load: false });
  slow.one("ais-pp-savenew").click();
  assert.equal(slow.one("ais-pp-bar-form").hidden, true, "not before the list is known");
  await settle();
  assert.equal(slow.msg(), "이미 프리셋 ‘가볍게’과 같은 내용이에요.", "the twin is found once the list is there");
  assert.deepEqual(slow.ops(), []);

  last?.close();
  const doc = installFakeDom();
  const server = fakeServer();
  server.seed("negative", "kept 1", "one");
  server.seed("negative", "kept 2", "two");
  server.fail = "503 Service Unavailable";
  const store = createPresetStore(presetApi(server.route));
  last = openPresetWindow("negative", { store, getText: () => "new one", setText() {} });
  await settle();
  const win = doc.body.querySelector(".ais-pp");
  win.one("ais-pp-savenew").click();
  await settle();
  assert.equal(win.one("ais-pp-msg").textContent, "프리셋을 못 읽었어요: 503 Service Unavailable");
  assert.equal(win.one("ais-pp-bar-form").hidden, true);
  assert.deepEqual(server.ops(), []);
  server.fail = null; // ComfyUI is back
  win.one("ais-pp-savenew").click();
  await settle();
  assert.equal(win.querySelectorAll(".ais-pp-card").length, 2, "the read was tried again");
  assert.equal(win.one("ais-pp-bar-form").hidden, false);
  win.one("ais-pp-commit").click();
  await settle();
  win.one("ais-pp-undo").click();
  await settle();
  assert.deepEqual(server.doc.sections.negative.map((p) => p.name), ["kept 1", "kept 2"]);
});

test("the name box saves what the box held when it was opened, and looks for a twin again", async () => {
  const env = await setup({ text: "my own text" });
  env.one("ais-pp-savenew").click();
  env.card("Anima 기본").one("ais-pp-replace").click(); // the box changes under the open name box
  env.one("ais-pp-newname").typeIn("my text");
  env.one("ais-pp-commit").click();
  await settle();
  assert.deepEqual(env.ops(), [{ op: "add", section: "negative", name: "my text", text: "my own text" }]);
  assert.equal(env.msg(), "이름 칸을 열 때 네거티브 칸에 있던 내용을 ‘my text’(으)로 저장했어요.");
  assert.equal(env.texts.negative, NEG);

  const twin = await setup({ text: "same text" });
  twin.one("ais-pp-savenew").click();
  await twin.store.add("negative", "먼저 저장", "same text, "); // saved meanwhile (another tab, the quick list)
  twin.one("ais-pp-commit").click();
  await settle();
  assert.equal(twin.msg(), "이미 프리셋 ‘먼저 저장’과 같은 내용이에요.");
  assert.equal(twin.ops().length, 1, "only the other save");
  assert.equal(twin.one("ais-pp-bar-form").hidden, true);
  assert.equal(twin.card("먼저 저장").classList.contains("sel"), true);
});

test("closing with unsaved edits asks first: one Esc, ✕ or outside click only says so, the next closes", async () => {
  const WARN = "‘Anima 기본’을(를) 고친 내용을 저장하지 않았어요. 한 번 더 누르면 저장하지 않고 닫혀요.";
  const env = await setup({ text: "x" });
  env.one("ais-pp-text").typeIn("a long edit");
  const esc = env.one("ais-pp-text").fire("keydown", { key: "Escape" });
  assert.ok(env.win(), "the edit is not thrown away by one key");
  assert.equal(esc.defaultPrevented, true);
  assert.equal(env.msg(), WARN);
  env.one("ais-pp-text").typeIn("a longer edit"); // typed on: it asks again
  env.one("ais-pp-icon").click();
  assert.ok(env.win());
  assert.equal(env.msg(), WARN);
  env.backdrop().fire("mousedown");
  env.backdrop().fire("click");
  assert.equal(env.win(), null, "asked once, closed on the next");
  assert.equal(env.closedCount, 1);
  assert.deepEqual(env.ops(), []);

  const other = await setup({ text: "x" });
  other.one("ais-pp-name").typeIn("renamed");
  other.backdrop().fire("mousedown");
  other.backdrop().fire("click");
  assert.ok(other.win());
  other.one("ais-pp-clear").click(); // something else was done since: it asks again
  other.one("ais-pp-icon").click();
  assert.ok(other.win());
  other.one("ais-pp-savedraft").click();
  await settle();
  other.one("ais-pp-icon").click();
  assert.equal(other.win(), null, "nothing unsaved: closes at once");

  const forced = await setup({ text: "x" });
  forced.one("ais-pp-name").typeIn("renamed");
  forced.handle.close(); // the caller's own close (another window opens) does not ask
  assert.equal(forced.win(), null);
});

test("'넣으면 닫기' leaves the window open while the editors hold unsaved edits", async () => {
  const env = await setup({ text: "x" });
  env.one("ais-pp-toggle").click();
  await settle();
  env.one("ais-pp-name").typeIn("renamed");
  env.card("가볍게").one("ais-pp-replace").click();
  assert.equal(env.texts.negative, LIGHT);
  assert.ok(env.win());
  assert.equal(env.msg(), "네거티브 칸을 ‘가볍게’(으)로 바꿨어요. ‘Anima 기본’을(를) 고친 내용은 저장하지 않아서 창은 열어 뒀어요.");
  assert.equal(env.one("ais-pp-name").value, "renamed");
});

test("closing gives the focus to what returnFocus finds when the opener left the page", async () => {
  const found = {};
  const env = await setup({ text: "x", opts: { returnFocus: () => found.label } });
  found.label = env.doc.createElement("button");
  env.doc.body.append(found.label);
  env.opener.remove(); // the panel redrew: the label that opened the window is gone
  env.one("ais-pp-icon").click();
  assert.equal(env.doc.activeElement, found.label);
});

test("a file that could not be parsed is said instead of 'no presets yet'", async () => {
  last?.close();
  const doc = installFakeDom();
  const server = fakeServer();
  server.broken = true;
  const store = createPresetStore(presetApi(server.route));
  last = openPresetWindow("negative", { store, getText: () => "", setText() {} });
  await settle();
  assert.equal(doc.body.querySelector(".ais-pp-empty").textContent,
    "프리셋 파일이 깨져서 읽지 못했어요. 새로 저장하면 원래 파일은 prompt_presets.json.bak 으로 남겨 둬요.");
});

test("Esc, the ✕ and a click on the dimmed page close it; the focus goes back where it was", async () => {
  const env = await setup({ text: "x" });
  const esc = env.one("ais-pp-search").fire("keydown", { key: "Escape" });
  assert.equal(env.backdrop(), null);
  assert.equal(esc.defaultPrevented, true);
  assert.equal(env.doc.activeElement, env.opener);
  assert.equal(env.closedCount, 1);
  await env.store.add("negative", "after close", "y"); // the closed window no longer listens
  env.handle.close();
  assert.equal(env.closedCount, 1);

  const byIcon = await setup({ text: "x" });
  byIcon.one("ais-pp-icon").click();
  assert.equal(byIcon.backdrop(), null);

  const byPage = await setup({ text: "x" });
  byPage.win().fire("mousedown");
  byPage.backdrop().fire("click"); // a drag that began inside (selecting text) and ended outside
  assert.ok(byPage.backdrop());
  byPage.backdrop().fire("mousedown");
  byPage.backdrop().fire("click");
  assert.equal(byPage.backdrop(), null);
});

test("keys typed in the window do not reach ComfyUI, and IME composing keys are left alone", async () => {
  const env = await setup({ text: "x" });
  let seen = 0;
  env.doc.addEventListener("keydown", () => { seen += 1; });
  env.one("ais-pp-text").fire("keydown", { key: "Delete" });
  env.one("ais-pp-search").fire("keydown", { key: "Enter", ctrlKey: true });
  assert.equal(seen, 0);
  env.one("ais-pp-search").fire("keydown", { key: "Escape", isComposing: true });
  assert.ok(env.win(), "Esc while composing Korean belongs to the IME");
});

test("Tab stays inside the window", async () => {
  const env = await setup({ text: "x" });
  const focusable = env.win().querySelectorAll("button, input, textarea").filter((el) => !el.disabled && el.getClientRects().length > 0);
  const lastControl = focusable[focusable.length - 1];
  lastControl.focus();
  const forward = lastControl.fire("keydown", { key: "Tab" });
  assert.equal(forward.defaultPrevented, true);
  assert.equal(env.doc.activeElement, focusable[0]);
  const back = focusable[0].fire("keydown", { key: "Tab", shiftKey: true });
  assert.equal(back.defaultPrevented, true);
  assert.equal(env.doc.activeElement, lastControl);
});

test("before the presets arrive the grid says so; a failed read says why", async () => {
  const env = await setup({ text: "x", load: false });
  assert.equal(env.one("ais-pp-empty").textContent, "읽는 중…");
  assert.equal(env.msg(), "");
  await settle();
  assert.equal(env.cardNames().length, 3);
  assert.equal(env.msg(), "네거티브 칸 프리셋 3개");

  last?.close();
  const doc = installFakeDom();
  const server = fakeServer();
  server.fail = "500 Internal Server Error";
  const store = createPresetStore(presetApi(server.route));
  last = openPresetWindow("prompt", { store, getText: () => "", setText() {} });
  await settle();
  assert.equal(doc.body.querySelector(".ais-pp-empty").textContent, "프리셋을 못 읽었어요: 500 Internal Server Error");
});

test("one window at a time, and the tabs can be limited to some boxes", async () => {
  const env = await setup({ text: "x" });
  last = openPresetWindow("prompt", { store: env.store, getText: () => "", setText() {}, sections: ["prompt", "negative"], labels: { prompt: "본문", negative: "네거" } });
  assert.equal(env.doc.body.querySelectorAll(".ais-pp-backdrop").length, 1);
  assert.equal(env.closedCount, 1, "the first window was closed");
  const win = env.doc.body.querySelector(".ais-pp");
  assert.deepEqual(win.querySelectorAll(".ais-pp-tab").map((tab) => tab.textContent), ["본문0", "네거3"]);
  assert.equal(win.querySelector("h1").textContent, "본문 프리셋");
  last.close();
  last = openPresetWindow("bogus", { store: env.store, getText: () => "", setText() {} });
  assert.equal(env.doc.body.querySelector("h1").textContent, "선행 고정 프리셋", "an unknown box opens on the first tab");
});

test("a preset saved in another tab shows up while the window is open, without touching the editors", async () => {
  const env = await setup({ text: "x" });
  env.one("ais-pp-name").typeIn("typing…");
  env.one("ais-pp-name").focus();
  await env.store.add("negative", "다른 탭", "from another tab");
  assert.deepEqual(env.cardNames(), ["Anima 기본", "가볍게", "글자 막기", "다른 탭"]);
  assert.equal(env.one("ais-pp-name").value, "typing…");
  assert.equal(env.doc.activeElement, env.one("ais-pp-name"));
});
