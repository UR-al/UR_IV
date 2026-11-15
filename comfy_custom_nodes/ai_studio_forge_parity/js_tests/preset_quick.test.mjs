// node:test with a fake document (preset_fakes.mjs): the quick preset list under a prompt box's label
// (web/ai_studio/preset_quick.js). Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { createPresetStore, presetApi } from "../web/ai_studio/preset_api.js";
import { openPresetQuick } from "../web/ai_studio/preset_quick.js";
import { fakeServer, hold, installFakeDom, settle, tick } from "./preset_fakes.mjs";

const NEG = "worst quality, low quality, lowres";

let last = null; // the list a test left open: the module shows one at a time, and the same box's label is a toggle

function setup({ presets = [["Anima 기본", NEG], ["가볍게", "worst quality,\nblurry"]], text = "" } = {}) {
  last?.close();
  const doc = installFakeDom();
  const server = fakeServer();
  const ids = presets.map(([name, value]) => server.seed("negative", name, value));
  const store = createPresetStore(presetApi(server.route));
  const anchor = doc.createElement("button");
  anchor.rect = { left: 100, top: 200, right: 160, bottom: 220, width: 60, height: 20 };
  doc.body.append(anchor);
  const texts = { negative: text };
  const env = { doc, server, store, anchor, texts, ids, notices: [], managed: [], sets: [] };
  env.open = (key = "negative", extra = {}) => {
    const handle = openPresetQuick(anchor, key, {
      store,
      getText: (k) => texts[k] ?? "",
      setText: (k, value) => { env.sets.push([k, value]); texts[k] = value; },
      onManage: (k) => env.managed.push(k),
      onNotice: (message) => env.notices.push(message),
      ...extra,
    });
    last = handle || last;
    return handle;
  };
  env.popup = () => doc.body.querySelector(".ais-pq");
  env.rows = () => env.popup().querySelectorAll(".ais-pq-row");
  env.note = () => env.popup().one("ais-pq-note");
  return env;
}

test("opening adds the stylesheet once and a labelled popup under the label", async () => {
  const env = setup();
  const handle = env.open();
  await settle();
  const links = env.doc.head.all((el) => el.tag === "link");
  assert.equal(links.length, 1);
  assert.equal(links[0].id, "aistudio-preset-css");
  assert.match(links[0].href, /\/web\/ai_studio\/preset\.css$/);
  const popup = env.popup();
  assert.equal(handle.element, popup);
  assert.equal(popup.parent, env.doc.body, "outside the canvas transform");
  assert.equal(popup.getAttribute("role"), "dialog");
  assert.equal(popup.getAttribute("aria-label"), "네거티브 프리셋 목록");
  assert.equal(popup.one("ais-pq-title").textContent, "네거티브 프리셋");
  assert.equal(popup.one("ais-pq-hint").textContent, "누르면 바꾸기 · + 는 뒤에 붙이기");
  assert.equal(env.anchor.getAttribute("aria-expanded"), "true");
  assert.equal(popup.style.left, "100px");
  assert.equal(popup.style.top, "224px");
  handle.close();
  env.open("prompt");
  assert.equal(env.doc.head.all((el) => el.tag === "link").length, 1, "the stylesheet is added once");
});

test("rows show the name and a one-line preview, with the dot on the preset the box holds", async () => {
  const env = setup({ text: ` ${NEG},\n` });
  env.open();
  assert.equal(env.popup().one("ais-pq-empty").textContent, "읽는 중…");
  await settle();
  const rows = env.rows();
  assert.deepEqual(rows.map((row) => row.one("ais-pq-name").textContent), ["Anima 기본", "가볍게"]);
  assert.deepEqual(rows.map((row) => row.one("ais-pq-text").textContent), [NEG, "worst quality, blurry"]);
  assert.deepEqual(rows.map((row) => row.one("ais-pq-dot").classList.contains("on")), [true, false]);
  assert.equal(rows[1].one("ais-pq-add").getAttribute("aria-label"), "가볍게 뒤에 붙이기");
  assert.equal(rows[1].one("ais-pq-pick").title, "worst quality,\nblurry", "the whole text on hover");
});

test("clicking a row replaces the box with the preset's text as written, stamps its use and closes", async () => {
  const env = setup({ text: "my own negative" });
  env.open();
  await settle();
  env.rows()[1].one("ais-pq-pick").click();
  assert.deepEqual(env.sets, [["negative", "worst quality,\nblurry"]]);
  assert.deepEqual(env.notices, ["네거티브 칸을 ‘가볍게’(으)로 바꿨어요."]);
  assert.equal(env.popup(), null);
  assert.equal(env.anchor.getAttribute("aria-expanded"), "false");
  assert.equal(env.doc.listenerCount(), 0, "no listener left on the document");
  await settle();
  assert.deepEqual(env.server.ops(), [{ op: "use", section: "negative", id: env.ids[1] }]);
});

test("+ on a row appends after the box's text with ', ' and closes", async () => {
  const env = setup({ text: "bad hands, \n" });
  env.open();
  await settle();
  env.rows()[1].one("ais-pq-add").click();
  assert.deepEqual(env.sets, [["negative", "bad hands, worst quality,\nblurry"]]);
  assert.deepEqual(env.notices, ["‘가볍게’을(를) 네거티브 칸 뒤에 붙였어요."]);
  assert.equal(env.popup(), null);
  await settle();
  assert.deepEqual(env.server.ops(), [{ op: "use", section: "negative", id: env.ids[1] }]);
});

test("'+ 지금 내용 저장' saves the box under the next '내 <칸> N' name and keeps the list open", async () => {
  const env = setup({ text: "text, logo, " });
  env.open();
  await settle();
  env.popup().one("ais-pq-save").click();
  await settle();
  assert.deepEqual(env.server.ops(), [{ op: "add", section: "negative", name: "내 네거티브 3", text: "text, logo, " }]);
  const rows = env.rows();
  assert.equal(rows.length, 3);
  assert.equal(rows[2].one("ais-pq-name").textContent, "내 네거티브 3");
  assert.equal(rows[2].one("ais-pq-dot").classList.contains("on"), true, "the box now equals the new preset");
  assert.equal(env.note().textContent, "지금 네거티브 칸을 ‘내 네거티브 3’(으)로 저장했어요. 이름은 ‘프리셋 관리…’에서 바꿔요.");
  assert.equal(env.note().hidden, false);
  assert.deepEqual(env.sets, [], "saving does not touch the box");
});

test("'+ 지금 내용 저장' says why when there is nothing to save, the box is already a preset, or the server refuses", async () => {
  const env = setup({ text: " ,\n" });
  env.open();
  await settle();
  const save = () => { env.popup().one("ais-pq-save").click(); return settle(); };
  assert.equal(env.note().hidden, true);
  await save();
  assert.equal(env.note().textContent, "네거티브 칸이 비어 있어서 저장할 게 없어요.");
  env.texts.negative = `${NEG}, `;
  await save();
  assert.equal(env.note().textContent, "이미 프리셋 ‘Anima 기본’과 같은 내용이에요.");
  assert.deepEqual(env.server.ops(), []);
  env.texts.negative = "new text";
  env.server.fail = "한 칸에 프리셋은 200개까지예요";
  await save();
  assert.equal(env.note().textContent, "저장하지 못했어요: 한 칸에 프리셋은 200개까지예요");
  assert.equal(env.rows().length, 2);
});

test("'프리셋 관리…' closes the list and asks for the window of that box", async () => {
  const env = setup();
  env.open();
  await settle();
  env.popup().one("ais-pq-manage").click();
  assert.equal(env.popup(), null);
  assert.deepEqual(env.managed, ["negative"]);
});

test("Esc and a press outside close it; a press inside or on the label does not", async () => {
  const env = setup();
  env.open();
  await settle();
  env.popup().one("ais-pq-title").fire("pointerdown");
  env.anchor.fire("pointerdown"); // the label's own click toggles it (next test)
  assert.ok(env.popup());
  env.doc.body.fire("pointerdown");
  assert.equal(env.popup(), null);

  env.open();
  const esc = env.popup().fire("keydown", { key: "Escape" });
  assert.equal(env.popup(), null);
  assert.equal(esc.defaultPrevented, true);

  env.open();
  env.doc.body.fire("keydown", { key: "Escape" }); // the focus is somewhere else
  assert.equal(env.popup(), null);

  env.open();
  env.doc.body.fire("wheel"); // the canvas under it zooms or pans: the label moves away
  assert.equal(env.popup(), null);
  assert.equal(env.doc.listenerCount(), 0);
});

test("keys typed in the list do not reach ComfyUI's shortcuts", async () => {
  const env = setup();
  let seen = 0;
  env.doc.addEventListener("keydown", () => { seen += 1; });
  env.open();
  await settle();
  env.popup().fire("keydown", { key: "Delete" });
  env.rows()[0].one("ais-pq-pick").fire("keydown", { key: "z", ctrlKey: true });
  assert.equal(seen, 0);
  assert.ok(env.popup());
});

test("up / down move between the rows", async () => {
  const env = setup();
  env.open();
  await settle();
  const picks = () => env.rows().map((row) => row.one("ais-pq-pick"));
  env.popup().fire("keydown", { key: "ArrowDown" });
  assert.equal(env.doc.activeElement, picks()[0]);
  picks()[0].fire("keydown", { key: "ArrowDown" });
  assert.equal(env.doc.activeElement, picks()[1]);
  picks()[1].fire("keydown", { key: "ArrowDown" });
  assert.equal(env.doc.activeElement, picks()[1], "stops at the last row");
  picks()[1].fire("keydown", { key: "ArrowUp" });
  assert.equal(env.doc.activeElement, picks()[0]);
});

test("asking for the open box again closes it; another box takes its place", async () => {
  const env = setup();
  const first = env.open("negative");
  assert.ok(first);
  assert.equal(env.open("negative"), null, "the label works as a toggle");
  assert.equal(env.popup(), null);
  env.open("negative");
  const other = env.open("prompt");
  assert.ok(other);
  assert.equal(env.doc.body.querySelectorAll(".ais-pq").length, 1);
  assert.equal(env.popup().one("ais-pq-title").textContent, "프롬프트 프리셋");
});

test("the popup stays inside the window: clamped at the right edge, above the label when there is no room below", async () => {
  const env = setup();
  env.anchor.rect = { left: 1200, top: 700, right: 1260, bottom: 720, width: 60, height: 20 };
  const handle = env.open();
  handle.element.offsetWidth = 330;
  handle.element.offsetHeight = 300;
  handle.place();
  assert.equal(handle.element.style.left, `${1280 - 330 - 8}px`);
  assert.equal(handle.element.style.top, `${700 - 300 - 4}px`);
  env.anchor.rect = { left: -50, top: 10, right: 10, bottom: 30, width: 60, height: 20 };
  handle.place();
  assert.equal(handle.element.style.left, "8px");
  assert.equal(handle.element.style.top, "34px");
});

test("a box with no presets says how to make one; a failed read says why", async () => {
  const env = setup({ presets: [] });
  env.open();
  await settle();
  assert.equal(env.popup().one("ais-pq-empty").textContent, "아직 프리셋이 없어요. 아래 ‘+ 지금 내용 저장’으로 만들어요.");
  env.popup().one("ais-pq-manage").click();

  const broken = setup();
  broken.server.fail = "500 Internal Server Error";
  broken.open();
  await settle();
  assert.equal(broken.popup().one("ais-pq-empty").textContent, "프리셋을 못 읽었어요: 500 Internal Server Error");
});

test("the list follows the store: a preset saved elsewhere shows up while it is open", async () => {
  const env = setup();
  env.open();
  await settle();
  await env.store.add("negative", "글자 막기", "text, logo");
  assert.deepEqual(env.rows().map((row) => row.one("ais-pq-name").textContent), ["Anima 기본", "가볍게", "글자 막기"]);
});

test("each open reads the file again, so another tab's presets are there", async () => {
  const env = setup();
  env.open();
  await settle();
  env.popup().one("ais-pq-manage").click();
  env.server.seed("negative", "다른 탭", "from another tab");
  env.open();
  assert.equal(env.rows().length, 2, "the known presets show at once");
  await settle();
  assert.equal(env.rows().length, 3);
});

test("after a preset replaced the box, the list offers to take that back while the box is unchanged", async () => {
  const env = setup({ text: "my long hand-written negative" });
  env.open();
  await settle();
  assert.equal(env.popup().querySelectorAll(".ais-pq-undo").length, 0);
  env.rows()[0].one("ais-pq-pick").click();
  env.open();
  await settle();
  const undo = env.popup().one("ais-pq-undo");
  assert.equal(undo.textContent, "되돌리기");
  undo.click();
  assert.equal(env.texts.negative, "my long hand-written negative");
  assert.equal(env.note().textContent, "되돌렸어요.");
  assert.equal(env.notices.at(-1), "네거티브 칸을 되돌렸어요.");
  assert.equal(env.popup().querySelectorAll(".ais-pq-undo").length, 0);
  assert.ok(env.popup(), "the list stays open");

  env.rows()[0].one("ais-pq-pick").click();
  env.texts.negative += ", typed more"; // the user went on writing: the old text is no longer one click away
  env.open();
  await settle();
  assert.equal(env.popup().querySelectorAll(".ais-pq-undo").length, 0);
});

test("a slow read that ends after the list closed changes nothing", async () => {
  const env = setup();
  const open = hold(env.server);
  const handle = env.open();
  handle.close();
  open();
  await settle();
  assert.equal(env.popup(), null);
});

test("'+ 지금 내용 저장' pressed again while the first save is on its way saves once", async () => {
  const env = setup({ text: "text, logo" });
  env.open();
  await settle();
  const open = hold(env.server); // the add's answer is slow
  const save = env.popup().one("ais-pq-save");
  save.click();
  await tick();
  assert.equal(save.getAttribute("aria-disabled"), "true");
  save.click();
  await tick();
  open();
  await settle();
  assert.deepEqual(env.server.ops(), [{ op: "add", section: "negative", name: "내 네거티브 3", text: "text, logo" }]);
  assert.equal(env.rows().length, 3);
  assert.equal(save.getAttribute("aria-disabled"), "false");
  env.texts.negative = "another text";
  save.click();
  await settle();
  assert.equal(env.rows().length, 4, "the button works again once the save has settled");
});

test("a file that could not be parsed is said instead of 'no presets yet'", async () => {
  const env = setup({ presets: [] });
  env.server.broken = true;
  env.open();
  await settle();
  assert.equal(env.popup().one("ais-pq-empty").textContent, "프리셋 파일이 깨져서 읽지 못했어요. 새로 저장하면 원래 파일은 .bak 으로 남겨 둬요.");
});

// ---- a panel that redraws: every render replaces its labels (panel_view.js render / promptRow)
function panelSetup({ text = "my hand-written negative" } = {}) {
  last?.close();
  const doc = installFakeDom();
  const server = fakeServer();
  server.seed("negative", "Anima 기본", NEG);
  const store = createPresetStore(presetApi(server.route));
  const panel = doc.createElement("div");
  doc.body.append(panel);
  const texts = { negative: text };
  const env = { doc, server, store, panel, texts, notices: [] };
  env.label = () => panel.querySelector('.ais-box-label[data-section="negative"]');
  env.open = (anchor) => {
    const handle = openPresetQuick(anchor, "negative", {
      store,
      getText: (k) => texts[k] ?? "",
      setText: (k, value) => { texts[k] = value; env.render(); }, // the panel's set(..., true): a full redraw
      onNotice: (message) => env.notices.push(message),
    });
    last = handle || last;
    return handle;
  };
  env.render = () => {
    const scroll = doc.createElement("div");
    for (const [key, top] of [["prompt", 200], ["negative", 300]]) {
      const label = doc.createElement("button");
      label.className = "ais-box-label";
      label.setAttribute("data-section", key);
      label.rect = { left: 100, top, right: 160, bottom: top + 20, width: 60, height: 20 };
      if (key === "negative") label.addEventListener("click", () => env.open(label)); // ctx.boxLabel(key, thatButton)
      scroll.append(label);
    }
    panel.replaceChildren(scroll);
  };
  env.popup = () => doc.body.querySelector(".ais-pq");
  env.render();
  return env;
}

test("a panel that redraws its labels: a pick gives the focus to the label that is on the page now", async () => {
  const env = panelSetup();
  const first = env.label();
  first.click();
  await settle();
  assert.equal(env.popup().style.top, "324px");
  env.popup().one("ais-pq-pick").click(); // setText -> the panel redraws -> the list closes
  assert.equal(env.popup(), null);
  assert.notEqual(env.label(), first);
  assert.equal(first.isConnected, false);
  assert.equal(env.doc.activeElement, env.label());
});

test("a panel that redraws while the list is open: the list stays under the new label, which still toggles it", async () => {
  const env = panelSetup();
  env.label().click();
  await settle();
  env.popup().one("ais-pq-pick").click();
  env.label().click(); // again: the list offers 되돌리기
  await settle();
  const before = env.label();
  env.popup().one("ais-pq-undo").click(); // setText -> the panel redraws under the open list
  assert.equal(env.texts.negative, "my hand-written negative");
  assert.notEqual(env.label(), before);
  assert.ok(env.popup(), "the list stays open");
  assert.equal(env.popup().style.left, "100px", "under the label, not in the window's corner");
  assert.equal(env.popup().style.top, "324px");
  assert.equal(env.label().getAttribute("aria-expanded"), "true");
  env.label().fire("pointerdown"); // the new label is the list's own: a press on it is not 'outside'
  assert.ok(env.popup());
  env.label().click();
  assert.equal(env.popup(), null, "the new label closes the list it opened");
  assert.equal(env.label().getAttribute("aria-expanded"), "false");
  assert.equal(env.doc.body.querySelectorAll(".ais-pq").length, 0);
});

test("a redraw from a store subscriber (the labels' marks) keeps the open list in place", async () => {
  const env = panelSetup({ text: "text, logo" });
  env.store.subscribe(() => env.render()); // the integrator redraws the panel after every read and change
  env.label().click();
  await settle(); // the forced read answers: the panel redraws
  assert.equal(env.popup().style.top, "324px");
  env.popup().one("ais-pq-save").click();
  await settle();
  assert.equal(env.popup().style.left, "100px");
  assert.equal(env.popup().style.top, "324px");
  assert.equal(env.popup().querySelectorAll(".ais-pq-row").length, 2);
  env.label().click();
  assert.equal(env.popup(), null);
});

test("the label can be given as a function; a label that is gone leaves the list where it was", async () => {
  const env = panelSetup();
  const handle = env.open(() => env.label());
  await settle();
  assert.equal(env.popup().style.top, "324px");
  env.render();
  env.label().rect = { left: 40, top: 500, right: 100, bottom: 520, width: 60, height: 20 };
  handle.place();
  assert.equal(env.popup().style.left, "40px");
  assert.equal(env.popup().style.top, "524px");
  env.label().fire("pointerdown");
  assert.ok(env.popup());
  env.panel.replaceChildren(); // the labels are gone (the card is hidden)
  handle.place();
  assert.equal(env.popup().style.top, "524px");
  handle.close();
  assert.equal(env.popup(), null);
});
