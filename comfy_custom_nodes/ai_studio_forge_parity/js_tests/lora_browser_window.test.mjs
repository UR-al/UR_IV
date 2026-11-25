// node:test for the LoRA picker window drawn into a tiny fake document (web/ai_studio/lora_browser.js): what stays where it
// was while the window redraws — the detail pane's scroll for the same run, and the 최근 사용 order and count the window
// opened with (the cards must not move under the pointer when something is added).
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { flush, installFakeDom } from "./fake_dom.mjs";
import { openLoraBrowser } from "../web/ai_studio/lora_browser.js";

const stemOf = (run, epoch, final) => (final ? run : `${run}-${String(epoch).padStart(6, "0")}`);
const item = (group, run, epoch, final = false) => ({
  name: `LoKr\\${group}\\${run}\\${stemOf(run, epoch, final)}.safetensors`, stem: stemOf(run, epoch, final),
  run_key: `LoKr/${group}/${run}`, epoch, final, num_epochs: 3, family: "2.9b", algo: "LoKr", group, group_label: group, run,
  run_short: run, trigger: "", date: "2026-11-01", has_meta: true, downloaded: false,
});
const GROUPS = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"];
const runOf = (group) => `${group.toLowerCase()}_2.9b`;
const keyOf = (group) => `LoKr/${group}/${runOf(group)}`;
const ITEMS = GROUPS.flatMap((group) => [item(group, runOf(group), 1), item(group, runOf(group), 2), item(group, runOf(group), 3, true)]);
const prefsWith = (extra = {}) => ({ favorites: [], recent: [], sort: "name", show_all: false, tree_open: false, close_after_add: false, ...extra });
const indexWith = (prefs) => ({ items: ITEMS, prefs: prefsWith(prefs), board_stars: {}, prompt_labels: {} });

/** Opens the window on `index` (`fresh` = what ↻ reads); resolves once the index is drawn. */
async function openWindow({ mode = "add", index, fresh } = {}) {
  installFakeDom();
  const seen = { saved: [], picked: [], details: [] };
  const handle = openLoraBrowser({
    mode, family: "2.9b",
    onPick: (pick) => { seen.picked.push(pick.name); return seen.picked.length - 1; },
    onPickMany: (list) => seen.picked.push(...list.map((pick) => pick.name)),
    api: {
      index: async (again) => (again && fresh ? fresh : index),
      detail: async (name) => { seen.details.push(name); return { summary: { has_meta: true }, epochs: [], cover: "" }; },
      thumbUrl: () => "",
      savePrefs: async (prefs) => { seen.saved.push(prefs); },
    },
  });
  await flush();
  const dialog = document.body.byClass("ais-lb")[0];
  const byKey = (key) => dialog.all((el) => el.attrs["data-k"] === key)[0];
  return {
    handle, seen, dialog, byKey,
    pane: dialog.byClass("ais-lb-dwrap")[0],
    list: dialog.byClass("ais-lb-scroll")[0],
    cards: () => dialog.byClass("ais-lb-card-main").map((card) => card.attrs["data-k"].slice(2)),
    recentCount: () => byKey("t:*recent").byClass("ais-lb-count")[0].textContent,
  };
}

/**
 * Makes the detail pane behave as a browser's at its worst: the new content is shorter than the old until its pictures load,
 * so the box is clamped back to the top when its children are replaced. Returns `{count}` = how often it was drawn.
 */
function clampOnRedraw(pane) {
  const drawn = { count: 0 };
  const replace = pane.replaceChildren.bind(pane);
  pane.replaceChildren = (...nodes) => {
    replace(...nodes);
    pane.scrollTop = 0;
    drawn.count += 1;
  };
  return drawn;
}

// ---- 11-16 model folders (D:\LoRA\<MODEL>\<algo>\<group>\<run>): tree heads, the folder filter on the same keys, card suffixes
const mItem = (model, algo, group, run, family, extra = {}) => ({
  name: `${model}\\${algo}\\${group}\\${run}\\${run}-000001.safetensors`, stem: `${run}-000001`, run_key: `${model}/${algo}/${group}/${run}`,
  epoch: 1, final: false, num_epochs: 1, family, algo: algo.toLowerCase(), model, group, group_label: group, run, run_short: run,
  trigger: "", date: "2026-11-16", has_meta: true, downloaded: false, ...extra,
});
const MODEL_ITEMS = [
  mItem("MOSS", "LoKr", "@enaa97", "@enaa97_moss", "moss"), mItem("ANIMA", "LoKr", "@enaa97", "@enaa97_2.9b_lokr", "2.9b"),
  mItem("ANIMA", "LoKr", "@enaa97", "@enaa97_3.8b", "3.8b"), mItem("SDXL", "LoRA", "HUSI", "HUSI", "sdxl"),
  mItem("ANIMA", "LoRA", "snapref-v1a", "snapref_v1a__52b", "anima"), mItem("ANIMA", "LoRA", "snapref-v1a", "snapref_v1a__40b", "anima"),
  item("Old", "old_2.9b", 1), // the old layout: no model folder
];
const modelIndex = (prefs) => ({ items: MODEL_ITEMS, prefs: prefsWith(prefs), board_stars: {}, prompt_labels: {} });
/** [key, label, count] of every tree button, in order. */
const treeNodes = (win) => win.dialog.byClass("ais-lb-tree")[0].all((el) => String(el.attrs["data-k"] || "").startsWith("t:"))
  .map((el) => [el.attrs["data-k"].slice(2), el.byClass("ais-lb-node-name")[0].textContent, el.byClass("ais-lb-count")[0].textContent]);

test("model folders: tree heads are \"<model> · <algo>\" by model then algorithm, and a click lists exactly that node's runs", async () => {
  const win = await openWindow({ index: modelIndex({ show_all: true }) });
  try {
    assert.deepEqual(treeNodes(win), [
      ["", "전체", "7"], ["*fav", "★ 즐겨찾기", "0"], ["*recent", "최근 사용", "0"],
      ["lokr", "LoKr", "1"], ["lokr/Old", "Old", "1"], // a run without a model folder: its algorithm folder, read case-blind
      ["ANIMA/lokr", "ANIMA · LoKr", "2"], ["ANIMA/lokr/@enaa97", "@enaa97", "2"],
      ["ANIMA/lora", "ANIMA · LoRA", "2"], ["ANIMA/lora/snapref-v1a", "snapref-v1a", "2"],
      ["MOSS/lokr", "MOSS · LoKr", "1"], ["MOSS/lokr/@enaa97", "@enaa97", "1"],
      ["SDXL/lora", "SDXL · LoRA", "1"], ["SDXL/lora/HUSI", "HUSI", "1"],
    ]);
    const lists = {
      "lokr": ["LoKr/Old/old_2.9b"], "ANIMA/lokr": ["ANIMA/LoKr/@enaa97/@enaa97_2.9b_lokr", "ANIMA/LoKr/@enaa97/@enaa97_3.8b"],
      "ANIMA/lora/snapref-v1a": ["ANIMA/LoRA/snapref-v1a/snapref_v1a__40b", "ANIMA/LoRA/snapref-v1a/snapref_v1a__52b"],
      "MOSS/lokr/@enaa97": ["MOSS/LoKr/@enaa97/@enaa97_moss"], "SDXL/lora": ["SDXL/LoRA/HUSI/HUSI"],
    };
    for (const [key, cards] of Object.entries(lists)) {
      win.byKey(`t:${key}`).fire("click");
      assert.deepEqual(win.cards(), cards, key);
      assert.equal(win.byKey(`t:${key}`).attrs["aria-pressed"], "true", key);
    }
    // the two snapref cards (one label) end in what tells them apart, drawn apart from the name so a cut name keeps it
    win.byKey("t:ANIMA/lora/snapref-v1a").fire("click");
    const suffixes = win.dialog.byClass("ais-lb-card-suffix").map((el) => el.textContent);
    assert.deepEqual(suffixes, [" · 40b", " · 52b"]);
    const names = win.dialog.byClass("ais-lb-card-name").map((el) => el.textContent);
    assert.deepEqual(names, ["snapref-v1a · snapref_v1a · 40b", "snapref-v1a · snapref_v1a · 52b"]);
    assert.equal(win.dialog.byClass("ais-lb-card-label")[0].textContent, "snapref-v1a · snapref_v1a");
  } finally {
    win.handle.close();
  }
});

test("model folders on a 2.9B checkpoint: MOSS / SDXL runs and their heads are hidden, ANIMA runs of unknown size stay", async () => {
  const win = await openWindow({ index: modelIndex() });
  try {
    assert.deepEqual(treeNodes(win).map(([key, , count]) => `${key}=${count}`), [
      "=4", "*fav=0", "*recent=0", "lokr=1", "lokr/Old=1", "ANIMA/lokr=1", "ANIMA/lokr/@enaa97=1", "ANIMA/lora=2", "ANIMA/lora/snapref-v1a=2",
    ]);
    const bar = win.dialog.byClass("ais-lb-bar")[0].children[0].textContent;
    assert.equal(bar, "4개 런 · 2.9B + Anima + 계열 모름 · 다른 계열 3개 숨김");
  } finally {
    win.handle.close();
  }
});

test("tree heads follow the algorithm FOLDER: runs whose header says lora_anima sit under \"ANIMA · LoRA\"; badges keep the header's", async () => {
  const HUSI = "ANIMA/LoRA/HUSI/husi_anima";
  const SNAP = "ANIMA/LoRA/snapref-v1a/snapref_v1a__40b";
  const items = [
    mItem("ANIMA", "LoRA", "snapref-v1a", "snapref_v1a__40b", "anima"),
    mItem("ANIMA", "LoRA", "HUSI", "husi_anima", "anima", { algo: "lora_anima" }), // kohya's networks.lora_anima
    mItem("MOSS", "LoRA", "HUSI", "husi_moss", "moss", { algo: "lora_anima" }),
  ];
  const win = await openWindow({ index: { items, prefs: prefsWith({ show_all: true }), board_stars: {}, prompt_labels: {} } });
  try {
    assert.deepEqual(treeNodes(win), [
      ["", "전체", "3"], ["*fav", "★ 즐겨찾기", "0"], ["*recent", "최근 사용", "0"],
      ["ANIMA/lora", "ANIMA · LoRA", "2"], ["ANIMA/lora/HUSI", "HUSI", "1"], ["ANIMA/lora/snapref-v1a", "snapref-v1a", "1"],
      ["MOSS/lora", "MOSS · LoRA", "1"], ["MOSS/lora/HUSI", "HUSI", "1"],
    ]);
    win.byKey("t:ANIMA/lora").fire("click");
    assert.deepEqual(win.cards(), [HUSI, SNAP], "the head lists the lora_anima run with the rest of its folder");
    win.byKey("t:ANIMA/lora/HUSI").fire("click");
    assert.deepEqual(win.cards(), [HUSI]);
    assert.equal(win.byKey("t:ANIMA/lora/HUSI").attrs["aria-pressed"], "true");
    const badges = (runKey) => win.byKey(`m:${runKey}`).byClass("ais-lb-badge").map((el) => el.textContent);
    assert.deepEqual(badges(HUSI), ["Anima", "lora_anima", "1에폭"]);
  } finally {
    win.handle.close();
  }
});

test("the detail pane stays where it was scrolled while the same run is drawn again; another run starts at the top", async () => {
  const win = await openWindow({ index: indexWith() });
  try {
    win.byKey(`m:${keyOf("Alpha")}`).fire("click");
    await flush(); // its detail arrived
    const drawn = clampOnRedraw(win.pane);
    win.pane.scrollTop = 92;

    win.byKey(`c:${stemOf(runOf("Alpha"), 1)}`).fire("click"); // an epoch chip
    assert.equal(drawn.count, 1, "the pane was drawn again");
    assert.equal(win.pane.scrollTop, 92);
    await flush(); // that epoch's detail arrives: drawn once more
    assert.equal(drawn.count, 2);
    assert.equal(win.pane.scrollTop, 92);
    assert.deepEqual(win.seen.details, ["alpha_2.9b", "alpha_2.9b-000001"].map((stem) => `LoKr\\Alpha\\alpha_2.9b\\${stem}.safetensors`));

    win.dialog.byClass("ais-lb-step")[1].fire("click"); // 세기 +
    win.byKey(`q:${keyOf("Alpha")}`).fire("click"); // the card's "+ 추가"
    assert.deepEqual(win.seen.picked, ["alpha_2.9b-000001"]);
    assert.equal(win.pane.scrollTop, 92);

    win.byKey(`m:${keyOf("Beta")}`).fire("click");
    assert.equal(win.pane.scrollTop, 0, "another run starts at the top");
    win.pane.scrollTop = 40;
    win.byKey(`m:${keyOf("Alpha")}`).fire("click");
    assert.equal(win.pane.scrollTop, 0, "and so does the first one when it is picked again");
  } finally {
    win.handle.close();
  }
});

test("xyz: checking an epoch chip keeps the detail pane's scroll", async () => {
  const win = await openWindow({ mode: "xyz", index: indexWith() });
  try {
    win.byKey(`m:${keyOf("Delta")}`).fire("click");
    await flush();
    const drawn = clampOnRedraw(win.pane);
    win.pane.scrollTop = 61;
    for (const epoch of [1, 2]) win.byKey(`c:${stemOf(runOf("Delta"), epoch)}`).fire("click");
    await flush();
    assert.ok(drawn.count >= 2, "every check draws the chips again");
    assert.equal(win.pane.scrollTop, 61);
    assert.equal(win.byKey(`c:${stemOf(runOf("Delta"), 2)}`).attrs["aria-pressed"], "true");
  } finally {
    win.handle.close();
  }
});

test("under 최근 사용 an add moves nothing: the cards, the list's scroll and the folder's count stay as the window opened", async () => {
  const beta = stemOf(runOf("Beta"), 2);
  const alpha = runOf("Alpha"); // its final file: what the card's button adds
  const win = await openWindow({
    index: indexWith({ sort: "recent", recent: [beta] }),
    fresh: indexWith({ sort: "recent", recent: [alpha, beta] }), // ↻ after the save: the server has the new order
  });
  try {
    const opened = win.cards();
    assert.equal(opened[0], keyOf("Beta"), "the used one first");
    assert.equal(opened[1], keyOf("Alpha"), "the rest by name");
    assert.equal(win.recentCount(), "1");
    win.list.scrollTop = 230; // one row down
    win.list.fire("scroll");

    win.byKey(`q:${keyOf("Alpha")}`).fire("click");
    assert.deepEqual(win.seen.picked, [alpha]);
    assert.equal(win.list.scrollTop, 230, "the list did not go back to the top");
    assert.deepEqual(win.cards(), opened);
    assert.equal(win.recentCount(), "1", "the tree counts what the folder shows");

    win.dialog.all((el) => String(el.attrs["aria-label"] || "").startsWith("다시 읽기"))[0].fire("click"); // ↻
    await flush();
    assert.deepEqual(win.cards(), opened, "↻ keeps the order of this window");
    assert.equal(win.recentCount(), "1");

    win.byKey("t:*recent").fire("click");
    assert.deepEqual(win.cards(), [keyOf("Beta")], "the folder lists what its count says");
  } finally {
    win.handle.close();
  }
  await flush();
  assert.deepEqual(win.seen.saved.at(-1).recent, [alpha, beta], "the pick is saved for the next open");

  const next = await openWindow({ index: indexWith({ sort: "recent", recent: [alpha, beta] }) });
  try {
    assert.deepEqual(next.cards().slice(0, 2), [keyOf("Alpha"), keyOf("Beta")], "the next open shows the new order");
    assert.equal(next.recentCount(), "2");
  } finally {
    next.handle.close();
  }
});
