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
