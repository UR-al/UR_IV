// node:test for the LoRA picker window's pure state (web/ai_studio/lora_browser_state.js). Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { cardPick, initialState, picks, quickPick, reduce, visibleRuns } from "../web/ai_studio/lora_browser_state.js";
import { listKey } from "../web/ai_studio/lora_browser_grid.js";

// Like the real index: an epoch file carries its epoch; the final file (no "-0000NN" suffix) carries the header epoch too
// (`epoch: N, final: true`), so that epoch number usually exists ONLY on the final file.
const it = (algo, group, run, epoch, extra = {}) => {
  const stem = extra.final ? run : `${run}-${String(epoch).padStart(6, "0")}`;
  return {
    name: `${algo}\\${group}\\${run}\\${stem}.safetensors`, stem, run_key: `${algo}/${group}/${run}`, epoch,
    final: false, num_epochs: 3, family: "2.9b", algo, group, group_label: group, run, run_short: run,
    trigger: "", date: "2026-11-01", has_meta: true, downloaded: false, ...extra,
  };
};
const fin = (algo, group, run, epoch, extra = {}) => it(algo, group, run, epoch, { final: true, num_epochs: epoch, ...extra });
const ITEMS = [
  it("LoKr", "Alpha", "alpha_2.9b_lokr", 1), it("LoKr", "Alpha", "alpha_2.9b_lokr", 2), fin("LoKr", "Alpha", "alpha_2.9b_lokr", 3),
  it("LoKr", "Beta", "beta_2.9b_lokr", 1, { date: "2026-11-05", trigger: "@beta" }), it("LoKr", "Beta", "beta_2.9b_lokr", 2, { date: "2026-11-05" }),
  it("LoRA", "Gamma", "gamma_base", 1, { family: "base" }),
  it("LoRA", "Delta", "delta_x", 1, { family: null }),
];
const PREFS = { favorites: [], recent: [], sort: "name", show_all: false, tree_open: [], close_after_add: false };
const make = (o = {}) =>
  initialState({ mode: "add", family: "2.9b", index: { items: ITEMS, prefs: PREFS, board_stars: {}, prompt_labels: {} }, prefs: PREFS, ...o });
const keys = (s) => visibleRuns(s).map((r) => r.runKey);
const A = "LoKr/Alpha/alpha_2.9b_lokr";
const B = "LoKr/Beta/beta_2.9b_lokr";

test("groups epoch files into one run", () => {
  const alpha = visibleRuns(make()).find((r) => r.runKey === A);
  assert.deepEqual(alpha.epochs.map((e) => [e.epoch, e.final]), [[1, false], [2, false], [3, true]]);
  assert.equal(alpha.groupLabel, "Alpha");
});

test("quickPick uses the selected epoch for the selected run and the final epoch otherwise", () => {
  let s = make();
  assert.equal(quickPick(s, A).name, "alpha_2.9b_lokr");
  s = reduce(s, { type: "select", runKey: A });
  s = reduce(s, { type: "epoch", value: 1 });
  assert.equal(quickPick(s, A).name, "alpha_2.9b_lokr-000001");
  assert.equal(quickPick(s, B).name, "beta_2.9b_lokr-000002");
  assert.equal(quickPick(s, A).strength, 1);
});

test("strengthStep clamps to 0..2 in 0.05 steps", () => {
  let s = make();
  s = reduce(s, { type: "strengthStep", delta: 0.05 });
  assert.equal(s.strength, 1.05);
  s = reduce(s, { type: "strength", value: 1.999 });
  assert.equal(s.strength, 2);
  s = reduce(s, { type: "strengthStep", delta: 0.05 });
  assert.equal(s.strength, 2);
  s = reduce(s, { type: "strength", value: -3 });
  assert.equal(s.strength, 0);
  s = reduce(s, { type: "strengthStep", delta: -0.05 });
  assert.equal(s.strength, 0);
  s = reduce(s, { type: "strength", value: 0.333 });
  assert.equal(s.strength, 0.33);
});

test("added then unadd leaves the rest in order", () => {
  let s = make();
  for (const k of ["a0", "a1", "a2"]) s = reduce(s, { type: "added", pick: { key: k, name: k, strength: 1, label: k } });
  s = reduce(s, { type: "unadd", key: "a1" });
  assert.deepEqual(s.added.map((p) => p.key), ["a0", "a2"]);
});

test("closeAfter starts from prefs.close_after_add", () => {
  assert.equal(make().closeAfter, false);
  const s = make({ prefs: { ...PREFS, close_after_add: true } });
  assert.equal(s.closeAfter, true);
  assert.equal(reduce(s, { type: "closeAfter", on: false }).closeAfter, false);
});

test("replace mode starts on the current file's run and epoch", () => {
  const s = make({ mode: "replace", current: "alpha_2.9b_lokr-000002" });
  assert.equal(s.selected, A);
  assert.deepEqual(picks(s), [{ name: "alpha_2.9b_lokr-000002", strength: 1 }]);
});

test("family filter hides base runs on 2.9b until showAll", () => {
  let s = make();
  assert.ok(!keys(s).includes("LoRA/Gamma/gamma_base"));
  s = reduce(s, { type: "showAll", on: true });
  assert.ok(keys(s).includes("LoRA/Gamma/gamma_base"));
});

test("unknown family stays visible", () => {
  assert.ok(keys(make()).includes("LoRA/Delta/delta_x"));
});

test("query narrows and keeps selection when still visible", () => {
  let s = reduce(make(), { type: "select", runKey: B });
  s = reduce(s, { type: "query", text: "beta" });
  assert.deepEqual(keys(s), [B]);
  assert.equal(s.selected, B);
  s = reduce(s, { type: "query", text: "alpha" });
  assert.equal(s.selected, A);
  s = reduce(s, { type: "query", text: "@beta" }); // trigger on any epoch file shows the run (R4)
  assert.deepEqual(keys(s), [B]);
  s = reduce(s, { type: "query", text: "alpha 1" }); // an epoch token moves the selected epoch
  assert.equal(picks(s)[0].name, "alpha_2.9b_lokr-000001");
});

test("arrow moves in a grid of cols", () => {
  let s = reduce(make(), { type: "showAll", on: true });
  const all = keys(s);
  s = reduce(s, { type: "select", runKey: all[0] });
  s = reduce(s, { type: "move", dx: 1, dy: 0, cols: 2 });
  assert.equal(s.selected, all[1]);
  s = reduce(s, { type: "move", dx: 0, dy: 1, cols: 2 });
  assert.equal(s.selected, all[3]);
  s = reduce(s, { type: "move", dx: 0, dy: -1, cols: 2 });
  assert.equal(s.selected, all[1]);
  s = reduce(s, { type: "move", dx: -1, dy: 0, cols: 2 });
  s = reduce(s, { type: "move", dx: -1, dy: 0, cols: 2 });
  assert.equal(s.selected, all[0]);
  s = reduce(s, { type: "move", dx: 0, dy: 9, cols: 2 });
  assert.equal(s.selected, all[all.length - 1]);
});

test("[ and ] step epochs and stop at the ends", () => {
  let s = reduce(make(), { type: "select", runKey: A });
  assert.equal(picks(s)[0].name, "alpha_2.9b_lokr"); // starts on the final file
  s = reduce(s, { type: "epoch", delta: 1 });
  assert.equal(picks(s)[0].name, "alpha_2.9b_lokr");
  for (let i = 0; i < 4; i++) s = reduce(s, { type: "epoch", delta: -1 });
  assert.equal(picks(s)[0].name, "alpha_2.9b_lokr-000001");
});

test("xyz picks are the checked epochs in order", () => {
  let s = reduce(make({ mode: "xyz" }), { type: "select", runKey: A });
  s = reduce(s, { type: "toggleEpoch", value: 2 });
  s = reduce(s, { type: "toggleEpoch", value: 1 });
  s = reduce(s, { type: "toggleEpoch", value: 3 }); // no such epoch
  assert.deepEqual(picks(s), [
    { name: "alpha_2.9b_lokr-000001", strength: null },
    { name: "alpha_2.9b_lokr-000002", strength: null },
  ]);
  s = reduce(s, { type: "toggleEpoch", value: 1 });
  assert.deepEqual(picks(s).map((p) => p.name), ["alpha_2.9b_lokr-000002"]);
});

test("favorites and recent folders", () => {
  let s = make({ prefs: { ...PREFS, recent: ["beta_2.9b_lokr-000002"] } });
  s = reduce(s, { type: "favorite", runKey: A });
  assert.deepEqual(s.favorites, [A]);
  assert.deepEqual(keys(reduce(s, { type: "folder", key: "*fav" })), [A]);
  assert.deepEqual(keys(reduce(s, { type: "folder", key: "*recent" })), [B]);
  assert.deepEqual(keys(reduce(s, { type: "folder", key: "LoKr/Beta" })), [B]);
  assert.equal(keys(reduce(s, { type: "folder", key: "LoKr" })).length, 2);
  assert.equal(visibleRuns(s).find((r) => r.runKey === A).favorite, true);
  s = reduce(s, { type: "favorite", runKey: A });
  assert.deepEqual(s.favorites, []);
});

test("sort by date puts the newest run first", () => {
  assert.equal(keys(reduce(make(), { type: "sort", by: "date" }))[0], B);
});

test("recent holds file stems: any epoch stem counts and the best rank sorts first", () => {
  const s = make({ prefs: { ...PREFS, recent: ["alpha_2.9b_lokr-000001", "beta_2.9b_lokr-000001"] } });
  assert.deepEqual(keys(reduce(s, { type: "folder", key: "*recent" })), [A, B]);
  assert.equal(keys(reduce(s, { type: "sort", by: "recent" }))[0], A);
  const t = make({ prefs: { ...PREFS, recent: ["beta_2.9b_lokr-000001"] } });
  assert.equal(keys(reduce(t, { type: "sort", by: "recent" }))[0], B);
});

test("replace mode shows the current run even when its family is hidden", () => {
  const s = make({ mode: "replace", current: "gamma_base-000001" });
  assert.equal(s.showAll, true);
  assert.equal(s.selected, "LoRA/Gamma/gamma_base");
  assert.ok(keys(s).includes("LoRA/Gamma/gamma_base"));
});

// ---- epoch numbers in the search over real-index-like runs (final-review items 1, 2, 4)
const GROUP = "묵밥 (@Bjunbae)";
const RUN = "@bjunbae_2.9b_lokr_came_b10_2e06";
const RUN16 = `${RUN}_fp16`;
const BJ = `LoKr/${GROUP}/${RUN}`;
const BJ16 = `LoKr/${GROUP}/${RUN16}`;
const bj = (run, epoch, last, final = false) => it("LoKr", GROUP, run, epoch, { trigger: "@bjunbae", num_epochs: last, final });
const REAL = [
  bj(RUN, 13, 15), bj(RUN, 14, 15), bj(RUN, 15, 15, true), // e15 exists only as the final file
  bj(RUN16, 14, 20), bj(RUN16, 19, 20), bj(RUN16, 20, 20, true),
  ...ITEMS,
];
const real = (o = {}) => initialState({ mode: "add", family: "2.9b", index: { items: REAL, prefs: PREFS }, prefs: PREFS, ...o });

test("\"run 15\" finds a run whose e15 exists only as the final file, selects it, and quickPick gives that final stem", () => {
  let s = reduce(real(), { type: "query", text: "bjunbae 15" });
  assert.deepEqual(keys(s), [BJ]);
  assert.equal(s.selected, BJ);
  assert.equal(quickPick(s, BJ).name, RUN);
  assert.match(quickPick(s, BJ).label, /최종 e15$/);
  assert.deepEqual(picks(s), [{ name: RUN, strength: 1 }]);
  s = reduce(make(), { type: "query", text: "alpha e3" }); // the fixture's alpha: e3 = its final file only
  assert.deepEqual(keys(s), [A]);
  assert.equal(quickPick(s, A).name, "alpha_2.9b_lokr");
});

test("an epoch token prefers the numbered file when the final file has the same epoch", () => {
  const items = [it("LoKr", "Zeta", "zeta_2.9b", 3), fin("LoKr", "Zeta", "zeta_2.9b", 3)];
  const s = reduce(initialState({ mode: "add", family: "2.9b", index: { items, prefs: PREFS } }), { type: "query", text: "zeta 3" });
  assert.equal(quickPick(s, "LoKr/Zeta/zeta_2.9b").name, "zeta_2.9b-000003");
});

test("under \"bjun 14\" the unselected card's button also adds e14, not its final epoch", () => {
  const s = reduce(real(), { type: "query", text: "bjun 14" });
  assert.deepEqual(new Set(keys(s)), new Set([BJ, BJ16]));
  const other = keys(s).find((k) => k !== s.selected);
  assert.ok(other);
  for (const runKey of [BJ, BJ16]) {
    assert.match(quickPick(s, runKey).name, /-000014$/, runKey);
    assert.match(quickPick(s, runKey).label, / · e14$/, runKey);
    assert.deepEqual(cardPick(s, runKey), quickPick(s, runKey), runKey); // add mode: the card button = quickPick
  }
});

test("the first card is selected after a search, in add and in xyz mode", () => {
  for (const mode of ["add", "xyz"]) {
    const s0 = real({ mode });
    assert.equal(s0.selected, null, mode);
    const s = reduce(s0, { type: "query", text: "bjun 14" });
    assert.equal(s.selected, keys(s)[0], mode);
    assert.equal(reduce(s0, { type: "query", text: "" }).selected, null, `${mode}: an empty query selects nothing`);
  }
  const x = reduce(real({ mode: "xyz" }), { type: "query", text: "bjun 14" });
  const other = keys(x).find((k) => k !== x.selected);
  assert.match(cardPick(x, other).name, /-000014$/); // the xyz card's check button checks the searched epoch
  assert.equal(cardPick(x, other).strength, null);
  assert.equal(cardPick(reduce(x, { type: "query", text: "" }), other).name, other === BJ ? RUN : RUN16); // its last epoch again
});

test("clearing the epoch token restores the epochs the query moved, but keeps an epoch the user picked meanwhile", () => {
  let s = reduce(real(), { type: "select", runKey: BJ });
  s = reduce(s, { type: "epoch", value: 13 }); // the user's own choice before searching
  s = reduce(s, { type: "query", text: "bjun 14" });
  assert.equal(picks(s)[0].name, `${RUN}-000014`);
  s = reduce(s, { type: "query", text: "bjun 1" }); // still typing: no e1 anywhere, the e14 moves are undone
  s = reduce(s, { type: "query", text: "bjun 14" });
  s = reduce(s, { type: "select", runKey: BJ16 });
  s = reduce(s, { type: "epoch", delta: 1 }); // e14 -> e19 by hand during the search
  s = reduce(s, { type: "query", text: "bjun" }); // the epoch token is gone
  assert.equal(picks(s)[0].name, `${RUN16}-000019`, "the user's own move stays");
  assert.equal(quickPick(s, BJ).name, RUN, "an unselected card is back on its last epoch");
  s = reduce(s, { type: "select", runKey: BJ });
  assert.equal(picks(s)[0].name, `${RUN}-000013`, "the epoch chosen before the search is back");
  s = reduce(reduce(s, { type: "query", text: "14" }), { type: "query", text: "" });
  assert.equal(picks(s)[0].name, `${RUN}-000013`);
});

test("a query typed before the index arrived is applied to the first state", () => {
  const s = real({ query: "bjun 14" });
  assert.equal(s.query, "bjun 14");
  assert.deepEqual(new Set(keys(s)), new Set([BJ, BJ16]));
  assert.equal(s.selected, keys(s)[0]);
  assert.match(quickPick(s, s.selected).name, /-000014$/);
  const r = real({ mode: "replace", current: `${RUN}-000013`, query: "bjun" });
  assert.equal(r.selected, BJ);
  assert.equal(picks(r)[0].name, `${RUN}-000013`);
});

// ---- 최근 사용 while the window is open: the order it opened with (a pick must not move the cards under the pointer)
test("a pick is remembered for the next open, but the 최근 사용 order and folder stay as the window opened", () => {
  const prefs = { ...PREFS, sort: "recent", recent: ["beta_2.9b_lokr-000001"] };
  let s = make({ prefs });
  const before = keys(s);
  assert.equal(before[0], B);
  const key = listKey(s, visibleRuns(s));
  s = reduce(s, { type: "remember", stems: ["alpha_2.9b_lokr-000002"] });
  assert.deepEqual(s.recent, ["alpha_2.9b_lokr-000002", "beta_2.9b_lokr-000001"], "what the prefs save");
  assert.deepEqual(keys(s), before, "the cards stay where they are");
  assert.equal(listKey(s, visibleRuns(s)), key, "so the grid does not go back to the top");
  assert.deepEqual(keys(reduce(s, { type: "folder", key: "*recent" })), [B], "the 최근 사용 folder too");
  const next = make({ prefs: { ...prefs, recent: s.recent } }); // the window opened again
  assert.equal(keys(next)[0], A);
  assert.deepEqual(keys(reduce(next, { type: "folder", key: "*recent" })), [A, B]);
});

test("remember puts the picked stems first, once each, and keeps at most 20", () => {
  let s = make({ prefs: { ...PREFS, recent: ["a", "b", "c"] } });
  s = reduce(s, { type: "remember", stems: ["c", "x"] });
  assert.deepEqual(s.recent, ["c", "x", "a", "b"]);
  s = reduce(s, { type: "remember", stems: Array.from({ length: 25 }, (_, i) => `n${i}`) });
  assert.equal(s.recent.length, 20);
  assert.equal(s.recent[0], "n0");
});
