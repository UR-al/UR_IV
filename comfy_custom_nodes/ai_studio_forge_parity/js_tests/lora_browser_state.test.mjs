// node:test for the LoRA picker window's pure state (web/ai_studio/lora_browser_state.js). Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { cardPick, folderKeys, initialState, labelSuffixes, picks, quickPick, reduce, visibleRuns } from "../web/ai_studio/lora_browser_state.js";
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
  assert.deepEqual(keys(reduce(s, { type: "folder", key: "lokr/Beta" })), [B]); // the algorithm folder "LoKr", read case-blind
  assert.equal(keys(reduce(s, { type: "folder", key: "lokr" })).length, 2);
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

// ---- 11-16 D:\LoRA\<MODEL>\<algo>\<group>\<run> (review items 1-4): tree keys, model-folder families, search, card labels.
// Like the real index: `run_key` keeps the folders' spelling, `algo` is the header's (lower case), `model` the model folder.
const mi = (model, algo, group, run, family, extra = {}) => ({
  name: `${model}\\${algo}\\${group}\\${run}\\${run}-000001.safetensors`, stem: `${run}-000001`, run_key: `${model}/${algo}/${group}/${run}`,
  epoch: 1, final: false, num_epochs: 1, family, algo: algo.toLowerCase(), model, group, group_label: group, run, run_short: run,
  trigger: "", date: "2026-11-16", has_meta: true, downloaded: false, ...extra,
});
const A29 = "ANIMA/LoKr/@enaa97/@enaa97_2.9b_lokr";
const A38 = "ANIMA/LoKr/@enaa97/@enaa97_3.8b";
const A0 = "ANIMA/LoKr/@enaa97/@enaa97";
const M1 = "MOSS/LoKr/@enaa97/@enaa97_moss";
const S40 = "ANIMA/LoRA/snapref-v1a/snapref_v1a__40b";
const S52 = "ANIMA/LoRA/snapref-v1a/snapref_v1a__52b";
const X1 = "SDXL/LoRA/HUSI/HUSI";
const K1 = "KREA2/LoRA/krea2/QuadView_krea2_v1";
const U1 = "_unsorted/LoRA/M3_Unlocked/M3_Unlocked_V2";
const OLD = "lokr/Old/old_2.9b";
const MODEL_ITEMS = [
  mi("ANIMA", "LoKr", "@enaa97", "@enaa97_2.9b_lokr", "2.9b"), mi("ANIMA", "LoKr", "@enaa97", "@enaa97_3.8b", "3.8b"),
  mi("ANIMA", "LoKr", "@enaa97", "@enaa97", "anima"), mi("MOSS", "LoKr", "@enaa97", "@enaa97_moss", "moss"),
  mi("ANIMA", "LoRA", "snapref-v1a", "snapref_v1a__40b", "anima"), mi("ANIMA", "LoRA", "snapref-v1a", "snapref_v1a__52b", "anima"),
  mi("SDXL", "LoRA", "HUSI", "HUSI", "sdxl"), mi("KREA2", "LoRA", "krea2", "QuadView_krea2_v1", "krea2"),
  mi("_unsorted", "LoRA", "M3_Unlocked", "M3_Unlocked_V2", null),
  it("lokr", "Old", "old_2.9b", 1), // the old layout: no model folder
];
const models = (prefs = {}, family = "2.9b") =>
  initialState({ mode: "add", family, index: { items: MODEL_ITEMS, prefs: { ...PREFS, ...prefs } } });

test("folderKeys: a model folder heads the tree as <model>/<algo>, a run without one keeps <algo>", () => {
  assert.deepEqual(folderKeys({ model: "ANIMA", algo: "lokr", group: "@enaa97" }), { head: "ANIMA/lokr", group: "ANIMA/lokr/@enaa97" });
  assert.deepEqual(folderKeys({ model: "", algo: "lokr", group: "Old" }), { head: "lokr", group: "lokr/Old" });
  assert.deepEqual(folderKeys({ algo: "", group: "g" }), { head: "", group: "/g" });
  assert.deepEqual(folderKeys(undefined), { head: "", group: "/" });
  // the algorithm folder is read case-blind (lower case, as the keys were when they came from the header)
  assert.deepEqual(folderKeys({ model: "ANIMA", algo: "LoRA", group: "HUSI" }), { head: "ANIMA/lora", group: "ANIMA/lora/HUSI" });
  assert.deepEqual(folderKeys({ model: "", algo: "LoKr", group: "Old" }), folderKeys({ model: "", algo: "lokr", group: "Old" }));
});

test("the tree places a run by its algorithm FOLDER: runs whose header says lora_anima sit in ANIMA/lora with the rest", () => {
  const husi = "ANIMA/LoRA/HUSI/husi_anima";
  const items = [
    mi("ANIMA", "LoRA", "snapref-v1a", "snapref_v1a__40b", "anima"),
    mi("ANIMA", "LoRA", "HUSI", "husi_anima", "anima", { algo: "lora_anima" }), // kohya's networks.lora_anima
    mi("ANIMA", "LoKr", "@enaa97", "@enaa97_2.9b_lokr", "2.9b", { algo: "lycoris" }),
  ];
  const s = initialState({ mode: "add", family: "2.9b", index: { items, prefs: { ...PREFS, show_all: true } } });
  const at = (key) => keys(reduce(s, { type: "folder", key }));
  assert.deepEqual(at("ANIMA/lora"), [husi, S40]);
  assert.deepEqual(at("ANIMA/lora/HUSI"), [husi]);
  assert.deepEqual(at("ANIMA/lokr"), [A29]);
  assert.deepEqual([at("ANIMA/lora_anima"), at("ANIMA/lycoris")], [[], []], "the header's algorithm is no tree key");
  // the run carries its place: the folders as spelled (the head's label) and the keys the filter compares
  assert.deepEqual(visibleRuns(s).find((r) => r.runKey === husi).tree,
    { model: "ANIMA", algo: "LoRA", head: "ANIMA/lora", group: "ANIMA/lora/HUSI" });
});

test("the folder filter takes the model keys exactly: ANIMA and MOSS runs of one group stay apart", () => {
  const s = models({ show_all: true });
  const at = (key) => keys(reduce(s, { type: "folder", key }));
  assert.deepEqual(at("ANIMA/lokr"), [A0, A29, A38]);
  assert.deepEqual(at("ANIMA/lokr/@enaa97"), [A0, A29, A38]);
  assert.deepEqual(at("MOSS/lokr"), [M1]);
  assert.deepEqual(at("MOSS/lokr/@enaa97"), [M1]);
  assert.deepEqual(at("ANIMA/lora/snapref-v1a"), [S40, S52]);
  assert.deepEqual(at("_unsorted/lora"), [U1]);
  assert.deepEqual(at("lokr"), [OLD], "the bare algorithm head is the old layout's only");
  assert.deepEqual(at("lokr/Old"), [OLD]);
  assert.deepEqual(at("lokr/@enaa97"), []);
  assert.deepEqual(at("ANIMA"), [], "a model folder alone is no tree key");
});

test("model-folder families: an Anima checkpoint hides MOSS / KREA2 / SDXL, a MOSS checkpoint all but MOSS, until 모든 모델", () => {
  const shown = (family, prefs) => new Set(keys(models(prefs, family)));
  assert.deepEqual(shown("2.9b"), new Set([A29, A0, S40, S52, U1, OLD]));
  assert.deepEqual(shown("3.8b"), new Set([A38, A0, S40, S52, U1]));
  assert.deepEqual(shown("base"), new Set([A0, S40, S52, U1]));
  assert.deepEqual(shown("moss"), new Set([M1, U1]));
  assert.deepEqual(shown("krea2"), new Set([K1, U1]));
  assert.equal(shown("moss", { show_all: true }).size, MODEL_ITEMS.length);
});

test("search leaves the model folder out: \"anima\" / \"sdxl\" find no run by its model folder alone", () => {
  const s = models({ show_all: true });
  const find = (text) => keys(reduce(s, { type: "query", text }));
  assert.deepEqual(find("anima"), []);
  assert.deepEqual(find("sdxl"), []);
  assert.deepEqual(find("moss"), [M1], "the run name has moss");
  assert.deepEqual(find("krea2"), [K1], "the group folder is krea2");
  assert.deepEqual(find("enaa97 3.8b"), [A38]);
});

test("labelSuffixes: the first run-name token none of the others has, else the model folder that tells it apart", () => {
  assert.deepEqual(labelSuffixes([
    { label: "s", run: "snapref_v1a__40b", model: "ANIMA" }, { label: "s", run: "snapref_v1a__52b", model: "ANIMA" },
    { label: "t", run: "@taesol9743_2.9b_lokr_came_b10_2e06", model: "ANIMA" }, { label: "t", run: "@taesol9743_3.8b", model: "ANIMA" },
    { label: "e", run: "@enaa97", model: "ANIMA" }, { label: "e", run: "@enaa97_3.8b", model: "ANIMA" },
    { label: "g", run: "r", model: "ANIMA" }, { label: "g", run: "r", model: "MOSS" }, { label: "g", run: "r", model: "" },
    { label: "alone", run: "x_40b", model: "ANIMA" },
  ]), ["40b", "52b", "2.9b", "3.8b", "", "3.8b", "ANIMA", "MOSS", "", ""]);
  // three of a kind: what none of the OTHERS has ("2.9b" is shared by two, so "lokr" / "loha" tell those apart)
  assert.deepEqual(labelSuffixes([
    { label: "a", run: "a_2.9b_lokr", model: "" }, { label: "a", run: "a_3.8b", model: "" }, { label: "a", run: "a_2.9b_loha", model: "" },
  ]), ["lokr", "3.8b", "loha"]);
  // case-blind: "LoKr" / "lokr" is no difference, so the model folders tell them apart
  assert.deepEqual(labelSuffixes([{ label: "a", run: "a_LoKr", model: "X" }, { label: "a", run: "a_lokr", model: "Y" }]), ["X", "Y"]);
  // labels that read the same (only "_" / " " or case apart) count as one: the real snapref_v2a_pre1 next to its __40b run
  assert.deepEqual(labelSuffixes([
    { label: "snapref-v2a-pre1 · snapref v2a pre1", run: "snapref_v2a_pre1", model: "ANIMA" },
    { label: "snapref-v2a-pre1 · snapref_v2a_pre1", run: "snapref_v2a_pre1__40b", model: "ANIMA" },
    { label: "Snapref-V2a-Pre1 · SNAPREF V2A PRE1", run: "snapref_v2a_pre1_moss", model: "ANIMA" },
    { label: "snapref-v2a-pre1 · snapref-v2a-pre1", run: "y", model: "ANIMA" }, // "-" is a real difference
  ]), ["", "40b", "moss", ""]);
  assert.deepEqual(labelSuffixes([]), []);
});

test("labelSuffixes: run names split at \"_\" and white space alike, so a spaced run name is no whole-name token", () => {
  // "snapref v2a pre1" was one token none of the others had: its suffix repeated the whole label (and the other got "snapref")
  assert.deepEqual(labelSuffixes([
    { label: "snapref-v2a-pre1 · snapref v2a pre1", run: "snapref v2a pre1", model: "ANIMA" },
    { label: "snapref-v2a-pre1 · snapref_v2a_pre1", run: "snapref_v2a_pre1__40b", model: "ANIMA" },
  ]), ["", "40b"]);
  // any white space (JS \s: a tab, U+3000) and runs of separators
  assert.deepEqual(labelSuffixes([
    { label: "a", run: "a\tv1　 40b", model: "" }, { label: "a", run: "a__v1_52b", model: "" },
  ]), ["40b", "52b"]);
});

test("labelSuffixes: cards still alike after the token / model suffix take their group label, then run name, then folders", () => {
  // three runs "x" under ANIMA, ANIMA, MOSS: the model folder leaves the two ANIMA cards alike, their groups tell them apart
  assert.deepEqual(labelSuffixes([
    { label: "x", run: "x", model: "ANIMA", groupLabel: "g1", runKey: "ANIMA/LoKr/g1/x" },
    { label: "x", run: "x", model: "ANIMA", groupLabel: "g2", runKey: "ANIMA/LoKr/g2/x" },
    { label: "x", run: "x", model: "MOSS", groupLabel: "g1", runKey: "MOSS/LoKr/g1/x" },
  ]), ["ANIMA · g1", "ANIMA · g2", "MOSS"]);
  // x_a_b / x_a_c / x_b_c: each token is another's, one model and one group (it tells nothing, so it is left out): run names
  assert.deepEqual(labelSuffixes([
    { label: "x", run: "x_a_b", model: "ANIMA", groupLabel: "g", runKey: "ANIMA/LoKr/g/x_a_b" },
    { label: "x", run: "x_a_c", model: "ANIMA", groupLabel: "g", runKey: "ANIMA/LoKr/g/x_a_c" },
    { label: "x", run: "x_b_c", model: "ANIMA", groupLabel: "g", runKey: "ANIMA/LoKr/g/x_b_c" },
  ]), ["x_a_b", "x_a_c", "x_b_c"]);
  // the same run name in two groups of one model
  assert.deepEqual(labelSuffixes([
    { label: "snapref v1", run: "snapref_v1", model: "ANIMA", groupLabel: "snap", runKey: "ANIMA/LoRA/snap/snapref_v1" },
    { label: "snapref v1", run: "snapref_v1", model: "ANIMA", groupLabel: "snapref", runKey: "ANIMA/LoRA/snapref/snapref_v1" },
  ]), ["snap", "snapref"]);
  // ... and in two groups that differ only in their "(@tag)" (one group label), or in two algorithm folders: the folders that differ
  assert.deepEqual(labelSuffixes([
    { label: "x", run: "x", model: "ANIMA", groupLabel: "x", runKey: "ANIMA/LoKr/x (@a)/x" },
    { label: "x", run: "x", model: "ANIMA", groupLabel: "x", runKey: "ANIMA/LoKr/x (@b)/x" },
    { label: "x", run: "x", model: "ANIMA", groupLabel: "x", runKey: "ANIMA/LoRA/x (@b)/x" },
  ]), ["LoKr/x (@a)", "LoKr/x (@b)", "LoRA/x (@b)"]);
  // only the spelling apart ("_" / " "): the run names show it
  assert.deepEqual(labelSuffixes([
    { label: "g · x y", run: "x y", model: "ANIMA", groupLabel: "g", runKey: "ANIMA/LoKr/g/x y" },
    { label: "g · x y", run: "x_y", model: "ANIMA", groupLabel: "g", runKey: "ANIMA/LoKr/g/x_y" },
  ]), ["x y", "x_y"]);
});

test("through the index: three or more cards of one label end up reading apart", () => {
  const items = [
    // groups "x (@a)" / "x (@b)" both read "x", as does the run: three cards "x"
    mi("ANIMA", "LoKr", "x (@a)", "x", "anima", { group_label: "x" }), mi("ANIMA", "LoKr", "x (@b)", "x", "anima", { group_label: "x" }),
    mi("MOSS", "LoKr", "x (@a)", "x", "moss", { group_label: "x" }),
    // only recipe tokens apart: all three read "@k"
    mi("ANIMA", "LoKr", "@k", "@k_2.9b_lokr", "2.9b"), mi("ANIMA", "LoKr", "@k", "@k_2.9b_came", "2.9b"), mi("ANIMA", "LoKr", "@k", "@k_lokr_came", "anima"),
    // the same run name in two groups whose names both start it
    mi("ANIMA", "LoRA", "snap", "snapref_v1", "anima"), mi("ANIMA", "LoRA", "snapref", "snapref_v1", "anima"),
  ];
  const s = initialState({ mode: "add", family: "2.9b", index: { items, prefs: { ...PREFS, show_all: true } } });
  const label = (runKey) => visibleRuns(s).find((r) => r.runKey === runKey).label;
  assert.deepEqual(items.map((i) => label(i.run_key)), [
    "x · ANIMA · x (@a)", "x · ANIMA · x (@b)", "x · MOSS",
    "@k · @k_2.9b_lokr", "@k · @k_2.9b_came", "@k · @k_lokr_came",
    "snapref v1 · snap", "snapref v1 · snapref",
  ]);
  const read = (text) => text.toLowerCase().replace(/_/g, " ");
  assert.equal(new Set(visibleRuns(s).map((r) => read(r.label))).size, items.length, "no two cards read the same");
});

test("cards that would show one label get their suffix, also while a filter shows one of them; the bare one stays bare", () => {
  const s = models({ show_all: true });
  const card = (key) => visibleRuns(s).find((r) => r.runKey === key);
  assert.deepEqual([S40, S52, A29, A38, A0].map((key) => card(key).label),
    ["snapref-v1a · snapref_v1a · 40b", "snapref-v1a · snapref_v1a · 52b", "@enaa97 · 2.9b", "@enaa97 · 3.8b", "@enaa97"]);
  assert.deepEqual([card(S40).labelBase, card(S40).labelSuffix], ["snapref-v1a · snapref_v1a", "40b"]);
  assert.deepEqual([card(A0).labelBase, card(A0).labelSuffix], ["@enaa97", ""]);
  assert.equal(card(M1).label, "@enaa97 moss", "a label of its own is left alone");
  const one = reduce(s, { type: "query", text: "40b" });
  assert.deepEqual(visibleRuns(one).map((r) => r.label), ["snapref-v1a · snapref_v1a · 40b"]);
  assert.equal(new Set(visibleRuns(s).map((r) => r.label)).size, MODEL_ITEMS.length, "no two cards read the same");
});

test("remember puts the picked stems first, once each, and keeps at most 20", () => {
  let s = make({ prefs: { ...PREFS, recent: ["a", "b", "c"] } });
  s = reduce(s, { type: "remember", stems: ["c", "x"] });
  assert.deepEqual(s.recent, ["c", "x", "a", "b"]);
  s = reduce(s, { type: "remember", stems: Array.from({ length: 25 }, (_, i) => `n${i}`) });
  assert.equal(s.recent.length, 20);
  assert.equal(s.recent[0], "n0");
});
