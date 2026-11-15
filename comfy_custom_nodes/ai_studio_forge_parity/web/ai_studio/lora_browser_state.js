// Pure state of the LoRA picker window: runs (one card per run, epoch files grouped), folder tree / search / family filter /
// sort, selection and epoch stepping, strength, the tray of picks added this session, xyz multi-check. No DOM, no fetch.
// The index payload (Task 6) arrives in snake_case; initialState converts it once and nothing else here sees snake_case.
import { matchesQuery, familyVisible, shortLoraName } from "./lora_names.js";

const SORTS = ["name", "date", "group", "recent"];
const RECENT_MAX = 20; // lora_store.RECENT_MAX
const EPOCH_TAIL = / · (?:e\d+|최종(?: e\d+)?)$/;
const clampStrength = (v) => {
  const n = Number(v);
  return Number.isFinite(n) ? Math.min(2, Math.max(0, Number(n.toFixed(2)))) : 1;
};
const byText = (a, b) => String(a).localeCompare(String(b), "ko");

/** Epoch files of a run: numbered epochs ascending, the final file last. */
function sortEpochs(epochs) {
  return epochs.sort((a, b) => Number(a.final) - Number(b.final) || (a.epoch ?? 0) - (b.epoch ?? 0));
}

function buildRuns(items, favorites) {
  const byKey = new Map();
  for (const raw of items || []) {
    const key = raw.run_key;
    if (!byKey.has(key)) {
      byKey.set(key, {
        runKey: key, algo: raw.algo || "", group: raw.group || "", groupLabel: raw.group_label || raw.group || "",
        family: null, numEpochs: raw.num_epochs || 0, hasMeta: false, downloaded: false, latestDate: "", epochs: [],
        // Each epoch file keeps its search text so a run matches when ANY of its files does (ruling R4).
        files: [],
      });
    }
    const run = byKey.get(key);
    run.family = run.family ?? raw.family ?? null;
    run.numEpochs = Math.max(run.numEpochs, raw.num_epochs || 0);
    run.hasMeta = run.hasMeta || Boolean(raw.has_meta);
    run.downloaded = run.downloaded || Boolean(raw.downloaded);
    if ((raw.date || "") > run.latestDate) run.latestDate = raw.date;
    run.epochs.push({ epoch: raw.epoch ?? null, name: raw.name, stem: raw.stem, final: Boolean(raw.final) });
    // The index epoch: a final file carries its header epoch ("e15 최종"), which its path does not show.
    run.files.push({ entry: raw.name, trigger: raw.trigger || "", epoch: raw.epoch ?? null });
    run.label = run.label || shortLoraName(raw.name).replace(EPOCH_TAIL, "");
  }
  const favSet = new Set(favorites);
  return [...byKey.values()].map((r) => ({ ...r, epochs: sortEpochs(r.epochs), favorite: favSet.has(r.runKey) }));
}

/** `query` = text typed in the search box before the index arrived (applied like a typed query). */
export function initialState({ mode = "add", current = "", family = "", index = {}, prefs, query = "" } = {}) {
  const p = prefs || index.prefs || {};
  const favorites = Array.isArray(p.favorites) ? [...p.favorites] : [];
  let state = {
    mode, family, query: "", folder: "", sort: SORTS.includes(p.sort) ? p.sort : "name", showAll: Boolean(p.show_all),
    treeOpen: Array.isArray(p.tree_open) ? [...p.tree_open] : [], closeAfter: Boolean(p.close_after_add),
    favorites, recent: Array.isArray(p.recent) ? [...p.recent] : [],
    // What 최근 사용 (the sort and the folder) shows: `recent` as it was when the window opened. A pick goes into `recent`
    // (saved for the next open) but not here, so adding does not reorder the cards under the pointer.
    recentShown: Array.isArray(p.recent) ? [...p.recent] : [],
    boardStars: index.board_stars || {}, promptLabels: index.prompt_labels || {},
    runs: buildRuns(index.items, favorites), selected: null, epochOf: {}, queryEpochs: {}, checked: [], strength: 1, added: [],
  };
  const start = mode === "replace" && current
    ? state.runs.map((run) => [run, run.epochs.find((e) => e.stem === current || e.name === current)]).find(([, hit]) => hit)
    : null;
  if (start) {
    const [run, hit] = start;
    // Ruling R9: the current run stays visible and selected even when the family filter would hide it.
    const showAll = state.showAll || !familyVisible(run.family, family, false);
    state = { ...state, showAll, selected: run.runKey, epochOf: { [run.runKey]: hit.stem } };
  }
  return String(query) ? reduce(state, { type: "query", text: query }) : state;
}

const findRun = (state, runKey) => state.runs.find((r) => r.runKey === runKey) || null;

/** The epoch file of `run` that is selected: the remembered one, else the final/last file. */
function chosenEpoch(state, run) {
  return run.epochs.find((e) => e.stem === state.epochOf[run.runKey]) || run.epochs[run.epochs.length - 1];
}

/**
 * `prefs.recent` lists file stems, newest first: a run's rank is the smallest index among its epoch stems (Infinity = never
 * used), in the order the window opened with (`recentShown`).
 */
function recentRank(state, run) {
  const shown = state.recentShown || state.recent;
  const idx = run.epochs.map((e) => shown.indexOf(e.stem)).filter((i) => i >= 0);
  return idx.length ? Math.min(...idx) : Infinity;
}

export function visibleRuns(state) {
  const { folder, query } = state;
  const runs = state.runs.filter((r) => {
    if (!familyVisible(r.family, state.family, state.showAll)) return false;
    if (folder === "*fav") { if (!state.favorites.includes(r.runKey)) return false; }
    else if (folder === "*recent") { if (recentRank(state, r) === Infinity) return false; }
    else if (folder && folder !== r.algo && folder !== `${r.algo}/${r.group}`) return false;
    return !query.trim() || r.files.some((f) => matchesQuery(f, query));
  });
  const rank = (r) => recentRank(state, r);
  const sorters = {
    name: (a, b) => byText(a.label, b.label),
    date: (a, b) => byText(b.latestDate, a.latestDate) || byText(a.label, b.label),
    group: (a, b) => byText(a.groupLabel, b.groupLabel) || byText(a.label, b.label),
    recent: (a, b) => (rank(a) === rank(b) ? byText(a.label, b.label) : rank(a) - rank(b)),
  };
  return runs.sort(sorters[state.sort] || sorters.name).map((r) => ({
    runKey: r.runKey, label: r.label, family: r.family, groupLabel: r.groupLabel, epochs: r.epochs.map(({ epoch, name, stem, final }) => ({ epoch, name, stem, final })),
    favorite: state.favorites.includes(r.runKey), latestDate: r.latestDate,
  }));
}

/**
 * After a filter change: keep the selection if its run is still shown, else take the first shown run. `pickFirst` (a search):
 * with nothing selected the first shown run is selected too, so the detail pane and Enter have a card.
 */
function fixSelection(state, pickFirst = false) {
  if (state.selected == null && !pickFirst) return state;
  const shown = visibleRuns(state);
  return shown.some((r) => r.runKey === state.selected) ? state : { ...state, selected: shown.length ? shown[0].runKey : null };
}

/**
 * Epoch tokens of the query ("14", "e14") move each run's selected epoch to that epoch (ruling R4): the numbered file, else
 * the final file of that epoch. `queryEpochs[runKey] = {prior, set}` remembers each move; the next query first undoes them
 * (back to `prior`), so the moves last only while the token is in the query — except a run whose epoch the user changed
 * meanwhile (no longer `set`): that choice stays.
 */
function applyQueryEpochs(state) {
  const epochOf = { ...state.epochOf };
  for (const [runKey, { prior, set }] of Object.entries(state.queryEpochs || {})) {
    if (epochOf[runKey] !== set) continue;
    if (prior == null) delete epochOf[runKey];
    else epochOf[runKey] = prior;
  }
  const nums = String(state.query).toLowerCase().split(/[\s\\/]+/).map((w) => /^e?(\d+)$/.exec(w)).filter(Boolean).map((m) => parseInt(m[1], 10));
  const queryEpochs = {};
  for (const run of nums.length ? state.runs : []) {
    const hit = run.epochs.find((e) => !e.final && nums.includes(e.epoch)) || run.epochs.find((e) => e.final && nums.includes(e.epoch));
    if (!hit) continue;
    queryEpochs[run.runKey] = { prior: epochOf[run.runKey] ?? null, set: hit.stem };
    epochOf[run.runKey] = hit.stem;
  }
  return { ...state, epochOf, queryEpochs };
}

function setEpoch(state, delta, value) {
  const run = findRun(state, state.selected);
  if (!run) return state;
  const cur = chosenEpoch(state, run);
  let target = null;
  if (delta != null) {
    const i = Math.min(run.epochs.length - 1, Math.max(0, run.epochs.indexOf(cur) + delta));
    target = run.epochs[i];
  } else {
    target = run.epochs.find((e) => (value === "final" ? e.final : !e.final && e.epoch === value));
  }
  return target ? { ...state, epochOf: { ...state.epochOf, [run.runKey]: target.stem } } : state;
}

export function reduce(state, action) {
  switch (action.type) {
    case "query": {
      const next = applyQueryEpochs({ ...state, query: String(action.text ?? "") });
      return fixSelection(next, Boolean(next.query.trim()));
    }
    case "folder": return fixSelection({ ...state, folder: action.key || "" });
    case "sort": return SORTS.includes(action.by) ? { ...state, sort: action.by } : state;
    case "showAll": return fixSelection({ ...state, showAll: Boolean(action.on) });
    case "move": {
      const shown = visibleRuns(state);
      if (!shown.length) return state;
      const i = shown.findIndex((r) => r.runKey === state.selected);
      if (i < 0) return { ...state, selected: shown[0].runKey };
      const step = (action.dx || 0) + (action.dy || 0) * Math.max(1, action.cols || 1);
      return { ...state, selected: shown[Math.min(shown.length - 1, Math.max(0, i + step))].runKey };
    }
    case "select": return findRun(state, action.runKey) ? { ...state, selected: action.runKey } : state;
    case "epoch": return action.delta != null ? setEpoch(state, action.delta) : setEpoch(state, null, action.value);
    case "toggleEpoch": {
      const run = findRun(state, state.selected);
      const hit = run && run.epochs.find((e) => (action.value === "final" ? e.final : !e.final && e.epoch === action.value));
      if (!hit) return state;
      const on = state.checked.includes(hit.stem);
      return { ...state, checked: on ? state.checked.filter((s) => s !== hit.stem) : [...state.checked, hit.stem] };
    }
    case "favorite": {
      const has = state.favorites.includes(action.runKey);
      return fixSelection({ ...state, favorites: has ? state.favorites.filter((k) => k !== action.runKey) : [...state.favorites, action.runKey] });
    }
    case "strength": return { ...state, strength: clampStrength(action.value) };
    case "strengthStep": return { ...state, strength: clampStrength(Math.round((state.strength + action.delta) * 20) / 20) };
    case "closeAfter": return { ...state, closeAfter: Boolean(action.on) };
    // R10: picked stems go to the front of prefs.recent (no duplicates, at most 20); `recentShown` stays as it is
    case "remember": return { ...state, recent: [...new Set([...action.stems, ...state.recent])].slice(0, RECENT_MAX) };
    case "added": return { ...state, added: [...state.added, { ...action.pick }] };
    case "unadd": return { ...state, added: state.added.filter((p) => p.key !== action.key) };
    default: return state;
  }
}

const lastEpoch = (run) => run.epochs[run.epochs.length - 1];
const pickOf = (run, ep, strength) => ({ name: ep.stem, strength, label: shortLoraName(ep.name, { num_epochs: run.numEpochs }) });

/**
 * Card button "+ eNN 추가": the selected epoch for the selected run and for a run the query's epoch token moved ("bjun 14"
 * → every card adds its e14), else the final/last epoch; strength as set now.
 */
export function quickPick(state, runKey) {
  const run = findRun(state, runKey);
  if (!run) return null;
  const moved = runKey === state.selected || Object.hasOwn(state.queryEpochs || {}, runKey);
  return pickOf(run, moved ? chosenEpoch(state, run) : lastEpoch(run), state.strength);
}

/**
 * What a card's own button acts on: add/replace = quickPick; xyz ("+ eNN 체크") = the epoch the query's token moved the run
 * to, else its last epoch (strength null) — the selected run's chips do not change it.
 */
export function cardPick(state, runKey) {
  if (state.mode !== "xyz") return quickPick(state, runKey);
  const run = findRun(state, runKey);
  if (!run) return null;
  return pickOf(run, Object.hasOwn(state.queryEpochs || {}, runKey) ? chosenEpoch(state, run) : lastEpoch(run), null);
}

/** What "추가"/"바꾸기"/"확인" hands back: add/replace = the selected epoch (strength as set); xyz = the checked epochs, no strength. */
export function picks(state) {
  if (state.mode === "xyz") {
    return state.runs.flatMap((r) => r.epochs).filter((e) => state.checked.includes(e.stem)).map((e) => ({ name: e.stem, strength: null }));
  }
  const run = findRun(state, state.selected);
  return run ? [{ name: chosenEpoch(state, run).stem, strength: state.strength }] : [];
}
