// LoRA picker window, left and middle: the folder tree and the card grid (one card per run, "+ eNN 추가" on every card).
// The grid is virtualised by rows of fixed height: only the visible rows ±2 are in the DOM, so only their pictures load
// (and those with `loading="lazy"`). Cards are kept per run and reused, so a picture is not fetched again on scrolling.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { h } from "./studio_dom.js";
import { algoLabel, epochText, initialsOf, runBadges, safeImg } from "./lora_browser_detail.js";

const CARD_MIN = 132; // px: narrowest card before the grid drops a column
const CARD_H = 218; // px: fixed card height (studio.css .ais-lb-card)
const GAP = 12;
const ROW_H = CARD_H + GAP;
const PAD_Y = 12; // .ais-lb-scroll padding
const PAD_X = 14;
const BUFFER_ROWS = 2;
const byText = (a, b) => String(a).localeCompare(String(b), "ko");

/** Card / detail info per run from the /loras/index items (the state keeps only what it filters on); `algo` is the header's. */
export function runInfo(items) {
  const map = new Map();
  for (const it of items || []) {
    const info = map.get(it.run_key) || { algo: it.algo || "", run: it.run || "", precision: "", numEpochs: 0, downloaded: false, trigger: "" };
    info.precision ||= it.precision || "";
    info.numEpochs = Math.max(info.numEpochs, it.num_epochs || 0);
    info.downloaded ||= Boolean(it.downloaded);
    info.trigger ||= it.trigger || "";
    map.set(it.run_key, info);
  }
  return map;
}

/**
 * What the card grid lists: search, folder, sort, 모든 모델 and the runs shown, in order. When it changes the grid goes back to
 * the top and then to the selected card (the one Enter / "+ 추가" act on); selection, epoch, strength, tray, a favourite
 * outside ★ or a pick under 최근 사용 (its order is the one the window opened with) keep the scroll position.
 */
export function listKey(state, runs) {
  return JSON.stringify([state.query, state.folder, state.sort, state.showAll, runs.map((r) => r.runKey)]);
}

/** Focuses the first candidate that really takes the focus (a hidden one does not); returns it, or null. */
export function focusFirst(doc, candidates) {
  for (const el of candidates) {
    if (!el) continue;
    el.focus({ preventScroll: true });
    if (doc.activeElement === el) return el;
  }
  return null;
}

/**
 * Redraws `container` by `draw()` and puts the focus back on the same control (by its `data-k`). When that control is gone or
 * cannot take the focus (e.g. the folded tree hid it), the first of `fallbacks` that can gets it — the focus must not drop to
 * <body>, where the window's Esc and Tab trap no longer see the keys.
 */
export function keepFocus(container, draw, ...fallbacks) {
  const doc = container.ownerDocument;
  const active = doc.activeElement;
  const inside = active && active !== container && container.contains(active);
  const key = inside ? active.getAttribute("data-k") : null;
  draw();
  if (!inside || doc.activeElement === active) return;
  const again = active.isConnected && container.contains(active)
    ? active
    : [...container.querySelectorAll("[data-k]")].find((el) => el.getAttribute("data-k") === key);
  focusFirst(doc, [again, ...fallbacks]);
}

/**
 * The folder tree: 전체 · ★ 즐겨찾기 · 최근 사용, then one head per model folder and algorithm FOLDER ("ANIMA · LoKr"; just
 * "LoKr" for a run without a model folder), by model then algorithm, with their author groups and run counts. `base` = the runs
 * the family filter shows (search and folder ignored); each run's `tree` (lora_browser_state.treePlace: its folders and their
 * folderKeys) places it, so a node lists exactly what the folder filter (visibleRuns) then shows. The header's algorithm
 * ("lora_anima" in ANIMA\LoRA) is not used here; the card badges keep it.
 */
export function drawTree(tree, { base, folder, recent, onFolder }) {
  const node = (key, label, count, extra = "") => h("button", {
    type: "button", class: `ais-lb-node${extra}${folder === key ? " on" : ""}`, "aria-pressed": String(folder === key),
    "data-k": `t:${key}`, title: label, onClick: () => onFolder(key),
  }, h("span", { class: "ais-lb-node-name" }, label), h("span", { class: "ais-lb-count" }, String(count)));
  const heads = new Map(); // head key -> {model, algo, n, groups: Map(group key -> {label, n})}
  for (const run of base) {
    const at = run.tree;
    if (!heads.has(at.head)) heads.set(at.head, { model: at.model, algo: at.algo, n: 0, groups: new Map() });
    const head = heads.get(at.head);
    head.n += 1;
    head.groups.set(at.group, { label: run.groupLabel, n: (head.groups.get(at.group)?.n || 0) + 1 });
  }
  const recentRun = (run) => run.epochs.some((e) => recent.includes(e.stem));
  const byHead = ([, a], [, b]) => byText(a.model, b.model) || byText(algoLabel(a.algo), algoLabel(b.algo));
  const nodes = [
    node("", "전체", base.length), node("*fav", "★ 즐겨찾기", base.filter((r) => r.favorite).length), node("*recent", "최근 사용", base.filter(recentRun).length),
    [...heads].sort(byHead).map(([key, head]) => [
      // a run without a model folder or algorithm has no folder key of its own ("" is 전체), so its head is only a label
      key ? node(key, [head.model, algoLabel(head.algo)].filter(Boolean).join(" · "), head.n, " head") : h("div", { class: "ais-lb-node head" }, "기타"),
      [...head.groups].sort((a, b) => byText(a[1].label, b[1].label)).map(([groupKey, g]) => node(groupKey, g.label || "(맨 위)", g.n)),
    ]),
  ];
  tree.replaceChildren(...nodes.flat(Infinity)); // replaceChildren does not flatten arrays (h() does)
}

/**
 * The card grid. `mode` as the window; `thumb(name, kind) -> url`; `on = {select(runKey), quick(run), favorite(runKey), scroll()}`;
 * A focused card that scrolls out leaves the focus on the grid itself (else on `fallback()`). Call `set(runs, view, emptyText)` after every state change with
 * `view = {selected, checked, infos, pickOf(runKey) -> cardPick}`; `measure()` after a resize (true if the columns changed).
 */
export function cardGrid({ mode, thumb, fallback, on }) {
  let runs = [];
  let view = { selected: null, checked: [], infos: new Map(), pickOf: () => null };
  let cols = 4;
  let drawn = "";
  const cards = new Map(); // runKey -> {el, main, update}
  const rowsEl = h("div", { class: "ais-lb-rows" });
  const spacer = h("div", { class: "ais-lb-spacer" }, rowsEl);
  const emptyEl = h("div", { class: "ais-lb-empty" }, "불러오는 중…");
  // overflow-anchor: the rows are placed by hand (draw). Chrome's scroll anchoring takes a card put back in place for content
  // that moved and scrolls the list up by a few rows on every click on a card.
  const scroller = h("div", { class: "ais-lb-scroll", tabindex: "-1", style: "overflow-anchor: none", onScroll: () => on.scroll?.() }, emptyEl, spacer);

  function makeCard(run) {
    const last = run.epochs[run.epochs.length - 1];
    const base = run.labelBase ?? run.label;
    // the suffix that tells apart cards with one label (" · 40b") sits outside the part the card cuts with "…", so it always shows
    const name = h("span", { class: "ais-lb-card-name" }, h("span", { class: "ais-lb-card-label" }, base),
      run.labelSuffix ? h("span", { class: "ais-lb-card-suffix" }, ` · ${run.labelSuffix}`) : null);
    const main = h("button", { type: "button", class: "ais-lb-card-main", "data-k": `m:${run.runKey}`, onClick: () => on.select(run.runKey) },
      safeImg(thumb(last.name, "cover"), initialsOf(base), "ais-lb-cover"),
      h("span", { class: "ais-lb-card-body" }, name, h("span", { class: "ais-lb-badges" }, runBadges(view.infos.get(run.runKey) || {}, run.family))));
    const quick = h("button", { type: "button", class: "ais-lb-quick", "data-k": `q:${run.runKey}` });
    const fav = h("button", { type: "button", class: "ais-lb-fav", "data-k": `f:${run.runKey}`, onClick: () => on.favorite(run.runKey) });
    const el = h("div", { class: "ais-lb-card" }, main, h("div", { class: "ais-lb-card-foot" }, quick), fav);
    let latest = run;
    quick.addEventListener("click", () => on.quick(latest));
    const update = (r) => {
      latest = r;
      const sel = r.runKey === view.selected;
      const tail = r.epochs[r.epochs.length - 1];
      const pick = view.pickOf(r.runKey);
      const ep = r.epochs.find((e) => e.stem === pick?.name) || tail; // the epoch this card's button acts on
      const checked = view.checked.includes(ep.stem);
      const what = (e) => (epochText(e) ? `${epochText(e)} ` : "");
      el.classList.toggle("sel", sel);
      main.setAttribute("aria-pressed", String(sel));
      main.setAttribute("aria-label", `${r.label} 고르기`);
      main.title = tail.name;
      fav.textContent = r.favorite ? "★" : "☆";
      fav.classList.toggle("on", r.favorite);
      fav.setAttribute("aria-pressed", String(r.favorite));
      fav.setAttribute("aria-label", `${r.favorite ? "즐겨찾기 빼기" : "즐겨찾기"}: ${r.label}`);
      quick.classList.toggle("on", mode === "xyz" && checked);
      if (mode === "xyz") quick.textContent = checked ? `✓ ${what(ep)}체크됨` : `+ ${what(ep)}체크`;
      else if (mode === "replace") quick.textContent = epochText(ep) ? `바꾸기 · ${epochText(ep)}` : "이걸로 바꾸기";
      else quick.textContent = `+ ${what(ep)}추가`;
      quick.title = mode === "xyz" ? `${pick?.label || r.label} 을 비교할 에폭으로 체크` : `${pick?.label || r.label} ${mode === "replace" ? "로 바꾸기" : "를 바로 추가"}`;
    };
    return { el, main, update };
  }

  /** Puts the cards of the visible rows ±2 in the DOM (`force`: even when the same rows are already there). */
  function draw(force) {
    const top = Math.max(0, scroller.scrollTop - PAD_Y);
    const rowCount = Math.ceil(runs.length / cols);
    const first = Math.max(0, Math.floor(top / ROW_H) - BUFFER_ROWS);
    const last = Math.min(rowCount - 1, Math.floor((top + scroller.clientHeight) / ROW_H) + BUFFER_ROWS);
    if (!force && drawn === `${first}:${last}:${cols}`) return;
    drawn = `${first}:${last}:${cols}`;
    rowsEl.style.top = `${first * ROW_H}px`;
    const slice = last < first ? [] : runs.slice(first * cols, (last + 1) * cols);
    keepFocus(rowsEl, () => rowsEl.replaceChildren(...slice.map((run) => {
      if (!cards.has(run.runKey)) cards.set(run.runKey, makeCard(run));
      const card = cards.get(run.runKey);
      card.update(run);
      return card.el;
    })), scroller, fallback?.());
  }

  return {
    scroller,
    get cols() { return cols; },
    set(nextRuns, nextView, emptyText) {
      runs = nextRuns;
      view = nextView;
      emptyEl.hidden = runs.length > 0;
      emptyEl.textContent = emptyText;
      spacer.style.height = `${Math.max(0, Math.ceil(runs.length / cols) * ROW_H - GAP)}px`;
      rowsEl.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
      draw(true);
    },
    draw,
    status(text) { emptyEl.hidden = false; emptyEl.textContent = text; },
    measure() {
      const next = Math.max(1, Math.floor((scroller.clientWidth - 2 * PAD_X + GAP) / (CARD_MIN + GAP)));
      if (next === cols) return false;
      cols = next;
      return true;
    },
    /** Scrolls the run's card into view (keyboard moves); `focusCard` also focuses it. */
    reveal(runKey, focusCard) {
      const i = runs.findIndex((r) => r.runKey === runKey);
      if (i < 0) return;
      const top = PAD_Y + Math.floor(i / cols) * ROW_H;
      if (top < scroller.scrollTop) scroller.scrollTop = top - PAD_Y;
      else if (top + CARD_H > scroller.scrollTop + scroller.clientHeight) scroller.scrollTop = top + CARD_H + PAD_Y - scroller.clientHeight;
      draw(true);
      if (focusCard) cards.get(runKey)?.main.focus({ preventScroll: true });
    },
    /** Drops a cached card (its picture is fetched again): one run, or all with no argument. */
    forget(runKey) {
      if (runKey === undefined) cards.clear();
      else cards.delete(runKey);
    },
    isGridTarget: (el) => el === scroller || Boolean(el?.classList?.contains("ais-lb-card-main")),
  };
}
