// LoRA picker window: an overlay on document.body (like the lightbox) — header and bottom tray (lora_browser_chrome.js),
// folder tree + virtualised card grid (lora_browser_grid.js) and the detail pane (lora_browser_detail.js).
// Mouse first: every card has its own "+ eNN 추가" button and each add goes into the panel at once; keys are extra.
// Every state change goes through lora_browser_state.js (pure); this file wires it to the DOM and to the injected `api`.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { h } from "./studio_dom.js";
import { cardPick, initialState, picks, quickPick, reduce, visibleRuns } from "./lora_browser_state.js";
import { coverVersions, withQuery } from "./lora_api.js";
import { actionBar, detailView, epochText, familyLabel } from "./lora_browser_detail.js";
import { cardGrid, drawTree, keepFocus, listKey, runInfo } from "./lora_browser_grid.js";
import { footBar, headerBar } from "./lora_browser_chrome.js";

const NARROW = 900; // px of window width: the tree folds into a button and the detail goes under the grid
const SAVE_DELAY = 400;
const ARROWS = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
const FOCUSABLE = 'button:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';
const errText = (err) => String(err?.message || err);

/**
 * What a key does. `focus`: "search" (the search box) or "grid" (cards / the window). Returns "add" (Enter: add and close),
 * "addKeep" (Shift+Enter), "close" (Esc), "focusSearch" (Ctrl/Cmd+F), a state action without its context
 * (`{type: "move", dx, dy}` — the caller adds `cols`; `{type: "epoch", delta}`; `{type: "favorite"}` — the caller adds the
 * run), or null. In the search box letters, `[` `]` and ←/→ stay text; ↑/↓ still move the cards.
 */
export function keyAction(event, focus) {
  const key = event?.key;
  if (key === "Escape") return "close";
  if (event?.ctrlKey || event?.metaKey) return key === "f" || key === "F" ? "focusSearch" : null;
  if (key === "Enter") return event.shiftKey ? "addKeep" : "add";
  const arrow = ARROWS[key];
  if (arrow && (focus === "grid" || arrow[0] === 0)) return { type: "move", dx: arrow[0], dy: arrow[1] };
  if (focus !== "grid") return null;
  if (key === "[" || key === "]") return { type: "epoch", delta: key === "[" ? -1 : 1 };
  if (key === "f" || key === "F") return { type: "favorite" };
  return null;
}

let openWindow = null; // the window on screen; one at a time

/**
 * Opens the picker; returns `{close()}`. `mode`: "add" — every pick goes into the panel at once (`onPick(pick) -> rowIndex`),
 * the tray's ✕ takes it out again (`onUnpick(rowIndex)`; return false if that row was not removed); "replace" — one
 * `onPick(pick)`, then the window closes; "xyz" — one `onPickMany(picks)` with the checked epochs, then it closes.
 * `pick = {name: stem, strength, label}`, `picks = [{name: stem, strength: null}]`. `current` = the stem (or list entry) a
 * replace starts on, `family` = the checkpoint's family, `strength` (optional) the row's strength to start from (null = no
 * strength controls: an XYZ row keeps its own strength),
 * `onTrigger(text)` = "작가 칸에 넣기". `api = {index(fresh), detail(name), thumbUrl(name, kind, epoch, i), savePrefs(prefs),
 * clearPreview?(runKey)}` (promises of the route payloads; `epoch`/`i` may be undefined = the route's default).
 */
export function openLoraBrowser(opts = {}) {
  const { mode = "add", current = "", family = "", api, onPick, onUnpick, onPickMany, onTrigger, strength } = opts;
  openWindow?.close();
  const doc = document;
  const opener = doc.activeElement;
  let state = null;
  let runs = [];
  let infos = new Map();
  let index = {};
  let closed = false, rafId = 0, needMeasure = false;
  let userShowAll = false, treeOpen = false, lastSaved = "", saveTimer = null, toastTimer = null;
  let loadTicket = 0, bust = 0, detailTick = 0, detailSig = "", detailRun = null, seq = 0, shownList = "";
  const details = new Map(); // entry name -> /loras/detail payload | {error}
  const lastOfRun = new Map(); // runKey -> its last loaded detail, shown while another epoch's detail loads
  const pending = new Set();
  const rowOf = new Map(); // tray key -> panel row index (null when the panel gave none)
  let coverOf = new Map(); // entry name -> its run's custom preview version (a new cover URL after it is set / cleared)
  const thumb = (name, kind, epoch, i) =>
    withQuery(api.thumbUrl(name, kind, epoch, i), { pv: kind === "cover" ? coverOf.get(name) : null, v: bust || null });

  // ---- skeleton (built once; parts are redrawn in place)
  const header = headerBar({
    mode, family,
    on: {
      query: (text) => dispatch({ type: "query", text }),
      showAll: () => { userShowAll = !state?.showAll; dispatch({ type: "showAll", on: userShowAll }); },
      sort: (by) => dispatch({ type: "sort", by }),
      closeAfter: () => dispatch({ type: "closeAfter", on: !state?.closeAfter }),
      tree: () => { treeOpen = !treeOpen; render(); },
      refresh: () => load(true),
      close: () => close(),
    },
  });
  const search = header.search;
  const tree = h("nav", { class: "ais-lb-tree", "aria-label": "폴더" });
  const grid = cardGrid({
    mode, thumb, fallback: () => dialog,
    on: {
      select: (runKey) => dispatch({ type: "select", runKey }),
      quick: (run) => quickAdd(run),
      favorite: (runKey) => dispatch({ type: "favorite", runKey }),
      scroll: () => schedule(false),
    },
  });
  const summary = h("span", {});
  const gridSec = h("section", { class: "ais-lb-gridsec", "aria-label": "LoRA 목록" },
    h("div", { class: "ais-lb-bar" }, summary, h("span", {}, "런 하나 = 카드 하나 · 에폭은 상세 칸에서")), grid.scroller);
  const detailBody = h("div", { class: "ais-lb-dwrap" });
  const actions = actionBar(mode, {
    strength: (value) => dispatch({ type: "strength", value }),
    step: (delta) => dispatch({ type: "strengthStep", delta }),
    commit: () => commitSelected(mode !== "add" || Boolean(state?.closeAfter)),
  }, { strength: strength !== null });
  const aside = h("aside", { class: "ais-lb-detail", "aria-label": "고른 LoRA" }, detailBody, actions.el);
  const footer = footBar({ mode, on: { unpick: (key) => unpick(key), done: () => close() } });
  const toast = h("div", { class: "ais-lb-toast", role: "status", "aria-live": "polite", hidden: true });
  const dialog = h("div", { class: "ais-lb", role: "dialog", "aria-modal": "true", "aria-label": "LoRA 고르기", tabindex: "-1" },
    header.el, h("div", { class: "ais-lb-body" }, tree, gridSec, aside), footer.el, toast);
  const backdrop = h("div", { class: "ais-lb-backdrop" }, dialog);

  // ---- state, prefs, loading
  function dispatch(action) {
    if (!state || closed) return;
    state = reduce(state, action);
    render();
  }
  const prefsOf = () => ({
    favorites: [...state.favorites], recent: [...state.recent], sort: state.sort, show_all: userShowAll, tree_open: treeOpen, close_after_add: state.closeAfter,
  });
  /** R10: changed prefs (favorites, recent, sort, show_all, tree_open, close_after_add) are saved after a short pause. */
  function savePrefsSoon() {
    clearTimeout(saveTimer);
    saveTimer = JSON.stringify(prefsOf()) === lastSaved ? null : setTimeout(flushPrefs, SAVE_DELAY);
  }
  function flushPrefs() {
    clearTimeout(saveTimer);
    saveTimer = null;
    const text = state ? JSON.stringify(prefsOf()) : lastSaved;
    if (text === lastSaved) return;
    lastSaved = text;
    const prefs = JSON.parse(text);
    Promise.resolve().then(() => api.savePrefs?.(prefs)).catch((err) => console.warn("[ai_studio] LoRA 설정 저장 실패", err));
  }
  /** R10: picked stems go to the front of prefs.recent. The cards keep the 최근 사용 order the window opened with. */
  function remember(stems) {
    state = reduce(state, { type: "remember", stems });
  }

  function load(fresh) {
    const ticket = ++loadTicket;
    if (state) showToast("다시 읽는 중…");
    Promise.resolve().then(() => api.index(Boolean(fresh))).then((payload) => {
      if (closed || ticket !== loadTicket) return;
      index = payload || {};
      infos = runInfo(index.items);
      coverOf = coverVersions(index);
      const prefs = index.prefs || {};
      // first load: text typed while the index was loading (the first build can take ~10 s) counts as a typed query
      const next = initialState({ mode, current, family, index, prefs, query: state ? "" : search.value });
      if (!state) {
        userShowAll = prefs.show_all === true;
        treeOpen = prefs.tree_open === true; // Python keeps tree_open as a flag: the folded tree (narrow window) is open
        state = Number.isFinite(strength) ? reduce(next, { type: "strength", value: strength }) : next;
        lastSaved = JSON.stringify(prefsOf());
        render();
        grid.reveal(state.selected, false);
      } else {
        // ↻ keeps what was done in this window, on the new list
        const keep = ["query", "folder", "sort", "showAll", "closeAfter", "favorites", "recent", "recentShown", "selected", "epochOf", "queryEpochs", "checked", "strength", "added"];
        state = reduce({ ...next, ...Object.fromEntries(keep.map((k) => [k, state[k]])) }, { type: "query", text: state.query });
        grid.forget();
        details.clear();
        lastOfRun.clear();
        bust += 1;
        render();
        showToast(index.error || "새로 읽었어요");
      }
    }, (err) => {
      if (closed || ticket !== loadTicket) return;
      if (state) showToast(`다시 읽지 못했어요: ${errText(err)}`);
      else grid.status(`목록을 못 읽었어요: ${errText(err)} — ↻ 로 다시`);
    });
  }

  // ---- picking
  function commitPick(pick, closeNow) {
    if (!pick || closed) return;
    let row;
    try {
      row = onPick?.({ name: pick.name, strength: pick.strength, label: pick.label });
    } catch (err) {
      return showToast(`넣지 못했어요: ${errText(err)}`);
    }
    remember([pick.name]);
    if (mode !== "add" || closeNow) return close();
    const key = `${pick.name}#${++seq}`;
    rowOf.set(key, Number.isInteger(row) ? row : null);
    dispatch({ type: "added", pick: { key, name: pick.name, strength: pick.strength, label: pick.label } });
    showToast(`추가됨: ${pick.label} — 계속 고를 수 있어요`);
  }
  function commitSelected(closeNow) {
    if (!state) return;
    if (mode === "xyz") {
      const list = picks(state);
      if (!list.length) return showToast("비교할 에폭을 체크하세요");
      try {
        onPickMany?.(list);
      } catch (err) {
        return showToast(`넣지 못했어요: ${errText(err)}`);
      }
      remember(list.map((p) => p.name));
      return close();
    }
    if (!state.selected && runs.length) dispatch({ type: "select", runKey: runs[0].runKey }); // "bjun 14" + Enter takes the first card
    if (state.selected) commitPick(quickPick(state, state.selected), closeNow);
  }
  /** Card button: add/replace that run at once; xyz: check its last epoch, or the searched one (and show it). */
  function quickAdd(run) {
    if (mode !== "xyz") return commitPick(quickPick(state, run.runKey), mode !== "add" || state.closeAfter);
    const ep = run.epochs.find((e) => e.stem === cardPick(state, run.runKey)?.name) || run.epochs[run.epochs.length - 1];
    const value = ep.final ? "final" : ep.epoch;
    state = reduce(reduce(state, { type: "select", runKey: run.runKey }), { type: "toggleEpoch", value });
    dispatch({ type: "epoch", value });
  }
  /** Tray ✕: the row leaves the panel; rows added after it move up one, so their indexes follow. */
  function unpick(key) {
    const row = rowOf.get(key);
    rowOf.delete(key);
    let removed = false;
    try {
      removed = row != null && onUnpick?.(row) !== false;
    } catch (err) {
      showToast(`빼지 못했어요: ${errText(err)}`);
    }
    if (removed) for (const [k, r] of rowOf) if (r != null && r > row) rowOf.set(k, r - 1);
    dispatch({ type: "unadd", key });
  }
  function clearPreview(run) {
    Promise.resolve().then(() => api.clearPreview(run.runKey)).then(() => {
      if (closed) return;
      for (const e of run.epochs) details.delete(e.name);
      lastOfRun.delete(run.runKey);
      grid.forget(run.runKey);
      bust += 1;
      render();
      showToast("미리보기 지정을 풀었어요");
    }, (err) => showToast(`풀지 못했어요: ${errText(err)}`));
  }

  // ---- drawing
  function render() {
    if (closed || !state) return;
    runs = visibleRuns(state);
    const famOnly = visibleRuns({ ...state, showAll: false, folder: "", query: "" });
    const hidden = visibleRuns({ ...state, showAll: true, folder: "", query: "" }).length - famOnly.length;
    header.update({ showAll: state.showAll, hidden, sort: state.sort, closeAfter: state.closeAfter, treeOpen });
    dialog.classList.toggle("ais-lb-tree-open", treeOpen);
    keepFocus(tree, () => drawTree(tree, {
      base: state.showAll ? visibleRuns({ ...state, folder: "", query: "" }) : famOnly, infos, folder: state.folder, recent: state.recentShown,
      onFolder: (key) => {
        const fold = treeOpen && dialog.classList.contains("ais-lb-narrow");
        if (fold) treeOpen = false;
        dispatch({ type: "folder", key });
        if (fold) search.focus({ preventScroll: true }); // the drawer just closed under the focused folder button
      },
    }), search, dialog);
    summary.textContent = `${runs.length}개 런` + (state.showAll ? " · 모든 모델" : family ? ` · ${familyLabel(family)} + 계열 모름` : "")
      + (hidden && !state.showAll ? ` · 다른 계열 ${hidden}개 숨김` : "");
    const key = listKey(state, runs);
    const reset = shownList !== "" && key !== shownList; // a new list starts at the top, then shows the selected card
    shownList = key;
    if (reset) grid.scroller.scrollTop = 0;
    grid.set(runs, { selected: state.selected, checked: state.checked, infos, pickOf: (runKey) => cardPick(state, runKey) },
      index.error ? `${index.error} — ↻ 로 다시` : hidden && !state.showAll ? "맞는 LoRA 가 없어요. '모든 모델' 을 켜 보세요." : "맞는 LoRA 가 없어요.");
    if (reset) grid.reveal(state.selected, false);
    drawDetail();
    keepFocus(footer.el, () => footer.update(state.added), dialog);
    savePrefsSoon();
  }

  function ensureDetail(run, ep) {
    if (details.has(ep.name) || pending.has(ep.name)) return;
    pending.add(ep.name);
    Promise.resolve().then(() => api.detail(ep.name)).then((d) => d || { error: "빈 응답" }, (err) => ({ error: errText(err) })).then((d) => {
      pending.delete(ep.name);
      if (closed) return;
      details.set(ep.name, d);
      if (!d.error) lastOfRun.set(run.runKey, d);
      detailTick += 1;
      drawDetail();
    });
  }

  function drawDetail() {
    const run = runs.find((r) => r.runKey === state.selected) || null;
    const pick = run ? quickPick(state, run.runKey) : null;
    const ep = run ? run.epochs.find((e) => e.stem === pick?.name) || null : null;
    if (run && ep) ensureDetail(run, ep);
    const sig = JSON.stringify([run?.runKey, ep?.stem, mode === "xyz" ? state.checked : 0, detailTick, bust]);
    if (sig !== detailSig) {
      detailSig = sig;
      const info = (run && infos.get(run.runKey)) || {};
      // the same run drawn again (an epoch chip, a check, its detail arriving) stays where it was scrolled; another run starts at the top
      const top = run && run.runKey === detailRun ? detailBody.scrollTop : 0;
      detailRun = run ? run.runKey : null;
      keepFocus(detailBody, () => detailBody.replaceChildren(detailView({
        run, ep, info, mode, checked: state.checked, thumb,
        detail: ep ? details.get(ep.name) || lastOfRun.get(run.runKey) || null : null,
        stars: (index.board_stars || {})[info.run] || [], labels: (index.prompt_labels || {})[info.run] || [],
        on: {
          epoch: (e) => {
            const value = e.final ? "final" : e.epoch;
            if (mode === "xyz") state = reduce(state, { type: "toggleEpoch", value });
            dispatch({ type: "epoch", value }); // in xyz the samples follow the last chip touched
          },
          trigger: onTrigger ? (text) => { onTrigger(text); showToast(`작가 칸에 넣음: ${text}`); } : null,
          clearPreview: api.clearPreview && run ? () => clearPreview(run) : null,
        },
      })), dialog);
      detailBody.scrollTop = top;
    }
    if (mode === "xyz") {
      const list = picks(state);
      actions.update({ text: `XYZ 에 ${list.length}개 넣기`, disabled: !list.length, title: list.map((p) => p.name).join("\n") });
    } else {
      const text = mode === "replace" ? "이걸로 바꾸기" : `+ ${epochText(ep) ? `${epochText(ep)} ` : ""}추가${state.closeAfter ? "하고 닫기" : ""}`;
      actions.update({ strength: state.strength, text, disabled: !ep, title: pick?.label || "" });
    }
  }

  function showToast(text) {
    if (closed) return;
    toast.textContent = text;
    toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toast.hidden = true; }, 2400);
  }

  // ---- layout, keys, closing
  function measure() {
    dialog.classList.toggle("ais-lb-narrow", dialog.clientWidth > 0 && dialog.clientWidth < NARROW);
    if (grid.measure()) render();
    else if (state) grid.draw(false);
  }
  const raf = globalThis.requestAnimationFrame || ((fn) => setTimeout(fn, 16));
  const cancelRaf = globalThis.cancelAnimationFrame || clearTimeout;
  function schedule(measureToo) {
    needMeasure ||= measureToo;
    if (rafId) return;
    rafId = raf(() => {
      rafId = 0;
      if (closed) return;
      if (needMeasure) measure();
      else if (state) grid.draw(false);
      needMeasure = false;
    });
  }
  // Measuring in the next frame keeps a layout change from re-triggering the observer within the same frame.
  const resizeObserver = globalThis.ResizeObserver ? new ResizeObserver(() => schedule(true)) : null;
  const onWindowResize = () => schedule(true);

  function trapTab(event) {
    const list = [...dialog.querySelectorAll(FOCUSABLE)].filter((el) => el.getClientRects().length > 0);
    if (!list.length) return;
    const active = doc.activeElement;
    const edge = event.shiftKey ? active === list[0] || active === dialog : active === list[list.length - 1];
    if (!edge && dialog.contains(active)) return;
    event.preventDefault();
    (event.shiftKey ? list[list.length - 1] : list[0]).focus();
  }

  function onKey(event) {
    event.stopPropagation(); // keys typed here are not ComfyUI shortcuts
    if (closed || doc.querySelector(".ais-lightbox")) return; // an open lightbox handles its own keys (Esc) first
    if (event.isComposing || event.keyCode === 229) return; // Korean IME composing: Enter / arrows belong to it
    if (event.key === "Tab") return trapTab(event);
    const target = event.target;
    const focus = target === search ? "search" : target === dialog || grid.isGridTarget(target) ? "grid" : null;
    const act = keyAction(event, focus || "grid");
    if (act == null || (focus == null && act !== "close" && act !== "focusSearch")) return; // other controls keep their own keys
    event.preventDefault();
    if (act === "close") close();
    else if (act === "focusSearch") {
      search.focus();
      search.select();
    } else if (act === "add" || act === "addKeep") commitSelected(act === "add");
    else if (act.type === "move") {
      dispatch({ ...act, cols: grid.cols });
      grid.reveal(state?.selected, focus === "grid");
    } else if (act.type === "favorite") {
      if (state?.selected) dispatch({ type: "favorite", runKey: state.selected });
    } else dispatch(act);
  }
  const stop = (event) => event.stopPropagation();

  let pressedOutside = false;
  backdrop.addEventListener("mousedown", (event) => { pressedOutside = event.target === backdrop; });
  backdrop.addEventListener("click", (event) => {
    if (pressedOutside && event.target === backdrop) close(); // a drag that started inside (the slider) does not close
    pressedOutside = false;
  });
  backdrop.addEventListener("keydown", onKey);
  for (const type of ["keyup", "keypress", "paste", "copy", "cut"]) backdrop.addEventListener(type, stop);

  function close() {
    if (closed) return;
    flushPrefs(); // before `closed`: the last pick's recent / a toggle saved just now still goes out
    closed = true;
    cancelRaf(rafId);
    clearTimeout(toastTimer);
    resizeObserver?.disconnect();
    globalThis.removeEventListener?.("resize", onWindowResize);
    backdrop.remove();
    if (openWindow === handle) openWindow = null;
    if (opener && opener !== doc.body && opener.isConnected && typeof opener.focus === "function") opener.focus({ preventScroll: true });
  }

  const handle = { close };
  openWindow = handle;
  doc.body.append(backdrop);
  if (resizeObserver) {
    resizeObserver.observe(dialog);
    resizeObserver.observe(grid.scroller);
  } else globalThis.addEventListener?.("resize", onWindowResize);
  measure();
  search.focus();
  load(false);
  return handle;
}
