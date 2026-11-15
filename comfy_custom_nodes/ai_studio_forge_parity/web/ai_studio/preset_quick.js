// Prompt-box presets, the quick list: a small popup under a box's label — a row per preset (name + one-line preview, a dot
// on the one the box holds). A row replaces the box, its "+" appends; the foot saves the box as a new preset and opens the
// preset window. The popup lives on document.body (the canvas transforms everything inside a node's DOM widget).
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { h } from "./studio_dom.js";
import { keepFocus } from "./lora_browser_grid.js";
import { addPresetStylesheet, afterPresetStyles, errText } from "./preset_dom.js";
import { PRESET_LABELS, appendText, autoName, matchPreset, presetEdge, previewLine, textProblem } from "./preset_logic.js";

const WIDTH = 330; // preset.css .ais-pq width, until the popup is measured
const GAP = 4;
const MARGIN = 8;

let openList = null; // the list on screen; one at a time

/**
 * `() => the label a box's list hangs under, as it is on the page now` (null: there is none).
 * `anchor` is that label, or a function giving it. The panel replaces every label when it redraws (panel_view.js render),
 * so an element that left the page is looked up again: `.ais-box-label[data-section="<key>"]` (panel_view.js promptRow)
 * under its nearest ancestor that is still on the page — the same panel, not another studio node's.
 */
export function anchorFinder(anchor, sectionKey) {
  if (typeof anchor === "function") return () => anchor() || null;
  let label = anchor || null;
  const ancestors = [];
  for (let node = label?.parentElement; node; node = node.parentElement) ancestors.push(node);
  const selector = /^[\w-]+$/.test(String(sectionKey)) ? `.ais-box-label[data-section="${sectionKey}"]` : null;
  return () => {
    if (label?.isConnected) return label;
    const root = selector ? ancestors.find((node) => node.isConnected) : null;
    const next = root?.querySelector?.(selector) || null;
    if (next) label = next;
    return next;
  };
}

/**
 * Opens the list under `anchor` (the box's label: the element, or a function giving the label that is on the page now —
 * see anchorFinder; the panel may redraw while the list is open); returns `{close(), place(), element, key}` — or null when
 * the list of this box was open: the label works as a toggle. Another box's open list is closed first.
 * `opts`: `store` (preset_api.js createPresetStore), `getText(key) -> the box's text`, `setText(key, text)` (write it into the
 * panel state), `onManage(key)` ("프리셋 관리…": open the preset window), `onNotice(message)` (optional: what was done, for the
 * panel's notice line), `label` (default: the box's panel label).
 */
export function openPresetQuick(anchor, sectionKey, opts = {}) {
  const { store, getText, setText, onManage, onNotice, label = PRESET_LABELS[sectionKey] || sectionKey } = opts;
  if (openList) {
    const same = openList.key === sectionKey;
    openList.close();
    if (same) return null;
  }
  addPresetStylesheet();
  const doc = document;
  const findAnchor = anchorFinder(anchor, sectionKey);
  let marked = null; // the label that says aria-expanded="true"
  let closed = false;
  let saving = false; // "+ 지금 내용 저장" is on its way to the server
  let message = "";
  const boxText = () => String(getText?.(sectionKey) ?? "");

  const rowsBox = h("div", { class: "ais-pq-rows" });
  const note = h("div", { class: "ais-pq-note", role: "status", "aria-live": "polite", hidden: true });
  const saveButton = h("button", { type: "button", class: "ais-pq-save", title: "지금 칸에 적힌 내용을 새 프리셋으로", "aria-disabled": "false", onClick: () => saveHere() }, "+ 지금 내용 저장");
  const popup = h("div", { class: "ais-pq", role: "dialog", "aria-label": `${label} 프리셋 목록`, tabindex: "-1" },
    h("div", { class: "ais-pq-head" },
      h("span", { class: "ais-pq-title" }, `${label} 프리셋`), h("span", { class: "ais-pq-hint" }, "누르면 바꾸기 · + 는 뒤에 붙이기")),
    rowsBox, note,
    h("div", { class: "ais-pq-foot" },
      saveButton,
      h("button", { type: "button", class: "ais-pq-manage", onClick: () => { close(); onManage?.(sectionKey); } }, "프리셋 관리…")));

  // ---- drawing
  function row(preset, current) {
    return h("div", { class: "ais-pq-row" },
      h("button", { type: "button", class: "ais-pq-pick", "data-k": `p:${preset.id}`, title: preset.text, onClick: () => put(preset, false) },
        h("span", { class: "ais-pq-line" },
          h("span", { class: `ais-pq-dot${preset === current ? " on" : ""}`, title: preset === current ? "지금 칸과 같음" : undefined }),
          h("span", { class: "ais-pq-name" }, preset.name)),
        h("span", { class: "ais-pq-text" }, previewLine(preset.text))),
      h("button", { type: "button", class: "ais-pq-add", "data-k": `a:${preset.id}`, "aria-label": `${preset.name} 뒤에 붙이기`, title: "뒤에 붙이기", onClick: () => put(preset, true) }, "+"));
  }
  function emptyText(list) {
    if (list.length) return "";
    if (store.loaded && store.broken) return "프리셋 파일이 깨져서 읽지 못했어요. 새로 저장하면 원래 파일은 .bak 으로 남겨 둬요.";
    if (store.loaded) return "아직 프리셋이 없어요. 아래 ‘+ 지금 내용 저장’으로 만들어요.";
    return store.error ? `프리셋을 못 읽었어요: ${store.error}` : "읽는 중…";
  }
  function render() {
    if (closed) return;
    const list = store.list(sectionKey);
    const current = matchPreset(list, boxText());
    const empty = emptyText(list);
    keepFocus(rowsBox, () => rowsBox.replaceChildren(...(empty ? [h("div", { class: "ais-pq-empty" }, empty)] : list.map((preset) => row(preset, current)))), popup);
    const before = store.boxUndo(sectionKey, boxText());
    const text = message || (before === null ? "" : "방금 프리셋을 넣기 전 내용으로 돌아갈 수 있어요.");
    keepFocus(note, () => note.replaceChildren(text,
      before === null ? "" : h("button", { type: "button", class: "ais-pq-undo", "data-k": "undo", onClick: () => undo(before) }, "되돌리기")), popup);
    note.hidden = !text;
    place();
  }
  function say(text) {
    message = text;
    render();
  }

  // ---- actions
  function put(preset, append) {
    const before = boxText();
    const after = append ? appendText(before, preset.text) : preset.text;
    try {
      setText?.(sectionKey, after);
    } catch (err) {
      return say(`넣지 못했어요: ${errText(err)}`);
    }
    store.noteBoxChange(sectionKey, before, after);
    store.touch(sectionKey, preset.id).catch((err) => console.warn("[AI Studio] preset use not saved:", err));
    onNotice?.(append ? `‘${preset.name}’을(를) ${label} 칸 뒤에 붙였어요.` : `${label} 칸을 ‘${preset.name}’(으)로 바꿨어요.`);
    close();
  }
  function undo(before) {
    try {
      setText?.(sectionKey, before);
    } catch (err) {
      return say(`되돌리지 못했어요: ${errText(err)}`);
    }
    store.clearBoxUndo();
    onNotice?.(`${label} 칸을 되돌렸어요.`);
    say("되돌렸어요.");
  }
  /** One save at a time: a second press while the first is on its way would save the same text twice under one name. */
  async function saveHere() {
    if (saving) return;
    saving = true;
    saveButton.setAttribute("aria-disabled", "true"); // not `disabled`: that would drop the focus (and the list's keys) to <body>
    try {
      await saveBox();
    } finally {
      saving = false;
      saveButton.setAttribute("aria-disabled", "false");
    }
  }
  async function saveBox() {
    try {
      await store.load();
    } catch (err) {
      return say(`프리셋을 못 읽었어요: ${errText(err)}`);
    }
    if (closed) return;
    const text = boxText();
    const list = store.list(sectionKey);
    if (!presetEdge(text)) return say(`${label} 칸이 비어 있어서 저장할 게 없어요.`);
    const problem = textProblem(text, store.limits);
    if (problem) return say(problem);
    const same = matchPreset(list, text);
    if (same) return say(`이미 프리셋 ‘${same.name}’과 같은 내용이에요.`);
    const name = autoName(list, label);
    try {
      await store.add(sectionKey, name, text);
    } catch (err) {
      return say(`저장하지 못했어요: ${errText(err)}`);
    }
    const done = `지금 ${label} 칸을 ‘${name}’(으)로 저장했어요. 이름은 ‘프리셋 관리…’에서 바꿔요.`;
    onNotice?.(done);
    say(done);
  }

  // ---- place, keys, closing
  /** Under the label as it is on the page now, kept inside the window; above it when there is no room below. */
  function place() {
    if (closed) return;
    const anchorEl = findAnchor();
    if (!anchorEl) { // no label for now (the card is hidden): the list stays where it was
      if (!popup.style.left) popup.style.left = popup.style.top = `${MARGIN}px`;
      return;
    }
    if (marked !== anchorEl) {
      marked = anchorEl;
      anchorEl.setAttribute?.("aria-expanded", "true"); // a label the panel has just drawn does not say so yet
    }
    const rect = anchorEl.getBoundingClientRect();
    const width = popup.offsetWidth || WIDTH;
    const height = popup.offsetHeight || 0;
    const viewWidth = globalThis.innerWidth || 0;
    const viewHeight = globalThis.innerHeight || 0;
    const left = viewWidth ? Math.min(rect.left, viewWidth - width - MARGIN) : rect.left;
    const below = rect.bottom + GAP;
    const above = rect.top - height - GAP;
    popup.style.left = `${Math.max(MARGIN, left)}px`;
    popup.style.top = `${viewHeight && below + height > viewHeight - MARGIN && above >= MARGIN ? above : below}px`;
  }
  function moveFocus(step) {
    const picks = [...rowsBox.querySelectorAll(".ais-pq-pick")];
    if (!picks.length) return;
    const at = picks.indexOf(doc.activeElement);
    picks[at < 0 ? (step > 0 ? 0 : picks.length - 1) : Math.max(0, Math.min(picks.length - 1, at + step))].focus();
  }
  function onKey(event) {
    event.stopPropagation(); // keys typed here are not ComfyUI shortcuts
    if (event.key === "Escape") {
      event.preventDefault();
      close();
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      moveFocus(event.key === "ArrowDown" ? 1 : -1);
    }
  }
  const stop = (event) => event.stopPropagation();
  const inside = (target) => popup.contains(target) || findAnchor()?.contains?.(target) === true; // the label's own click toggles
  const onOutside = (event) => { if (!inside(event.target)) close(); };
  const onDocKey = (event) => { if (event.key === "Escape" && !popup.contains(event.target)) close(); };
  const onResize = () => close();

  function close() {
    if (closed) return;
    closed = true;
    const hadFocus = popup.contains(doc.activeElement);
    unsubscribe();
    doc.removeEventListener("pointerdown", onOutside, true);
    doc.removeEventListener("wheel", onOutside, true);
    doc.removeEventListener("keydown", onDocKey, true);
    globalThis.removeEventListener?.("resize", onResize);
    popup.remove();
    const anchorEl = findAnchor();
    for (const el of new Set([marked, anchorEl])) el?.setAttribute?.("aria-expanded", "false");
    if (openList === handle) openList = null;
    if (hadFocus && typeof anchorEl?.focus === "function") anchorEl.focus({ preventScroll: true });
  }

  popup.addEventListener("keydown", onKey);
  for (const type of ["keyup", "keypress", "paste", "copy", "cut"]) popup.addEventListener(type, stop);
  doc.addEventListener("pointerdown", onOutside, true);
  doc.addEventListener("wheel", onOutside, true);
  doc.addEventListener("keydown", onDocKey, true);
  globalThis.addEventListener?.("resize", onResize);
  const unsubscribe = store.subscribe(render);
  const handle = { close, place, element: popup, key: sectionKey };
  openList = handle;
  doc.body.append(popup);
  render(); // place() marks the label aria-expanded
  afterPresetStyles(place); // the very first list may be drawn before its stylesheet: its size, so its place, changes
  popup.focus({ preventScroll: true });
  store.load(true).catch(() => {}); // render() shows store.error; a fresh read picks up another tab's changes
  return handle;
}
