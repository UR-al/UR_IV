// Prompt-box preset window (프리셋 관리): an overlay on document.body in the look of the LoRA picker — a tab per prompt box,
// a card per preset (바꾸기 / + 붙이기 go into the panel's box at once), the detail pane to rename, edit, reorder and delete,
// and a foot line with 되돌리기 for the last change. Mouse first; Esc closes, Ctrl+F searches, Ctrl+S saves an edit.
// 되돌리기 is the reverse of that one change (delete the added preset, put the deleted one back, the old name / text, one
// step back) — never a copy of the list: another tab may have saved since.
// The parts are in preset_window_view.js, the rules in preset_logic.js, the data in the store (preset_api.js).
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { h } from "./studio_dom.js";
import { keepFocus } from "./lora_browser_grid.js";
import { addPresetStylesheet, errText, trapTab } from "./preset_dom.js";
import {
  PRESET_LABELS, PRESET_SECTIONS, appendText, autoName, boxStatus, cleanName, filterPresets, pickSelection, presetEdge, presetMeta, textProblem,
} from "./preset_logic.js";
import { cardGrid, detailPane, footBar, gridBar, headBar } from "./preset_window_view.js";

let openWindow = null; // the window on screen; one at a time

/**
 * Opens the window on the tab of `sectionKey`; returns `{close()}`.
 * `opts`: `store` (preset_api.js createPresetStore), `getText(key) -> that box's text`, `setText(key, text)` (write it into the
 * panel state; every 바꾸기 / + 붙이기 / 칸 비우기 calls it at once), `sections` (the boxes to show as tabs, default all eight),
 * `labels` ({key: label}, default the panel's labels), `onClose()` (optional), `returnFocus()` (optional: the element that
 * gets the focus on closing when the one that had it at opening left the page — the panel redraws its labels).
 * `close()` closes at once; Esc, the ✕ and a click on the dimmed page ask first while the editors hold unsaved edits.
 */
export function openPresetWindow(sectionKey, opts = {}) {
  const { store, getText, setText, sections = PRESET_SECTIONS, labels = PRESET_LABELS, onClose, returnFocus } = opts;
  openWindow?.close();
  addPresetStylesheet();
  const doc = document;
  const opener = doc.activeElement;
  const labelOf = (key) => labels[key] || key;
  const boxText = (key) => String(getText?.(key) ?? "");
  let sec = sections.includes(sectionKey) ? sectionKey : sections[0];
  let query = "";
  let selected = null; // preset id shown in the detail pane
  let draftOf = null; // "<box>:<id>" whose name / text are in the editors
  let saving = false; // the inline "새 프리셋 이름" box is open
  let savingText = ""; // what the box held when it was opened: that is what gets saved
  let closeAsked = ""; // the "unsaved edits" warning a first Esc / ✕ / outside click showed
  let committing = false; // a new preset is on its way to the server
  let closeAfter = false;
  let prefsRead = false;
  let message = "";
  let undo = null; // () => void | Promise: takes the last change back
  let closed = false;

  // ---- skeleton (built once)
  const head = headBar({
    tabs: sections.map((key) => ({ key, label: labelOf(key) })),
    on: { query: (text) => { query = text; render(); }, closeAfter: () => toggleCloseAfter(), close: () => askClose(), tab: (key) => pickTab(key) },
  });
  const bar = gridBar({ on: { start: () => startSave(), commit: () => commitSave(), cancel: () => cancelSave() } });
  const grid = cardGrid({ on: { select: (id) => select(id), replace: (id) => put(find(id), false), append: (id) => put(find(id), true) } });
  const detail = detailPane({
    on: {
      clear: () => clearBox(), draft: () => { closeAsked = ""; render(); }, save: () => saveDraft(),
      use: () => putDraft(false), add: () => putDraft(true), move: (delta) => move(delta), remove: () => remove(),
    },
  });
  const foot = footBar({ on: { undo: () => runUndo() } });
  const dialog = h("div", { class: "ais-pp", role: "dialog", "aria-modal": "true", "aria-label": "프롬프트 프리셋", tabindex: "-1" },
    head.el,
    h("div", { class: "ais-pp-body" }, h("section", { class: "ais-pp-gridsec", "aria-label": "프리셋 목록" }, bar.el, grid.el), detail.el),
    foot.el);
  const backdrop = h("div", { class: "ais-pp-backdrop" }, dialog);

  // ---- reading the state
  const list = () => store.list(sec);
  const find = (id) => list().find((preset) => preset.id === id) || null;
  const selectedPreset = () => find(selected);
  const isDirty = (preset) => Boolean(preset) && draftOf === `${sec}:${preset.id}` && (detail.nameInput.value !== preset.name || detail.textInput.value !== preset.text);

  function say(text) {
    message = text;
    render();
  }
  function emptyLine(all, shown) {
    if (shown.length) return "";
    if (!store.loaded) return store.error ? `프리셋을 못 읽었어요: ${store.error}` : "읽는 중…";
    if (store.broken && !all.length) return "프리셋 파일이 깨져서 읽지 못했어요. 새로 저장하면 원래 파일은 prompt_presets.json.bak 으로 남겨 둬요.";
    if (!all.length) return `${labelOf(sec)} 칸 프리셋이 아직 없어요. 위의 ‘+ 지금 칸 내용 저장’으로 만들 수 있어요.`;
    return "맞는 프리셋이 없어요. 위의 ‘+ 지금 칸 내용 저장’으로 만들 수 있어요.";
  }

  // ---- drawing
  function render() {
    if (closed) return;
    if (!prefsRead && store.loaded) {
      prefsRead = true;
      closeAfter = store.prefs.close_after_insert;
    }
    const label = labelOf(sec);
    const all = list();
    const box = boxText(sec);
    const status = boxStatus(all, box);
    selected = pickSelection(all, box, selected);
    const preset = selectedPreset();
    const draftKey = preset ? `${sec}:${preset.id}` : null;
    if (draftKey !== draftOf) {
      draftOf = draftKey;
      detail.fill(preset);
    }
    const shown = filterPresets(all, query);
    const now = Date.now();
    head.update({ label, closeAfter, sec, counts: Object.fromEntries(sections.map((key) => [key, store.list(key).length])) });
    bar.update({
      summary: `프리셋 ${shown.length}개 · 내 순서 · 카드를 누르면 오른쪽에서 고칠 수 있어요`, saving,
      placeholder: `비워 두면 ‘${autoName(all, label)}’`,
    });
    keepFocus(grid.el, () => grid.draw(shown.map((p) => ({
      id: p.id, name: p.name, text: p.text, meta: presetMeta(p, now), selected: p.id === selected, current: status.preset === p,
    })), emptyLine(all, shown)), dialog);
    const index = preset ? all.indexOf(preset) : -1;
    detail.update({ label, box, status, preset, dirty: isDirty(preset), first: index <= 0, last: index < 0 || index === all.length - 1 });
    foot.update({ msg: message, canUndo: Boolean(undo) });
    keepKeys();
  }
  /** The focus never rests on <body> or on a control that was just disabled or hidden: the window's keys would stop working. */
  function keepKeys() {
    const active = doc.activeElement;
    if (!active || !dialog.contains(active) || active.disabled || active.getClientRects().length === 0) dialog.focus({ preventScroll: true });
  }
  /** What to say about the editors' unsaved changes when the selection is about to move on. */
  function leftUnsaved() {
    const preset = selectedPreset();
    return isDirty(preset) ? `‘${preset.name}’을(를) 고친 내용은 저장하지 않았어요.` : "";
  }

  // ---- the box (the panel's text)
  function writeBox(after, done, presetId) {
    const key = sec;
    const before = boxText(key);
    try {
      setText?.(key, after);
    } catch (err) {
      say(`넣지 못했어요: ${errText(err)}`);
      return false;
    }
    store.noteBoxChange(key, before, after);
    undo = () => {
      setText?.(key, before);
      store.clearBoxUndo();
    };
    if (presetId) store.touch(key, presetId).catch((err) => console.warn("[AI Studio] preset use not saved:", err));
    say(done);
    return true;
  }
  function putText(name, text, append, presetId) {
    const label = labelOf(sec);
    const problem = textProblem(text, store.limits);
    if (problem) return say(problem);
    const done = append ? `‘${name}’을(를) ${label} 칸 뒤에 붙였어요.` : `${label} 칸을 ‘${name}’(으)로 바꿨어요.`;
    if (!writeBox(append ? appendText(boxText(sec), text) : text, done, presetId) || !closeAfter) return;
    const edited = selectedPreset();
    if (isDirty(edited)) say(`${done} ‘${edited.name}’을(를) 고친 내용은 저장하지 않아서 창은 열어 뒀어요.`);
    else close();
  }
  const put = (preset, append) => { if (preset) putText(preset.name, preset.text, append, preset.id); };
  /** The detail pane's buttons put in what its editors show, saved or not. */
  function putDraft(append) {
    const preset = selectedPreset();
    if (preset) putText(detail.nameInput.value.trim() || preset.name, detail.textInput.value, append, preset.id);
  }
  function clearBox() {
    if (boxText(sec) === "") return say(`${labelOf(sec)} 칸은 이미 비어 있어요.`);
    writeBox("", `${labelOf(sec)} 칸을 비웠어요.`, null);
  }

  // ---- the presets (the store)
  /**
   * Runs one store change of this box. On success `back(key, result)` gives what takes just that change back: its reverse,
   * never the list as this window knew it — another tab may have saved since, and that list may not even be read yet.
   */
  async function change(run, done, back) {
    const key = sec;
    let result;
    try {
      result = await run();
    } catch (err) {
      say(`저장하지 못했어요: ${errText(err)}`);
      return undefined;
    }
    undo = back(key, result);
    message = done;
    return result ?? null;
  }
  async function startSave() {
    if (!store.loaded) { // the twin check and the new name need the list: wait for it (a failed read is tried again)
      try {
        await store.load();
      } catch (err) {
        return say(`프리셋을 못 읽었어요: ${errText(err)}`);
      }
      if (closed) return;
    }
    const label = labelOf(sec);
    const box = boxText(sec);
    if (!presetEdge(box)) return say(`${label} 칸이 비어 있어서 저장할 게 없어요.`);
    const problem = textProblem(box, store.limits);
    if (problem) return say(problem);
    const same = boxStatus(list(), box).preset;
    if (same) {
      selected = same.id;
      return say(`이미 프리셋 ‘${same.name}’과 같은 내용이에요.`);
    }
    saving = true;
    savingText = box; // 바꾸기 on a card while the name is typed does not change what is saved
    bar.nameInput.value = "";
    render();
    bar.nameInput.focus();
  }
  function cancelSave() {
    saving = false;
    render();
    bar.start.focus();
  }
  async function commitSave() {
    if (!saving || committing) return;
    const label = labelOf(sec);
    const text = savingText;
    const same = boxStatus(list(), text).preset; // saved since the name box opened (another tab, the quick list)
    if (same) {
      saving = false;
      selected = same.id;
      say(`이미 프리셋 ‘${same.name}’과 같은 내용이에요.`);
      return bar.start.focus();
    }
    const typed = bar.nameInput.value;
    const name = typed.trim() ? cleanName(typed, store.limits) : autoName(list(), label);
    if (!name) return say(`이름은 ${store.limits.name}자까지예요.`);
    const done = boxText(sec) === text ? `지금 ${label} 칸 내용을 ‘${name}’(으)로 저장했어요.` : `이름 칸을 열 때 ${label} 칸에 있던 내용을 ‘${name}’(으)로 저장했어요.`;
    committing = true;
    const preset = await change(() => store.add(sec, name, text), done, (key, added) => (added ? () => store.remove(key, added.id) : null));
    committing = false;
    if (closed || preset === undefined) return;
    saving = false;
    if (preset) selected = preset.id;
    render();
    bar.start.focus();
  }
  async function saveDraft() {
    const preset = selectedPreset();
    if (!isDirty(preset)) return;
    const name = cleanName(detail.nameInput.value, store.limits);
    if (!name) return say(detail.nameInput.value.trim() ? `이름은 ${store.limits.name}자까지예요.` : "이름을 적어 주세요.");
    const text = detail.textInput.value;
    const problem = textProblem(text, store.limits);
    if (problem) return say(problem);
    const { id } = preset;
    const was = { name: preset.name, text: preset.text };
    const draft = `${sec}:${id}`;
    const done = await change(() => store.update(sec, id, { name, text }), `‘${name}’을(를) 고쳐서 저장했어요.`, (key) => async () => {
      await store.update(key, id, was);
      if (sec !== key) return;
      selected = id;
      draftOf = null; // the editors show the preset as it is again, not what was typed
    });
    if (closed || done === undefined) return;
    if (draftOf === draft) detail.nameInput.value = name; // as the store keeps it (spaces tidied)
    render();
  }
  async function move(delta) {
    const preset = selectedPreset();
    if (!preset) return;
    const { id } = preset;
    const done = await change(() => store.move(sec, id, delta), delta < 0 ? "위로 옮겼어요." : "아래로 옮겼어요.", (key) => async () => {
      await store.move(key, id, -delta);
      if (sec === key) selected = id;
    });
    if (!closed && done !== undefined) render();
  }
  async function remove() {
    const preset = selectedPreset();
    if (!preset) return;
    const all = list();
    const at = all.indexOf(preset);
    const next = (all[at + 1] || all[at - 1])?.id ?? null;
    const item = { ...preset };
    const done = await change(() => store.remove(sec, item.id), `‘${item.name}’을(를) 지웠어요.`, (key) => async () => {
      await store.insert(key, item, at); // where it was; presets saved elsewhere since stay
      if (sec === key) selected = item.id;
    });
    if (closed || done === undefined) return;
    selected = next;
    render();
  }
  async function runUndo() {
    const run = undo;
    if (!run) return;
    undo = null;
    try {
      await run();
    } catch (err) {
      return say(`되돌리지 못했어요: ${errText(err)}`);
    }
    if (!closed) say("되돌렸어요.");
  }

  // ---- choosing
  function select(id) {
    if (id === selected) return;
    const note = leftUnsaved();
    selected = id;
    if (note) message = note;
    render();
  }
  function pickTab(key) {
    if (key === sec) return;
    const note = leftUnsaved();
    sec = key;
    query = "";
    head.search.value = "";
    saving = false;
    selected = null;
    say(`${labelOf(key)} 칸 프리셋 ${store.list(key).length}개${note ? ` · ${note}` : ""}`);
  }
  function toggleCloseAfter() {
    closeAfter = !closeAfter;
    prefsRead = true;
    store.setPrefs({ close_after_insert: closeAfter }).catch((err) => console.warn("[AI Studio] preset prefs not saved:", err));
    say(closeAfter ? "넣으면 닫기 켬: 바꾸기 · 붙이기 한 번에 창이 닫혀요." : "넣으면 닫기 끔");
  }

  // ---- keys, closing
  function onKey(event) {
    event.stopPropagation(); // keys typed here are not ComfyUI shortcuts
    if (closed) return;
    if (event.isComposing || event.keyCode === 229) return; // Korean IME composing: Enter / Esc belong to it
    const key = event.key;
    if (key === "Tab") return trapTab(dialog, event);
    const command = event.ctrlKey || event.metaKey;
    const inEditor = event.target === detail.nameInput || event.target === detail.textInput;
    let handled = true;
    if (key === "Escape") {
      if (saving) cancelSave();
      else askClose();
    } else if (command && (key === "f" || key === "F")) {
      head.search.focus();
      head.search.select();
    } else if (key === "Enter" && event.target === bar.nameInput) commitSave();
    else if (inEditor && command && (key === "s" || key === "S" || key === "Enter")) saveDraft();
    else handled = false;
    if (handled) event.preventDefault();
  }
  const stop = (event) => event.stopPropagation();
  let pressedOutside = false;
  backdrop.addEventListener("mousedown", (event) => { pressedOutside = event.target === backdrop; });
  backdrop.addEventListener("click", (event) => {
    if (pressedOutside && event.target === backdrop) askClose(); // a drag that started inside (selecting text) does not close
    pressedOutside = false;
  });
  backdrop.addEventListener("keydown", onKey);
  for (const type of ["keyup", "keypress", "paste", "copy", "cut"]) backdrop.addEventListener(type, stop);

  /** Esc, the ✕ and a click on the dimmed page: with unsaved edits in the editors the first one only says so. */
  function askClose() {
    const preset = selectedPreset();
    if (isDirty(preset)) {
      const warn = `‘${preset.name}’을(를) 고친 내용을 저장하지 않았어요. 한 번 더 누르면 저장하지 않고 닫혀요.`;
      if (closeAsked !== warn || message !== warn) { // not asked yet, or something else was done or typed since
        say(warn);
        closeAsked = warn;
        return;
      }
    }
    close();
  }
  function close() {
    if (closed) return;
    closed = true;
    unsubscribe();
    backdrop.remove();
    if (openWindow === handle) openWindow = null;
    const back = opener && opener !== doc.body && opener.isConnected ? opener : returnFocus?.(); // the panel redraws its labels
    if (typeof back?.focus === "function") back.focus({ preventScroll: true });
    onClose?.();
  }

  const unsubscribe = store.subscribe(render);
  const handle = { close };
  openWindow = handle;
  doc.body.append(backdrop);
  const start = boxStatus(list(), boxText(sec));
  message = start.preset ? `${labelOf(sec)} 칸이 프리셋 ‘${start.preset.name}’과 같아요.` : store.loaded ? `${labelOf(sec)} 칸 프리셋 ${list().length}개` : "";
  render();
  head.search.focus();
  store.load(true).then(() => {
    if (closed || message) return;
    const now = boxStatus(list(), boxText(sec));
    say(now.preset ? `${labelOf(sec)} 칸이 프리셋 ‘${now.preset.name}’과 같아요.` : `${labelOf(sec)} 칸 프리셋 ${list().length}개`);
  }, () => {}); // render() shows store.error
  return handle;
}
