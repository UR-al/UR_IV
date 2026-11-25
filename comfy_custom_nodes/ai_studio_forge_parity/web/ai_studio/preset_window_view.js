// Prompt-box preset window, the parts: header (title, search, "넣으면 닫기", ✕, the box tabs), the bar over the cards
// ("+ 지금 칸 내용 저장" with its inline name box), the card grid, the detail pane and the foot line. Each part is built once
// and updated in place — only the cards are redrawn — so a text box keeps its focus and caret while the window redraws.
// preset_window.js wires them. Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { h } from "./studio_dom.js";
import { presetIcon } from "./preset_dom.js";
import { PRESET_FILE_NOTE } from "./preset_logic.js";

const button = (cls, text, onClick, props = {}) => h("button", { type: "button", class: cls, onClick, ...props }, text);

/**
 * `tabs` = [{key, label}] (fixed). `on = {query(text), closeAfter(), close(), tab(key)}`.
 * `update({label, closeAfter, sec, counts: {key: n}})`.
 */
export function headBar({ tabs, on }) {
  const title = h("h1", {});
  const search = h("input", { type: "text", class: "ais-pp-search", placeholder: "이름이나 내용으로 검색   (Ctrl+F)", "aria-label": "프리셋 검색",
    autocomplete: "off", onInput: (e) => on.query(e.target.value) });
  search.spellcheck = false; // h() skips false props
  const closeAfter = button("ais-pp-toggle", "넣으면 닫기", () => on.closeAfter(), { title: "켜면 바꾸기 · 붙이기 한 번에 창이 닫혀요" });
  const tabButtons = tabs.map(({ key, label }) => {
    const count = h("span", { class: "ais-pp-count" });
    return { key, count, el: button("ais-pp-tab", [label, count], () => on.tab(key), { role: "tab", "data-k": `t:${key}` }) };
  });
  return {
    search,
    el: h("div", { class: "ais-pp-head" },
      h("div", { class: "ais-pp-top" },
        title, h("span", { class: "ais-pp-hint" }, "칸 이름을 눌러 열림"),
        h("label", { class: "ais-pp-searchbox" }, presetIcon("search"), search),
        closeAfter,
        button("ais-pp-icon", presetIcon("close"), () => on.close(), { "aria-label": "창 닫기 (Esc)", title: "닫기 (Esc)" })),
      h("div", { class: "ais-pp-tabs", role: "tablist", "aria-label": "프롬프트 칸" }, tabButtons.map((tab) => tab.el))),
    update({ label, closeAfter: closing, sec, counts }) {
      title.textContent = `${label} 프리셋`;
      closeAfter.setAttribute("aria-pressed", String(closing));
      closeAfter.classList.toggle("on", closing);
      for (const tab of tabButtons) {
        tab.el.setAttribute("aria-selected", String(tab.key === sec));
        tab.el.classList.toggle("on", tab.key === sec);
        tab.count.textContent = String(counts[tab.key] ?? 0);
      }
    },
  };
}

/** The bar over the cards. `on = {start(), commit(), cancel()}`; `update({summary, saving, placeholder})`. */
export function gridBar({ on }) {
  const summary = h("span", { class: "ais-pp-summary" });
  const start = button("ais-pp-savenew", "+ 지금 칸 내용 저장", () => on.start(), { title: "지금 칸에 적힌 내용을 새 프리셋으로" });
  const nameInput = h("input", { type: "text", class: "ais-pp-newname", autocomplete: "off" });
  nameInput.spellcheck = false;
  const info = h("div", { class: "ais-pp-bar-info" }, summary, start);
  const form = h("div", { class: "ais-pp-bar-form", hidden: true },
    h("label", { class: "ais-pp-newlabel" }, "새 프리셋 이름", nameInput),
    button("ais-pp-commit", "저장", () => on.commit()), button("ais-pp-btn ais-pp-cancel", "취소", () => on.cancel()));
  return {
    nameInput, start,
    el: h("div", { class: "ais-pp-bar" }, info, form),
    update({ summary: text, saving, placeholder }) {
      summary.textContent = text;
      nameInput.placeholder = placeholder;
      info.hidden = saving;
      form.hidden = !saving;
    },
  };
}

/**
 * The card grid. `on = {select(id), replace(id), append(id)}`; `draw(cards, empty)` with
 * `cards = [{id, name, text, meta, selected, current}]`, `empty` = the line shown instead of cards ("" = none).
 */
export function cardGrid({ on }) {
  const grid = h("div", { class: "ais-pp-grid" });
  const emptyLine = h("div", { class: "ais-pp-empty", hidden: true });
  const card = (c) => h("div", { class: `ais-pp-card${c.selected ? " sel" : ""}${c.current ? " cur" : ""}` },
    h("button", { type: "button", class: "ais-pp-card-main", "data-k": `c:${c.id}`, "aria-label": `${c.name} 고르기`, "aria-pressed": String(c.selected), onClick: () => on.select(c.id) },
      h("span", { class: "ais-pp-card-head" },
        h("span", { class: "ais-pp-card-name" }, c.name), c.current ? h("span", { class: "ais-pp-badge" }, "지금 칸") : null),
      h("span", { class: "ais-pp-card-text" }, c.text),
      h("span", { class: "ais-pp-card-meta" }, c.meta)),
    h("div", { class: "ais-pp-card-foot" },
      button("ais-pp-replace", "바꾸기", () => on.replace(c.id), { "data-k": `r:${c.id}`, title: "칸 내용을 이 프리셋으로 바꾸기" }),
      button("ais-pp-append", "+ 붙이기", () => on.append(c.id), { "data-k": `a:${c.id}`, title: "칸 내용 뒤에 붙이기" })));
  return {
    el: h("div", { class: "ais-pp-scroll" }, emptyLine, grid),
    draw(cards, empty) {
      emptyLine.textContent = empty;
      emptyLine.hidden = !empty;
      grid.replaceChildren(...cards.map(card));
    },
  };
}

/**
 * The right pane: "지금 <칸> 칸" (read-only text, status badge, 칸 비우기) and the selected preset's editors and buttons.
 * `on = {clear(), draft(), save(), use(), add(), move(delta), remove()}`. `update({label, box, status: {kind, text}, preset,
 * dirty, first, last})`; `fill(preset)` puts a preset's name and text into the editors (the caller decides when: a new
 * selection, never while the user types).
 */
export function detailPane({ on }) {
  const cap = h("span", { class: "ais-pp-cap" });
  const status = h("span", { class: "ais-pp-status" });
  const current = h("div", { class: "ais-pp-current" });
  const clear = button("ais-pp-btn ais-pp-clear", "칸 비우기", () => on.clear());
  const nameInput = h("input", { type: "text", class: "ais-pp-name", autocomplete: "off", onInput: () => on.draft() });
  const textInput = h("textarea", { class: "ais-pp-text", rows: 9, onInput: () => on.draft() });
  nameInput.spellcheck = false;
  textInput.spellcheck = false;
  const save = button("ais-pp-savedraft", "고친 내용 저장", () => on.save(), { title: "이름 · 내용을 고친 대로 프리셋에 저장 (Ctrl+S)" });
  const up = button("ais-pp-btn ais-pp-up", "↑ 위로", () => on.move(-1), { "aria-label": "위로" });
  const down = button("ais-pp-btn ais-pp-down", "↓ 아래로", () => on.move(1), { "aria-label": "아래로" });
  const edit = h("div", { class: "ais-pp-edit" },
    h("span", { class: "ais-pp-cap" }, "고르는 프리셋"),
    h("label", { class: "ais-pp-field" }, "이름", nameInput),
    h("label", { class: "ais-pp-field" }, "내용", textInput),
    h("div", { class: "ais-pp-btns" },
      save, button("ais-pp-use", "이걸로 바꾸기", () => on.use(), { title: "칸 내용을 여기 적힌 내용으로 바꾸기" }),
      button("ais-pp-btn ais-pp-add", "+ 붙이기", () => on.add(), { title: "칸 내용 뒤에 여기 적힌 내용 붙이기" })),
    h("div", { class: "ais-pp-btns" }, up, down, button("ais-pp-delete", "삭제", () => on.remove())));
  const none = h("div", { class: "ais-pp-none", hidden: true }, "카드를 누르면 여기서 이름과 내용을 고칠 수 있어요.");
  return {
    nameInput, textInput,
    el: h("aside", { class: "ais-pp-detail", "aria-label": "지금 칸과 고르는 프리셋" },
      h("div", { class: "ais-pp-dwrap" },
        h("div", { class: "ais-pp-now" }, h("div", { class: "ais-pp-row2" }, cap, status), current, clear),
        h("div", { class: "ais-pp-sep" }), edit, none)),
    fill(preset) {
      nameInput.value = preset?.name ?? "";
      textInput.value = preset?.text ?? "";
    },
    update({ label, box, status: state, preset, dirty, first, last }) {
      cap.textContent = `지금 ${label} 칸`;
      status.textContent = state.text;
      status.className = `ais-pp-status ${state.kind}`;
      current.textContent = state.kind === "empty" ? "(비어 있음)" : box;
      current.classList.toggle("dim", state.kind === "empty");
      clear.disabled = box === "";
      edit.hidden = !preset;
      none.hidden = Boolean(preset);
      save.disabled = !dirty;
      up.disabled = first;
      down.disabled = last;
    },
  };
}

/** The foot line. `on = {undo()}`; `update({msg, canUndo})`. */
export function footBar({ on }) {
  const msg = h("span", { class: "ais-pp-msg", role: "status", "aria-live": "polite" });
  const undo = button("ais-pp-btn ais-pp-undo", "되돌리기", () => on.undo(), { hidden: true, title: "방금 한 일을 되돌리기" });
  return {
    el: h("div", { class: "ais-pp-foot" }, msg, undo, h("span", { class: "ais-pp-path" }, PRESET_FILE_NOTE)),
    update({ msg: text, canUndo }) {
      msg.textContent = text;
      undo.hidden = !canUndo;
    },
  };
}
