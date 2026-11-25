// LoRA picker window chrome: the header bar (title, mode, search, 모든 모델, 정렬, 추가하면 닫기, ↻, ✕, and the "폴더" button of
// a narrow window) and the bottom bar ("이번에 추가" chips with ✕, the shortcut hint, 완료/닫기). Built once, updated in place,
// so the search box keeps its focus and caret while the rest of the window redraws.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { h } from "./studio_dom.js";
import { middleEllipsis } from "./lora_names.js";
import { familyLabel } from "./lora_browser_detail.js";

const MODE_LABELS = { add: "추가", replace: "바꾸기", xyz: "XYZ 에폭 비교" };
const SORTS = [["date", "최근 학습"], ["name", "이름"], ["group", "작가 폴더"], ["recent", "최근 사용"]];
const KEY_HELP = "↑↓←→ 카드 · [ ] 에폭 · Enter 추가하고 닫기 · Shift+Enter 추가하고 계속 · F 즐겨찾기 · Ctrl+F 검색 · Esc 닫기";
const FOOT_NOTES = {
  add: "'+ 추가' 를 누를 때마다 패널에 바로 들어가고 여기 쌓여요 (✕ = 그 줄 빼기)",
  replace: "고르면 그 줄이 바뀌고 창이 닫혀요",
  xyz: "카드를 고르고 에폭 칩을 체크한 뒤 'XYZ 에 넣기'",
};
const SVG = (body) => `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">${body}</svg>`;
const SEARCH_SVG = SVG('<circle cx="11" cy="11" r="7"/><line x1="16.5" y1="16.5" x2="21" y2="21"/>');
const REFRESH_SVG = SVG('<path d="M20 11a8 8 0 1 0-2.3 5.7"/><polyline points="20 4 20 11 13 11"/>');
const CLOSE_SVG = SVG('<line x1="6" y1="6" x2="18" y2="18"/><line x1="18" y1="6" x2="6" y2="18"/>');

/** An `<svg>` from the static markup above (no user text). */
function svg(markup) {
  const template = document.createElement("template");
  template.innerHTML = markup;
  return template.content.firstElementChild;
}

/**
 * The header. `on = {query(text), showAll(), sort(by), closeAfter(), tree(), refresh(), close()}`; the "추가하면 닫기" toggle
 * is there in add mode only. `update({showAll, hidden, sort, closeAfter, treeOpen})`.
 */
export function headerBar({ mode, family, on }) {
  const search = h("input", { type: "text", class: "ais-lb-search", placeholder: "검색 — 예: bjun 14, @gt dune   (Ctrl+F)", "aria-label": "LoRA 검색",
    autocomplete: "off", onInput: (e) => on.query(e.target.value) });
  search.spellcheck = false; // h() skips false props
  const allBtn = h("button", { type: "button", class: "ais-lb-toggle", title: "다른 모델 계열로 학습한 LoRA 도 보기", onClick: () => on.showAll() });
  const sortSel = h("select", { class: "ais-lb-select", "aria-label": "정렬", onChange: (e) => on.sort(e.target.value) },
    SORTS.map(([value, text]) => h("option", { value }, text)));
  const closeAfterBtn = mode === "add"
    ? h("button", { type: "button", class: "ais-lb-toggle ais-lb-small", title: "끄면 여러 개를 연달아 추가하고 아래 '완료'로 닫아요", onClick: () => on.closeAfter() })
    : null;
  const treeBtn = h("button", { type: "button", class: "ais-lb-toggle ais-lb-treebtn", onClick: () => on.tree() }, "폴더");
  const iconBtn = (label, markup, onClick) => h("button", { type: "button", class: "ais-lb-icon", "aria-label": label, title: label, onClick }, svg(markup));
  const toggle = (button, on, text) => {
    button.textContent = text;
    button.setAttribute("aria-pressed", String(on));
    button.classList.toggle("on", on);
  };
  const bar = {
    search,
    el: h("div", { class: "ais-lb-head" },
      h("h1", {}, "LoRA 고르기"), h("span", { class: "ais-lb-mode" }, MODE_LABELS[mode] || mode), treeBtn,
      h("label", { class: "ais-lb-searchbox" }, svg(SEARCH_SVG), search),
      family ? h("span", { class: "ais-lb-dim ais-lb-nowrap" }, `체크포인트 ${familyLabel(family)}`) : null, allBtn,
      h("label", { class: "ais-lb-sort" }, "정렬", sortSel), closeAfterBtn,
      iconBtn("다시 읽기 (새 에폭 반영)", REFRESH_SVG, () => on.refresh()), iconBtn("닫기 (Esc)", CLOSE_SVG, () => on.close())),
    update({ showAll, hidden, sort, closeAfter, treeOpen }) {
      toggle(allBtn, showAll, showAll ? "모든 모델 켬" : `모든 모델 · 숨김 ${hidden}`);
      if (sortSel.value !== sort) sortSel.value = sort;
      if (closeAfterBtn) toggle(closeAfterBtn, closeAfter, `추가하면 닫기 · ${closeAfter ? "켬" : "끔"}`);
      treeBtn.setAttribute("aria-expanded", String(treeOpen));
    },
  };
  bar.update({ showAll: false, hidden: 0, sort: "name", closeAfter: false, treeOpen: false }); // until the index arrives
  return bar;
}

/**
 * The bottom bar. `on = {unpick(key), done()}`; `update(added)` with the window's `state.added` (add mode shows them as chips
 * whose ✕ takes that row out of the panel again; "완료 · N개 추가됨" closes).
 */
export function footBar({ mode, on }) {
  const tray = h("div", { class: "ais-lb-tray" });
  const doneBtn = h("button", { type: "button", class: "ais-lb-done", onClick: () => on.done() }, "닫기");
  const chip = (a) => h("span", { class: "ais-lb-added", title: a.name }, `${middleEllipsis(a.label, 44)} · ${Number(a.strength).toFixed(2)}`,
    h("button", { type: "button", "aria-label": `빼기: ${a.label}`, title: "패널에서 이 줄 빼기", "data-k": `x:${a.key}`, onClick: () => on.unpick(a.key) }, "✕"));
  const bar = {
    el: h("div", { class: "ais-lb-foot" },
      mode === "add" ? h("span", { class: "ais-lb-dim ais-lb-nowrap" }, "이번에 추가") : null, tray,
      h("span", { class: "ais-lb-keys", title: KEY_HELP }, "단축키도 있음"), doneBtn),
    update(added) {
      const many = mode === "add" && added.length > 0;
      tray.replaceChildren(...(many ? added.map(chip) : [h("span", { class: "ais-lb-dim" }, FOOT_NOTES[mode] || "")]));
      doneBtn.textContent = many ? `완료 · ${added.length}개 추가됨` : "닫기";
    },
  };
  bar.update([]);
  return bar;
}
