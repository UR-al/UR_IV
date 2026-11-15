// LoRA picker window, right-hand side: the detail pane of the selected run (epoch chips with board stars, the epoch's training
// samples, trigger, training info) and the bottom action bar (strength −/+, slider, number and the add / replace / XYZ button),
// plus the small label helpers the cards share. Drawn with `h()` only; data comes in, clicks go out through `on`.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { h } from "./studio_dom.js";

const FAMILY_LABELS = { "2.9b": "2.9B", "3.8b": "3.8B", base: "Base", moss: "MOSS", krea2: "Krea2" };
const ALGO_LABELS = { lokr: "LoKr", lora: "LoRA", locon: "LoCon", loha: "LoHa", dora: "DoRA", full: "Full", tsm: "TSM", ia3: "IA3" };
const COVER_CAPTIONS = { custom: "직접 지정한 미리보기", output: "최근 출력 그림 · EXIF 에 이 LoRA", sample: "마지막 에폭 샘플" };

export const familyLabel = (family) => (family ? FAMILY_LABELS[family] || String(family) : "계열 모름");
export const algoLabel = (algo) => (algo ? ALGO_LABELS[String(algo).toLowerCase()] || String(algo) : "");

/** "e14" / "e15 최종" / "최종" / "" (a lone file without an epoch). */
export function epochText(ep) {
  if (!ep) return "";
  if (ep.final) return ep.epoch != null ? `e${ep.epoch} 최종` : "최종";
  return ep.epoch != null ? `e${ep.epoch}` : "";
}

/** Two letters for a card without a picture: the run part of the label, without the leading "@". */
export function initialsOf(label) {
  const run = String(label || "").split(" · ").pop().replace(/^@/, "");
  return Array.from(run).slice(0, 2).join("").toUpperCase() || "?";
}

/**
 * A picture over a text placeholder. A missing file (the routes answer 404) removes the `<img>`, so the placeholder shows
 * and never a broken-image icon; `loading="lazy"` leaves off-screen pictures unfetched.
 */
export function safeImg(url, fallback, className, ...extra) {
  const box = h("div", { class: className }, h("span", { class: "ais-lb-ph" }, fallback), extra);
  if (url) {
    const img = h("img", { alt: "", loading: "lazy", decoding: "async" });
    img.addEventListener("error", () => img.remove());
    img.src = url;
    box.append(img);
  }
  return box;
}

/** Card badges: model family, algorithm, precision, epoch count, "다운로드" (warning colour for unknown family / downloads). */
export function runBadges(info, family) {
  const list = [
    { t: familyLabel(family), warn: !family },
    { t: algoLabel(info.algo) },
    { t: info.precision && info.precision !== "no" ? info.precision : "" },
    { t: info.numEpochs ? `${info.numEpochs}에폭` : "" },
    { t: info.downloaded ? "다운로드" : "", warn: true },
  ];
  return list.filter((b) => b.t).map((b) => h("span", { class: `ais-lb-badge${b.warn ? " warn" : ""}` }, b.t));
}

/** "LoKr + DoRA · CAME 2e-06" from the header summary (Task 2 `header_summary`). */
function methodText(s) {
  const args = s.network_args && typeof s.network_args === "object" ? s.network_args : {};
  const dora = ["dora_wd", "dora", "use_dora"].some((k) => String(args[k]).toLowerCase() === "true");
  const optimizer = s.optimizer ? String(s.optimizer).split("(")[0].split(".").pop() : "";
  const head = [algoLabel(s.algo), dora ? "DoRA" : ""].filter(Boolean).join(" + ");
  return [head, [optimizer, s.lr].filter(Boolean).join(" ")].filter(Boolean).join(" · ");
}

function infoRows(run, info, detail) {
  const s = detail && !detail.error ? detail.summary : null;
  if (!s) return [["모델", familyLabel(run.family)], ["정보", detail?.error ? "못 읽었어요" : "읽는 중…"]];
  const trained = [s.date, s.num_epochs ? `${s.num_epochs}에폭` : "", s.steps ? `${s.steps}스텝` : ""].filter(Boolean).join(" · ");
  return [
    ["모델", s.base_model || familyLabel(run.family)],
    ["방식", methodText(s) || algoLabel(info.algo) || "—"],
    ["정밀도", s.precision || "—"],
    ["학습", trained || "—"],
    s.dataset_dirs?.length ? ["데이터", s.dataset_dirs.join(", ")] : null,
    s.has_meta ? null : ["메타", "정보 없음"],
  ].filter(Boolean);
}

function preview(ep, detail, labels, thumb, clearPreview) {
  const d = detail && !detail.error ? detail : null;
  if (!d) return h("div", { class: "ais-lb-nocover" }, detail?.error ? "정보를 못 읽었어요" : "읽는 중…");
  const hit = ep.epoch != null ? (d.epochs || []).find((e) => e.epoch === ep.epoch) : null;
  const count = Math.min(4, Number(hit?.samples) || 0);
  const clear = d.cover === "custom" && clearPreview
    ? h("button", { type: "button", class: "ais-lb-link", "data-k": "clear-preview", onClick: clearPreview }, "미리보기 지정 해제")
    : null;
  if (count > 0) {
    const tiles = Array.from({ length: count }, (_, i) =>
      safeImg(thumb(ep.name, "sample", ep.epoch, i), "샘플 없음", "ais-lb-sample", h("span", { class: "ais-lb-cap" }, `e${ep.epoch} · ${labels[i] || i + 1}`)));
    return [h("div", { class: "ais-lb-samples" }, tiles), clear];
  }
  if (d.cover) {
    const caption = COVER_CAPTIONS[d.cover] || "미리보기";
    return [safeImg(thumb(ep.name, "cover"), "미리보기 없음", "ais-lb-bigcover", h("span", { class: "ais-lb-cap" }, caption)), clear];
  }
  return h("div", { class: "ais-lb-nocover" }, h("span", {}, "미리보기 없음"), h("span", {}, "갤러리 그림의 'LoRA 미리보기로' 로 지정할 수 있어요"));
}

/**
 * The detail pane of the selected run. `ctx = {run, ep, info, detail, mode, checked, stars, labels, thumb, on}`:
 * `run` from visibleRuns, `ep` its chosen epoch file, `info` the card info, `detail` the /loras/detail payload (`null` while
 * loading, `{error}` when it failed), `stars`/`labels` the index's board stars / prompt labels of the run (fallback until the
 * detail arrives), `thumb(name, kind, epoch, i) -> url`, `on = {epoch(ep), trigger(text)?, clearPreview()?}`.
 */
export function detailView(ctx) {
  const { run, ep, info = {}, detail, mode, checked = [], thumb, on } = ctx;
  if (!run || !ep) return h("div", { class: "ais-lb-none" }, "카드를 고르면 여기에 에폭과 샘플이 보여요");
  const xyz = mode === "xyz";
  const d = detail && !detail.error ? detail : null;
  const starred = new Set([...(ctx.stars || []), ...(d?.epochs || []).filter((e) => e.starred).map((e) => e.epoch)]);
  const labels = d?.prompt_labels?.length ? d.prompt_labels : ctx.labels || [];
  const isOn = (e) => (xyz ? checked.includes(e.stem) : e.stem === ep.stem);
  const lone = run.epochs.length === 1 && !epochText(run.epochs[0]);
  const chips = run.epochs.map((e) =>
    h("button", {
      type: "button", class: `ais-lb-chip${isOn(e) ? " on" : ""}`, "aria-pressed": isOn(e) ? "true" : "false", "data-k": `c:${e.stem}`,
      title: e.name, onClick: () => on.epoch(e),
    }, (epochText(e) || "최종") + (e.epoch != null && starred.has(e.epoch) ? " ★" : "")));
  const epochHead = xyz ? `비교할 에폭 체크 (${run.epochs.filter(isOn).length}개)` : "에폭";
  const trigger = String(d?.trigger || info.trigger || "").trim();
  return h("div", { class: "ais-lb-dbody" },
    h("div", { class: "ais-lb-dhead" },
      h("h2", {}, [run.label, xyz ? "" : epochText(ep)].filter(Boolean).join(" · ")),
      h("span", { class: "ais-lb-path" }, ep.name)),
    h("div", { class: "ais-lb-epochs" },
      h("div", { class: "ais-lb-row2" }, h("span", {}, epochHead), starred.size ? h("span", {}, "★ = 보드에서 별 찍은 에폭") : null),
      lone ? h("span", { class: "ais-lb-dim" }, "에폭 정보 없음 (파일 하나)") : h("div", { class: "ais-lb-chips" }, chips)),
    preview(ep, detail, labels, thumb, on.clearPreview),
    trigger ? h("div", { class: "ais-lb-trigger" },
      h("span", { class: "ais-lb-dim" }, "트리거"), h("code", {}, trigger),
      on.trigger ? h("button", { type: "button", class: "ais-lb-btn", "data-k": "trigger", onClick: () => on.trigger(trigger) }, "작가 칸에 넣기") : null) : null,
    h("dl", { class: "ais-lb-info" }, infoRows(run, info, detail).map(([k, v]) => [h("dt", {}, k), h("dd", { title: v }, v)])),
  );
}

/**
 * The bottom of the detail pane, built once and updated in place so the slider keeps working while it is dragged.
 * add/replace: strength (−/+ by 0.05, slider 0..2, number) + the main button; xyz: a note + "XYZ 에 N개 넣기".
 * `on = {strength(value), step(delta), commit()}`; `update({strength, text, disabled, title})`. `opts.strength: false` leaves
 * the strength controls out (a replace from an XYZ row, whose strength the window does not set).
 */
export function actionBar(mode, on, opts = {}) {
  const xyz = mode === "xyz";
  const withStrength = !xyz && opts.strength !== false;
  let shown = 1; // the strength last drawn (state value, already clamped)
  const range = h("input", { type: "range", class: "ais-lb-range", min: "0", max: "2", step: "0.05", "aria-label": "세기",
    onInput: (e) => on.strength(Number(e.target.value)) });
  const num = h("input", { type: "number", class: "ais-lb-num", min: "0", max: "2", step: "0.05", "aria-label": "세기 값",
    onChange: (e) => {
      const value = Number(e.target.value);
      if (e.target.value !== "" && Number.isFinite(value)) on.strength(value);
      num.value = shown.toFixed(2); // show the clamped value (or put back the old one after a bad entry)
    } });
  const step = (delta, text, label) => h("button", { type: "button", class: "ais-lb-step", "aria-label": label, title: label, onClick: () => on.step(delta) }, text);
  const strength = !withStrength ? null : h("div", { class: "ais-lb-strength" },
    h("span", { class: "ais-lb-dim" }, "세기"), step(-0.05, "−", "세기 0.05 내리기"), range, step(0.05, "+", "세기 0.05 올리기"), num);
  const note = xyz ? h("div", { class: "ais-lb-dim" }, "체크한 에폭이 XYZ LoRA 줄로 하나씩 들어가요 · 기준 줄은 그대로") : null;
  const main = h("button", { type: "button", class: "ais-lb-main", onClick: () => on.commit() });
  return {
    el: h("div", { class: "ais-lb-actions" }, note, strength, main),
    update({ strength: value = 1, text, disabled = false, title = "" }) {
      if (withStrength) {
        shown = Number(value);
        range.value = String(shown);
        if (num.ownerDocument.activeElement !== num) num.value = shown.toFixed(2); // never under the user's typing
      }
      main.textContent = text;
      main.disabled = disabled;
      main.title = title;
    },
  };
}
