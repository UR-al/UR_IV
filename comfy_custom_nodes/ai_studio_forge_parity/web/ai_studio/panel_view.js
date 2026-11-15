// Anima studio: the control panel (C4 left column, 790px) drawn into AIStudioPanel's DOM widget.
// Its state lives as JSON in the node's hidden `state` STRING widget, so it is saved with the workflow.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { dropBlockedExt, extMemory, renderExtensions } from "./ext_view.js";
import { nodeInstalled, referenceBlock, renderReference } from "./reference_view.js";
import { createAutocomplete } from "./studio_autocomplete.js";
import { addSession, listedStem, replaceRow, rowFace } from "./lora_rows.js";
import { hiresCard, hiresSizeText, postCard } from "./post_view.js";
import { labelMark } from "./preset_logic.js";
import { FAMILY_LABELS } from "./studio_ext.js";
import { BOX_LABELS, BOX_SWITCHES, CROSS_ATTN_MODES, DORA_MODES, INPAINT_FILLS, LORA_BASE_MAX, MOSS_SKIP_LAYER, PROMPT_SWITCHES, defaultModel, inpaintByDetailer, familyHasNegpip, imageSize, modelFamily, negpipKey, pickCheckpoint, sizeStep, withDefaults } from "./studio_logic.js";
import { pulledState } from "./studio_settings.js";
import { autoGrow, card, guardCanvasKeys, h, numberBox, optionItems, pill, row, select } from "./studio_dom.js";

const CHOICE_KINDS = ["unets", "tes", "vaes", "samplers", "schedulers", "loras", "upscalers", "sam3", "controlnets", "face_models", "attentions", "wildcards"];
// Model picks filled with an Anima default when their list arrives empty-handed (an empty pick would run the
// loader's first file; the run controller refuses a run while one is still empty).
const MODEL_KEYS = { unets: "unet", tes: "te", vaes: "vae" };
const NEGPIP_MISSING = "불가: ComfyUI-ppm 노드 팩(CLIPNegPip)이 설치돼 있지 않음";
const NEGPIP_OLD_GRAPH = "불가: 이 워크플로에는 NegPiP 노드가 없음. AI Studio Anima.json 을 새로 여세요";
const NEGPIP_NO_FAMILY = "불가: Krea2 는 가중치를 못 읽음. 빼고 싶은 것은 네거티브에";
const NEGPIP_TITLE = { anima: "음수 가중치 (단어:-1.0)", moss: "음수 가중치 (단어:-1.0). 켜면 (단어:1.2) 도 읽고, 글의 괄호도 가중치가 됨: 글자 괄호는 \\( \\)" };
const STREAM_MISSING = "불가: ComfyUI-MOSS-Toolkit 노드 팩(MOSSStreamFP32)이 설치돼 있지 않음";
const STREAM_OLD_GRAPH = "불가: 이 워크플로에는 fp32 노드가 없음. AI Studio Anima.json 을 새로 여세요";
const SKIP_OLD_GRAPH = "불가: 이 워크플로에는 MOSS 노드가 없음. AI Studio Anima.json 을 새로 여세요";
const SKIP_LABEL = `스킵 ${MOSS_SKIP_LAYER}`;
const SKIP_TITLE =
  `MOSS 공식 샘플링: 네거티브(uncond) 패스에서 ${MOSS_SKIP_LAYER}번 블록을 건너뜀 (전 구간, 모든 패스). ` +
  "끄면 1.19 까지의 그림. CFG 1 에서는 네거티브를 안 돌려서 차이 없음";
// The 스텝 row's switches of the two fixed boxes: [box, pill text, tooltip]; off, the box keeps its text. The same two flags
// as those boxes' switches on the prompt card: either one moves both.
const STEP_BOX_SWITCHES = [
  ["prompt_prefix", "선행", "선행 고정 칸을 프롬프트에 넣기. 끄면 칸의 글은 두고 프롬프트에서만 뺌 (XYZ·와일드카드·저장 정보도). 프롬프트 카드의 스위치와 같음"],
  ["prompt_suffix", "후행", "후행 고정 칸을 프롬프트에 넣기. 끄면 칸의 글은 두고 프롬프트에서만 뺌 (XYZ·와일드카드·저장 정보도). 프롬프트 카드의 스위치와 같음"],
];
// The prompt card's switch left of every box's name (studio_logic BOX_SWITCHES): what it says on mouse-over, and its row when off.
const BOX_SWITCH_TITLE = "프롬프트에 넣기/빼기. 끄면 칸의 글은 두고 프롬프트에서만 뺌 (XYZ·와일드카드·저장 정보도)";
const NEGATIVE_SWITCH_TITLE = "네거티브 넣기/빼기. 끄면 칸의 글은 두고 빈 네거티브로 실행 (XYZ·와일드카드·저장 정보도)";
const BOX_OFF_TITLE = "꺼짐: 프롬프트에 안 들어감 (이름 왼쪽 스위치로 켬)";
const NEGATIVE_OFF_TITLE = "꺼짐: 네거티브 없이 실행 (이름 왼쪽 스위치로 켬)";
const INPAINT_OLD_GRAPH = "불가: 이 워크플로에는 인페인트 노드가 없음. AI Studio Anima.json 을 새로 여세요";
const INPAINT_AREA_TITLE = "전체: 그림 전체를 위 크기로 다시 그림. 마스크만: 마스크 둘레를 잘라 위 크기로 그리고 원본에 붙임 (Forge의 Only masked)";
const ATTENTION_OLD_GRAPH = "불가: 이 워크플로에는 어텐션 노드가 없음. AI Studio Anima.json 을 새로 여세요";
const LORA_CTL_OLD_GRAPH = "불가: 이 워크플로에는 LoRA 제어 노드가 없음. AI Studio Anima.json 을 새로 여세요";
const LORA_BASE_TITLE =
  "메인 샘플링의 처음 N% 스텝은 LoRA 없이(Base), 나머지는 LoRA로. 구도는 Base가 프롬프트대로 잡고 화풍은 LoRA가 입힌다. " +
  "Hires·SAM3·후처리는 LoRA 그대로, 텍스트 인코더 LoRA는 늘 적용. SPEED·Spectrum·Restart·UniPC와는 같이 못 씀. 0 = 끔";
const LORA_BASE_INPAINT = "인페인트는 메인 샘플러를 안 써서 Base 먼저가 듣지 않음 (다른 탭에서는 그대로)";
const LORA_CROSS_FAMILY = "Anima 에서만 들음 (MOSS·Krea2 에는 교차 어텐션이 없음)";
const LORA_CROSS_TITLE =
  "LoRA가 Anima의 글자→그림 경로(교차 어텐션, LLM 어댑터)는 건드리지 않게 한다(+변조 = 교차 어텐션의 adaLN 변조까지). 모든 패스에 적용";
const ATTENTION_FORGE = "Forge처럼: PyTorch 어텐션";
const ATTENTION_LABELS = { launch: "실행 옵션대로", pytorch: "PyTorch", sage: "Sage", flash: "Flash", comfy_kitchen_int8: "Kitchen" };
const FILL_LABELS = { original: "원본", fill: "채우기", "latent noise": "latent noise", "latent nothing": "latent nothing" };

// "Forge처럼" items; greyed ones follow studio_forge_parity.md (불가 rows).
const FORGE_ITEMS = [
  { key: "noise", label: "노이즈", title: "초기·SDE 노이즈를 Forge처럼 (그림마다 시드 + i)" },
  { key: "prompt_syntax", label: "프롬프트 편집", title: "[a:b:0.5] · [a|b] 를 Forge처럼 스텝 구간으로" },
  { key: "dora", label: "DoRA", title: "DoRA 합치기를 학습과 같은 LyCORIS로 (실제 Forge 설정)" },
  { key: "attention", label: "어텐션", title: "실행 옵션과 상관없이 PyTorch 어텐션으로 (실제 Forge 설정)" },
];

const group = (title, color, extraClass, ...children) =>
  h("div", { class: `ais-grp ${extraClass}`, style: `--c:${color}` }, title ? h("div", { class: "ais-gh" }, title) : null, children);

export class PanelView {
  /**
   * `widget`: the node's hidden `state` widget. `ctx`: `choices(kind)` -> Promise<{items}>, `fetchChoices(class)` ->
   * Promise<`/object_info/<class>` input>, `hasRole(role)`, `generate()`, `stop()`, `changed()`, optional `tags(query, limit)` ->
   * Promise<[{name, category, count, alias?, ko?}]> (prompt tag autocomplete), `pickedSeed()` -> Promise<seed of the image
   * picked in the gallery, or null>, `engineRoom` {collapsed(),
   * toggle()}, `i2i` {get(), set(value), url(value), upload(file) -> {name, width, height}, maskEditor()},
   * `reference`, `reference2`, `reference3` {url(value), upload(file) -> {name}} (the reference images' nodes),
   * `loraBrowser({mode, current, strength, onPick, onUnpick, onPickMany})` (the LoRA picker window; studio_main adds the rest),
   * optional `boxLabel(key, anchorEl)`: a prompt-card name was pressed (`key`: a box of studio_logic BOX_SWITCHES, `anchorEl`:
   * the name's button). With it the names are buttons, without it plain text.
   * Optional `boxPreset(key, text)` -> the preset whose text is the box's, null when there is none, undefined while the presets
   * are not read: with it every name has a small line under it (the preset's name / "저장 안 됨" / nothing for an empty box).
   * Call `refreshBoxMarks()` when the presets change.
   */
  constructor(widget, ctx) {
    this.widget = widget;
    this.ctx = ctx;
    this.running = false;
    this.percent = 0;
    this.forgeOpen = false;
    this.extMemory = extMemory();
    this.choices = {};
    // One popup for both prompt boxes; the LoRA list (ComfyUI file names) feeds the `<lora:` suggestions.
    this.autocomplete = ctx.tags
      ? createAutocomplete({ fetchTags: ctx.tags, loras: () => this.choices.loras || [], wildcards: () => this.choices.wildcards || [] })
      : null;
    this.state = this.read();
    // Delete / Backspace on a focused button (a box name after its preset list closed) must not delete this node.
    this.element = guardCanvasKeys(h("div", { class: "ais-root ais-panel" }));
    this.render();
    this.loadChoices();
    // The first render has no size yet (the DOM widget is attached later).
    if (globalThis.ResizeObserver) new ResizeObserver(() => this.growPrompts()).observe(this.element);
  }

  /** The node was deleted: the autocomplete popup (on document.body) must not outlive it. */
  dispose() {
    this.autocomplete?.dispose();
  }

  /** The checkpoint's family: `"anima" | "moss" | "krea2"`. */
  get family() {
    return modelFamily(this.state.unet);
  }

  /** Why NegPiP cannot run here ("" = it can): the family has no NegPiP, the provider pack is missing, or this graph predates the node. */
  negpipBlock() {
    if (!familyHasNegpip(this.family)) return NEGPIP_NO_FAMILY;
    if (!nodeInstalled("CLIPNegPip")) return NEGPIP_MISSING;
    return this.ctx.hasRole && !this.ctx.hasRole("negpip") ? NEGPIP_OLD_GRAPH : "";
  }

  /** Why the attention cannot be picked here ("" = it can): this graph predates the node. */
  attentionBlock() {
    return this.ctx.hasRole && !this.ctx.hasRole("attention") ? ATTENTION_OLD_GRAPH : "";
  }

  /** Why the LoRA card's Base 먼저 / 교차 어텐션 cannot be set here ("" = they can). */
  loraCtlBlock() {
    return this.ctx.hasRole && !this.ctx.hasRole("lora_ctl") ? LORA_CTL_OLD_GRAPH : "";
  }

  /** Why the inpaint settings cannot run here ("" = they can): this graph predates the nodes. */
  inpaintBlock() {
    return this.ctx.hasRole && !this.ctx.hasRole("inpaint_detailer") ? INPAINT_OLD_GRAPH : "";
  }

  /** Why MOSS's fp32 stream cannot run here ("" = it can). */
  streamBlock() {
    if (!nodeInstalled("MOSSStreamFP32")) return STREAM_MISSING;
    return this.ctx.hasRole && !this.ctx.hasRole("moss_stream") ? STREAM_OLD_GRAPH : "";
  }

  /** Why MOSS's layer skip cannot run here ("" = it can): this graph predates the MOSS node. It needs no other pack. */
  layerSkipBlock() {
    return this.ctx.hasRole && !this.ctx.hasRole("moss_stream") ? SKIP_OLD_GRAPH : "";
  }

  growPrompts() {
    this.element.querySelectorAll("textarea.ais-prompt, textarea.ais-negative").forEach(autoGrow);
  }

  loadChoices() {
    for (const kind of CHOICE_KINDS) {
      this.ctx.choices(kind).then((data) => {
        this.choices[kind] = data?.items || [];
        if (kind === "attentions") this.launchAttention = data?.launch || "";
        this.pickDefault(kind, data);
        this.scheduleRender();
      });
    }
  }

  /** The LoRA list without its cache: an adapter file may have arrived since. */
  refreshLoras() {
    return Promise.resolve(this.ctx.choices("loras", true)).then((data) => {
      const items = data?.items || [];
      if (!items.length && this.choices.loras?.length) return; // a failed request answers []: keep the list we have
      if (JSON.stringify(items) === JSON.stringify(this.choices.loras)) return;
      this.choices.loras = items;
      this.scheduleRender();
    });
  }

  pickDefault(kind, data) {
    const key = MODEL_KEYS[kind];
    if (!key || this.state[key]) return;
    const name = defaultModel(kind, data?.items, data?.families, this.family);
    if (name) this.set((s) => { s[key] = name; });
  }

  read() {
    try {
      return withDefaults(JSON.parse(this.widget?.value || "{}"));
    } catch {
      return withDefaults({});
    }
  }

  /** Re-reads the saved state (after the workflow was loaded into the node). */
  load() {
    this.state = this.read();
    this.render();
    this.loadChoices();
  }

  set(mutate, rerender = false) {
    mutate(this.state);
    if (this.widget) this.widget.value = JSON.stringify(this.state);
    this.ctx.changed?.();
    if (rerender) this.render();
    else this.sizeText?.replaceChildren(hiresSizeText(this.state)); // no redraw: it would take the focus from the size boxes
  }

  /** The complete panel state; the i2i image is the LoadImage node's value (the mask editor writes there). */
  getState() {
    const state = structuredClone(this.state);
    const image = this.ctx.i2i.get();
    if (image) state.i2i.image = image;
    if (this.negpipBlock()) Object.assign(state, { negpip: false, moss_negpip: false });
    if (this.streamBlock()) state.moss_stream = false;
    if (this.layerSkipBlock()) state.moss_layer_skip = false;
    if (this.loraCtlBlock()) Object.assign(state, { lora_base: 0, lora_cross_attn: CROSS_ATTN_MODES[0] });
    if (this.attentionBlock()) {
      state.attention = "launch";
      state.forge_like.attention = false;
    }
    if (this.inpaintBlock()) state.i2i.inpaint_area = "whole";
    if (referenceBlock(this.family, state.reference.method, this.ctx.hasRole)) state.reference.on = false;
    dropBlockedExt(state.ext, this.ctx.hasRole, this.family);
    return state;
  }

  setSeed(seed) {
    this.set((s) => {
      s.seed = seed;
    }, true);
  }

  /** Forge's ♻: the seed of the image picked in the gallery. `ctx.pickedSeed` says so itself when there is none. */
  async pinSeed() {
    const seed = await this.ctx.pickedSeed?.();
    if (Number.isInteger(seed)) this.setSeed(seed);
  }

  /** The EXIF panel's 설정 가져오기 (`read`: studio_settings.pulledState); answers the names this ComfyUI has no file for. */
  pullSettings(read) {
    const { state, missing } = pulledState(this.state, read, this.choices);
    this.set((s) => { Object.assign(s, state); }, true);
    this.loadChoices(); // an empty model pick takes its family's default
    return missing;
  }

  /** Forge's send-to: the image goes to the i2i tab and its size (`{width, height}`) becomes that tab's size. */
  showI2i(image, mode, size = null) {
    this.set((s) => {
      s.tab = "i2i";
      s.i2i_mode = mode === "inpaint" ? "inpaint" : "img2img";
      s.i2i.image = image;
      Object.assign(s.i2i, imageSize(size?.width, size?.height, sizeStep(this.family)) || {});
    }, true);
  }

  onStudioEvent(type, payload) {
    if (type === "running") {
      this.running = payload.running;
      this.percent = 0;
      this.renderTop();
    } else if (type === "progress") {
      const step = payload.max > 0 ? payload.value / payload.max : 0;
      this.percent = payload.cell ? ((payload.cell.index - 1 + step) * 100) / payload.cell.total : step * 100;
      if (this.bar) this.bar.style.width = `${Math.min(100, this.percent)}%`;
    }
  }

  scheduleRender() {
    if (this.renderQueued) return;
    this.renderQueued = true;
    queueMicrotask(() => {
      this.renderQueued = false;
      this.render();
    });
  }

  render() {
    const s = this.state;
    this.autocomplete?.close(); // the prompt boxes below are rebuilt
    this.sizeText = null; // drawn again with the t2i tab
    const scrolled = this.element.querySelector(".ais-scroll")?.scrollTop ?? 0;
    const active = globalThis.document?.activeElement; // a LoRA button the picker window gave the focus back to keeps it, and a 스텝 row switch its own
    const focusKey = active && this.element.contains?.(active) ? active.getAttribute?.("data-focus-key") : null;
    this.top = h("div", { class: "ais-top" });
    this.thin = h("div", { class: "ais-thin" }, (this.bar = h("div")));
    const scroll = h("div", { class: "ais-scroll" }, this.common(s), this.tabs(s), this.extensions(s), this.engineBar());
    this.element.replaceChildren(this.top, this.thin, scroll);
    if (focusKey) this.element.querySelector(`[data-focus-key="${focusKey}"]`)?.focus({ preventScroll: true });
    this.renderTop();
    this.growPrompts(); // first: the prompts' height decides how far the column can scroll
    scroll.scrollTop = scrolled;
    if (globalThis.requestAnimationFrame) requestAnimationFrame(() => this.growPrompts());
  }

  renderTop() {
    if (!this.top) return;
    const label = this.running ? "정지" : "생성 (Ctrl+Enter)";
    const button = h(
      "button",
      { type: "button", class: `ais-run ${this.running ? "stop" : ""}`, "aria-label": label, onClick: () => (this.running ? this.ctx.stop() : this.ctx.generate()) },
      label,
    );
    this.top.replaceChildren(h("strong", {}, `${FAMILY_LABELS[this.family]} 생성`), button);
    this.thin.hidden = !this.running;
    this.bar.style.width = `${Math.min(100, this.percent)}%`;
  }

  // ------------------------------------------------------------------------------------------- 공통 설정

  common(s) {
    const typed = (key) => (e) => { autoGrow(e.target); this.set((st) => { st[key] = e.target.value; }); this.markBox(key); };
    const blocked = this.negpipBlock();
    const key = negpipKey(this.family);
    const negpip = pill(s[key] && !blocked, () => this.set((st) => { st[key] = !st[key]; }, true), {
      on: "NegPiP",
      off: "NegPiP",
      title: blocked || NEGPIP_TITLE[this.family],
      disabled: Boolean(blocked),
    });
    const keys = Object.keys(BOX_SWITCHES); // the boxes in the order they join, then the negative
    const boxes = keys.map((key) =>
      h("textarea", { class: `ais-f ${key === "prompt" ? "ais-prompt" : "ais-negative"}`, "aria-label": BOX_LABELS[key], value: s[key], onInput: typed(key) }));
    boxes.forEach((box) => this.autocomplete?.attach(box)); // after the panel's own input handler: a picked tag is saved too
    const rows = boxes.map((box, index) => this.promptRow(s, keys[index], box));
    const prompt = card("프롬프트", negpip, rows, "ais-prompts");
    return group("공통 설정", "#4b6a8f", "", h("div", { class: "ais-stack" }, prompt, this.modelCard(s), this.loraCard(s), postCard(s, this.postView())));
  }

  /**
   * One row of the prompt card: the box's on/off switch, its name, the box. Off, the box keeps its text (and can still be
   * typed in) and the run leaves it out; the row is dimmed and says so.
   * The name is an element of its own, `.ais-box-label[data-section="<key>"]`, never the switch: with `ctx.boxLabel` it is a
   * button that calls `ctx.boxLabel(key, thatButton)` (the preset list opens under it).
   * It is ONE element a box for the panel's life (`this.boxNames`), like "+ LoRA": the list keeps the name it opened under
   * as its anchor (its place, a press on it, aria-expanded, the focus back when it closes) while the panel redraws under it.
   * So nothing that changes with the state may be set on the name itself: its look when off comes from the row's classes.
   * With `ctx.boxPreset` a third element follows, `.ais-box-mark[data-for="<key>"]`: what the box holds (markBox).
   */
  promptRow(s, key, box) {
    const flag = BOX_SWITCHES[key];
    const on = s[flag] !== false;
    const negative = key === "negative";
    const toggle = h("button", {
      type: "button",
      class: `ais-box-switch ${on ? "on" : "off"}`,
      role: "switch",
      "aria-checked": on ? "true" : "false",
      "aria-label": `${BOX_LABELS[key]} ${negative ? "쓰기" : "프롬프트에 넣기"}`,
      title: negative ? NEGATIVE_SWITCH_TITLE : BOX_SWITCH_TITLE,
      "data-switch": flag,
      "data-focus-key": `box-switch-${key}`,
      onClick: () => this.set((st) => { st[flag] = st[flag] === false; }, true),
    });
    const open = this.ctx.boxLabel;
    const name = ((this.boxNames ??= {})[key] ??= h(open ? "button" : "span", {
      type: open ? "button" : undefined,
      class: "ais-box-label",
      "data-section": key,
      "data-focus-key": open ? `box-label-${key}` : undefined,
      onClick: open ? () => open(key, this.boxNames[key]) : undefined,
    }, BOX_LABELS[key]));
    const mark = this.ctx.boxPreset ? ((this.boxMarks ??= {})[key] ??= h("span", { class: "ais-box-mark", "data-for": key, hidden: true })) : null;
    this.markBox(key);
    return h(
      "div",
      { class: `ais-row ${negative ? "ais-neg" : "ais-pos"}${on ? "" : " ais-off"}`, "data-box": key, title: on ? undefined : negative ? NEGATIVE_OFF_TITLE : BOX_OFF_TITLE },
      h("span", { class: "ais-label" }, toggle, name, mark),
      h("span", { class: "ais-ctl" }, box),
    );
  }

  /**
   * The line under a box's name: the preset whose text the box holds, "저장 안 됨" for a text no preset has, nothing for an
   * empty box or while the presets are not read (preset_logic.js labelMark). One element a box, filled in place: typing
   * changes it without a redraw.
   */
  markBox(key) {
    const mark = this.boxMarks?.[key];
    if (!mark) return;
    const { kind, text, title } = labelMark(this.ctx.boxPreset(key, this.state[key]), this.state[key]);
    mark.textContent = text;
    mark.title = title;
    mark.hidden = kind === "empty";
    mark.setAttribute("data-mark", kind);
  }

  /** The presets changed (read, saved, renamed, deleted): every box's mark again, without drawing the panel. */
  refreshBoxMarks() {
    for (const key of Object.keys(this.boxMarks || {})) this.markBox(key);
  }

  modelCard(s) {
    const pick = (kind, key, empty) =>
      select(optionItems(this.choices[kind], s[key], { empty }), s[key], (value) => this.set((st) => { st[key] = value; }), { title: empty });
    const num = (key, opts) => numberBox(s[key], (value) => this.set((st) => { st[key] = value; }), opts);
    const head = h(
      "span",
      { class: "ais-head-ctl" },
      pill(s.forge_like.master, () => this.set((st) => { st.forge_like.master = !st.forge_like.master; }, true), { on: "Forge처럼", off: "Forge처럼", title: "Forge처럼 맞추기 (전체)" }),
      h("button", { type: "button", class: "ais-caret", "aria-label": "Forge처럼 항목", title: "Forge처럼 항목", onClick: () => { this.forgeOpen = !this.forgeOpen; this.render(); } }, this.forgeOpen ? "▴" : "▾"),
    );
    const unet = select(optionItems(this.choices.unets, s.unet, { empty: "[체크포인트]" }), s.unet,
      (value) => this.set((st) => pickCheckpoint(st, value, { tes: this.choices.tes, vaes: this.choices.vaes }), true), { title: "[체크포인트]" });
    const stream = this.streamBlock();
    return card("모델 · 샘플링", head, [
      unet,
      pick("tes", "te", "[TE]"),
      pick("vaes", "vae", "[VAE]"),
      this.family === "moss"
        ? row("fp32 stream", pill(s.moss_stream && !stream, () => this.set((st) => { st.moss_stream = !st.moss_stream; }, true), {
          title: stream || "손가락·글자 오차 감소. VRAM 약 0.6GB 더, 속도 약 1.5~10% 느림",
          disabled: Boolean(stream),
        }))
        : null,
      this.forgeOpen ? this.forgeItems(s) : null,
      row("샘플러", pick("samplers", "sampler")),
      row("스케줄러", pick("schedulers", "scheduler")),
      row("어텐션", this.attentionPick(s)),
      row("스텝", this.stepSwitches(s), num("steps", { min: 1, max: 150, step: 1, width: 46 }), "CFG", num("cfg", { min: 0, max: 30, step: 0.5, width: 44 }),
        "Shift", num("shift", { min: 0, max: 100, step: 0.5, width: 44 })),
      row("시드", num("seed", { min: -1, step: 1, width: 120, title: "-1 = 랜덤, 그 밖의 값은 그대로 유지", live: true }), // Ctrl+Enter comes before the box is left
        h("button", { type: "button", class: "ais-mini", title: "현재 이미지의 시드 가져오기", "aria-label": "현재 이미지 시드 가져오기", onClick: () => this.pinSeed() }, "♻"),
        h("button", { type: "button", class: "ais-mini", title: "랜덤 (-1)", "aria-label": "랜덤 시드", onClick: () => this.setSeed(-1) }, "랜덤")),
    ]);
  }

  /** The 스텝 row's switches, right before the steps box: MOSS's layer skip (MOSS only), then the two fixed prompt boxes. */
  stepSwitches(s) {
    const flip = (key) => () => this.set((st) => { st[key] = st[key] === false; }, true);
    const blocked = this.layerSkipBlock();
    const skip = this.family === "moss"
      ? pill(s.moss_layer_skip !== false && !blocked, flip("moss_layer_skip"), { on: SKIP_LABEL, off: SKIP_LABEL, title: blocked || SKIP_TITLE, disabled: Boolean(blocked), focusKey: "switch-moss_layer_skip" })
      : null;
    const boxes = STEP_BOX_SWITCHES.map(([box, label, title]) =>
      pill(s[PROMPT_SWITCHES[box]] !== false, flip(PROMPT_SWITCHES[box]), { on: label, off: label, title, focusKey: `switch-${PROMPT_SWITCHES[box]}` }));
    return h("span", { class: "ais-step-switches" }, skip, boxes);
  }

  attentionPick(s) {
    const blocked = this.attentionBlock();
    const forge = s.forge_like.master && s.forge_like.attention;
    const label = (mode) => {
      const started = ATTENTION_LABELS[this.launchAttention];
      return mode === "launch" && started ? `${ATTENTION_LABELS.launch} (${started})` : ATTENTION_LABELS[mode];
    };
    const items = (this.choices.attentions || ["launch"]).map((mode) => ({ value: mode, label: label(mode) }));
    return select(optionItems(items, s.attention), forge ? "pytorch" : s.attention, (value) => this.set((st) => { st.attention = value; }),
      { disabled: Boolean(blocked || forge), title: blocked || (forge ? ATTENTION_FORGE : "ComfyUI 를 다시 켜지 않고 바꾼다"), "aria-label": "어텐션" });
  }

  forgeItems(s) {
    const items = FORGE_ITEMS.map((item) =>
      pill(Boolean(s.forge_like[item.key]) && !item.disabled, () => this.set((st) => { st.forge_like[item.key] = !st.forge_like[item.key]; }, true), {
        on: item.label,
        off: item.label,
        title: item.disabled || item.title,
        disabled: Boolean(item.disabled),
      }));
    const doraForge = s.forge_like.master && s.forge_like.dora;
    const dora = doraForge
      ? null
      : row("DoRA 방식", select(optionItems(DORA_MODES, s.dora_mode), s.dora_mode, (value) => this.set((st) => { st.dora_mode = value; })));
    return h("div", { class: `ais-forge ${s.forge_like.master ? "" : "dim"}` }, h("div", { class: "ais-wrap" }, items), dora);
  }

  loraCard(s) {
    const rows = s.loras.map((lora, index) => {
      const face = rowFace(lora.name, this.choices.loras);
      return h("div", { class: "ais-row ais-lora" },
        pill(lora.on, () => this.set((st) => { st.loras[index].on = !st.loras[index].on; }, true)),
        h("button", { type: "button", class: `ais-f ais-grow ais-lb-pick${face.missing ? " missing" : ""}`, title: face.title, "data-focus-key": `lora-${index}`, onClick: () => this.replaceLora(index) },
          h("span", { class: "ais-lb-pick-text" }, face.text)),
        numberBox(lora.strength, (value) => this.set((st) => { st.loras[index].strength = value; }), { min: -10, max: 10, step: 0.05, width: 50 }),
        h("button", { type: "button", class: "ais-mini", title: "빼기", "aria-label": "LoRA 빼기", onClick: () => this.set((st) => { st.loras.splice(index, 1); }, true) }, "✕"));
    });
    // One element across redraws: the window gives the focus back to it when it closes, after the picks redrew the panel.
    const add = (this.loraAdd ??= h("button", { type: "button", class: "ais-add", "data-focus-key": "lora-add", onClick: () => this.addLoras() }, "+ LoRA"));
    const blocked = this.loraCtlBlock();
    const baseOff = blocked || (inpaintByDetailer(s) ? LORA_BASE_INPAINT : "");
    const crossOff = blocked || (this.family !== "anima" ? LORA_CROSS_FAMILY : "");
    const base = numberBox(blocked ? 0 : s.lora_base, (value) => this.set((st) => { st.lora_base = Math.max(0, Math.min(LORA_BASE_MAX, Math.trunc(value))); }),
      { min: 0, max: LORA_BASE_MAX, step: 5, width: 46, title: baseOff || LORA_BASE_TITLE, disabled: Boolean(baseOff) });
    const cross = select(optionItems(CROSS_ATTN_MODES, s.lora_cross_attn), crossOff ? CROSS_ATTN_MODES[0] : s.lora_cross_attn,
      (value) => this.set((st) => { st.lora_cross_attn = value; }),
      { disabled: Boolean(crossOff), title: crossOff || LORA_CROSS_TITLE, "aria-label": "LoRA 교차 어텐션 제외" });
    const controls = [row("Base 먼저", base, h("span", { title: baseOff || LORA_BASE_TITLE }, "%")), row("교차 어텐션 제외", cross)];
    return card("LoRA", null, [rows, add, ...controls]);
  }

  /** "+ LoRA": the picker window (lora_rows.addSession) puts each pick in as a row at once; its tray ✕ takes the row out again. */
  addLoras() {
    const session = addSession();
    this.ctx.loraBrowser?.({
      mode: "add",
      onPick: (pick) => this.editLoras((loras) => session.onPick(loras, pick), pick.name),
      onUnpick: (row) => this.editLoras((loras) => session.onUnpick(loras, row)),
    });
  }

  /** A row's name: the window opens on that LoRA and strength ("바꾸기"), one pick, then it closes. */
  replaceLora(index) {
    const lora = this.state.loras[index];
    if (!lora) return;
    this.ctx.loraBrowser?.({ mode: "replace", current: lora.name, strength: lora.strength,
      onPick: (pick) => this.editLoras((loras) => replaceRow(loras, index, pick), pick.name) });
  }

  /** A row edit from the window: saved now, drawn after the window's step (its close() gives the focus back first). */
  editLoras(edit, picked = "") {
    let result;
    this.set((st) => { result = edit(st.loras); });
    this.scheduleRender();
    if (picked && !listedStem(this.choices.loras, picked)) this.refreshLoras(); // a file newer than the panel's list
    return result;
  }

  /** What post_view.js draws with. */
  postView() {
    const lists = {
      upscalers: () => (this.choices.upscalers || []).filter((name) => !name.startsWith("latent:")),
      upscaler_methods: () => this.choices.upscalers,
    };
    return {
      set: (mutate, rerender) => this.set(mutate, rerender),
      memory: this.extMemory,
      rerender: () => this.scheduleRender(),
      lists: (kind) => (lists[kind] ? lists[kind]() : this.choices[kind]),
      attachText: (box) => this.autocomplete?.attach(box),
      keepSizeText: (element) => { this.sizeText = element; },
    };
  }

  // ------------------------------------------------------------------------------------------- t2i | i2i

  tabs(s) {
    const tab = (name) =>
      h("button", { type: "button", class: `ais-tab ${s.tab === name ? "active" : ""}`, "aria-pressed": s.tab === name ? "true" : "false",
        onClick: () => this.set((st) => { st.tab = name; }, true) }, name);
    const body = s.tab === "t2i"
      ? group(null, "#3f7a52", "ais-tabgrp", h("div", { class: "ais-grid2" }, this.sizeCard(s), hiresCard(s, this.postView())))
      : group(null, "#8a5a7a", "ais-tabgrp", h("div", { class: "ais-grid-i2i" }, this.imageBox(), this.i2iCard(s)));
    return h("div", { class: "ais-tabs-wrap" }, h("div", { class: "ais-tabs" }, tab("t2i"), tab("i2i")), body);
  }

  /** `section`: where the size lives, the state itself (t2i) or `i2i`. */
  sizeBox(s, key, section = null) {
    const owner = (st) => (section ? st[section] : st);
    return numberBox(owner(s)[key], (value) => this.set((st) => { owner(st)[key] = value; }), { min: 64, max: 4096, step: sizeStep(this.family), width: 62 });
  }

  sizeCard(s) {
    const size = (key) => this.sizeBox(s, key);
    const swap = h("button", { type: "button", class: "ais-mini", title: "가로 세로 바꾸기", "aria-label": "가로 세로 바꾸기",
      onClick: () => this.set((st) => { [st.width, st.height] = [st.height, st.width]; }, true) }, "⇅");
    return card("크기", null, [
      row("크기", size("width"), "×", size("height"), swap),
      row("장수", numberBox(s.batch, (value) => this.set((st) => { st.batch = value; }), { min: 1, max: 16, step: 1, width: 62 })),
    ]);
  }

  imageBox() {
    const value = this.ctx.i2i.get() || this.state.i2i.image;
    const file = h("input", { type: "file", accept: "image/*", hidden: true, onChange: async (e) => {
      const chosen = e.target.files?.[0];
      if (!chosen) return;
      const uploaded = await this.ctx.i2i.upload(chosen); // {name, width, height}
      if (!uploaded?.name) return;
      this.set((st) => {
        st.i2i.image = uploaded.name;
        Object.assign(st.i2i, imageSize(uploaded.width, uploaded.height, sizeStep(this.family)) || {});
      }, true);
    } });
    const picture = value
      ? h("img", { class: "ais-i2i-img", src: this.ctx.i2i.url(value), alt: "입력 이미지" })
      : h("span", {}, "입력 이미지");
    const box = h("div", { class: "ais-i2i-box", title: "눌러서 이미지 고르기", onClick: (e) => { if (e.target === box || e.target === picture) file.click(); },
      onPointerenter: () => { if ((this.ctx.i2i.get() || "") !== (value || "")) this.render(); } },
    picture, file,
    h("button", { type: "button", class: "ais-pill off", disabled: !value, onClick: () => this.ctx.i2i.maskEditor() }, "마스크 그리기"));
    return box;
  }

  i2iCard(s) {
    const mode = (name, text) =>
      pill(s.i2i_mode === name, () => this.set((st) => { st.i2i_mode = name; }, true), { on: text, off: text });
    const blocked = this.inpaintBlock();
    const areas = [{ value: "whole", label: "전체" }, { value: "masked", label: "마스크만" }];
    const area = select(areas, blocked ? "whole" : s.i2i.inpaint_area, (value) => this.set((st) => { st.i2i.inpaint_area = value; }, true),
      { disabled: Boolean(blocked), title: blocked || INPAINT_AREA_TITLE, "aria-label": "인페인트 영역" });
    const inpainting = !blocked && s.i2i_mode === "inpaint";
    const fills = INPAINT_FILLS.map((value) => ({ value, label: FILL_LABELS[value] }));
    return card("i2i 설정", null, [
      row("모드", mode("img2img", "i2i"), mode("inpaint", "인페인트")),
      row("크기", this.sizeBox(s, "width", "i2i"), "×", this.sizeBox(s, "height", "i2i")), // the image's size when sent / uploaded
      row("디노이즈", numberBox(s.i2i.denoise, (value) => this.set((st) => { st.i2i.denoise = value; }), { min: 0, max: 1, step: 0.05, width: 62 })),
      row("인페인트 영역", area),
      inpainting && s.i2i.inpaint_area === "masked" ? row("패딩", numberBox(s.i2i.inpaint_padding, (value) => this.set((st) => { st.i2i.inpaint_padding = value; }), { min: 0, max: 2048, step: 4, width: 62, title: "마스크 둘레에 함께 자를 픽셀" })) : null,
      inpainting ? row("마스크 내용", select(fills, s.i2i.inpaint_fill, (value) => this.set((st) => { st.i2i.inpaint_fill = value; }), { "aria-label": "마스크 내용" })) : null,
      row("마스크 흐림", numberBox(s.i2i.mask_blur, (value) => this.set((st) => { st.i2i.mask_blur = value; }), { min: 0, max: 64, step: 1, width: 62 })),
    ]);
  }

  // ------------------------------------------------------------------------------------------- 확장기능

  extensions(s) {
    return renderExtensions(s, {
      set: (mutate, rerender) => this.set(mutate, rerender),
      hasRole: this.ctx.hasRole,
      fetchChoices: this.ctx.fetchChoices,
      memory: this.extMemory,
      rerender: () => this.scheduleRender(),
      family: this.family,
      leading: renderReference(s, {
        set: (mutate, rerender) => this.set(mutate, rerender),
        hasRole: this.ctx.hasRole,
        family: this.family,
        memory: this.extMemory,
        rerender: () => this.scheduleRender(),
        loras: this.choices.loras || [],
        refreshLoras: () => this.refreshLoras(),
        image: this.ctx.reference,
        image2: this.ctx.reference2,
        image3: this.ctx.reference3,
      }),
    });
  }

  // ------------------------------------------------------------------------------------------- 엔진룸

  engineBar() {
    const collapsed = this.ctx.engineRoom.collapsed();
    const text = `엔진룸 (${collapsed ? "접힘" : "펼침"} · 위에서 아래로 ①~⑩)`;
    const label = collapsed ? "펼치기" : "접기";
    return h("div", { class: "ais-engine" }, h("span", {}, text),
      h("button", { type: "button", class: "ais-pill off", "aria-label": `엔진룸 ${label}`, onClick: () => { this.ctx.engineRoom.toggle(); this.render(); } }, label));
  }
}
