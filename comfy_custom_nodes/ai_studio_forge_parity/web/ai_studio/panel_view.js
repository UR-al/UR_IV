// Anima studio: the control panel (C4 left column, 790px) drawn into AIStudioPanel's DOM widget.
// Its state lives as JSON in the node's hidden `state` STRING widget, so it is saved with the workflow.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { dropBlockedExt, extMemory, renderExtensions } from "./ext_view.js";
import { nodeInstalled, referenceBlock, renderReference } from "./reference_view.js";
import { createAutocomplete } from "./studio_autocomplete.js";
import { hiresCard, hiresSizeText, postCard } from "./post_view.js";
import { FAMILY_LABELS } from "./studio_ext.js";
import { DORA_MODES, INPAINT_FILLS, PROMPT_SECTIONS, defaultModel, familyHasNegpip, fileStem, imageSize, modelFamily, negpipKey, pickCheckpoint, sizeStep, withDefaults } from "./studio_logic.js";
import { autoGrow, card, h, numberBox, optionItems, pill, row, select } from "./studio_dom.js";

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
const INPAINT_OLD_GRAPH = "불가: 이 워크플로에는 인페인트 노드가 없음. AI Studio Anima.json 을 새로 여세요";
const INPAINT_AREA_TITLE = "전체: 그림 전체를 위 크기로 다시 그림. 마스크만: 마스크 둘레를 잘라 위 크기로 그리고 원본에 붙임 (Forge의 Only masked)";
const ATTENTION_OLD_GRAPH = "불가: 이 워크플로에는 어텐션 노드가 없음. AI Studio Anima.json 을 새로 여세요";
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

const PROMPT_LABELS = {
  prompt_prefix: "선행 고정",
  prompt_char_count: "인물 수",
  prompt_character: "캐릭터",
  prompt_copyright: "작품",
  prompt_artist: "작가",
  prompt: "프롬프트",
  prompt_suffix: "후행 고정",
  negative: "네거티브",
};

const group = (title, color, extraClass, ...children) =>
  h("div", { class: `ais-grp ${extraClass}`, style: `--c:${color}` }, title ? h("div", { class: "ais-gh" }, title) : null, children);

export class PanelView {
  /**
   * `widget`: the node's hidden `state` widget. `ctx`: `choices(kind)` -> Promise<{items}>, `fetchChoices(class)` ->
   * Promise<`/object_info/<class>` input>, `hasRole(role)`, `generate()`, `stop()`, `changed()`, optional `tags(query, limit)` ->
   * Promise<[{name, category, count, alias?, ko?}]> (prompt tag autocomplete), `pickedSeed()` -> Promise<seed of the image
   * picked in the gallery, or null>, `engineRoom` {collapsed(),
   * toggle()}, `i2i` {get(), set(value), url(value), upload(file) -> {name, width, height}, maskEditor()},
   * `reference`, `reference2` {url(value), upload(file) -> {name}} (the reference images' nodes).
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
    this.element = h("div", { class: "ais-root ais-panel" });
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

  /** Why the inpaint settings cannot run here ("" = they can): this graph predates the nodes. */
  inpaintBlock() {
    return this.ctx.hasRole && !this.ctx.hasRole("inpaint_detailer") ? INPAINT_OLD_GRAPH : "";
  }

  /** Why MOSS's fp32 stream cannot run here ("" = it can). */
  streamBlock() {
    if (!nodeInstalled("MOSSStreamFP32")) return STREAM_MISSING;
    return this.ctx.hasRole && !this.ctx.hasRole("moss_stream") ? STREAM_OLD_GRAPH : "";
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
    this.top = h("div", { class: "ais-top" });
    this.thin = h("div", { class: "ais-thin" }, (this.bar = h("div")));
    const scroll = h("div", { class: "ais-scroll" }, this.common(s), this.tabs(s), this.extensions(s), this.engineBar());
    this.element.replaceChildren(this.top, this.thin, scroll);
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
    const typed = (key) => (e) => { autoGrow(e.target); this.set((st) => { st[key] = e.target.value; }); };
    const blocked = this.negpipBlock();
    const key = negpipKey(this.family);
    const negpip = pill(s[key] && !blocked, () => this.set((st) => { st[key] = !st[key]; }, true), {
      on: "NegPiP",
      off: "NegPiP",
      title: blocked || NEGPIP_TITLE[this.family],
      disabled: Boolean(blocked),
    });
    const keys = [...PROMPT_SECTIONS, "negative"];
    const boxes = keys.map((key) =>
      h("textarea", { class: `ais-f ${key === "prompt" ? "ais-prompt" : "ais-negative"}`, "aria-label": PROMPT_LABELS[key], value: s[key], onInput: typed(key) }));
    boxes.forEach((box) => this.autocomplete?.attach(box)); // after the panel's own input handler: a picked tag is saved too
    const prompt = card("프롬프트", negpip, boxes.map((box, index) => row(PROMPT_LABELS[keys[index]], box)), "ais-prompts");
    return group("공통 설정", "#4b6a8f", "", h("div", { class: "ais-stack" }, prompt, this.modelCard(s), this.loraCard(s), postCard(s, this.postView())));
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
      row("스텝", num("steps", { min: 1, max: 150, step: 1, width: 46 }), "CFG", num("cfg", { min: 0, max: 30, step: 0.5, width: 44 }),
        "Shift", num("shift", { min: 0, max: 100, step: 0.5, width: 44 })),
      row("시드", num("seed", { min: -1, step: 1, width: 120, title: "-1 = 랜덤, 그 밖의 값은 그대로 유지", live: true }), // Ctrl+Enter comes before the box is left
        h("button", { type: "button", class: "ais-mini", title: "현재 이미지의 시드 가져오기", "aria-label": "현재 이미지 시드 가져오기", onClick: () => this.pinSeed() }, "♻"),
        h("button", { type: "button", class: "ais-mini", title: "랜덤 (-1)", "aria-label": "랜덤 시드", onClick: () => this.setSeed(-1) }, "랜덤")),
    ]);
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
    const files = (this.choices.loras || []).map((entry) => ({ value: fileStem(entry), label: entry }));
    const rows = s.loras.map((lora, index) =>
      h("div", { class: "ais-row ais-lora" },
        pill(lora.on, () => this.set((st) => { st.loras[index].on = !st.loras[index].on; }, true)),
        select(optionItems(files, lora.name, { empty: "[LoRA]" }), lora.name, (value) => this.set((st) => { st.loras[index].name = value; }), { class: "ais-f ais-grow" }),
        numberBox(lora.strength, (value) => this.set((st) => { st.loras[index].strength = value; }), { min: -10, max: 10, step: 0.05, width: 50 }),
        h("button", { type: "button", class: "ais-mini", title: "빼기", "aria-label": "LoRA 빼기", onClick: () => this.set((st) => { st.loras.splice(index, 1); }, true) }, "✕")));
    const add = h("button", { type: "button", class: "ais-add", onClick: () => this.set((st) => { st.loras.push({ on: true, name: "", strength: 1.0 }); }, true) }, "+ LoRA");
    return card("LoRA", null, [rows, add]);
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
