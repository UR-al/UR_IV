// Anima studio: the control panel (C4 left column, 790px) drawn into AIStudioPanel's DOM widget.
// Its state lives as JSON in the node's hidden `state` STRING widget, so it is saved with the workflow.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { dropBlockedExt, extMemory, renderExtensions } from "./ext_view.js";
import { createAutocomplete } from "./studio_autocomplete.js";
import { DORA_MODES, defaultModel, imageSize, withDefaults } from "./studio_logic.js";
import { autoGrow, h, numberBox, optionItems, pill, select } from "./studio_dom.js";

const CHOICE_KINDS = ["unets", "tes", "vaes", "samplers", "schedulers", "loras", "upscalers", "sam3"];
// Model picks filled with an Anima default when their list arrives empty-handed (an empty pick would run the
// loader's first file; the run controller refuses a run while one is still empty).
const MODEL_KEYS = { unets: "unet", tes: "te", vaes: "vae" };
const NEGPIP_MISSING = "불가: ComfyUI-ppm 노드 팩(CLIPNegPip)이 설치돼 있지 않음";
const NEGPIP_OLD_GRAPH = "불가: 이 워크플로에는 NegPiP 노드가 없음. AI Studio Anima.json 을 새로 여세요";
const INPAINT_AREA_REASON =
  "불가: 지금 인페인트는 늘 그림 전체를 이 크기로 다시 그린다 (Forge의 '전체'). '마스크만'(잘라서 크게 그리기)은 아직 없음";

// "Forge처럼" items; greyed ones follow studio_forge_parity.md (불가 rows).
const FORGE_ITEMS = [
  { key: "noise", label: "노이즈", title: "초기·SDE 노이즈를 Forge처럼 (그림마다 시드 + i)" },
  { key: "prompt_syntax", label: "프롬프트 편집", title: "[a:b:0.5] · [a|b] 를 Forge처럼 스텝 구간으로" },
  { key: "dora", label: "DoRA", title: "DoRA 합치기를 학습과 같은 LyCORIS로 (실제 Forge 설정)" },
  {
    key: "attention",
    label: "어텐션",
    disabled: "불가: 어텐션은 ComfyUI 실행 옵션이라 여기서 못 바꿈. Forge와 비교할 땐 run_nvidia_gpu.bat(PyTorch 어텐션)으로 다시 켜기",
  },
];

const fileStem = (entry) => entry.replace(/\\/g, "/").split("/").pop().replace(/\.[^.]+$/, "");

const group = (title, color, extraClass, ...children) =>
  h("div", { class: `ais-grp ${extraClass}`, style: `--c:${color}` }, title ? h("div", { class: "ais-gh" }, title) : null, children);

const card = (title, head, body, extraClass = "") =>
  h("div", { class: `ais-node ${extraClass}` }, h("div", { class: "ais-nt" }, h("span", {}, title), head), h("div", { class: "ais-nb" }, body));

const row = (label, ...controls) => h("div", { class: "ais-row" }, h("span", { class: "ais-label" }, label), h("span", { class: "ais-ctl" }, controls));

// Absent LiteGraph (tests) counts as installed; a missing provider greys the switch and runs without it.
const negpipInstalled = () => {
  const types = globalThis.LiteGraph?.registered_node_types;
  return !types || Boolean(types.CLIPNegPip);
};

export class PanelView {
  /**
   * `widget`: the node's hidden `state` widget. `ctx`: `choices(kind)` -> Promise<{items}>, `fetchChoices(class)` ->
   * Promise<`/object_info/<class>` input>, `hasRole(role)`, `generate()`, `stop()`, `changed()`, optional `tags(query, limit)` ->
   * Promise<[{name, category, count, alias?, ko?}]> (prompt tag autocomplete), `engineRoom` {collapsed(),
   * toggle()}, `i2i` {get(), set(value), url(value), upload(file) -> {name, width, height}, maskEditor()}.
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
    this.autocomplete = ctx.tags ? createAutocomplete({ fetchTags: ctx.tags, loras: () => this.choices.loras || [] }) : null;
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

  /** Why NegPiP cannot run here ("" = it can): the provider pack is missing, or this graph predates the NegPiP node. */
  negpipBlock() {
    if (!negpipInstalled()) return NEGPIP_MISSING;
    return this.ctx.hasRole && !this.ctx.hasRole("negpip") ? NEGPIP_OLD_GRAPH : "";
  }

  growPrompts() {
    this.element.querySelectorAll("textarea.ais-prompt, textarea.ais-negative").forEach(autoGrow);
  }

  loadChoices() {
    for (const kind of CHOICE_KINDS) {
      this.ctx.choices(kind).then((data) => {
        this.choices[kind] = data?.items || [];
        this.pickDefault(kind, data);
        this.scheduleRender();
      });
    }
  }

  pickDefault(kind, data) {
    const key = MODEL_KEYS[kind];
    if (!key || this.state[key]) return;
    const name = defaultModel(kind, data?.items, data?.families);
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
  }

  /** The complete panel state; the i2i image is the LoadImage node's value (the mask editor writes there). */
  getState() {
    const state = structuredClone(this.state);
    const image = this.ctx.i2i.get();
    if (image) state.i2i.image = image;
    if (this.negpipBlock()) state.negpip = false;
    dropBlockedExt(state.ext, this.ctx.hasRole);
    return state;
  }

  setSeed(seed) {
    this.set((s) => {
      s.seed = seed;
    }, true);
  }

  /** Forge's send-to: the image goes to the i2i tab and its size (`{width, height}`) becomes that tab's size. */
  showI2i(image, mode, size = null) {
    this.set((s) => {
      s.tab = "i2i";
      s.i2i_mode = mode === "inpaint" ? "inpaint" : "img2img";
      s.i2i.image = image;
      Object.assign(s.i2i, imageSize(size?.width, size?.height) || {});
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
    this.top.replaceChildren(h("strong", {}, "Anima 생성"), button);
    this.thin.hidden = !this.running;
    this.bar.style.width = `${Math.min(100, this.percent)}%`;
  }

  // ------------------------------------------------------------------------------------------- 공통 설정

  common(s) {
    const typed = (key) => (e) => { autoGrow(e.target); this.set((st) => { st[key] = e.target.value; }); };
    const blocked = this.negpipBlock();
    const negpip = pill(s.negpip && !blocked, () => this.set((st) => { st.negpip = !st.negpip; }, true), {
      on: "NegPiP",
      off: "NegPiP",
      title: blocked || "음수 가중치 (단어:-1.0)",
      disabled: Boolean(blocked),
    });
    const boxes = [
      h("textarea", { class: "ais-f ais-prompt", placeholder: "프롬프트", value: s.prompt, onInput: typed("prompt") }),
      h("textarea", { class: "ais-f ais-negative", placeholder: "네거티브", value: s.negative, onInput: typed("negative") }),
    ];
    boxes.forEach((box) => this.autocomplete?.attach(box)); // after the panel's own input handler: a picked tag is saved too
    const prompt = card("프롬프트", negpip, boxes, "span3");
    return group("공통 설정", "#4b6a8f", "", h("div", { class: "ais-grid3" }, prompt, this.modelCard(s), this.loraCard(s), this.postCard(s)));
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
    return card("모델 · 샘플링", head, [
      pick("unets", "unet", "[체크포인트]"),
      pick("tes", "te", "[TE]"),
      pick("vaes", "vae", "[VAE]"),
      this.forgeOpen ? this.forgeItems(s) : null,
      row("샘플러", pick("samplers", "sampler")),
      row("스케줄러", pick("schedulers", "scheduler")),
      row("스텝", num("steps", { min: 1, max: 150, step: 1, width: 46 }), "CFG", num("cfg", { min: 0, max: 30, step: 0.5, width: 44 }),
        "Shift", num("shift", { min: 0, max: 100, step: 0.5, width: 44 })),
      row("시드", num("seed", { min: -1, step: 1, width: 120, title: "-1 = 랜덤" }),
        h("button", { type: "button", class: "ais-mini", title: "랜덤 (-1)", "aria-label": "랜덤 시드", onClick: () => this.setSeed(-1) }, "-1")),
    ]);
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

  postCard(s) {
    const toggle = (section) => pill(s[section].on, () => this.set((st) => { st[section].on = !st[section].on; }, true));
    const models = optionItems((this.choices.upscalers || []).filter((name) => !name.startsWith("latent:")), s.upscale.model, { empty: "[업스케일 모델]" });
    const upscale = s.upscale.on
      ? [select(models, s.upscale.model, (value) => this.set((st) => { st.upscale.model = value; })),
        row("배율", numberBox(s.upscale.scale, (value) => this.set((st) => { st.upscale.scale = value; }), { min: 1, max: 8, step: 0.25, width: 50 }))]
      : null;
    const sam3 = s.sam3.on
      ? [select(optionItems(this.choices.sam3, s.sam3.checkpoint, { empty: "[SAM3 모델]" }), s.sam3.checkpoint, (value) => this.set((st) => { st.sam3.checkpoint = value; })),
        row("찾을 것", h("input", { class: "ais-f", value: s.sam3.prompt, onChange: (e) => this.set((st) => { st.sam3.prompt = e.target.value || "face"; }) })),
        row("디노이즈", numberBox(s.sam3.denoise, (value) => this.set((st) => { st.sam3.denoise = value; }), { min: 0, max: 1, step: 0.05, width: 50 }))]
      : null;
    return card("후처리 (켜면 바로 적용)", null, [
      row("Hires fix", toggle("hires")),
      row("업스케일", toggle("upscale")), upscale,
      row("SAM3 디테일러", toggle("sam3")), sam3,
    ]);
  }

  // ------------------------------------------------------------------------------------------- t2i | i2i

  tabs(s) {
    const tab = (name) =>
      h("button", { type: "button", class: `ais-tab ${s.tab === name ? "active" : ""}`, "aria-pressed": s.tab === name ? "true" : "false",
        onClick: () => this.set((st) => { st.tab = name; }, true) }, name);
    const body = s.tab === "t2i"
      ? group(null, "#3f7a52", "ais-tabgrp", h("div", { class: "ais-grid2" }, this.sizeCard(s), this.hiresCard(s)))
      : group(null, "#8a5a7a", "ais-tabgrp", h("div", { class: "ais-grid-i2i" }, this.imageBox(), this.i2iCard(s)));
    return h("div", { class: "ais-tabs-wrap" }, h("div", { class: "ais-tabs" }, tab("t2i"), tab("i2i")), body);
  }

  /** `section`: where the size lives, the state itself (t2i) or `i2i`. */
  sizeBox(s, key, section = null) {
    const owner = (st) => (section ? st[section] : st);
    return numberBox(owner(s)[key], (value) => this.set((st) => { owner(st)[key] = value; }), { min: 64, max: 4096, step: 8, width: 62 });
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

  hiresCard(s) {
    const hires = s.hires;
    const num = (key, opts) => numberBox(hires[key], (value) => this.set((st) => { st.hires[key] = value; }), opts);
    const samplers = optionItems(this.choices.samplers, hires.sampler, { empty: "메인과 같음" });
    return card("Hires fix", null, [
      row("업스케일러", select(optionItems(this.choices.upscalers, hires.upscaler), hires.upscaler, (value) => this.set((st) => { st.hires.upscaler = value; }))),
      row("배율", num("scale", { min: 1, max: 8, step: 0.05, width: 46 }), "디노이즈", num("denoise", { min: 0, max: 1, step: 0.05, width: 46 }),
        "스텝", num("steps", { min: 1, max: 150, step: 1, width: 42 })),
      row("샘플러", select(samplers, hires.sampler, (value) => this.set((st) => { st.hires.sampler = value; }))),
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
        Object.assign(st.i2i, imageSize(uploaded.width, uploaded.height) || {});
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
    // Greyed like 어텐션: no node input takes the area, every inpaint is whole-picture (studio_forge_parity.md).
    const areas = [{ value: "whole", label: "전체" }, { value: "masked", label: "마스크만" }];
    const area = h("span", { title: INPAINT_AREA_REASON }, select(areas, "whole", () => {}, { disabled: true, "aria-label": "인페인트 영역" }));
    return card("i2i 설정", null, [
      row("모드", mode("img2img", "i2i"), mode("inpaint", "인페인트")),
      row("크기", this.sizeBox(s, "width", "i2i"), "×", this.sizeBox(s, "height", "i2i")), // the image's size when sent / uploaded
      row("디노이즈", numberBox(s.i2i.denoise, (value) => this.set((st) => { st.i2i.denoise = value; }), { min: 0, max: 1, step: 0.05, width: 62 })),
      row("인페인트 영역", area),
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
