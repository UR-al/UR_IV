// Anima studio: runs what the panel asks for — one generation, an X/Y/Z plot, or a gallery post-processing pass —
// and follows ComfyUI's websocket events for the prompts it queued (images, progress, preview, failures, stop).
// No DOM and no ComfyUI import: ComfyUI's `api`, the graph and the routes come in through `deps` (studio_main.js).
// Pure ES module without side effects (ComfyUI also loads every .js under web/ as an extension).
import {
  applyStateToPrompt,
  hasNegativeWeights,
  missingLoras,
  missingPicks,
  postPrompt,
  postRunState,
  recordFromParameters,
  stageModes,
  withDefaults,
} from "./studio_logic.js";
import { withExtraParams } from "./studio_pnginfo.js";

const EVENTS = Object.freeze([
  "execution_start",
  "executed",
  "execution_success",
  "execution_error",
  "execution_interrupted",
  "progress",
  "executing",
  "b_preview",
]);
const EARLY_EVENT_LIMIT = 500; // events of prompts whose id the queue call has not returned yet

export const NOTICES = Object.freeze({
  busy: "이미 생성 중",
  noImage: "고른 이미지가 없음",
  noInfo: "이 이미지엔 생성 정보가 없음",
  noCells: "격자로 만들 이미지가 없음",
  failed: "실행 오류",
  noLora: "LoRA 없음 (빼고 생성)",
  noNegpip: "NegPiP 꺼짐: 음수 가중치가 그림을 망침",
});

const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);

/** A queue error's text: ComfyUI's PromptExecutionError carries `{error, node_errors}` in `response`. */
function errorText(error) {
  const response = error?.response;
  const nodeErrors = Object.values(isObject(response?.node_errors) ? response.node_errors : {})
    .flatMap((node) => node?.errors || [])
    .slice(0, 2)
    .map((item) => [item.message, item.details].filter(Boolean).join(": "));
  const head = response?.error?.message || error?.message || String(error);
  return [head, ...nodeErrors].join("\n");
}

function savedState(output, nodeId) {
  try {
    const state = JSON.parse(output?.[nodeId]?.inputs?.studio_state);
    return isObject(state) ? state : null;
  } catch {
    return null;
  }
}

/** The seed image `index` of a batch was drawn with: seed + index with Forge-like noise (Forge), else the batch seed. */
function imageSeed(state, index) {
  const s = withDefaults(state);
  return s.forge_like.master && s.forge_like.noise ? s.seed + index : s.seed;
}

const POST_SETTINGS = Object.freeze(["hires", "upscale", "sam3"]);

/**
 * 같은 시드's state: the record's settings (seed included) with the current post-processing sections as they are,
 * batch 1. A record read from PNG `parameters` has no tab: it regenerates as t2i, never as img2img over whatever
 * the panel's i2i box holds now.
 */
function sameSeedState(record, currentState) {
  const current = withDefaults(currentState);
  const state = postRunState(record, current, null);
  if (!record?.tab) state.tab = "t2i";
  for (const section of POST_SETTINGS) state[section] = { ...current[section] };
  state.batch = 1; // the picked image only
  return state;
}

const imageRef = (image) => ({ filename: image.filename, subfolder: image.subfolder || "", type: image.type || "output" });

export class RunController {
  /**
   * `deps`: `api` (ComfyUI api: queuePrompt, interrupt, deleteItem, addEventListener), `basePrompt(modes)` (graphToPrompt
   * with those node modes -> `{output, workflow}`), `roleIds()`, `getState()`, `getXyz()` (`{on, axes, options}`),
   * `route(path, body)` (studio route JSON; throws on an HTTP error), `readParameters(imageRef)` (PNG `parameters` or
   * null), `families()`, `upscalers()`, `loras()` (LoRA file list or undefined), `emit(type, payload)` for the views,
   * `now()` (ms).
   */
  constructor(deps) {
    this.deps = deps;
    this.job = null;
    this.starting = false; // claimed by a run that is still awaiting its plan / PNG read (see claim())
    this.early = new Map();
    this.progressBase = null;
    for (const name of EVENTS) deps.api.addEventListener(name, (event) => this.handleEvent(name, event?.detail));
  }

  get running() {
    return this.job !== null || this.starting;
  }

  /** Resolves when the current run (if any) has finished, its grid included. */
  whenIdle() {
    return this.job ? this.job.finished : Promise.resolve();
  }

  now() {
    return this.deps.now ? this.deps.now() : Date.now();
  }

  emit(type, payload) {
    this.deps.emit?.(type, payload);
  }

  notice(text) {
    this.emit("notice", { text });
    return { notice: text };
  }

  options() {
    return { families: this.deps.families?.() || undefined, upscalers: this.deps.upscalers?.() || undefined };
  }

  /** The panel state applied to an API prompt built with that state's stage modes (also used for Ctrl+Enter). */
  async panelPrompt(state = this.deps.getState()) {
    const options = this.options();
    const base = await this.deps.basePrompt(stageModes(state, null, options));
    return { output: applyStateToPrompt(base.output, this.deps.roleIds(), state, options), workflow: base.workflow };
  }

  /**
   * Runs `work` with the controller claimed: a second 생성 / gallery click while the first one still awaits its
   * plan or PNG read gets the busy notice instead of starting a second job (which would orphan the first). The
   * claim is released when `work` returns; by then run() has set `this.job` (or nothing was started).
   */
  async claim(work) {
    if (this.running) return this.notice(NOTICES.busy);
    this.starting = true;
    try {
      return await work();
    } finally {
      this.starting = false;
    }
  }

  /** Notices that never block: `<lora:…>` tags matching no file (the LoRA node skips them), `(x:-1)` without NegPiP. */
  warnRun(state) {
    const missing = missingLoras(state, this.deps.loras?.());
    if (missing.length) this.notice(`${NOTICES.noLora}: ${missing.join(", ")}`);
    if (!state.negpip && hasNegativeWeights(state)) this.notice(NOTICES.noNegpip);
  }

  /** 생성: the X/Y/Z plot when its panel is on (R17), else one generation of the panel state. */
  generate() {
    return this.claim(() => {
      const state = this.deps.getState();
      const missing = missingPicks(state);
      if (missing) return this.notice(missing);
      if (this.deps.getXyz?.()?.on) return this.generateXyz(state);
      this.warnRun(state);
      return this.run("single", 1, async (job) => {
        const prompt = await this.panelPrompt(state);
        await this.queueSlot(job, 0, prompt, undefined, String(this.deps.roleIds().save));
      });
    });
  }

  /** The X/Y/Z run (called by generate() with the controller claimed); the planner checks the cells' LoRAs. */
  async generateXyz(state) {
    const xyz = this.deps.getXyz?.() || {};
    let plan;
    try {
      plan = await this.deps.route("/xyz/plan", { state, axes: xyz.axes || [], options: xyz.options || {} });
    } catch (error) {
      return this.notice(errorText(error));
    }
    if (plan?.errors?.length) return this.notice(plan.errors.join("\n")); // R13: a rejected plan is a 200 with errors
    const cells = plan?.cells || [];
    if (!cells.length) return this.notice(NOTICES.noCells);
    return this.run("xyz", cells.length, async (job) => {
      job.plan = plan;
      job.xyzOptions = xyz.options || {};
      const options = this.options();
      const roleIds = this.deps.roleIds();
      const bases = new Map(); // R16: one base prompt per stage-mode signature
      for (const index of plan.order) {
        if (job.stopped) {
          this.resolveSlot(job, index, null);
          continue;
        }
        const modes = stageModes(cells[index].state, null, options);
        const key = JSON.stringify(modes);
        if (!bases.has(key)) bases.set(key, await this.deps.basePrompt(modes));
        const base = bases.get(key);
        const output = applyStateToPrompt(base.output, roleIds, cells[index].state, options);
        await this.queueSlot(job, index, { output, workflow: base.workflow }, undefined, String(roleIds.save));
      }
    });
  }

  /** The settings an image was made with (flat state incl. `seed`): its session record, else its PNG `parameters`. */
  async imageRecord(item) {
    if (!item?.image) return null;
    const record = item.record;
    if (isObject(record?.state)) return { ...record.state, seed: record.seed ?? record.state.seed };
    return recordFromParameters(await this.readParameters(item.image));
  }

  /** Gallery 업스케일 / Hires / SAM3 on one image: its own settings, the current post-processing settings (spec §4). */
  runPost(action, item) {
    return this.claim(() => this.startPost(action, item));
  }

  async startPost(action, item) {
    if (!item?.image) return this.notice(NOTICES.noImage);
    const flat = await this.imageRecord(item);
    if (!flat) return this.notice(NOTICES.noInfo);
    const record = { ...flat, image: item.image };
    const current = this.deps.getState();
    const runState = postRunState(record, current, action);
    const missing = missingPicks(runState, action);
    if (missing) return this.notice(missing);
    if (action !== "upscale") this.warnRun(runState);
    return this.run("post", 1, async (job) => {
      const options = this.options();
      const roleIds = this.deps.roleIds();
      const base = await this.deps.basePrompt(stageModes(runState, action, options));
      const { prompt, outputNodeIds } = postPrompt(base.output, roleIds, record, action, current, options);
      const data = { output: prompt, workflow: base.workflow };
      await this.queueSlot(job, 0, data, { partialExecutionTargets: outputNodeIds }, String(roleIds.post_save));
    });
  }

  /**
   * 같은 시드 on one image (spec §3.2): generate it again right away — its own settings and seed (session record,
   * else PNG `parameters`), the current post-processing settings, one image, never the X/Y/Z plot.
   */
  sameSeed(item) {
    return this.claim(async () => {
      if (!item?.image) return this.notice(NOTICES.noImage);
      const record = await this.imageRecord(item);
      if (!record) return this.notice(NOTICES.noInfo);
      const state = sameSeedState(record, this.deps.getState());
      const missing = missingPicks(state);
      if (missing) return this.notice(missing);
      this.warnRun(state);
      const result = await this.run("single", 1, async (job) => {
        const prompt = await this.panelPrompt(state);
        await this.queueSlot(job, 0, prompt, undefined, String(this.deps.roleIds().save));
      });
      return result.queued ? { ...result, seed: record.seed } : result;
    });
  }

  /**
   * 정지: drop this controller's pending prompts, interrupt the running one (R11); unfinished cells become empty.
   * A started prompt is given up too, so a lost end event (websocket reconnect, ComfyUI restart) cannot leave the
   * run stuck; its late events are ignored.
   */
  async stop() {
    const job = this.job;
    if (!job || job.stopped) return;
    job.stopped = true;
    const pending = [...job.entries.values()].filter((entry) => !entry.done && !entry.started);
    pending.forEach((entry) => {
      entry.done = true;
    });
    try {
      await this.cancelPrompts(pending.map((entry) => entry.id));
    } finally {
      pending.forEach((entry) => this.resolveSlot(job, entry.slot, null));
      for (const entry of job.entries.values()) {
        if (entry.done) continue;
        entry.done = true;
        this.resolveSlot(job, entry.slot, null);
      }
    }
  }

  // ------------------------------------------------------------------------------------------- jobs

  async readParameters(image) {
    try {
      return (await this.deps.readParameters(image)) || null;
    } catch {
      return null;
    }
  }

  async cancelPrompts(ids) {
    const { api } = this.deps;
    for (const id of ids) await api.deleteItem("queue", id).catch(() => {}); // before the interrupt: else the next one starts
    await api.interrupt(null).catch(() => {});
  }

  async run(kind, slots, body) {
    let resolve;
    const finished = new Promise((done) => {
      resolve = done;
    });
    const job = { kind, slots, results: new Array(slots).fill(undefined), remaining: slots, entries: new Map() };
    Object.assign(job, { finished, resolve, stopped: false, notice: "", durations: [] });
    this.job = job;
    this.early.clear();
    this.progressBase = null;
    this.emit("running", { running: true, kind });
    try {
      await body(job);
    } catch (error) {
      this.noticeOnce(job, errorText(error));
      const queued = new Set([...job.entries.values()].map((entry) => entry.slot));
      for (let slot = 0; slot < slots; slot++) if (!queued.has(slot)) this.resolveSlot(job, slot, null);
    }
    if (!job.entries.size && job.notice) return { notice: job.notice };
    return { queued: job.entries.size };
  }

  noticeOnce(job, text) {
    if (job.notice) return;
    job.notice = text;
    this.emit("notice", { text });
  }

  async queueSlot(job, slot, data, options, outputId) {
    if (job.stopped) return this.resolveSlot(job, slot, null);
    let response;
    try {
      response = await this.deps.api.queuePrompt(0, data, options);
    } catch (error) {
      this.noticeOnce(job, errorText(error));
      return this.resolveSlot(job, slot, null);
    }
    const id = response?.prompt_id === undefined || response?.prompt_id === null ? "" : String(response.prompt_id);
    if (!id) return this.resolveSlot(job, slot, null);
    const entry = { id, slot, outputId, state: savedState(data.output, outputId), images: [] };
    Object.assign(entry, { started: false, done: false, startedAt: 0 });
    job.entries.set(id, entry);
    if (job.stopped) {
      entry.done = true;
      this.resolveSlot(job, slot, null);
      return this.cancelPrompts([id]);
    }
    const early = this.early.get(id) || [];
    this.early.delete(id);
    for (const [name, detail] of early) this.handleEvent(name, detail);
    return undefined;
  }

  resolveSlot(job, slot, images) {
    if (job.results[slot] !== undefined) return;
    job.results[slot] = images && images.length ? images : null;
    job.remaining -= 1;
    if (job.remaining === 0) this.complete(job);
  }

  async complete(job) {
    try {
      if (job.kind === "xyz") await this.drawGrid(job);
    } catch (error) {
      this.notice(errorText(error));
    } finally {
      if (this.job === job) {
        this.job = null;
        this.early.clear();
      }
      this.emit("running", { running: false, kind: job.kind });
      job.resolve();
    }
  }

  async drawGrid(job) {
    const { plan, xyzOptions: options } = job;
    const cells = plan.cells.map((cell, index) => ({
      ix: cell.ix,
      iy: cell.iy,
      iz: cell.iz,
      image: job.results[index]?.[0] || null,
    }));
    const first = cells[0].image || cells.find((cell) => cell.image)?.image;
    if (!first) {
      this.notice(NOTICES.noCells);
      return;
    }
    const labels = plan.labels || {};
    const count = (letter) => (Array.isArray(labels[letter]) && labels[letter].length) || 1;
    const result = await this.deps.route("/xyz/grid", {
      nx: count("x"),
      ny: count("y"),
      nz: count("z"),
      labels,
      options: {
        draw_legend: options.draw_legend !== false,
        include_sub_grids: Boolean(options.include_sub_grids),
        margin: Number(options.margin) || 0,
        row_count: Number(options.row_count) || 0,
      },
      cells,
      infotext: withExtraParams(await this.readParameters(first), plan.infotext_extra),
    });
    const subgrids = (result.subgrids || []).map((image) => ({ image, record: null }));
    this.emit("grid", { item: { image: result.grid, record: null }, subgrids });
  }

  // ------------------------------------------------------------------------------------------- events

  handleEvent(name, detail) {
    const job = this.job;
    if (!job) return;
    if (name === "b_preview") {
      if (detail) this.emit("preview", { blob: detail });
      return;
    }
    if (name === "executing") {
      this.progressBase = null; // a new node: its own progress rate
      return;
    }
    const id = detail?.prompt_id === undefined || detail?.prompt_id === null ? "" : String(detail.prompt_id);
    const entry = job.entries.get(id);
    if (!entry) {
      if (id) this.keepEarly(id, name, detail);
      return;
    }
    if (entry.done) return;
    if (name === "execution_start") {
      entry.started = true;
      entry.startedAt = this.now();
      this.progressBase = null;
    } else if (name === "executed") {
      if ([detail.node, detail.display_node].some((node) => String(node) === entry.outputId)) {
        entry.images.push(...(detail.output?.images || []).filter((image) => image?.filename).map(imageRef));
      }
    } else if (name === "execution_success") {
      this.finishEntry(job, entry, true);
    } else if (name === "execution_error") {
      this.noticeOnce(job, detail.exception_message || NOTICES.failed);
      this.finishEntry(job, entry, false);
    } else if (name === "execution_interrupted") {
      this.finishEntry(job, entry, false);
    } else if (name === "progress") {
      this.onProgress(job, entry, detail);
    }
  }

  keepEarly(id, name, detail) {
    const total = [...this.early.values()].reduce((sum, list) => sum + list.length, 0);
    if (total >= EARLY_EVENT_LIMIT) this.early.delete(this.early.keys().next().value);
    if (!this.early.has(id)) this.early.set(id, []);
    this.early.get(id).push([name, detail]);
  }

  finishEntry(job, entry, ok) {
    entry.done = true;
    const images = ok ? entry.images : [];
    if (ok && entry.startedAt) job.durations.push(this.now() - entry.startedAt);
    if (images.length && (job.kind !== "xyz" || job.xyzOptions?.include_images)) {
      const items = images.map((image, index) => {
        if (!entry.state) return { image, record: null };
        const seed = imageSeed(entry.state, index);
        return { image, record: { state: { ...entry.state, seed }, seed } };
      });
      this.emit("images", { items });
    }
    this.resolveSlot(job, entry.slot, images);
  }

  onProgress(job, entry, detail) {
    const value = Number(detail.value) || 0;
    const max = Number(detail.max) || 0;
    const now = this.now();
    const base = this.progressBase;
    if (!base || base.node !== detail.node || value < base.value) this.progressBase = { node: detail.node, value, time: now };
    const start = this.progressBase;
    let eta = value > start.value ? (((now - start.time) / (value - start.value)) * (max - value)) / 1000 : null;
    const resolved = job.slots - job.remaining;
    const waiting = job.remaining - 1; // cells after the running one
    if (eta !== null && job.kind === "xyz" && waiting > 0 && job.durations.length) {
      eta += (job.durations.reduce((a, b) => a + b, 0) / job.durations.length / 1000) * waiting;
    }
    this.emit("progress", {
      value,
      max,
      percent: max > 0 ? Math.round((value * 100) / max) : 0,
      eta,
      cell: job.kind === "xyz" ? { index: resolved + 1, total: job.slots } : null,
    });
  }
}
