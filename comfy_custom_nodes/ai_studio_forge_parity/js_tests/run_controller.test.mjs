// node:test for the studio's run controller (web/ai_studio/run_controller.js): the state machine only, no DOM.
// A fake ComfyUI `api` records queue / delete / interrupt calls and fires websocket events in any order.
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { deflateSync } from "node:zlib";

import { NOTICES, RunController } from "../web/ai_studio/run_controller.js";
import { DEFAULT_STATE, stageModes, withDefaults } from "../web/ai_studio/studio_logic.js";
import { readPngText, withExtraParams } from "../web/ai_studio/studio_pnginfo.js";
import { ROLES } from "../web/ai_studio/studio_roles.js";

const golden = JSON.parse(readFileSync(new URL("./fixtures/studio_logic_golden.json", import.meta.url), "utf-8"));

const ROLE_IDS = Object.fromEntries(ROLES.map((role, i) => [role, String(i + 1)]));
const SAVE = ROLE_IDS.save;
const POST_SAVE = ROLE_IDS.post_save;

class FakeApi {
  constructor() {
    this.listeners = new Map();
    this.queued = [];
    this.log = [];
    this.failNext = null;
  }

  addEventListener(name, fn) {
    if (!this.listeners.has(name)) this.listeners.set(name, []);
    this.listeners.get(name).push(fn);
  }

  fire(name, detail) {
    for (const fn of this.listeners.get(name) || []) fn({ detail });
  }

  async queuePrompt(number, data, options) {
    if (this.failNext) {
      const error = this.failNext;
      this.failNext = null;
      throw error;
    }
    const id = `p${this.queued.length}`;
    this.queued.push({ id, number, data, options });
    this.log.push(["queue", id]);
    if (this.onQueue) this.onQueue(id);
    return { prompt_id: id, number: this.queued.length, node_errors: {} };
  }

  async deleteItem(type, id) {
    this.log.push(["delete", type, id]);
  }

  async interrupt(id) {
    this.log.push(["interrupt", id]);
  }
}

// graphToPrompt stand-in: every role node, bypassed (mode 4) ones omitted like ComfyUI does.
function fakeBasePrompt(calls) {
  return async (modes) => {
    calls.push(modes);
    const output = {};
    for (const role of ROLES) {
      if (modes[role] === 4) continue;
      output[ROLE_IDS[role]] = { class_type: `Fake_${role}`, inputs: {}, _meta: { title: role } };
    }
    return { output, workflow: { nodes: [] } };
  };
}

const ref = (filename) => ({ filename, subfolder: "AIStudio", type: "output" });

// A 2x2 X/Y plan whose generation order is not the cell order (X loops outermost).
function plan2x2(stateOf = () => ({})) {
  const cells = [];
  for (let iy = 0; iy < 2; iy++) {
    for (let ix = 0; ix < 2; ix++) {
      const index = ix + iy * 2;
      cells.push({ ix, iy, iz: 0, state: withDefaults({ seed: 100 + index, ...stateOf(index) }) });
    }
  }
  return {
    cells,
    order: [0, 2, 1, 3],
    labels: { x: ["Seed: 100", "Seed: 101"], y: ["Steps: 20", "Steps: 30"], z: [""] },
    fixed: {},
    infotext_extra: { Script: "X/Y/Z plot", "X Type": "Seed", "X Values": "100, 101", "Y Type": "Steps", "Y Values": "20,30" },
    errors: [],
  };
}

// The panel's model picks (missingPicks refuses a run while any is empty).
const PICKS = { unet: "anima-2.9b.safetensors", te: "qwen_3_06b_base.safetensors", vae: "qwen_image_vae.safetensors" };

function setup({ plan = plan2x2(), xyz = { on: false }, parameters = null, state = {}, route = null, loras = undefined, extFiles = undefined, reloadLoras = undefined, roleClass = undefined, now = () => 0 } = {}) {
  const api = new FakeApi();
  const emitted = [];
  const routes = [];
  const baseCalls = [];
  const readCalls = [];
  const controller = new RunController({
    api,
    basePrompt: fakeBasePrompt(baseCalls),
    roleIds: () => ROLE_IDS,
    roleClass,
    getState: () => withDefaults({ ...structuredClone(DEFAULT_STATE), ...PICKS, ...state }),
    getXyz: () => xyz,
    route: async (path, body) => {
      routes.push([path, body]);
      if (route) return route(path, body);
      if (path === "/xyz/plan") return plan;
      if (path === "/xyz/grid") return { grid: ref("xyz_grid_00001.png"), subgrids: [] };
      throw new Error(`unexpected route ${path}`);
    },
    readParameters: async (image) => {
      readCalls.push(image);
      return typeof parameters === "function" ? parameters(image) : parameters;
    },
    families: () => ({}),
    upscalers: () => undefined,
    loras: () => (typeof loras === "function" ? loras() : loras),
    reloadLoras,
    extFiles: () => extFiles,
    emit: (type, payload) => emitted.push([type, payload]),
    now,
  });
  return { api, controller, emitted, routes, baseCalls, readCalls };
}

const XYZ_ON = {
  on: true,
  axes: [{ axis: "seed", text: "100, 101" }, { axis: "steps", text: "20,30" }, { axis: "nothing", text: "" }],
  options: { draw_legend: true, include_images: false, include_sub_grids: false, margin: 0, row_count: 0 },
};

function finishCell(api, id, filename) {
  api.fire("execution_start", { prompt_id: id });
  api.fire("executed", { prompt_id: id, node: SAVE, display_node: SAVE, output: { images: [ref(filename)] } });
  api.fire("execution_success", { prompt_id: id });
}

const gridBody = (routes) => routes.find(([path]) => path === "/xyz/grid")?.[1];
const cellImages = (body) =>
  [...body.cells].sort((a, b) => a.ix + a.iy * 2 - (b.ix + b.iy * 2)).map((cell) => cell.image && cell.image.filename);
const emittedOf = (emitted, type) => emitted.filter(([kind]) => kind === type).map(([, payload]) => payload);

// ---------------------------------------------------------------------------------------------- XYZ

test("xyz run collects images in plan order", async () => {
  const plan = plan2x2();
  const { api, controller, emitted, routes } = setup({ plan, xyz: XYZ_ON, parameters: golden.infotext.text });
  await controller.generate();
  assert.equal(api.queued.length, 4);
  // queue k holds cell plan.order[k]
  api.queued.forEach((entry, k) => {
    assert.equal(entry.data.output[ROLE_IDS.sampler].inputs.seed, 100 + plan.order[k]);
  });
  assert.equal(controller.running, true);
  // a preview node's executed event is not a cell image
  api.fire("executed", { prompt_id: "p0", node: ROLE_IDS.decode, output: { images: [ref("preview.png")] } });
  for (const k of [3, 1, 0, 2]) finishCell(api, `p${k}`, `cell${plan.order[k]}.png`);
  await controller.whenIdle();

  const body = gridBody(routes);
  assert.deepEqual([body.nx, body.ny, body.nz], [2, 2, 1]);
  assert.deepEqual(cellImages(body), ["cell0.png", "cell1.png", "cell2.png", "cell3.png"]);
  assert.deepEqual(body.labels, plan.labels);
  assert.deepEqual(body.options, { draw_legend: true, include_sub_grids: false, margin: 0, row_count: 0 });
  assert.match(body.infotext, /Seed: 42, .*Script: X\/Y\/Z plot, X Type: Seed, X Values: "100, 101", Y Type: Steps, Y Values: "20,30"$/);
  const [grid] = emittedOf(emitted, "grid");
  assert.deepEqual(grid.item.image, ref("xyz_grid_00001.png"));
  assert.equal(controller.running, false);
  assert.deepEqual(emittedOf(emitted, "running").map((e) => e.running), [true, false]);
  assert.equal(emittedOf(emitted, "images").length, 0); // include_images off
});

test("stop marks remaining cells null", async () => {
  const plan = plan2x2();
  const { api, controller, routes } = setup({ plan, xyz: XYZ_ON });
  await controller.generate();
  finishCell(api, "p0", `cell${plan.order[0]}.png`);
  api.fire("execution_start", { prompt_id: "p1" });
  await controller.stop();
  // pending prompts are deleted before the global interrupt (else the server would start the next one)
  assert.deepEqual(api.log.slice(4), [["delete", "queue", "p2"], ["delete", "queue", "p3"], ["interrupt", null]]);
  api.fire("execution_interrupted", { prompt_id: "p1" });
  api.fire("execution_success", { prompt_id: "p2" }); // too late: already given up
  await controller.whenIdle();
  const images = cellImages(gridBody(routes));
  assert.deepEqual(images, [`cell${plan.order[0]}.png`, null, null, null]);
  assert.equal(controller.running, false);
});

test("stop during queueing never queues the remaining cells", async () => {
  const plan = plan2x2();
  const { api, controller, routes } = setup({ plan, xyz: XYZ_ON });
  api.onQueue = (id) => {
    if (id === "p1") controller.stop();
  };
  await controller.generate();
  await controller.whenIdle();
  assert.equal(api.queued.length, 2);
  assert.ok(api.log.some((entry) => entry[0] === "delete" && entry[2] === "p1"));
  assert.equal(gridBody(routes), undefined); // no cell finished: nothing to draw
});

test("stop ends a started prompt whose end event never comes (lost websocket event)", async () => {
  const { api, controller, emitted } = setup();
  await controller.generate();
  api.fire("execution_start", { prompt_id: "p0" });
  await controller.stop(); // the interrupt is never answered
  assert.equal(controller.running, false); // before whenIdle(), which would never resolve if this failed
  await controller.whenIdle();
  assert.deepEqual(api.log.slice(1), [["interrupt", null]]);
  assert.deepEqual(emittedOf(emitted, "running").map((e) => e.running), [true, false]);
  api.fire("executed", { prompt_id: "p0", node: SAVE, output: { images: [ref("late.png")] } });
  api.fire("execution_success", { prompt_id: "p0" }); // too late: ignored
  assert.deepEqual(emittedOf(emitted, "images"), []);
  assert.deepEqual(await controller.generate(), { queued: 1 }); // not stuck on "이미 생성 중"
});

test("failed cell becomes null", async () => {
  const plan = plan2x2();
  const { api, controller, emitted, routes } = setup({ plan, xyz: XYZ_ON });
  await controller.generate();
  finishCell(api, "p0", `cell${plan.order[0]}.png`);
  api.fire("execution_start", { prompt_id: "p1" });
  api.fire("execution_error", { prompt_id: "p1", exception_message: "CUDA out of memory" });
  finishCell(api, "p2", `cell${plan.order[2]}.png`);
  finishCell(api, "p3", `cell${plan.order[3]}.png`);
  await controller.whenIdle();
  const images = cellImages(gridBody(routes));
  const failed = plan.order[1];
  assert.equal(images[failed], null);
  assert.equal(images.filter(Boolean).length, 3);
  assert.ok(emittedOf(emitted, "notice").some((n) => n.text.includes("CUDA out of memory")));
});

test("xyz plan errors are shown and nothing is queued (R13)", async () => {
  const plan = { cells: [], order: [], labels: {}, errors: ['Prompt S/R did not find "x" in either prompt or negative prompt'] };
  const { api, controller, emitted } = setup({ plan, xyz: XYZ_ON });
  const result = await controller.generate();
  assert.equal(api.queued.length, 0);
  assert.match(result.notice, /Prompt S\/R did not find/);
  assert.match(emittedOf(emitted, "notice")[0].text, /Prompt S\/R did not find/);
  assert.equal(controller.running, false);
});

test("xyz cells get the stage modes of their own state (R16)", async () => {
  const plan = plan2x2((index) => ({ sam3: { on: index % 2 === 1 } }));
  const { api, controller, baseCalls } = setup({ plan, xyz: XYZ_ON });
  await controller.generate();
  assert.equal(baseCalls.length, 2); // one base prompt per stage-mode signature
  assert.deepEqual(baseCalls.map((modes) => modes.sam3_detailer).sort(), [0, 4]);
  api.queued.forEach((entry, k) => {
    const detailer = entry.data.output[ROLE_IDS.sam3_detailer];
    if (plan.order[k] % 2 === 1) assert.equal(detailer.inputs.enabled, true);
    else assert.equal(detailer, undefined);
  });
});

test("cell images go to the gallery when include_images is on", async () => {
  const plan = plan2x2();
  const xyz = { ...XYZ_ON, options: { ...XYZ_ON.options, include_images: true } };
  const { api, controller, emitted } = setup({ plan, xyz });
  await controller.generate();
  for (const k of [0, 1, 2, 3]) finishCell(api, `p${k}`, `cell${plan.order[k]}.png`);
  await controller.whenIdle();
  const items = emittedOf(emitted, "images").flatMap((e) => e.items);
  assert.equal(items.length, 4);
  assert.equal(items[0].record.state.seed, 100 + plan.order[0]);
  assert.equal(items[0].record.seed, 100 + plan.order[0]);
});

// ---------------------------------------------------------------------------------------------- single run

test("generate queues the panel state with its stage modes and records each image's seed", async () => {
  const { api, controller, emitted, baseCalls } = setup({ state: { seed: 7, batch: 2, steps: 20 } });
  const result = await controller.generate();
  assert.deepEqual(result, { queued: 1 });
  assert.deepEqual(baseCalls, [stageModes(withDefaults({ ...PICKS, seed: 7, batch: 2, steps: 20 }), null, { families: {} })]);
  const [{ data, number }] = api.queued;
  assert.equal(number, 0);
  assert.equal(data.output[ROLE_IDS.sampler].inputs.steps, 20);
  assert.equal(data.output[ROLE_IDS.sampler].inputs.seed, 7);
  api.fire("execution_start", { prompt_id: "p0" });
  api.fire("progress", { prompt_id: "p0", node: ROLE_IDS.sampler, value: 5, max: 20 });
  api.fire("progress", { prompt_id: "other", node: "9", value: 1, max: 2 }); // someone else's prompt
  api.fire("executed", { prompt_id: "p0", node: SAVE, output: { images: [ref("a.png"), ref("b.png")] } });
  api.fire("execution_success", { prompt_id: "p0" });
  await controller.whenIdle();
  const progress = emittedOf(emitted, "progress");
  assert.equal(progress.length, 1);
  assert.deepEqual([progress[0].value, progress[0].max, progress[0].percent], [5, 20, 25]);
  const [{ items }] = emittedOf(emitted, "images");
  assert.deepEqual(items.map((item) => item.image.filename), ["a.png", "b.png"]);
  assert.deepEqual(items.map((item) => item.record.seed), [7, 8]); // Forge-like noise: image i uses seed + i
  assert.equal(items[1].record.state.seed, 8);
  assert.equal(items[1].record.state.steps, 20);
});

test("events that arrive before queuePrompt answers are not lost", async () => {
  const { api, controller, emitted } = setup();
  api.onQueue = (id) => {
    api.fire("execution_start", { prompt_id: id });
    api.fire("executed", { prompt_id: id, node: SAVE, output: { images: [ref("fast.png")] } });
    api.fire("execution_success", { prompt_id: id });
  };
  await controller.generate();
  await controller.whenIdle();
  assert.equal(emittedOf(emitted, "images")[0].items[0].image.filename, "fast.png");
});

test("a prompt the server rejects ends the run with a notice", async () => {
  const { api, controller, emitted } = setup();
  api.failNext = Object.assign(new Error("Prompt outputs failed validation"), {});
  const result = await controller.generate();
  assert.match(result.notice, /failed validation/);
  assert.equal(controller.running, false);
  assert.deepEqual(emittedOf(emitted, "running").map((e) => e.running), [true, false]);
});

test("generate while running answers with a notice and queues nothing more", async () => {
  const { api, controller } = setup();
  await controller.generate();
  const second = await controller.generate();
  assert.ok(second.notice);
  assert.equal(api.queued.length, 1);
});

test("panelPrompt is the panel state applied to a prompt built with its stage modes", async () => {
  const { controller, baseCalls } = setup({ state: { tab: "i2i", steps: 12 } });
  const { output, workflow } = await controller.panelPrompt();
  assert.equal(baseCalls[0].i2i_image, 0);
  assert.equal(output[ROLE_IDS.sampler].inputs.steps, 12);
  assert.equal(output[ROLE_IDS.latent].inputs.mode, "img2img");
  assert.deepEqual(workflow, { nodes: [] });
});

// ---------------------------------------------------------------------------------------------- post-processing

test("post without record uses parameters", async () => {
  const { api, controller, emitted, baseCalls, readCalls } = setup({ parameters: golden.infotext.text });
  const item = { image: ref("old.png"), record: null };
  const result = await controller.runPost("hires", item);
  assert.deepEqual(result, { queued: 1 });
  assert.deepEqual(readCalls, [item.image]);
  assert.equal(baseCalls[0].post_hires, 0);
  assert.equal(baseCalls[0].post_image, 0);
  const [{ data, options }] = api.queued;
  assert.deepEqual(options, { partialExecutionTargets: [POST_SAVE] });
  assert.equal(data.output[ROLE_IDS.post_image].inputs.image, "AIStudio/old.png [output]");
  assert.equal(data.output[ROLE_IDS.post_hires].inputs.enabled, true);
  assert.equal(data.output[POST_SAVE].inputs.seed, 42);
  assert.equal(data.output[ROLE_IDS.lora_stack].inputs.positive, "1girl, solo <lora:meow:1>");
  api.fire("executed", { prompt_id: "p0", node: POST_SAVE, output: { images: [ref("hires.png")] } });
  api.fire("execution_success", { prompt_id: "p0" });
  await controller.whenIdle();
  const [{ items }] = emittedOf(emitted, "images");
  assert.equal(items[0].image.filename, "hires.png");
  assert.equal(items[0].record.seed, 42);
});

test("post with a session record uses it, not the image's parameters", async () => {
  const { api, controller, readCalls } = setup({ parameters: golden.infotext.text, state: { upscale: { model: "4x-UltraSharp.pth" } } });
  const state = withDefaults({ ...PICKS, prompt: "from the session", seed: 9 });
  await controller.runPost("upscale", { image: ref("s.png"), record: { state, seed: 9 } });
  assert.deepEqual(readCalls, []);
  const [{ data }] = api.queued;
  assert.equal(data.output[ROLE_IDS.lora_stack].inputs.positive, "from the session");
  assert.equal(data.output[ROLE_IDS.post_upscale].inputs.enabled, true);
  assert.equal(data.output[POST_SAVE].inputs.seed, 9);
});

test("post without anything returns notice", async () => {
  const { api, controller, emitted, baseCalls } = setup({ parameters: null });
  const result = await controller.runPost("sam3", { image: ref("disk.png"), record: null });
  assert.ok(result.notice);
  assert.equal(api.queued.length, 0);
  assert.equal(baseCalls.length, 0);
  assert.equal(emittedOf(emitted, "notice")[0].text, result.notice);
  assert.ok((await controller.runPost("sam3", null)).notice); // no image picked
});

test("imageRecord gives the seed of a session image or of the PNG parameters", async () => {
  const { controller } = setup({ parameters: (image) => (image.filename === "info.png" ? golden.infotext.text : null) });
  assert.equal((await controller.imageRecord({ image: ref("s.png"), record: { state: withDefaults({ seed: 5 }), seed: 6 } })).seed, 6);
  assert.equal((await controller.imageRecord({ image: ref("info.png"), record: null })).seed, 42);
  assert.equal(await controller.imageRecord({ image: ref("none.png"), record: null }), null);
});

test("same seed regenerates the picked image's settings with the current post settings (spec §3.2)", async () => {
  const current = {
    prompt: "the panel now",
    seed: 1,
    batch: 4,
    hires: { on: true, denoise: 0.3 },
    sam3: { on: true, prompt: "hands" },
  };
  const { api, controller, emitted, routes, baseCalls, readCalls } = setup({ state: current, xyz: XYZ_ON });
  const state = withDefaults({ ...PICKS, prompt: "from the session", seed: 9, steps: 20, batch: 2, hires: { on: false, denoise: 0.7 } });
  const result = await controller.sameSeed({ image: ref("s.png"), record: { state, seed: 10 } });
  assert.deepEqual(result, { queued: 1, seed: 10 }); // studio_main puts the seed into the panel too
  assert.deepEqual(readCalls, []);
  assert.deepEqual(routes, []); // one image, not the X/Y/Z plot the panel has on
  assert.equal(baseCalls[0].sam3_detailer, 0); // the current SAM3 toggle
  assert.equal(baseCalls[0].post_hires, 4);
  const [{ data, options }] = api.queued;
  assert.equal(options, undefined);
  const inputs = (role) => data.output[ROLE_IDS[role]].inputs;
  assert.equal(inputs("lora_stack").positive, "from the session");
  assert.equal(inputs("sampler").seed, 10);
  assert.equal(inputs("sampler").steps, 20);
  assert.equal(inputs("latent").batch_size, 1); // the picked image only
  assert.deepEqual([inputs("hires").enabled, inputs("hires").denoise], [true, 0.3]);
  assert.deepEqual([inputs("sam3_detailer").enabled, inputs("sam3_mask").prompt], [true, "hands"]);
  finishCell(api, "p0", "again.png");
  await controller.whenIdle();
  const [{ items }] = emittedOf(emitted, "images");
  assert.equal(items[0].image.filename, "again.png");
  assert.equal(items[0].record.seed, 10);
});

test("same seed without a session record uses the PNG parameters, else a notice", async () => {
  const { api, controller } = setup({ parameters: (image) => (image.filename === "info.png" ? golden.infotext.text : null) });
  await controller.sameSeed({ image: ref("info.png"), record: null });
  assert.equal(api.queued[0].data.output[ROLE_IDS.sampler].inputs.seed, 42);
  assert.ok((await controller.sameSeed({ image: ref("other.png"), record: null })).notice); // still busy
  finishCell(api, "p0", "x.png");
  await controller.whenIdle();
  assert.ok((await controller.sameSeed({ image: ref("none.png"), record: null })).notice);
  assert.ok((await controller.sameSeed(null)).notice);
  assert.equal(api.queued.length, 1);
});

// ---------------------------------------------------------------------------------------------- studio_pnginfo.js

test("withExtraParams appends keys to the parameters line, Forge-quoted", () => {
  const text = "a cat\nNegative prompt: dog\nSteps: 20, Seed: 1";
  assert.equal(
    withExtraParams(text, { Script: "X/Y/Z plot", "X Values": "1, 2", "Y Values": "a:b" }),
    'a cat\nNegative prompt: dog\nSteps: 20, Seed: 1, Script: X/Y/Z plot, X Values: "1, 2", Y Values: "a:b"',
  );
  assert.equal(withExtraParams("", { Script: "X/Y/Z plot" }), "Script: X/Y/Z plot");
  assert.equal(withExtraParams(null, {}), "");
});

function pngChunk(type, data) {
  const head = Buffer.alloc(8);
  head.writeUInt32BE(data.length, 0);
  head.write(type, 4, "latin1");
  return Buffer.concat([head, data, Buffer.alloc(4)]);
}

test("readPngText reads tEXt and iTXt (plain and compressed) chunks", async () => {
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const korean = "소녀, 1girl\nSteps: 20, Seed: 3";
  const itxt = (key, text, compressed) =>
    Buffer.concat([
      Buffer.from(`${key}\0`, "latin1"),
      Buffer.from([compressed ? 1 : 0, 0]),
      Buffer.from("\0\0", "latin1"),
      compressed ? deflateSync(Buffer.from(text, "utf-8")) : Buffer.from(text, "utf-8"),
    ]);
  const png = Buffer.concat([
    signature,
    pngChunk("IHDR", Buffer.alloc(13)),
    pngChunk("tEXt", Buffer.from("prompt\0{\"1\": {}}", "latin1")),
    pngChunk("iTXt", itxt("parameters", korean, false)),
    pngChunk("iTXt", itxt("workflow", "{}", true)),
    pngChunk("IEND", Buffer.alloc(0)),
  ]);
  const text = await readPngText(png.buffer.slice(png.byteOffset, png.byteOffset + png.length));
  assert.equal(text.parameters, korean);
  assert.equal(text.prompt, '{"1": {}}');
  assert.equal(text.workflow, "{}");
  assert.deepEqual(await readPngText(new ArrayBuffer(4)), {});
});

// ---------------------------------------------------------------------------------------------- final fix wave

function deferred() {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

test("a second 생성 while the X/Y/Z plan is still awaited gets the busy notice and starts no second job", async () => {
  const plan = plan2x2();
  const pending = deferred();
  const { api, controller, routes, emitted } = setup({
    xyz: XYZ_ON,
    route: (path) => (path === "/xyz/plan" ? pending.promise : { grid: ref("xyz_grid_00001.png"), subgrids: [] }),
  });
  const first = controller.generate();
  const second = await controller.generate(); // double-click / held Ctrl+Enter during the plan request
  assert.equal(second.notice, "이미 생성 중");
  assert.equal(controller.running, true);
  pending.resolve(plan);
  assert.deepEqual(await first, { queued: 4 });
  assert.equal(routes.filter(([path]) => path === "/xyz/plan").length, 1);
  assert.equal(api.queued.length, 4); // one job
  for (const k of [0, 1, 2, 3]) finishCell(api, `p${k}`, `cell${plan.order[k]}.png`);
  await controller.whenIdle();
  assert.equal(controller.running, false);
  assert.deepEqual(emittedOf(emitted, "running").map((e) => e.running), [true, false]);
});

test("a rejected plan releases the claim, so the next 생성 runs", async () => {
  const plan = { cells: [], order: [], labels: {}, errors: ["Steps: no values"] };
  const { controller } = setup({ plan, xyz: XYZ_ON });
  assert.match((await controller.generate()).notice, /no values/);
  assert.equal(controller.running, false);
  assert.match((await controller.generate()).notice, /no values/); // not "이미 생성 중"
});

test("a second gallery click while the PNG is read gets the busy notice", async () => {
  const pending = deferred();
  const { api, controller } = setup({ parameters: () => pending.promise });
  const first = controller.runPost("hires", { image: ref("old.png"), record: null });
  assert.equal((await controller.runPost("sam3", { image: ref("old.png"), record: null })).notice, "이미 생성 중");
  assert.equal((await controller.sameSeed({ image: ref("old.png"), record: null })).notice, "이미 생성 중");
  assert.equal((await controller.generate()).notice, "이미 생성 중");
  pending.resolve(golden.infotext.text);
  assert.deepEqual(await first, { queued: 1 });
  assert.equal(api.queued.length, 1);
});

test("an image without generation info releases the claim", async () => {
  const { controller } = setup({ parameters: null });
  assert.ok((await controller.runPost("hires", { image: ref("x.png"), record: null })).notice);
  assert.equal(controller.running, false);
  assert.deepEqual(await controller.generate(), { queued: 1 });
});

test("generate refuses empty model picks and 업스케일 without a model, with one notice", async () => {
  for (const state of [{ unet: "" }, { te: "" }, { vae: "" }]) {
    const { api, controller, emitted } = setup({ state });
    assert.equal((await controller.generate()).notice, "체크포인트·TE·VAE 고르기");
    assert.equal(api.queued.length, 0);
    assert.deepEqual(emittedOf(emitted, "notice").map((n) => n.text), ["체크포인트·TE·VAE 고르기"]);
    assert.equal(controller.running, false);
  }
  const upscale = setup({ state: { upscale: { on: true, model: "" } } });
  assert.equal((await upscale.controller.generate()).notice, "업스케일 모델 고르기");
  assert.equal(upscale.api.queued.length, 0);
  const xyz = setup({ state: { vae: "" }, xyz: XYZ_ON });
  assert.ok((await xyz.controller.generate()).notice);
  assert.deepEqual(xyz.routes, []); // refused before the plan request
});

test("post runs refuse empty picks: 업스케일 needs only its model", async () => {
  const session = { state: withDefaults({ ...PICKS, seed: 3 }), seed: 3 };
  const noModel = setup();
  assert.equal((await noModel.controller.runPost("upscale", { image: ref("a.png"), record: session })).notice, "업스케일 모델 고르기");
  const upscaleOnly = setup({ state: { unet: "", te: "", vae: "", upscale: { model: "4x.pth" } } });
  const record = { state: withDefaults({ seed: 3 }), seed: 3 }; // a session record without picks
  assert.deepEqual(await upscaleOnly.controller.runPost("upscale", { image: ref("a.png"), record }), { queued: 1 });
  const hires = setup({ state: { te: "" } });
  assert.equal((await hires.controller.runPost("hires", { image: ref("a.png"), record })).notice, "체크포인트·TE·VAE 고르기");
  assert.equal(hires.api.queued.length, 0);
});

test("a LoRA tag that matches no file gives a notice but still generates", async () => {
  const loras = ["meow.safetensors", "sub/na-e10.safetensors"];
  const { api, controller, emitted } = setup({
    loras,
    state: { prompt: "1girl <lora:na-e10:1> <lora:na-e11:0.5>", loras: [{ on: true, name: "Meow", strength: 1 }, { on: true, name: "gone", strength: 1 }] },
  });
  assert.deepEqual(await controller.generate(), { queued: 1 });
  assert.equal(api.queued.length, 1);
  assert.deepEqual(emittedOf(emitted, "notice").map((n) => n.text), ["LoRA 없음 (빼고 생성): na-e11, gone"]);
  const quiet = setup({ loras: undefined, state: { prompt: "<lora:x:1>" } }); // list not loaded yet: no notice
  await quiet.controller.generate();
  assert.deepEqual(emittedOf(quiet.emitted, "notice"), []);
});

test("a tag missing from the cached LoRA list is looked up once more before the notice", async () => {
  let list = ["old.safetensors"];
  let reloads = 0;
  const fresh = setup({
    loras: () => list,
    reloadLoras: async () => { reloads += 1; list = ["old.safetensors", "new-e10.safetensors"]; },
    state: { prompt: "1boy <lora:new-e10:1>" },
  });
  assert.deepEqual(await fresh.controller.generate(), { queued: 1 });
  assert.equal(reloads, 1);
  assert.deepEqual(emittedOf(fresh.emitted, "notice"), []);
  const gone = setup({ loras: () => list, reloadLoras: async () => { reloads += 1; }, state: { prompt: "<lora:gone:1>" } });
  await gone.controller.generate();
  assert.deepEqual(emittedOf(gone.emitted, "notice").map((n) => n.text), ["LoRA 없음 (빼고 생성): gone"]);
  reloads = 0;
  const known = setup({ loras: () => list, reloadLoras: async () => { reloads += 1; }, state: { prompt: "<lora:old:1>" } });
  await known.controller.generate();
  assert.equal(reloads, 0); // no request when nothing is missing
});

test("negative weights without NegPiP warn but still run; with NegPiP stay quiet", async () => {
  const off = setup({ state: { prompt: "1boy, (crowd, big eyes:-1.0), abs", negpip: false } });
  assert.deepEqual(await off.controller.generate(), { queued: 1 });
  assert.deepEqual(emittedOf(off.emitted, "notice").map((n) => n.text), ["NegPiP 꺼짐: 음수 가중치가 그림을 망침"]);
  const on = setup({ state: { prompt: "1boy, (crowd:-1.0)", negpip: true } });
  await on.controller.generate();
  assert.deepEqual(emittedOf(on.emitted, "notice"), []);
  const plain = setup({ state: { prompt: "1boy, (crowd:1.2), [a:b:0.5]", negative: "x", negpip: false } });
  await plain.controller.generate();
  assert.deepEqual(emittedOf(plain.emitted, "notice"), []);
});

test("same seed on a PNG-only image regenerates as t2i even when the panel is on i2i", async () => {
  const { api, controller, baseCalls } = setup({ parameters: golden.infotext.text, state: { tab: "i2i", i2i_mode: "inpaint", i2i: { image: "box.png" } } });
  await controller.sameSeed({ image: ref("info.png"), record: null });
  assert.equal(baseCalls[0].i2i_image, 4); // the i2i input is bypassed
  const inputs = api.queued[0].data.output[ROLE_IDS.latent].inputs;
  assert.equal(inputs.mode, "txt2img");
  assert.equal(api.queued[0].data.output[ROLE_IDS.sampler].inputs.denoise, 1.0);
  assert.equal(api.queued[0].data.output[ROLE_IDS.sampler].inputs.seed, 42);
  // a session record keeps its own tab
  finishCell(api, "p0", "x.png");
  await controller.whenIdle();
  const i2iRecord = { state: withDefaults({ ...PICKS, tab: "i2i", seed: 5, i2i: { image: "src.png" } }), seed: 5 };
  await controller.sameSeed({ image: ref("s.png"), record: i2iRecord });
  assert.equal(api.queued[1].data.output[ROLE_IDS.latent].inputs.mode, "img2img");
});

// ---------------------------------------------------------------------------------------------- idle extensions

const NO_FILES = { ext: { lora_block: { on: true }, vae_2x: { on: true } } };
const noticesOf = (emitted) => emittedOf(emitted, "notice").map((payload) => payload.text);

test("a switched-on extension without its file warns, and the run still starts", async () => {
  const { api, controller, emitted } = setup({ state: NO_FILES });
  const result = await controller.generate();
  assert.equal(result.queued, 1);
  assert.equal(api.queued.length, 1);
  assert.deepEqual(noticesOf(emitted), [`${NOTICES.noExtFile}: LoRA 블록, VAE 2x`]);
});

test("no extension notice when nothing is idle", async () => {
  const { controller, emitted } = setup({ state: { ext: { skim: { on: true } } } });
  await controller.generate();
  assert.deepEqual(noticesOf(emitted), []);
});

test("DeGrid's notice reads the node's file list", async () => {
  const lists = new Map([["ForgeNeoAnimaVAEDeGrid", { required: { model_name: [["auto"], {}] } }]]);
  const { controller, emitted } = setup({ state: { ext: { degrid: { on: true } } }, extFiles: lists });
  await controller.generate();
  assert.deepEqual(noticesOf(emitted), [`${NOTICES.noExtFile}: VAE DeGrid`]);
});

test("an X/Y/Z run gets the extension notice too", async () => {
  const { controller, emitted } = setup({ state: NO_FILES, xyz: XYZ_ON, parameters: golden.infotext.text });
  await controller.generate();
  assert.deepEqual(noticesOf(emitted), [`${NOTICES.noExtFile}: LoRA 블록, VAE 2x`]);
});

test("a gallery Hires run warns about the model stack only, an upscale about nothing", async () => {
  const item = { image: ref("a.png"), record: { state: { ...PICKS, ...NO_FILES }, seed: 1 } };
  const hires = setup({ state: NO_FILES });
  await hires.controller.runPost("hires", item);
  assert.deepEqual(noticesOf(hires.emitted), [`${NOTICES.noExtFile}: LoRA 블록`]);
  const upscale = setup({ state: { ...NO_FILES, upscale: { ...DEFAULT_STATE.upscale, model: "m.pth" } } });
  await upscale.controller.runPost("upscale", item);
  assert.deepEqual(noticesOf(upscale.emitted), []);
});

// ---------------------------------------------------------------------------------------------- model families

const MOSS = { unet: "moss_image_sft_bf16.safetensors", te: "qwen3.5_9b_moss_image_sft_bf16.safetensors", vae: "moss_image_vae_fp32.safetensors" };
const oldGraph = () => "CLIPLoader";

test("MOSS on an older workflow's core CLIPLoader is refused with what to do", async () => {
  const { api, controller, emitted } = setup({ state: MOSS, roleClass: oldGraph });
  await controller.generate();
  assert.equal(api.queued.length, 0);
  assert.deepEqual(noticesOf(emitted), [NOTICES.oldTeNode]);
});

test("the studio's own loader runs MOSS, and an older workflow still runs Anima", async () => {
  const moss = setup({ state: MOSS, roleClass: () => "AIStudioTextEncoderLoader" });
  await moss.controller.generate();
  assert.equal(moss.api.queued.length, 1);
  const anima = setup({ roleClass: oldGraph });
  await anima.controller.generate();
  assert.equal(anima.api.queued.length, 1);
});

test("the warnings of a run share one notice, so none hides another", async () => {
  const { controller, emitted } = setup({
    state: { ...MOSS, prompt: "(detailed:1.2) <lora:typo:1>", ext: { vae_2x: { on: false }, skim: { on: true } } },
    loras: () => ["other.safetensors"],
    roleClass: () => "AIStudioTextEncoderLoader",
  });
  await controller.generate();
  assert.deepEqual(noticesOf(emitted), [`${NOTICES.noLora}: typo · ${NOTICES.noWeights}`]);
});

test("an X/Y/Z run gets the state's warnings too", async () => {
  const { controller, emitted } = setup({ state: { ...MOSS, prompt: "(detailed:1.2)" }, xyz: XYZ_ON, parameters: golden.infotext.text });
  await controller.generate();
  assert.deepEqual(noticesOf(emitted), [NOTICES.noWeights]);
});

test("Same seed and Hires refuse an image another family made, upscaling does not care", async () => {
  const krea = "Steps: 8, Seed: 1, Size: 1024x1024, Model: krea2_turbo_bf16";
  const same = setup({ parameters: krea });
  await same.controller.sameSeed({ image: ref("k.png"), record: null });
  assert.equal(same.api.queued.length, 0);
  assert.deepEqual(noticesOf(same.emitted), [NOTICES.otherFamily]);
  const hires = setup({ parameters: krea });
  await hires.controller.runPost("hires", { image: ref("k.png"), record: null });
  assert.equal(hires.api.queued.length, 0);
  const upscale = setup({ parameters: krea, state: { upscale: { ...DEFAULT_STATE.upscale, model: "x.pth" } } });
  await upscale.controller.runPost("upscale", { image: ref("k.png"), record: null });
  assert.equal(upscale.api.queued.length, 1);
});

// ---------------------------------------------------------------------------------------------- how long a run took

test("the run's end says how long it took, from the click to the last image", async () => {
  let clock = 1000;
  const { api, controller, emitted } = setup({ now: () => clock });
  await controller.generate();
  clock = 4000; // waiting in ComfyUI's queue counts, like Forge's Time taken
  api.fire("execution_start", { prompt_id: "p0" });
  clock = 13500;
  api.fire("executed", { prompt_id: "p0", node: SAVE, display_node: SAVE, output: { images: [ref("a.png")] } });
  api.fire("execution_success", { prompt_id: "p0" });
  await controller.whenIdle();
  assert.deepEqual(emittedOf(emitted, "running"), [{ running: true, kind: "single" }, { running: false, kind: "single", elapsed: 12500 }]);
});

test("an X/Y/Z run's time covers all its cells", async () => {
  let clock = 0;
  const { api, controller, emitted } = setup({ xyz: XYZ_ON, now: () => clock });
  await controller.generate();
  for (const id of ["p0", "p1", "p2", "p3"]) {
    clock += 2000;
    finishCell(api, id, `${id}.png`);
  }
  await controller.whenIdle();
  assert.equal(emittedOf(emitted, "running").at(-1).elapsed, 8000);
});

test("a run that made no image reports no time", async () => {
  let clock = 0;
  const { api, controller, emitted } = setup({ now: () => clock });
  await controller.generate();
  clock = 5000;
  api.fire("execution_error", { prompt_id: "p0", exception_message: "boom" });
  await controller.whenIdle();
  assert.equal(emittedOf(emitted, "running").at(-1).elapsed, null);
});

test("a stopped X/Y/Z run with finished cells still reports its time", async () => {
  let clock = 0;
  const { api, controller, emitted } = setup({ xyz: XYZ_ON, now: () => clock });
  await controller.generate();
  clock = 3000;
  finishCell(api, "p0", "p0.png");
  await controller.stop();
  await controller.whenIdle();
  assert.equal(emittedOf(emitted, "running").at(-1).elapsed, 3000);
});

// ---------------------------------------------------------------------------------------------- 갤러리 자동 비우기

const replaceFlags = (emitted) => emitted.filter(([kind]) => kind === "images" || kind === "grid").map(([kind, payload]) => [kind, payload.replace]);

test("a generation's first images carry the replace flag", async () => {
  const { api, controller, emitted } = setup();
  await controller.generate();
  finishCell(api, "p0", "a.png");
  await controller.whenIdle();
  assert.deepEqual(replaceFlags(emitted), [["images", true]]);
});

test("the next generation carries it again", async () => {
  const { api, controller, emitted } = setup();
  await controller.generate();
  finishCell(api, "p0", "a.png");
  await controller.whenIdle();
  await controller.generate();
  finishCell(api, "p1", "b.png");
  await controller.whenIdle();
  assert.deepEqual(replaceFlags(emitted), [["images", true], ["images", true]]);
});

test("an X/Y/Z run with cell images replaces once, its later cells and the grid join them", async () => {
  const xyz = { ...XYZ_ON, options: { ...XYZ_ON.options, include_images: true } };
  const { api, controller, emitted } = setup({ xyz });
  await controller.generate();
  for (const id of ["p0", "p1", "p2", "p3"]) finishCell(api, id, `${id}.png`);
  await controller.whenIdle();
  assert.deepEqual(replaceFlags(emitted), [["images", true], ["images", false], ["images", false], ["images", false], ["grid", false]]);
});

test("an X/Y/Z run without cell images replaces with its grid", async () => {
  const { api, controller, emitted } = setup({ xyz: XYZ_ON });
  await controller.generate();
  for (const id of ["p0", "p1", "p2", "p3"]) finishCell(api, id, `${id}.png`);
  await controller.whenIdle();
  assert.deepEqual(replaceFlags(emitted), [["grid", true]]);
});

test("같은 시드 is a generation and replaces", async () => {
  const item = { image: ref("a.png"), record: { state: { ...PICKS }, seed: 5 } };
  const { api, controller, emitted } = setup();
  await controller.sameSeed(item);
  finishCell(api, "p0", "again.png");
  await controller.whenIdle();
  assert.deepEqual(replaceFlags(emitted), [["images", true]]);
});

test("a gallery post-processing run keeps the other images", async () => {
  const item = { image: ref("a.png"), record: { state: { ...PICKS }, seed: 5 } };
  const { api, controller, emitted } = setup();
  await controller.runPost("hires", item);
  api.fire("execution_start", { prompt_id: "p0" });
  api.fire("executed", { prompt_id: "p0", node: POST_SAVE, display_node: POST_SAVE, output: { images: [ref("post.png")] } });
  api.fire("execution_success", { prompt_id: "p0" });
  await controller.whenIdle();
  assert.deepEqual(replaceFlags(emitted), [["images", false]]);
});
