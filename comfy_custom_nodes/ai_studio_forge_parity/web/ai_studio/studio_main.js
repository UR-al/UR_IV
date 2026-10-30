// Anima studio web extension: draws the three UI nodes (AIStudioPanel / AIStudioResultView / AIStudioXYZView) as
// fixed-size DOM widgets and wires them to ComfyUI — the run controller, the studio routes (/aistudio/studio/*), the
// role-tagged engine nodes (engine room, i2i image, mask editor), the prompt tag autocomplete and the queue (생성 button, Ctrl+Enter).
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { PanelView } from "./panel_view.js";
import { ResultView } from "./result_view.js";
import { RunController } from "./run_controller.js";
import { refFromValue } from "./studio_dom.js";
import { engineRoomCollapsed, toggleEngineRoom } from "./studio_engine_room.js";
import { readPngText } from "./studio_pnginfo.js";
import { ROLE_PROPERTY } from "./studio_roles.js";
import { XyzView } from "./xyz_view.js";

const ROUTE = "/aistudio/studio";
const CSS_ID = "aistudio-studio-css";
const PANEL = "AIStudioPanel";
const RESULT = "AIStudioResultView";
const XYZ = "AIStudioXYZView";
// Node sizes of the studio workflow (canvas units); the DOM widget fills the node body (R11: pixel heights).
const UI_NODES = {
  [PANEL]: { widget: "state", width: 790, height: 960, View: PanelView },
  [RESULT]: { widget: "view", width: 680, height: 960, View: ResultView },
  [XYZ]: { widget: "xyz", width: 400, height: 960, View: XyzView },
};
const BODY_INSET = 16; // node body height - DOM widget height (widget margins)
const MASK_EDITOR_WAIT_MS = 3000;
const CHOICES_TTL_MS = 60000;

const views = new WeakMap(); // UI node -> its view
const choiceCache = new Map();
const loaded = { families: undefined, upscalers: undefined, loras: undefined };
let controller = null;
let originalGraphToPrompt = null;

const graph = () => app.rootGraph ?? app.graph;
const viewOf = (comfyClass) => {
  const node = (graph()?.nodes || []).find((item) => item.comfyClass === comfyClass && views.has(item));
  return node ? views.get(node) : null;
};

function roleNodes() {
  const byRole = new Map();
  for (const node of graph()?.nodes || []) {
    const role = node.properties?.[ROLE_PROPERTY];
    if (role && !UI_NODES[node.comfyClass]) byRole.set(role, node);
  }
  return byRole;
}

const roleIds = () => Object.fromEntries([...roleNodes()].map(([role, node]) => [role, String(node.id)]));

function emit(type, payload) {
  for (const comfyClass of Object.keys(UI_NODES)) viewOf(comfyClass)?.onStudioEvent?.(type, payload);
  if (type === "notice" && !viewOf(RESULT)) app.extensionManager?.toast?.add({ severity: "info", summary: "AI Studio", detail: payload.text, life: 6000 });
}

const notice = (text) => emit("notice", { text });

async function route(path, body) {
  const init = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const response = await api.fetchApi(ROUTE + path, init);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `${response.status} ${response.statusText}`);
  return data;
}

/** `/choices?kind=` JSON, cached for CHOICES_TTL_MS (new LoRA epochs show up without a page reload); `fresh` skips the cache. */
function choices(kind, fresh = false) {
  const cached = choiceCache.get(kind);
  if (fresh || !cached || Date.now() - cached.at > CHOICES_TTL_MS) {
    const request = route(`/choices?kind=${encodeURIComponent(kind)}`)
      .then((data) => {
        if (kind === "unets") loaded.families = data.families || {};
        if (kind === "upscalers") loaded.upscalers = data.items || [];
        if (kind === "loras") loaded.loras = data.items || [];
        return data;
      })
      .catch((error) => {
        choiceCache.delete(kind);
        console.warn(`[AI Studio] choices ${kind}:`, error);
        return { items: [] };
      });
    choiceCache.set(kind, { at: Date.now(), request });
  }
  return choiceCache.get(kind).request;
}

/** A node class's inputs from `/object_info/<class>` (its COMBO file lists); throws on an HTTP error. */
async function nodeInputs(comfyClass) {
  const response = await api.fetchApi(`/object_info/${encodeURIComponent(comfyClass)}`);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return (await response.json())[comfyClass]?.input ?? {};
}

function imageUrl(ref, thumb = false) {
  const query = new URLSearchParams({ filename: ref.filename, subfolder: ref.subfolder || "", type: ref.type || "output" });
  if (thumb) query.set("preview", "webp;50");
  return api.apiURL(`/view?${query}`);
}

async function readParameters(ref) {
  const query = new URLSearchParams({ filename: ref.filename, subfolder: ref.subfolder || "", type: ref.type || "output" });
  const response = await api.fetchApi(`/view?${query}`);
  if (!response.ok) return null;
  return (await readPngText(await response.arrayBuffer())).parameters || null;
}

/** graphToPrompt with the role nodes set to `modes` (R7 bypass) for this one call; the graph keeps its own modes. */
async function basePrompt(modes) {
  const nodes = roleNodes();
  const saved = [];
  for (const [role, mode] of Object.entries(modes)) {
    const node = nodes.get(role);
    if (!node) continue;
    saved.push([node, node.mode]);
    node.mode = mode;
  }
  try {
    return await originalGraphToPrompt.call(app, graph());
  } finally {
    for (const [node, mode] of saved) node.mode = mode;
  }
}

// ---------------------------------------------------------------------------------------------- the i2i image

function i2iWidget() {
  return roleNodes().get("i2i_image")?.widgets?.find((widget) => widget.name === "image") || null;
}

const i2i = {
  get: () => String(i2iWidget()?.value || ""),
  set(value) {
    const widget = i2iWidget();
    if (!widget) return;
    widget.value = value;
    widget.callback?.(value); // LoadImage loads its preview (node.imgs) from the new value
    graph()?.setDirtyCanvas(true, false);
  },
  url: (value) => imageUrl(refFromValue(value)),
  async upload(file) {
    const body = new FormData();
    body.append("image", file, file.name);
    const response = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!response.ok) {
      notice(`업로드 실패: ${response.status}`);
      return "";
    }
    const data = await response.json();
    const value = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
    i2i.set(value);
    return { name: value, ...(await fileImageSize(file)) };
  },
  maskEditor: () => openMaskEditor(),
};

/** `{width, height}` of a picked image file (Forge's send-to size), or `{}` when the browser cannot decode it. */
async function fileImageSize(file) {
  try {
    const bitmap = await createImageBitmap(file);
    const size = { width: bitmap.width, height: bitmap.height };
    bitmap.close?.();
    return size;
  } catch {
    return {};
  }
}

async function openMaskEditor() {
  const node = roleNodes().get("i2i_image");
  if (!node) return notice("i2i 이미지 노드가 없음");
  const until = Date.now() + MASK_EDITOR_WAIT_MS;
  while (!node.imgs?.length && Date.now() < until) await new Promise((done) => setTimeout(done, 100));
  if (!node.imgs?.length) return notice("입력 이미지를 먼저 고르기");
  app.canvas.selectNode(node); // the mask editor opens for the single selected node
  return app.extensionManager.command.execute("Comfy.MaskEditor.OpenMaskEditor");
}

// ---------------------------------------------------------------------------------------------- engine room

const engineRoom = {
  nodes: () => (graph()?.nodes || []).filter((node) => node.properties?.[ROLE_PROPERTY] && !UI_NODES[node.comfyClass]),
  collapsed: () => engineRoomCollapsed(engineRoom.nodes()),
  toggle() {
    const nodes = engineRoom.nodes();
    if (!nodes.length) return;
    toggleEngineRoom(graph().groups || [], nodes, app.canvas?.ctx);
    graph().setDirtyCanvas(true, true);
  },
};

// ---------------------------------------------------------------------------------------------- gallery buttons

async function galleryAction(name, item) {
  try {
    if (name === "folder") {
      await route("/open_output", {});
      return;
    }
    if (!item) return notice("고른 이미지가 없음");
    if (name === "i2i" || name === "inpaint") {
      // a copy in input/ and its size, Forge's "send to" (the i2i run uses the image's size)
      const { name: value, width, height } = await route("/to_input", item.image);
      i2i.set(value);
      viewOf(PANEL)?.showI2i(value, name === "inpaint" ? "inpaint" : "img2img", { width, height });
      if (name === "inpaint") await openMaskEditor();
      return;
    }
    if (name === "seed") {
      // spec §3.2: runs right away on the picked image (its settings and seed, the current post-processing settings)
      const result = await controller.sameSeed(item);
      if (result?.queued) viewOf(PANEL)?.setSeed(result.seed); // the panel keeps that seed too, like Forge's ♻
      return;
    }
    await controller.runPost(name, item);
  } catch (error) {
    notice(String(error?.message || error));
  }
}

const generate = () => controller.generate().catch((error) => notice(String(error?.message || error)));

function contextFor(comfyClass) {
  if (comfyClass === PANEL) {
    return {
      choices,
      tags: (query, limit) => route(`/tags?q=${encodeURIComponent(query)}&limit=${limit}`).then((data) => data.items || []),
      fetchChoices: nodeInputs,
      generate,
      stop: () => controller.stop(),
      changed: () => emit("state", null),
      engineRoom,
      i2i,
      hasRole: (role) => roleNodes().has(role),
    };
  }
  if (comfyClass === RESULT) return { imageUrl, action: galleryAction };
  return { choices, plan: (body) => route("/xyz/plan", body), getState: () => viewOf(PANEL)?.getState() || {} };
}

// ---------------------------------------------------------------------------------------------- ComfyUI hooks

function addStylesheet() {
  if (document.getElementById(CSS_ID)) return;
  const link = document.createElement("link");
  link.id = CSS_ID;
  link.rel = "stylesheet";
  link.href = new URL("./studio.css", import.meta.url).href;
  document.head.append(link);
}

function chain(node, method, after) {
  const before = node[method];
  node[method] = function (...args) {
    const result = before?.apply(this, args);
    after();
    return result;
  };
}

function fixSize(node, spec) {
  node.setSize([spec.width, spec.height]);
  node.resizable = false;
}

/** R11: no pre-queue hook in frontend 1.57.4 — Ctrl+Enter / Run reach app.queuePrompt and app.graphToPrompt. */
function wrapQueue() {
  originalGraphToPrompt = app.graphToPrompt;
  app.graphToPrompt = async function (target, ...rest) {
    if (viewOf(PANEL) && (target === undefined || target === graph())) {
      try {
        return await controller.panelPrompt(); // the panel state written into the engine nodes
      } catch (error) {
        console.warn("[AI Studio] panel state not applied:", error);
      }
    }
    return originalGraphToPrompt.call(this, target, ...rest);
  };
  const originalQueuePrompt = app.queuePrompt;
  app.queuePrompt = async function (number, batchCount, options) {
    const partial = Array.isArray(options) ? options.length > 0 : Boolean(options?.queueNodeIds?.length);
    if (partial || !viewOf(PANEL)) return originalQueuePrompt.call(this, number, batchCount, options);
    await generate(); // tracked like the 생성 button; X/Y/Z when its panel is on (R17)
    return true;
  };
}

app.registerExtension({
  name: "AIStudio.Studio",

  // init: runs before the node types are registered, so the controller exists before any UI node is created.
  init() {
    addStylesheet();
    controller = new RunController({
      api,
      basePrompt,
      roleIds,
      getState: () => {
        const panel = viewOf(PANEL);
        if (!panel) throw new Error("제어판(AI Studio Panel) 노드가 없음");
        return panel.getState();
      },
      getXyz: () => viewOf(XYZ)?.getXyz() || { on: false },
      route,
      readParameters,
      families: () => loaded.families,
      upscalers: () => loaded.upscalers,
      loras: () => loaded.loras,
      reloadLoras: () => choices("loras", true),
      extFiles: () => viewOf(PANEL)?.extMemory.lists,
      emit,
    });
    wrapQueue();
  },

  afterConfigureGraph() {
    viewOf(PANEL)?.render(); // the engine-room button and the i2i image depend on nodes configured after the panel
  },

  nodeCreated(node) {
    const spec = UI_NODES[node.comfyClass];
    if (!spec) return;
    const widget = node.widgets?.find((item) => item.name === spec.widget) || null;
    const view = new spec.View(widget, contextFor(node.comfyClass));
    const height = spec.height - BODY_INSET;
    const dom = node.addDOMWidget(`aistudio_${spec.widget}`, "aistudio_view", view.element, {
      hideOnZoom: false,
      serialize: false, // not an API prompt input
      getValue: () => "",
      setValue: () => {},
      getMinHeight: () => height,
      getMaxHeight: () => height,
    });
    dom.serialize = false; // not in widgets_values: the hidden STRING widget holds the saved state
    views.set(node, view);
    fixSize(node, spec);
    chain(node, "onConfigure", () => {
      view.load(); // widgets_values were just restored into the hidden widget
      fixSize(node, spec);
    });
    chain(node, "onRemoved", () => {
      views.delete(node);
      view.dispose?.();
    });
  },
});
