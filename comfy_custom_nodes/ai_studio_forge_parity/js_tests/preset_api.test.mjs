// node:test for the prompt-box presets' route wrapper and the shared store (web/ai_studio/preset_api.js); the server is a
// small in-memory fake of preset_routes.py (preset_fakes.mjs). Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { createPresetStore, presetApi } from "../web/ai_studio/preset_api.js";
import { PRESET_LIMITS } from "../web/ai_studio/preset_logic.js";
import { fakeServer, hold, tick } from "./preset_fakes.mjs";

test("presetApi maps each call to GET /presets or one POST /presets change", async () => {
  const calls = [];
  const api = presetApi((path, body) => { calls.push([path, body]); return Promise.resolve({ ok: true }); });
  await api.load();
  await api.add("negative", "N", "t");
  await api.update("negative", "a", { name: "N2" });
  await api.update("negative", "a", { text: "t2", name: undefined });
  await api.remove("negative", "a");
  await api.move("negative", "a", -1);
  await api.touch("negative", "a");
  await api.insert("negative", { id: "a" }, 2);
  await api.savePrefs({ close_after_insert: true });
  assert.deepEqual(calls, [
    ["/presets", undefined],
    ["/presets", { op: "add", section: "negative", name: "N", text: "t" }],
    ["/presets", { op: "update", section: "negative", id: "a", name: "N2" }],
    ["/presets", { op: "update", section: "negative", id: "a", text: "t2" }],
    ["/presets", { op: "delete", section: "negative", id: "a" }],
    ["/presets", { op: "move", section: "negative", id: "a", delta: -1 }],
    ["/presets", { op: "use", section: "negative", id: "a" }],
    ["/presets", { op: "insert", section: "negative", item: { id: "a" }, index: 2 }],
    ["/presets", { op: "prefs", prefs: { close_after_insert: true } }],
  ]);
});

test("a new store is empty and not loaded; load() reads once and later calls reuse it", async () => {
  const server = fakeServer();
  server.doc.sections.negative.push({ id: "a", name: "A", text: "x", created: 1, used: 0 });
  const store = createPresetStore(presetApi(server.route));
  assert.equal(store.loaded, false);
  assert.deepEqual(store.list("negative"), []);
  assert.deepEqual(store.prefs, { close_after_insert: false });
  assert.deepEqual({ ...store.limits }, { ...PRESET_LIMITS });
  const [one, two] = await Promise.all([store.load(), store.load()]);
  assert.equal(one, two);
  assert.equal(server.calls.length, 1, "two callers share one request");
  assert.equal(store.loaded, true);
  assert.deepEqual(store.list("negative").map((p) => p.id), ["a"]);
  await store.load();
  assert.equal(server.calls.length, 1, "cached");
  await store.load(true);
  assert.equal(server.calls.length, 2, "forced");
});

test("every change answers with the new document and tells the subscribers once", async () => {
  const server = fakeServer();
  const store = createPresetStore(presetApi(server.route));
  let told = 0;
  const off = store.subscribe(() => { told += 1; });
  const added = await store.add("negative", "A", "x");
  assert.deepEqual(added, { id: "id1", name: "A", text: "x", created: 1001, used: 0 });
  assert.equal(store.loaded, true, "a change's answer is a full document");
  assert.equal(told, 1);
  await store.add("negative", "B", "y");
  await store.update("negative", "id1", { name: "A2" });
  await store.move("negative", "id2", -1);
  assert.deepEqual(store.list("negative").map((p) => p.name), ["B", "A2"]);
  await store.touch("negative", "id2");
  assert.equal(store.list("negative")[0].used, 4242);
  const gone = store.list("negative")[1];
  await store.remove("negative", "id1");
  assert.deepEqual(store.list("negative").map((p) => p.id), ["id2"]);
  await store.insert("negative", gone, 1);
  assert.deepEqual(store.list("negative"), [store.list("negative")[0], gone]);
  await store.setPrefs({ close_after_insert: true });
  assert.deepEqual(store.prefs, { close_after_insert: true });
  assert.equal(told, 8);
  off();
  await store.add("prompt", "P", "z");
  assert.equal(told, 8, "unsubscribed");
  assert.equal(store.match("negative", " y, ").id, "id2");
  assert.equal(store.match("negative", "nothing"), null);
});

test("changes reach the server in the order they were made, one at a time", async () => {
  const server = fakeServer();
  const store = createPresetStore(presetApi(server.route));
  const open = hold(server);
  const first = store.add("negative", "A", "x");
  const second = store.add("negative", "B", "y");
  await tick();
  assert.equal(server.calls.length, 1, "the second waits for the first");
  open();
  await Promise.all([first, second]);
  assert.deepEqual(store.list("negative").map((p) => p.name), ["A", "B"]);
});

test("a failed change rejects, keeps the document and does not block the next one", async () => {
  const server = fakeServer();
  const store = createPresetStore(presetApi(server.route));
  await store.add("negative", "A", "x");
  server.fail = "이름을 적어 주세요";
  await assert.rejects(store.add("negative", "", "y"), /이름을 적어 주세요/);
  assert.deepEqual(store.list("negative").map((p) => p.name), ["A"]);
  server.fail = null;
  await store.add("negative", "B", "y");
  assert.deepEqual(store.list("negative").map((p) => p.name), ["A", "B"]);
});

test("a failed load rejects, is remembered as the error and can be tried again", async () => {
  const server = fakeServer();
  const store = createPresetStore(presetApi(server.route));
  let told = 0;
  store.subscribe(() => { told += 1; });
  server.fail = "500 Internal Server Error";
  await assert.rejects(store.load(), /500/);
  assert.equal(store.loaded, false);
  assert.match(store.error, /500/);
  assert.equal(told, 1);
  server.fail = null;
  await store.load();
  assert.equal(store.loaded, true);
  assert.equal(store.error, null);
});

test("a subscriber that throws does not stop the others", async () => {
  const server = fakeServer();
  const store = createPresetStore(presetApi(server.route));
  let told = 0;
  store.subscribe(() => { throw new Error("boom"); });
  store.subscribe(() => { told += 1; });
  const warn = console.warn;
  console.warn = () => {};
  try {
    await store.load();
  } finally {
    console.warn = warn;
  }
  assert.equal(told, 1);
});

test("the last box change can be taken back while the box still holds what was put in", () => {
  const store = createPresetStore(presetApi(() => Promise.resolve({})));
  assert.equal(store.boxUndo("negative", "new"), null);
  store.noteBoxChange("negative", "old text", "new");
  assert.equal(store.boxUndo("negative", "new"), "old text");
  assert.equal(store.boxUndo("negative", "new, typed more"), null, "the box moved on: nothing to take back");
  assert.equal(store.boxUndo("prompt", "new"), null, "another box");
  store.noteBoxChange("prompt", "", "x");
  assert.equal(store.boxUndo("prompt", "x"), "", "an empty box is a state to go back to");
  assert.equal(store.boxUndo("negative", "new"), null, "only the last change is kept");
  store.clearBoxUndo();
  assert.equal(store.boxUndo("prompt", "x"), null);
});

test("no change replaces a whole list: the api and the store have no 'restore'", () => {
  const server = fakeServer();
  const api = presetApi(server.route);
  const store = createPresetStore(api);
  assert.equal(api.restore, undefined);
  assert.equal(store.restore, undefined);
});

test("the store says when the file could not be parsed, until a change wrote it anew", async () => {
  const server = fakeServer();
  const store = createPresetStore(presetApi(server.route));
  assert.equal(store.broken, false);
  server.broken = true;
  await store.load();
  assert.equal(store.broken, true);
  await store.add("negative", "A", "x"); // the server kept the old file as .bak and wrote a whole one
  assert.equal(store.broken, false);
});
