// node:test for the panel's LoRA rows <-> picker window helpers (web/ai_studio/lora_rows.js) and the window's route api
// (web/ai_studio/lora_api.js). Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { coverVersions, loraApi, thumbPath, withQuery } from "../web/ai_studio/lora_api.js";
import { addSession, listedStem, pickedRow, replaceRow, rowFace } from "../web/ai_studio/lora_rows.js";
import { withDefaults } from "../web/ai_studio/studio_logic.js";

const BJUN = "LoKr\\묵밥 (@Bjunbae)\\@bjunbae_2.9b_lokr_came_b10_2e06_fp16\\@bjunbae_2.9b_lokr_came_b10_2e06_fp16-000014.safetensors";
const BJUN_STEM = "@bjunbae_2.9b_lokr_came_b10_2e06_fp16-000014";

// ----------------------------------------------------------------------------------------------- rows

test("onPick then onUnpick restores the rows", () => {
  const loras = [{ on: false, name: "kept", strength: 0.5 }];
  const before = structuredClone(loras);
  const session = addSession();
  const first = session.onPick(loras, { name: "b", strength: 0.8, label: "b" });
  const second = session.onPick(loras, { name: "c", strength: 1.2, label: "c" });
  assert.deepEqual([first, second], [1, 2]);
  assert.deepEqual(loras.slice(1), [{ on: true, name: "b", strength: 0.8 }, { on: true, name: "c", strength: 1.2 }]);
  assert.equal(session.onUnpick(loras, first), true);
  assert.deepEqual(loras.map((row) => row.name), ["kept", "c"]);
  assert.equal(session.onUnpick(loras, second - 1), true); // the window moves later rows up one, as the panel did
  assert.deepEqual(loras, before);
});

test("onUnpick ignores a row the user already changed", () => {
  const loras = [];
  const session = addSession();
  const index = session.onPick(loras, { name: "b", strength: 1 });
  loras[index].name = "other";
  assert.equal(session.onUnpick(loras, index), false);
  assert.deepEqual(loras, [{ on: true, name: "other", strength: 1 }]);
  const gone = session.onPick(loras, { name: "c", strength: 1 });
  loras.splice(gone, 1); // the row was taken out in the panel
  assert.equal(session.onUnpick(loras, gone), false);
  assert.equal(session.onUnpick(loras, 0), false); // never handed out by this window
  assert.deepEqual(loras, [{ on: true, name: "other", strength: 1 }]);
});

test("onUnpick of the same LoRA added twice takes the rows out one by one", () => {
  const loras = [];
  const session = addSession();
  const a = session.onPick(loras, { name: "b", strength: 1 });
  const b = session.onPick(loras, { name: "b", strength: 0.5 });
  assert.equal(session.onUnpick(loras, a), true);
  assert.deepEqual(loras, [{ on: true, name: "b", strength: 0.5 }]);
  assert.equal(session.onUnpick(loras, b - 1), true);
  assert.deepEqual(loras, []);
});

test("picked rows keep the saved form {on, name, strength} through the state round trip", () => {
  const row = pickedRow({ name: BJUN_STEM, strength: 0.85, label: "묵밥 · @bjunbae fp16 · e14" });
  assert.deepEqual(row, { on: true, name: BJUN_STEM, strength: 0.85 });
  assert.deepEqual(Object.keys(row), ["on", "name", "strength"]);
  assert.deepEqual(withDefaults({ loras: [row] }).loras, [row]);
  assert.deepEqual(JSON.parse(JSON.stringify(withDefaults({ loras: [row] }))).loras, [row]);
  assert.equal(pickedRow({ name: "x", strength: null }).strength, 1.0);
});

test("replaceRow: the picked LoRA and strength; the switch stays", () => {
  const loras = [{ on: false, name: "old", strength: 0.7 }];
  assert.equal(replaceRow(loras, 0, { name: "new", strength: 1.1 }), true);
  assert.deepEqual(loras[0], { on: false, name: "new", strength: 1.1 });
  assert.equal(replaceRow(loras, 5, { name: "x", strength: 1 }), false);
  assert.equal(loras.length, 1);
});

test("replaceRow: slider untouched → the row's own strength is kept (the window started it clamped to 0~2, two decimals)", () => {
  const loras = [{ on: true, name: "neg", strength: -1 }, { on: true, name: "big", strength: 3 }, { on: true, name: "fine", strength: 0.833 }];
  replaceRow(loras, 0, { name: "a", strength: 0 });
  replaceRow(loras, 1, { name: "b", strength: 2 });
  replaceRow(loras, 2, { name: "c", strength: 0.83 });
  assert.deepEqual(loras, [{ on: true, name: "a", strength: -1 }, { on: true, name: "b", strength: 3 }, { on: true, name: "c", strength: 0.833 }]);
});

test("replaceRow: slider moved → the picked strength is applied, also over a strength the window could not show", () => {
  const loras = [{ on: true, name: "neg", strength: -1 }, { on: true, name: "big", strength: 3 }];
  replaceRow(loras, 0, { name: "a", strength: 1.2 });
  replaceRow(loras, 1, { name: "b", strength: 1.95 });
  assert.deepEqual(loras, [{ on: true, name: "a", strength: 1.2 }, { on: true, name: "b", strength: 1.95 }]);
});

// ----------------------------------------------------------------------------------------------- the row button's face

test("rowFace: the short name, the full path on hover; a name missing from the loaded list is marked", () => {
  const list = ["other.safetensors", BJUN];
  assert.deepEqual(rowFace(BJUN_STEM, list), { text: "묵밥 · @bjunbae fp16 · e14", title: BJUN, missing: false });
  const gone = rowFace("@gone_2.9b_lokr_fp16-000003", list);
  assert.equal(gone.missing, true);
  assert.equal(gone.text, "@gone fp16 · e3 (없음)");
  assert.match(gone.title, /^@gone_2\.9b_lokr_fp16-000003 /);
  const loading = rowFace(BJUN_STEM, []); // the list is still loading (or failed): no mark
  assert.equal(loading.missing, false);
  assert.equal(loading.text, "@bjunbae fp16 · e14");
  assert.equal(rowFace("", list).text, "[LoRA]");
  assert.equal(rowFace("", list).missing, false);
});

test("rowFace: a long name is shortened in the middle to 40 characters, the epoch stays", () => {
  const entry = "LoKr\\아주 긴 그룹 이름 (@someone)\\@a_very_long_run_name_for_testing_2.9b_lokr_came_b10_2e06_fp16\\@a_very_long_run_name_for_testing_2.9b_lokr_came_b10_2e06_fp16-000012.safetensors";
  const face = rowFace("@a_very_long_run_name_for_testing_2.9b_lokr_came_b10_2e06_fp16-000012", [entry]);
  assert.equal(Array.from(face.text).length, 40);
  assert.match(face.text, /…/);
  assert.match(face.text, / · e12$/);
});

test("listedStem: by file stem, either slash", () => {
  assert.equal(listedStem([BJUN], BJUN_STEM), true);
  assert.equal(listedStem(["a/b/c.safetensors"], "c"), true);
  assert.equal(listedStem(["a/b/c.safetensors"], "b"), false);
  assert.equal(listedStem(undefined, "c"), false);
});

// ----------------------------------------------------------------------------------------------- the window's api

test("thumbPath leaves out an empty epoch / index and encodes the name", () => {
  assert.equal(thumbPath(BJUN, "cover"), `/loras/thumb?name=${encodeURIComponent(BJUN)}&kind=cover`);
  assert.equal(thumbPath("a b", "cover", "", null), "/loras/thumb?name=a%20b&kind=cover");
  assert.equal(thumbPath("a", "sample", 14, 2), "/loras/thumb?name=a&kind=sample&epoch=14&i=2");
  assert.equal(thumbPath("a", "sample", 0, 0), "/loras/thumb?name=a&kind=sample&epoch=0&i=0");
  assert.equal(thumbPath("a", "sample", Number.NaN, undefined), "/loras/thumb?name=a&kind=sample");
  const odd = "LoRA\\ໂ (@9an9ta)\\x+y&z=1 #2.safetensors";
  const query = new URLSearchParams(thumbPath(odd, "cover").split("?")[1]);
  assert.equal(query.get("name"), odd);
  assert.deepEqual([...query.keys()], ["name", "kind"]);
});

test("loraApi: index / detail / thumb / prefs / preview over the studio route helper", async () => {
  const calls = [];
  const route = async (path, body) => { calls.push(body === undefined ? [path] : [path, body]); return { ok: true }; };
  const api = loraApi(route, (path) => `http://host/api/aistudio/studio${path}`);
  await api.index(false);
  await api.index(true);
  await api.detail(BJUN);
  await api.savePrefs({ favorites: ["r"], recent: [] });
  await api.clearPreview("LoKr/묵밥 (@Bjunbae)/run");
  assert.deepEqual(calls, [
    ["/loras/index"],
    ["/loras/index?fresh=1"],
    [`/loras/detail?name=${encodeURIComponent(BJUN)}`],
    ["/loras/prefs", { favorites: ["r"], recent: [] }],
    ["/loras/preview", { run_key: "LoKr/묵밥 (@Bjunbae)/run", clear: true }],
  ]);
  assert.equal(api.thumbUrl("a", "sample", 3), "http://host/api/aistudio/studio/loras/thumb?name=a&kind=sample&epoch=3");
});

test("a run's custom preview version goes on its cover URL, so a new / cleared preview is not served from the browser cache", () => {
  const index = {
    items: [{ name: BJUN, run_key: "LoKr/묵밥/run" }, { name: "x.safetensors", run_key: "LoRA//x" }],
    cover_versions: { "LoKr/묵밥/run": 1731300000123 },
  };
  const versions = coverVersions(index);
  assert.equal(versions.get(BJUN), 1731300000123);
  assert.equal(versions.has("x.safetensors"), false);
  assert.equal(coverVersions({}).size, 0); // an older route without the field
  assert.equal(withQuery("/t?name=a&kind=cover", { pv: versions.get(BJUN), v: null }), "/t?name=a&kind=cover&pv=1731300000123");
  assert.equal(withQuery("/t?name=a&kind=cover", { pv: undefined, v: 2 }), "/t?name=a&kind=cover&v=2");
  assert.equal(withQuery("/t", { pv: 5, v: 2 }), "/t?pv=5&v=2");
  assert.equal(withQuery("/t?a=1", {}), "/t?a=1");
  assert.equal(withQuery("", { v: 1 }), ""); // no URL stays no URL
});
