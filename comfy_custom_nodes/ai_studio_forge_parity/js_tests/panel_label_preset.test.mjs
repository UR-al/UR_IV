// node:test for the seam between the control panel's prompt card (web/ai_studio/panel_view.js) and the preset quick list
// (web/ai_studio/preset_quick.js): a box's name opens the list through `ctx.boxLabel(key, anchorEl)`, and the list keeps
// that name as its anchor while the panel redraws under it (a picked preset, 되돌리기, choices arriving).
// Uses the preset tests' fake document and fake preset routes (preset_fakes.mjs). Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { PanelView } from "../web/ai_studio/panel_view.js";
import { createPresetStore, presetApi } from "../web/ai_studio/preset_api.js";
import { openPresetQuick } from "../web/ai_studio/preset_quick.js";
import { fakeServer, installFakeDom, settle } from "./preset_fakes.mjs";

const PRESET = "worst quality, low quality";
const PLACE = { left: 40, top: 300, right: 100, bottom: 316, width: 60, height: 16 };

let last = null; // the list a test left open: the module shows one at a time

/** A panel in the page whose names open the quick list, wired the way the studio does it. */
async function setup() {
  last?.close();
  const doc = installFakeDom();
  const server = fakeServer();
  server.seed("negative", "Anima 기본", PRESET);
  const store = createPresetStore(presetApi(server.route));
  const env = { doc };
  const ctx = {
    choices: async () => ({ items: [] }),
    engineRoom: { collapsed: () => true, toggle() {} },
    i2i: { get: () => "", set() {}, url: () => "", upload: async () => ({}), maskEditor() {} },
    hasRole: () => true,
    fetchChoices: async () => ({}),
    boxLabel: (key, anchorEl) => {
      last = openPresetQuick(anchorEl, key, {
        store,
        getText: (box) => env.view.state[box],
        setText: (box, text) => env.view.set((state) => { state[box] = text; }, true),
      }) || last;
    },
  };
  env.view = new PanelView({ value: JSON.stringify({ negative: "my negative" }) }, ctx);
  doc.body.append(env.view.element);
  await settle();
  env.label = () => env.view.element.querySelector('[data-section="negative"]');
  env.popup = () => doc.body.querySelector(".ais-pq");
  env.place = () => (env.popup() ? `${env.popup().style.left},${env.popup().style.top}` : "closed");
  env.label().rect = PLACE; // where the browser would lay the name out
  return env;
}

test("picking a preset from the list under a name: the box changes and the focus goes back to that name", async () => {
  const env = await setup();
  const name = env.label();
  name.focus();
  name.click();
  await settle();
  assert.ok(env.popup().contains(env.doc.activeElement), "the list takes the focus");
  assert.equal(name.getAttribute("aria-expanded"), "true");
  const pick = env.popup().querySelector(".ais-pq-pick");
  pick.focus();
  pick.click(); // setText redraws the panel while the focus is inside the list
  await settle();
  assert.equal(env.view.state.negative, PRESET);
  assert.equal(env.popup(), null);
  assert.equal(env.label(), name);
  assert.equal(env.doc.activeElement, name, "the keyboard is back on the name, not on the page");
  assert.equal(name.getAttribute("aria-expanded"), "false");
});

test("되돌리기 redraws the panel under the open list: the list stays under its name", async () => {
  const env = await setup();
  env.label().click();
  await settle();
  assert.equal(env.place(), "40px,320px");
  env.popup().querySelector(".ais-pq-pick").click(); // replace the box: the list closes
  await settle();
  env.label().click(); // open again: the note offers 되돌리기
  await settle();
  assert.equal(env.place(), "40px,320px");
  env.popup().querySelector(".ais-pq-undo").click();
  await settle();
  assert.equal(env.view.state.negative, "my negative");
  assert.equal(env.place(), "40px,320px"); // not the window's corner
  assert.equal(env.label().getAttribute("aria-expanded"), "true");
});

test("after a redraw under the open list the name is still its toggle", async () => {
  const env = await setup();
  env.label().click();
  await settle();
  env.view.render(); // choices or LoRA names arriving, any switch
  assert.ok(env.popup());
  assert.equal(env.label().getAttribute("aria-expanded"), "true");
  env.label().fire("pointerdown"); // a press on the name is not a press outside the list
  assert.ok(env.popup(), "the press alone does not close the list");
  env.label().click(); // the click closes it: the name is a toggle
  await settle();
  assert.equal(env.popup(), null);
  assert.equal(env.label().getAttribute("aria-expanded"), "false");
});
