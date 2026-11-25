// node:test for the engine-room fold / unfold (web/ai_studio/studio_engine_room.js) with LiteGraph-like fakes.
// Run: cd frontend && npm run test:pack
import assert from "node:assert/strict";
import { test } from "node:test";

import { engineRoomCollapsed, toggleEngineRoom } from "../web/ai_studio/studio_engine_room.js";

const TITLE = 30; // LiteGraph.NODE_TITLE_HEIGHT
const FOLDED_WIDTH = 80;

/** Like LGraphNode: `boundingRect` is a cache the canvas refreshes once per frame (`updateArea`), not on collapse(). */
class FakeNode {
  constructor(title, pos, size, collapsed) {
    Object.assign(this, { title, pos, size, flags: { collapsed } });
    this.rect = [0, 0, 0, 0];
    this.updateArea();
  }
  get boundingRect() {
    return this.rect;
  }
  updateArea() {
    const folded = this.flags.collapsed;
    this.rect = [this.pos[0], this.pos[1] - TITLE, folded ? FOLDED_WIDTH : this.size[0], folded ? TITLE : this.size[1] + TITLE];
  }
  collapse() {
    this.flags.collapsed = !this.flags.collapsed; // LGraphNode.collapse(): the flag only, the rect stays until the next frame
  }
}

/** Like LGraphGroup: members are the nodes whose rect centre is inside; resizeTo fits the members' rects + padding. */
class FakeGroup {
  constructor(title, bounding, all) {
    Object.assign(this, { title, bounding, all, nodes: [] });
  }
  recomputeInsideNodes() {
    const [x, y, w, h] = this.bounding;
    this.nodes = this.all.filter(({ boundingRect: [nx, ny, nw, nh] }) => {
      const cx = nx + nw / 2;
      const cy = ny + nh / 2;
      return cx > x && cx < x + w && cy > y && cy < y + h;
    });
  }
  resizeTo(members, padding = 10) {
    const rects = members.map((node) => node.boundingRect);
    const x0 = Math.min(...rects.map((r) => r[0])) - padding;
    const y0 = Math.min(...rects.map((r) => r[1])) - padding;
    const x1 = Math.max(...rects.map((r) => r[0] + r[2])) + padding;
    const y1 = Math.max(...rects.map((r) => r[1] + r[3])) + padding;
    this.bounding = [x0, y0 - TITLE, x1 - x0, y1 - y0 + TITLE];
  }
}

const inside = (group, node) => {
  const [gx, gy, gw, gh] = group.bounding;
  const [x, y, w, h] = node.boundingRect;
  return x >= gx && y >= gy && x + w <= gx + gw && y + h <= gy + gh;
};

/** Two rows as the saved workflow has them: nodes folded, each group sized for its unfolded nodes. */
function studio() {
  const all = [];
  const unet = new FakeNode("UNet", [10, 1080], [340, 112], true);
  const te = new FakeNode("TE", [400, 1080], [340, 112], true);
  const sampler = new FakeNode("샘플러", [10, 1300], [360, 500], true);
  const panel = new FakeNode("Panel", [0, 0], [790, 960], false);
  all.push(unet, te, sampler, panel);
  const groups = [new FakeGroup("① 모델", [0, 1040, 760, 200], all), new FakeGroup("④ 샘플링", [0, 1260, 400, 560], all)];
  return { groups, engine: [unet, te, sampler], panel };
}

test("unfolding fits every group around its unfolded nodes", () => {
  const { groups, engine } = studio();
  assert.equal(engineRoomCollapsed(engine), true);
  assert.equal(toggleEngineRoom(groups, engine), false);
  assert.equal(engineRoomCollapsed(engine), false);
  for (const node of engine) node.updateArea(); // the next frame
  const [model, sampling] = groups;
  for (const [group, nodes] of [[model, engine.slice(0, 2)], [sampling, engine.slice(2)]]) {
    for (const node of nodes) assert.ok(inside(group, node), `${node.title} inside ${group.title}: ${group.bounding}`);
    group.recomputeInsideNodes();
    assert.deepEqual(group.nodes.map((node) => node.title), nodes.map((node) => node.title));
  }
});

test("folding shrinks the groups to the title rows, and unfolding again finds the same members", () => {
  const { groups, engine } = studio();
  toggleEngineRoom(groups, engine); // open
  assert.equal(toggleEngineRoom(groups, engine), true); // fold
  for (const node of engine) node.updateArea();
  const [model] = groups;
  assert.equal(model.bounding[3], TITLE + 2 * 10 + TITLE);
  for (const node of engine.slice(0, 2)) assert.ok(inside(model, node));
  toggleEngineRoom(groups, engine); // open again
  for (const node of engine) node.updateArea();
  for (const node of engine.slice(0, 2)) assert.ok(inside(model, node), `${node.title} inside after the second open`);
});

test("a mixed state folds everything; groups without engine nodes and the UI nodes are left alone", () => {
  const { groups, engine, panel } = studio();
  engine[0].collapse();
  engine[0].updateArea();
  const other = new FakeGroup("UI", [-10, -40, 810, 1010], [panel]);
  const before = [...other.bounding];
  assert.equal(toggleEngineRoom([...groups, other], engine), true);
  assert.equal(engineRoomCollapsed(engine), true);
  assert.equal(panel.flags.collapsed, false);
  assert.deepEqual(other.bounding, before);
  assert.equal(toggleEngineRoom(groups, []), false);
  assert.equal(engineRoomCollapsed([]), false);
});
