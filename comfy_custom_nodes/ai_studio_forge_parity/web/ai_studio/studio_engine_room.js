// Anima studio: fold / unfold the engine room (the role-tagged engine nodes) and fit their groups to them.
// No ComfyUI import: `groups` and `nodes` are LiteGraph objects, so node:test can drive it with fakes.

/** True when there are engine nodes and every one is folded. */
export function engineRoomCollapsed(nodes) {
  return nodes.length > 0 && nodes.every((node) => node.flags?.collapsed);
}

/**
 * Fold the engine nodes when any is open, otherwise unfold them all, then fit every group that holds one of them.
 * `ctx` is the canvas 2D context (the folded title width is measured with it). Returns whether they are now folded.
 */
export function toggleEngineRoom(groups, nodes, ctx) {
  if (!nodes.length) return false;
  const collapse = !engineRoomCollapsed(nodes);
  const engine = new Set(nodes);
  // LiteGraph caches boundingRect once per frame (updateArea) and collapse() does not refresh it: measure the nodes
  // before the groups pick their members and again after folding, or the groups fit the old size.
  const measure = () => nodes.forEach((node) => node.updateArea?.(ctx));
  measure();
  const fitted = groups
    .map((group) => {
      group.recomputeInsideNodes();
      return [group, [...group.nodes]];
    })
    .filter(([, members]) => members.some((node) => engine.has(node)));
  for (const node of nodes) if (Boolean(node.flags?.collapsed) !== collapse) node.collapse(); // collapse() toggles (R11)
  measure();
  for (const [group, members] of fitted) group.resizeTo(members);
  return collapse;
}
