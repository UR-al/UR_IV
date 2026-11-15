// Prompt-box presets: the route wrapper and the store the panel, the quick list and the preset window share.
// `route(path, body)` = studio_main's JSON helper (GET when `body` is undefined, POST JSON otherwise, throws on an HTTP error).
// The server (preset_routes.py) answers every call with the whole document `{doc, limits}` (the read also `broken`, `add` also
// `id`), so the store just keeps the last one. A refusal is an HTTP error (400, 404 for a preset that is gone, 500 when the
// file cannot be read or written) and `route` throws its text.
// A change is always one preset's (never a whole list): what another tab saved meanwhile is not this page's to overwrite.
// Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).
import { PRESET_LIMITS, matchPreset, sectionList } from "./preset_logic.js";

const PATH = "/presets";
const given = (fields) => Object.fromEntries(Object.entries(fields).filter(([, value]) => value !== undefined));

/** `{load(), add(section, name, text), update(section, id, {name?, text?}), remove(section, id), move(section, id, delta), touch(section, id), insert(section, item, index), savePrefs(prefs)}` — promises of `{doc, limits, id?}` (`load` also `broken`). */
export function presetApi(route) {
  const change = (op, fields) => route(PATH, { op, ...given(fields) });
  return {
    load: () => route(PATH),
    add: (section, name, text) => change("add", { section, name, text }),
    update: (section, id, { name, text } = {}) => change("update", { section, id, name, text }),
    remove: (section, id) => change("delete", { section, id }),
    move: (section, id, delta) => change("move", { section, id, delta }),
    touch: (section, id) => change("use", { section, id }),
    insert: (section, item, index) => change("insert", { section, item, index }),
    savePrefs: (prefs) => change("prefs", { prefs }),
  };
}

/**
 * One store per page. Reading: `doc` (null until loaded), `loaded`, `error` (the last load's failure text or null), `broken`
 * (the file was there but could not be parsed: it reads as empty and the first change keeps it as .bak), `limits`,
 * `prefs` ({close_after_insert}), `list(key)` (a box's presets in the user's order), `match(key, text)` (the preset equal to
 * the box's text or null). `load(force)` reads the file once (again with `force`); `subscribe(fn)` -> unsubscribe, `fn()` runs
 * after every load and change. Changes — `add` (resolves to the new preset), `update`, `remove`, `move`, `touch` (stamps
 * `used`), `insert` (a removed preset back at its place), `setPrefs` — go to the server one at a time in the order made; each
 * resolves once the store holds the answer and rejects (store unchanged) when the server refuses.
 * `noteBoxChange(key, before, after)` remembers the last text a preset put into a box; `boxUndo(key, text)` gives `before`
 * back while the box still holds `after` (else null); `clearBoxUndo()` forgets it.
 */
export function createPresetStore(api) {
  let doc = null;
  let limits = PRESET_LIMITS;
  let error = null;
  let broken = false;
  let loading = null;
  let queue = Promise.resolve();
  let boxChange = null;
  const subscribers = new Set();

  function notify() {
    for (const fn of [...subscribers]) {
      try {
        fn();
      } catch (err) {
        console.warn("[AI Studio] preset subscriber failed:", err);
      }
    }
  }
  function accept(answer) {
    if (answer?.doc) doc = answer.doc;
    if (answer?.limits) limits = answer.limits;
    broken = answer?.broken === true; // only a read says so; a change wrote a whole file
    error = null;
    notify();
    return answer;
  }
  /** Runs `call` after everything asked before it (an answer never overtakes a later one). */
  function inOrder(call) {
    const run = queue.then(call);
    queue = run.catch(() => {});
    return run;
  }
  const change = (call) => inOrder(call).then(accept);

  return {
    get doc() { return doc; },
    get loaded() { return doc !== null; },
    get error() { return error; },
    get broken() { return broken; },
    get limits() { return limits; },
    get prefs() { return { close_after_insert: doc?.prefs?.close_after_insert === true }; },
    list: (key) => sectionList(doc, key),
    match: (key, text) => matchPreset(sectionList(doc, key), text),
    load(force = false) {
      if (doc && !force) return Promise.resolve(doc);
      loading ||= inOrder(() => api.load()).then((answer) => accept(answer).doc, (err) => {
        error = String(err?.message || err);
        notify();
        throw err;
      }).finally(() => { loading = null; });
      return loading;
    },
    subscribe(fn) {
      subscribers.add(fn);
      return () => subscribers.delete(fn);
    },
    add: (key, name, text) => change(() => api.add(key, name, text)).then((answer) => sectionList(answer.doc, key).find((p) => p.id === answer.id) || null),
    update: (key, id, changes) => change(() => api.update(key, id, changes)),
    remove: (key, id) => change(() => api.remove(key, id)),
    move: (key, id, delta) => change(() => api.move(key, id, delta)),
    touch: (key, id) => change(() => api.touch(key, id)),
    insert: (key, item, index) => change(() => api.insert(key, item, index)),
    setPrefs: (prefs) => change(() => api.savePrefs(prefs)),
    noteBoxChange(key, before, after) { boxChange = { key, before: String(before ?? ""), after: String(after ?? "") }; },
    boxUndo: (key, text) => (boxChange && boxChange.key === key && boxChange.after === String(text ?? "") ? boxChange.before : null),
    clearBoxUndo() { boxChange = null; },
  };
}
