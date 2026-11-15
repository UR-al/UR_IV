// The LoRA picker window's `api` (lora_browser.js) over the studio's LoRA routes (lora_routes.py, under /aistudio/studio).
// `route(path, body)` = studio_main's JSON helper (GET when `body` is undefined, POST JSON otherwise, throws on an HTTP error);
// `url(path)` = the full URL of a studio route path for an <img src>. Names are list entries (Korean, Thai marks, spaces,
// brackets, '@'), always URL-encoded. Pure ES module: nothing runs at import (ComfyUI also loads this file as an extension).

const given = (value) => value !== undefined && value !== null && value !== "" && !Number.isNaN(value);

/** `/loras/thumb?…`; `epoch` / `i` only when given: the route takes its defaults then and answers 400 for a non-integer. */
export function thumbPath(name, kind, epoch, i) {
  const query = [`name=${encodeURIComponent(name)}`, `kind=${encodeURIComponent(kind)}`];
  if (given(epoch)) query.push(`epoch=${encodeURIComponent(epoch)}`);
  if (given(i)) query.push(`i=${encodeURIComponent(i)}`);
  return `/loras/thumb?${query.join("&")}`;
}

/** `url` plus the given query params (null / undefined ones left out; the thumb route ignores params it does not know). */
export function withQuery(url, params) {
  const extra = Object.entries(params || {}).filter(([, v]) => given(v)).map(([k, v]) => `${k}=${encodeURIComponent(v)}`);
  return url && extra.length ? `${url}${url.includes("?") ? "&" : "?"}${extra.join("&")}` : url;
}

/** List entry -> its run's custom preview version (`/loras/index` `cover_versions`, keyed by run_key): a cover URL param. */
export function coverVersions(index) {
  const versions = index?.cover_versions || {};
  return new Map((index?.items || []).filter((it) => given(versions[it.run_key])).map((it) => [it.name, versions[it.run_key]]));
}

/** `{index(fresh), detail(name), thumbUrl(name, kind, epoch, i), savePrefs(prefs), setPreview(runKey, image), clearPreview(runKey)}`. */
export function loraApi(route, url) {
  return {
    index: (fresh) => route(fresh ? "/loras/index?fresh=1" : "/loras/index"),
    detail: (name) => route(`/loras/detail?name=${encodeURIComponent(name)}`),
    thumbUrl: (name, kind, epoch, i) => url(thumbPath(name, kind, epoch, i)),
    savePrefs: (prefs) => route("/loras/prefs", prefs),
    setPreview: (runKey, image) => route("/loras/preview", { run_key: runKey, image }),
    clearPreview: (runKey) => route("/loras/preview", { run_key: runKey, clear: true }),
  };
}
