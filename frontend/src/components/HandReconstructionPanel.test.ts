import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import * as Vue from 'vue'
import { createRenderer, h, nextTick, ref, type App } from 'vue'
import { parse, compileScript } from '@vue/compiler-sfc'
import ts from 'typescript'
import source from './HandReconstructionPanel.vue?raw'
import inpaintSource from '../views/InpaintView.vue?raw'
import * as hostBridge from '../bridge.js'
import * as widgetStore from '../stores/widgetStore.js'
import * as media from '../utils/media.js'

const host = vi.hoisted(() => ({ action: vi.fn(), off: vi.fn(), event: vi.fn(), web: false }))
vi.mock('../bridge.js', () => ({ onBackendEvent: (_name: string, callback: unknown) => { host.event(callback); return host.off } }))
vi.mock('../stores/widgetStore.js', () => ({ requestAction: host.action }))
vi.mock('../utils/media.js', () => ({ isWebMode: () => host.web }))
const script = compileScript(parse(source).descriptor, { id: 'hand-reconstruction-test', inlineTemplate: true })
const code = ts.transpileModule(script.content, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }).outputText
const compiled: { default?: any } = {}
new Function('require', 'exports', code)((name: string) => {
  if (name === 'vue') return Vue
  if (name === '../bridge.js') return hostBridge
  if (name === '../stores/widgetStore.js') return widgetStore
  if (name === '../utils/media.js') return media
  throw Error(`Unexpected component dependency: ${name}`)
}, compiled)

class Node {
  children: Node[] = []
  parent: Node | null = null
  props: Record<string, any> = {}
  open = false
  constructor(public tag: string, public text = '') {}
  get textContent(): string { return this.text + this.children.map(child => child.textContent).join('') }
  showModal() { this.open = true }
  close() { this.open = false; this.props.onClose?.() }
}
function insert(node: Node, parent: Node, anchor: Node | null = null) {
  if (node.parent) node.parent.children.splice(node.parent.children.indexOf(node), 1)
  node.parent = parent
  const index = anchor ? parent.children.indexOf(anchor) : -1
  parent.children.splice(index < 0 ? parent.children.length : index, 0, node)
}
const renderer = createRenderer<Node, Node>({
  createElement: tag => new Node(tag), createText: text => new Node('#text', text), createComment: () => new Node('#comment'),
  insert, remove: node => { if (node.parent) node.parent.children.splice(node.parent.children.indexOf(node), 1) },
  setText: (node, text) => { node.text = text }, setElementText: (node, text) => { node.text = text; node.children = [] },
  parentNode: node => node.parent, nextSibling: node => node.parent?.children[node.parent.children.indexOf(node) + 1] ?? null,
  patchProp: (node, key, _old, value) => { node.props[key] = value }, setScopeId: () => {},
  insertStaticContent: (text, parent, anchor) => { const node = new Node('#static', text); insert(node, parent, anchor); return [node, node] },
})
function all(root: Node, predicate: (node: Node) => boolean): Node[] {
  return [...(predicate(root) ? [root] : []), ...root.children.flatMap(child => all(child, predicate))]
}
function button(root: Node, label: string) { return all(root, node => node.tag === 'button' && node.textContent.includes(label))[0]! }
function checkbox(root: Node) { return all(root, node => node.props.type === 'checkbox')[0]! }
function range(root: Node, key: string) { return all(root, node => node.props.type === 'range' && node.props.id.endsWith(`-${key}`))[0]! }
function receive(event: Record<string, unknown>) { host.event.mock.calls[0]![0](JSON.stringify(event)) }
const DATA = 'data:image/png;base64,iVBORw0KGgo='
const AFTER = 'data:image/png;base64,QUFB'
const MASK = 'data:image/png;base64,TUFTSw=='
const GENERATE = 'hand_reconstruction_generate'
const CANCEL = 'hand_reconstruction_cancel'
const EXPORT = 'hand_reconstruction_export'
const DEFAULT_PROMPT = 'An anatomically coherent hand matching the existing gesture and wrist alignment. Preserve the interaction with nearby objects and the surrounding style.'
let app: App | undefined
async function mount(getInput: () => any = () => ({ image: DATA, mask: MASK })) {
  const root = new Node('root')
  const revision = ref(1)
  const hasImage = ref(true)
  const hasMask = ref(true)
  app = renderer.createApp({ setup: () => () => h(compiled.default, { sourceRevision: revision.value, hasImage: hasImage.value, hasMask: hasMask.value, getInput }) })
  app.mount(root)
  await nextTick()
  return { root, revision, hasImage, hasMask }
}
async function enable(root: Node) { checkbox(root).props.onChange({ target: { checked: true } }); await nextTick() }
async function start(root: Node) {
  await button(root, '손 재구성 후보 생성').props.onClick()
  await nextTick()
  const calls = host.action.mock.calls.filter(([action]) => action === GENERATE)
  return calls[calls.length - 1]![1]
}
// The backend no longer echoes the source: the panel compares against the image it sent.
function finalEvent(requestId: string, extras: Record<string, unknown> = {}) {
  return { action: GENERATE, requestId, phase: 'complete', ok: true, prepared: MASK,
    candidates: [{ index: 0, seed: 20, image: DATA }, { index: 1, seed: 21, image: AFTER }], ...extras }
}
async function complete(root: Node) {
  const payload = await start(root)
  receive(finalEvent(payload.requestId))
  await nextTick()
  return payload
}
beforeEach(() => { host.action.mockReset(); host.event.mockReset(); host.off.mockReset(); host.web = false; vi.useFakeTimers() })
afterEach(() => { app?.unmount(); app = undefined; vi.clearAllTimers(); vi.useRealTimers(); vi.unstubAllGlobals() })

it('is collapsed/off by default and controls do not automatically generate', async () => {
  const { root } = await mount()
  expect(all(root, node => node.tag === 'details')[0]!.props.open).toBeUndefined()
  expect(checkbox(root).props.checked).toBe(false)
  expect(host.action).not.toHaveBeenCalled()
  await enable(root)
  range(root, 'strength').props.onInput({ target: { value: '0.85' } })
  range(root, 'resolution').props.onInput({ target: { value: '700' } })
  await nextTick()
  expect(range(root, 'resolution').props.value).toBe(768)
  expect(host.action).not.toHaveBeenCalled()
  expect(root.textContent).toContain('자동 판정하거나 정상 손을 보장')
})

it('uses exact image/mask snapshots and sends explicit reconstruction settings without a file path', async () => {
  const snapshot = vi.fn(() => ({ image: DATA, mask: MASK }))
  const { root } = await mount(snapshot)
  await enable(root)
  const payload = await start(root)
  expect(snapshot).toHaveBeenCalledOnce()
  expect(payload).toEqual({ requestId: expect.any(String), image: DATA, mask: MASK,
    settings: { enabled: true, strength: 0.9, candidates: 2, padding: 64, resolution: 768, feather: 4 }, prompt: DEFAULT_PROMPT })
  expect(payload).not.toHaveProperty('image_path')
  expect(all(root, node => node.tag === 'fieldset')[0]!.props.disabled).toBe(true)
})

it('requires an image and mask, and rejects external URLs without fetching', async () => {
  const { root, hasMask } = await mount(() => ({ image: 'https://external.invalid/hand.png', mask: MASK }))
  await enable(root)
  hasMask.value = false
  await nextTick()
  expect(button(root, '손 재구성 후보 생성').props.disabled).toBe(true)
  await button(root, '손 재구성 후보 생성').props.onClick()
  expect(host.action).not.toHaveBeenCalled()
  hasMask.value = true
  await nextTick()
  await button(root, '손 재구성 후보 생성').props.onClick()
  await nextTick()
  expect(root.textContent).toContain('이미지로 읽지 못했습니다')
  expect(host.action.mock.calls.some(([action]) => action === GENERATE)).toBe(false)
})

it('lets the user choose, compare, and save a candidate without changing the source canvas', async () => {
  const { root, revision } = await mount()
  await enable(root)
  const payload = await complete(root)
  expect(root.textContent).toContain('원본은 변경하지 않았습니다')
  const select = all(root, node => node.tag === 'select')[0]!
  select.props.onChange({ target: { value: '1' } })
  await nextTick()
  await button(root, '크게 전후 비교').props.onClick()
  await nextTick()
  const dialog = all(root, node => node.tag === 'dialog')[0]!
  expect(dialog.open).toBe(true)
  expect(all(dialog, node => node.tag === 'img').map(node => node.props.src)).toEqual([DATA, AFTER, MASK])
  expect(all(root, node => node.tag === 'button' && /적용|원본으로 사용/.test(node.textContent))).toHaveLength(0)
  button(root, '선택 후보 별도 PNG 저장').props.onClick()
  await nextTick()
  const saved = host.action.mock.calls.find(([action]) => action === EXPORT)![1]
  expect(saved).toEqual({ requestId: expect.any(String), previewRequestId: payload.requestId, candidateIndex: 1 })
  expect(saved.requestId).not.toBe(payload.requestId)
  receive({ action: EXPORT, requestId: saved.requestId, phase: 'complete', ok: true, path: 'generated_images/hand_reconstruction/candidate_2.png' })
  await nextTick()
  expect(root.textContent).toContain('별도 PNG로 저장했습니다')
  expect(revision.value).toBe(1)
  button(root, '비교 닫기').props.onClick()
  expect(dialog.open).toBe(false)
})

it('compares against the exact image sent with that request and ignores any source echoed by the backend', async () => {
  const SENT = 'data:image/png;base64,U0VOVA=='
  const NEXT = 'data:image/jpeg;base64,TkVYVA=='
  let image = SENT
  const { root, revision } = await mount(() => ({ image, mask: MASK }))
  await enable(root)
  const first = await start(root)
  expect(first.image).toBe(SENT)
  receive(finalEvent(first.requestId, { source: 'https://external.invalid/echo.png' }))
  await nextTick()
  await button(root, '크게 전후 비교').props.onClick()
  await nextTick()
  const dialog = all(root, node => node.tag === 'dialog')[0]!
  expect(all(dialog, node => node.tag === 'img')[0]!.props.src).toBe(SENT)
  // A new source revision + request pairs the comparison with the NEW input, not the old one.
  image = NEXT
  revision.value++
  await nextTick()
  const second = await start(root)
  receive(finalEvent(first.requestId))          // stale completion for the old request is ignored
  receive(finalEvent(second.requestId))
  await nextTick()
  await button(root, '크게 전후 비교').props.onClick()
  await nextTick()
  expect(all(all(root, node => node.tag === 'dialog')[0]!, node => node.tag === 'img')[0]!.props.src).toBe(NEXT)
})

it('accepts progress only for the current request and ignores cancel acknowledgement until final partial results', async () => {
  const { root } = await mount()
  await enable(root)
  const payload = await start(root)
  receive({ action: GENERATE, requestId: 'stale', phase: 'progress', ok: true, candidate: 4, count: 4 })
  receive({ action: GENERATE, requestId: payload.requestId, phase: 'progress', ok: true, candidate: 1, count: 2, step: 3, total: 20 })
  await nextTick()
  expect(root.textContent).toContain('후보 1 / 2 생성 중 · 3/20 단계')
  button(root, '후속 후보 취소').props.onClick()
  receive({ action: CANCEL, requestId: payload.requestId, phase: 'cancel_requested', ok: true })
  await nextTick()
  expect(root.textContent).toContain('실행 중인 서버 요청은 끝날 수 있습니다')
  expect(all(root, node => node.tag === 'fieldset')[0]!.props.disabled).toBe(true)
  receive(finalEvent(payload.requestId, { canceled: true, candidates: [{ index: 0, seed: 20, image: DATA }], warning: '두 번째 후보는 완료되지 않았습니다.' }))
  await nextTick()
  expect(root.textContent).toContain('후속 생성을 취소했습니다. 1개 후보')
  expect(root.textContent).toContain('두 번째 후보는 완료되지 않았습니다')
  expect(host.action).toHaveBeenCalledWith(CANCEL, { requestId: payload.requestId })
})

it('invalidates pending work on mask revision even when hasMask remains true and never accepts stale results', async () => {
  const { root, revision } = await mount()
  await enable(root)
  const payload = await start(root)
  revision.value++
  await nextTick()
  expect(host.action).toHaveBeenCalledWith(CANCEL, { requestId: payload.requestId })
  receive(finalEvent(payload.requestId))
  await nextTick()
  expect(all(root, node => node.tag === 'select')).toHaveLength(0)
  expect(button(root, '손 재구성 후보 생성').props.disabled).toBe(false)
})

it('does not dispatch an asynchronous snapshot if the source changed while it was being read', async () => {
  let resolveInput!: (value: { image: string; mask: string }) => void
  const { root, revision } = await mount(() => new Promise(resolve => { resolveInput = resolve }))
  await enable(root)
  const pending = button(root, '손 재구성 후보 생성').props.onClick()
  revision.value++
  await nextTick()
  resolveInput({ image: DATA, mask: MASK })
  await pending
  expect(host.action.mock.calls.some(([action]) => action === GENERATE)).toBe(false)
})

it('does not dispatch when cancellation arrives while preparing an asynchronous snapshot', async () => {
  let resolveInput!: (value: { image: string; mask: string }) => void
  const { root } = await mount(() => new Promise(resolve => { resolveInput = resolve }))
  await enable(root)
  const pending = button(root, '손 재구성 후보 생성').props.onClick()
  await nextTick()
  button(root, '후속 후보 취소').props.onClick()
  resolveInput({ image: DATA, mask: MASK })
  await pending
  await nextTick()
  expect(host.action.mock.calls.some(([action]) => action === GENERATE)).toBe(false)
  expect(root.textContent).toContain('생성 요청을 보내지 않았습니다')
})

it('expires preview/cache and closes comparison when source or settings change', async () => {
  const { root, revision } = await mount()
  await enable(root)
  const payload = await complete(root)
  await button(root, '크게 전후 비교').props.onClick()
  const dialog = all(root, node => node.tag === 'dialog')[0]!
  revision.value++
  await nextTick()
  expect(dialog.open).toBe(false)
  expect(host.action).toHaveBeenCalledWith(CANCEL, { requestId: payload.requestId })
  await complete(root)
  range(root, 'padding').props.onInput({ target: { value: '80' } })
  await nextTick()
  expect(all(root, node => node.tag === 'select')).toHaveLength(0)
})

it('rejects malformed/duplicate or external candidate images and reports backend failures', async () => {
  const { root } = await mount()
  await enable(root)
  let payload = await start(root)
  receive(finalEvent(payload.requestId, { candidates: [{ index: 0, seed: 1, image: 'https://external.invalid/candidate.png' }] }))
  await nextTick()
  expect(root.textContent).toContain('올바른 후보 결과')
  expect(all(root, node => node.tag === 'select')).toHaveLength(0)
  payload = await start(root)
  receive({ action: GENERATE, requestId: payload.requestId, phase: 'complete', ok: false, error: '사용자 워크플로에서는 실행할 수 없습니다.' })
  await nextTick()
  expect(root.textContent).toContain('사용자 워크플로')
})

it('disables the feature in the web runtime', async () => {
  host.web = true
  const { root } = await mount()
  expect(checkbox(root).props.disabled).toBe(true)
  await enable(root)
  expect(all(root, node => node.tag === 'button')).toHaveLength(0)
  expect(host.action).not.toHaveBeenCalled()
})

it('cancels and releases listeners after timeout or unmount', async () => {
  const { root } = await mount()
  await enable(root)
  const payload = await start(root)
  await vi.advanceTimersByTimeAsync(600000)
  expect(root.textContent).toContain('서버 응답이 지연')
  expect(host.action).toHaveBeenCalledWith(CANCEL, { requestId: payload.requestId })
  app!.unmount(); app = undefined
  expect(host.off).toHaveBeenCalledOnce()
})

it('wires Inpaint to source bytes/mask snapshots and revisions, never stale native image paths', () => {
  expect(inpaintSource).toContain('<HandReconstructionPanel :source-revision="handSourceRevision"')
  expect(inpaintSource).toContain(':get-input="getHandReconstructionInput"')
  // 새 이미지 읽기를 시작하는 순간(파일은 FileReader 전) 원본 revision 이 바뀐다 — 모든 로드가 beginImageLoad 를 거친다.
  expect(inpaintSource).toMatch(/function beginImageLoad\(\)[^{]*\{[^}]*handSourceRevision\.value\+\+/)
  expect(inpaintSource).toMatch(/function loadFile[\s\S]*?beginImageLoad\(\)[\s\S]*?readAsDataURL/)
  expect(inpaintSource).toMatch(/function renderDirty[\s\S]*?handSourceRevision\.value\+\+/)
  expect(inpaintSource).toContain("image: input.sourceKind === 'canvas' ? imgRef.value.toDataURL('image/png') : input.image")
  expect(inpaintSource).toMatch(/async function loadFromPath[\s\S]*?path\.startsWith\('blob:'\)[\s\S]*?imagePath\.value = ''[\s\S]*?initCanvas\(path, beginImageLoad\(\)\)/)
  expect(source).toContain('@keydown.stop')
  expect(source).toContain('prefers-reduced-motion: reduce')
})

it('discloses canvas-only metadata/pixel limits throughout preview and comparison', async () => {
  const { root } = await mount(() => ({ image: DATA, mask: MASK, sourceKind: 'canvas' }))
  await enable(root)
  await complete(root)
  expect(root.textContent).toContain('원본 메타데이터는 포함하지 않으며')
  await button(root, '크게 전후 비교').props.onClick()
  expect(all(root, node => node.tag === 'dialog')[0]!.textContent).toContain('캔버스 스냅샷 기준 · 원본 메타데이터 미포함')
})

// Execute the production source-reader functions without mounting the drawing
// canvas. State and browser I/O are controlled, but parsing/limits stay real.
const sourceReaderSection = inpaintSource.slice(inpaintSource.indexOf('const HAND_SOURCE_MAX_BYTES'), inpaintSource.indexOf('\nfunction generate()'))
const sourceReaderCode = ts.transpileModule(sourceReaderSection, { compilerOptions: { target: ts.ScriptTarget.ES2020 } }).outputText
function sourceReader(initial: string, getBackend = async (): Promise<any> => ({})) {
  const image = ref(initial)
  const revision = ref(1)
  const ready = ref(true)
  const canvas = { toDataURL: vi.fn(() => AFTER) }
  const read = new Function('imageSrc', 'handSourceRevision', 'handImageReady', 'imgRef', 'hasMask', 'getMaskBase64', 'getBackend',
    `let drawing = false; ${sourceReaderCode}; return getHandReconstructionInput;`)(image, revision, ready, ref(canvas), ref(true), () => MASK, getBackend)
  return { read, image, revision, ready, canvas }
}
class MemoryFileReader {
  result = ''
  onload: (() => void) | undefined
  onerror: (() => void) | undefined
  readAsDataURL(blob: Blob) {
    blob.arrayBuffer().then(buffer => {
      this.result = `data:${blob.type};base64,${btoa(String.fromCharCode(...new Uint8Array(buffer)))}`
      this.onload?.()
    }, () => this.onerror?.())
  }
}

it('forwards uploaded supported raster bytes unchanged, retaining metadata/alpha rather than canvas re-encoding', async () => {
  const fetcher = vi.fn()
  vi.stubGlobal('fetch', fetcher)
  for (const mime of ['png', 'jpeg', 'webp']) {
    const original = `data:image/${mime};base64,${btoa('metadata-and-transparent-rgb-sentinel')}`
    const { read, canvas } = sourceReader(original)
    expect(await read()).toEqual({ image: original, mask: MASK, sourceKind: 'original' })
    expect(canvas.toDataURL).not.toHaveBeenCalled()
  }
  expect(fetcher).not.toHaveBeenCalled()
})

it('reads only current local/blob source bytes with redirect denial, then preserves their exact bytes', async () => {
  const bytes = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10, 65, 76, 80, 72, 65, 77, 69, 84, 65])
  const fetcher = vi.fn(async () => new Response(bytes, { headers: { 'content-type': 'application/octet-stream' } }))
  vi.stubGlobal('fetch', fetcher)
  vi.stubGlobal('FileReader', MemoryFileReader)
  for (const url of ['file:///C:/images/current.png', 'blob:http://localhost/current']) {
    const { read, canvas } = sourceReader(url)
    expect(await read()).toEqual({ image: `data:image/png;base64,${btoa(String.fromCharCode(...bytes))}`, mask: MASK, sourceKind: 'original' })
    expect(fetcher).toHaveBeenLastCalledWith(url, { redirect: 'error', credentials: 'omit', signal: expect.any(AbortSignal) })
    expect(canvas.toDataURL).not.toHaveBeenCalled()
  }
})

it('rejects remote/UNC URLs and unavailable/oversized original files without silent canvas fallback', async () => {
  const fetcher = vi.fn(async () => new Response('x', { headers: { 'content-length': String(64 * 1024 * 1024 + 1) } }))
  vi.stubGlobal('fetch', fetcher)
  for (const url of ['https://external.invalid/hand.png', 'file:////server/share/hand.png', 'file:///%5c%5cserver/share/hand.png', 'file:///%2F%2Fserver/share/hand.png']) {
    const { read, canvas } = sourceReader(url)
    await expect(read()).rejects.toThrow('외부 이미지 URL은 읽지 않습니다')
    expect(canvas.toDataURL).not.toHaveBeenCalled()
  }
  expect(fetcher).not.toHaveBeenCalled()
  const oversized = sourceReader('file:///C:/images/large.png')
  await expect(oversized.read()).rejects.toThrow('64 MB 이하')
  expect(oversized.canvas.toDataURL).not.toHaveBeenCalled()
  fetcher.mockRejectedValueOnce(Error('CORS or local file blocked'))
  const blocked = sourceReader('file:///C:/images/blocked.png')
  await expect(blocked.read()).rejects.toThrow('캔버스로 자동 대체하지 않습니다')
  expect(blocked.canvas.toDataURL).not.toHaveBeenCalled()
})

it('uses a clearly identified PNG canvas fallback only for displayed unsupported image formats', async () => {
  const fetcher = vi.fn()
  vi.stubGlobal('fetch', fetcher)
  for (const mime of ['svg+xml', 'bmp', 'gif']) {
    const { read, canvas } = sourceReader(`data:image/${mime};base64,QUFB`)
    expect(await read()).toEqual({ image: AFTER, mask: MASK, sourceKind: 'canvas' })
    expect(canvas.toDataURL).toHaveBeenCalledWith('image/png')
  }
  expect(fetcher).not.toHaveBeenCalled()
})

it('rechecks both source and mask revision after asynchronous original-file reads', async () => {
  let resolveResponse!: (value: Response) => void
  const fetcher = vi.fn(() => new Promise<Response>(resolve => { resolveResponse = resolve }))
  vi.stubGlobal('fetch', fetcher)
  vi.stubGlobal('FileReader', MemoryFileReader)
  const { read, revision, canvas } = sourceReader('file:///C:/images/hand.png')
  const pending = read()
  revision.value++
  resolveResponse(new Response(new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10])))
  await expect(pending).rejects.toThrow('원본 또는 마스크가 바뀌었습니다')
  expect(canvas.toDataURL).not.toHaveBeenCalled()
})

it('uses the existing native original-byte reader for local Gallery images instead of Chromium file fetch', async () => {
  const nativeRead = vi.fn((_source: string, callback: (data: string) => void) => callback(DATA))
  const backend = vi.fn(async () => ({ loadImageBase64: nativeRead }))
  const fetcher = vi.fn()
  vi.stubGlobal('window', { qt: { webChannelTransport: {} } })
  vi.stubGlobal('fetch', fetcher)
  const { read, canvas } = sourceReader('file:///C:/gallery/current.png', backend)
  expect(await read()).toEqual({ image: DATA, mask: MASK, sourceKind: 'original' })
  expect(nativeRead).toHaveBeenCalledWith('file:///C:/gallery/current.png', expect.any(Function))
  expect(backend).toHaveBeenCalledOnce()
  expect(fetcher).not.toHaveBeenCalled()
  expect(canvas.toDataURL).not.toHaveBeenCalled()
})

it('rejects missing/invalid native original bytes without fetch or silent canvas fallback', async () => {
  let response = ''
  const nativeRead = vi.fn((_source: string, callback: (data: string) => void) => callback(response))
  const backend = vi.fn(async () => ({ loadImageBase64: nativeRead }))
  const fetcher = vi.fn()
  vi.stubGlobal('window', { qt: { webChannelTransport: {} } })
  vi.stubGlobal('fetch', fetcher)
  for (const data of ['', 'https://external.invalid/image.png', 'data:image/png;base64,???', 'data:image/png;base64,QUFB']) {
    response = data
    const { read, canvas } = sourceReader('file:///C:/gallery/current.png', backend)
    await expect(read()).rejects.toThrow('캔버스로 자동 대체하지 않습니다')
    expect(canvas.toDataURL).not.toHaveBeenCalled()
  }
  expect(fetcher).not.toHaveBeenCalled()
})

it('keeps the native original-byte route unavailable for external or UNC sources', async () => {
  const backend = vi.fn(async () => ({ loadImageBase64: vi.fn() }))
  vi.stubGlobal('window', { qt: { webChannelTransport: {} } })
  for (const url of ['https://external.invalid/hand.png', 'file:////server/share/hand.png']) {
    await expect(sourceReader(url, backend).read()).rejects.toThrow('외부 이미지 URL은 읽지 않습니다')
  }
  expect(backend).not.toHaveBeenCalled()
})

it('times out the native callback and rechecks source revision before dispatching original bytes', async () => {
  let callback!: (data: string) => void
  const backend = async () => ({ loadImageBase64: (_source: string, done: (data: string) => void) => { callback = done } })
  vi.stubGlobal('window', { qt: { webChannelTransport: {} } })
  const timedOut = sourceReader('file:///C:/gallery/timeout.png', backend)
  const pending = timedOut.read()
  const timeoutAssertion = expect(pending).rejects.toThrow('현재 로컬 원본 파일을 읽지 못했습니다')
  await vi.advanceTimersByTimeAsync(30000)
  await timeoutAssertion
  callback(DATA)
  expect(timedOut.canvas.toDataURL).not.toHaveBeenCalled()
  const stale = sourceReader('file:///C:/gallery/stale.png', backend)
  const second = stale.read()
  await nextTick()
  stale.revision.value++
  callback(DATA)
  await expect(second).rejects.toThrow('원본 또는 마스크가 바뀌었습니다')
})
