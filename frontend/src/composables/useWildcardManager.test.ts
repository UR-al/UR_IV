import { afterEach, expect, it, vi } from 'vitest'
import { reactive } from 'vue'
import { createWildcardManager } from './useWildcardManager'

const TREE = [
  { name: 'hair', file: 'hair.txt', tags: ['red hair, blue hair'], lines: ['# colours', 'red hair, blue hair'] },
  { name: 'pose', file: 'pose.txt', tags: ['standing'] },
]

function setup(backend: Record<string, any> = {}) {
  const storeWidgets = reactive<Record<string, any>>({ main_prompt_text: '1girl', prefix_prompt_text: '' })
  const addToast = vi.fn()
  const writeClipboard = vi.fn()
  const bk = {
    getWildcardTree: vi.fn((cb: (json: string) => void) => cb(JSON.stringify(TREE))),
    saveWildcard: vi.fn((_file: string, _content: string, cb: (json: string) => void) => cb('{"ok":true}')),
    // ui/vue_bridge.py deleteWildcard 의 성공 응답 모양 그대로 — 응답 없는 cb() 는 실패로 읽힌다
    deleteWildcard: vi.fn((_name: string, cb: (json: string) => void) => cb('{"ok":true}')),
    renameWildcard: vi.fn((_old: string, next: string, cb: (json: string) => void) => cb(JSON.stringify({ ok: true, name: next }))),
    createWildcard: vi.fn((name: string, cb: (json: string) => void) => cb(JSON.stringify({ ok: true, name, created: true }))),
    ...backend,
  }
  const mgr = createWildcardManager({ getBackend: async () => bk, addToast, storeWidgets, writeClipboard })
  mgr.loadWildcardTree(bk)
  return { mgr, bk, addToast, writeClipboard, storeWidgets }
}

afterEach(() => { vi.unstubAllGlobals() })

it('edits the raw lines (comments kept) and saves them back with the counted tags', async () => {
  const { mgr, bk, addToast } = setup()
  mgr.selectWildcard('hair')
  expect(mgr.wcEditLines.value).toEqual(['# colours', 'red hair, blue hair'])
  mgr.appendWcLine()
  mgr.wcEditLines.value[2] = 'green hair'
  await mgr.saveCurrentWildcard()
  expect(bk.saveWildcard).toHaveBeenCalledWith('hair.txt', '# colours\nred hair, blue hair\ngreen hair', expect.any(Function))
  expect(addToast).toHaveBeenCalledWith('success', '와일드카드 저장됨')
  expect(mgr.selectedWcData.value?.tags).toEqual(['red hair, blue hair', 'green hair'])
})

it('reports a failed save and keeps the local entry untouched', async () => {
  const { mgr, addToast } = setup({ saveWildcard: (_f: string, _c: string, cb: (json: string) => void) => cb('{"error":"disk"}') })
  mgr.selectWildcard('pose')
  mgr.wcEditLines.value = ['sitting']
  await mgr.saveCurrentWildcard()
  expect(addToast).toHaveBeenCalledWith('error', '와일드카드 저장 실패')
  expect(mgr.selectedWcData.value?.tags).toEqual(['standing'])
})

it('USE inserts the syntax into the chosen prompt field, or copies it', () => {
  const { mgr, storeWidgets, writeClipboard } = setup()
  mgr.selectWildcard('hair')
  mgr.useWcSyntax()
  expect(storeWidgets.main_prompt_text).toBe('1girl, __hair__, ')
  mgr.wcInsertTarget.value = 'prefix'
  mgr.useWcSyntax()
  expect(storeWidgets.prefix_prompt_text).toBe('__hair__, ')
  mgr.wcInsertTarget.value = 'clipboard'
  mgr.useWcSyntax()
  expect(writeClipboard).toHaveBeenCalledWith('__hair__')
})

it('openWildcardByName (PromptPanel chip) opens the manager on that file', () => {
  const { mgr } = setup()
  mgr.openWildcardByName('pose')
  expect(mgr.showWcManager.value).toBe(true)
  expect(mgr.selectedWc.value).toBe('pose')
  expect(mgr.wcEditLines.value).toEqual(['standing'])   // lines 가 없는 옛 항목은 tags
})

it('creating an existing file opens it instead of emptying it', async () => {
  const { mgr, addToast } = setup({
    createWildcard: (_n: string, cb: (json: string) => void) => cb(JSON.stringify({ ok: true, name: 'hair', created: false })),
  })
  vi.stubGlobal('prompt', () => 'Hair')
  await mgr.createNewWildcard()
  expect(addToast).toHaveBeenCalledWith('info', "'hair' 은(는) 이미 있습니다 — 그 파일을 엽니다")
  expect(mgr.selectedWc.value).toBe('hair')
  expect(mgr.wildcards.value).toHaveLength(2)
})

it('creating a new file adds an empty entry and selects it', async () => {
  const { mgr } = setup()
  vi.stubGlobal('prompt', () => '  eyes ')
  await mgr.createNewWildcard()
  expect(mgr.wildcards.value.map(w => w.name)).toEqual(['hair', 'pose', 'eyes'])
  expect(mgr.selectedWc.value).toBe('eyes')
  expect(mgr.wcEditLines.value).toEqual([])
})

it('rename in place follows the name the backend actually saved', async () => {
  const { mgr, addToast } = setup({
    renameWildcard: (_o: string, _n: string, cb: (json: string) => void) => cb(JSON.stringify({ ok: true, name: 'hair_styles' })),
  })
  mgr.selectWildcard('hair')
  mgr.startWcRename()
  expect(mgr.wcRenaming.value).toBe(true)
  expect(mgr.wcNewName.value).toBe('hair')
  mgr.wcNewName.value = 'hair styles'
  await mgr.finishWcRename()
  expect(mgr.wcRenaming.value).toBe(false)
  expect(mgr.selectedWc.value).toBe('hair_styles')
  expect(mgr.wildcards.value[0].file).toBe('hair_styles.txt')
  expect(addToast).toHaveBeenCalledWith('success', '이름 변경됨')
})

it('delete asks first and clears the editor when the open file goes', async () => {
  const { mgr, bk } = setup()
  mgr.selectWildcard('pose')
  vi.stubGlobal('confirm', () => false)
  await mgr.deleteWildcard('pose')
  expect(bk.deleteWildcard).not.toHaveBeenCalled()
  vi.stubGlobal('confirm', () => true)
  await mgr.deleteWildcard('pose')
  expect(mgr.wildcards.value.map(w => w.name)).toEqual(['hair'])
  expect(mgr.selectedWc.value).toBe('')
  expect(mgr.wcEditLines.value).toEqual([])
})

it('a failed delete keeps the list, selection and edits and reports the error', async () => {
  // 잠긴 파일(WinError 32)·읽기 전용(WinError 5) — 백엔드는 {error} 를 돌려주고 파일은 디스크에 남는다
  const { mgr, addToast } = setup({
    deleteWildcard: (_n: string, cb: (json: string) => void) => cb(JSON.stringify({ error: '[WinError 32] in use' })),
  })
  mgr.selectWildcard('pose')
  mgr.wcEditLines.value = ['sitting']
  vi.stubGlobal('confirm', () => true)
  await mgr.deleteWildcard('pose')
  expect(mgr.wildcards.value.map(w => w.name)).toEqual(['hair', 'pose'])
  expect(mgr.selectedWc.value).toBe('pose')
  expect(mgr.wcEditLines.value).toEqual(['sitting'])
  expect(addToast).toHaveBeenCalledWith('error', '와일드카드 삭제 실패: [WinError 32] in use')
  expect(addToast).not.toHaveBeenCalledWith('success', expect.anything())
})

it('an empty or malformed delete reply is a failure too, not a silent success', async () => {
  for (const reply of ['', 'not json', 'null', '{}']) {
    const { mgr, addToast } = setup({ deleteWildcard: (_n: string, cb: (json: string) => void) => cb(reply) })
    vi.stubGlobal('confirm', () => true)
    await mgr.deleteWildcard('hair')
    expect(mgr.wildcards.value.map(w => w.name)).toEqual(['hair', 'pose'])
    expect(addToast).toHaveBeenCalledWith('error', '와일드카드 삭제 실패')
  }
})

it('a successful delete of a file that is not open leaves the editor alone and says so', async () => {
  const { mgr, addToast } = setup()
  mgr.selectWildcard('hair')
  vi.stubGlobal('confirm', () => true)
  await mgr.deleteWildcard('pose')
  expect(mgr.wildcards.value.map(w => w.name)).toEqual(['hair'])
  expect(mgr.selectedWc.value).toBe('hair')
  expect(mgr.wcEditLines.value).toEqual(['# colours', 'red hair, blue hair'])
  expect(addToast).toHaveBeenCalledWith('success', '와일드카드 삭제됨')
})

it('create and rename failures carry the backend error text', async () => {
  const { mgr, addToast } = setup({
    createWildcard: (_n: string, cb: (json: string) => void) => cb(JSON.stringify({ error: 'bad name' })),
    renameWildcard: (_o: string, _n: string, cb: (json: string) => void) => cb(JSON.stringify({ error: 'exists' })),
  })
  vi.stubGlobal('prompt', () => 'eyes')
  await mgr.createNewWildcard()
  expect(addToast).toHaveBeenCalledWith('error', '와일드카드를 만들지 못했습니다: bad name')
  vi.stubGlobal('prompt', () => 'pose')
  await mgr.renameWildcard('hair')
  expect(addToast).toHaveBeenCalledWith('error', '이름 변경 실패: exists')
  expect(mgr.wildcards.value.map(w => w.name)).toEqual(['hair', 'pose'])
})

it('the substitution switch mirrors the wildcard_enabled widget (missing = on)', () => {
  const { mgr, storeWidgets } = setup()
  expect(mgr.wildcardEnabled.value).toBe(true)
  mgr.wildcardEnabled.value = false
  expect(storeWidgets.wildcard_enabled).toBe('false')
  mgr.wildcardEnabled.value = true
  expect(storeWidgets.wildcard_enabled).toBe('true')
})

it('line helpers insert after, append and remove', () => {
  const { mgr } = setup()
  mgr.wcEditLines.value = ['a', 'b']
  mgr.addWcLine(0)
  expect(mgr.wcEditLines.value).toEqual(['a', '', 'b'])
  mgr.removeWcLine(1)
  mgr.appendWcLine()
  expect(mgr.wcEditLines.value).toEqual(['a', 'b', ''])
})
