import { describe, expect, it } from 'vitest'
import { nextTick, ref } from 'vue'
import { MESH_MESSAGES, MESH_TOOL, useEditorMeshWarp, type MeshCanvasApi } from './useEditorMeshWarp'

/** 마지막 원소 — tsconfig lib 이 Array.prototype.at 이전이라 직접 */
const last = <T>(arr: readonly T[]): T | undefined => arr[arr.length - 1]

const GRID = Array.from({ length: 16 }, (_, i) => [(i % 4) / 3, Math.floor(i / 4) / 3])
const moved = () => GRID.map((p, i) => (i === 5 ? [0.4, 0.35] : [...p]))

function setup(opts: { endResult?: number[][] | null; begin?: boolean } = {}) {
  const log: string[] = []
  const previews: Array<[string, Record<string, unknown>]> = []
  const ops: Array<[string, Record<string, unknown>]> = []
  const toasts: Array<[string, string]> = []
  const canvas: MeshCanvasApi = {
    beginMeshWarp: (pin) => { log.push(`begin:${pin}`); return opts.begin ?? true },
    endMeshWarp: () => { log.push('end'); return opts.endResult === undefined ? moved() : opts.endResult },
    cancelMeshWarp: () => { log.push('cancel') },
    resetMeshWarp: () => { log.push('reset'); return true },
    undoMeshStep: () => { log.push('undo'); return true },
    setMeshPinEdges: (on) => { log.push(`pin:${on}`) },
  }
  const currentTool = ref('box')
  const imagePath = ref('C:/img.png')
  const api = useEditorMeshWarp({
    canvas: () => canvas,
    hasImage: () => !!imagePath.value,
    currentTool,
    imagePath,
    clearPreview: () => { log.push('clearPreview') },
    schedulePreview: (op, params) => { previews.push([op, params]) },
    doOp: (op, params) => { ops.push([op, params]) },
    cancelOtherModes: () => { log.push('cancelOthers') },
    toast: (type, msg) => { toasts.push([type, msg]) },
  })
  return { api, log, previews, ops, toasts, currentTool, imagePath }
}

function key(k: string, mods: Partial<KeyboardEvent> = {}) {
  const flags = { prevented: false, stopped: false }
  const e = {
    key: k, ctrlKey: false, shiftKey: false, altKey: false, ...mods,
    preventDefault: () => { flags.prevented = true },
    stopImmediatePropagation: () => { flags.stopped = true },
  } as unknown as KeyboardEvent
  return { e, flags }
}

describe('useEditorMeshWarp', () => {
  it('start cancels other modes, clears the preview and takes the canvas tool', () => {
    const { api, log, toasts, currentTool } = setup()
    api.startMesh()
    expect(log).toEqual(['cancelOthers', 'clearPreview', 'begin:true'])
    expect(api.meshActive.value).toBe(true)
    expect(currentTool.value).toBe(MESH_TOOL)
    expect(toasts).toEqual([['info', MESH_MESSAGES.started]])
  })

  it('does not start when the canvas has no image', () => {
    const { api, currentTool } = setup({ begin: false })
    api.startMesh()
    expect(api.meshActive.value).toBe(false)
    expect(currentTool.value).toBe('box')
  })

  it('previews moved grids and clears the preview for the untouched grid', () => {
    const { api, log, previews } = setup()
    api.startMesh()
    api.onMeshChanged(moved())
    expect(previews).toEqual([['mesh_warp', { grid: 4, dst: moved(), pin_edges: true }]])
    api.onMeshChanged(GRID)
    expect(last(log)).toBe('clearPreview')
    expect(previews).toHaveLength(1)
  })

  it('confirm sends the operation and returns the tool', () => {
    const { api, ops, currentTool } = setup()
    api.startMesh()
    api.confirmMesh()
    expect(ops).toEqual([['mesh_warp', { grid: 4, dst: moved(), pin_edges: true }]])
    expect(api.meshActive.value).toBe(false)
    expect(currentTool.value).toBe('box')
  })

  it('confirm without moved points only tells the user', () => {
    const { api, ops, toasts } = setup({ endResult: GRID })
    api.startMesh()
    api.confirmMesh()
    expect(ops).toHaveLength(0)
    expect(last(toasts)).toEqual(['info', MESH_MESSAGES.unchanged])
  })

  it('switching tools or changing the image cancels the session', async () => {
    const a = setup()
    a.api.startMesh()
    a.currentTool.value = 'brush'
    await nextTick()
    expect(a.api.meshActive.value).toBe(false)
    expect(a.log).toContain('cancel')
    expect(a.currentTool.value).toBe('brush')

    const b = setup()
    b.api.startMesh()
    b.imagePath.value = 'C:/other.png'
    await nextTick()
    expect(b.api.meshActive.value).toBe(false)
  })

  it('keys: Ctrl+Z undoes a point move, Esc cancels, Enter applies', () => {
    const { api, log, ops } = setup()
    expect(api.handleMeshKey(key('z', { ctrlKey: true }).e)).toBe(false)   // 진행 중이 아니면 손대지 않는다
    api.startMesh()
    const z = key('z', { ctrlKey: true })
    expect(api.handleMeshKey(z.e)).toBe(true)
    expect(z.flags).toEqual({ prevented: true, stopped: true })
    expect(last(log)).toBe('undo')
    expect(api.handleMeshKey(key('a').e)).toBe(false)
    expect(api.handleMeshKey(key('Enter').e)).toBe(true)
    expect(ops).toHaveLength(1)
    api.startMesh()
    expect(api.handleMeshKey(key('Escape').e)).toBe(true)
    expect(api.meshActive.value).toBe(false)
  })

  it('an undo that restored the pin only updates the checkbox (the canvas already has it)', () => {
    const { api, log, previews } = setup()
    api.setMeshPinEdges(true)
    api.startMesh()
    log.length = 0
    api.onMeshPinRestored(false)
    expect(api.meshPinEdges.value).toBe(false)
    expect(log).toEqual([])                       // 캔버스에 다시 보내 테두리를 되돌리지 않는다
    api.onMeshChanged(moved())
    expect(last(previews)?.[1].pin_edges).toBe(false)
    api.setMeshPinEdges(true)
  })

  it('pin-edges is passed to the canvas and the payload', () => {
    const { api, log, previews } = setup()
    api.setMeshPinEdges(false)
    expect(log).toContain('pin:false')
    api.startMesh()
    expect(log).toContain('begin:false')
    api.onMeshChanged(moved())
    expect(previews[0][1].pin_edges).toBe(false)
    api.setMeshPinEdges(true)
  })
})
