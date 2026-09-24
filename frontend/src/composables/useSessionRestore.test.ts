import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { nextTick, reactive } from 'vue'
import { useSessionRestore } from './useSessionRestore'

// 크래시 복구: 부팅이 프롬프트를 이미 채워도(load_settings) 백업이 다르면 제안하고, 사용자가 고르기
// 전엔 백업을 덮지 않는다. 정상 종료(clean)면 제안하지 않는다(감사 #160).
function setup(backup: unknown, prompts = { main_prompt_text: 'saved prompt', neg_prompt_text: 'neg' }) {
  const storeWidgets = reactive({ ...prompts })
  const saved: any[] = []
  const bk = {
    getSession: (cb: (json: string) => void) => cb(typeof backup === 'string' ? backup : JSON.stringify(backup)),
    saveSession: (json: string, _cb: () => void) => { saved.push(JSON.parse(json)) },
  }
  const addToast = vi.fn()
  const goToTab = vi.fn()
  const api = useSessionRestore({
    storeWidgets, currentTab: () => 't2i', goToTab, getBackend: async () => bk, addToast,
  })
  return { ...api, storeWidgets, saved, addToast, goToTab }
}

async function settle() {
  await vi.advanceTimersByTimeAsync(1200)
  await nextTick()
}

beforeEach(() => { vi.useFakeTimers() })
afterEach(() => { vi.useRealTimers() })

it('offers the crash backup even though the saved prompt was loaded, and keeps it until decided', async () => {
  const s = setup({ tab: 'i2i', prompt: 'crash edit', negative: 'neg', clean: false })
  s.startSessionBackup()
  await settle()
  expect(s.sessionRestore.value).toMatchObject({ prompt: 'crash edit' })
  // 결정 전 편집·주기 저장은 백업을 덮지 않는다
  s.storeWidgets.main_prompt_text = 'typing'
  await nextTick()
  await vi.advanceTimersByTimeAsync(31000)
  expect(s.saved).toEqual([])
  s.applySessionRestore()
  await vi.advanceTimersByTimeAsync(0)
  expect(s.storeWidgets.main_prompt_text).toBe('crash edit')
  expect(s.goToTab).toHaveBeenCalledWith('i2i')
  expect(s.addToast).toHaveBeenCalledWith('success', '이전 세션을 복원했습니다')
  expect(s.saved[s.saved.length - 1]).toMatchObject({ prompt: 'crash edit' })
})

it('does not offer after a clean shutdown and resumes backups right away', async () => {
  const s = setup({ prompt: 'older', negative: 'neg', clean: true })
  s.startSessionBackup()
  await settle()
  expect(s.sessionRestore.value).toBeNull()
  await vi.advanceTimersByTimeAsync(0)
  expect(s.saved[s.saved.length - 1]).toMatchObject({ prompt: 'saved prompt', negative: 'neg' })
  s.storeWidgets.main_prompt_text = 'new text'
  await nextTick()
  await vi.advanceTimersByTimeAsync(2500)
  expect(s.saved[s.saved.length - 1]).toMatchObject({ prompt: 'new text' })
})

it('dismiss resumes backups with the current prompt', async () => {
  const s = setup({ prompt: 'crash edit', negative: '' })
  s.startSessionBackup()
  await settle()
  expect(s.sessionRestore.value).not.toBeNull()
  s.dismissSessionRestore()
  await vi.advanceTimersByTimeAsync(0)
  expect(s.sessionRestore.value).toBeNull()
  expect(s.saved[s.saved.length - 1]).toMatchObject({ prompt: 'saved prompt' })
})

it('an error reply is treated as nothing to restore', async () => {
  const s = setup('{"error":"disk"}')
  s.startSessionBackup()
  await settle()
  expect(s.sessionRestore.value).toBeNull()
  await vi.advanceTimersByTimeAsync(0)
  expect(s.saved).toHaveLength(1)
})
