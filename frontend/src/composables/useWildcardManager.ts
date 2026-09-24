import { computed, ref } from 'vue'
import {
  createdWildcard, editableLines, insertWildcardSyntax, savedEntryLines, savedWildcardName,
  wildcardContent, wildcardFailureMessage, wildcardReplyOk, wildcardSyntax,
} from '../utils/wildcardFile'
import { appManagerDeps, type ManagerDeps } from './managerDeps'

/** lines = 주석까지 담은 원문(편집·저장용), tags = 해석에 쓰이는 줄(개수 표시) — utils/wildcardFile */
export interface Wildcard { name: string; file: string; tags: string[]; lines?: string[]; [k: string]: any }

/** '삽입 위치' 선택 → 위젯 키. 'clipboard' 는 위젯이 아니라 클립보드로 복사한다. */
const INSERT_TARGETS: Record<string, string> = {
  main: 'main_prompt_text',
  prefix: 'prefix_prompt_text',
  suffix: 'suffix_prompt_text',
}

/**
 * 파일 와일드카드(wildcards/*.txt) 관리 모달의 상태와 동작 — App.vue 에서 추출(App.vue 분할 ④).
 * 화면은 components/managers/WildcardManagerModal.vue. PromptPanel 의 와일드카드 칩이 모달 밖에서
 * `openWildcardByName` 으로 이 상태를 연다 — 그래서 모달 안이 아니라 여기(모듈 싱글턴)에 있다.
 */
export function createWildcardManager(
  deps: Pick<ManagerDeps, 'getBackend' | 'addToast' | 'storeWidgets'> & {
    writeClipboard?: (text: string) => void
  },
) {
  const writeClipboard = deps.writeClipboard ?? ((text: string) => { navigator.clipboard?.writeText(text) })
  const storeWidgets = deps.storeWidgets

  const wildcards = ref<Wildcard[]>([])
  // 와일드카드 시스템 ON/OFF — Python CheckBoxProxy 'wildcard_enabled'(값의 주인은 core/prompt_settings_extras)
  const wildcardEnabled = computed({
    get: () => storeWidgets.wildcard_enabled !== 'false',
    set: (v: boolean) => { storeWidgets.wildcard_enabled = v ? 'true' : 'false' },
  })
  const showWcManager = ref(false)
  const selectedWc = ref('')
  const selectedWcData = computed(() => wildcards.value.find(w => w.name === selectedWc.value) || null)
  const wcEditLines = ref<string[]>([])
  const wcInsertTarget = ref('main')
  const wcRenaming = ref(false)
  const wcNewName = ref('')

  /** 부팅 때 목록을 받는다(App onMounted) — getWildcardTree 가 없는 백엔드면 건너뛴다. */
  function loadWildcardTree(bk: any) {
    if (bk && bk.getWildcardTree) bk.getWildcardTree((json: string) => { try { wildcards.value = JSON.parse(json) } catch {} })
  }

  function selectWildcard(name: string) {
    selectedWc.value = name
    // 주석(#)까지 담은 원문으로 편집한다 — tags 로 편집하면 저장할 때 주석이 사라진다.
    wcEditLines.value = editableLines(wildcards.value.find(w => w.name === name))
  }

  async function saveCurrentWildcard() {
    if (!selectedWc.value) return
    const bk = await deps.getBackend()
    const name = selectedWc.value
    const lines = [...wcEditLines.value]
    bk.saveWildcard(name + '.txt', wildcardContent(lines), (json: string) => {
      if (savedWildcardName(json, name) === null) { deps.addToast('error', '와일드카드 저장 실패'); return }
      deps.addToast('success', '와일드카드 저장됨')
      // 로컬 목록 — 원문(lines)과 개수용 줄(tags)을 함께 맞춘다
      const wc = wildcards.value.find(w => w.name === name)
      if (wc) Object.assign(wc, savedEntryLines(lines))
    })
  }

  async function createNewWildcard() {
    const name = (prompt('새 와일드카드 이름:') || '').trim()
    if (!name) return
    const bk = await deps.getBackend()
    // 만들기 전용 슬롯 — '같은 파일'(대소문자만 다름·이름 규칙으로 같아짐 포함)이 이미 있으면 백엔드가
    // 비우지 않고 그 파일의 이름을 돌려준다. 예전엔 목록과 글자 그대로만 비교하고 saveWildcard(이름, '')
    // 로 써서 'Hairstyle' · 'hairstyle.' 같은 이름이 기존 hairstyle.txt(주석 포함)를 빈 파일로 덮었다.
    bk.createWildcard(name, (json: string) => {
      const res = createdWildcard(json)
      if (!res) {
        deps.addToast('error', wildcardFailureMessage(json, '와일드카드를 만들지 못했습니다'))
        return
      }
      if (!res.created) deps.addToast('info', `'${res.name}' 은(는) 이미 있습니다 — 그 파일을 엽니다`)
      if (wildcards.value.some(w => w.name === res.name)) { selectWildcard(res.name); return }
      if (res.created) {
        wildcards.value.push({ name: res.name, file: res.name + '.txt', tags: [], lines: [] })
        selectWildcard(res.name)
        return
      }
      // 목록에 없던 기존 파일(밖에서 만들었다) — 목록을 새로 받아 원문과 함께 연다
      bk.getWildcardTree((tree: string) => {
        try { wildcards.value = JSON.parse(tree) } catch {}
        selectWildcard(res.name)
      })
    })
  }

  async function deleteWildcard(name: string) {
    if (!confirm(`"${name}" 와일드카드를 삭제할까요?`)) return
    const bk = await deps.getBackend()
    bk.deleteWildcard(name, (json: string) => {
      // 응답을 읽는다 — 파일이 잠겼거나(WinError 32) 읽기 전용이면(WinError 5) 백엔드가 {error} 를
      // 돌려주고 파일은 디스크에 남는다. 그때 목록·선택·편집 중인 줄을 지우면 '삭제됨'처럼 보이지만
      // 프롬프트의 __이름__ 은 계속 풀리고 재시작하면 되살아난다 — 그대로 두고 실패를 알린다.
      if (!wildcardReplyOk(json)) {
        deps.addToast('error', wildcardFailureMessage(json, '와일드카드 삭제 실패'))
        return
      }
      wildcards.value = wildcards.value.filter(w => w.name !== name)
      if (selectedWc.value === name) { selectedWc.value = ''; wcEditLines.value = [] }
      deps.addToast('success', '와일드카드 삭제됨')
    })
  }

  function applyWildcardRename(oldName: string, json: string, requested: string) {
    const saved = savedWildcardName(json, requested)
    if (saved === null) {
      deps.addToast('error', wildcardFailureMessage(json, '이름 변경 실패'))
      return false
    }
    const wc = wildcards.value.find(w => w.name === oldName)
    if (wc) { wc.name = saved; wc.file = saved + '.txt' }
    if (selectedWc.value === oldName) selectedWc.value = saved
    return true
  }

  /** 목록의 이름을 더블클릭 — 프롬프트 창으로 새 이름을 받는다. */
  async function renameWildcard(oldName: string) {
    const newName = (prompt('새 이름:', oldName) || '').trim()
    if (!newName || newName === oldName) return
    const bk = await deps.getBackend()
    bk.renameWildcard(oldName, newName, (json: string) => { applyWildcardRename(oldName, json, newName) })
  }

  /** 편집 영역 제목을 더블클릭 — 제자리 입력으로 바꾼다(포커스는 모달이 준다). */
  function startWcRename() {
    wcRenaming.value = true
    wcNewName.value = selectedWc.value
  }

  async function finishWcRename() {
    wcRenaming.value = false
    const newName = wcNewName.value.trim()
    if (!newName || newName === selectedWc.value) return
    const bk = await deps.getBackend()
    const oldName = selectedWc.value
    bk.renameWildcard(oldName, newName, (json: string) => {
      if (applyWildcardRename(oldName, json, newName)) deps.addToast('success', '이름 변경됨')
    })
  }

  // USE = 와일드카드 사용 문법 삽입 (__name__ — 해석기가 ~/name/~ 와 같은 뜻으로 푼다.
  // __name__ 으로 못 가리키는 이름이면 ~/name/~ — utils/wildcardFile.wildcardSyntax)
  function useWcSyntax() {
    if (!selectedWc.value) return
    const syntax = wildcardSyntax(selectedWc.value)
    if (wcInsertTarget.value === 'clipboard') {
      writeClipboard(syntax)
      deps.addToast('info', '문법 복사됨: ' + syntax)
    } else {
      const key = INSERT_TARGETS[wcInsertTarget.value] || 'main_prompt_text'
      storeWidgets[key] = insertWildcardSyntax(storeWidgets[key] || '', syntax)
      deps.addToast('success', syntax + ' 삽입됨')
    }
  }

  function addWcLine(afterIdx: number) { wcEditLines.value.splice(afterIdx + 1, 0, '') }
  function appendWcLine() { wcEditLines.value.push('') }
  function removeWcLine(index: number) { wcEditLines.value.splice(index, 1) }

  /** PromptPanel 의 와일드카드 칩 — 모달을 열고 그 파일을 고른다. */
  function openWildcardByName(name: string) {
    showWcManager.value = true
    selectWildcard(name)
  }
  function openWcManager() { showWcManager.value = true }
  function closeWcManager() { showWcManager.value = false }

  return {
    wildcards, wildcardEnabled, showWcManager, selectedWc, selectedWcData, wcEditLines, wcInsertTarget,
    wcRenaming, wcNewName,
    loadWildcardTree, selectWildcard, saveCurrentWildcard, createNewWildcard, deleteWildcard,
    renameWildcard, startWcRename, finishWcRename, useWcSyntax,
    addWcLine, appendWcLine, removeWcLine,
    openWildcardByName, openWcManager, closeWcManager,
  }
}

export type WildcardManager = ReturnType<typeof createWildcardManager>

let _app: WildcardManager | null = null
export function useWildcardManager(): WildcardManager {
  if (!_app) _app = createWildcardManager(appManagerDeps())
  return _app
}
