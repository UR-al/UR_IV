// 조건부 프롬프트 — Search 탭 + 조건부 모달이 공유하는 상태/로직.
// (규칙 편집은 STUDIO TOOLS의 '조건부' 모달, 적용은 Search 탭의 결과 전송 시.)
//
// 저장 규칙 (감사 #108 — utils/condRulesSource.ts 설명 참고):
// - 파일(config/cond_rules.json)이 주인, localStorage 'searchCondRules' 는 첫 렌더 캐시.
// - 부팅: 캐시로 먼저 그리고, 파일이 오면 파일이 이긴다. 캐시가 더 최신 편집(updatedAt)이면
//   캐시를 조용히 파일로 올린다(os._exit 로 800ms 디바운스 안의 편집이 유실되는 경우의 복구).
// - 편집: 내용이 파일과 다를 때만 800ms 뒤 조용히 저장(자동저장은 토스트 없음).
//   마스터 토글은 생성이 파일에서 읽으므로 즉시 저장. '즉시 저장' 버튼만 _manual 로 토스트.
import { reactive, ref, watch } from 'vue'
import { requestAction } from '../stores/widgetStore.js'
import { onBackendEvent } from '../bridge.js'
import { condRulesContent, condRulesContentKey, condRulesUpdatedAt, pickCondRulesSource } from '../utils/condRulesSource'

function _readStorage(key) {
  try { return window.localStorage.getItem(key) } catch { return null }
}
function _writeStorage(key, value) {
  try { window.localStorage.setItem(key, value) } catch {}
}

/**
 * 조건부 프롬프트 규칙 1개. (lang="ts" 컴포넌트가 템플릿에서 rule.* 접근 시 타입 필요)
 * @typedef {Object} CondRule
 * @property {boolean} enabled
 * @property {string} condition  조건 태그
 * @property {boolean} exists    true=있으면, false=없으면
 * @property {string} target     대상 태그
 * @property {string} action     add | remove | replace
 * @property {string} location   main | prefix | suffix
 */
/** @type {CondRule[]} */
export const condPositive = reactive([])
/** @type {CondRule[]} */
export const condNegative = reactive([])
export const condEnabled = ref(_readStorage('condEnabled') !== 'false')  // 기본 ON

/** 자동저장 디바운스(ms) */
export const COND_RULES_SAVE_DELAY_MS = 800

let _updatedAt = 0          // 마지막 사용자 편집 시각(ms) — 캐시·파일에 같이 저장해 부팅 때 최신을 가린다
let _syncedKey = null       // 파일과 같다고 확인된 내용(복원·전송 직후). 같으면 다시 보내지 않는다
let _fileKnown = false      // 파일 사본을 받았는가 — 받기 전 편집은 캐시에만 두고(더 최신 updatedAt),
                            // 파일이 오면 pickCondRulesSource 가 그 편집을 파일로 올린다(유실 없음)
let _saveTimer = null

function _content() {
  return condRulesContent({
    enabled: condEnabled.value,   // 마스터 토글 — 백엔드가 cond_rules.json에서 읽어 ON/OFF 판단
    positive: condPositive,
    negative: condNegative,
  })
}
function _payload() {
  return { ..._content(), updatedAt: _updatedAt }
}
function _cache(p) {
  _writeStorage('searchCondRules', JSON.stringify(p))
  _writeStorage('condEnabled', String(p.enabled))
}
function _send(p, { manual = false } = {}) {
  clearTimeout(_saveTimer)
  _saveTimer = null
  _syncedKey = condRulesContentKey(p)
  // _manual 은 제어 플래그 — Python 이 파일에 쓰기 전에 뺀다(core/cond_rules_store.py)
  requestAction('save_cond_rules', manual ? { ...p, _manual: true } : p)
}

/** 편집 감시 — 복원이 일으킨 변경·빈 규칙 추가처럼 내용이 파일과 같으면 아무것도 하지 않는다. */
function _onEdited({ immediate = false } = {}) {
  if (condRulesContentKey(_content()) === _syncedKey) return
  _updatedAt = Date.now()
  const p = _payload()
  _cache(p)
  clearTimeout(_saveTimer)
  _saveTimer = null
  if (!_fileKnown) return
  if (immediate) _send(p)
  else _saveTimer = setTimeout(() => _send(_payload()), COND_RULES_SAVE_DELAY_MS)
}

/** '즉시 저장' 버튼 — 늘 보내고, 이것만 저장 토스트를 띄운다. */
export function saveCondRules() {
  if (condRulesContentKey(_content()) !== _syncedKey) _updatedAt = Date.now()
  const p = _payload()
  _cache(p)
  _send(p, { manual: true })
}
watch(condPositive, () => _onEdited(), { deep: true })
watch(condNegative, () => _onEdited(), { deep: true })
// 마스터 토글은 생성 경로가 파일에서 읽는다 — 디바운스 없이 곧바로
watch(condEnabled, () => _onEdited({ immediate: true }))

export function addCondRule(which) {
  const arr = which === 'neg' ? condNegative : condPositive
  arr.push({ enabled: true, condition: '', exists: true, target: '', action: 'add', location: 'main' })
}
export function removeCondRule(which, i) {
  (which === 'neg' ? condNegative : condPositive).splice(i, 1)
}

/** 사본 하나를 화면에 싣는다 — 이 내용은 '파일과 같다'로 기록해 watch 가 되쓰지 않게 한다. */
function _apply(snapshot) {
  const c = condRulesContent(snapshot)
  _syncedKey = condRulesContentKey(c)
  _updatedAt = condRulesUpdatedAt(snapshot)
  condPositive.splice(0, condPositive.length, ...c.positive.map(r => ({ ...r })))
  condNegative.splice(0, condNegative.length, ...c.negative.map(r => ({ ...r })))
  condEnabled.value = c.enabled
  return c
}

function _readLocalSnapshot() {
  const raw = _readStorage('searchCondRules')
  if (!raw) return null
  try {
    const d = JSON.parse(raw)
    if (!d || typeof d !== 'object') return null
    // 옛 캐시는 enabled 를 따로('condEnabled') 들고 있었다
    const storedEnabled = _readStorage('condEnabled')
    if (typeof d.enabled !== 'boolean' && storedEnabled !== null) d.enabled = storedEnabled !== 'false'
    return d
  } catch {
    return null
  }
}

let _loaded = false
export function loadCondRules() {
  if (_loaded) return
  _loaded = true
  // 1) 첫 렌더 캐시 — 파일이 오기 전에 바로 그린다(아직 파일과 같은지 모르니 보내지 않는다)
  const local = _readLocalSnapshot()
  if (local) _apply(local)
  // 2) 파일(config/cond_rules.json, getInitialConfig pull) — 파일이 이긴다. 캐시가 더 최신 편집이면
  //    캐시를 조용히 파일로 올린다.
  onBackendEvent('condRulesLoaded', (json) => {
    let file = null
    try { file = JSON.parse(json) } catch { return }
    if (!file || typeof file !== 'object') return
    _fileKnown = true
    const source = pickCondRulesSource(_readLocalSnapshot(), file)
    if (source === 'file') {
      _apply(file)
      _cache(_payload())
    } else if (source === 'local') {
      const current = _readLocalSnapshot()
      if (current) _apply(current)
      if (condRulesContentKey(_content()) !== condRulesContentKey(file)) _send(_payload())
      else _syncedKey = condRulesContentKey(file)
    }
  })
}

// 생성/큐 전송 시 적용할 규칙 (condEnabled가 꺼져있으면 빈 목록)
export function condRulesPayload() {
  if (!condEnabled.value) return { cond_positive: [], cond_negative: [] }
  return {
    cond_positive: condPositive.filter(r => r.enabled && r.condition && r.target),
    cond_negative: condNegative.filter(r => r.enabled && r.condition && r.target),
  }
}
