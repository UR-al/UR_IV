/**
 * 전역 조건식의 두 사본(localStorage 'searchCondRules' 캐시 ↔ config/cond_rules.json) 중 무엇을 쓸지,
 * 그리고 '같은 내용인가'의 판단 — 순수 로직 (감사 #108).
 *
 * 예전엔 부팅 복원이 자동저장을 불러 규칙이 있는 사용자는 매번 파일을 다시 쓰고 거짓 '저장되었습니다'
 * 토스트를 봤고, 파일 값은 로컬 배열이 비었을 때만 반영돼 수동 편집·다른 기기(웹 모드)의 변경이 다음
 * 부팅에 조용히 되돌려졌다. 이제
 * - 파일이 주인이다(부팅 시 파일이 이긴다).
 * - 단, 로컬 사본이 **더 최신 편집**(updatedAt 이 더 큼)이면 로컬을 쓰고 파일로 올린다 — 앱은
 *   os._exit 로 끝나서 800ms 디바운스 안에 있던 편집이 파일에 못 간 채 사라질 수 있다. 그 편집은
 *   localStorage 에만 남아 있다(이 복구를 잃지 않으려고 '파일 무조건 우선'으로 하지 않는다).
 * - 내용이 같으면 보내지 않는다(condRulesContentKey).
 *
 * save_cond_rules 말고 파일을 통째로 바꾸는 Python 경로는 이 규칙에 맞춰 updatedAt 을 정한다
 * (core/cond_rules_store.py). 설정 백업 가져오기는 **가져온 시각**을 찍어 그 전에 편집한 캐시를
 * 이긴다 — 백업 속 시각(내보낸 때·옛 백업은 없음)을 두면 재시작 부팅에서 캐시가 가져온 규칙을
 * 되덮었다. 레거시 이관은 시각 없이(0) 쓴다 — 시각이 있는 캐시는 늘 그보다 새 편집이다.
 */

export interface CondRulesSnapshot {
  enabled?: unknown
  positive?: unknown
  negative?: unknown
  updatedAt?: unknown
}

export type CondRulesSource = 'local' | 'file' | 'none'

type Rule = Record<string, unknown>

function rulesOf(value: unknown): Rule[] {
  if (!Array.isArray(value)) return []
  return value.filter((r): r is Rule => !!r && typeof r === 'object' && !!((r as Rule).condition || (r as Rule).target))
}

function stable(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stable)
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {}
    for (const key of Object.keys(value as Rule).sort()) out[key] = stable((value as Rule)[key])
    return out
  }
  return value
}

/** 저장할 내용만(마스터 토글 + 채워진 규칙) — updatedAt·제어 플래그는 빠진다. */
export function condRulesContent(snapshot: CondRulesSnapshot | null | undefined) {
  const s = snapshot && typeof snapshot === 'object' ? snapshot : {}
  return {
    enabled: typeof s.enabled === 'boolean' ? s.enabled : true,   // 기본 ON (condRules.js 와 같다)
    positive: rulesOf(s.positive),
    negative: rulesOf(s.negative),
  }
}

/** 내용 비교 키 — 규칙 객체의 키 순서가 달라도 같은 내용이면 같은 키. */
export function condRulesContentKey(snapshot: CondRulesSnapshot | null | undefined): string {
  return JSON.stringify(stable(condRulesContent(snapshot)))
}

export function condRulesUpdatedAt(snapshot: CondRulesSnapshot | null | undefined): number {
  const n = Number(snapshot?.updatedAt)
  return Number.isFinite(n) && n > 0 ? n : 0
}

function hasRules(snapshot: CondRulesSnapshot | null | undefined): boolean {
  const c = condRulesContent(snapshot)
  return c.positive.length > 0 || c.negative.length > 0
}

/**
 * 부팅 때 어느 사본을 쓸지.
 * - 둘 다 없으면 'none', 한쪽만 있으면 그쪽.
 * - 로컬이 파일보다 **엄격히** 최신이면 'local'(파일에 못 간 편집 복구).
 * - 둘 다 시각이 없는 옛 데이터: 파일이 비었고 로컬에 규칙이 있으면 'local', 아니면 'file'.
 * - 그 밖엔 'file'.
 */
export function pickCondRulesSource(
  local: CondRulesSnapshot | null | undefined,
  file: CondRulesSnapshot | null | undefined,
): CondRulesSource {
  const hasLocal = !!local && typeof local === 'object'
  const hasFile = !!file && typeof file === 'object'
  if (!hasLocal && !hasFile) return 'none'
  if (!hasFile) return 'local'
  if (!hasLocal) return 'file'
  const localAt = condRulesUpdatedAt(local)
  const fileAt = condRulesUpdatedAt(file)
  if (localAt > fileAt) return 'local'
  if (!localAt && !fileAt && !hasRules(file) && hasRules(local)) return 'local'
  return 'file'
}
