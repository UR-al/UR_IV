/**
 * T2I 생성 family(Standard / Krea2) 콤보의 옵션과 판정 — 순수 로직.
 *
 * Python ComboBoxProxy 는 Vue 에서 온 값을 항목 목록과 **대소문자까지 정확히** 비교한다.
 * 그래서 이 옵션 라벨은 core/generation_family.py 의 GENERATION_FAMILY_ITEMS 와 글자 그대로
 * 같아야 한다 (tests/test_generation_family.py 가 고정). 예전엔 Python 이 'KREA2', Vue 가
 * 'Krea2' 라서 Krea2 선택이 무시되고 숨은 Standard 체크포인트가 steps 8·CFG 1 로 생성됐다.
 *
 * localStorage('generationFamily')에는 대문자('STANDARD'/'KREA2')가 저장된다 — 비교는 항상
 * 대소문자 무시로 한다.
 */

export const GENERATION_FAMILY_ITEMS = ['Standard', 'Krea2'] as const
export type GenerationFamilyLabel = typeof GENERATION_FAMILY_ITEMS[number]

/** 저장값(대문자)·옛 값·빈 값을 표시 라벨로. 모르는 값은 Standard. */
export function toFamilyLabel(value: unknown): GenerationFamilyLabel {
  const key = String(value ?? '').trim().toUpperCase()
  return GENERATION_FAMILY_ITEMS.find((item) => item.toUpperCase() === key) ?? GENERATION_FAMILY_ITEMS[0]
}

/** 라벨·대문자 저장값 어느 쪽이든 Krea2 인지. */
export function isKrea2Family(value: unknown): boolean {
  return String(value ?? '').trim().toUpperCase() === 'KREA2'
}

/**
 * 시작 시 복원할 라벨 — 이 브라우저에 저장된 선택이 없으면 null(Python 현재값을 그대로 둔다).
 * 웹 모드의 새 브라우저가 기본값 'Standard' 를 밀어 공유 Python 콤보를 바꾸지 않게 한다.
 */
export function storedFamilyLabel(stored: unknown): GenerationFamilyLabel | null {
  return String(stored ?? '').trim() ? toFamilyLabel(stored) : null
}

/**
 * 시작 시 localStorage 선택을 복원할 때 steps/CFG 를 어떻게 할지.
 * - 'restore-standard': Standard 로 돌아가며 남아 있던 Standard 스냅샷(steps/CFG)을 되살린다
 * - 'none': 콤보 대입이 watch 를 발동시켜 처리하거나(Standard → Krea2 전환이면 watch 가 스냅샷을
 *   만들고 8/1 을 적용), 할 일이 없음
 *
 * 저장값도 Krea2, 콤보도 **이미** Krea2 면 'none' 이다 — Python 이 페이지 새로고침(웹 모드 새로고침,
 * QWebEngine 재로드)을 넘어 상태를 들고 있다는 뜻이라 steps/CFG 도 사용자가 Krea2 에서 맞춘 현재
 * 값이다. 예전엔 여기서 8/1 을 다시 덮어써 새로고침할 때마다 조정값이 사라졌고, 그 Krea2 값으로
 * Standard 스냅샷까지 만들 수 있었다. Krea2 기본값은 처음 전환할 때 watch 가 이미 적용했다.
 */
export type FamilyRestoreAction = 'restore-standard' | 'none'

export function familyRestoreAction(
  saved: unknown,
  _current: unknown,
  hasStandardSnapshot: boolean,
): FamilyRestoreAction {
  if (isKrea2Family(saved)) return 'none'
  return hasStandardSnapshot ? 'restore-standard' : 'none'
}
