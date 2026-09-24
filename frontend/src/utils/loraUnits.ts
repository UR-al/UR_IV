/**
 * LoRA 가중치 단위 변환 — Vue 저장 형식(정수 퍼센트) ↔ 브리지 형식(배율).
 *
 * 경계마다 단위가 고정이다 (core/lora_stack.py 와 같은 계약):
 * - Vue loraStack · localStorage · ui_prefs.loraStack : 정수 퍼센트 (80 = 0.80배)
 * - set_lora_stack / loraStackLoaded · Python _vue_lora_entries : 배율 (0.8)
 *
 * 예전엔 Python 쪽에 두 단위가 섞여 들어가 프로파일 적용 시 LoRA 가 100배(<lora:x:95.00>)가 됐다.
 * 변환은 이 두 함수에서만 한다.
 */
import type { LoraEntry } from '../types/bridge'

const DEFAULT_PERCENT = 80

// 기존 동작 그대로: Number() 로 바꿔 유한하면 그 값(null/'' → 0), 아니면 기본값.
function finite(value: unknown): number | null {
  const n = Number(value)
  return Number.isFinite(n) ? n : null
}

function triggerWords(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((w): w is string => typeof w === 'string') : []
}

/** Vue 저장 항목(정수 %) → set_lora_stack 항목(배율). 숫자가 아니면 0.8배. */
export function toBridgeLoraEntries(stack: ReadonlyArray<Partial<LoraEntry>> | null | undefined): LoraEntry[] {
  return (stack ?? []).map((l) => {
    const percent = finite(l?.weight)
    return {
      name: String(l?.name || ''),
      weight: (percent ?? DEFAULT_PERCENT) / 100,
      enabled: l?.enabled !== false,
      triggerWords: triggerWords(l?.triggerWords),
    }
  })
}

/** loraStackLoaded 항목(배율) → Vue 저장 항목(정수 %). 배열이 아니면 null(무시). */
export function fromBridgeLoraEntries(entries: unknown): LoraEntry[] | null {
  if (!Array.isArray(entries)) return null
  return entries
    .filter((e): e is Record<string, unknown> => !!e && typeof e === 'object')
    .map((e) => {
      const multiplier = finite(e.weight)
      return {
        name: String(e.name || ''),
        weight: multiplier === null ? DEFAULT_PERCENT : Math.round(multiplier * 100),
        enabled: e.enabled !== false,
        triggerWords: triggerWords(e.triggerWords),
      }
    })
}
