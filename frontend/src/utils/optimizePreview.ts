/**
 * 프롬프트 최적화(딥 클리너) 미리보기가 낡았는지 — PromptPanel 의 [적용] 가드.
 *
 * 최적화 대상은 메인 태그뿐이고, 나머지 6칸(인물수·캐릭터·작품·작가·접두·접미)은 context 로 보내
 * '다른 칸에 이미 있는 태그'를 메인에서 빼는 데 쓴다(core/prompt_deep_clean.py). 그래서 미리보기의
 * `after` 는 메인 **과 6칸** 모두에 달려 있다. 예전 [적용]은 메인만 비교해서, 미리보기를 연 채
 * 접두에서 'red hair' 를 지우고 적용하면 메인의 'red hair' 도 (옛 접두 기준으로) 빠져 태그가
 * 프롬프트에서 통째로 사라졌다.
 */

/** 최적화 context 로 보내는 칸 — 메인 태그와 합쳐 최종 프롬프트가 되는 나머지 6칸. */
export const OPTIMIZE_CONTEXT_KEYS = [
  'char_count_input', 'character_input', 'copyright_input',
  'artist_input', 'prefix_prompt_text', 'suffix_prompt_text',
] as const

export interface OptimizeInputs {
  main: string
  /** 칸별 원문 — 비운 칸도 빈 문자열로 남긴다(지운 것도 변화로 알아챈다) */
  context: Record<string, string>
}

type WidgetValues = Record<string, unknown>

const text = (v: unknown): string => (v === undefined || v === null ? '' : String(v))

/** 요청 시점의 입력 스냅숏. */
export function snapshotOptimizeInputs(widgets: WidgetValues): OptimizeInputs {
  const context: Record<string, string> = {}
  for (const key of OPTIMIZE_CONTEXT_KEYS) context[key] = text(widgets[key])
  return { main: text(widgets.main_prompt_text), context }
}

/** 백엔드로 보내는 context 목록 — 빈 칸은 뺀다(예전 동작 그대로). */
export function optimizeContextPayload(snap: OptimizeInputs): string[] {
  return OPTIMIZE_CONTEXT_KEYS.map((k) => snap.context[k] ?? '').filter((t) => t.trim())
}

/** 미리보기를 만든 뒤 메인이나 6칸 중 하나라도 바뀌었으면 true — 옛 결과로 덮으면 안 된다. */
export function isOptimizePreviewStale(snap: OptimizeInputs, widgets: WidgetValues): boolean {
  if (text(widgets.main_prompt_text) !== snap.main) return true
  return OPTIMIZE_CONTEXT_KEYS.some((k) => text(widgets[k]) !== (snap.context[k] ?? ''))
}
