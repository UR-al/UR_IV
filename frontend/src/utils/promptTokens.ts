import { stripPromptComments } from './promptComments'

/**
 * SD/CLIP 근사 토큰 수 — 실제 토크나이저보다 약간 보수적.
 * 규칙: 프롬프트 주석(// · /* *\/)을 제외한 뒤 쉼표 구분 태그별 단어 수 합산 + 쉼표 수.
 * NavRail.vue 와 PromptPanel.vue 가 같은 함수를 써서 숫자가 갈라지지 않게 한다.
 */
export function approxTokens(text: unknown): number {
  const sent = stripPromptComments(String(text ?? ''))
  if (!sent.trim()) return 0
  const chunks = sent.split(',').map((s) => s.trim()).filter(Boolean)
  let total = 0
  for (const c of chunks) {
    total += c.split(/\s+/).filter(Boolean).length
  }
  return total + Math.max(0, chunks.length - 1)
}
