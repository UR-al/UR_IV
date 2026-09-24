/**
 * 자동화 조종석 '다음 프롬프트' 편집의 순수 규칙 (AutomationPanel.vue).
 *
 * 편집 상태 = 기준 프롬프트(basePrompt) + 뺀 태그 인덱스(removed) + 더한 태그(added).
 * 편집 결과는 전문(finalPrompt)으로 백엔드에 '일회성 덮어쓰기'로 보낸다. 백엔드는 곧바로
 * 그 전문을 automationStatus.prompt 로 되돌려 주는데, 이 메아리를 기준으로 삼으면 removed
 * 인덱스와 added 가 한 번 더 적용된다('a, b, c' 에서 b 를 빼고 d 를 더하면 'a, d, d').
 * 그래서 기준은 메아리가 아닌 진짜 새 프롬프트가 올 때만 바꾼다(:func:`isOverrideEcho`).
 */

/** 괄호·꺾쇠 안의 쉼표는 태그 구분자가 아니다 — `(a, b:1.2)` · `<lora:x:1>` 을 안 쪼갠다. */
export function splitPrompt(text: string): string[] {
  const out: string[] = []
  let depth = 0
  let cur = ''
  for (const ch of text || '') {
    if (ch === '(' || ch === '[' || ch === '{' || ch === '<') depth += 1
    else if (ch === ')' || ch === ']' || ch === '}' || ch === '>') depth = Math.max(0, depth - 1)
    if (ch === ',' && depth === 0) {
      const t = cur.trim()
      if (t) out.push(t)
      cur = ''
      continue
    }
    cur += ch
  }
  const tail = cur.trim()
  if (tail) out.push(tail)
  return out
}

/** 기준 태그에서 removed 인덱스를 빼고 added 를 뒤에 붙인 전문. */
export function composeEditedPrompt(baseTags: string[], removed: number[], added: string[]): string {
  const gone = new Set(removed)
  return [...baseTags.filter((_, i) => !gone.has(i)), ...added].join(', ')
}

/**
 * 들어온 nextPrompt 가 우리가 방금 보낸 덮어쓰기의 메아리인가.
 * 메아리면 같은 장이라 편집(취소선·추가 태그)과 기준을 그대로 둔다. 보낸 것이 없으면('')
 * 메아리가 아니다 — 빈 프롬프트가 와도 기준을 새로 잡아야 한다.
 */
export function isOverrideEcho(incoming: string, lastSent: string): boolean {
  return lastSent !== '' && (incoming || '') === lastSent
}
