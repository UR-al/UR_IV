import { hasPromptComment, joinPromptBlocks, splitPromptBlocks, stripPromptComments } from './promptComments'

/**
 * 최종 프롬프트(블록 모드)에서 블록을 지우거나 옮기면 원래 칸에 되쓴다 — PromptPanel.onTotalBlockChange 에서 떼어 냈다.
 *
 * 최종 프롬프트는 칸마다 주석을 뗀 합본이다(core/prompt_order.compose_prompt). 그래서 주석이 든 칸을 예전처럼
 * 쉼표로만 나눠 비교하면 주석 토큰·주석이 붙은 태그가 '최종에 없는 태그' 로 보여 칸에서 지워졌다.
 * - 주석이 없는 칸: 예전 그대로 — 최종에서 사라진 태그를 빼고, 남은 태그를 최종 순서로 정렬해 `', '` 로 잇는다.
 * - 주석이 있는 칸: 블록 규칙(splitPromptBlocks — 줄 주석 뒤 줄바꿈 = 경계)으로 나누고, 주석만 있는 토큰은 늘
 *   남기고, 나머지는 주석을 뗀 글자로 최종에 있는지 본다. 순서는 바꾸지 않는다(주석은 제자리에). 지운 게 있을 때만 쓴다.
 */
export interface TotalTags {
  /** 최종의 쉼표 토큰(소문자) */
  tags: Set<string>
  /** 쉼표 토큰 → 처음 나온 위치 */
  index: Map<string, number>
  /** 쉼표 토큰을 줄로도 나눈 조각(소문자) — 주석을 뗀 칸은 줄바꿈이 남아 'a\nb' 처럼 합쳐 보인다 */
  pieces: Set<string>
}

const LINE_BREAK = /\r\n|\n|\r/
const PIECE_BREAK = /,|\r\n|\n|\r/

export function totalTags(newTotal: string): TotalTags {
  const order = stripPromptComments(newTotal || '').split(',').map(t => t.trim()).filter(Boolean)
  const index = new Map<string, number>()
  order.forEach((t, i) => { const k = t.toLowerCase(); if (!index.has(k)) index.set(k, i) })
  const pieces = new Set<string>()
  for (const t of order) {
    for (const piece of t.split(LINE_BREAK)) {
      const k = piece.trim().toLowerCase()
      if (k) pieces.add(k)
    }
  }
  return { tags: new Set(order.map(t => t.toLowerCase())), index, pieces }
}

/** 칸 하나 → 새 텍스트. 바꿀 것이 없으면 null(그대로 둔다). */
export function syncFieldToTotal(cur: string, total: string | TotalTags): string | null {
  const t = typeof total === 'string' ? totalTags(total) : total
  const text = cur || ''
  if (!hasPromptComment(text)) {
    const kept = text.split(',').map(s => s.trim()).filter(s => s && t.tags.has(s.toLowerCase()))
    kept.sort((a, b) => (t.index.get(a.toLowerCase()) ?? 0) - (t.index.get(b.toLowerCase()) ?? 0))
    const result = kept.join(', ')
    return result !== text.trim() ? result : null
  }
  const tokens = splitPromptBlocks(text)
  const kept = tokens.filter(token => {
    const code = stripPromptComments(token).trim().toLowerCase()
    if (!code || t.tags.has(code)) return true
    // 괄호 묶음('(a, b:1.2)')·줄바꿈이 든 태그는 최종 쪽 쉼표·줄 조각으로 나눠 모두 있는지 본다
    const pieces = code.split(PIECE_BREAK).map(p => p.trim()).filter(Boolean)
    return pieces.length > 0 && pieces.every(p => t.pieces.has(p))
  })
  return kept.length === tokens.length ? null : joinPromptBlocks(kept)
}
