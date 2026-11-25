/** 프롬프트 칸의 키와 기본 순서 — core/prompt_order.py 의 DEFAULT_ORDER 와 같다(tests/test_prompt_order_sections.py). */
export const DEFAULT_PROMPT_SECTIONS = ['char_count', 'character', 'copyright', 'artist', 'prefix', 'main', 'suffix'] as const

export type PromptSectionKey = typeof DEFAULT_PROMPT_SECTIONS[number]

const isSection = (key: unknown): key is PromptSectionKey => (DEFAULT_PROMPT_SECTIONS as readonly unknown[]).includes(key)

/** 받은 순서를 칸 목록으로 — 모르는 키는 버리고 빠진 칸은 끝에 붙인다(core/prompt_order.load_order 와 같은 규칙). 입력 칸은 사라지면 안 된다. */
export function orderedSections(keys: readonly unknown[]): PromptSectionKey[] {
  const known = [...new Set(keys.filter(isSection))]
  return [...known, ...DEFAULT_PROMPT_SECTIONS.filter((key) => !known.includes(key))]
}
