import { describe, expect, it } from 'vitest'
import source from './PromptPanel.vue?raw'

/**
 * 프롬프트 최적화 [적용] 가드 — 판단은 utils/optimizePreview(단위 테스트)에 있고, 여기서는
 * 패널이 그것을 실제로 쓰는지만 소스로 고정한다. 메인 칸만 비교하던 가드로 되돌아가면
 * 미리보기 뒤 다른 칸을 바꾼 채 적용했을 때 태그가 프롬프트에서 사라진다.
 */
const script = source.slice(source.indexOf('<script'), source.indexOf('</script>'))

function body(name: string): string {
  const start = script.search(new RegExp(`(?:async )?function ${name}\\(`))
  expect(start, name).toBeGreaterThanOrEqual(0)
  const end = script.indexOf('\n}', start)
  return script.slice(start, end + 2)
}

describe('PromptPanel optimize preview guard', () => {
  it('builds the request and the preview from one snapshot of all seven fields', () => {
    const optimize = body('optimizePrompt')
    expect(optimize).toContain('snapshotOptimizeInputs(widgets)')
    expect(optimize).toContain('optimizeContextPayload(inputs)')
    expect(optimize).toMatch(/^\s+inputs,\s*$/m)
  })

  it('apply re-optimizes when any field changed, not only the main tags', () => {
    const apply = body('applyOptimize')
    expect(apply).toContain('isOptimizePreviewStale(p.inputs, widgets)')
    expect(apply).not.toMatch(/main_prompt_text \|\| ''\) !== p\.before/)
  })

  it('keeps no private copy of the context key list', () => {
    expect(script).not.toMatch(/const OPTIMIZE_CONTEXT_KEYS\s*=/)
  })
})
