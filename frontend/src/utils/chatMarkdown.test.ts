import { describe, expect, it, vi } from 'vitest'
import { createMarkdownMemo, escapeHtml, inlineMarkdown, renderMarkdown } from './chatMarkdown'

describe('createMarkdownMemo — 스트리밍 중 바뀌지 않은 답은 다시 렌더하지 않는다', () => {
  it('같은 메시지·같은 원문은 캐시를, 바뀐 원문만 다시 렌더한다', () => {
    const render = vi.fn((source: string) => `<p>${source}</p>`)
    const md = createMarkdownMemo(render)
    const done = { content: '긴 이전 답' }
    const streaming = { content: '부분' }
    expect(md(done)).toBe('<p>긴 이전 답</p>')
    expect(md(streaming)).toBe('<p>부분</p>')
    for (let packet = 0; packet < 25; packet++) {   // 초당 약 25패킷
      streaming.content += '.'
      md(done)
      md(streaming)
    }
    expect(render.mock.calls.filter(([source]) => source === '긴 이전 답')).toHaveLength(1)
    expect(render).toHaveBeenCalledTimes(27)
    expect(md(streaming)).toBe(`<p>${streaming.content}</p>`)
  })

  it('기본 렌더러는 renderMarkdown 그대로라 이스케이프 보장이 유지된다', () => {
    const md = createMarkdownMemo()
    const message = { content: '<script>x</script> **굵게**' }
    expect(md(message)).toBe(renderMarkdown(message.content))
    expect(md(message)).not.toContain('<script')
    expect(md({ content: undefined })).toBe('')
  })
})

describe('chatMarkdown — 모델 출력은 신뢰할 수 없는 문자열이다', () => {
  it('HTML 을 먼저 이스케이프한다 — 모델이 <script> 를 뱉어도 글자로 남는다', () => {
    const html = renderMarkdown('<script>alert(1)</script> 와 <img onerror=x>')
    expect(html).not.toContain('<script')
    expect(html).not.toContain('<img')
    expect(html).toContain('&lt;script&gt;')
  })

  it('코드 펜스는 안의 서식을 건드리지 않고, 언어 이름을 남긴다', () => {
    const html = renderMarkdown('```python\nprint("**not bold**")\n```')
    expect(html).toContain('<pre data-lang="python"><code>')
    expect(html).toContain('print(&quot;**not bold**&quot;)')
    expect(html).not.toContain('<strong>')
  })

  it('스트리밍 중 아직 안 닫힌 펜스도 코드 블록으로 보인다', () => {
    const html = renderMarkdown('설명\n```js\nconst a = 1')
    expect(html).toContain('<p>설명</p>')
    expect(html).toContain('<pre data-lang="js"><code>const a = 1</code></pre>')
  })

  it('인라인 코드 안의 별표는 서식이 아니다', () => {
    expect(inlineMarkdown(escapeHtml('`a * b * c` 와 **굵게**'))).toBe('<code>a * b * c</code> 와 <strong>굵게</strong>')
  })

  it('링크는 http(s) 만 새 창으로 — javascript: 는 링크가 되지 않는다', () => {
    const ok = renderMarkdown('[문서](https://example.com/a?b=1&c=2) 그리고 https://x.y/z')
    expect(ok).toContain('<a href="https://example.com/a?b=1&amp;c=2" target="_blank" rel="noopener noreferrer">문서</a>')
    expect(ok).toContain('<a href="https://x.y/z" target="_blank" rel="noopener noreferrer">https://x.y/z</a>')
    const bad = renderMarkdown('[x](javascript:alert(1))')
    expect(bad).not.toContain('<a ')
  })

  it('제목 · 목록 · 인용 · 구분선 · 문단 줄바꿈', () => {
    const html = renderMarkdown('# 제목\n- 하나\n- 둘\n1. 첫째\n2. 둘째\n> 인용\n---\n한 줄\n두 줄')
    expect(html).toContain('<h1>제목</h1>')
    expect(html).toContain('<ul><li>하나</li><li>둘</li></ul>')
    expect(html).toContain('<ol><li>첫째</li><li>둘째</li></ol>')
    expect(html).toContain('<blockquote>인용</blockquote>')
    expect(html).toContain('<hr>')
    expect(html).toContain('<p>한 줄<br>두 줄</p>')
  })

  it('빈 입력은 빈 문자열', () => {
    expect(renderMarkdown('')).toBe('')
  })
})
