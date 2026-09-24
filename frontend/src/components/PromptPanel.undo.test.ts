import { describe, expect, it } from 'vitest'
import source from './PromptPanel.vue?raw'
import { PROMPT_UNDO_KEYS } from '../utils/promptUndoKeys'

/**
 * `[data-prompt-undo]` 표시와 스냅숏 추적 키(PROMPT_UNDO_KEYS)의 짝 (#3).
 *
 * 표시한 칸에서는 Ctrl+Z 가 네이티브 실행 취소 대신 패널 스냅숏 Undo 가 된다. 그런데 그 칸의
 * 값이 추적 키가 아니면 Ctrl+Z 가 **그 칸은 못 되돌리고 다른 칸을 되돌린다** — 인물수(char_count_input)
 * 가 그랬다. 템플릿을 읽어 짝이 어긋나면 실패한다.
 */

interface StartTag { name: string; attrs: string }

/** 따옴표 안의 '>'('() => ...')를 건너뛰며 시작 태그를 모은다 */
function startTags(template: string): StartTag[] {
  const out: StartTag[] = []
  let i = 0
  while ((i = template.indexOf('<', i)) !== -1) {
    const m = /^<([A-Za-z][\w-]*)/.exec(template.slice(i, i + 64))
    if (!m) { i++; continue }
    let j = i + m[0].length
    let quote = ''
    for (; j < template.length; j++) {
      const ch = template[j]
      if (quote) { if (ch === quote) quote = '' }
      else if (ch === '"' || ch === "'") quote = ch
      else if (ch === '>') break
    }
    out.push({ name: m[1], attrs: template.slice(i + m[0].length, j) })
    i = j + 1
  }
  return out
}

const template = source.slice(0, source.indexOf('<script')).replace(/<!--[\s\S]*?-->/g, '')
const script = source.slice(source.indexOf('<script'))
const tags = startTags(template)
const hasUndoAttr = (t: StartTag) => /(^|\s)data-prompt-undo(?=[\s/=]|$)/.test(t.attrs)
const vModelKey = (t: StartTag) => /(?:^|\s)v-model="widgets\.(\w+)"/.exec(t.attrs)?.[1]
const FIELD_TAGS = new Set(['TagBlockField', 'input', 'textarea'])

/** onTotalBlockChange 가 블록 모드 최종 프롬프트 편집을 되쓰는 칸들 */
function totalBlockTargets(): string[] {
  const body = /function onTotalBlockChange[\s\S]*?for \(const key of \[([\s\S]*?)\]\)/.exec(script)?.[1] || ''
  return [...body.matchAll(/'(\w+)'/g)].map(m => m[1])
}

describe('PromptPanel undo scope ↔ PROMPT_UNDO_KEYS', () => {
  const marked = tags.filter(hasUndoAttr)

  it('finds the marked prompt fields', () => {
    expect(marked.length).toBeGreaterThanOrEqual(PROMPT_UNDO_KEYS.length * 2)
  })

  it('every [data-prompt-undo] field writes a tracked key', () => {
    for (const t of marked) {
      const key = vModelKey(t)
      if (key) {
        expect(PROMPT_UNDO_KEYS, `${t.name} v-model=${key}`).toContain(key)
        continue
      }
      // v-model 이 없는 유일한 표시 칸: 블록 모드 최종 프롬프트 — 추적 칸으로만 되쓴다
      expect(t.attrs).toMatch(/:model-value="widgets\.total_prompt_display"/)
      expect(t.attrs).toMatch(/@update:model-value="onTotalBlockChange"/)
    }
  })

  it('the block-mode final prompt only rewrites tracked keys', () => {
    const targets = totalBlockTargets()
    expect(targets).toContain('char_count_input')
    for (const key of targets) expect(PROMPT_UNDO_KEYS, key).toContain(key)
  })

  it('every tracked key is marked in both block and text mode, and never left unmarked', () => {
    for (const key of PROMPT_UNDO_KEYS) {
      const bound = tags.filter(t => FIELD_TAGS.has(t.name) && vModelKey(t) === key)
      expect(bound.some(t => t.name === 'TagBlockField'), `${key} block field`).toBe(true)
      expect(bound.some(t => t.name !== 'TagBlockField'), `${key} text field`).toBe(true)
      for (const t of bound) expect(hasUndoAttr(t), `${key} <${t.name}> needs data-prompt-undo`).toBe(true)
    }
  })

  it('uses the shared history composable with the shared keys', () => {
    expect(script).toMatch(/usePromptUndo\(widgets\)/)
    expect(script).not.toMatch(/const UNDO_KEYS\s*=/)
  })
})
