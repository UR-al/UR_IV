import { describe, expect, it } from 'vitest'
import {
  DEFAULT_OLLAMA_URL,
  findInstalledModel,
  isOllamaModelInstalled,
  resolveInstalledModel,
  storedOllamaModel,
  storedOllamaUrl,
} from './ollamaPrefs'

function storage(values: Record<string, string | null>) {
  return { getItem: (key: string) => values[key] ?? null }
}

describe('stored Ollama preferences', () => {
  it('falls back to the shared default URL but never invents a model', () => {
    expect(storedOllamaUrl(storage({}))).toBe(DEFAULT_OLLAMA_URL)
    expect(storedOllamaUrl(storage({ ollamaUrl: '  http://gpu-box:11434 ' }))).toBe('http://gpu-box:11434')
    expect(storedOllamaModel(storage({}))).toBe('')
    expect(storedOllamaModel(storage({ ollamaModel: ' qwen3:8b ' }))).toBe('qwen3:8b')
  })

  it('survives a storage that throws or is missing', () => {
    const blocked = { getItem: () => { throw new Error('denied') } }
    expect(storedOllamaUrl(blocked)).toBe(DEFAULT_OLLAMA_URL)
    expect(storedOllamaModel(blocked)).toBe('')
    expect(storedOllamaUrl(null)).toBe(DEFAULT_OLLAMA_URL)
  })
})

// 골든 케이스 — tests/test_ollama_model_resolution.py ResolveModelTests 와 같은 입력/출력
describe('resolveInstalledModel (mirrors core.ollama_client.resolve_model)', () => {
  it('keeps an exact installed name', () => {
    expect(resolveInstalledModel('gemma3:4b', ['llava:7b', 'gemma3:4b'])).toBe('gemma3:4b')
  })

  it('treats name and name:latest as the same model, ignoring case', () => {
    expect(resolveInstalledModel('qwen3', ['llava:7b', 'qwen3:latest'])).toBe('qwen3:latest')
    expect(resolveInstalledModel('Qwen3:Latest', ['qwen3'])).toBe('qwen3')
  })

  it('substitutes the installed tag of the same family', () => {
    expect(resolveInstalledModel('gemma3:4b', ['llava:7b', 'gemma3:12b'])).toBe('gemma3:12b')
  })

  it('falls back to the first installed model for unknown or empty requests', () => {
    expect(resolveInstalledModel('mistral:7b', ['llava:7b', 'gemma3:12b'])).toBe('llava:7b')
    expect(resolveInstalledModel('', ['llava:7b'])).toBe('llava:7b')
  })

  it('keeps the request when nothing is known to be installed', () => {
    expect(resolveInstalledModel('custom:1b', [])).toBe('custom:1b')
    expect(resolveInstalledModel('', [])).toBe('')
    expect(resolveInstalledModel('x', null)).toBe('x')
  })

  it('does not split registry ports or HF quantisation tags on the wrong colon', () => {
    const installed = ['host:5000/team/model', 'hf.co/org/repo:Q4_K_M']
    expect(resolveInstalledModel('host:5000/team/model:latest', installed)).toBe('host:5000/team/model')
    expect(resolveInstalledModel('hf.co/org/repo:BF16', installed)).toBe('hf.co/org/repo:Q4_K_M')
  })
})

describe('installed badge for recommended models', () => {
  it('does not mark a different size of the same family as installed', () => {
    expect(isOllamaModelInstalled('gemma3:4b', ['gemma3:12b'])).toBe(false)
    expect(isOllamaModelInstalled('qwen3.5:9b', ['qwen3.5:4b'])).toBe(false)
    expect(isOllamaModelInstalled('minicpm-v4.5', ['minicpm-v4.5:latest'])).toBe(true)
    expect(findInstalledModel('minicpm-v4.5', ['minicpm-v4.5:latest'])).toBe('minicpm-v4.5:latest')
    expect(findInstalledModel('', ['a'])).toBe('')
  })
})
