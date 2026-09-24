/**
 * Ollama 연결 설정의 단일 출처 — 기본 URL, 저장된 URL/모델 읽기, 설치 모델 대조.
 *
 * 모델 기본값은 두지 않는다: 저장된 모델이 없으면 '' 를 보내고 백엔드가 설치 목록을 보고
 * 정한다(core/ollama_client.py resolve_model — 워커 스레드에서). 예전엔 'gemma3:4b' ·
 * 'gemma4:e4b' 리터럴이 파일마다 달리 박혀 서로 어긋났다.
 *
 * 대조 규칙은 파이썬 resolve_model 과 같다(골든 케이스: ollamaPrefs.test.ts ↔
 * tests/test_ollama_model_resolution.py). 두 구현이 갈라지면 화면의 모델과 실제 요청 모델이 달라진다.
 */

export const DEFAULT_OLLAMA_URL = 'http://localhost:11434'

type StorageLike = Pick<Storage, 'getItem'>

function browserStorage(): StorageLike | null {
  try {
    return typeof localStorage === 'undefined' ? null : localStorage
  } catch {
    return null   // 차단된 저장소(웹 모드 사생활 설정 등)
  }
}

function read(storage: StorageLike | null, key: string): string {
  try {
    return String(storage?.getItem(key) || '').trim()
  } catch {
    return ''
  }
}

/** 저장된 Ollama 서버 주소 — 없으면 기본 주소. */
export function storedOllamaUrl(storage: StorageLike | null = browserStorage()): string {
  return read(storage, 'ollamaUrl') || DEFAULT_OLLAMA_URL
}

/** 저장된 Ollama 모델 — 없으면 '' (백엔드가 설치 모델로 정한다). */
export function storedOllamaModel(storage: StorageLike | null = browserStorage()): string {
  return read(storage, 'ollamaModel')
}

/** 'repo:tag' → [repo, tag] 소문자. 태그가 없으면 'latest'. 레지스트리 포트(host:5000/x)의 콜론은 태그가 아니다. */
function splitTag(name: string): [string, string] {
  const text = String(name || '').trim().toLowerCase()
  const at = text.lastIndexOf(':')
  const base = at > 0 ? text.slice(0, at) : ''
  const tag = at > 0 ? text.slice(at + 1) : ''
  if (!base || !tag || tag.includes('/')) return [text, 'latest']
  return [base, tag]
}

function cleanNames(installed: readonly unknown[] | null | undefined): string[] {
  return (installed || []).filter((m): m is string => typeof m === 'string' && !!m.trim()).map(m => m.trim())
}

/** 같은 모델로 설치된 이름(정확히 같거나 name ≡ name:latest, 대소문자 무시) — 다른 태그는 다른 모델이다. */
export function findInstalledModel(name: string, installed: readonly unknown[] | null | undefined): string {
  const want = String(name || '').trim()
  if (!want) return ''
  const names = cleanNames(installed)
  if (names.includes(want)) return want
  const [base, tag] = splitTag(want)
  return names.find(m => { const [b, t] = splitTag(m); return b === base && t === tag }) || ''
}

/** 추천 목록의 '설치됨' 표시 — gemma3:4b 와 gemma3:12b 를 같은 모델로 보지 않는다. */
export function isOllamaModelInstalled(name: string, installed: readonly unknown[] | null | undefined): boolean {
  return !!findInstalledModel(name, installed)
}

/**
 * 요청 모델을 설치 목록에 맞춘다: 1) 같은 모델 2) 같은 계열의 설치된 다른 태그
 * 3) 첫 설치 모델. 목록이 비면 판단 근거가 없으므로 요청 그대로('' 이면 '').
 */
export function resolveInstalledModel(requested: string, installed: readonly unknown[] | null | undefined): string {
  const want = String(requested || '').trim()
  const names = cleanNames(installed)
  if (!names.length) return want
  if (want) {
    const same = findInstalledModel(want, names)
    if (same) return same
    const family = splitTag(want)[0]
    const sibling = names.find(m => splitTag(m)[0] === family)
    if (sibling) return sibling
  }
  return names[0]
}
