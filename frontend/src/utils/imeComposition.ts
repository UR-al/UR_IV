/**
 * IME(한글·일본어) 조합 중인 키 입력인가.
 *
 * 조합 중 Enter 는 '글자 확정' 이지 '제출/추가' 가 아니다. 브라우저(특히 macOS 웹 모드)는
 * 조합 확정 Enter 를 keydown 'Enter' 로 보내면서 isComposing=true 를 싣는다 — 이걸 제출로
 * 받으면 마지막 음절이 잘리거나(v-model 은 조합 중 갱신되지 않는다) 같은 태그가 두 번 들어간다.
 * 옛 WebKit/일부 IME 는 isComposing 없이 keyCode 229 만 준다.
 * (Windows QtWebEngine 은 조합 중 keydown 을 'Process'/229 로 보내 Enter 핸들러에 닿지 않지만,
 *  웹 모드 브라우저는 다르다 — 방어적으로 모든 확정 Enter 핸들러가 이걸 먼저 본다.)
 */
export interface ImeKeyLike {
  isComposing?: boolean
  keyCode?: number
}

export function isImeComposing(e: ImeKeyLike | null | undefined): boolean {
  if (!e) return false
  return e.isComposing === true || e.keyCode === 229
}
