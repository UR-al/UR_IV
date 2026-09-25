import { effectScope, getCurrentScope, onScopeDispose, type EffectScope } from 'vue'

/**
 * 쓰는 곳이 있는 동안만 사는 모듈 싱글턴.
 *
 * 대화 탭(views/ChatView.vue)과 우하단 도크의 작은 대화 패널(components/dock/ChatMiniPanel.vue)이
 * 같은 대화 목록·스트리밍 상태를 나눠 써야 한다. 그냥 모듈 최상위 싱글턴이면 (1) import 시점에
 * localStorage 를 읽어 테스트가 매번 새 상태로 시작할 수 없고, (2) 백엔드 이벤트 구독이 영영
 * 풀리지 않는다. 그래서 첫 사용자가 부를 때 만들고(effectScope 안 — watch 가 컴포넌트에 묶이지
 * 않는다), 마지막 사용자의 scope 가 끝나면 dispose 한 뒤 버린다. 다음 사용자는 새로 만든다.
 *
 * 사용자 = `use()` 를 부른 effect scope(보통 컴포넌트 setup). scope 밖에서 부르면 해제 시점을 알
 * 수 없어 영영 산 것으로 센다.
 */
export interface RefCountedParts<T> {
  api: T
  /** 마지막 사용자가 떠날 때 — 구독 해제 · 대기 중인 저장 flush 등. scope 의 watch 는 뒤이어 멈춘다. */
  dispose: () => void
}

export interface RefCountedSingleton<T> {
  (): T
  /** 지금 살아 있는 사용자 수(테스트·진단용). */
  users(): number
}

export function createRefCountedSingleton<T>(create: () => RefCountedParts<T>): RefCountedSingleton<T> {
  let current: { parts: RefCountedParts<T>; scope: EffectScope } | null = null
  let count = 0

  function use(): T {
    if (!current) {
      const scope = effectScope(true)
      const parts = scope.run(create)
      if (!parts) { scope.stop(); throw new Error('refCountedSingleton: create() returned nothing') }
      current = { parts, scope }
    }
    const mine = current
    count++
    if (getCurrentScope()) {
      onScopeDispose(() => {
        count--
        if (count > 0 || current !== mine) return
        current = null
        try { mine.parts.dispose() } finally { mine.scope.stop() }
      })
    }
    return mine.parts.api
  }
  use.users = () => count
  return use
}
