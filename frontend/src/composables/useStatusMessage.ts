/**
 * 하단 계기 스트립의 상태 한 줄 — Python `show_status` 가 보내는 `statusMessage` 를 받아 보인다.
 *
 * 예전 `show_status` 는 파이썬 쪽 더미 라벨에만 써서 79곳의 진행·완료·실패 문구가 어디에도
 * 나오지 않았다. 토스트가 아니다(스텝마다 오는 진행 문구도 있다) — 한 줄을 덮어쓰고
 * timeoutMs 뒤 지운다(0 이면 다음 문구까지 유지).
 *
 * 마운트할 때 `getStatusMessage` 로 마지막 문구를 한 번 읽는다: Vue 가 뜨기 전(설정 불러오기
 * 실패 등)이나 웹 클라이언트가 붙기 전에 보낸 문구도 남은 시간만큼 보인다.
 */
import { onMounted, onUnmounted, shallowRef } from 'vue'
import { getBackend, onBackendEvent } from '../bridge.js'
import { createStatusLine, type StatusMessage } from '../utils/statusMessage'

export function useStatusMessage() {
  const message = shallowRef<StatusMessage | null>(null)
  // 표시 규칙(덮어쓰기·만료·늦게 읽은 문구의 남은 시간)은 utils/statusMessage 의 createStatusLine 이 맡는다.
  const line = createStatusLine({ onChange: (next) => { message.value = next } })
  let disconnect: (() => void) | null = null

  onMounted(async () => {
    disconnect = onBackendEvent('statusMessage', (json: string) => { line.live(json) })
    try {
      const backend: any = await getBackend()
      if (typeof backend?.getStatusMessage !== 'function') return
      // 마운트 뒤 실시간 문구가 이미 왔으면 그게 더 새롭다 — replay 가 옛 문구로 덮지 않는다.
      backend.getStatusMessage((json: string) => { line.replay(json) })
    } catch { /* 브리지가 없으면(개발 서버) 줄만 비어 있다 */ }
  })

  onUnmounted(() => {
    line.dispose()
    disconnect?.()
    disconnect = null
  })

  return { message }
}
