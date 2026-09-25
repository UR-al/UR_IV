import { readonly, ref } from 'vue'
import { onBackendBound, onBackendEvent } from '../bridge.js'
import { getValue, requestAction } from '../stores/widgetStore.js'
import type { SamExtraCapabilitiesEvent, SamExtraFeature } from '../types/bridge'
import {
  mayUseFeature, missingEnabledFeatures, missingFeaturesMessage, parseSamExtraCapabilities,
} from '../utils/samExtraCapabilities'

/**
 * sam-extra 런타임 기능 스냅샷 — 모듈 싱글턴. 파이썬(ui/sam_extra_capabilities_actions.py)이
 * WebUI 에 연결될 때 GET 으로 확인해 `samExtraCapabilities` 로 보낸 마지막 값을 든다.
 *
 * 뒤 패키지(카드 배지·비활성화)는 `capabilities` / `mayUse(feature)` 를 읽는다. 지금 화면 변화는
 * 하나뿐이다: 확장(또는 그 스크립트)이 없는데 그 기능을 켜 두었으면 경고 토스트를 한 번 띄운다.
 *
 * 늦게 붙은 구독자: 연결 알림은 Vue 마운트 전에 올 수 있어서, 백엔드가 붙을 때마다
 * `sam_extra_capabilities_get`(refresh 없음)으로 마지막 스냅샷을 다시 받는다.
 */

const capabilities = ref<SamExtraCapabilitiesEvent | null>(null)
let bound = false
let toast: ((type: string, msg: string) => void) | null = null
let lastToastKey = ''

function onSnapshot(json: string) {
  const next = parseSamExtraCapabilities(json)
  if (!next) return
  capabilities.value = next
  const missing = missingEnabledFeatures(next, getValue)
  const message = missingFeaturesMessage(next, missing)
  // 같은 확인 결과(checked_at)로는 한 번만 — 재생(get)으로 같은 스냅샷이 다시 와도 조용하다.
  const key = `${next.checked_at || ''}|${message}`
  if (message && toast && key !== lastToastKey) {
    lastToastKey = key
    toast('warning', message)
  }
}

function bind() {
  if (bound) return
  bound = true
  onBackendEvent('samExtraCapabilities', onSnapshot)
  onBackendBound(() => requestAction('sam_extra_capabilities_get', { refresh: false }))
}

/** 다시 확인 — 결과는 samExtraCapabilities 이벤트로 온다. 파이썬이 30초에 한 번까지만 받아 준다
 *  (그 사이 요청은 마지막 스냅샷을 다시 보낸다). 이 수동 확인만 Forge 확장 목록(다시 스캔하는 GET)도 새로 묻는다. */
function refresh() {
  requestAction('sam_extra_capabilities_get', { refresh: true })
}

export function useSamExtraCapabilities(opts: { addToast?: (type: string, msg: string) => void } = {}) {
  if (opts.addToast) toast = opts.addToast
  bind()
  return {
    capabilities: readonly(capabilities),
    mayUse: (feature: SamExtraFeature) => mayUseFeature(capabilities.value, feature),
    refresh,
  }
}
