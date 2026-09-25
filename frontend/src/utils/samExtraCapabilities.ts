import type { SamExtraCapabilitiesEvent, SamExtraFeature } from '../types/bridge'

/**
 * sam-extra 기능 스냅샷(samExtraCapabilities 이벤트)을 읽는 순수 로직.
 * 상태는 composables/useSamExtraCapabilities.ts 가 들고, 여기는 계산만 한다.
 *
 * 원칙: `known` 이 false 면 **아무것도 막지 않는다**(모르면 지금처럼 보낸다 — 파이썬 may_use 와 같은 규칙).
 */

/**
 * 앱에서 켤 수 있는 기능 → '켜짐'을 판단하는 위젯 id (하나라도 'true' 면 켜짐).
 * 가이던스 키는 core/anima_guidance.py `_ACTIVATION_KEYS` 의 거울이다 — 앞에 `_` 를 붙인 위젯 id.
 * tests/test_sam_extra_capabilities.py 가 파이썬 표와 대조한다(하나라도 빠지면 실패).
 * `_guid_rdc_enabled` 는 넣지 않는다 — RDC 는 DCW 안에서만 돌아 혼자서는 아무것도 켜지 않는다(DCW 스위치가 이미 키).
 */
export const FEATURE_ENABLE_WIDGETS: Readonly<Partial<Record<SamExtraFeature, readonly string[]>>> = {
  sam3: ['sam3_group'],
  anima_guidance: [
    '_guid_enabled', '_guid_slg_on', '_guid_apg_enabled', '_guid_adg_enabled',
    '_guid_smc_enabled', '_guid_smc_master_enabled', '_guid_cwm_enabled',
    '_guid_dcw_enabled',
    '_guid_dave_enabled', '_guid_cns_enabled', '_guid_mod_enabled',
    '_guid_experimental_stack',
  ],
  skimmed_cfg: ['_skim_enabled'],
  detail_daemon: ['_dd_enabled'],
}

export const FEATURE_LABELS: Readonly<Partial<Record<SamExtraFeature, string>>> = {
  sam3: 'SAM3',
  anima_guidance: 'Anima 가이던스',
  skimmed_cfg: 'Skimmed CFG',
  detail_daemon: 'Detail Daemon',
}

/** 시그널 페이로드(JSON 문자열) → 이벤트. 깨졌거나 모양이 다르면 null. */
export function parseSamExtraCapabilities(json: string): SamExtraCapabilitiesEvent | null {
  try {
    const value = JSON.parse(json || 'null')
    if (!value || typeof value !== 'object' || typeof value.status !== 'string') return null
    if (!value.features || typeof value.features !== 'object') return null
    return value as SamExtraCapabilitiesEvent
  } catch {
    return null
  }
}

/** 기능을 써도 되나 — 모르면 true. */
export function mayUseFeature(caps: SamExtraCapabilitiesEvent | null, feature: SamExtraFeature): boolean {
  if (!caps || !caps.known) return true
  return caps.features?.[feature] === true
}

/** 켜 둔 기능 중 연결된 Forge 에 없는 것. 스냅샷을 모르면 빈 배열. */
export function missingEnabledFeatures(
  caps: SamExtraCapabilitiesEvent | null,
  readWidget: (id: string) => unknown,
): SamExtraFeature[] {
  if (!caps || !caps.known) return []
  const missing: SamExtraFeature[] = []
  for (const [feature, ids] of Object.entries(FEATURE_ENABLE_WIDGETS) as Array<[SamExtraFeature, readonly string[]]>) {
    const enabled = ids.some(id => String(readWidget(id) ?? '') === 'true')
    if (enabled && !mayUseFeature(caps, feature)) missing.push(feature)
  }
  return missing
}

/** 경고 토스트 문구 — 없으면 ''. */
export function missingFeaturesMessage(caps: SamExtraCapabilitiesEvent | null, missing: SamExtraFeature[]): string {
  if (!caps || !missing.length) return ''
  const names = missing.map(f => FEATURE_LABELS[f] || f).join(', ')
  return caps.installed
    ? `연결된 Forge 의 sam-extra 에 없는 기능이 켜져 있습니다: ${names} — 확장을 업데이트하거나 기능을 끄세요.`
    : `연결된 WebUI 에 sam-extra 확장이 없습니다. 켜 둔 ${names} 때문에 생성이 거절(422)될 수 있습니다.`
}
