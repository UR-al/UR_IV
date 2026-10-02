import type { SamExtraCapabilitiesEvent } from '../types/bridge'

/**
 * Anima 가이던스 디테일 묶음(S²·Adaptive SMC·TSR·Momentum·HiGS·HiFlow — Perturbation 인자 62-90)의 안내.
 *
 * 디테일 묶음이 시작하는 인자 자리 — core/anima_guidance.py `PAG_DETAIL_SUITE_FROM` 의 거울
 * (tests/test_anima_guidance.py 가 대조한다). 인자가 이 수 이하인 sam-extra 는 그 칸을 모르고, Forge 는 넘친
 * 위치 인자를 버린다 — 생성은 되지만 켠 기능만 빠진다.
 */
export const PAG_DETAIL_SUITE_FROM = 62

/**
 * 연결된 Forge 의 sam-extra 가 디테일 묶음을 모르면 안내 문구, 아니면 ''.
 * 모르면(확인 전·ComfyUI — known false) 안내하지 않는다. 파이썬 `detail_suite_note` 와 같은 판단이다.
 */
export function detailSuiteNote(
  caps: Pick<SamExtraCapabilitiesEvent, 'known' | 'anima_guidance_argc'> | null | undefined,
): string {
  if (!caps?.known) return ''
  const argc = caps.anima_guidance_argc
  if (typeof argc !== 'number' || argc > PAG_DETAIL_SUITE_FROM) return ''
  return `연결된 sam-extra 가 이 설정을 모릅니다(Perturbation 인자 ${argc}개 빌드) — 켜도 적용되지 않습니다. `
    + 'sam-extra 를 v0.30.0 이상으로 업데이트하세요.'
}
