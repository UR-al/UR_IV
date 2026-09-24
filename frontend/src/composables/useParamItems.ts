import { computed } from 'vue'
import { useWidgetStore } from '../stores/widgetStore.js'

/** SAM3 검출 연산 장치 — cuda 권장(auto 는 CPU 로 떨어질 수 있다) */
export const SAM3_DEVICE_ITEMS = ['cuda', 'auto', 'cpu']
/** SAM3 인페인트 'Masked content' — Forge Neo 확장과 같은 네 가지 */
export const SAM3_FILL_ITEMS = ['fill', 'original', 'latent noise', 'latent nothing']

type PropertyReader = (id: string, prop: string, def?: any) => any

/**
 * 파라미터 열의 드롭다운 목록 — Python 프록시가 위젯 속성(`items`)으로 채운다.
 * 목록이 아직 안 왔을 때의 기본값은 예전 App.vue 와 같다('Use same …' 계열은 Python 이 접두까지 채운다).
 * ADetailer 슬롯의 별도 Checkpoint/VAE 목록은 s1/s2 가 같아서 Python 이 `_ad_s1_*` 에만 채운다.
 */
export function useParamItems(getProperty: PropertyReader = useWidgetStore().getProperty) {
  const items = (id: string, fallback: string[]) => computed<string[]>(() => getProperty(id, 'items') || fallback)
  return {
    samplerItems: items('sampler_combo', []),
    schedulerItems: items('scheduler_combo', []),
    upscalerItems: items('upscaler_combo', []),
    sam3CheckpointItems: items('_sam3_checkpoint', ['sam3.pt']),
    adCheckpointItems: items('_ad_s1_ckpt', ['Use same checkpoint']),
    adVaeItems: items('_ad_s1_vae', ['Use same VAE']),
    hiresCheckpointItems: items('hires_checkpoint_combo', ['Use same checkpoint']),
    hiresSamplerItems: items('hires_sampler_combo', ['Use same sampler']),
    hiresSchedulerItems: items('hires_scheduler_combo', ['Use same scheduler']),
  }
}
