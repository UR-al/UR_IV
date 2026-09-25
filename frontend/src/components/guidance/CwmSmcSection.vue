<template>
  <label class="ext-check-row">
    <ToggleSwitch :model-value="b('guid_cwm_enabled')" @update:model-value="setB('guid_cwm_enabled', $event)" size="sm" />
    <span>Enable CWM</span>
  </label>
  <template v-if="b('guid_cwm_enabled')">
    <div class="ext-row">
      <div class="ext-field"><label>CWM alpha low (초반 저주파 · 0 = 표준 CFG)</label>
        <input type="number" v-model="w._guid_cwm_alpha_low" step="0.01" min="-1" max="2" /></div>
      <div class="ext-field"><label>CWM alpha high (후반 고주파 · 0 = 표준 CFG)</label>
        <input type="number" v-model="w._guid_cwm_alpha_high" step="0.01" min="-1" max="2" /></div>
    </div>
    <div class="ext-note">원본 기본 0 / 0(그 대역은 표준 CFG). 양수 = 그 대역 guidance 를 키움, 음수 = 줄임.
      원본 안내 시작값: low 0.1–0.3(Flow 계열 0.2–0.5) · high 0.1–0.2 — high 가 크면 과선명.</div>
  </template>

  <label class="ext-check-row">
    <ToggleSwitch :model-value="b('guid_smc_master_enabled')" @update:model-value="setB('guid_smc_master_enabled', $event)" size="sm" />
    <span>Enable SMC</span>
  </label>
  <div class="ext-field"><label>SMC preset</label>
    <CustomSelect v-model="w._guid_smc_preset" :options="smcPresets" placeholder="Auto" /></div>
  <div class="ext-row" v-if="w._guid_smc_preset === 'Custom'">
    <div class="ext-field"><label>Custom SMC lambda</label>
      <input type="number" v-model="w._guid_smc_lambda" step="0.1" min="0.5" max="30" /></div>
    <div class="ext-field"><label>Custom SMC k</label>
      <input type="number" v-model="w._guid_smc_k" step="0.01" min="0" max="5" /></div>
  </div>
  <div class="ext-note">Auto는 모델을 감지하며 Anima는 Cosmos / Wan 프리셋을 사용합니다.</div>

  <details class="ag-sub">
    <summary>Legacy CFG base 라디오 (상호배타)</summary>
    <div class="ext-field"><label>CFG base mode</label>
      <CustomSelect v-model="w._guid_cfg_mode" :options="cfgModes" placeholder="Preserve incoming" /></div>
    <label class="ext-check-row">
      <ToggleSwitch :model-value="b('guid_experimental_stack')" @update:model-value="setB('guid_experimental_stack', $event)" size="sm" />
      <span>Experimental stack: SMC → APG → CWM</span>
    </label>
    <label class="ext-check-row">
      <ToggleSwitch :model-value="b('guid_smc_enabled')" @update:model-value="setB('guid_smc_enabled', $event)" size="sm" />
      <span>Enable SMC (legacy)</span>
    </label>
  </details>
</template>

<script setup lang="ts">
import ToggleSwitch from '../ToggleSwitch.vue'
import CustomSelect from '../CustomSelect.vue'
import { useGuidanceWidgets } from './guidanceWidgets'

/**
 * CWM / SMC — CFG base 그룹(원본: namemechan/ComfyUI-DCW 의 CWM·SMC)과 예전 CFG base 라디오.
 * 그룹 제목과 APG 칸(ApgSection)은 AnimaGuidancePanel 이 앞에 둔다.
 * 칸 범위·step 은 원본 노드 입력과 같다: alpha_l/alpha_h −1~2 step 0.01, smc_lambda 0.5~30 step 0.1,
 * smc_k 0~5 step 0.01 (origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:675-755 — 숫자만 옮겼다,
 * GPL 코드는 복사하지 않는다). 원본의 smc_preset 목록 하나(기본 Off)를 여기서는 마스터 스위치 + 목록으로
 * 나눈다(호스트 차이, 1:1 대응). guidanceOriginInputs.test.ts 가 범위를 지킨다.
 *
 * 조각(fragment) 컴포넌트다 — 감싸는 요소 없이 AnimaGuidancePanel 의 그룹 <details> 안에 그대로 놓인다.
 * 그래서 패널의 scoped 속성을 받지 않는다: .ag-sub · .ext-note 모양은 패널이 :deep 으로 입힌다.
 */
const props = defineProps<{ widgets: Record<string, any> }>()
const { w, b, setB } = useGuidanceWidgets(props)

const cfgModes = ['Preserve incoming', 'APG', 'CWM', 'SMC', 'SMC + CWM']
const smcPresets = [
  'Auto', 'SD1.5 / SD2', 'SDXL', 'SD3 / SD3.5',
  'Flux', 'Qwen-Image', 'Cosmos / Wan', 'Custom',
]
</script>
