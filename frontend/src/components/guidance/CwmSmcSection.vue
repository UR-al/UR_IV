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
  <div class="ext-field"><label>SMC controller</label>
    <CustomSelect v-model="w._guid_smc_mode" :options="smcModes" placeholder="Unit-L2" /></div>
  <template v-if="!adaptive">
    <div class="ext-field"><label>SMC preset</label>
      <CustomSelect v-model="w._guid_smc_preset" :options="smcPresets" placeholder="Auto" /></div>
    <div class="ext-row" v-if="w._guid_smc_preset === 'Custom'">
      <div class="ext-field"><label>Custom SMC lambda</label>
        <input type="number" v-model="w._guid_smc_lambda" step="0.1" min="0.5" max="30" /></div>
      <div class="ext-field"><label>Custom SMC k</label>
        <input type="number" v-model="w._guid_smc_k" step="0.01" min="0" max="5" /></div>
    </div>
    <div class="ext-note">Auto는 모델을 감지하며 Anima는 Cosmos / Wan 프리셋을 사용합니다. Unit-L2(원본 식)는 1MP Anima
      잠재에서 원소당 보정이 약 4e-4 라 거의 효과가 없습니다.</div>
  </template>
  <template v-else>
    <p v-if="detailNote" class="ext-note">{{ detailNote }}</p>
    <div class="ext-row">
      <div class="ext-field"><label>Adaptive SMC α (이득 = α·mean|e|)</label>
        <input type="number" v-model="w._guid_smc_adaptive_alpha" step="0.01" min="0" max="1" /></div>
      <div class="ext-field"><label>Adaptive SMC λ</label>
        <input type="number" v-model="w._guid_smc_adaptive_lambda" step="0.1" min="0.5" max="30" /></div>
    </div>
    <div class="ext-note" :title="adaptiveTitle">Adaptive sign — sorryhyun 의 Anima 판(원소별 sign, 이득 α·mean|e|).
      프리셋·Custom 값은 쓰지 않습니다. 원작 기본 α 0.2 · λ 5 — 어두워지거나 거칠면 α 를 낮추세요.</div>
  </template>

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
import { computed } from 'vue'
import ToggleSwitch from '../ToggleSwitch.vue'
import CustomSelect from '../CustomSelect.vue'
import { useSamExtraCapabilities } from '../../composables/useSamExtraCapabilities'
import { detailSuiteNote } from '../../utils/guidanceDetailSuite'
import { useGuidanceWidgets } from './guidanceWidgets'

/**
 * CWM / SMC — CFG base 그룹(원본: namemechan/ComfyUI-DCW 의 CWM·SMC)과 예전 CFG base 라디오.
 * 그룹 제목과 APG 칸(ApgSection)은 AnimaGuidancePanel 이 앞에 둔다.
 * 칸 범위·step 은 원본 노드 입력과 같다: alpha_l/alpha_h −1~2 step 0.01, smc_lambda 0.5~30 step 0.1,
 * smc_k 0~5 step 0.01 (origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:675-755 — 숫자만 옮겼다,
 * GPL 코드는 복사하지 않는다). 원본의 smc_preset 목록 하나(기본 Off)를 여기서는 마스터 스위치 + 목록으로
 * 나눈다(호스트 차이, 1:1 대응). guidanceOriginInputs.test.ts 가 범위를 지킨다.
 * SMC controller 와 Adaptive α/λ(인자 68-70, 확장 v0.30 디테일 묶음)는 확장 슬라이더 범위 그대로다 — 원작은
 * sorryhyun/anima_lora smc_cfg.py(MIT). Adaptive 를 고르면 프리셋·Custom 값을 쓰지 않으므로 숨긴다.
 *
 * 조각(fragment) 컴포넌트다 — 감싸는 요소 없이 AnimaGuidancePanel 의 그룹 <details> 안에 그대로 놓인다.
 * 그래서 패널의 scoped 속성을 받지 않는다: .ag-sub · .ext-note 모양은 패널이 :deep 으로 입힌다.
 */
const props = defineProps<{ widgets: Record<string, any> }>()
const { w, b, setB } = useGuidanceWidgets(props)

// 선택지 표기 = 확장 라디오 그대로(core/anima_guidance.py SMC_MODE_*)
const smcModes = ['Unit-L2', 'Adaptive sign']
const adaptive = computed(() => w._guid_smc_mode === 'Adaptive sign')
const adaptiveTitle =
  '속도 공간 식을 x0 공간에서 같게 계산합니다(σ_t/σ_prev 보정). 원작자는 λ 를 낮추면 어두워짐이 줄었다고 '
  + '보고했습니다. α 0 = 보정 없음.'

// 연결된 Forge 의 sam-extra 가 SMC controller(인자 68-70)를 모르면 안내 — ComfyUI 는 번들 팩이 처리한다
const { capabilities } = useSamExtraCapabilities()
const detailNote = computed(() => detailSuiteNote(capabilities.value))

const cfgModes = ['Preserve incoming', 'APG', 'CWM', 'SMC', 'SMC + CWM']
const smcPresets = [
  'Auto', 'SD1.5 / SD2', 'SDXL', 'SD3 / SD3.5',
  'Flux', 'Qwen-Image', 'Cosmos / Wan', 'Custom',
]
</script>
