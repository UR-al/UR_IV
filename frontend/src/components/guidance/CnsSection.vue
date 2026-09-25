<template>
  <label class="ext-check-row">
    <ToggleSwitch :model-value="b('guid_cns_enabled')" @update:model-value="setB('guid_cns_enabled', $event)" size="sm" />
    <span>Enable CNS (wavelet noise 재색칠)</span>
  </label>
  <template v-if="b('guid_cns_enabled')">
    <div class="ext-row">
      <div class="ext-field"><label :title="strengthTitle">CNS strength (원본 기본 1.0)</label>
        <input type="number" v-model="w._guid_cns_strength" step="0.05" min="0" max="1" /></div>
      <div class="ext-field"><label :title="powerTitle">CNS gamma power (원본 기본 0.5)</label>
        <input type="number" v-model="w._guid_cns_gamma_power" step="0.05" min="0.1" max="2" /></div>
    </div>
    <div class="ext-field"><label :title="scaleTitle">CNS gamma scale (기본 2.0 · Anima+cfg_pp 권장 3.0)</label>
      <input type="number" v-model="w._guid_cns_gamma_scale" step="0.1" min="0.1" max="25" /></div>
    <div class="ext-note">ancestral·SDE 샘플러가 스텝마다 넣는 노이즈만 바꿉니다 — euler 같은 ODE 샘플러에서는 효과가
      없습니다. 원본 README 는 Anima + euler_ancestral_cfg_pp 에 gamma scale 3.0 을 권장합니다.</div>
  </template>
</template>

<script setup lang="ts">
import ToggleSwitch from '../ToggleSwitch.vue'
import { useGuidanceWidgets } from './guidanceWidgets'

/**
 * CNS — wavelet 노이즈 재색칠(원본: namemechan/comfyui-cns_sampler_patch). "DCW / RDC / DAVE / CNS" 그룹의 끝 칸.
 * 칸 범위·step 은 원본 노드 입력과 같다: strength 0~1 step 0.05, gamma_power 0.1~2 step 0.05,
 * gamma_scale 0.1~25 step 0.1, 기본 2.0 (origin: namemechan/comfyui-cns_sampler_patch@42278b13:
 * cns_sampler_patch.py:396-437 — 숫자만 옮겼다, GPL 코드는 복사하지 않는다). 3.0 은 원본 README.md:112 표의
 * 'Flux / Anima + euler_ancestral_cfg_pp' 권장값이다. guidanceOriginInputs.test.ts 가 범위를 지킨다.
 *
 * 조각(fragment) 컴포넌트다 — 감싸는 요소 없이 AnimaGuidancePanel 의 그룹 <details> 안에 그대로 놓인다.
 * 그래서 패널의 scoped 속성을 받지 않는다: .ag-sub · .ext-note 모양은 패널이 :deep 으로 입힌다.
 */
const props = defineProps<{ widgets: Record<string, any> }>()
const { w, b, setB } = useGuidanceWidgets(props)

// 툴팁은 사실만 적는다 — 범위 · 원본 기본값 · 값을 올리고 내릴 때의 방향(계산에서 읽은 것). 원본(GPL) 툴팁 문장을
// 옮기지 않는다(번역도 하지 않는다). 권장 시작값은 위 라벨의 README 권장값만 쓴다.
const strengthTitle = '0 = 흰 노이즈 그대로, 1 = 색칠한 노이즈만(원본 기본 1.0), 사이 값은 둘을 섞음. 범위 0~1 step 0.05'
const powerTitle =
  '대역별 부족분(1 − 에너지 비율)에 거는 지수. 클수록 대역 간 차이가 커지고, 작을수록 대역이 고르게 되어 흰 노이즈에 '
  + '가까워짐. 원본 기본 0.5, 범위 0.1~2 step 0.05'
const scaleTitle =
  '대역 에너지 비율을 이 값으로 나눈 뒤 부족분을 구함. 작을수록 색칠이 강해지고, 클수록 흰 노이즈에 가까워짐. '
  + '원본 기본 2.0, 범위 0.1~25 step 0.1'
</script>
