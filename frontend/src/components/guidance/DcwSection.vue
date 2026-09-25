<template>
  <label class="ext-check-row">
    <ToggleSwitch :model-value="b('guid_dcw_enabled')" @update:model-value="setB('guid_dcw_enabled', $event)" size="sm" />
    <span>Enable DCW (post-CFG wavelet correction)</span>
  </label>
  <template v-if="b('guid_dcw_enabled')">
    <div class="ext-row">
      <div class="ext-field"><label :title="lowTitle">DCW lambda low (초반 저주파 · 원본 기본 0.05)</label>
        <input type="number" v-model="w._guid_dcw_lambda_low" step="0.005" min="-0.5" max="0.5" /></div>
      <div class="ext-field"><label :title="highTitle">DCW lambda high (후반 고주파 · 원본 기본 0.01)</label>
        <input type="number" v-model="w._guid_dcw_lambda_high" step="0.001" min="-0.3" max="0.3" /></div>
    </div>
    <div class="ext-note">양수 = x_t 쪽으로(low: 신호↑ · high: 디테일↑), 음수 = 반대, 0 = 그 대역 보정 끔.
      원본 안내: DDPM/EDM 은 low 0.04–0.07 · high 0.008–0.015, Flow 계열(Anima 포함)은 그 약 2배에서 시작.</div>
  </template>
</template>

<script setup lang="ts">
import ToggleSwitch from '../ToggleSwitch.vue'
import { useGuidanceWidgets } from './guidanceWidgets'

/**
 * DCW — post-CFG wavelet 보정(원본: namemechan/ComfyUI-DCW). "DCW / RDC / DAVE / CNS" 그룹의 첫 칸.
 * 칸 범위·step 은 원본 노드 입력과 같다: lambda_l ±0.5 step 0.005, lambda_h ±0.3 step 0.001
 * (origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:637-667 — 숫자만 옮겼다, GPL 코드는 복사하지 않는다).
 * 안내 문구는 원본 툴팁의 뜻을 우리말로 새로 쓴 것이다. guidanceOriginInputs.test.ts 가 범위를 지킨다.
 *
 * 조각(fragment) 컴포넌트다 — 감싸는 요소 없이 AnimaGuidancePanel 의 그룹 <details> 안에 그대로 놓인다.
 * 그래서 패널의 scoped 속성을 받지 않는다: .ag-sub · .ext-note 모양은 패널이 :deep 으로 입힌다.
 */
const props = defineProps<{ widgets: Record<string, any> }>()
const { w, b, setB } = useGuidanceWidgets(props)

const lowTitle = '저주파(LL) 보정 — 주로 초반 스텝에 작용. 범위 ±0.5, step 0.005 (원본 노드와 같음)'
const highTitle = '고주파(HH) 보정 — 주로 후반 스텝에 작용. 범위 ±0.3, step 0.001 (원본 노드와 같음)'
</script>
