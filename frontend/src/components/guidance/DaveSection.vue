<template>
  <label class="ext-check-row">
    <ToggleSwitch :model-value="b('guid_dave_enabled')" @update:model-value="setB('guid_dave_enabled', $event)" size="sm" />
    <span>Enable DAVE (block DC attenuation)</span>
  </label>
  <template v-if="b('guid_dave_enabled')">
    <div class="ext-row">
      <div class="ext-field"><label :title="strengthTitle">DAVE strength (원본 기본 0.30 · 0 = 끔)</label>
        <input type="number" v-model="w._guid_dave_strength" step="0.01" min="0" max="1" /></div>
      <div class="ext-field"><label :title="tauTitle">DAVE tau (초반 스텝 비율 · ≤ 0.10 권장 · 0 = 모든 스텝)</label>
        <input type="number" v-model="w._guid_dave_tau" step="0.01" min="0" max="1" /></div>
    </div>
    <div class="ext-field"><label>DAVE block indices (빈칸 = 8-18, 원본 기본 마스크)</label>
      <input type="text" v-model="w._guid_dave_blocks" placeholder="8-18" /></div>
    <div class="ext-note">strength 는 '클수록 강함'이 아닙니다 — 구도 다양성은 낮은 값(0.05–0.2)부터 찾고, 0.30 이 무난한
      기본, DC 를 가장 많이 빼려면 tau 0.10 에서 약 0.80. tau 를 넓히는 쪽이 글자·손을 더 망가뜨립니다(0.15 부터).</div>
  </template>
</template>

<script setup lang="ts">
import ToggleSwitch from '../ToggleSwitch.vue'
import { useGuidanceWidgets } from './guidanceWidgets'

/**
 * DAVE — 블록 DC 감쇠(원본: sorryhyun/ComfyUI-Anima-DAVE, MIT). "DCW / RDC / DAVE / CNS" 그룹.
 * 칸 범위·step 은 원본 노드 입력과 같다: strength 0~1 step 0.01, tau 0~1 step 0.01
 * (origin: sorryhyun/ComfyUI-Anima-DAVE@83143e8d:nodes.py:121-147). 원본의 마스크 선택은 블록 텍스트 칸이
 * 대신하고(호스트 차이), 빈칸은 원본 기본 마스크 dave_alpha.npz 와 같은 8-18 로 보낸다(nodes.py:119,
 * core/anima_guidance.py build_args). 안내는 원본 README.md:50-64 의 사용 안내를 줄여 옮긴 것이다.
 * guidanceOriginInputs.test.ts 가 범위를 지킨다.
 *
 * 조각(fragment) 컴포넌트다 — 감싸는 요소 없이 AnimaGuidancePanel 의 그룹 <details> 안에 그대로 놓인다.
 * 그래서 패널의 scoped 속성을 받지 않는다: .ag-sub · .ext-note 모양은 패널이 :deep 으로 입힌다.
 */
const props = defineProps<{ widgets: Record<string, any> }>()
const { w, b, setB } = useGuidanceWidgets(props)

const strengthTitle = 'DC 제거량 s = 1 − α = strength · w(블록). 0 = 끔, 범위 0~1 step 0.01 (원본 노드와 같음)'
const tauTitle =
  '초반(σ 가 높은) 스텝 중 DAVE 가 켜지는 비율 — 스케줄 인덱스 < round(tau · 스텝 수)(최소 1스텝). '
  + '0 = 모든 스텝, 범위 0~1 step 0.01 (원본 노드와 같음)'
</script>
