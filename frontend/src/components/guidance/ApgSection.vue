<template>
  <label class="ext-check-row">
    <ToggleSwitch :model-value="b('guid_apg_enabled')" @update:model-value="setB('guid_apg_enabled', $event)" size="sm" />
    <span>Enable APG (실험 · CFG &gt; 1)</span>
  </label>
  <template v-if="b('guid_apg_enabled')">
    <div class="ext-row">
      <div class="ext-field"><label>APG eta</label>
        <input type="number" v-model="w._guid_apg_eta" step="0.05" min="-10" max="10" /></div>
      <div class="ext-field"><label>APG norm (0=off)</label>
        <input type="number" v-model="w._guid_apg_norm" step="0.5" min="0" max="50" /></div>
    </div>
    <div class="ext-field"><label>APG momentum (음수 권장 · 0=off)</label>
      <input type="number" v-model="w._guid_apg_momentum" step="0.05" min="-1" max="1" /></div>
    <label class="ext-check-row">
      <ToggleSwitch :model-value="b('guid_apg_autooff')" @update:model-value="setB('guid_apg_autooff', $event)" size="sm" />
      <span>APG 켜지면 PAG rescale 자동 끄기</span>
    </label>
  </template>
</template>

<script setup lang="ts">
import ToggleSwitch from '../ToggleSwitch.vue'
import { useGuidanceWidgets } from './guidanceWidgets'

/**
 * APG — CFG base 그룹의 첫 칸(Adaptive Projected Guidance). 그룹 제목은 AnimaGuidancePanel 이 든다.
 *
 * 조각(fragment) 컴포넌트다 — 감싸는 요소 없이 AnimaGuidancePanel 의 그룹 <details> 안에 그대로 놓인다.
 * 그래서 패널의 scoped 속성을 받지 않는다: .ag-sub · .ext-note 모양은 패널이 :deep 으로 입힌다.
 */
const props = defineProps<{ widgets: Record<string, any> }>()
const { w, b, setB } = useGuidanceWidgets(props)
</script>
