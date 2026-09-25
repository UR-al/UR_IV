<template>
  <label class="ext-check-row">
    <ToggleSwitch :model-value="b('guid_adg_enabled')" @update:model-value="setB('guid_adg_enabled', $event)" size="sm" />
    <span>Enable Adaptive Guidance (후반 uncond 생략)</span>
  </label>
  <div class="ext-row" v-if="b('guid_adg_enabled')">
    <div class="ext-field"><label>Skip after</label>
      <input type="number" v-model="w._guid_adg_start" step="0.01" min="0" max="1" /></div>
    <div class="ext-field"><label>Keep every N (0=항상 생략)</label>
      <input type="number" v-model="w._guid_adg_interval" step="1" min="0" max="10" /></div>
  </div>
</template>

<script setup lang="ts">
import ToggleSwitch from '../ToggleSwitch.vue'
import { useGuidanceWidgets } from './guidanceWidgets'

/**
 * Adaptive Guidance — 후반 uncond 생략. "Adaptive Guidance / CLIP Modulation" 그룹의 첫 칸.
 *
 * 조각(fragment) 컴포넌트다 — 감싸는 요소 없이 AnimaGuidancePanel 의 그룹 <details> 안에 그대로 놓인다.
 * 그래서 패널의 scoped 속성을 받지 않는다: .ag-sub · .ext-note 모양은 패널이 :deep 으로 입힌다.
 */
const props = defineProps<{ widgets: Record<string, any> }>()
const { w, b, setB } = useGuidanceWidgets(props)
</script>
