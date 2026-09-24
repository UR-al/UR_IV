<template>
  <details class="ext-card">
    <summary class="ext-title">ADetailer</summary>
    <label class="ext-check-row"><ToggleSwitch v-model="ad_enabled" size="sm" /><span>ADetailer 활성화</span></label>
    <!-- Slot 1 -->
    <div class="ext-sub-title">Slot 1</div>
    <AdSlotFields :slot-no="1" :model-items="modelItems" prompt-placeholder="ADetailer prompt..." />

    <!-- Slot 2 -->
    <details class="ext-sub">
      <summary>Slot 2</summary>
      <AdSlotFields :slot-no="2" :model-items="modelItems" prompt-placeholder="Slot 2 prompt..." />
    </details>
  </details>
</template>

<script setup lang="ts">
/**
 * 파라미터 열 — ADetailer 카드. 슬롯 필드는 AdSlotFields 하나로 그린다(App.vue 분할 ④).
 * 모델 목록은 App 이 adetailerModelsReady 로 받아 내려준다(부팅 요청 시점을 App 이 정한다).
 */
import ToggleSwitch from '../ToggleSwitch.vue'
import AdSlotFields from './AdSlotFields.vue'
import { useWidgetStore } from '../../stores/widgetStore.js'
import { widgetFlag } from '../../composables/widgetFlag'

defineProps<{ modelItems: string[] }>()

const ad_enabled = widgetFlag(useWidgetStore().widgets, 'adetailer_group')
</script>
