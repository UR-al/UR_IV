<template>
  <label class="ext-check-row">
    <ToggleSwitch :model-value="b('guid_mod_enabled')" @update:model-value="setB('guid_mod_enabled', $event)" size="sm" />
    <span>Enable Anima Modulation Guidance (보조 CLIP-L)</span>
  </label>
  <template v-if="b('guid_mod_enabled')">
    <div class="ext-field"><label>CLIP-L model (models/text_encoder)</label>
      <input type="text" v-model="w._guid_mod_clip_model" placeholder="clip_l.safetensors" /></div>
    <div class="ext-field"><label>Direction weight w</label>
      <input type="number" v-model="w._guid_mod_weight" step="0.05" min="-20" max="20" /></div>
    <div class="ext-row">
      <div class="ext-field"><label>Start block</label>
        <input type="number" v-model="w._guid_mod_start_layer" step="1" min="0" max="63" /></div>
      <div class="ext-field"><label>End block (-1=마지막)</label>
        <input type="number" v-model="w._guid_mod_end_layer" step="1" min="-1" max="63" /></div>
    </div>
    <div class="ext-field"><label>Base CLIP prompt source</label>
      <CustomSelect v-model="w._guid_mod_base_source" :options="['Main positive', 'Custom']" placeholder="Main positive" /></div>
    <div class="ext-field" v-if="w._guid_mod_base_source === 'Custom'">
      <label>Custom base CLIP prompt</label>
      <input type="text" v-model="w._guid_mod_base_prompt" /></div>
    <div class="ext-field"><label>Positive direction prompt</label>
      <input type="text" v-model="w._guid_mod_positive_prompt" /></div>
    <div class="ext-field"><label>Negative direction source</label>
      <CustomSelect v-model="w._guid_mod_negative_source" :options="['Main negative', 'Custom']" placeholder="Main negative" /></div>
    <div class="ext-field" v-if="w._guid_mod_negative_source === 'Custom'">
      <label>Custom negative direction prompt</label>
      <input type="text" v-model="w._guid_mod_negative_prompt" /></div>
    <div class="ext-field"><label>Adapter source</label>
      <CustomSelect v-model="w._guid_mod_adapter_mode"
        :options="['Auto-download official', 'Local file']" placeholder="Auto-download official" /></div>
    <div class="ext-field" v-if="w._guid_mod_adapter_mode === 'Local file'">
      <label>Local adapter path</label>
      <input type="text" v-model="w._guid_mod_adapter_path" /></div>
  </template>
</template>

<script setup lang="ts">
import ToggleSwitch from '../ToggleSwitch.vue'
import CustomSelect from '../CustomSelect.vue'
import { useGuidanceWidgets } from './guidanceWidgets'

/**
 * Anima Modulation Guidance — 보조 CLIP-L 방향. "Adaptive Guidance / CLIP Modulation" 그룹의 끝 칸.
 *
 * 조각(fragment) 컴포넌트다 — 감싸는 요소 없이 AnimaGuidancePanel 의 그룹 <details> 안에 그대로 놓인다.
 * 그래서 패널의 scoped 속성을 받지 않는다: .ag-sub · .ext-note 모양은 패널이 :deep 으로 입힌다.
 */
const props = defineProps<{ widgets: Record<string, any> }>()
const { w, b, setB } = useGuidanceWidgets(props)
</script>
