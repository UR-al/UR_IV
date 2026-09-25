<template>
  <label class="ext-check-row">
    <ToggleSwitch :model-value="b('guid_enabled')" @update:model-value="setB('guid_enabled', $event)" size="sm" />
    <span>Enable Perturbation Guidance</span>
  </label>

  <template v-if="b('guid_enabled')">
    <div class="ext-field"><label>Attention method (PAG/SEG는 택1)</label>
      <CustomSelect v-model="w._guid_attn_method" :options="['PAG', 'SEG', 'None']" placeholder="PAG" /></div>
    <div class="ext-row">
      <div class="ext-field"><label>Attn Scale (cond−weak 배율)</label>
        <input type="number" v-model="w._guid_scale" step="0.1" min="0" max="100" /></div>
      <div class="ext-field"><label>Perturbation strength (1=전체)</label>
        <input type="number" v-model="w._guid_official_strength" step="0.01" min="0" max="1" /></div>
    </div>
    <div class="ext-field" v-if="w._guid_attn_method === 'SEG'">
      <label>SEG query blur sigma (&gt;9999 = uniform)</label>
      <input type="number" v-model="w._guid_seg_sigma" step="1" min="0" max="10000" /></div>
    <div class="ext-row">
      <div class="ext-field"><label>Block indices (기본 18)</label>
        <input type="text" v-model="w._guid_block_indices" placeholder="18 또는 18-20" /></div>
      <div class="ext-field"><label>Head indices (빈칸=전체)</label>
        <input type="text" v-model="w._guid_head_indices" placeholder="0,2,4-7" /></div>
    </div>

    <label class="ext-check-row">
      <ToggleSwitch :model-value="b('guid_slg_on')" @update:model-value="setB('guid_slg_on', $event)" size="sm" />
      <span>Enable SLG (skip layers · PAG/SEG와 병용 가능)</span>
    </label>
    <div class="ext-row" v-if="b('guid_slg_on')">
      <div class="ext-field"><label>SLG scale</label>
        <input type="number" v-model="w._guid_slg_scale" step="0.1" min="0" max="15" /></div>
      <div class="ext-field"><label>SLG skip blocks</label>
        <input type="text" v-model="w._guid_slg_blocks" placeholder="18" /></div>
    </div>

    <div class="ext-row">
      <div class="ext-field"><label>Start percent</label>
        <input type="number" v-model="w._guid_start_percent" step="0.001" min="0" max="1" /></div>
      <div class="ext-field"><label>End percent</label>
        <input type="number" v-model="w._guid_end_percent" step="0.001" min="0" max="1" /></div>
    </div>
    <div class="ext-row">
      <div class="ext-field"><label>Rescale (과대비 억제)</label>
        <input type="number" v-model="w._guid_rescale" step="0.01" min="0" max="1" /></div>
      <div class="ext-field"><label>Rescale mode</label>
        <CustomSelect v-model="w._guid_rescale_mode" :options="['full', 'partial']" placeholder="full" /></div>
    </div>

    <details class="ag-sub">
      <summary>Legacy Soft/Approx 호환</summary>
      <label class="ext-check-row">
        <ToggleSwitch :model-value="b('guid_legacy_attn')" @update:model-value="setB('guid_legacy_attn', $event)" size="sm" />
        <span>기존 Soft PAG / SEG-Approx 사용</span>
      </label>
      <div class="ext-field" v-if="b('guid_legacy_attn')"><label>Legacy strength</label>
        <input type="number" v-model="w._guid_legacy_strength" step="0.01" min="0" max="1" /></div>
    </details>
  </template>
</template>

<script setup lang="ts">
import ToggleSwitch from '../ToggleSwitch.vue'
import CustomSelect from '../CustomSelect.vue'
import { useGuidanceWidgets } from './guidanceWidgets'

/**
 * PAG / SEG / SLG — attention perturbation guidance(확장 anima_safe_pag.py 의 perturbation 인자).
 * 원본: iljung1106/comfyui-anima-safe-pag. 그룹 제목은 AnimaGuidancePanel 이 든다.
 * 원본 노드에 있는 칸의 범위·step 은 원본 입력과 같다(origin: iljung1106/comfyui-anima-safe-pag@905b0107:
 * __init__.py:201-207): scale 0~100 step 0.1, perturbation_strength 0~1 step 0.01, start/end 0~1 step 0.001,
 * rescale 0~1 step 0.01. guidanceOriginInputs.test.ts 가 지킨다(scale 은 PagSection.test.ts 도).
 * SEG sigma · SLG · Legacy strength 는 원본 노드에 없는 확장 기능이라 확장 범위를 그대로 둔다.
 *
 * 조각(fragment) 컴포넌트다 — 감싸는 요소 없이 AnimaGuidancePanel 의 그룹 <details> 안에 그대로 놓인다.
 * 그래서 패널의 scoped 속성을 받지 않는다: .ag-sub · .ext-note 모양은 패널이 :deep 으로 입힌다.
 */
const props = defineProps<{ widgets: Record<string, any> }>()
const { w, b, setB } = useGuidanceWidgets(props)
</script>
