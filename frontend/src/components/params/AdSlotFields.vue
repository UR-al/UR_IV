<template>
  <label class="ext-check-row"><ToggleSwitch v-model="slotEnabled" size="sm" /><span>Slot {{ slotNo }} 활성화</span></label>
  <div class="ext-field"><label>Model</label>
    <CustomSelect v-model="storeWidgets[k('model')]" :options="modelItems" placeholder="AD Model..." /></div>
  <div class="ext-field"><label>Prompt</label>
    <input type="text" v-model="storeWidgets[k('prompt')]" :placeholder="promptPlaceholder" /></div>
  <div class="ext-field"><label>Negative Prompt</label>
    <input type="text" v-model="storeWidgets[k('neg')]" placeholder="AD negative..." /></div>
  <div class="ext-row">
    <div class="ext-field"><label>Confidence</label><input type="number" v-model="storeWidgets[k('confidence')]" step="0.05" min="0" max="1" /></div>
    <div class="ext-field"><label>Denoise</label><input type="number" v-model="storeWidgets[k('denoise')]" step="0.05" min="0" max="1" /></div>
  </div>
  <div class="ext-row">
    <div class="ext-field"><label>Mask Blur</label><input type="number" v-model="storeWidgets[k('mask_blur')]" min="0" /></div>
    <div class="ext-field"><label>Padding</label><input type="number" v-model="storeWidgets[k('padding')]" min="0" /></div>
    <div class="ext-field"><label>Dilate/Erode</label><input type="number" v-model="storeWidgets[k('dilate_erode')]" /></div>
  </div>
  <div class="ext-field"><label>마스크 병합</label>
    <CustomSelect v-model="storeWidgets[k('mask_merge')]" :options="['None', 'Merge', 'Merge and Invert']" placeholder="None" /></div>
  <!-- Separate settings -->
  <label class="ext-check-row"><ToggleSwitch :model-value="isOn('use_inp_size')" @update:model-value="setOn('use_inp_size', $event)" size="sm" /><span>별도 Inpaint 크기</span></label>
  <div class="ext-row" v-if="isOn('use_inp_size')">
    <div class="ext-field"><label>너비</label><input type="number" v-model="storeWidgets[k('inp_w')]" /></div>
    <div class="ext-field"><label>높이</label><input type="number" v-model="storeWidgets[k('inp_h')]" /></div>
  </div>
  <label class="ext-check-row"><ToggleSwitch :model-value="isOn('use_steps')" @update:model-value="setOn('use_steps', $event)" size="sm" /><span>별도 Steps</span></label>
  <div class="ext-row" v-if="isOn('use_steps')">
    <div class="ext-field"><label>스텝</label><input type="number" v-model="storeWidgets[k('steps')]" min="1" /></div>
  </div>
  <label class="ext-check-row"><ToggleSwitch :model-value="isOn('use_cfg')" @update:model-value="setOn('use_cfg', $event)" size="sm" /><span>별도 CFG</span></label>
  <div class="ext-row" v-if="isOn('use_cfg')">
    <div class="ext-field"><label>CFG</label><input type="number" v-model="storeWidgets[k('cfg')]" step="0.5" /></div>
  </div>
  <label class="ext-check-row"><ToggleSwitch :model-value="isOn('use_sampler')" @update:model-value="setOn('use_sampler', $event)" size="sm" /><span>별도 Sampler</span></label>
  <div class="ext-row" v-if="isOn('use_sampler')">
    <div class="ext-field"><label>Sampler</label>
      <CustomSelect v-model="storeWidgets[k('sampler')]" :options="samplerItems" placeholder="Sampler" /></div>
    <div class="ext-field"><label>Scheduler</label>
      <CustomSelect v-model="storeWidgets[k('scheduler')]" :options="schedulerItems" placeholder="Scheduler" /></div>
  </div>
  <label class="ext-check-row"><ToggleSwitch :model-value="isOn('use_ckpt')" @update:model-value="setOn('use_ckpt', $event)" size="sm" /><span>별도 Checkpoint</span></label>
  <div class="ext-field" v-if="isOn('use_ckpt')"><label>Checkpoint</label>
    <CustomSelect v-model="storeWidgets[k('ckpt')]" :options="adCheckpointItems" placeholder="Use same checkpoint" /></div>
  <label class="ext-check-row"><ToggleSwitch :model-value="isOn('use_vae')" @update:model-value="setOn('use_vae', $event)" size="sm" /><span>별도 VAE</span></label>
  <div class="ext-field" v-if="isOn('use_vae')"><label>VAE</label>
    <CustomSelect v-model="storeWidgets[k('vae')]" :options="adVaeItems" placeholder="Use same VAE" /></div>
</template>

<script setup lang="ts">
/**
 * ADetailer 슬롯 하나의 필드 — 슬롯 1·2 는 위젯 키의 `_ad_s1_` / `_ad_s2_` 만 다른 같은 모양이라
 * 한 컴포넌트로 그린다(예전 App.vue 에선 23개 필드를 두 번 베껴 적었다). 키 규칙은
 * `adSlotKey`(Python ADetailer 프록시 widget_id 와 같다 — ui/generator_ui_setup.py).
 * 슬롯을 감싸는 모양(슬롯 1 은 소제목, 슬롯 2 는 접는 <details>)은 AdetailerCard 가 정한다.
 */
import { computed } from 'vue'
import CustomSelect from '../CustomSelect.vue'
import ToggleSwitch from '../ToggleSwitch.vue'
import { useWidgetStore } from '../../stores/widgetStore.js'
import { useParamItems } from '../../composables/useParamItems'
import { adSlotGroupKey, adSlotKey, type AdSlotField } from '../../utils/adSlotKeys'

const props = defineProps<{
  /** 1 | 2 — utils/adSlotKeys 의 AdSlotNo 와 같다(매크로가 다른 파일의 타입을 풀지 않아도 되게 여기 적는다) */
  slotNo: 1 | 2
  /** ADetailer 모델 목록(adetailerModelsReady) — App 이 받아 내려준다 */
  modelItems: string[]
  promptPlaceholder: string
}>()

const storeWidgets = useWidgetStore().widgets
const { samplerItems, schedulerItems, adCheckpointItems, adVaeItems } = useParamItems()

const k = (field: AdSlotField) => adSlotKey(props.slotNo, field)
const isOn = (field: AdSlotField) => storeWidgets[k(field)] === 'true'
function setOn(field: AdSlotField, on: boolean) { storeWidgets[k(field)] = on ? 'true' : 'false' }
const slotEnabled = computed({
  get: () => storeWidgets[adSlotGroupKey(props.slotNo)] === 'true',
  set: (v: boolean) => { storeWidgets[adSlotGroupKey(props.slotNo)] = v ? 'true' : 'false' },
})
</script>
