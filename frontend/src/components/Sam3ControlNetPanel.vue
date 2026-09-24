<template>
  <details class="ext-card cn-card">
    <summary class="ext-title cn-title">
      ControlNet
      <span v-if="enabled" class="cn-badge" :title="summary">{{ summary }}</span>
    </summary>

    <p class="cn-note">
      SAM3 인페인트 패스에 ControlNet 을 주입합니다. Mode 가 <b>Inpaint</b> 이고 Forge 에
      sd_forge_controlnet 이 있을 때만 동작합니다.
    </p>

    <label class="ext-check-row">
      <ToggleSwitch :model-value="enabled" @update:model-value="setBool('cn_enable', $event)" size="sm" />
      <span>Enable ControlNet</span>
    </label>

    <template v-if="enabled">
      <label class="ext-check-row" title="이미 켜 둔 외부 ControlNet 유닛이 있어도 이 설정으로 덮어씁니다">
        <ToggleSwitch :model-value="bool('cn_override_external')" @update:model-value="setBool('cn_override_external', $event)" size="sm" />
        <span>Override external ControlNet units</span>
      </label>

      <div class="ext-field"><label>Model</label>
        <input type="text" v-model="w[id('cn_model')]" placeholder="None" spellcheck="false" /></div>
      <div class="ext-field"><label>Module (전처리기)</label>
        <CustomSelect v-model="w[id('cn_module')]" :options="moduleOptions" placeholder="inpaint_only" /></div>

      <div class="ext-row">
        <div class="ext-field"><label>Weight</label>
          <input type="number" v-model="w[id('cn_weight')]" step="0.05" min="0" max="2" /></div>
        <div class="ext-field"><label>Processor res</label>
          <input type="number" v-model="w[id('cn_processor_res')]" min="0" step="64" /></div>
      </div>
      <div class="ext-row">
        <div class="ext-field"><label>Guidance start</label>
          <input type="number" v-model="w[id('cn_guidance_start')]" step="0.01" min="0" max="1" /></div>
        <div class="ext-field"><label>Guidance end</label>
          <input type="number" v-model="w[id('cn_guidance_end')]" step="0.01" min="0" max="1" /></div>
      </div>

      <div class="ext-field"><label>Control mode</label>
        <CustomSelect v-model="w[id('cn_control_mode')]" :options="controlModeOptions" placeholder="Balanced" /></div>
      <div class="ext-field"><label>Resize mode</label>
        <CustomSelect v-model="w[id('cn_resize_mode')]" :options="resizeModeOptions" placeholder="Crop and Resize" /></div>

      <div class="ext-row">
        <div class="ext-field"><label>Threshold A (-1 = 기본)</label>
          <input type="number" v-model="w[id('cn_threshold_a')]" step="1" /></div>
        <div class="ext-field"><label>Threshold B (-1 = 기본)</label>
          <input type="number" v-model="w[id('cn_threshold_b')]" step="1" /></div>
      </div>

      <label class="ext-check-row">
        <ToggleSwitch :model-value="bool('cn_pixel_perfect')" @update:model-value="setBool('cn_pixel_perfect', $event)" size="sm" />
        <span>Pixel perfect</span>
      </label>

      <button type="button" class="cn-reset" @click="resetAll">기본값으로</button>
    </template>
  </details>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import ToggleSwitch from './ToggleSwitch.vue'
import CustomSelect from './CustomSelect.vue'
import { getProperty } from '../stores/widgetStore.js'
import { SAM3_CN_FIELDS, choiceOptions, isTrue, sam3CnId } from '../utils/sam3ControlNet'

/**
 * SAM3 ControlNet 13필드 패널 (Forge 확장의 SAM3 > ControlNet 아코디언과 1:1).
 *
 * `widgets` 는 widget id(`_sam3_cn_*`) → 문자열 값 객체다.
 *  - T2I: 위젯 스토어(storeWidgets) — 값이 Python 프록시와 동기화되고 설정 저장/복원된다.
 *  - SAM3 Refine·배치 SAM3: `sam3CnDefaults()` 로 만든 로컬 reactive — `sam3CnSettings` 로 전송.
 * 전처리기·control/resize mode 선택지는 Python(core/sam3_controlnet → sam3_args)이 T2I
 * 프록시 items 로 보낸 목록 하나를 세 곳이 같이 쓴다.
 */
const props = defineProps<{ widgets: Record<string, any> }>()
const w = computed(() => props.widgets).value

function id(key: string): string {
  return sam3CnId(key)
}
function bool(key: string): boolean {
  return isTrue(w[id(key)])
}
function setBool(key: string, value: boolean) {
  w[id(key)] = value ? 'true' : 'false'
}

const enabled = computed(() => bool('cn_enable'))

// 선택지가 아직 안 왔으면 getProperty 는 '' — choiceOptions 가 현재 값 하나로 대신한다.
const moduleOptions = computed(() =>
  choiceOptions(getProperty('_sam3_cn_module', 'items'), w[id('cn_module')], 'inpaint_only'))
const controlModeOptions = computed(() =>
  choiceOptions(getProperty('_sam3_cn_control_mode', 'items'), w[id('cn_control_mode')], 'Balanced'))
const resizeModeOptions = computed(() =>
  choiceOptions(getProperty('_sam3_cn_resize_mode', 'items'), w[id('cn_resize_mode')], 'Crop and Resize'))

const summary = computed(() => {
  const model = String(w[id('cn_model')] ?? '').trim()
  const module = String(w[id('cn_module')] ?? '').trim() || 'inpaint_only'
  return model && model !== 'None' ? `${module} · ${model}` : module
})

function resetAll() {
  // 켜 둔 상태는 유지하고 나머지만 확장 기본값으로 되돌린다.
  for (const field of SAM3_CN_FIELDS) {
    if (field.key === 'cn_enable') continue
    w[id(field.key)] = field.def
  }
}
</script>

<style scoped>
.cn-card { padding: 10px 12px; }
.cn-title { display: flex; align-items: center; gap: 8px; }
.cn-badge {
  font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--accent);
  max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.cn-note {
  margin: 4px 0 8px; padding: 6px 8px; border-radius: 6px;
  background: rgba(255, 255, 255, 0.03);
  color: var(--text-muted); font-size: var(--fs-label); line-height: 1.5;
}
.ext-field { margin-bottom: 8px; }
.ext-field label {
  display: block; margin-bottom: 3px;
  font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-muted);
}
.ext-field input { width: 100%; box-sizing: border-box; }
.ext-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.ext-check-row {
  display: flex; align-items: center; gap: 6px; max-width: 100%;
  margin-bottom: 6px; cursor: pointer;
  font-size: var(--fs-label); color: var(--text-secondary);
}
.ext-check-row span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cn-reset {
  height: 26px; padding: 0 10px; margin-top: 4px;
  background: transparent; border: 1px dashed var(--border); border-radius: 5px;
  color: var(--text-muted); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer;
}
.cn-reset:hover { border-color: var(--text-muted); color: var(--text-primary); }
</style>
