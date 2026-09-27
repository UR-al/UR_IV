<template>
  <details class="ext-card dora-card">
    <summary class="ext-title">DoRA 추론 방식<span v-if="summaryText" class="dora-summary">{{ summaryText }}</span></summary>
    <label class="ext-check-row"><ToggleSwitch v-model="enabled" size="sm" /><span>DoRA 추론 방식 사용</span></label>
    <div class="ext-field">
      <label>계산 방식</label>
      <CustomSelect v-model="modeLabel" :options="modeLabels" />
    </div>
    <div class="ext-field">
      <label>끼워 넣은 블록 (상향 블록 변환 시)</label>
      <CustomSelect v-model="insertedLabel" :options="insertLabels" />
    </div>
    <div v-if="weakFields" class="ext-row">
      <div class="ext-field"><label>약한 복사 강도</label>
        <input type="number" v-model="storeWidgets[WIDGET_IDS.weak_strength]"
          :min="WEAK_STRENGTH_RANGE.min" :max="WEAK_STRENGTH_RANGE.max" :step="WEAK_STRENGTH_RANGE.step" /></div>
      <div class="ext-field"><label>약한 복사 범위</label>
        <CustomSelect v-model="scopeLabel" :options="scopeLabels" /></div>
    </div>
    <label class="ext-check-row" title="Forge 는 img2img 탭에 아코디언이 따로 있다 — 사용자 Forge 설정은 꺼짐">
      <ToggleSwitch v-model="applyImg2img" size="sm" /><span>I2I·인페인트에도 적용</span></label>
    <div class="ext-note" :class="{ 'dora-alert': status.status === 'missing' }">{{ status.text }}</div>
    <div class="dora-actions">
      <button type="button" class="dora-reset" title="Forge 설정(txt2img)과 같은 앱 기본값으로" @click="resetDefaults">기본값</button>
    </div>
  </details>
</template>

<script setup lang="ts">
/**
 * 파라미터 열 — DoRA 추론 방식 카드(sam-extra DoRA Inference Mode). 값은 위젯 스토어(_dora_*)에 두고 파이썬
 * 프록시(ui/dora_infer_mode_ui.py)가 같은 문자열을 읽는다. 선택 칸은 확장 라벨을 보이고 키를 저장한다.
 * 보낼지(게이트·순정 생략·ComfyUI)는 파이썬이 정한다 — 여기는 표시만. 새 브리지 액션은 없다.
 */
import { computed, type WritableComputedRef } from 'vue'
import CustomSelect from '../CustomSelect.vue'
import ToggleSwitch from '../ToggleSwitch.vue'
import { useWidgetStore } from '../../stores/widgetStore.js'
import { widgetFlag } from '../../composables/widgetFlag'
import { useSamExtraCapabilities } from '../../composables/useSamExtraCapabilities'
import {
  DEFAULTS, INSERT_OPTIONS, MODE_OPTIONS, SCOPE_OPTIONS, WEAK_STRENGTH_RANGE, WIDGET_IDS,
  cardStatus, keyOf, labelOf, readValues, showWeakFields, summary,
  type DoraOption, type DoraWidgetKey,
} from '../../utils/doraMode'

const storeWidgets = useWidgetStore().widgets as Record<string, any>
const { capabilities } = useSamExtraCapabilities()

const enabled = widgetFlag(storeWidgets, WIDGET_IDS.enabled)
const applyImg2img = widgetFlag(storeWidgets, WIDGET_IDS.apply_img2img)

const values = computed(() => readValues(storeWidgets))

function labelModel(key: DoraWidgetKey, options: readonly DoraOption[]): WritableComputedRef<string> {
  return computed({
    get: () => labelOf(options, values.value[key]),
    set: (label: string | number) => { storeWidgets[WIDGET_IDS[key]] = keyOf(options, String(label)) },
  })
}

const modeLabel = labelModel('mode', MODE_OPTIONS)
const insertedLabel = labelModel('inserted', INSERT_OPTIONS)
const scopeLabel = labelModel('weak_scope', SCOPE_OPTIONS)
const modeLabels = MODE_OPTIONS.map(o => o.label)
const insertLabels = INSERT_OPTIONS.map(o => o.label)
const scopeLabels = SCOPE_OPTIONS.map(o => o.label)

const weakFields = computed(() => showWeakFields(values.value))
const summaryText = computed(() => summary(values.value))
const status = computed(() => cardStatus(capabilities.value, values.value))

function resetDefaults() {
  for (const [key, value] of Object.entries(DEFAULTS) as Array<[DoraWidgetKey, string]>) {
    storeWidgets[WIDGET_IDS[key]] = value
  }
}
</script>

<style scoped>
.dora-summary { margin-left: var(--sp-2); font-weight: var(--fw-medium); color: var(--state-info-fg); font-size: var(--fs-label); }
.ext-note { margin: 3px 0 6px; color: var(--text-muted); font-size: var(--fs-label); line-height: 1.45; }
.ext-note.dora-alert { color: var(--state-alert-fg); }
.dora-actions { display: flex; justify-content: flex-end; margin-top: var(--sp-2); }
.dora-reset {
  height: 28px; padding: 0 12px; font-size: var(--fs-meta); font-weight: var(--fw-medium); border-radius: 5px; cursor: pointer;
  background: transparent; border: 1px dashed var(--border); color: var(--text-muted);
}
.dora-reset:hover { border-color: var(--text-muted); color: var(--text-primary); }
</style>
