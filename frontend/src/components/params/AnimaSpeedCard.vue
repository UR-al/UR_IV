<template>
  <details class="ext-card speed-card">
    <summary class="ext-title">Anima SPEED<span v-if="badge" class="speed-summary"
      :class="{ 'speed-alert': messages.length }">{{ badge }}</span></summary>
    <label class="ext-check-row" title="sam-extra 'Anima SPEED' — 초반 스텝을 저해상도로 돌려 빠르게(이미지가 달라진다)">
      <ToggleSwitch v-model="enabled" size="sm" /><span>{{ LABELS.enabled }}</span></label>
    <div class="ext-note">{{ LEAD_NOTE }}</div>
    <div class="ext-field">
      <label>{{ LABELS.mode }}</label>
      <CustomSelect v-model="modeLabel" :options="modeLabels" />
    </div>
    <div class="ext-field">
      <label>{{ LABELS.threshold }}</label>
      <CustomSelect v-model="thresholdLabel" :options="thresholdLabels" />
    </div>
    <div v-if="shows('preset')" class="ext-field">
      <label>{{ LABELS.preset }}</label>
      <CustomSelect v-model="preset" :options="presetOptions" />
    </div>
    <div class="ext-field">
      <label>{{ LABELS.scales }}</label>
      <input type="text" v-model="storeWidgets[WIDGET_IDS.scales]" :placeholder="DEFAULTS.scales" :title="HINTS.scales"
        spellcheck="false" />
    </div>
    <div v-if="shows('delta') || shows('sigma_divisor')" class="ext-row">
      <div v-if="shows('delta')" class="ext-field"><label>{{ LABELS.delta }}</label>
        <input type="number" v-model="storeWidgets[WIDGET_IDS.delta]" :title="HINTS.delta"
          :min="DELTA_RANGE.min" :max="DELTA_RANGE.max" :step="DELTA_RANGE.step" /></div>
      <div v-if="shows('sigma_divisor')" class="ext-field"><label>{{ LABELS.sigma_divisor }}</label>
        <input type="number" v-model="storeWidgets[WIDGET_IDS.sigma_divisor]" :title="HINTS.sigma_divisor"
          :min="DIVISOR_RANGE.min" :max="DIVISOR_RANGE.max" :step="DIVISOR_RANGE.step" /></div>
    </div>
    <div v-if="shows('manual_sigmas')" class="ext-field">
      <label>{{ LABELS.manual_sigmas }}</label>
      <input type="text" v-model="storeWidgets[WIDGET_IDS.manual_sigmas]" :placeholder="DEFAULTS.manual_sigmas"
        :title="HINTS.manual_sigmas" spellcheck="false" />
    </div>
    <label v-if="shows('adaptive')" class="ext-check-row" :title="HINTS.adaptive">
      <ToggleSwitch v-model="adaptive" size="sm" /><span>{{ LABELS.adaptive }}</span></label>
    <label class="ext-check-row" :title="HINTS.hires">
      <ToggleSwitch v-model="hires" size="sm" /><span>{{ LABELS.hires }}</span></label>
    <label class="ext-check-row" :title="HINTS.apply_img2img">
      <ToggleSwitch v-model="applyImg2img" size="sm" /><span>{{ LABELS.apply_img2img }}</span></label>
    <details class="ext-sub">
      <summary>세부값</summary>
      <div class="ext-field">
        <label>{{ LABELS.transform }}</label>
        <CustomSelect v-model="transformLabel" :options="transformLabels" />
        <p class="speed-hint">{{ HINTS.transform }}</p>
      </div>
      <div v-if="shows('spectrum_A')" class="ext-row">
        <div class="ext-field"><label>{{ LABELS.spectrum_A }}</label>
          <input type="number" v-model="storeWidgets[WIDGET_IDS.spectrum_A]" step="any" /></div>
        <div class="ext-field"><label>{{ LABELS.spectrum_beta }}</label>
          <input type="number" v-model="storeWidgets[WIDGET_IDS.spectrum_beta]" step="any" /></div>
      </div>
      <div class="ext-field">
        <label>{{ LABELS.seed }}</label>
        <input type="number" v-model="storeWidgets[WIDGET_IDS.seed]" min="-1" step="1" />
      </div>
    </details>
    <div v-if="messages.length" class="ext-note speed-alert" role="alert">
      <p v-for="message in messages" :key="message" class="speed-line">{{ message }}</p>
      <p class="speed-line">{{ INVALID_NOTE }}</p>
    </div>
    <div class="ext-note">{{ SKIP_NOTE }}</div>
    <div class="ext-note" :class="{ 'speed-warnline': note !== FORGE_NOTE }" role="status">{{ note }}</div>
    <div class="speed-actions">
      <button type="button" class="speed-reset" :class="{ 'speed-muted': atDefaults }" :title="RESET_TITLE"
        @click="resetDefaults">기본값</button>
    </div>
  </details>
</template>

<script setup lang="ts">
/**
 * 파라미터 열 — Anima SPEED 카드(sam-extra "Anima SPEED", Anima38 카드 다음 — 계획 R8). 값은 위젯 스토어(_aspeed_*)에
 * 두고 파이썬 프록시(ui/anima_speed_ui.py)가 같은 문자열을 읽는다. 선택지 칸은 앱 라벨을 보이고 키를 저장한다. 확장이
 * 읽지 않는 칸(threshold·preset 에 따라)은 숨기지만 값은 그대로 두고 보낸다(요청·저장이 그대로 되돌아오게).
 * 검사 줄은 utils/animaSpeed.validate(파이썬 validate 의 거울) — 막지 않고 알린다(Forge 가 'invalid settings' 로 기록하고
 * 평소처럼 샘플링한다). 보낼지(대상·백엔드·게이트)는 파이썬이 정한다 — 여기는 표시만. 새 브리지 이름 없음.
 */
import { computed } from 'vue'
import CustomSelect from '../CustomSelect.vue'
import ToggleSwitch from '../ToggleSwitch.vue'
import { useWidgetStore } from '../../stores/widgetStore.js'
import { widgetFlag } from '../../composables/widgetFlag'
import { useSamExtraCapabilities } from '../../composables/useSamExtraCapabilities'
import { isKrea2Family } from '../../utils/generationFamily'
import {
  DEFAULTS, DELTA_RANGE, DIVISOR_RANGE, FORGE_NOTE, HINTS, INVALID_NOTE, LABELS, LEAD_NOTE, MODE_OPTIONS,
  PRESET_NAMES, RESET_TITLE, SKIP_NOTE, THRESHOLD_OPTIONS, TRANSFORM_OPTIONS, WIDGET_IDS, cardNote, isDefault,
  readSettings, summary, validationMessages, visibleFields, type SpeedKey, type SpeedOption, type VisibleField,
} from '../../utils/animaSpeed'

const store = useWidgetStore()
const storeWidgets = store.widgets as Record<string, any>
const { capabilities } = useSamExtraCapabilities()

const settings = computed(() => readSettings(storeWidgets))

const enabled = widgetFlag(storeWidgets, WIDGET_IDS.enabled)
const hires = widgetFlag(storeWidgets, WIDGET_IDS.hires)
const applyImg2img = widgetFlag(storeWidgets, WIDGET_IDS.apply_img2img)
// 적응형은 기본이 켬 — 스토어에 값이 없거나 빈 칸이면 파이썬처럼 기본값(켬)을 보인다(widgetFlag 는 'true' 만 켬)
const adaptive = computed({
  get: () => settings.value.adaptive,
  set: (value: boolean) => { storeWidgets[WIDGET_IDS.adaptive] = value ? 'true' : 'false' },
})
const visible = computed(() => new Set<VisibleField>(visibleFields(settings.value)))
const shows = (field: VisibleField) => visible.value.has(field)

function optionModel<K extends string>(options: readonly SpeedOption<K>[], key: SpeedKey) {
  return computed({
    get: () => options.find(o => o.key === settings.value[key])?.label ?? options[0]!.label,
    set: (label: string | number) => {
      const picked = options.find(o => o.label === String(label))
      if (picked) storeWidgets[WIDGET_IDS[key]] = picked.key
    },
  })
}
const modeLabel = optionModel(MODE_OPTIONS, 'mode')
const thresholdLabel = optionModel(THRESHOLD_OPTIONS, 'threshold')
const transformLabel = optionModel(TRANSFORM_OPTIONS, 'transform')
const modeLabels = MODE_OPTIONS.map(o => o.label)
const thresholdLabels = THRESHOLD_OPTIONS.map(o => o.label)
const transformLabels = TRANSFORM_OPTIONS.map(o => o.label)
const presetOptions = [...PRESET_NAMES]
const preset = computed({
  get: () => settings.value.preset,
  set: (name: string | number) => { storeWidgets[WIDGET_IDS.preset] = String(name) },
})

const messages = computed(() => validationMessages(settings.value))
const badge = computed(() => summary(settings.value))
const krea2 = computed(() => isKrea2Family(storeWidgets.generation_family_combo))
const note = computed(() => cardNote(capabilities.value, krea2.value))
const atDefaults = computed(() => isDefault(settings.value))

function resetDefaults() {
  for (const [key, value] of Object.entries(DEFAULTS) as Array<[SpeedKey, string]>) {
    storeWidgets[WIDGET_IDS[key]] = value
  }
}
</script>

<style scoped>
.speed-summary { margin-left: var(--sp-2); font-weight: var(--fw-medium); color: var(--state-info-fg); font-size: var(--fs-label); }
.speed-summary.speed-alert { color: var(--state-alert-fg); }
.ext-note { margin: 3px 0 6px; color: var(--text-muted); font-size: var(--fs-label); line-height: 1.45; }
.ext-note.speed-alert { color: var(--state-alert-fg); }
.ext-note.speed-warnline { color: var(--state-warn-fg); }
.speed-line { margin: 0; }
.speed-hint { margin: 4px 0 0; color: var(--text-muted); font-size: var(--fs-label); line-height: 1.4; }
.speed-actions { display: flex; justify-content: flex-end; margin-top: var(--sp-2); }
.speed-reset {
  height: 28px; padding: 0 12px; font-size: var(--fs-meta); font-weight: var(--fw-medium); border-radius: 5px; cursor: pointer;
  background: transparent; border: 1px dashed var(--border); color: var(--text-muted);
}
.speed-reset:hover { border-color: var(--text-muted); color: var(--text-primary); }
.speed-reset.speed-muted { opacity: 0.55; }
</style>
