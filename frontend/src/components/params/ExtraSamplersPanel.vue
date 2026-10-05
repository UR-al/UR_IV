<template>
  <div v-if="hints.length" class="xsa-hints">
    <p v-for="hint in hints" :key="hint" class="xsa-hint">{{ hint }}</p>
  </div>
  <section v-if="passes.length" class="xsa-fields" aria-label="ER SDE 설정">
    <div class="xsa-head">
      <span class="xsa-title">ER SDE 설정 (sam-extra Extra Samplers)</span>
      <span class="xsa-subtitle">{{ subtitleText }}</span>
      <button type="button" class="xsa-reset" :disabled="atDefaults"
        title="확장 기본값으로 — 차수 3 · η 1 · 잡음 구간 끔(0.2–0.8)" @click="resetDefaults">기본값</button>
    </div>
    <div class="ext-row">
      <div class="ext-field">
        <label>풀이 차수 (max stage)</label>
        <div class="xsa-stages" role="radiogroup" aria-label="풀이 차수">
          <button v-for="option in STAGE_OPTIONS" :key="option" type="button" role="radio"
            :aria-checked="stage === option" :class="{ active: stage === option }"
            @click="storeWidgets[WIDGET_IDS.max_stage] = option">{{ option }}</button>
        </div>
      </div>
      <div class="ext-field"><label>η (eta)</label>
        <input type="number" v-model="storeWidgets[WIDGET_IDS.eta]" :disabled="!etaUsed"
          :min="ETA_RANGE.min" :max="ETA_RANGE.max" :step="ETA_RANGE.step" /></div>
    </div>
    <p class="xsa-note">{{ STAGE_HELP }}</p>
    <p class="xsa-note">{{ etaUsed ? ETA_HELP : ODE_NOTE }}</p>
    <div class="ext-row xsa-window" :class="{ off: !etaUsed }">
      <div class="ext-field">
        <label class="ext-check-row" :aria-disabled="!etaUsed">
          <ToggleSwitch :model-value="settings.noise_window" size="sm" @update:model-value="setWindow" /><span>잡음 구간</span>
        </label>
      </div>
      <div class="ext-field"><label>시작</label>
        <input type="number" v-model="storeWidgets[WIDGET_IDS.noise_start]" :disabled="!etaUsed" title="샘플링 진행률 0–1"
          :min="NOISE_RANGE.min" :max="NOISE_RANGE.max" :step="NOISE_RANGE.step" /></div>
      <div class="ext-field"><label>끝</label>
        <input type="number" v-model="storeWidgets[WIDGET_IDS.noise_end]" :disabled="!etaUsed" title="샘플링 진행률 0–1"
          :min="NOISE_RANGE.min" :max="NOISE_RANGE.max" :step="NOISE_RANGE.step" /></div>
    </div>
    <p class="xsa-note">{{ WINDOW_HELP }}</p>
    <p v-if="emptyWindow" class="xsa-note xsa-warnline" role="alert">{{ WINDOW_EMPTY_WARNING }}</p>
    <p v-if="windowDropped" class="xsa-note xsa-warnline" role="status">{{ WINDOW_UNSUPPORTED_TEXT }}</p>
    <p v-if="note" class="xsa-note xsa-warnline" role="status">{{ note }}</p>
    <p v-else class="xsa-note">{{ paramsLine }}</p>
  </section>
  <p v-else-if="!atDefaults" class="xsa-hidden">
    <span>{{ hiddenText }}</span>
    <button type="button" class="xsa-reset" title="확장 기본값으로 — 차수 3 · η 1 · 잡음 구간 끔(0.2–0.8)"
      @click="resetDefaults">기본값</button>
  </p>
</template>

<script setup lang="ts">
/**
 * 파라미터 카드의 Sampler/Scheduler 줄 바로 아래(계획 R8 — Extra Schedulers 값 칸 앞) — Extra Samplers(sam-extra) ER SDE 값
 * 칸. 다음 요청의 어느 패스(메인·Hires·ADetailer·SAM3)가 ER SDE (Reverse-time)·(ODE)·(Tunable) 을 쓸 때만 보인다(확장은 한
 * 생성에 값 한 벌 — 그 패스들이 같은 값을 쓴다). 풀이 차수는 셋 모두, η 와 잡음 구간(체크 + 시작·끝, 샘플링 진행률 0–1)은
 * Reverse-time·Tunable 만 읽는다 — ODE 만 쓰면 그 칸들은 꺼진다. 구간을 켰는데 시작 ≥ 끝이면 잡음이 없다는 경고 줄, 연결된
 * Forge 의 sam-extra 가 인자 2개(0.31·0.32)라 구간을 버리면 그 줄(계획 X10) — 막지는 않는다. 값은 위젯 스토어(_xsamp_*)에
 * 두고 파이썬 프록시(ui/extra_samplers_ui.py)가 같은 문자열을 읽는다 — 보낼지(스냅샷·패스·백엔드)는 파이썬이 정한다
 * (core/extra_samplers.plan). 칸이 안 보일 때 값이 기본값이 아니면 한 줄로 알린다(값은 그대로 남는다).
 *
 * Sampler 줄 바로 아래 힌트(v0.33.0 샘플러 설명, CFG++ 의 CFG 경고)는 칸과 상관없이 보인다. 새 브리지 이름은 없다.
 * App.vue 를 거치지 않고 스토어를 직접 읽는다(ParamsBasicCard 안에서 그린다).
 */
import { computed } from 'vue'
import ToggleSwitch from '../ToggleSwitch.vue'
import { useWidgetStore } from '../../stores/widgetStore.js'
import { useSamExtraCapabilities } from '../../composables/useSamExtraCapabilities'
import { isKrea2Family } from '../../utils/generationFamily'
import {
  DEFAULTS, ETA_HELP, ETA_RANGE, NOISE_RANGE, ODE_NOTE, STAGE_HELP, STAGE_OPTIONS, WIDGET_IDS,
  WINDOW_EMPTY_WARNING, WINDOW_HELP, WINDOW_UNSUPPORTED_TEXT, cardNote, hiddenNote, isDefault, paramsNote, parseValues,
  passesUsing, readValues, samplerHints, subtitle, usesEta, windowEmpty, windowUnsupported, type XsampKey,
} from '../../utils/extraSamplers'

const storeWidgets = useWidgetStore().widgets as Record<string, any>
const { capabilities } = useSamExtraCapabilities()

const passes = computed(() => passesUsing(storeWidgets))
// ComfyUI(스냅샷 'not_applicable' — paramsNote 와 같은 판정)는 Hires CFG 가 끔이면 메인 CFG 로 돈다(Forge 는 1)
const hints = computed(() => samplerHints(storeWidgets, capabilities.value?.status === 'not_applicable'))
const values = computed(() => readValues(storeWidgets))
const settings = computed(() => parseValues(values.value))
const stage = computed(() => String(settings.value.max_stage))
const etaUsed = computed(() => usesEta(passes.value))
const emptyWindow = computed(() => etaUsed.value && windowEmpty(settings.value))
const windowDropped = computed(() => etaUsed.value && windowUnsupported(settings.value, capabilities.value))
const atDefaults = computed(() => isDefault(values.value))
const subtitleText = computed(() => subtitle(passes.value))
const hiddenText = computed(() => hiddenNote(settings.value))
const note = computed(() => cardNote(capabilities.value, isKrea2Family(storeWidgets.generation_family_combo)))
const paramsLine = computed(() => paramsNote(capabilities.value))

function setWindow(on: boolean) {
  if (etaUsed.value) storeWidgets[WIDGET_IDS.noise_window] = on ? '1' : '0'
}

function resetDefaults() {
  for (const [key, value] of Object.entries(DEFAULTS) as Array<[XsampKey, string]>) {
    storeWidgets[WIDGET_IDS[key]] = value
  }
}
</script>

<style scoped>
.xsa-hints { margin: -2px 0 6px; }
.xsa-hint { margin: 2px 0; color: var(--state-warn-fg); font-size: var(--fs-label); line-height: 1.4; }
.xsa-fields {
  margin: 2px 0 8px; padding: 8px 10px; border: 1px solid var(--border); border-radius: var(--radius-base);
  display: flex; flex-direction: column; gap: 4px;
}
.xsa-head { display: flex; align-items: center; gap: var(--sp-2); flex-wrap: wrap; }
.xsa-title { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-primary); }
.xsa-subtitle { font-size: var(--fs-label); color: var(--state-info-fg); flex: 1 1 auto; }
.xsa-stages { display: flex; gap: 4px; }
.xsa-stages button {
  flex: 1 1 0; height: 28px; font-size: var(--fs-meta); border-radius: 5px; cursor: pointer;
  background: transparent; border: 1px solid var(--border); color: var(--text-secondary);
}
.xsa-stages button.active { border-color: var(--accent); color: var(--text-primary); font-weight: var(--fw-medium); }
.xsa-stages button:focus-visible, .xsa-reset:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.xsa-window { align-items: flex-end; }
.xsa-window.off { opacity: 0.5; }
.xsa-window.off .ext-check-row { pointer-events: none; }
.xsa-note { margin: 2px 0 4px; color: var(--text-muted); font-size: var(--fs-label); line-height: 1.45; }
.xsa-note.xsa-warnline { color: var(--state-warn-fg); }
.xsa-hidden {
  margin: -2px 0 6px; display: flex; align-items: center; gap: var(--sp-2); flex-wrap: wrap;
  color: var(--text-muted); font-size: var(--fs-label); line-height: 1.4;
}
.xsa-reset {
  height: 24px; padding: 0 10px; font-size: var(--fs-meta); font-weight: var(--fw-medium); border-radius: 5px; cursor: pointer;
  background: transparent; border: 1px dashed var(--border); color: var(--text-muted);
}
.xsa-reset:hover:not(:disabled) { border-color: var(--text-muted); color: var(--text-primary); }
.xsa-reset:disabled { opacity: 0.5; cursor: default; }
</style>
