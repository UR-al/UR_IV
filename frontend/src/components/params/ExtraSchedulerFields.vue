<template>
  <div v-if="hints.length" class="xs-hints">
    <p v-for="hint in hints" :key="hint" class="xs-hint">{{ hint }}</p>
  </div>
  <section v-if="passes.length" class="xs-fields" aria-label="Extra Schedulers 값">
    <div class="xs-head">
      <span class="xs-title">Extra Schedulers 값</span>
      <span class="xs-subtitle">{{ subtitleText }}</span>
      <button type="button" class="xs-reset" :disabled="atDefaults"
        title="확장 기본값으로 — 식 M * (m / M) ** x · 목록 [1.0, 0.6, 0.25, 0.1, 0.0] · μ 0 · β 0.5"
        @click="resetDefaults">기본값</button>
    </div>
    <template v-if="showCustom">
      <div class="ext-field">
        <label>custom 방식</label>
        <div class="xs-modes" role="radiogroup" aria-label="custom 방식">
          <button v-for="option in MODE_OPTIONS" :key="option.key" type="button" role="radio"
            :aria-checked="values.custom_mode === option.key" :class="{ active: values.custom_mode === option.key }"
            @click="storeWidgets[WIDGET_IDS.custom_mode] = option.key">{{ option.label }}</button>
        </div>
      </div>
      <div v-if="values.custom_mode === 'expression'" class="ext-field">
        <label>custom 식</label>
        <input type="text" v-model="storeWidgets[WIDGET_IDS.custom_expression]" placeholder="m + (M - m) * (1 - x) ** 3"
          spellcheck="false" />
        <p v-if="expressionError" class="xs-error" role="alert">식을 쓸 수 없습니다: {{ expressionError }}</p>
        <p class="xs-note">변수 m(σ min)·M(σ max)·n(스텝 수)·s(이번 스텝, 0부터)·x(= s/(n−1)), 상수 phi·pi·e, 연산
          + − * / **, 함수 abs sqrt exp log log2 log10 sin cos tan asin acos atan atan2 sinh cosh tanh floor ceil min max.
          값은 0보다 커야 하고 마지막 0은 자동으로 붙습니다. 기본 식은 Exponential 과 같은 곡선. flow 모델(Anima 등)은
          M(=1)을 넘지 않게.</p>
      </div>
      <div v-else class="ext-field">
        <label>custom 시그마 목록</label>
        <input type="text" v-model="storeWidgets[WIDGET_IDS.custom_sigmas]" placeholder="[1.0, 0.6, 0.25, 0.1, 0.0]"
          spellcheck="false" />
        <p v-if="sigmasError" class="xs-error" role="alert">시그마 목록을 쓸 수 없습니다: {{ sigmasError }}</p>
        <p class="xs-note">1.0 으로 시작해 0.0 으로 끝나면 σmax~σmin 사이로 늘려 쓰고, 그 밖의 목록은 시그마 값 그대로 씁니다.
          개수가 스텝 수와 다르면 로그-선형 보간(Align Your Steps 와 같은 방식).</p>
      </div>
    </template>
    <template v-if="showLaplace">
      <div class="ext-row">
        <div class="ext-field"><label>Laplace μ</label>
          <input type="number" v-model="storeWidgets[WIDGET_IDS.laplace_mu]"
            :min="MU_RANGE.min" :max="MU_RANGE.max" :step="MU_RANGE.step" /></div>
        <div class="ext-field"><label>Laplace β</label>
          <input type="number" v-model="storeWidgets[WIDGET_IDS.laplace_beta]"
            :min="BETA_RANGE.min" :max="BETA_RANGE.max" :step="BETA_RANGE.step" /></div>
      </div>
      <p class="xs-note">시그마가 e^μ 근처에 모이고 β 가 클수록 넓게 퍼집니다. flow 모델은 μ 0 그대로(음수 μ 는 물 빠진 이미지),
        샘플러는 Euler 계열 권장. 같은 시그마가 되풀이되면 범위 안의 곡선에 스텝을 고르게 다시 놓습니다 — ComfyUI 기본
        LaplaceScheduler 와 다른 점이고, 앱은 Forge·ComfyUI 모두 sam-extra 규칙으로 계산합니다.</p>
    </template>
    <p v-if="note" class="xs-note xs-warnline" role="status">{{ note }}</p>
  </section>
</template>

<script setup lang="ts">
/**
 * 파라미터 카드의 Sampler/Scheduler 줄 아래 — Extra Schedulers(sam-extra) 값 칸. 다음 요청의 어느 패스(메인·Hires·
 * ADetailer·SAM3)가 laplace·custom 을 쓸 때만 보인다(확장은 한 생성에 값 한 벌 — 그 패스들이 같은 값을 쓴다). Laplace 칸은
 * Laplace 를 쓸 때만, custom 칸은 custom 을 쓸 때만. 값은 위젯 스토어(_xsched_*)에 두고 파이썬 프록시(ui/
 * extra_schedulers_ui.py)가 같은 문자열을 읽는다 — 보낼지(스냅샷·패스·백엔드)는 파이썬이 정한다(core/extra_schedulers.
 * plan). 칸이 안 보여도 값은 그대로 남는다(다음에 laplace·custom 을 고르면 그 값으로).
 *
 * Scheduler 줄 바로 아래 힌트(flow 에 맞지 않는 스케줄러, Karras Dynamic 의 ρ)는 칸과 상관없이 보인다. 식·목록 오류 줄은
 * 파이썬이 `pushWidgetProperty(…, 'error', …)` 로 보낸 문구다(계획 C7 — 그 전에는 비어 있다). 새 브리지 이름은 없다.
 * App.vue 를 거치지 않고 스토어를 직접 읽는다(ParamsBasicCard 안에서 그린다).
 */
import { computed } from 'vue'
import { useWidgetStore } from '../../stores/widgetStore.js'
import { useSamExtraCapabilities } from '../../composables/useSamExtraCapabilities'
import { isKrea2Family } from '../../utils/generationFamily'
import {
  BETA_RANGE, DEFAULTS, ERROR_PROPERTY, MODE_OPTIONS, MU_RANGE, WIDGET_IDS, cardNote, isDefault, passesUsing, readValues,
  schedulerHints, subtitle, usesCustom, usesLaplace, type XschedKey,
} from '../../utils/extraSchedulers'

const store = useWidgetStore()
const storeWidgets = store.widgets as Record<string, any>
const getProperty = store.getProperty as (id: string, prop: string, def?: unknown) => unknown
const { capabilities } = useSamExtraCapabilities()

const passes = computed(() => passesUsing(storeWidgets))
const hints = computed(() => schedulerHints(storeWidgets))
const values = computed(() => readValues(storeWidgets))
const subtitleText = computed(() => subtitle(passes.value))
const showCustom = computed(() => usesCustom(passes.value))
const showLaplace = computed(() => usesLaplace(passes.value))
const atDefaults = computed(() => isDefault(values.value))
const note = computed(() => cardNote(capabilities.value, isKrea2Family(storeWidgets.generation_family_combo)))

function errorText(key: XschedKey): string {
  const value = getProperty(WIDGET_IDS[key], ERROR_PROPERTY, null)
  return typeof value === 'string' ? value.trim() : ''
}
const expressionError = computed(() => errorText('custom_expression'))
const sigmasError = computed(() => errorText('custom_sigmas'))

function resetDefaults() {
  for (const [key, value] of Object.entries(DEFAULTS) as Array<[XschedKey, string]>) {
    storeWidgets[WIDGET_IDS[key]] = value
  }
}
</script>

<style scoped>
.xs-hints { margin: -2px 0 6px; }
.xs-hint { margin: 2px 0; color: var(--state-warn-fg); font-size: var(--fs-label); line-height: 1.4; }
.xs-fields {
  margin: 2px 0 8px; padding: 8px 10px; border: 1px solid var(--border); border-radius: var(--radius-base);
  display: flex; flex-direction: column; gap: 4px;
}
.xs-head { display: flex; align-items: center; gap: var(--sp-2); flex-wrap: wrap; }
.xs-title { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-primary); }
.xs-subtitle { font-size: var(--fs-label); color: var(--state-info-fg); flex: 1 1 auto; }
.xs-modes { display: flex; gap: 4px; }
.xs-modes button {
  flex: 1 1 0; height: 28px; font-size: var(--fs-meta); border-radius: 5px; cursor: pointer;
  background: transparent; border: 1px solid var(--border); color: var(--text-secondary);
}
.xs-modes button.active { border-color: var(--accent); color: var(--text-primary); font-weight: var(--fw-medium); }
.xs-modes button:focus-visible, .xs-reset:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.xs-note { margin: 2px 0 4px; color: var(--text-muted); font-size: var(--fs-label); line-height: 1.45; }
.xs-note.xs-warnline { color: var(--state-warn-fg); }
.xs-error { margin: 3px 0 0; color: var(--state-alert-fg); font-size: var(--fs-label); line-height: 1.4; }
.xs-reset {
  height: 24px; padding: 0 10px; font-size: var(--fs-meta); font-weight: var(--fw-medium); border-radius: 5px; cursor: pointer;
  background: transparent; border: 1px dashed var(--border); color: var(--text-muted);
}
.xs-reset:hover:not(:disabled) { border-color: var(--text-muted); color: var(--text-primary); }
.xs-reset:disabled { opacity: 0.5; cursor: default; }
</style>
