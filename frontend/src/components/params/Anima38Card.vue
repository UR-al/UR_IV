<template>
  <details v-if="view.visible" class="ext-card a38-card">
    <summary class="ext-title">Anima 3.8B<span v-if="summaryText" class="a38-summary">{{ summaryText }}</span></summary>
    <label class="ext-check-row" title="Use adapter on negative prompt — 끄면 부정 프롬프트는 순정 Anima 인코더로">
      <ToggleSwitch v-model="negative" size="sm" /><span>부정 프롬프트도 Qwen3.5 커넥터로</span></label>
    <div v-if="view.v2FixedNote" class="ext-note">v2 번들은 부정 커넥터 강도가 1.0 으로 고정됩니다.</div>
    <div v-if="view.showNegativeStrength" class="ext-field">
      <label>부정 어댑터 강도 (v1)</label>
      <input type="number" v-model="storeWidgets[WIDGET_IDS.negative_strength]"
        :min="STRENGTH_RANGE.min" :max="STRENGTH_RANGE.max" :step="STRENGTH_RANGE.step" />
    </div>
    <label v-if="view.showBypass" class="ext-check-row" title="v2 번들의 자동 켜짐과 v1 어댑터를 끕니다 — 순정 확인·A/B 비교용">
      <ToggleSwitch v-model="bypass" size="sm" /><span>Bypass — 순정 Anima(0.6B)로 생성</span></label>
    <div v-if="view.v1BypassedNote" class="ext-note a38-alert">Bypass 가 켜져 있어 v1 어댑터를 쓰지 않습니다 — v1 을 쓰려면 Bypass 를 끄세요.</div>
    <template v-if="view.showV1">
      <label class="ext-check-row" title="Forge 는 아코디언을 켤 때만 v1 어댑터를 씁니다(v2 번들은 자동)">
        <ToggleSwitch v-model="v1Enabled" size="sm" /><span>v1 어댑터 켜기 (구형 Anima)</span></label>
      <div class="ext-row">
        <div class="ext-field">
          <label>어댑터{{ adapters.unverified ? ' (설치 여부 미확인)' : '' }}</label>
          <CustomSelect v-model="storeWidgets[WIDGET_IDS.adapter]" :options="adapters.options" />
        </div>
        <div class="ext-field">
          <label>어댑터 강도</label>
          <input type="number" v-model="storeWidgets[WIDGET_IDS.strength]"
            :min="STRENGTH_RANGE.min" :max="STRENGTH_RANGE.max" :step="STRENGTH_RANGE.step" />
        </div>
      </div>
    </template>
    <label class="ext-check-row" title="Forge 는 img2img 탭에 아코디언이 따로 있다 — 사용자 Forge 설정은 모두 꺼짐">
      <ToggleSwitch v-model="applyImg2img" size="sm" /><span>I2I·인페인트에도 적용</span></label>
    <div class="ext-note" :class="{ 'a38-alert': status.status === 'missing' }">{{ status.text }}</div>
    <div class="a38-actions">
      <button type="button" class="a38-reset" title="Forge 설정(txt2img)과 같은 앱 기본값으로 — 부정 커넥터 켬, v1 끔"
        @click="resetDefaults">기본값</button>
    </div>
  </details>
</template>

<script setup lang="ts">
/**
 * 파라미터 열 — Anima 3.8B 카드(sam-extra "Anima 3.8B (Qwen3.5 / v2)"). 값은 위젯 스토어(_a38_*)에 두고 파이썬
 * 프록시(ui/anima38_ui.py)가 같은 문자열을 읽는다. 모델 종류(model_combo 의 animaKinds 속성 — 파이썬이 연결 때
 * 체크포인트 헤더로 판정)에 따라 쓰이는 칸만 보인다. 비 Anima·Krea2 면 숨긴다(설정은 저장된다). 보낼지(모델 종류·
 * 게이트·앱 기본값 규칙)는 파이썬이 정한다 — 여기는 표시만. 새 브리지 액션은 없다.
 */
import { computed } from 'vue'
import CustomSelect from '../CustomSelect.vue'
import ToggleSwitch from '../ToggleSwitch.vue'
import { useWidgetStore } from '../../stores/widgetStore.js'
import { widgetFlag } from '../../composables/widgetFlag'
import { useSamExtraCapabilities } from '../../composables/useSamExtraCapabilities'
import { isKrea2Family } from '../../utils/generationFamily'
import {
  COMFY_ADAPTER_PROPERTY, DEFAULTS, KIND_PROPERTY, STRENGTH_RANGE, WIDGET_IDS, adapterChoices, cardStatus, cardView,
  modelKind, readValues, summary, type Anima38WidgetKey,
} from '../../utils/anima38Card'

const store = useWidgetStore()
const storeWidgets = store.widgets as Record<string, any>
const getProperty = store.getProperty as (id: string, prop: string, def?: unknown) => unknown
const { capabilities } = useSamExtraCapabilities()

const negative = widgetFlag(storeWidgets, WIDGET_IDS.negative)
const bypass = widgetFlag(storeWidgets, WIDGET_IDS.bypass)
const v1Enabled = widgetFlag(storeWidgets, WIDGET_IDS.enabled)
const applyImg2img = widgetFlag(storeWidgets, WIDGET_IDS.apply_img2img)

const values = computed(() => readValues(storeWidgets))
const kind = computed(() => modelKind(getProperty('model_combo', KIND_PROPERTY, {}), storeWidgets.model_combo))
const krea2 = computed(() => isKrea2Family(storeWidgets.generation_family_combo))
const view = computed(() => cardView(kind.value, values.value, krea2.value))
const summaryText = computed(() => summary(kind.value, values.value))
const status = computed(() => cardStatus(capabilities.value, kind.value))
// ComfyUI 는 기능 스냅샷 대신 object_info 의 어댑터 목록(파이썬이 연결 때 _a38_adapter 속성으로 보낸다)
const adapters = computed(() => adapterChoices(
  capabilities.value, values.value.adapter, getProperty(WIDGET_IDS.adapter, COMFY_ADAPTER_PROPERTY, null)))

function resetDefaults() {
  for (const [key, value] of Object.entries(DEFAULTS) as Array<[Anima38WidgetKey, string]>) {
    storeWidgets[WIDGET_IDS[key]] = value
  }
}
</script>

<style scoped>
.a38-summary { margin-left: var(--sp-2); font-weight: var(--fw-medium); color: var(--state-info-fg); font-size: var(--fs-label); }
.ext-note { margin: 3px 0 6px; color: var(--text-muted); font-size: var(--fs-label); line-height: 1.45; }
.ext-note.a38-alert { color: var(--state-alert-fg); }
.a38-actions { display: flex; justify-content: flex-end; margin-top: var(--sp-2); }
.a38-reset {
  height: 28px; padding: 0 12px; font-size: var(--fs-meta); font-weight: var(--fw-medium); border-radius: 5px; cursor: pointer;
  background: transparent; border: 1px dashed var(--border); color: var(--text-muted);
}
.a38-reset:hover { border-color: var(--text-muted); color: var(--text-primary); }
</style>
