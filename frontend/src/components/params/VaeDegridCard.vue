<template>
  <details class="ext-card degrid-card">
    <summary class="ext-title">VAE DeGrid<span v-if="summaryText" class="degrid-summary">{{ summaryText }}</span></summary>
    <label class="ext-check-row" title="sam-extra 'Anima VAE DeGrid (NAFNet)' — VAE 디코딩이 남기는 격자 무늬를 지운다">
      <ToggleSwitch v-model="enabled" size="sm" /><span>VAE DeGrid 사용</span></label>
    <div class="ext-field">
      <label>모델{{ models.unverified ? ' (목록 미확인)' : '' }}</label>
      <div class="degrid-model-row">
        <CustomSelect class="degrid-model-input" v-model="modelLabel" :options="modelLabels" />
        <button type="button" class="degrid-icon-btn" :title="refreshHint"
          aria-label="모델 목록 다시 확인" @click="refresh()"><Icon name="refresh" /></button>
      </div>
      <p v-if="models.missing" class="degrid-warn">목록에 없는 모델입니다 — 파일을 옮겼거나 이름이 바뀌었으면 다시 고르세요.</p>
    </div>
    <div class="ext-field">
      <label>모드</label>
      <CustomSelect v-model="modeLabel" :options="modeLabels" />
    </div>
    <div class="ext-row">
      <div class="ext-field"><label>강도</label>
        <input type="number" v-model="storeWidgets[WIDGET_IDS.strength]"
          :min="STRENGTH_RANGE.min" :max="STRENGTH_RANGE.max" :step="STRENGTH_RANGE.step" /></div>
      <div class="ext-field"><label>타일 크기 (0 = 나누지 않음)</label>
        <input type="number" v-model="storeWidgets[WIDGET_IDS.tile]"
          :min="TILE_RANGE.min" :max="TILE_RANGE.max" :step="TILE_RANGE.step" /></div>
    </div>
    <label class="ext-check-row" title="Forge 는 img2img 탭에 아코디언이 따로 있다 — 기본은 T2I 계열에만">
      <ToggleSwitch v-model="applyImg2img" size="sm" /><span>I2I·인페인트에도 적용</span></label>
    <div class="ext-note">모든 후처리(ADetailer·SAM3) 뒤, 저장 직전에 이미지마다 한 번 적용합니다. 보조 작업(Refine·단독
      SAM3·ADetailer·손 재구성)에는 걸지 않습니다. Forge 설정에서 Extras 판을 메인 탭에도 켜 두면 두 번 걸립니다.</div>
    <div class="ext-note" :class="toneClass" role="status">{{ status.text }}</div>
    <details class="ext-sub" @toggle="onOptionsToggle">
      <summary>장치·메모리 (Forge 설정)</summary>
      <ul class="degrid-options">
        <li v-for="row in optionRows" :key="row.key" :class="{ 'degrid-alert': row.blocked }">
          <span class="degrid-option-label">{{ row.label }}</span><span>{{ row.text }}</span>
        </li>
      </ul>
      <div class="ext-note">{{ optionsNote }}</div>
    </details>
    <div class="degrid-actions">
      <button type="button" class="degrid-reset" title="확장 기본값으로 — 꺼짐 · 자동 · Full · 1 · 512, I2I 끔"
        @click="resetDefaults">기본값</button>
    </div>
  </details>
</template>

<script setup lang="ts">
/**
 * 파라미터 열 — VAE DeGrid 카드(sam-extra "Anima VAE DeGrid (NAFNet)"). 값은 위젯 스토어(_degrid_*)에 두고 파이썬
 * 프록시(ui/vae_degrid_ui.py)가 같은 문자열을 읽는다. 모드 칸은 확장 라벨을 보이고 키를 저장한다. 모델은 ''(자동)
 * 또는 Forge 식 이름 — 목록은 Forge 기능 스냅샷(choices.degrid_models), ComfyUI 면 `_degrid_model` 의 comfyModels 속성.
 * ↻ 는 백엔드와 상관없이 기존 `sam_extra_capabilities_get {refresh: true}` 를 보낸다 — 파이썬이 Forge 면 기능 스냅샷을,
 * ComfyUI 면 object_info 를 다시 받아 comfyModels 를 다시 보낸다(새 브리지 이름 없음). 버튼 설명도 백엔드를 따른다.
 * 보낼지(대상·백엔드·게이트)는 파이썬이 정한다 — 여기는 표시만. 칸은 막지 않는다(상태 줄이 말한다).
 *
 * '장치·메모리' 칸은 Forge 설정(sam3_degrid_*) 읽기 전용 거울이다 — 앱 덮어쓰기(설정 › Forge, P10)는 ui_prefs 에
 * 있으므로 펼칠 때마다 getUiPrefs 로 다시 읽는다(새 브리지 이름 없음). ComfyUI 는 노드 고정값(auto · fp32 · 끔).
 */
import { computed, onUnmounted, ref } from 'vue'
import CustomSelect from '../CustomSelect.vue'
import ToggleSwitch from '../ToggleSwitch.vue'
import { getBackend } from '../../bridge.js'
import { useWidgetStore } from '../../stores/widgetStore.js'
import { widgetFlag } from '../../composables/widgetFlag'
import { useSamExtraCapabilities } from '../../composables/useSamExtraCapabilities'
import { isKrea2Family } from '../../utils/generationFamily'
import { KIND_PROPERTY, modelKind } from '../../utils/anima38Card'
import { PREF_KEY, normalizeOverrides, type Overrides } from '../../utils/forgeOptionOverrides'
import {
  COMFY_MODELS_PROPERTY, DEFAULTS, MODE_OPTIONS, OPTION_KEYS, STRENGTH_RANGE, TILE_RANGE, WIDGET_IDS,
  cardStatus, effectiveOption, keyOf, labelOf, modelChoices, readValues, refreshTitle, summary, type DegridWidgetKey,
} from '../../utils/vaeDegrid'

const store = useWidgetStore()
const storeWidgets = store.widgets as Record<string, any>
const getProperty = store.getProperty as (id: string, prop: string, def?: unknown) => unknown
const { capabilities, refresh } = useSamExtraCapabilities()

const enabled = widgetFlag(storeWidgets, WIDGET_IDS.enabled)
const applyImg2img = widgetFlag(storeWidgets, WIDGET_IDS.apply_img2img)

const values = computed(() => readValues(storeWidgets))
const comfyModels = computed(() => getProperty(WIDGET_IDS.model, COMFY_MODELS_PROPERTY, null))
const kind = computed(() => modelKind(getProperty('model_combo', KIND_PROPERTY, {}), storeWidgets.model_combo))
const krea2 = computed(() => isKrea2Family(storeWidgets.generation_family_combo))
const isComfy = computed(() => capabilities.value?.status === 'not_applicable')
const refreshHint = computed(() => refreshTitle(capabilities.value))

const models = computed(() => modelChoices(capabilities.value, values.value.model, comfyModels.value))
const modelLabels = computed(() => models.value.options.map(o => o.label))
const modelLabel = computed({
  get: () => models.value.options.find(o => o.value === values.value.model)?.label ?? models.value.options[0]!.label,
  set: (label: string | number) => {
    const picked = models.value.options.find(o => o.label === String(label))
    if (picked) storeWidgets[WIDGET_IDS.model] = picked.value
  },
})

const modeLabel = computed({
  get: () => labelOf(MODE_OPTIONS, values.value.mode),
  set: (label: string | number) => { storeWidgets[WIDGET_IDS.mode] = keyOf(MODE_OPTIONS, String(label)) },
})
const modeLabels = MODE_OPTIONS.map(o => o.label)

const summaryText = computed(() => summary(values.value))
const status = computed(() => cardStatus({
  caps: capabilities.value, values: values.value, kind: kind.value, krea2: krea2.value, comfyModels: comfyModels.value,
}))
const toneClass = computed(() => ({
  'degrid-alert': status.value.tone === 'alert', 'degrid-warnline': status.value.tone === 'warn',
}))

// 장치·메모리 — 앱 덮어쓰기(ui_prefs.forgeOptionOverrides)는 펼칠 때마다 다시 읽는다
const overrides = ref<Overrides>({})
let disposed = false
const optionRows = computed(() => OPTION_KEYS.map(key => effectiveOption(key, overrides.value, capabilities.value)))
const optionsNote = computed(() => isComfy.value
  ? 'ComfyUI 는 Forge 설정을 쓰지 않습니다 — 노드가 auto · fp32 · 끔(확장 기본값)으로 돕니다.'
  : '바꾸려면 설정 › Forge 의 sam-extra 설정(요청마다 적용)에서 고릅니다. Forge 값은 Forge 가 시작할 때 읽은 것입니다.')

async function loadOverrides() {
  const backend = await getBackend()
  if (disposed) return
  backend?.getUiPrefs?.((raw: string) => {
    if (disposed) return
    try {
      overrides.value = normalizeOverrides((JSON.parse(raw) || {})[PREF_KEY])
    } catch {
      /* 못 읽으면 지난 값을 둔다 — 표시 전용 */
    }
  })
}

function onOptionsToggle(event: Event) {
  if ((event.target as HTMLDetailsElement | null)?.open) void loadOverrides()
}

function resetDefaults() {
  for (const [key, value] of Object.entries(DEFAULTS) as Array<[DegridWidgetKey, string]>) {
    storeWidgets[WIDGET_IDS[key]] = value
  }
}

onUnmounted(() => { disposed = true })
</script>

<style scoped>
.degrid-summary { margin-left: var(--sp-2); font-weight: var(--fw-medium); color: var(--state-info-fg); font-size: var(--fs-label); }
.ext-note { margin: 3px 0 6px; color: var(--text-muted); font-size: var(--fs-label); line-height: 1.45; }
.ext-note.degrid-alert { color: var(--state-alert-fg); }
.ext-note.degrid-warnline { color: var(--state-warn-fg); }
.degrid-model-row { display: flex; align-items: stretch; gap: 6px; }
.degrid-model-input { flex: 1 1 auto; min-width: 0; }
.degrid-icon-btn {
  flex: 0 0 auto; width: 32px; padding: 0;
  display: inline-flex; align-items: center; justify-content: center;
  background: transparent; border: 1px solid var(--border); border-radius: var(--radius-base);
  color: var(--text-muted); cursor: pointer;
}
.degrid-icon-btn:hover { border-color: var(--text-muted); color: var(--text-primary); }
.degrid-icon-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.degrid-warn { margin: 4px 0 0; color: var(--state-warn-fg); font-size: var(--fs-label); line-height: 1.4; }
.degrid-options { list-style: none; margin: 6px 0 4px; padding: 0; display: flex; flex-direction: column; gap: 4px; }
.degrid-options li {
  display: flex; flex-wrap: wrap; justify-content: space-between; gap: 2px 8px;
  font-size: var(--fs-label); color: var(--text-secondary); line-height: 1.4;
}
.degrid-options li.degrid-alert { color: var(--state-alert-fg); }
.degrid-option-label { color: var(--text-muted); }
.degrid-actions { display: flex; justify-content: flex-end; margin-top: var(--sp-2); }
.degrid-reset {
  height: 28px; padding: 0 12px; font-size: var(--fs-meta); font-weight: var(--fw-medium); border-radius: 5px; cursor: pointer;
  background: transparent; border: 1px dashed var(--border); color: var(--text-muted);
}
.degrid-reset:hover { border-color: var(--text-muted); color: var(--text-primary); }
</style>
