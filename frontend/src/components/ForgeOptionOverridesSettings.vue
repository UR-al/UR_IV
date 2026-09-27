<template>
  <section class="glass-card mt-16 forge-options" aria-labelledby="forge-options-title">
    <div class="fo-head">
      <span id="forge-options-title" class="field-label fo-title">sam-extra 설정 (요청마다 적용)</span>
      <span class="fo-summary">{{ summaryText }}</span>
    </div>
    <p class="fo-lead">
      Forge 설정(config.json)은 바꾸지 않습니다. 앱이 보내는 생성 요청에만 적용되고, 요청이 끝나면 Forge 가 원래 값으로
      되돌립니다. 기본은 모두 'Forge 설정 따름'(보내지 않음)이라 Forge UI 와 같게 생성합니다. 이 Forge 에 없는 설정이나
      확인하지 못한 설정은 보내지 않습니다.
    </p>
    <p v-if="isComfy" class="fo-note" role="status">ComfyUI 백엔드에는 해당 없음 — Forge 로 생성할 때만 보냅니다.</p>
    <div v-for="group in GROUPS" :key="group.id" class="fo-group">
      <h3 class="fo-group-title">{{ group.title }}</h3>
      <div v-for="spec in specsOf(group.id)" :key="spec.key" class="fo-row">
        <div class="fo-label">
          <span class="field-label">{{ spec.label }}</span>
          <small>{{ DESCRIPTIONS[spec.key] }}</small>
          <small v-if="spec.infotext" class="fo-infotext">결과 infotext: {{ spec.infotext }}</small>
        </div>
        <div class="fo-control">
          <CustomSelect :model-value="choiceLabel(choiceOf(overrides, spec.key))" :options="choiceLabels"
            @update:model-value="setChoice(spec.key, $event)" />
        </div>
        <div class="fo-meta">
          <span class="fo-status" :class="{ 'fo-alert': blocked(spec.key) }">{{ statusOf(spec.key).text }}</span>
          <span v-if="warningOf(spec.key)" class="fo-warning">{{ warningOf(spec.key) }}</span>
        </div>
      </div>
    </div>
    <p v-if="error" class="fo-error" role="alert">{{ error }}</p>
  </section>
</template>

<script setup lang="ts">
/**
 * 설정 › Forge — sam-extra Forge 옵션 요청별 덮어쓰기(P10). 값은 ui_prefs.forgeOptionOverrides({키: boolean}, 키 없음 =
 * 'Forge 설정 따름')이고 저장은 기존 save_ui_prefs, 읽기는 getUiPrefs + uiPrefsLoaded(SpectrumSettings 와 같은 방식 —
 * 새 브리지 이름 없음). 보낼지·병합·거절 시 다시 보내기는 파이썬 백엔드(core/forge_override_settings)가 정한다.
 * 상태 한 줄은 sam-extra 기능 스냅샷(options·options_known)으로 표시만 한다.
 */
import { computed, onMounted, onUnmounted, ref } from 'vue'
import CustomSelect from './CustomSelect.vue'
import { getBackend, onBackendEvent } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'
import { useSamExtraCapabilities } from '../composables/useSamExtraCapabilities'
import {
  CHOICES, DESCRIPTIONS, GROUPS, PREF_KEY, choiceFromLabel, choiceLabel, choiceOf, isBlocked, normalizeOverrides,
  optionStatus, rowWarning, specsOf, summary, withChoice, type OptionStatus, type Overrides,
} from '../utils/forgeOptionOverrides'

const { capabilities } = useSamExtraCapabilities()
const overrides = ref<Overrides>({})
const error = ref('')
const choiceLabels = CHOICES.map(choice => choice.label)
let edited = false
let disposed = false
let disconnect: (() => void) | undefined

const isComfy = computed(() => capabilities.value?.status === 'not_applicable')
const summaryText = computed(() => summary(overrides.value))

function statusOf(key: string): OptionStatus {
  return optionStatus(capabilities.value, key)
}

function blocked(key: string): boolean {
  return isBlocked(statusOf(key), choiceOf(overrides.value, key))
}

function warningOf(key: string): string {
  return rowWarning(key, choiceOf(overrides.value, key), statusOf(key).forgeValue)
}

function apply(raw: string) {
  // 시작 때 한 번 오는 값 — 사용자가 이미 고쳤으면 덮지 않는다(늦게 온 getUiPrefs 응답).
  if (disposed || edited) return
  try {
    const prefs = JSON.parse(raw) || {}
    overrides.value = normalizeOverrides(prefs[PREF_KEY])
    error.value = ''
  } catch {
    error.value = 'sam-extra 설정을 읽지 못했습니다.'
  }
}

function setChoice(key: string, label: string | number) {
  edited = true
  overrides.value = withChoice(overrides.value, key, choiceFromLabel(label))
  // dict 전체를 보낸다 — 'Forge 설정 따름'은 키를 지운 dict 로 파일 값을 바꾼다(prefs.update).
  requestAction('save_ui_prefs', { [PREF_KEY]: { ...overrides.value } })
}

onMounted(async () => {
  disconnect = onBackendEvent('uiPrefsLoaded', apply)
  const backend = await getBackend()
  if (disposed) return
  backend?.getUiPrefs?.((raw: string) => { if (!disposed && !edited) apply(raw) })
})
onUnmounted(() => { disposed = true; disconnect?.() })
</script>

<style scoped>
.forge-options { color: var(--text-primary); overflow-wrap: anywhere; }
.fo-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
.fo-title { font-weight: var(--fw-bold); }
.fo-summary { font-size: var(--fs-label); color: var(--state-info-fg); }
.fo-lead, .fo-note { margin: 8px 0 0; font-size: var(--fs-label); line-height: 1.6; color: var(--text-muted); }
.fo-note { color: var(--state-warn-fg); }
.fo-group { margin-top: 16px; }
.fo-group-title { margin: 0 0 6px; font-size: var(--fs-label); font-weight: var(--fw-medium); color: var(--text-secondary); }
.fo-row {
  display: grid; grid-template-columns: minmax(0, 1fr) 150px; gap: 4px 12px; align-items: center;
  padding: 10px 0; border-top: 1px solid var(--border);
}
.fo-label { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.fo-label small { font-size: var(--fs-label); color: var(--text-muted); line-height: 1.45; }
.fo-infotext { font-family: var(--font-mono, monospace); }
.fo-control { min-width: 0; }
.fo-meta { grid-column: 1 / -1; display: flex; flex-direction: column; gap: 2px; font-size: var(--fs-label); }
.fo-status { color: var(--text-muted); }
.fo-status.fo-alert { color: var(--state-alert-fg); }
.fo-warning { color: var(--state-warn-fg); }
.fo-error { margin: 8px 0 0; font-size: var(--fs-label); color: var(--state-alert-fg); }
@media (max-width: 560px) {
  .fo-row { grid-template-columns: minmax(0, 1fr); }
}
</style>
