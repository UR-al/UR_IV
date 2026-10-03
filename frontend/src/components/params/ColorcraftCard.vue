<template>
  <details class="ext-card colorcraft-card">
    <summary class="ext-title">Colorcraft — 샘플링 중 색 보정<span v-if="badgeText" class="cc-summary"
      :class="{ 'cc-summary-warn': status.level === 'warn' }">{{ badgeText }}</span></summary>
    <div class="cc-switches">
      <label class="ext-check-row" title="Enable Colorcraft — sam-extra 'Colorcraft (sam-extra)'">
        <ToggleSwitch :modelValue="card.enabled" size="sm" @update:modelValue="setFlag('enabled', $event)" />
        <span>Colorcraft 사용</span></label>
      <label class="ext-check-row" title="Enable masking">
        <ToggleSwitch :modelValue="card.masking" size="sm" @update:modelValue="setFlag('masking', $event)" />
        <span>마스크 사용</span></label>
    </div>
    <div class="ext-note cc-status" :class="`cc-status-${status.level}`" role="status">{{ status.text }}</div>
    <div v-if="card.stateError" class="ext-note cc-warnline" role="alert">
      저장된 Colorcraft 상태를 읽지 못해 기본값으로 보입니다 — 값을 바꾸면 새 상태로 저장됩니다. ({{ card.stateError }})</div>
    <div class="cc-section-label">수정자</div>
    <ColorcraftNodePicker :marks="view.modifierMarks" :selected="modifierTag" label="수정자 고르기"
      @select="modifierTag = $event" />
    <ColorcraftModifierEditor :config="config" :tag="modifierTag" :hiresOn="hiresOn"
      @update="(name, value) => update(modifierTag, name, value)" @reset="reset(modifierTag)">
      <template #schedule="{ modifier }">
        <ColorcraftSchedulePreview :modifier="modifier" :steps="steps" />
      </template>
    </ColorcraftModifierEditor>
    <template v-if="card.masking">
      <div class="cc-section-label">마스크</div>
      <ColorcraftNodePicker :marks="view.maskMarks" :selected="maskTag" label="마스크 고르기" dividerBefore="C1"
        @select="maskTag = $event" />
      <p v-if="noMaskedModifier" class="ext-note">마스크를 켰지만 마스크를 고른 켜진 수정자가 없습니다 — 수정자의 '마스크' 칸에서
        고르세요(Basic 은 마스크를 받지 않습니다).</p>
      <ColorcraftMaskEditor :config="config" :tag="maskTag" @update="(name, value) => update(maskTag, name, value)"
        @reset="reset(maskTag)" />
    </template>
    <label class="ext-check-row" title="I2I·인페인트·채팅 이미지 편집 요청에도 보낸다(Forge img2img 패널은 따로, 기본 끔)">
      <ToggleSwitch :modelValue="card.img2img" size="sm" @update:modelValue="setFlag('img2img', $event)" />
      <span>I2I·인페인트에도 적용</span></label>
    <div class="ext-note">{{ PASS_NOTE }}</div>
    <div class="ext-note">{{ OVERVIEW_LEAD }}{{ view.text }}</div>
    <details class="ext-sub cc-io">
      <summary>가져오기 / 내보내기</summary>
      <div class="ext-field">
        <label title="SAM Extra Colorcraft — Forge 가 이미지에 남기는 값과 같은 문자열">현재 값 (infotext)</label>
        <div class="cc-io-row">
          <input type="text" class="cc-infotext" :value="infotext" readonly spellcheck="false" />
          <button type="button" class="cc-btn" :disabled="!card.enabled" @click="copyInfotext">{{ copied ? '복사함' : '복사' }}</button>
        </div>
      </div>
      <div class="ext-field">
        <label :title="IMPORT_HINT">붙여 넣기 (infotext · 'SAM Extra Colorcraft: …' · 'Colorcraft: …' · 그 값)</label>
        <textarea v-model="pasteText" class="cc-paste" rows="3" spellcheck="false" :placeholder="PASTE_PLACEHOLDER" />
        <div class="cc-io-row cc-io-end">
          <button type="button" class="cc-btn" :disabled="!pasteText.trim()" @click="applyPaste">가져오기</button>
        </div>
      </div>
    </details>
    <div class="cc-actions">
      <button type="button" class="cc-btn cc-reset-all" :class="{ 'cc-muted': atDefaults }"
        title="Colorcraft 를 끄고 모든 수정자·마스크를 확장 기본값으로" @click="resetAll">전체 초기화</button>
    </div>
  </details>
</template>

<script setup lang="ts">
/**
 * 파라미터 열 — Colorcraft 카드(sam-extra "Colorcraft (sam-extra)", Anima 가이던스 다음 — 계획 R8). 확장 공유 편집기처럼
 * 수정자·마스크 고르개(●/○) + 편집기 하나씩. 값은 위젯 스토어 네 칸(_cc_enabled·_cc_masking·_cc_img2img·_cc_state)에 두고
 * 파이썬 프록시(ui/colorcraft_ui.py)가 같은 문자열을 읽는다. 상태 문자열은 utils/colorcraft.dumpState(파이썬 dump_state 와
 * 같은 바이트). 보낼지(대상·백엔드·게이트)는 파이썬이 정한다 — 여기는 표시만(cardStatus). 새 브리지 이름 없음.
 */
import { computed, ref } from 'vue'
import ToggleSwitch from '../ToggleSwitch.vue'
import ColorcraftNodePicker from '../colorcraft/ColorcraftNodePicker.vue'
import ColorcraftModifierEditor from '../colorcraft/ColorcraftModifierEditor.vue'
import ColorcraftMaskEditor from '../colorcraft/ColorcraftMaskEditor.vue'
import ColorcraftSchedulePreview from '../colorcraft/ColorcraftSchedulePreview.vue'
import { useWidgetStore } from '../../stores/widgetStore.js'
import { useSamExtraCapabilities } from '../../composables/useSamExtraCapabilities'
import { isKrea2Family } from '../../utils/generationFamily'
import { importSamExtra } from '../../utils/samExtraImport'
import {
  DEFAULT_STATE, MASK_NONE, WIDGET_IDS, badge, cardStatus, dumpState, kindOf, overview, readCard, resetNode, setPath,
  takesMask, toInfotext, type CardKey,
} from '../../utils/colorcraft'

const PASS_NOTE = 'Base = 1차 패스와 I2I, Hires = Hires.fix 패스. 보조 작업(Refine·단독 SAM3·ADetailer·손 재구성)과 생성 안의 '
  + 'SAM3·ADetailer 패스에는 걸지 않습니다(확장과 같음). 노출·색·디테일·마스크는 Anima·Krea 2·Qwen-Image·Flux·Z-Image·'
  + 'Flux 2 계열에서만 — SD1.5/SDXL 은 대비·색 이동만.'
const OVERVIEW_LEAD = '한눈에: '
const IMPORT_HINT = "이미지 정보 전체나 한 줄, 또는 값만 — sam-extra 자기 형식(v1;mods=…), 원본 Forge 스크립트(MOD(…)), "
  + "aoleg 포크(v1;…)를 확장과 같은 규칙으로 읽는다. 앱 전용 'I2I·인페인트에도 적용' 은 그대로 둔다"
const PASTE_PLACEHOLDER = 'v1;mods=I;I.exposure=0.3'

const store = useWidgetStore()
const storeWidgets = store.widgets as Record<string, any>
const { capabilities } = useSamExtraCapabilities()

const card = computed(() => readCard(storeWidgets))
const config = computed(() => card.value.config)
const view = computed(() => overview(config.value))
const modifierTag = ref('I')
const maskTag = ref('M1')
const hiresOn = computed(() => String(storeWidgets.hires_options_group ?? '') === 'true')
const steps = computed(() => {
  const value = Number.parseInt(String(storeWidgets.steps_input ?? ''), 10)
  return Number.isFinite(value) && value > 0 ? value : 0
})
const krea2 = computed(() => isKrea2Family(storeWidgets.generation_family_combo))
const status = computed(() => cardStatus(config.value, capabilities.value, krea2.value))
const badgeText = computed(() => badge(config.value))
const infotext = computed(() => toInfotext(config.value))
const noMaskedModifier = computed(() => !config.value.modifiers.some(m =>
  m.active && takesMask(kindOf(m)) && m.mask !== MASK_NONE))
const atDefaults = computed(() => !card.value.enabled && !card.value.masking && !card.value.img2img
  && card.value.state === DEFAULT_STATE)
const copied = ref(false)
const pasteText = ref('')

function setFlag(key: Exclude<CardKey, 'state'>, value: boolean) {
  storeWidgets[WIDGET_IDS[key]] = value ? 'true' : 'false'
}

function update(tag: string, name: string, value: unknown) {
  storeWidgets[WIDGET_IDS.state] = dumpState(setPath(config.value, `${tag}.${name}`, value))
}

function reset(tag: string) {
  storeWidgets[WIDGET_IDS.state] = dumpState(resetNode(config.value, tag))
}

function resetAll() {
  storeWidgets[WIDGET_IDS.enabled] = 'false'
  storeWidgets[WIDGET_IDS.masking] = 'false'
  storeWidgets[WIDGET_IDS.img2img] = 'false'
  storeWidgets[WIDGET_IDS.state] = DEFAULT_STATE
  modifierTag.value = 'I'
  maskTag.value = 'M1'
}

/** 붙여 넣은 글을 Colorcraft 만 가져온다(액션 sam_extra_import — 파이썬이 읽고 위젯에 쓰고 토스트로 알린다). */
function applyPaste() {
  if (importSamExtra(pasteText.value, 'colorcraft')) pasteText.value = ''
}

async function copyInfotext() {
  try {
    await navigator.clipboard?.writeText(infotext.value)
    copied.value = true
    setTimeout(() => { copied.value = false }, 1500)
  } catch {
    copied.value = false
  }
}
</script>

<style scoped>
.cc-summary { margin-left: var(--sp-2); font-weight: var(--fw-medium); color: var(--state-info-fg); font-size: var(--fs-label); }
.cc-summary.cc-summary-warn { color: var(--state-warn-fg); }
.cc-switches { display: flex; flex-wrap: wrap; gap: var(--sp-2) 16px; }
.ext-note { margin: 3px 0 6px; color: var(--text-muted); font-size: var(--fs-label); line-height: 1.45; }
.ext-note.cc-status-warn, .ext-note.cc-warnline { color: var(--state-warn-fg); }
.ext-note.cc-status-ok { color: var(--state-info-fg); }
.cc-section-label { margin-top: 6px; font-size: var(--fs-label); font-weight: var(--fw-medium); color: var(--text-muted); }
.cc-io-row { display: flex; gap: var(--sp-2); align-items: center; }
.cc-infotext { flex: 1; min-width: 0; font-family: var(--font-mono, monospace); font-size: var(--fs-meta); }
.cc-paste { width: 100%; box-sizing: border-box; resize: vertical; font-family: var(--font-mono, monospace); font-size: var(--fs-meta); }
.cc-io-end { justify-content: flex-end; margin-top: 4px; }
.cc-actions { display: flex; justify-content: flex-end; margin-top: var(--sp-2); }
.cc-btn {
  height: 28px; padding: 0 12px; font-size: var(--fs-meta); font-weight: var(--fw-medium); border-radius: 5px; cursor: pointer;
  background: transparent; border: 1px dashed var(--border); color: var(--text-muted);
}
.cc-btn:hover:not(:disabled) { border-color: var(--text-muted); color: var(--text-primary); }
.cc-btn:disabled { opacity: 0.5; cursor: default; }
.cc-btn.cc-muted { opacity: 0.55; }
</style>
