<template>
  <details class="glass-card tile-repair-panel" @toggle="onToggle">
    <summary>Anima Tile &amp; Repair</summary>
    <p class="tile-help">Forge sam-extra 의 Anima ControlNet-LLLite(Tile &amp; Repair)로 원본을 다시 그립니다 — 흐림·압축 흔적을 줄이고 구도는 유지합니다. 결과는 새 PNG 로 저장하고 원본은 바꾸지 않습니다.</p>
    <p v-if="unavailable" class="tile-warning" role="status">{{ unavailable }}</p>
    <template v-else>
      <fieldset :disabled="tile.busy.value">
        <legend>모델</legend>
        <label>LLLite
          <select v-model="tile.settings.model">
            <option value="">{{ defaultChoiceLabel(options?.default_model) }}</option>
            <option v-for="name in options?.models || []" :key="name" :value="name">{{ name }}</option>
          </select>
        </label>
        <p v-if="options && !options.models.length" class="tile-warning">Forge models/ControlNet 에 3채널 Anima LLLite(예: animaTileRepair_v20)가 없습니다.</p>
        <p v-if="options && !options.available" class="tile-warning">Forge 에 Anima 벤더(sd-scripts)가 없습니다 — sam-extra install.py 를 다시 실행하세요.</p>
        <details class="tile-advanced">
          <summary>DiT · Text Encoder · VAE</summary>
          <label v-for="field in MODEL_FIELDS" :key="field.key">{{ field.label }}
            <select v-model="tile.settings[field.key]">
              <option value="">{{ defaultChoiceLabel(defaultOf(field.key)) }}</option>
              <option v-for="name in choicesOf(field.key)" :key="name" :value="name">{{ name }}</option>
            </select>
          </label>
        </details>
        <button type="button" class="tile-refresh" :disabled="tile.loadingOptions.value" @click="tile.loadOptions()">
          {{ tile.loadingOptions.value ? '불러오는 중…' : '목록 다시 받기' }}
        </button>
        <p v-if="tile.optionsError.value" class="tile-error" role="alert">{{ tile.optionsError.value }}</p>
      </fieldset>

      <fieldset :disabled="tile.busy.value">
        <legend>프롬프트</legend>
        <textarea v-model="tile.settings.prompt" rows="3" aria-label="Tile & Repair 프롬프트"></textarea>
        <textarea v-model="tile.settings.negative_prompt" rows="2" placeholder="네거티브 (기본 비움)" aria-label="Tile & Repair 네거티브"></textarea>
      </fieldset>

      <fieldset :disabled="tile.busy.value" class="tile-numbers">
        <legend>샘플링</legend>
        <label v-for="field in NUMBER_FIELDS" :key="field.key" :for="`${id}-${field.key}`">
          <span>{{ field.label }}</span>
          <input :id="`${id}-${field.key}`" type="number" :min="TILE_REPAIR_RANGES[field.key][0]" :max="TILE_REPAIR_RANGES[field.key][1]"
            :step="TILE_REPAIR_INCREMENTS[field.key]" :value="tile.settings[field.key]" @change="setNumber(field.key, $event)" />
        </label>
        <label :for="`${id}-seed`">
          <span>시드 (−1 = 랜덤)</span>
          <input :id="`${id}-seed`" type="text" inputmode="numeric" :value="tile.settings.seed" @change="setSeed" />
        </label>
        <label class="tile-check"><input v-model="tile.settings.unload_forge_before" type="checkbox" /> 실행 전에 Forge 모델 내리기 (16 GB 이하 GPU 권장)</label>
        <button type="button" class="tile-reset" @click="tile.resetSettings()">원본 기본값으로</button>
      </fieldset>
      <p class="tile-help">짧은 변을 정하면 긴 변은 원본 비율을 따릅니다(32 배수). 기본값은 원본(kohya sd-scripts · ComfyUI-Anima-LLLite)과 같습니다: 50 스텝 · CFG 3.5 · shift 5.0 · 배율 1.0.</p>

      <div class="tile-run-row">
        <button type="button" class="tile-run" :disabled="tile.busy.value || !hasSource" @click="start">
          {{ tile.busy.value ? 'Forge 에서 복원 중…' : '▶ Tile & Repair 실행' }}
        </button>
        <button v-if="tile.busy.value" type="button" class="tile-cancel" :disabled="tile.cancelling.value" @click="tile.cancel()">
          {{ tile.cancelling.value ? '멈추는 중…' : '취소' }}
        </button>
      </div>
      <p v-if="!hasSource" class="tile-help">I2I 원본 이미지를 먼저 올리세요.</p>
      <p v-if="tile.busy.value" class="tile-help">첫 실행은 Anima 모델을 디스크에서 올리느라 20~40초 더 걸립니다. Forge 에서 다른 생성이 돌고 있으면 끝난 뒤 시작합니다.</p>
      <p v-if="tile.error.value" class="tile-error" role="alert">{{ tile.error.value }}</p>
      <p class="tile-notice" role="status" aria-live="polite">{{ tile.notice.value }}</p>

      <template v-if="tile.result.value">
        <figure class="tile-result">
          <img :src="mediaUrl(tile.result.value.path)" alt="Tile & Repair 결과" />
          <figcaption>{{ tile.result.value.width }}×{{ tile.result.value.height }} · 시드 {{ tile.result.value.seed ?? '?' }} · {{ tile.result.value.model }}</figcaption>
        </figure>
        <div class="tile-actions">
          <button type="button" :disabled="tile.busy.value" @click="apply">결과를 I2I 원본으로 사용</button>
          <button type="button" :disabled="tile.result.value.seed == null" @click="tile.reuseSeed()">이 시드 고정</button>
        </div>
      </template>
    </template>
  </details>
</template>

<script setup lang="ts">
import { computed, ref, useId } from 'vue'
import { useSamExtraCapabilities } from '../composables/useSamExtraCapabilities'
import { useTileRepair } from '../composables/useTileRepair'
import { isWebMode, mediaUrl } from '../utils/media.js'
import {
  TILE_REPAIR_INCREMENTS, TILE_REPAIR_RANGES, clampTileRepairNumber, defaultChoiceLabel,
  normalizeTileRepairSeed, tileRepairSource, type TileRepairNumberKey,
} from '../utils/tileRepair'

const props = defineProps<{ imageSrc: string; imagePath: string }>()
const emit = defineEmits<{ apply: [result: { path: string }] }>()

type ModelKey = 'dit' | 'text_encoder' | 'vae'
const MODEL_FIELDS: Array<{ key: ModelKey; label: string }> = [
  { key: 'dit', label: 'DiT' }, { key: 'text_encoder', label: 'Text Encoder' }, { key: 'vae', label: 'VAE' },
]
const NUMBER_FIELDS: Array<{ key: TileRepairNumberKey; label: string }> = [
  { key: 'steps', label: '스텝' },
  { key: 'cfg_scale', label: 'CFG' },
  { key: 'flow_shift', label: 'Flow shift' },
  { key: 'multiplier', label: 'LLLite 배율' },
  { key: 'short_side', label: '짧은 변(px)' },
]

const id = `tile-repair-${useId()}`
const tile = useTileRepair()
const { capabilities } = useSamExtraCapabilities()
const options = computed(() => tile.options.value)
const hasSource = computed(() => tileRepairSource(props.imagePath, props.imageSrc) !== null)
const webMode = ref(isWebMode())

const unavailable = computed(() => {
  if (webMode.value) return 'Tile & Repair 는 로컬 앱 전용입니다. 웹 모드에서는 실행할 수 없습니다.'
  const caps = capabilities.value
  if (caps?.status === 'not_applicable') return 'Tile & Repair 는 Forge(WebUI) 백엔드에서만 쓸 수 있습니다.'
  if (caps?.known && !caps.features?.tile_repair_route) {
    return caps.installed
      ? '연결된 Forge 의 sam-extra 에 Tile & Repair 라우트가 없습니다 — 확장을 업데이트하세요.'
      : '연결된 WebUI 에 sam-extra 확장이 없습니다.'
  }
  return ''
})

function defaultOf(key: ModelKey): string | null {
  const value = options.value?.defaults?.[key]
  return typeof value === 'string' ? value : null
}
function choicesOf(key: ModelKey): string[] {
  return options.value?.[key] || []
}
function onToggle(event: Event) {
  const open = (event.target as HTMLDetailsElement).open
  if (open && !unavailable.value && !tile.options.value && !tile.loadingOptions.value) tile.loadOptions()
}
function setNumber(key: TileRepairNumberKey, event: Event) {
  const input = event.target as HTMLInputElement
  const value = clampTileRepairNumber(key, input.value)
  if (value !== null) tile.settings[key] = value
  input.value = String(tile.settings[key])
}
function setSeed(event: Event) {
  const input = event.target as HTMLInputElement
  tile.settings.seed = normalizeTileRepairSeed(input.value)
  input.value = String(tile.settings.seed)
}
function start() {
  if (unavailable.value) return
  tile.run(props.imagePath, props.imageSrc)
}
function apply() {
  const path = tile.result.value?.path
  if (path) emit('apply', { path })
}
</script>

<style scoped>
.tile-repair-panel { min-width: 0; }
.tile-repair-panel summary { cursor: pointer; color: var(--text-primary); font-size: 12px; font-weight: var(--fw-bold); }
.tile-repair-panel summary:focus-visible, .tile-repair-panel :is(input, button, select, textarea):focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.tile-help, .tile-notice { font-size: 11px; line-height: 1.5; color: var(--text-muted); overflow-wrap: anywhere; margin: 8px 0; }
.tile-warning { color: var(--state-warn-fg); font-size: 11px; line-height: 1.5; overflow-wrap: anywhere; }
.tile-error { color: var(--state-alert-fg); font-size: 11px; line-height: 1.5; overflow-wrap: anywhere; }
.tile-repair-panel fieldset { min-width: 0; margin: 12px 0; padding: 8px; border: 1px solid var(--border); border-radius: 5px; }
.tile-repair-panel legend { color: var(--text-secondary); font-size: 11px; padding: 0 4px; }
.tile-repair-panel label { display: block; margin: 4px 0 8px; color: var(--text-secondary); font-size: 11px; }
.tile-repair-panel select, .tile-repair-panel textarea, .tile-repair-panel input[type="number"], .tile-repair-panel input[type="text"] {
  display: block; width: 100%; min-width: 0; box-sizing: border-box; margin-top: 4px; padding: 6px 8px;
  background: var(--bg-input); border: 1px solid var(--border); border-radius: 5px; color: var(--text-primary); font-size: 11px;
}
.tile-repair-panel textarea { resize: vertical; margin-bottom: 6px; }
.tile-numbers { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 0 8px; }
.tile-numbers legend { grid-column: 1 / -1; }
.tile-numbers .tile-check, .tile-numbers .tile-reset { grid-column: 1 / -1; }
.tile-repair-panel .tile-check { display: flex; gap: 7px; align-items: center; color: var(--text-primary); }
.tile-check input { width: auto; accent-color: var(--accent-fill); }
.tile-advanced { margin: 6px 0; }
.tile-advanced summary { font-size: 11px; font-weight: normal; color: var(--text-secondary); }
.tile-repair-panel button { min-height: 32px; padding: 6px 10px; border: 1px solid var(--border); border-radius: 5px; background: var(--bg-button); color: var(--text-primary); font-size: 11px; cursor: pointer; white-space: normal; }
.tile-repair-panel button:disabled { opacity: 0.5; cursor: default; }
.tile-run-row { display: flex; gap: 6px; }
.tile-repair-panel .tile-run { flex: 1; background: var(--accent-fill); color: var(--on-accent); }
.tile-repair-panel .tile-cancel { border-color: var(--state-alert-fg); color: var(--state-alert-fg); background: transparent; }
.tile-result { min-width: 0; margin: 8px 0; }
.tile-result img { width: 100%; height: 180px; object-fit: contain; background: var(--bg-input); border-radius: 4px; }
.tile-result figcaption { font-size: 10px; color: var(--text-muted); line-height: 1.5; overflow-wrap: anywhere; }
.tile-actions { display: flex; gap: 6px; flex-wrap: wrap; }
.tile-actions button { flex: 1; }
@media (pointer: coarse) { .tile-repair-panel button { min-height: 44px; } }
</style>
