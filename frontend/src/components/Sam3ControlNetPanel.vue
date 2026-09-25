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
        <div class="cn-model-row">
          <CustomSelect v-if="liveModels && !modelTyping" class="cn-model-input" v-model="w[id('cn_model')]"
            :options="modelOptions" placeholder="None" />
          <input v-else class="cn-model-input" type="text" v-model="w[id('cn_model')]" placeholder="None"
            spellcheck="false" aria-label="ControlNet 모델 이름" />
          <button v-if="liveModels" type="button" class="cn-icon-btn" :aria-pressed="modelTyping"
            :title="modelTyping ? 'Forge 목록에서 고르기' : '목록에 없는 이름 직접 입력 (models/sam3 에 새로 넣은 LLLite 등)'"
            @click="modelTyping = !modelTyping"><Icon :name="modelTyping ? 'chevron-down' : 'pencil'" /></button>
          <button v-if="canRefresh" type="button" class="cn-icon-btn"
            title="연결된 Forge 에서 ControlNet 모델·전처리기 목록 다시 받기 (30초에 한 번)"
            @click="refresh()"><Icon name="refresh" /></button>
        </div>
        <p v-if="modelMissing" class="cn-warn">연결 때 받은 Forge 목록에 없는 모델입니다. models/sam3 의 파일(LLLite 등)이면
          확장이 생성 때 다시 찾지만, 그 밖의 이름이면 SAM3 패스가 실패합니다(KeyError). 새로 넣은 모델이면 목록을 다시 받아 보세요.</p>
      </div>
      <div class="ext-field"><label>Module (전처리기)</label>
        <CustomSelect v-model="w[id('cn_module')]" :options="moduleOptions" placeholder="inpaint_only" />
        <p v-if="moduleMissing" class="cn-warn">연결된 Forge 에 없는 전처리기입니다 — SAM3 패스가 실패합니다. 목록에서 다시 고르세요.</p>
        <p v-if="llliteTile" class="cn-info">Anima Tile &amp; Repair LLLite 모델입니다. 원본(kohya
          sd-scripts·ComfyUI-Anima-LLLite)처럼 고칠 그림을 그대로 받으므로 전처리기는 None 으로 고정합니다
          (inpaint_only 는 고칠 영역을 비웁니다). 확장도 모델 파일로 확인해 같은 값으로 돌립니다.</p>
        <p v-else-if="llliteChannels === 3" class="cn-info">3채널 Anima LLLite 모델입니다. 제어 이미지는 인페인트 입력
          이미지라 lineart·canny·depth 모델이면 맞는 전처리기를 고르세요. 원본은 마스크를 쓰지 않으므로 inpaint_*
          전처리기(마스크 영역을 비움)만 None 으로 바꿉니다.</p>
        <p v-else-if="llliteChannels === 4" class="cn-info">LLLite 인페인트(4채널) 모델입니다. inpaint_* 전처리기는
          마스크를 버려 LLLite 가 실패하므로 None 으로 바꿉니다.</p>
      </div>

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
import { computed, ref, watch } from 'vue'
import ToggleSwitch from './ToggleSwitch.vue'
import CustomSelect from './CustomSelect.vue'
import { getProperty } from '../stores/widgetStore.js'
import { useSamExtraCapabilities } from '../composables/useSamExtraCapabilities'
import {
  CN_NONE, SAM3_CN_FIELDS, canonicalCnName, choiceOptions, cnModelOptions, cnModuleOptions,
  cnModuleOptionsForModel, cnNameMissing, isTrue, sam3CnId, sam3CnLiveLists, sam3CnLlliteChannels,
  sam3CnLlliteModule, sam3CnLlliteTileRepair,
} from '../utils/sam3ControlNet'

/**
 * SAM3 ControlNet 13필드 패널 (Forge 확장의 SAM3 > ControlNet 아코디언과 1:1).
 *
 * `widgets` 는 widget id(`_sam3_cn_*`) → 문자열 값 객체다.
 *  - T2I: 위젯 스토어(storeWidgets) — 값이 Python 프록시와 동기화되고 설정 저장/복원된다.
 *  - SAM3 Refine·배치 SAM3: `sam3CnDefaults()` 로 만든 로컬 reactive — `sam3CnSettings` 로 전송.
 * control/resize mode 선택지는 Python(core/sam3_controlnet → sam3_args)이 T2I 프록시 items 로 보낸
 * 목록 하나를 세 곳이 같이 쓴다.
 *
 * 전처리기·모델은 연결된 Forge 의 라이브 목록(sam-extra 기능 스냅샷의 /controlnet/module_list ·
 * model_list — models/sam3 LLLite 포함)이 먼저다. 모르면 전처리기는 Python 정적 폴백 items, 모델은
 * 자유 입력. Forge 는 이름을 대소문자까지 그대로 찾아서(소문자 'none' = KeyError → SAM3 패스 실패)
 * 대소문자만 다른 값은 목록 표기로 고치고, 라이브 목록에 없는 값은 경고한다(P3).
 * 모델 목록은 연결 때 것이고 확장은 SAM3 패스마다 models/sam3 를 다시 찾으므로, 목록을 알아도 연필 버튼으로
 * 직접 입력할 수 있고 경고도 '실패'라고 단정하지 않는다. ↻ 는 스냅샷을 다시 받는다(useSamExtraCapabilities.refresh).
 * Anima ControlNet-LLLite 모델은 원본(kohya)처럼 사용자가 준 제어 이미지를 그대로 받는다 — Tile & Repair 면 전처리기를
 * None 으로 고정하고, 그 밖의 Anima LLLite(3채널 lineart·canny·depth, 4채널 인페인트)는 inpaint_* 만 None 으로
 * 바꾼다(sam3CnLlliteModule, 저장값 inpaint_only 보정). lineart_anime 같은 다른 전처리기는 고른 그대로 둔다.
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

const { capabilities, refresh } = useSamExtraCapabilities()
const live = computed(() => sam3CnLiveLists(capabilities.value))
// ComfyUI(not_applicable)는 Forge 목록이 없다 — 다시 받을 것도 없다.
const canRefresh = computed(() => capabilities.value?.status !== 'not_applicable')
// 라이브 모델 목록이 있어도 목록 밖 이름(시작 뒤 models/sam3 에 넣은 LLLite 등)을 칠 수 있게.
const modelTyping = ref(false)

// 라이브 목록 → 정적 폴백 items → (선택지가 아직 안 왔으면 getProperty 는 '') 현재 값 하나.
const allModuleOptions = computed(() =>
  cnModuleOptions(live.value.modules, getProperty('_sam3_cn_module', 'items'), w[id('cn_module')]))
// Anima LLLite 모델이면 원본처럼 — Tile & Repair 는 'None' 하나, 그 밖의 Anima LLLite 는 inpaint_* 를 뺀다.
const moduleOptions = computed(() => cnModuleOptionsForModel(allModuleOptions.value, w[id('cn_model')]))
const llliteChannels = computed(() => sam3CnLlliteChannels(w[id('cn_model')]))
const llliteTile = computed(() => sam3CnLlliteTileRepair(w[id('cn_model')]))
// 라이브 모델 목록이 있으면 드롭다운(+ 직접 입력 전환), 모르면 자유 입력(null).
const liveModels = computed(() => live.value.models)
const modelOptions = computed(() => cnModelOptions(liveModels.value, w[id('cn_model')]))
const moduleMissing = computed(() => cnNameMissing(w[id('cn_module')], live.value.modules))
const modelMissing = computed(() => cnNameMissing(w[id('cn_model')], live.value.models))

// 대소문자만 다른 값(예전 저장값 'none' 등)은 목록 표기로 — 목록 밖 이름은 바꾸지 않는다.
function fixCase(key: string, options: readonly string[]) {
  const current = w[id(key)]
  const next = canonicalCnName(current, options)
  if (next !== current) w[id(key)] = next
}
// 저장값·기본값 inpaint_only 가 Anima LLLite 와 함께 남아 있으면(Tile & Repair 는 어떤 전처리기든) None 으로 — 파이썬
// core/sam3_cn_names.guard_lllite_module · 확장 inject_controlnet_unit 과 같은 규칙(대소문자를 맞춘 뒤에).
function guardLllite() {
  const guard = sam3CnLlliteModule(w[id('cn_module')], w[id('cn_model')])
  if (guard.forced) w[id('cn_module')] = guard.module
}
watch(
  () => [allModuleOptions.value, liveModels.value, w[id('cn_module')], w[id('cn_model')]] as const,
  ([modules, models]) => {
    fixCase('cn_module', modules)
    fixCase('cn_model', models ?? [CN_NONE])
    guardLllite()
  },
  { immediate: true },
)
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
.cn-model-row { display: flex; align-items: stretch; gap: 6px; }
.cn-model-input { flex: 1 1 auto; min-width: 0; }
.cn-icon-btn {
  flex: 0 0 auto; width: 32px; padding: 0;
  display: inline-flex; align-items: center; justify-content: center;
  background: transparent; border: 1px solid var(--border); border-radius: var(--radius-base);
  color: var(--text-muted); cursor: pointer;
}
.cn-icon-btn:hover, .cn-icon-btn[aria-pressed="true"] { border-color: var(--text-muted); color: var(--text-primary); }
.cn-icon-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.cn-warn {
  margin: 4px 0 0; color: var(--state-warn-fg);
  font-size: var(--fs-label); line-height: 1.4;
}
.cn-info {
  margin: 4px 0 0; color: var(--text-muted);
  font-size: var(--fs-label); line-height: 1.4;
}
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
