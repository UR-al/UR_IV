<template>
  <div class="fc-fields">
    <label class="ext-check-row" :title="toggleTitle">
      <ToggleSwitch :model-value="enabled" @update:model-value="setEnabled" size="sm" />
      <span>얼굴 안에서만 찾기 (face_yolov8 얼굴 crop 마다 SAM3)</span>
    </label>
    <template v-if="enabled">
      <div class="ext-row">
        <div class="ext-field"><label>얼굴 검출 모델 (models/adetailer)</label>
          <CustomSelect v-model="w[MODEL_ID]" :options="modelOptions" :placeholder="FACE_CROP_DEFAULT_MODEL" /></div>
        <div class="ext-field"><label>얼굴 검출 신뢰도</label>
          <input type="number" v-model="w[CONFIDENCE_ID]" step="0.01" min="0" max="1" aria-label="얼굴 검출 신뢰도" /></div>
      </div>
      <p class="fc-note">얼굴마다 넓힌 정사각 영역 안에서만 Detect·Exclude 를 찾습니다 — 작은 얼굴·여러 얼굴의 눈·입에
        좋고, 영역 밖으로 뻗은 머리카락은 잘립니다. Individual 이면 얼굴마다 따로 인페인트합니다. 얼굴을 못 찾은 이미지는
        그대로 둡니다.</p>
      <p v-if="support === 'unsupported'" class="fc-warn">연결된 Forge 의 sam-extra 가 이 기능을 모릅니다(0.34.0 이상,
        업데이트 뒤 Forge 재시작) — 켜도 전체 이미지에서 찾습니다.</p>
      <p v-else-if="support === 'comfy'" class="fc-warn">ComfyUI 에서는 아직 쓸 수 없습니다 — 켠 채로 생성하면 멈춥니다
        (Forge sam-extra 전용).</p>
    </template>
  </div>
</template>

<script setup lang="ts">
/**
 * SAM3 '얼굴 안에서만 찾기' 한 줄 — T2I SAM3 카드(Sam3MaskCard)·SAM3 Refine(RefinePanel)·배치 SAM3(BatchView) 가 같이 쓴다
 * (Sam3ControlNetPanel 과 같은 방식). `widgets` 는 widget id(`_sam3_face_crop*`) → 문자열 값 — T2I 는 위젯 스토어,
 * Refine·배치는 sam3FaceCropDefaults() 로 만든 로컬 상태다. 값 변환은 utils/sam3FaceCrop.ts·Python build_state.
 * 확장 SAM3 Mask 아코디언의 줄(Search inside faces only · SAM3 Face Detector · Confidence)과 같은 세 칸.
 */
import { computed } from 'vue'
import ToggleSwitch from './ToggleSwitch.vue'
import CustomSelect from './CustomSelect.vue'
import { getProperty } from '../stores/widgetStore.js'
import { useSamExtraCapabilities } from '../composables/useSamExtraCapabilities'
import {
  FACE_CROP_DEFAULT_MODEL, faceCropModelOptions, faceCropSupport, sam3FaceCropId,
} from '../utils/sam3FaceCrop'

const props = defineProps<{ widgets: Record<string, any> }>()
const w = computed(() => props.widgets).value

const ENABLE_ID = sam3FaceCropId('face_crop')
const MODEL_ID = sam3FaceCropId('face_crop_model')
const CONFIDENCE_ID = sam3FaceCropId('face_crop_confidence')
const toggleTitle = 'ADetailer 의 얼굴 모델(models/adetailer)로 얼굴 상자를 찾고, 상자를 넓힌 정사각 영역마다 SAM3 를 돌립니다.\n'
  + '모델은 받지 않습니다 — 파일이 없으면 생성 결과 알림이 알려 줍니다.'

const enabled = computed(() => String(w[ENABLE_ID] ?? '').trim().toLowerCase() === 'true' || w[ENABLE_ID] === true)
function setEnabled(value: boolean) {
  w[ENABLE_ID] = value ? 'true' : 'false'
}

// 추천 목록은 Python 이 ADetailer 모델 목록에서 골라 `_sam3_face_crop_model` items 로 보낸다(T2I·Refine·배치 공용)
const modelOptions = computed(() => faceCropModelOptions(getProperty(MODEL_ID, 'items'), w[MODEL_ID]))
const { capabilities } = useSamExtraCapabilities()
const support = computed(() => faceCropSupport(capabilities.value))
</script>

<style scoped>
.fc-fields { margin-bottom: 4px; }
.ext-check-row {
  display: flex; align-items: center; gap: 6px; max-width: 100%;
  margin-bottom: 6px; cursor: pointer;
  font-size: var(--fs-label); color: var(--text-secondary);
}
.ext-check-row span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ext-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.ext-field { margin-bottom: 8px; min-width: 0; }
.ext-field label {
  display: block; margin-bottom: 3px;
  font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-muted);
}
.ext-field input { width: 100%; box-sizing: border-box; }
.fc-note {
  margin: 0 0 6px; color: var(--text-muted);
  font-size: var(--fs-label); line-height: 1.45;
}
.fc-warn {
  margin: 0 0 6px; color: var(--state-warn-fg);
  font-size: var(--fs-label); line-height: 1.4;
}
</style>
