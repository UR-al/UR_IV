<template>
  <label class="ext-check-row">
    <ToggleSwitch :model-value="b('dd_enabled')" @update:model-value="setB('dd_enabled', $event)" size="sm" />
    <span>Enable Detail Daemon</span>
  </label>
  <template v-if="b('dd_enabled')">
    <label class="ext-check-row">
      <ToggleSwitch :model-value="b('dd_hires')" @update:model-value="setB('dd_hires', $event)" size="sm" />
      <span>Hires Pass (켜면 hires 패스에만 · 끄면 base 패스에만)</span>
    </label>
    <p v-if="hiresNote" class="ext-note">{{ hiresNote }}</p>
    <div class="ext-field">
      <label>Detail Amount (음수=매끈 · 양수=디테일↑)</label>
      <input type="number" v-model="w._dd_amount" step="0.01" min="-5" max="5"
        :placeholder="placeholder.dd_amount" @change="commit('dd_amount')" />
      <p class="ext-note" :title="amountTitle">ComfyUI Detail Daemon 노드의 detail_amount 와 같은 값 · 범위 ±5 · 기본 0.10</p>
    </div>
    <details class="ag-sub">
      <summary>세부 스케줄</summary>
      <div class="ext-row">
        <div class="ext-field"><label>Start</label>
          <input type="number" v-model="w._dd_start" step="0.01" min="0" max="1"
            :placeholder="placeholder.dd_start" @change="commit('dd_start')" /></div>
        <div class="ext-field"><label>End</label>
          <input type="number" v-model="w._dd_end" step="0.01" min="0" max="1"
            :placeholder="placeholder.dd_end" @change="commit('dd_end')" /></div>
      </div>
      <div class="ext-row">
        <div class="ext-field"><label>Start Offset</label>
          <input type="number" v-model="w._dd_start_offset" step="0.01" min="-1" max="1"
            :placeholder="placeholder.dd_start_offset" @change="commit('dd_start_offset')" /></div>
        <div class="ext-field"><label>End Offset</label>
          <input type="number" v-model="w._dd_end_offset" step="0.01" min="-1" max="1"
            :placeholder="placeholder.dd_end_offset" @change="commit('dd_end_offset')" /></div>
      </div>
      <div class="ext-row">
        <div class="ext-field"><label>Bias</label>
          <input type="number" v-model="w._dd_bias" step="0.01" min="0" max="1"
            :placeholder="placeholder.dd_bias" @change="commit('dd_bias')" /></div>
        <div class="ext-field"><label>Exponent</label>
          <input type="number" v-model="w._dd_exponent" step="0.05" min="0" max="10"
            :placeholder="placeholder.dd_exponent" @change="commit('dd_exponent')" /></div>
      </div>
      <div class="ext-row">
        <div class="ext-field"><label>Fade</label>
          <input type="number" v-model="w._dd_fade" step="0.05" min="0" max="1"
            :placeholder="placeholder.dd_fade" @change="commit('dd_fade')" /></div>
      </div>
      <label class="ext-check-row">
        <ToggleSwitch :model-value="b('dd_smooth')" @update:model-value="setB('dd_smooth', $event)" size="sm" />
        <span>Smooth (코사인 스무딩)</span>
      </label>
    </details>
  </template>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import ToggleSwitch from '../ToggleSwitch.vue'
import { useSamExtraCapabilities } from '../../composables/useSamExtraCapabilities'
import { useGuidanceWidgets } from './guidanceWidgets'
import { DD_FLOAT_INPUTS, commitDdInput, ddPlaceholder, type DdFloatKey } from '../../utils/detailDaemonInputs'

/**
 * Detail Daemon — 확장 anima_detail_daemon.py 의 인자. 그룹 제목은 AnimaGuidancePanel 이 든다.
 *
 * 칸·범위·기본값은 원본 노드 Jonseed/ComfyUI-Detail-Daemon(DetailDaemonSamplerNode INPUT_TYPES)과 같고
 * (표: utils/detailDaemonInputs.ts — min/max/step 은 템플릿에 글자로 둔다: tests/test_detail_daemon_origin.py 가
 * 소스에서 읽고, DetailDaemonSection.test.ts 가 렌더 결과를 그 표와 대조한다), 값은 그 단위 그대로 저장하고
 * 그대로 보낸다(core/anima_guidance.py
 * DETAIL_DAEMON_SPEC — 변환·프리셋 없음, 엔진이 ×0.1×CFG 를 곱한다). 라벨과 칸 순서는
 * muerrilla/sd-webui-detail-daemon 의 UI(Hires Pass → Detail Amount → Start/End → Offset → Bias/Exponent → Fade
 * → Smooth)를 따른다. 노드에 없는 Hires Pass 도 muerrilla 와 같다(기본 끔).
 * 빈 칸은 placeholder 의 기본값이 나가고, 범위 밖 값은 확정(change) 때 노드 범위로 잘린다 — 백엔드와 같은 규칙.
 *
 * 조각(fragment) 컴포넌트다 — 감싸는 요소 없이 AnimaGuidancePanel 의 그룹 <details> 안에 그대로 놓인다.
 * 그래서 패널의 scoped 속성을 받지 않는다: .ag-sub · .ext-note 모양은 패널이 :deep 으로 입힌다.
 */
const props = defineProps<{ widgets: Record<string, any> }>()
const { w, b, setB } = useGuidanceWidgets(props)

const placeholder = Object.fromEntries(
  (Object.keys(DD_FLOAT_INPUTS) as DdFloatKey[]).map(key => [key, ddPlaceholder(key)]),
) as Record<DdFloatKey, string>

function commit(key: DdFloatKey) {
  commitDdInput(w, key)
}

const amountTitle =
  'σ × (1 − amount × 0.1 × CFG) — Forge sam-extra 화면·원본 노드와 같은 숫자입니다. 범위 ±5(노드와 같다), '
  + 'Start/End Offset 은 ±1. 빈 칸은 기본값을 보내고, 범위 밖 값은 범위 끝으로 잘립니다.'

// Hires Pass 가 그대로 적용되지 않는 백엔드 — Hires Pass 칸(arg 13)이 없는 옛 sam-extra 뿐이다.
// ComfyUI(status 'not_applicable')는 컴파일러가 dd_hires 를 따른다(DD-C: 켜면 hires 패스에만 DD 모델) — 경고 없음.
const { capabilities } = useSamExtraCapabilities()
const hiresNote = computed(() => {
  if (!b('dd_hires')) return ''
  const caps = capabilities.value
  if (caps?.known && caps.detail_daemon_hires === false) {
    return '연결된 sam-extra 에 Hires Pass 칸이 없습니다 — 모든 패스에 적용됩니다. 확장을 업데이트하세요.'
  }
  return ''
})
</script>
