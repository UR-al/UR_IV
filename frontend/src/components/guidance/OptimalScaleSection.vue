<template>
  <label class="ext-check-row">
    <ToggleSwitch :model-value="b('ocfg_enabled')" @update:model-value="setB('ocfg_enabled', $event)" size="sm" />
    <span>Enable Anima Optimal Scale (실험)</span>
  </label>
  <template v-if="b('ocfg_enabled')">
    <div class="ext-field"><label>Optimal-scale blend (1 = 식 그대로)</label>
      <input type="number" v-model="w._ocfg_blend" step="0.05" min="0" max="1" /></div>
    <div class="ext-row">
      <div class="ext-field"><label>Start (σ 기준 %)</label>
        <input type="number" v-model="w._ocfg_start" step="0.01" min="0" max="1" /></div>
      <div class="ext-field"><label>End (σ 기준 %)</label>
        <input type="number" v-model="w._ocfg_end" step="0.01" min="0" max="1" /></div>
    </div>
    <p v-if="windowEmpty" class="ext-note">Start 가 End 보다 작아야 붙습니다 — 지금 값이면 이 보정은 돌지 않습니다.</p>
    <p class="ext-note" :title="noteTitle">CFG-Zero* 의 optimized-scale 식만 씁니다(초반 스텝을 0 으로 만드는 zero-init 은
      없음). Anima · CFG &gt; 1 전용이고, Skimmed CFG 와 함께면 건너뜁니다. SMC·APG·CWM 을 켜면 그 CFG 기반이 결과를 다시
      만들어 이 보정은 남지 않습니다.</p>
  </template>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import ToggleSwitch from '../ToggleSwitch.vue'
import { useGuidanceWidgets } from './guidanceWidgets'

/**
 * Anima Optimal Scale — 확장 scripts/anima_cfg_optimal_scale.py 의 인자 4개(2026-10-02 검토 제안 편입, 실험·기본 끔).
 * 그룹 제목은 AnimaGuidancePanel 이 든다. 칸 범위·step 은 확장 슬라이더 그대로다(blend 0~1 / .05, start·end 0~1 /
 * .01 — tests/test_guidance_detail_inputs.py 가 픽스처와 대조한다). start·end 는 확장이 predictor.percent_to_sigma 로
 * 바꾸는 σ 기준 % 이고, start ≥ end 면 확장이 붙지 않는다(빈 창).
 *
 * 조각(fragment) 컴포넌트다 — 감싸는 요소 없이 AnimaGuidancePanel 의 그룹 <details> 안에 그대로 놓인다.
 * 그래서 패널의 scoped 속성을 받지 않는다: .ag-sub · .ext-note 모양은 패널이 :deep 으로 입힌다.
 */
const props = defineProps<{ widgets: Record<string, any> }>()
const { w, b, setB } = useGuidanceWidgets(props)

// 빈 칸은 백엔드가 기본값(0 / 1)으로 보낸다 — 판단도 같은 기본값으로
const windowEmpty = computed(() => {
  const start = w._ocfg_start === '' || w._ocfg_start == null ? 0 : Number(w._ocfg_start)
  const end = w._ocfg_end === '' || w._ocfg_end == null ? 1 : Number(w._ocfg_end)
  return Number.isFinite(start) && Number.isFinite(end) && start >= end
})
const noteTitle =
  'optimized scale s* = ⟨v_c, v_u⟩/‖v_u‖² 로 바뀐 만큼을 x0 공간에서 blend 배 더합니다. 1 이면 표준 CFG 에서 '
  + '논문 식 그대로, 이 블렌드는 확장이 더한 실험 조절값입니다. Anima 화질 효과는 아직 확인 전입니다.'
</script>
