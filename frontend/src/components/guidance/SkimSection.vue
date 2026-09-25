<template>
  <label class="ext-check-row">
    <ToggleSwitch :model-value="b('skim_enabled')" @update:model-value="setB('skim_enabled', $event)" size="sm" />
    <span>Enable Skimmed CFG</span>
  </label>
  <template v-if="b('skim_enabled')">
    <div class="ext-field"><label>Skimming CFG (-1 = 현재 CFG)</label>
      <input type="number" v-model="w._skim_skimming_cfg" step="0.5" min="-1" max="10" /></div>
    <label class="ext-check-row">
      <ToggleSwitch :model-value="b('skim_full_skim_negative')" @update:model-value="setB('skim_full_skim_negative', $event)" size="sm" />
      <span>Full skim negative</span>
    </label>
    <label class="ext-check-row">
      <ToggleSwitch :model-value="b('skim_disable_flipping_filter')" @update:model-value="setB('skim_disable_flipping_filter', $event)" size="sm" />
      <span>Disable flipping filter</span>
    </label>
    <div class="ext-row">
      <div class="ext-field"><label :title="windowTitle">Start at (σ 기준 %)</label>
        <input type="number" v-model="w._skim_start_percent" step="0.01" min="0" max="1" /></div>
      <div class="ext-field"><label :title="windowTitle">End at (σ 기준 %)</label>
        <input type="number" v-model="w._skim_end_percent" step="0.01" min="0" max="1" /></div>
    </div>
    <div class="ext-field"><label :title="flipTitle">Flip at (σ 기준 %) · 0 = 사용 안 함</label>
      <input type="number" v-model="w._skim_flip_at" step="0.01" min="0" max="1" /></div>
    <div class="ext-note">%는 스텝 수가 아니라 노이즈 스케줄(σ) 위치입니다 — 원본 노드와 같습니다. Start 0 · denoise 1
      (txt2img 첫 패스)이면 Anima 는 첫 스텝의 σ 가 시작점(0%)의 σ 와 같아서 원본처럼 첫 스텝은 깎지 않습니다.
      hires·img2img 처럼 denoise 가 1 보다 작은 패스는 첫 σ 가 그보다 작아 첫 스텝부터 깎습니다.</div>
  </template>
</template>

<script setup lang="ts">
import ToggleSwitch from '../ToggleSwitch.vue'
import { useGuidanceWidgets } from './guidanceWidgets'

/**
 * Skimmed CFG — anti-burn(원본: Extraltodeus/Skimmed_CFG, Apache-2.0). 그룹 제목은 AnimaGuidancePanel 이 든다.
 *
 * 칸 범위·step 은 원본 주 노드 CFG_Skimming_Single_Scale_Pre_CFG 와 같다: skimming_cfg step 0.5(MAX_SCALE 10,
 * STEP_STEP 2), start/end/flip 0~1 step 0.01 (origin: Extraltodeus/Skimmed_CFG@d8300583:skimmed_CFG.py:5-6,
 * :93-134). skimming_cfg 의 −1(현재 CFG)은 원본 Clean Skim / Timed flip 노드가 넘기는 값(:204-282)이라 칸이 허용한다
 * — 칸 하나로 세 노드를 대신하는 호스트 차이. 구간은 원본처럼 σ 로 잰다: 패치 때 percent_to_sigma 로 start·end·flip
 * 의 σ 를 구하고, end_σ < σ < start_σ 일 때만 깎는다(엄격 부등호, :153-172). flow 모델(Anima)은
 * percent_to_sigma(0) = 1.0 = 첫 σ 라 denoise 1 패스에서는 첫 스텝이 빠진다 — denoise < 1(hires·img2img)은 잘린
 * 스케줄의 첫 σ 가 1.0 보다 작아 원본도 첫 스텝을 깎는다(예: Comfy shift 3 · simple 20스텝 · denoise 0.5 → 첫 σ 0.75). flip 은 σ > flip_σ(flip 지점 전)에서 필터를 뒤집는다(:178-179).
 * guidanceOriginInputs.test.ts 가 범위와 안내를 지킨다.
 *
 * 조각(fragment) 컴포넌트다 — 감싸는 요소 없이 AnimaGuidancePanel 의 그룹 <details> 안에 그대로 놓인다.
 * 그래서 패널의 scoped 속성을 받지 않는다: .ag-sub · .ext-note 모양은 패널이 :deep 으로 입힌다.
 */
const props = defineProps<{ widgets: Record<string, any> }>()
const { w, b, setB } = useGuidanceWidgets(props)

const windowTitle =
  '0 = 시작(σ 최대), 1 = 끝. 모델의 σ 스케줄에서 이 위치의 σ 를 구해, σ 가 End 의 σ 보다 크고 Start 의 σ 보다 '
  + '작은 스텝만 깎습니다(경계는 제외) — 같은 값이라도 스케줄러·denoise 에 따라 깎이는 스텝 수가 달라집니다.'
const flipTitle =
  '이 위치의 σ 보다 σ 가 큰 스텝(이 지점 전)에서 flipping filter 를 반대로 씁니다. 0 = 사용 안 함, step 0.01'
</script>
