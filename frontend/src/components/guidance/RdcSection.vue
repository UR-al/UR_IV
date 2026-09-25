<template>
  <label class="ext-check-row">
    <ToggleSwitch :model-value="b('guid_rdc_enabled')" @update:model-value="setB('guid_rdc_enabled', $event)" size="sm" />
    <span>Enable RDC (band-wise reverse drift compensation)</span>
  </label>
  <template v-if="b('guid_rdc_enabled')">
    <div class="ext-field"><label :title="tauTitle">RDC tau (EMA 기억 구간 · 0 = 끔 · 원본 기본 0)</label>
      <input type="number" v-model="w._guid_rdc_tau" step="0.01" min="0" max="0.5" /></div>
    <div class="ext-row">
      <div class="ext-field"><label>RDC alpha LL (구조 drift)</label>
        <input type="number" v-model="w._guid_rdc_alpha_ll" step="0.005" min="0" max="0.3" /></div>
      <div class="ext-field"><label>RDC alpha HH (텍스처 drift)</label>
        <input type="number" v-model="w._guid_rdc_alpha_hh" step="0.001" min="0" max="0.1" /></div>
    </div>
    <div class="ext-note">원본에는 따로 켜는 스위치가 없고 tau 가 0 보다 클 때만 켜집니다 — tau 0 이면 이 스위치를 켜도
      RDC 는 꺼져 있습니다. 시작값: tau 0.05–0.1(빠른 반응)~0.2–0.3(느리고 부드러움) · LL 0.02–0.05 ·
      HH 0(텍스처가 흐려짐 — 쓰더라도 0.01 이하).</div>
    <div v-if="!b('guid_dcw_enabled')" class="ext-note">DCW 가 꺼져 있어 RDC 가 적용되지 않습니다 — 원본처럼 RDC 는
      DCW 보정 안에서 돕니다. 위의 Enable DCW 를 켜세요(lambda 를 0 으로 두면 RDC 만 씁니다).</div>
  </template>
</template>

<script setup lang="ts">
import ToggleSwitch from '../ToggleSwitch.vue'
import { useGuidanceWidgets } from './guidanceWidgets'

/**
 * RDC — band-wise reverse drift compensation(원본: namemechan/ComfyUI-DCW). "DCW / RDC / DAVE / CNS" 그룹.
 *
 * 원본은 켜기 스위치 없이 rdc_tau 하나로 켜고(0 = 끔, 기본 0), RDC 는 DCW 훅 안에서 돌아 dcw_enabled 가
 * 꺼지면 같이 꺼진다(origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:757-815 — 숫자·뜻만 옮겼다, GPL 코드는
 * 복사하지 않는다). 앱의 Enable RDC 스위치는 저장된 값과 팩 스위트(guid_rdc_enabled)가 읽는 자리라 남기고,
 * 켜짐 = 스위치 · DCW · tau > 0 셋 다임을 안내한다. guidanceOriginInputs.test.ts 가 범위를 지킨다.
 *
 * 조각(fragment) 컴포넌트다 — 감싸는 요소 없이 AnimaGuidancePanel 의 그룹 <details> 안에 그대로 놓인다.
 * 그래서 패널의 scoped 속성을 받지 않는다: .ag-sub · .ext-note 모양은 패널이 :deep 으로 입힌다.
 */
const props = defineProps<{ widgets: Record<string, any> }>()
const { w, b, setB } = useGuidanceWidgets(props)

// 툴팁은 사실만 적는다 — 범위 · 원본 기본값 · 방향. 원본(GPL) 툴팁 문장을 옮기지 않는다(번역도 하지 않는다).
const tauTitle =
  '0 = RDC 끔(원본 기본), 0 보다 크면 켬. 값은 EMA 기억 길이(σ_norm 단위) — 클수록 예전 스텝을 오래 기억해 '
  + '천천히 보정함. 범위 0~0.5 step 0.01'
</script>
