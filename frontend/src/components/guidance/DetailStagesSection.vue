<template>
  <p class="ext-note">PAG/SEG/SLG 항 뒤, DCW 앞에서 HiFlow → Momentum → HiGS → TSR 순서로 돕니다. 추가 모델 호출이 없고
    모두 기본 꺼짐입니다. Anima 화질 비교는 아직 없습니다 — 고정 시드로 하나씩 비교하세요.</p>
  <p v-if="detailNote" class="ext-note">{{ detailNote }}</p>

  <label class="ext-check-row">
    <ToggleSwitch :model-value="b('guid_tsr_enabled')" @update:model-value="setB('guid_tsr_enabled', $event)" size="sm" />
    <span>Enable TSR (Temporal Score Rescaling)</span>
  </label>
  <template v-if="b('guid_tsr_enabled')">
    <div class="ext-row">
      <div class="ext-field"><label>TSR k (1 = 끔 · 낮을수록 디테일)</label>
        <input type="number" v-model="w._guid_tsr_k" step="0.005" min="0.5" max="1.5" /></div>
      <div class="ext-field"><label>TSR sigma (클수록 일찍 적용)</label>
        <input type="number" v-model="w._guid_tsr_sigma" step="0.05" min="0.1" max="10" /></div>
    </div>
    <p class="ext-note" :title="tsrTitle">원본 노드 기본 k 0.95 · sigma 1.0(논문 SD3 최적 0.93 · 3.0). 지저분해지면 k 를
      1 쪽으로 올리세요.</p>
  </template>

  <label class="ext-check-row">
    <ToggleSwitch :model-value="b('guid_mg_enabled')" @update:model-value="setB('guid_mg_enabled', $event)" size="sm" />
    <span>Enable Momentum Guidance</span>
  </label>
  <template v-if="b('guid_mg_enabled')">
    <div class="ext-row">
      <div class="ext-field"><label>MG α (밀어내는 세기)</label>
        <input type="number" v-model="w._guid_mg_alpha" step="0.05" min="0" max="3" /></div>
      <div class="ext-field"><label>MG β (지수평균 기억)</label>
        <input type="number" v-model="w._guid_mg_beta" step="0.05" min="0" max="0.95" /></div>
    </div>
    <label class="ext-check-row">
      <ToggleSwitch :model-value="b('guid_mg_normalize')" @update:model-value="setB('guid_mg_normalize', $event)" size="sm" />
      <span>Normalize momentum (‖m‖ 을 ‖v‖ 에 맞춤 · 논문 §8.2)</span>
    </label>
    <div class="ext-row">
      <div class="ext-field"><label>MG window min (노이즈 수준)</label>
        <input type="number" v-model="w._guid_mg_min" step="0.01" min="0" max="1" /></div>
      <div class="ext-field"><label>MG window max (노이즈 수준)</label>
        <input type="number" v-model="w._guid_mg_max" step="0.01" min="0" max="1" /></div>
    </div>
    <p class="ext-note" :title="mgTitle">CFG 가 낮을수록 효과가 크고, 2차·멀티스텝 샘플러는 이미 외삽을 해서 겹칩니다.
      HiGS 와 같은 기록을 써서 함께 켜면 두 번 셉니다.</p>
  </template>

  <label class="ext-check-row">
    <ToggleSwitch :model-value="b('guid_higs_enabled')" @update:model-value="setB('guid_higs_enabled', $event)" size="sm" />
    <span>Enable HiGS (History-Guided Sampling)</span>
  </label>
  <template v-if="b('guid_higs_enabled')">
    <div class="ext-row">
      <div class="ext-field"><label>HiGS weight w</label>
        <input type="number" v-model="w._guid_higs_weight" step="0.05" min="0" max="3" /></div>
      <div class="ext-field"><label>HiGS η (예측 방향 성분 비중)</label>
        <input type="number" v-model="w._guid_higs_eta" step="0.05" min="0" max="1" /></div>
    </div>
    <p class="ext-note">논문 기본 w 1.75. Res Multistep 같은 멀티스텝 샘플러에서는 1.75 가 이미지를 무너뜨렸습니다(Anima 3.8B
      실측) — 0.5 이하부터 시작하세요. η 0 = 예측과 수직 성분만(과채도 방지).</p>
    <details class="ag-sub">
      <summary>HiGS 세부값</summary>
      <div class="ext-row">
        <div class="ext-field"><label>History α (지수평균 갱신)</label>
          <input type="number" v-model="w._guid_higs_alpha" step="0.05" min="0.05" max="0.95" /></div>
        <div class="ext-field"><label>High-pass cutoff R_c</label>
          <input type="number" v-model="w._guid_higs_cutoff" step="0.005" min="0" max="0.5" /></div>
      </div>
      <div class="ext-row">
        <div class="ext-field"><label>t min (노이즈 수준 · 이하에서 끔)</label>
          <input type="number" v-model="w._guid_higs_t_min" step="0.01" min="0" max="1" /></div>
        <div class="ext-field"><label>t max (노이즈 수준)</label>
          <input type="number" v-model="w._guid_higs_t_max" step="0.01" min="0" max="1" /></div>
      </div>
    </details>
  </template>

  <label class="ext-check-row">
    <ToggleSwitch :model-value="b('guid_hiflow_enabled')" @update:model-value="setB('guid_hiflow_enabled', $event)" size="sm" />
    <span>Enable HiFlow (Hires.fix 전용)</span>
  </label>
  <template v-if="b('guid_hiflow_enabled')">
    <p v-if="hiresOff" class="ext-note">Hires.fix 가 꺼져 있어 HiFlow 는 아무것도 하지 않습니다 — T2I 에서 Hires.fix 를 켜세요.</p>
    <div class="ext-row">
      <div class="ext-field"><label>HiFlow direction α</label>
        <input type="number" v-model="w._guid_hiflow_alpha" step="0.05" min="0" max="2" /></div>
      <div class="ext-field"><label>HiFlow acceleration β</label>
        <input type="number" v-model="w._guid_hiflow_beta" step="0.05" min="0" max="1" /></div>
    </div>
    <div class="ext-field"><label>HiFlow low-pass cutoff D (Butterworth)</label>
      <input type="number" v-model="w._guid_hiflow_cutoff" step="0.01" min="0.05" max="1" /></div>
    <p class="ext-note" :title="hiflowTitle">1차 패스의 x0 궤적에 hires 패스의 저주파(방향)와 스텝 간 변화(가속도)를 맞춰 1차
      구도를 지킵니다. 가중치는 hires 스텝 동안 줄어듭니다.</p>
  </template>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import ToggleSwitch from '../ToggleSwitch.vue'
import { useSamExtraCapabilities } from '../../composables/useSamExtraCapabilities'
import { detailSuiteNote } from '../../utils/guidanceDetailSuite'
import { useGuidanceWidgets } from './guidanceWidgets'

/**
 * 디테일 단계 — TSR · Momentum Guidance · HiGS · HiFlow(확장 anima_safe_pag.py 인자 71-90, v0.30 디테일 묶음).
 * 그룹 제목은 AnimaGuidancePanel 이 든다. 칸 범위·step 은 확장 슬라이더 그대로다(tests/test_guidance_detail_inputs.py
 * 가 픽스처와 대조한다). 출처: TSR = ComfyUI nodes_eps.py(arXiv 2510.01184), Momentum = arXiv 2602.20360,
 * HiGS = arXiv 2509.22300, HiFlow = Bujiazi/HiFlow(arXiv 2504.06232) — 셋은 논문 식, HiFlow 는 공식 코드를 옮겼다.
 *
 * 조각(fragment) 컴포넌트다 — 감싸는 요소 없이 AnimaGuidancePanel 의 그룹 <details> 안에 그대로 놓인다.
 * 그래서 패널의 scoped 속성을 받지 않는다: .ag-sub · .ext-note 모양은 패널이 :deep 으로 입힌다.
 */
const props = defineProps<{ widgets: Record<string, any> }>()
const { w, b, setB } = useGuidanceWidgets(props)

const tsrTitle =
  'SNR 에 따라 예측 점수를 다시 배율합니다. k < 1 이면 노이즈가 거의 없을 때 예측 노이즈를 조금 더 남겨 잔디테일이 '
  + '늘고, k > 1 이면 매끈해집니다. sigma 는 언제부터 적용할지(원본 1.0 = Anima 에서 σ≈0.5 부터).'
const mgTitle =
  '앞 스텝 속도의 지수평균에서 멀어지는 쪽으로 현재 속도를 밉니다(D + α·σ·(v − m)). 창은 노이즈 수준(flow σ)이고 '
  + '논문 t∈[0.05, 0.7] 이 σ 0.30–0.95 입니다.'
const hiflowTitle =
  '공식 설정은 1차 30스텝 · hires 16스텝 · denoise 0.53 근처입니다. 1차 구도에 너무 묶이면 α 를, 얼룩지면 β 를 '
  + '낮추세요. cutoff 는 공식 코드 0.2(논문 본문 0.4).'

// 연결된 Forge 의 sam-extra 가 디테일 단계(인자 71-90)를 모르면, 하나라도 켰을 때 안내한다
const { capabilities } = useSamExtraCapabilities()
const anyOn = computed(() => ['guid_tsr_enabled', 'guid_mg_enabled', 'guid_higs_enabled', 'guid_hiflow_enabled']
  .some(key => b(key)))
const detailNote = computed(() => (anyOn.value ? detailSuiteNote(capabilities.value) : ''))

// HiFlow 는 T2I Hires.fix 의 hires 패스에서만 돈다(확장 _hiflow_attach: enable_hr · txt2img)
const hiresOff = computed(() => String(w.hires_options_group ?? '') !== 'true')
</script>
