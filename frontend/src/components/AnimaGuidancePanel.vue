<template>
  <details class="ext-card">
    <summary class="ext-title">
      ANIMA 가이던스
      <span v-if="activeSummary" class="ag-badge" :title="activeSummary">{{ activeSummary }}</span>
    </summary>

    <p class="ag-note">
      Anima/Cosmos/Predict2 계열 <b>DiT 전용</b> guidance. 전부 OFF면 Forge 결과 그대로.
      다른 엔진에서는 확장이 알아서 폴백합니다.
    </p>

    <!-- ── PAG / SEG / SLG ─────────────────────────────────────────── -->
    <details class="ag-group">
      <summary>PAG / SEG / SLG — Attention perturbation</summary>

      <label class="ext-check-row">
        <ToggleSwitch :model-value="b('guid_enabled')" @update:model-value="setB('guid_enabled', $event)" size="sm" />
        <span>Enable Perturbation Guidance</span>
      </label>

      <template v-if="b('guid_enabled')">
        <div class="ext-field"><label>Attention method (PAG/SEG는 택1)</label>
          <CustomSelect v-model="w._guid_attn_method" :options="['PAG', 'SEG', 'None']" placeholder="PAG" /></div>
        <div class="ext-row">
          <div class="ext-field"><label>Attn Scale (cond−weak 배율)</label>
            <input type="number" v-model="w._guid_scale" step="0.1" min="0" max="15" /></div>
          <div class="ext-field"><label>Perturbation strength (1=전체)</label>
            <input type="number" v-model="w._guid_official_strength" step="0.01" min="0" max="1" /></div>
        </div>
        <div class="ext-field" v-if="w._guid_attn_method === 'SEG'">
          <label>SEG query blur sigma (&gt;9999 = uniform)</label>
          <input type="number" v-model="w._guid_seg_sigma" step="1" min="0" max="10000" /></div>
        <div class="ext-row">
          <div class="ext-field"><label>Block indices (기본 18)</label>
            <input type="text" v-model="w._guid_block_indices" placeholder="18 또는 18-20" /></div>
          <div class="ext-field"><label>Head indices (빈칸=전체)</label>
            <input type="text" v-model="w._guid_head_indices" placeholder="0,2,4-7" /></div>
        </div>

        <label class="ext-check-row">
          <ToggleSwitch :model-value="b('guid_slg_on')" @update:model-value="setB('guid_slg_on', $event)" size="sm" />
          <span>Enable SLG (skip layers · PAG/SEG와 병용 가능)</span>
        </label>
        <div class="ext-row" v-if="b('guid_slg_on')">
          <div class="ext-field"><label>SLG scale</label>
            <input type="number" v-model="w._guid_slg_scale" step="0.1" min="0" max="15" /></div>
          <div class="ext-field"><label>SLG skip blocks</label>
            <input type="text" v-model="w._guid_slg_blocks" placeholder="18" /></div>
        </div>

        <div class="ext-row">
          <div class="ext-field"><label>Start percent</label>
            <input type="number" v-model="w._guid_start_percent" step="0.01" min="0" max="1" /></div>
          <div class="ext-field"><label>End percent</label>
            <input type="number" v-model="w._guid_end_percent" step="0.01" min="0" max="1" /></div>
        </div>
        <div class="ext-row">
          <div class="ext-field"><label>Rescale (과대비 억제)</label>
            <input type="number" v-model="w._guid_rescale" step="0.01" min="0" max="1" /></div>
          <div class="ext-field"><label>Rescale mode</label>
            <CustomSelect v-model="w._guid_rescale_mode" :options="['full', 'partial']" placeholder="full" /></div>
        </div>

        <details class="ag-sub">
          <summary>Legacy Soft/Approx 호환</summary>
          <label class="ext-check-row">
            <ToggleSwitch :model-value="b('guid_legacy_attn')" @update:model-value="setB('guid_legacy_attn', $event)" size="sm" />
            <span>기존 Soft PAG / SEG-Approx 사용</span>
          </label>
          <div class="ext-field" v-if="b('guid_legacy_attn')"><label>Legacy strength</label>
            <input type="number" v-model="w._guid_legacy_strength" step="0.01" min="0" max="1" /></div>
        </details>
      </template>
    </details>

    <!-- ── CFG base: APG / CWM / SMC ────────────────────────────────── -->
    <details class="ag-group">
      <summary>APG / CWM / SMC — CFG base</summary>

      <label class="ext-check-row">
        <ToggleSwitch :model-value="b('guid_apg_enabled')" @update:model-value="setB('guid_apg_enabled', $event)" size="sm" />
        <span>Enable APG (실험 · CFG &gt; 1)</span>
      </label>
      <template v-if="b('guid_apg_enabled')">
        <div class="ext-row">
          <div class="ext-field"><label>APG eta</label>
            <input type="number" v-model="w._guid_apg_eta" step="0.05" min="-10" max="10" /></div>
          <div class="ext-field"><label>APG norm (0=off)</label>
            <input type="number" v-model="w._guid_apg_norm" step="0.5" min="0" max="50" /></div>
        </div>
        <div class="ext-field"><label>APG momentum (음수 권장 · 0=off)</label>
          <input type="number" v-model="w._guid_apg_momentum" step="0.05" min="-1" max="1" /></div>
        <label class="ext-check-row">
          <ToggleSwitch :model-value="b('guid_apg_autooff')" @update:model-value="setB('guid_apg_autooff', $event)" size="sm" />
          <span>APG 켜지면 PAG rescale 자동 끄기</span>
        </label>
      </template>

      <label class="ext-check-row">
        <ToggleSwitch :model-value="b('guid_cwm_enabled')" @update:model-value="setB('guid_cwm_enabled', $event)" size="sm" />
        <span>Enable CWM</span>
      </label>
      <div class="ext-row" v-if="b('guid_cwm_enabled')">
        <div class="ext-field"><label>CWM alpha low (초반 저주파)</label>
          <input type="number" v-model="w._guid_cwm_alpha_low" step="0.01" min="-1" max="1" /></div>
        <div class="ext-field"><label>CWM alpha high (후반 고주파)</label>
          <input type="number" v-model="w._guid_cwm_alpha_high" step="0.01" min="-1" max="1" /></div>
      </div>

      <label class="ext-check-row">
        <ToggleSwitch :model-value="b('guid_smc_master_enabled')" @update:model-value="setB('guid_smc_master_enabled', $event)" size="sm" />
        <span>Enable SMC</span>
      </label>
      <div class="ext-field"><label>SMC preset</label>
        <CustomSelect v-model="w._guid_smc_preset" :options="smcPresets" placeholder="Auto" /></div>
      <div class="ext-row" v-if="w._guid_smc_preset === 'Custom'">
        <div class="ext-field"><label>Custom SMC lambda</label>
          <input type="number" v-model="w._guid_smc_lambda" step="0.1" min="0.5" max="30" /></div>
        <div class="ext-field"><label>Custom SMC k</label>
          <input type="number" v-model="w._guid_smc_k" step="0.01" min="0" max="5" /></div>
      </div>
      <div class="ext-note">Auto는 모델을 감지하며 Anima는 Cosmos / Wan 프리셋을 사용합니다.</div>

      <details class="ag-sub">
        <summary>Legacy CFG base 라디오 (상호배타)</summary>
        <div class="ext-field"><label>CFG base mode</label>
          <CustomSelect v-model="w._guid_cfg_mode" :options="cfgModes" placeholder="Preserve incoming" /></div>
        <label class="ext-check-row">
          <ToggleSwitch :model-value="b('guid_experimental_stack')" @update:model-value="setB('guid_experimental_stack', $event)" size="sm" />
          <span>Experimental stack: SMC → APG → CWM</span>
        </label>
        <label class="ext-check-row">
          <ToggleSwitch :model-value="b('guid_smc_enabled')" @update:model-value="setB('guid_smc_enabled', $event)" size="sm" />
          <span>Enable SMC (legacy)</span>
        </label>
      </details>
    </details>

    <!-- ── Skimmed CFG ──────────────────────────────────────────────── -->
    <details class="ag-group">
      <summary>Skimmed CFG — anti-burn</summary>
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
          <div class="ext-field"><label>Start at (%)</label>
            <input type="number" v-model="w._skim_start_percent" step="0.01" min="0" max="1" /></div>
          <div class="ext-field"><label>End at (%)</label>
            <input type="number" v-model="w._skim_end_percent" step="0.01" min="0" max="1" /></div>
        </div>
        <div class="ext-field"><label>Flip at (%) · 0 = 사용 안 함</label>
          <input type="number" v-model="w._skim_flip_at" step="0.05" min="0" max="1" /></div>
      </template>
    </details>

    <!-- ── DCW / RDC / DAVE / CNS ───────────────────────────────────── -->
    <details class="ag-group">
      <summary>DCW / RDC / DAVE / CNS</summary>

      <label class="ext-check-row">
        <ToggleSwitch :model-value="b('guid_dcw_enabled')" @update:model-value="setB('guid_dcw_enabled', $event)" size="sm" />
        <span>Enable DCW (post-CFG wavelet correction)</span>
      </label>
      <div class="ext-row" v-if="b('guid_dcw_enabled')">
        <div class="ext-field"><label>DCW lambda low</label>
          <input type="number" v-model="w._guid_dcw_lambda_low" step="0.005" min="-0.5" max="0.5" /></div>
        <div class="ext-field"><label>DCW lambda high</label>
          <input type="number" v-model="w._guid_dcw_lambda_high" step="0.005" min="-0.5" max="0.5" /></div>
      </div>

      <label class="ext-check-row">
        <ToggleSwitch :model-value="b('guid_rdc_enabled')" @update:model-value="setB('guid_rdc_enabled', $event)" size="sm" />
        <span>Enable RDC (band-wise reverse drift compensation)</span>
      </label>
      <template v-if="b('guid_rdc_enabled')">
        <div class="ext-field"><label>RDC tau (EMA 기억 구간)</label>
          <input type="number" v-model="w._guid_rdc_tau" step="0.01" min="0" max="0.5" /></div>
        <div class="ext-row">
          <div class="ext-field"><label>RDC alpha LL (구조 drift)</label>
            <input type="number" v-model="w._guid_rdc_alpha_ll" step="0.005" min="0" max="0.3" /></div>
          <div class="ext-field"><label>RDC alpha HH (텍스처 drift)</label>
            <input type="number" v-model="w._guid_rdc_alpha_hh" step="0.001" min="0" max="0.1" /></div>
        </div>
        <div class="ext-note">권장 시작값: tau 0.15 · LL 0.03 · HH 0. 텍스처가 흐려지면 HH를 0으로 유지하세요.</div>
      </template>

      <label class="ext-check-row">
        <ToggleSwitch :model-value="b('guid_dave_enabled')" @update:model-value="setB('guid_dave_enabled', $event)" size="sm" />
        <span>Enable DAVE (block DC attenuation)</span>
      </label>
      <template v-if="b('guid_dave_enabled')">
        <div class="ext-row">
          <div class="ext-field"><label>DAVE strength</label>
            <input type="number" v-model="w._guid_dave_strength" step="0.01" min="0" max="1" /></div>
          <div class="ext-field"><label>DAVE tau (0=전 구간)</label>
            <input type="number" v-model="w._guid_dave_tau" step="0.01" min="0" max="1" /></div>
        </div>
        <div class="ext-field"><label>DAVE block indices</label>
          <input type="text" v-model="w._guid_dave_blocks" placeholder="8-18" /></div>
      </template>

      <label class="ext-check-row">
        <ToggleSwitch :model-value="b('guid_cns_enabled')" @update:model-value="setB('guid_cns_enabled', $event)" size="sm" />
        <span>Enable CNS (wavelet noise 재색칠)</span>
      </label>
      <template v-if="b('guid_cns_enabled')">
        <div class="ext-row">
          <div class="ext-field"><label>CNS strength</label>
            <input type="number" v-model="w._guid_cns_strength" step="0.01" min="0" max="1" /></div>
          <div class="ext-field"><label>CNS gamma power</label>
            <input type="number" v-model="w._guid_cns_gamma_power" step="0.05" min="0.05" max="2" /></div>
        </div>
        <div class="ext-field"><label>CNS gamma scale (Anima 3.0)</label>
          <input type="number" v-model="w._guid_cns_gamma_scale" step="0.25" min="0.25" max="25" /></div>
      </template>
    </details>

    <!-- ── Detail Daemon ────────────────────────────────────────────── -->
    <details class="ag-group">
      <summary>Detail Daemon</summary>
      <label class="ext-check-row">
        <ToggleSwitch :model-value="b('dd_enabled')" @update:model-value="setB('dd_enabled', $event)" size="sm" />
        <span>Enable Detail Daemon</span>
      </label>
      <template v-if="b('dd_enabled')">
        <div class="ext-field"><label>Preset</label>
          <CustomSelect v-model="w._dd_preset" :options="['Custom', 'Subtle', 'Medium', 'Strong']" placeholder="Medium" /></div>
        <div class="ext-field" v-if="w._dd_preset === 'Custom'">
          <label>Detail amount (음수=매끈)</label>
          <input type="number" v-model="w._dd_amount" step="0.01" min="-1" max="1" /></div>
        <details class="ag-sub">
          <summary>세부 스케줄</summary>
          <div class="ext-row">
            <div class="ext-field"><label>Start</label>
              <input type="number" v-model="w._dd_start" step="0.01" min="0" max="1" /></div>
            <div class="ext-field"><label>End</label>
              <input type="number" v-model="w._dd_end" step="0.01" min="0" max="1" /></div>
          </div>
          <div class="ext-row">
            <div class="ext-field"><label>Bias</label>
              <input type="number" v-model="w._dd_bias" step="0.01" min="0" max="1" /></div>
            <div class="ext-field"><label>Exponent</label>
              <input type="number" v-model="w._dd_exponent" step="0.05" min="0" max="10" /></div>
          </div>
          <div class="ext-row">
            <div class="ext-field"><label>Start offset</label>
              <input type="number" v-model="w._dd_start_offset" step="0.01" min="-1" max="1" /></div>
            <div class="ext-field"><label>End offset</label>
              <input type="number" v-model="w._dd_end_offset" step="0.01" min="-1" max="1" /></div>
          </div>
          <div class="ext-row">
            <div class="ext-field"><label>Fade</label>
              <input type="number" v-model="w._dd_fade" step="0.05" min="0" max="1" /></div>
            <div class="ext-field"><label>Multiplier</label>
              <input type="number" v-model="w._dd_multiplier" step="0.05" min="0" max="2" /></div>
          </div>
          <label class="ext-check-row">
            <ToggleSwitch :model-value="b('dd_smooth')" @update:model-value="setB('dd_smooth', $event)" size="sm" />
            <span>Smooth (코사인 스무딩)</span>
          </label>
          <label class="ext-check-row">
            <ToggleSwitch :model-value="b('dd_cfg_couple')" @update:model-value="setB('dd_cfg_couple', $event)" size="sm" />
            <span>Couple to CFG scale</span>
          </label>
        </details>
      </template>
    </details>

    <!-- ── Adaptive Guidance / Modulation ───────────────────────────── -->
    <details class="ag-group">
      <summary>Adaptive Guidance / CLIP Modulation</summary>

      <label class="ext-check-row">
        <ToggleSwitch :model-value="b('guid_adg_enabled')" @update:model-value="setB('guid_adg_enabled', $event)" size="sm" />
        <span>Enable Adaptive Guidance (후반 uncond 생략)</span>
      </label>
      <div class="ext-row" v-if="b('guid_adg_enabled')">
        <div class="ext-field"><label>Skip after</label>
          <input type="number" v-model="w._guid_adg_start" step="0.01" min="0" max="1" /></div>
        <div class="ext-field"><label>Keep every N (0=항상 생략)</label>
          <input type="number" v-model="w._guid_adg_interval" step="1" min="0" max="10" /></div>
      </div>

      <label class="ext-check-row">
        <ToggleSwitch :model-value="b('guid_mod_enabled')" @update:model-value="setB('guid_mod_enabled', $event)" size="sm" />
        <span>Enable Anima Modulation Guidance (보조 CLIP-L)</span>
      </label>
      <template v-if="b('guid_mod_enabled')">
        <div class="ext-field"><label>CLIP-L model (models/text_encoder)</label>
          <input type="text" v-model="w._guid_mod_clip_model" placeholder="clip_l.safetensors" /></div>
        <div class="ext-field"><label>Direction weight w</label>
          <input type="number" v-model="w._guid_mod_weight" step="0.05" min="-20" max="20" /></div>
        <div class="ext-row">
          <div class="ext-field"><label>Start block</label>
            <input type="number" v-model="w._guid_mod_start_layer" step="1" min="0" max="63" /></div>
          <div class="ext-field"><label>End block (-1=마지막)</label>
            <input type="number" v-model="w._guid_mod_end_layer" step="1" min="-1" max="63" /></div>
        </div>
        <div class="ext-field"><label>Base CLIP prompt source</label>
          <CustomSelect v-model="w._guid_mod_base_source" :options="['Main positive', 'Custom']" placeholder="Main positive" /></div>
        <div class="ext-field" v-if="w._guid_mod_base_source === 'Custom'">
          <label>Custom base CLIP prompt</label>
          <input type="text" v-model="w._guid_mod_base_prompt" /></div>
        <div class="ext-field"><label>Positive direction prompt</label>
          <input type="text" v-model="w._guid_mod_positive_prompt" /></div>
        <div class="ext-field"><label>Negative direction source</label>
          <CustomSelect v-model="w._guid_mod_negative_source" :options="['Main negative', 'Custom']" placeholder="Main negative" /></div>
        <div class="ext-field" v-if="w._guid_mod_negative_source === 'Custom'">
          <label>Custom negative direction prompt</label>
          <input type="text" v-model="w._guid_mod_negative_prompt" /></div>
        <div class="ext-field"><label>Adapter source</label>
          <CustomSelect v-model="w._guid_mod_adapter_mode"
            :options="['Auto-download official', 'Local file']" placeholder="Auto-download official" /></div>
        <div class="ext-field" v-if="w._guid_mod_adapter_mode === 'Local file'">
          <label>Local adapter path</label>
          <input type="text" v-model="w._guid_mod_adapter_path" /></div>
      </template>
    </details>

    <div class="ag-actions">
      <button class="ag-import" @click="importFromForge"
        title="현재 Forge API의 /sdapi/v1/script-info가 공개하는 Anima 설정을 가져옵니다. Forge 브라우저에서 아직 적용되지 않은 입력값은 Forge 버전에 따라 포함되지 않을 수 있습니다."><Icon name="rotate-cw" /> Forge에서 가져오기
      </button>
      <button class="ag-reset" @click="resetAll" title="Anima Guidance 전체를 확장 기본값으로">전체 초기화</button>
    </div>
  </details>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import ToggleSwitch from './ToggleSwitch.vue'
import CustomSelect from './CustomSelect.vue'
import { requestAction } from '../stores/widgetStore.js'

/**
 * Anima Guidance Suite 패널.
 *
 * 위젯 id 는 `_` + core/anima_guidance.py 의 스펙 키 (예: `_guid_enabled`).
 * 실제 alwayson_scripts 인자 배열은 백엔드가 그 스펙 순서대로 만든다 —
 * 확장이 args 를 **위치로만** 읽으므로 순서를 프론트에서 다루지 않는 게 핵심이다.
 * 순서 검증은 tests/test_anima_guidance.py 가 담당(설치된 확장과 교차검증).
 */
const props = defineProps<{ widgets: Record<string, any> }>()
const w = computed(() => props.widgets).value

// 스토어는 값을 문자열로 들고 있다 ('true'/'false') — 백엔드 coercion 과 동일 규칙
function b(key: string): boolean {
  return String(w[`_${key}`] ?? '') === 'true'
}
function setB(key: string, val: boolean) {
  w[`_${key}`] = val ? 'true' : 'false'
}

const cfgModes = ['Preserve incoming', 'APG', 'CWM', 'SMC', 'SMC + CWM']
const smcPresets = [
  'Auto', 'SD1.5 / SD2', 'SDXL', 'SD3 / SD3.5',
  'Flux', 'Qwen-Image', 'Cosmos / Wan', 'Custom',
]

// 전체 초기화 — 기본값은 Python 이 core/anima_guidance.py 스펙(default_settings)에서 채운다.
// 예전엔 여기에 82개 기본값 사본(DEFAULTS)을 들고 있어, 스펙이 바뀌면 이 버튼만 옛 값을 쓰거나
// 새 키를 빠뜨릴 수 있었다. 값은 배치 한 번으로 돌아온다(generator_settings._reset_anima_guidance).
function resetAll() {
  requestAction('reset_anima_guidance')
}

function importFromForge() {
  requestAction('import_anima_from_forge')
}

// 켜져 있는 기능 요약 — 아코디언을 접어둬도 뭐가 도는지 보이게
const activeSummary = computed(() => {
  const parts: string[] = []
  if (b('guid_enabled') && w._guid_attn_method !== 'None') parts.push(String(w._guid_attn_method || 'PAG'))
  const beforeSmc: Array<[string, string]> = [
    ['guid_slg_on', 'SLG'], ['guid_apg_enabled', 'APG'], ['guid_adg_enabled', 'ADG'],
  ]
  for (const [key, label] of beforeSmc) if (b(key)) parts.push(label)
  if (b('guid_smc_master_enabled') || b('guid_smc_enabled')) parts.push('SMC')
  const afterSmc: Array<[string, string]> = [
    ['guid_cwm_enabled', 'CWM'], ['guid_dcw_enabled', 'DCW'], ['guid_rdc_enabled', 'RDC'],
    ['guid_dave_enabled', 'DAVE'], ['guid_cns_enabled', 'CNS'],
    ['guid_mod_enabled', 'MOD'], ['skim_enabled', 'Skim'], ['dd_enabled', 'DD'],
  ]
  for (const [key, label] of afterSmc) if (b(key)) parts.push(label)
  return parts.join(' · ')
})
</script>

<style scoped>
.ag-note {
  font-size: var(--fs-label); line-height: 1.5; color: var(--text-muted);
  margin: 6px 0 10px; padding: 6px 8px;
  background: rgba(255, 255, 255, 0.03); border-radius: 6px;
}
.ag-badge {
  margin-left: 8px; font-size: var(--fs-label); font-weight: var(--fw-bold);
  color: var(--accent); opacity: 0.9;
  max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  display: inline-block; vertical-align: bottom;
}
.ag-group { margin: 6px 0; border-left: 2px solid var(--border); padding-left: 8px; }
.ag-group > summary {
  cursor: pointer; font-size: 11px; font-weight: var(--fw-bold);
  color: var(--text-secondary); padding: 4px 0; list-style: none;
}
.ag-group > summary::-webkit-details-marker { display: none; }
.ag-group > summary::before { content: '▸ '; color: var(--text-muted); }
.ag-group[open] > summary::before { content: '▾ '; }
.ag-group > summary:hover { color: var(--text-primary); }
.ag-sub { margin: 6px 0 6px 4px; }
.ag-sub > summary {
  cursor: pointer; font-size: var(--fs-label); font-weight: var(--fw-bold);
  color: var(--text-muted); padding: 3px 0;
}
.ext-note {
  margin: 3px 0 6px; color: var(--text-muted);
  font-size: var(--fs-label); line-height: 1.45;
}
.ag-actions { display: flex; justify-content: space-between; gap: 6px; margin-top: 10px; }
.ag-reset, .ag-import {
  height: 26px; padding: 0 10px; font-size: var(--fs-label); font-weight: var(--fw-bold);
  border-radius: 5px; cursor: pointer;
}
.ag-reset { background: transparent; border: 1px dashed var(--border); color: var(--text-muted); }
.ag-reset:hover { border-color: var(--text-muted); color: var(--text-primary); }
.ag-import {
  background: rgba(96, 165, 250, 0.1); border: 1px solid rgba(96, 165, 250, 0.55);
  color: var(--state-info-fg);
}
.ag-import:hover { background: rgba(96, 165, 250, 0.2); border-color: var(--state-info-fg); color: var(--text-primary); }

/* Anima 패널은 9~11px의 컴팩트 입력 체계다. 공용 CustomSelect의 14px 기본값을
   이 패널 안에서만 맞춰 다른 드롭다운과 입력의 글자 크기가 튀지 않게 한다. */
:deep(.csel-display) {
  min-height: 32px; padding: 6px 8px; font-size: 11px;
}
:deep(.csel-option) { padding: 6px 8px; font-size: 11px; }
:deep(.csel-arrow) { font-size: var(--fs-label); }
</style>
