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
      <PagSection :widgets="widgets" />
    </details>

    <!-- ── CFG base: APG / CWM / SMC ────────────────────────────────── -->
    <details class="ag-group">
      <summary>APG / CWM / SMC — CFG base</summary>
      <ApgSection :widgets="widgets" />
      <CwmSmcSection :widgets="widgets" />
    </details>

    <!-- ── Skimmed CFG ──────────────────────────────────────────────── -->
    <details class="ag-group">
      <summary>Skimmed CFG — anti-burn</summary>
      <SkimSection :widgets="widgets" />
    </details>

    <!-- ── DCW / RDC / DAVE / CNS ───────────────────────────────────── -->
    <details class="ag-group">
      <summary>DCW / RDC / DAVE / CNS</summary>
      <DcwSection :widgets="widgets" />
      <RdcSection :widgets="widgets" />
      <DaveSection :widgets="widgets" />
      <CnsSection :widgets="widgets" />
    </details>

    <!-- ── Detail Daemon ────────────────────────────────────────────── -->
    <details class="ag-group">
      <summary>Detail Daemon</summary>
      <DetailDaemonSection :widgets="widgets" />
    </details>

    <!-- ── Adaptive Guidance / Modulation ───────────────────────────── -->
    <details class="ag-group">
      <summary>Adaptive Guidance / CLIP Modulation</summary>
      <AdgSection :widgets="widgets" />
      <ModulationSection :widgets="widgets" />
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
import { requestAction } from '../stores/widgetStore.js'
import { useGuidanceWidgets } from './guidance/guidanceWidgets'
import PagSection from './guidance/PagSection.vue'
import ApgSection from './guidance/ApgSection.vue'
import CwmSmcSection from './guidance/CwmSmcSection.vue'
import SkimSection from './guidance/SkimSection.vue'
import DcwSection from './guidance/DcwSection.vue'
import RdcSection from './guidance/RdcSection.vue'
import DaveSection from './guidance/DaveSection.vue'
import CnsSection from './guidance/CnsSection.vue'
import DetailDaemonSection from './guidance/DetailDaemonSection.vue'
import AdgSection from './guidance/AdgSection.vue'
import ModulationSection from './guidance/ModulationSection.vue'

/**
 * Anima Guidance Suite 패널 — 카드 · 그룹 아코디언 · 요약 배지 · 가져오기/초기화 버튼만 든다.
 * 기능별 칸은 components/guidance/*Section.vue 에 있다(그룹 <details> 안에 조각으로 놓인다).
 *
 * 위젯 id 는 `_` + core/anima_guidance.py 의 스펙 키 (예: `_guid_enabled`).
 * 실제 alwayson_scripts 인자 배열은 백엔드가 그 스펙 순서대로 만든다 —
 * 확장이 args 를 **위치로만** 읽으므로 순서를 프론트에서 다루지 않는 게 핵심이다.
 * 순서 검증은 tests/test_anima_guidance.py 가 담당(설치된 확장과 교차검증).
 */
const props = defineProps<{ widgets: Record<string, any> }>()
const { w, b } = useGuidanceWidgets(props)

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
  // RDC 는 보내는 값과 같은 원본 켜짐 규칙으로만 표시한다 — 스위치 · DCW · tau > 0 셋 다
  // (core/anima_guidance.py _rdc_on / describe_active). 스위치만 켜고 tau 0(원본 기본)이면 RDC 는 꺼져 있다.
  const rdcOn = b('guid_rdc_enabled') && b('guid_dcw_enabled') && Number(w._guid_rdc_tau) > 0
  const afterSmc: Array<[boolean, string]> = [
    [b('guid_cwm_enabled'), 'CWM'], [b('guid_dcw_enabled'), 'DCW'], [rdcOn, 'RDC'],
    [b('guid_dave_enabled'), 'DAVE'], [b('guid_cns_enabled'), 'CNS'],
    [b('guid_mod_enabled'), 'MOD'], [b('skim_enabled'), 'Skim'], [b('dd_enabled'), 'DD'],
  ]
  for (const [on, label] of afterSmc) if (on) parts.push(label)
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
/* 섹션(guidance/*Section.vue)은 감싸는 요소 없는 조각 컴포넌트라 이 패널의 scoped 속성을 받지 않는다 —
   섹션 안의 .ag-sub · .ext-note 는 :deep 으로 입힌다. 특이도는 예전 `.ag-sub[data-v]` 와 같다(0,2,0). */
:deep(.ag-sub) { margin: 6px 0 6px 4px; }
:deep(.ag-sub > summary) {
  cursor: pointer; font-size: var(--fs-label); font-weight: var(--fw-bold);
  color: var(--text-muted); padding: 3px 0;
}
:deep(.ext-note) {
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
