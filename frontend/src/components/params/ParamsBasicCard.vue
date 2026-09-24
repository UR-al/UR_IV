<template>
  <div id="sec-params" class="ext-card">
    <!-- Parameters (기본) — `#sec-params` 는 파라미터 열의 첫 카드다(tests/test_nav_rail_contract.py). -->
    <div class="ext-title">파라미터</div>
    <div class="ext-field">
      <label>Resolution</label>
      <div class="ext-res-row">
        <input type="number" v-model="storeWidgets.width_input" />
        <span>×</span>
        <input type="number" v-model="storeWidgets.height_input" />
        <button class="ext-mini-btn" @click="requestAction('swap_resolution')"><Icon name="arrows-horizontal" /></button>
      </div>
      <div class="ext-res-opts">
        <label class="ext-check-sm"><ToggleSwitch v-model="randomResEnabled" size="sm" /><span>랜덤</span></label>
        <label class="ext-check-sm"><ToggleSwitch v-model="autoResEnabled" size="sm" /><span>자동(Parquet)</span></label>
        <label class="ext-check-sm hr-toggle" :class="{ active: highResEnabled }"
          title="입력 해상도 × 배율로 처음부터 더 크게 생성 (hires.fix와 다른 단일 패스)&#10;&#10;⚠ SAM3 자동 검열과 동시 사용 시 VRAM OOM 위험&#10;⚠ 16GB GPU + 1.5× + SAM3 = inpaint 단계에서 메모리 부족&#10;⚠ 모델 학습 해상도(보통 1024±)를 크게 넘으면 이중 캐릭터/왜곡 가능&#10;&#10;권장: 1.5× 단독 사용 또는 SAM3 단독 사용 (둘 중 하나)">
          <ToggleSwitch v-model="highResEnabled" size="sm" />
          <span>고해상도 {{ highResFactor.toFixed(2) }}×<span v-if="highResEnabled" class="hr-warn"><Icon name="alert" /></span></span>
        </label>
      </div>
      <!-- 고해상도 미리보기 + 배율 슬라이더 -->
      <div v-if="highResEnabled" class="hr-preview">
        <div class="hr-row">
          <span class="hr-label">배율</span>
          <input type="range" min="1.1" max="2.5" step="0.05" v-model.number="highResFactor"
            class="hr-slider" />
          <span class="hr-val">{{ highResFactor.toFixed(2) }}×</span>
        </div>
        <div class="hr-result">
          실제 생성: <strong>{{ hrActualW }}×{{ hrActualH }}</strong>
          <span class="hr-note">(8 배수 정렬)</span>
        </div>
        <div class="hr-warn-banner"><Icon name="alert" /> SAM3·ADetailer 등 후처리 동시 사용 시 VRAM OOM 위험.
          16GB GPU + 1.5× = 한계.
        </div>
      </div>
      <!-- 랜덤 해상도 편집기 -->
      <div v-if="randomResEnabled" class="rand-res-editor">
        <div class="rand-res-list">
          <div v-for="(r, i) in randomResList" :key="i" class="rand-res-item">
            <span class="rand-res-val">{{ r[0] }}×{{ r[1] }}</span>
            <span class="rand-res-desc">{{ r[2] }}</span>
            <button class="rand-res-del" @click="removeRandomRes(i)"><Icon name="close" /></button>
          </div>
          <div v-if="!randomResList.length" class="rand-res-empty">해상도를 추가하세요</div>
        </div>
        <div class="rand-res-add">
          <input type="number" v-model.number="newResW" placeholder="W" class="rand-res-input" />
          <span>×</span>
          <input type="number" v-model.number="newResH" placeholder="H" class="rand-res-input" />
          <button class="rand-res-btn" @click="addRandomRes">+</button>
        </div>
      </div>
    </div>
    <div class="ext-row">
      <div class="ext-field"><label>Sampler</label>
        <CustomSelect v-model="storeWidgets.sampler_combo" :options="samplerItems" placeholder="Sampler..." />
      </div>
      <div class="ext-field"><label>Scheduler</label>
        <CustomSelect v-model="storeWidgets.scheduler_combo" :options="schedulerItems" placeholder="Scheduler..." />
      </div>
    </div>
    <div class="ext-row">
      <div class="ext-field"><label>스텝</label><input type="number" v-model="storeWidgets.steps_input" min="1" max="150" /></div>
      <div class="ext-field"><label>CFG</label><input type="number" v-model="storeWidgets.cfg_input" step="0.5" /></div>
      <div class="ext-field"><label>Shift</label><input type="number" v-model="storeWidgets.shift_input" step="0.5" min="0" max="24" title="0이면 미사용 (Distilled CFG Scale)" /></div>
      <div class="ext-field"><label>Seed</label><input type="text" v-model="storeWidgets.seed_input" /></div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 파라미터 열 — 기본 카드(해상도 · 랜덤/자동/고해상도 · 샘플러 · 스텝/CFG/Shift/Seed).
 * App.vue 에서 추출(App.vue 분할 ④). 고해상도·랜덤 해상도 상태의 주인은 App(부팅 복원·로드 시점을
 * App 이 정한다)이고, 여기는 그 객체를 prop 으로 받아 그린다.
 */
import type { PropType } from 'vue'
import CustomSelect from '../CustomSelect.vue'
import ToggleSwitch from '../ToggleSwitch.vue'
import { requestAction, useWidgetStore } from '../../stores/widgetStore.js'
import { useParamItems } from '../../composables/useParamItems'
import { widgetFlag } from '../../composables/widgetFlag'
import type { useHighRes } from '../../composables/useHighRes.js'
import type { RandomResolutions } from '../../composables/useRandomResolutions'

const props = defineProps({
  highRes: { type: Object as PropType<ReturnType<typeof useHighRes>>, required: true },
  randomRes: { type: Object as PropType<RandomResolutions>, required: true },
})

const storeWidgets = useWidgetStore().widgets
const { samplerItems, schedulerItems } = useParamItems()
const { highResEnabled, highResFactor, hrActualW, hrActualH } = props.highRes
const { randomResEnabled, randomResList, newResW, newResH, addRandomRes, removeRandomRes } = props.randomRes
const autoResEnabled = widgetFlag(storeWidgets, 'auto_res_check')
</script>

<style scoped>
.ext-res-row { display: flex; align-items: center; gap: 6px; }
.ext-res-row input { text-align: center; flex: 1; }
.ext-res-row span { color: var(--text-muted); }
.ext-mini-btn { width: 32px; height: 32px; background: var(--bg-button); border: 1px solid var(--border-strong); border-radius: 4px; color: var(--text-primary); cursor: pointer; flex-shrink: 0; }
.ext-res-opts { display: flex; gap: 8px; margin-top: 4px; flex-wrap: wrap; }
.ext-check-sm { display: flex; align-items: center; gap: 3px; font-size: var(--fs-label); color: var(--text-secondary); cursor: pointer; white-space: nowrap; }
.ext-check-sm input { width: 12px; height: 12px; margin: 0; }
/* 고해상도 토글 — 활성 시 골드 강조 */
.hr-toggle.active { color: var(--accent); font-weight: var(--fw-bold); }
.hr-toggle.active span { text-shadow: 0 0 4px rgba(250, 204, 21, 0.3); }

/* 고해상도 미리보기 박스 */
.hr-preview {
  margin-top: 8px; padding: 8px 10px;
  background: var(--accent-dim); border: 1px solid rgba(250, 204, 21, 0.3);
  border-radius: 6px; display: flex; flex-direction: column; gap: 6px;
}
.hr-row { display: flex; align-items: center; gap: 8px; }
.hr-label { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--accent); letter-spacing: 0; min-width: 26px; }
.hr-slider { flex: 1; accent-color: var(--accent); cursor: pointer; height: 4px; }
.hr-val {
  font-family: 'Consolas', monospace; font-size: var(--fs-label); font-weight: var(--fw-bold);
  color: var(--accent); min-width: 36px; text-align: right;
}
.hr-result {
  font-size: var(--fs-label); color: var(--text-secondary);
  font-family: 'Consolas', monospace;
}
.hr-result strong { color: var(--accent); font-weight: var(--fw-bold); font-size: 11px; }
.hr-note { color: var(--text-muted); font-size: var(--fs-label); margin-left: 4px; }
.hr-warn { color: var(--state-warn-fg); margin-left: 4px; font-weight: var(--fw-bold); }
.hr-warn-banner {
  margin-top: 4px; padding: 6px 8px;
  background: rgba(251, 146, 60, 0.08);
  border: 1px solid rgba(251, 146, 60, 0.3);
  border-radius: 4px;
  font-size: 9.5px; line-height: 1.5;
  color: var(--state-warn-fg);
}

/* 랜덤 해상도 편집기 */
.rand-res-editor { margin-top: 6px; border: 1px solid var(--border); border-radius: 6px; padding: 6px; background: rgba(0,0,0,0.15); }
.rand-res-list { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 6px; }
.rand-res-item { display: flex; align-items: center; gap: 4px; padding: 2px 6px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 4px; font-size: var(--fs-label); }
.rand-res-val { color: var(--text-primary); font-weight: var(--fw-bold); font-family: monospace; }
.rand-res-desc { color: var(--text-muted); font-size: var(--fs-label); }
.rand-res-del { background: none; border: none; color: var(--state-alert-fg); cursor: pointer; font-size: var(--fs-label); padding: 0 2px; }
.rand-res-empty { font-size: var(--fs-label); color: var(--text-muted); padding: 4px; }
.rand-res-add { display: flex; align-items: center; gap: 4px; }
.rand-res-add span { color: var(--text-muted); font-size: var(--fs-label); }
.rand-res-input { width: 50px; padding: 3px 4px; font-size: var(--fs-label); text-align: center; }
.rand-res-btn { width: 24px; height: 24px; background: var(--accent-fill); border: none; border-radius: 4px; color: var(--on-accent); font-weight: var(--fw-bold); cursor: pointer; font-size: 14px; }
</style>
