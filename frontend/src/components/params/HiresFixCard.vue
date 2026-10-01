<template>
  <details class="ext-card">
    <summary class="ext-title">Hires.fix</summary>
    <label class="ext-check-row"><ToggleSwitch v-model="hires_enabled" size="sm" /><span>Hires.fix 활성화</span></label>
    <div class="ext-field">
      <label>Upscaler</label>
      <CustomSelect v-model="storeWidgets.upscaler_combo" :options="upscalerItems" placeholder="Upscaler..." />
      <p v-if="degridWarning" class="hires-warn" role="alert">{{ degridWarning }}</p>
    </div>
    <div class="ext-row">
      <div class="ext-field"><label>스텝</label><input type="number" v-model="storeWidgets.hires_steps_input" /></div>
      <div class="ext-field"><label>Denoise</label><input type="number" v-model="storeWidgets.hires_denoising_input" step="0.05" /></div>
    </div>
    <div class="ext-row">
      <div class="ext-field"><label>Scale</label><input type="number" v-model="storeWidgets.hires_scale_input" step="0.1" min="1" /></div>
      <div class="ext-field"><label>CFG (0=off)</label><input type="number" v-model="storeWidgets.hires_cfg_input" step="0.5" /></div>
    </div>
    <div class="ext-field"><label>Checkpoint</label>
      <CustomSelect v-model="storeWidgets.hires_checkpoint_combo" :options="hiresCheckpointItems" placeholder="Use same checkpoint" /></div>
    <div class="ext-row">
      <div class="ext-field"><label>Sampler</label>
        <CustomSelect v-model="storeWidgets.hires_sampler_combo" :options="hiresSamplerItems" placeholder="Use same sampler" /></div>
      <div class="ext-field"><label>Scheduler</label>
        <CustomSelect v-model="storeWidgets.hires_scheduler_combo" :options="hiresSchedulerItems" placeholder="Use same scheduler" /></div>
    </div>
    <div class="ext-field"><label>Hires Prompt (비우면 메인 사용)</label>
      <input type="text" v-model="storeWidgets.hires_prompt_text" placeholder="비워두면 메인 프롬프트 사용" /></div>
    <div class="ext-field"><label>Hires Negative Prompt</label>
      <input type="text" v-model="storeWidgets.hires_neg_prompt_text" placeholder="비워두면 메인 네거티브 사용" /></div>
  </details>
</template>

<script setup lang="ts">
/** 파라미터 열 — Hires.fix 카드(hires_* 위젯). App.vue 에서 추출(App.vue 분할 ④).
 *
 * Upscaler 에 VAE DeGrid 모델(NAFNet 잔차 모델 — Forge 는 models/ESRGAN 의 것을 업스케일러 목록에도 올린다)을 고르면
 * 경고한다. Forge 스냅샷의 choices.degrid_models 와 ComfyUI 의 `_degrid_model` comfyModels 속성 둘 다와 비교한다
 * (composables/useUpscalerDegridWarning — 배치 업스케일과 같은 규칙). 표시만 — 고른 값은 막지 않는다. */
import CustomSelect from '../CustomSelect.vue'
import ToggleSwitch from '../ToggleSwitch.vue'
import { useWidgetStore } from '../../stores/widgetStore.js'
import { useParamItems } from '../../composables/useParamItems'
import { widgetFlag } from '../../composables/widgetFlag'
import { useUpscalerDegridWarning } from '../../composables/useUpscalerDegridWarning'

const store = useWidgetStore()
const storeWidgets = store.widgets
const { upscalerItems, hiresCheckpointItems, hiresSamplerItems, hiresSchedulerItems } = useParamItems()
const hires_enabled = widgetFlag(storeWidgets, 'hires_options_group')
const degridWarning = useUpscalerDegridWarning(() => storeWidgets.upscaler_combo)
</script>

<style scoped>
.hires-warn { margin: 4px 0 0; color: var(--state-warn-fg); font-size: var(--fs-label); line-height: 1.4; }
</style>
