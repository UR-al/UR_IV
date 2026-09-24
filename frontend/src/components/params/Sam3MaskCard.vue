<template>
  <details class="ext-card">
    <summary class="ext-title">SAM3 Mask</summary>
    <label class="ext-check-row"><ToggleSwitch v-model="sam3_enabled" size="sm" /><span>Enable SAM3</span></label>

    <div class="ext-field"><label>SAM3 Detect Prompt</label>
      <input type="text" v-model="storeWidgets._sam3_detect_prompt" placeholder="face" /></div>
    <div class="ext-field"><label>SAM3 Exclude Prompt</label>
      <input type="text" v-model="storeWidgets._sam3_exclude_prompt" placeholder="메인 마스크에서 검출+제외. 예: 'face, eyes' 보호" /></div>
    <div class="ext-field"><label>SAM3 Inpaint Prompt</label>
      <input type="text" v-model="storeWidgets._sam3_inpaint_prompt" placeholder="비워두면 메인 프롬프트 사용" /></div>
    <div class="ext-field"><label>SAM3 Negative Prompt</label>
      <input type="text" v-model="storeWidgets._sam3_neg_prompt" placeholder="비워두면 메인 네거티브 사용" /></div>

    <div class="ext-row">
      <div class="ext-field"><label>SAM3 Mode</label>
        <CustomSelect v-model="storeWidgets._sam3_mode" :options="['Inpaint', 'Mask only']" placeholder="Inpaint" /></div>
      <div class="ext-field"><label>마스크 처리</label>
        <CustomSelect v-model="storeWidgets._sam3_mask_mode" :options="['Individual', 'Combined']" placeholder="Individual" /></div>
    </div>
    <div class="ext-row">
      <div class="ext-field"><label>SAM3 Threshold</label><input type="number" v-model="storeWidgets._sam3_threshold" step="0.01" min="0" max="1" /></div>
      <div class="ext-field"><label>Mask Dilation (px)</label><input type="number" v-model="storeWidgets._sam3_mask_dilation" min="0" /></div>
    </div>
    <div class="ext-row">
      <label class="ext-check-row" style="flex:1" title="머리카락 가닥 등을 감싸 마스크를 채움"><ToggleSwitch :model-value="storeWidgets._sam3_mask_hull === 'true'" @update:model-value="storeWidgets._sam3_mask_hull = $event ? 'true' : 'false'" size="sm" /><span>볼록 껍질 (가닥 감싸기)</span></label>
      <div class="ext-field"><label>Outline expand (edge-aware, px)</label><input type="number" v-model="storeWidgets._sam3_mask_outline_px" min="0" /></div>
    </div>
    <div class="ext-field"><label>SAM3 Checkpoint</label>
      <CustomSelect v-model="storeWidgets._sam3_checkpoint" :options="sam3CheckpointItems" placeholder="sam3.pt" /></div>
    <div class="ext-field">
      <label>SAM3 Device (검출 연산 장치)</label>
      <CustomSelect v-model="storeWidgets._sam3_device" :options="sam3DeviceItems" placeholder="cuda" />
      <div class="ext-note">cuda 권장 — auto는 CPU로 떨어져 검출이 느려질 수 있음</div>
    </div>
    <label class="ext-check-row"><ToggleSwitch :model-value="storeWidgets._sam3_preview_overlay === 'true'" @update:model-value="storeWidgets._sam3_preview_overlay = $event ? 'true' : 'false'" size="sm" /><span>Replace output with overlay preview</span></label>
    <label class="ext-check-row"><ToggleSwitch :model-value="storeWidgets._sam3_save_artifacts === 'true'" @update:model-value="storeWidgets._sam3_save_artifacts = $event ? 'true' : 'false'" size="sm" /><span>Save mask/overlay artifacts</span></label>
    <label class="ext-check-row" title="검출 직후 SAM3(~3.5GB) VRAM 회수 — 16GB GPU 권장"><ToggleSwitch :model-value="storeWidgets._sam3_unload_after === 'true'" @update:model-value="storeWidgets._sam3_unload_after = $event ? 'true' : 'false'" size="sm" /><span>Unload SAM3 from VRAM after detection (~3.5GB)</span></label>

    <!-- 인페인트 하위 섹션 -->
    <details class="ext-card" open style="margin-top:8px">
      <summary class="ext-title">인페인트</summary>
      <div class="ext-row">
        <div class="ext-field"><label>디노이즈 강도</label><input type="number" v-model="storeWidgets._sam3_denoise" step="0.01" min="0" max="1" /></div>
        <div class="ext-field"><label>Mask Blur</label><input type="number" v-model="storeWidgets._sam3_mask_blur" min="0" /></div>
      </div>
      <div class="ext-row">
        <div class="ext-field"><label>Masked content (init for masked area)</label>
          <CustomSelect v-model="storeWidgets._sam3_inpainting_fill" :options="sam3FillItems" placeholder="original" /></div>
        <div class="ext-field"><label>Inpaint padding</label><input type="number" v-model="storeWidgets._sam3_padding" min="0" /></div>
      </div>
      <label class="ext-check-row"><ToggleSwitch :model-value="storeWidgets._sam3_inpaint_only_masked === 'true'" @update:model-value="storeWidgets._sam3_inpaint_only_masked = $event ? 'true' : 'false'" size="sm" /><span>마스크된 영역만</span></label>

      <label class="ext-check-row"><ToggleSwitch :model-value="storeWidgets._sam3_use_inp_size === 'true'" @update:model-value="storeWidgets._sam3_use_inp_size = $event ? 'true' : 'false'" size="sm" /><span>Use separate inpaint width/height</span></label>
      <div class="ext-row" v-if="storeWidgets._sam3_use_inp_size === 'true'">
        <div class="ext-field"><label>Inpaint Width</label><input type="number" v-model="storeWidgets._sam3_inp_w" /></div>
        <div class="ext-field"><label>Inpaint Height</label><input type="number" v-model="storeWidgets._sam3_inp_h" /></div>
      </div>

      <label class="ext-check-row"><ToggleSwitch :model-value="storeWidgets._sam3_use_steps === 'true'" @update:model-value="storeWidgets._sam3_use_steps = $event ? 'true' : 'false'" size="sm" /><span>별도의 단계 사용</span></label>
      <div class="ext-row" v-if="storeWidgets._sam3_use_steps === 'true'">
        <div class="ext-field"><label>단계</label><input type="number" v-model="storeWidgets._sam3_steps" min="1" /></div>
      </div>
      <label class="ext-check-row"><ToggleSwitch :model-value="storeWidgets._sam3_use_cfg === 'true'" @update:model-value="storeWidgets._sam3_use_cfg = $event ? 'true' : 'false'" size="sm" /><span>별도의 CFG 스케일 사용</span></label>
      <div class="ext-row" v-if="storeWidgets._sam3_use_cfg === 'true'">
        <div class="ext-field"><label>CFG 스케일</label><input type="number" v-model="storeWidgets._sam3_cfg" step="0.5" /></div>
      </div>
      <label class="ext-check-row" title="OFF면 base 생성의 sampler 상속&#10;ON이고 'Use same sampler' 이외면 SAM3 단계에서 override"><ToggleSwitch :model-value="storeWidgets._sam3_use_sampler === 'true'" @update:model-value="storeWidgets._sam3_use_sampler = $event ? 'true' : 'false'" size="sm" /><span>별도의 샘플러 사용</span></label>
      <div class="ext-field" v-if="storeWidgets._sam3_use_sampler === 'true'"><label>샘플러</label>
        <CustomSelect v-model="storeWidgets._sam3_sampler" :options="['Use same sampler', ...samplerItems]" placeholder="Use same sampler" /></div>
      <label class="ext-check-row" title="OFF면 base 생성의 scheduler 상속"><ToggleSwitch :model-value="storeWidgets._sam3_use_scheduler === 'true'" @update:model-value="storeWidgets._sam3_use_scheduler = $event ? 'true' : 'false'" size="sm" /><span>Use separate scheduler</span></label>
      <div class="ext-field" v-if="storeWidgets._sam3_use_scheduler === 'true'"><label>Scheduler</label>
        <CustomSelect v-model="storeWidgets._sam3_scheduler" :options="['Use same scheduler', ...schedulerItems]" placeholder="Use same scheduler" /></div>

      <label class="ext-check-row"><ToggleSwitch :model-value="storeWidgets._sam3_use_seed === 'true'" @update:model-value="storeWidgets._sam3_use_seed = $event ? 'true' : 'false'" size="sm" /><span>Use specified seed (instead of parent's)</span></label>
      <div class="ext-field" v-if="storeWidgets._sam3_use_seed === 'true'"><label>Seed (-1 = random)</label>
        <input type="number" v-model="storeWidgets._sam3_seed" /></div>

      <label class="ext-check-row"><ToggleSwitch :model-value="storeWidgets._sam3_use_noise_mul === 'true'" @update:model-value="storeWidgets._sam3_use_noise_mul = $event ? 'true' : 'false'" size="sm" /><span>Use noise multiplier</span></label>
      <div class="ext-row" v-if="storeWidgets._sam3_use_noise_mul === 'true'">
        <div class="ext-field"><label>Noise Multiplier</label><input type="number" v-model="storeWidgets._sam3_noise_mul" step="0.01" min="0" max="2" /></div>
      </div>
      <label class="ext-check-row"><ToggleSwitch :model-value="storeWidgets._sam3_restore_face === 'true'" @update:model-value="storeWidgets._sam3_restore_face = $event ? 'true' : 'false'" size="sm" /><span>Restore face</span></label>
    </details>
    <!-- ControlNet 13필드 (_sam3_cn_*) — 인페인트 패스에 주입, 설정 저장/복원 포함 -->
    <Sam3ControlNetPanel :widgets="storeWidgets" style="margin-top:8px" />
  </details>
</template>

<script setup lang="ts">
/**
 * 파라미터 열 — SAM3 Mask 카드(Forge Neo SAM3 확장과 1:1, _sam3_* 위젯). App.vue 에서 추출(App.vue 분할 ④).
 * ControlNet 13필드는 Sam3ControlNetPanel(RefinePanel · BatchView 와 같은 컴포넌트)이 그린다.
 */
import CustomSelect from '../CustomSelect.vue'
import ToggleSwitch from '../ToggleSwitch.vue'
import Sam3ControlNetPanel from '../Sam3ControlNetPanel.vue'
import { useWidgetStore } from '../../stores/widgetStore.js'
import { SAM3_DEVICE_ITEMS, SAM3_FILL_ITEMS, useParamItems } from '../../composables/useParamItems'
import { widgetFlag } from '../../composables/widgetFlag'

const storeWidgets = useWidgetStore().widgets
const { samplerItems, schedulerItems, sam3CheckpointItems } = useParamItems()
const sam3DeviceItems = SAM3_DEVICE_ITEMS
const sam3FillItems = SAM3_FILL_ITEMS
const sam3_enabled = widgetFlag(storeWidgets, 'sam3_group')
</script>
