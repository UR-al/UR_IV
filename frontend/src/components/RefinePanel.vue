<template>
  <div class="refine-workspace">
    <!-- 좌: 설정 -->
    <aside class="sidebar">
      <div class="sidebar-scroll">
        <div class="glass-card">
          <label>원본 이미지</label>
          <div class="source-thumb" @click="$emit('pick-image')">
            <img v-if="imageSrc" :src="imageSrc" />
            <div v-else class="upload-hint">비어 있음</div>
            <div class="edit-overlay">이미지 변경</div>
          </div>
          <p class="hint mt-8">
            갤러리/히스토리에서 이미지를 끌어오거나 클릭해서 선택하세요.
            결과를 다시 선택하면 <b>체인 refine</b>이 됩니다.
          </p>
        </div>

        <!-- Target / Replacement — Refine의 본체 -->
        <div class="glass-card">
          <label>Target <span class="sub">(마스킹 + 프롬프트에서 제거)</span></label>
          <input v-model="target" type="text" placeholder="shirt, necktie, belt" spellcheck="false" />

          <label class="mt-12">Exclude <span class="sub">(보호할 영역)</span></label>
          <input v-model="exclude" type="text" placeholder="face, eyes, hand" spellcheck="false" />

          <label class="mt-12 accent">Replacement <span class="sub">(대체할 내용)</span></label>
          <input v-model="replacement" type="text" placeholder="nude" spellcheck="false" />

          <label class="mt-12 danger">네거티브</label>
          <textarea v-model="negative" rows="2" placeholder="(선택)"></textarea>

          <label class="ext-check-row mt-12">
            <ToggleSwitch v-model="inheritMain" size="sm" />
            <span>Inherit main t2i prompt</span>
          </label>
          <label class="ext-check-row">
            <ToggleSwitch v-model="inheritNegative" size="sm" />
            <span>Inherit main t2i negative</span>
          </label>
        </div>

        <!-- 마스크 후처리 -->
        <details class="glass-card" open>
          <summary class="card-header">마스크</summary>
          <div class="grid-2 mt-12">
            <div class="input-unit"><label>Threshold</label>
              <input v-model.number="threshold" type="number" step="0.01" min="0" max="1" /></div>
            <div class="input-unit"><label>Dilation (px)</label>
              <input v-model.number="maskDilation" type="number" min="0" /></div>
          </div>
          <div class="grid-2 mt-8">
            <div class="input-unit"><label>Mask Blur</label>
              <input v-model.number="maskBlur" type="number" min="0" /></div>
            <div class="input-unit"><label>Outline (px)</label>
              <input v-model.number="maskOutline" type="number" min="0" /></div>
          </div>
          <label class="ext-check-row mt-8">
            <ToggleSwitch v-model="maskHull" size="sm" />
            <span>Convex Hull (머리카락 가닥 감싸기)</span>
          </label>
          <label class="mt-12">Mask Mode</label>
          <CustomSelect v-model="maskMode" :options="['Individual', 'Combined']" placeholder="Individual" />
        </details>

        <!-- 인페인트 파라미터 -->
        <details class="glass-card">
          <summary class="card-header">인페인트</summary>
          <div class="grid-2 mt-12">
            <div class="input-unit"><label>Denoising</label>
              <input v-model.number="denoise" type="number" step="0.01" min="0" max="1" /></div>
            <div class="input-unit"><label>Padding</label>
              <input v-model.number="padding" type="number" min="0" /></div>
          </div>
          <label class="ext-check-row mt-8">
            <ToggleSwitch v-model="onlyMasked" size="sm" />
            <span>Inpaint only masked</span>
          </label>
          <label class="mt-12">Inpainting fill</label>
          <CustomSelect v-model="fill"
            :options="['fill', 'original', 'latent noise', 'latent nothing']" placeholder="original" />
          <div class="grid-2 mt-12">
            <div class="input-unit"><label>스텝</label>
              <input v-model.number="steps" type="number" min="1" /></div>
            <div class="input-unit"><label>CFG</label>
              <input v-model.number="cfg" type="number" step="0.5" min="0" /></div>
          </div>
          <label class="mt-12">Sampler</label>
          <CustomSelect v-model="sampler" :options="samplerOptions" placeholder="Use same sampler" />
          <label class="mt-12">Scheduler</label>
          <CustomSelect v-model="scheduler" :options="schedulerOptions" placeholder="Use same scheduler" />
          <label class="mt-12">Seed (−1 = 랜덤)</label>
          <div class="seed-row">
            <input v-model="seed" type="text" class="seed-input" placeholder="-1" />
            <button class="seed-btn" @click="seed = '-1'" title="랜덤으로 초기화"><Icon name="dice" /></button>
          </div>
          <label class="mt-12">SAM3 Checkpoint</label>
          <CustomSelect v-model="checkpoint" :options="checkpointOptions" placeholder="sam3.pt" />
          <label class="ext-check-row mt-8" title="인페인트 동안 SAM3(~3.5GB) VRAM 해제 — 16GB 이하 권장">
            <ToggleSwitch v-model="unloadAfter" size="sm" />
            <span>Unload SAM3 from VRAM after detection</span>
          </label>
        </details>

        <!-- ControlNet — T2I·배치 SAM3 와 같은 13필드 패널 (선택지는 Python 한 벌) -->
        <Sam3ControlNetPanel class="glass-card" :widgets="cn" />
      </div>

      <div class="sidebar-footer">
        <button class="btn-generate primary" @click="run" :disabled="!imagePath || busy">
          <Icon v-if="!busy && imagePath" name="play" /> {{ busy ? 'REFINING…' : (!imagePath ? 'SELECT IMAGE FIRST' : 'REFINE') }}
        </button>
      </div>
    </aside>

    <!-- 우: 결과 -->
    <section class="canvas-area">
      <div class="result-grid">
        <div class="result-pane">
          <div class="pane-title">이전</div>
          <img v-if="imageSrc" :src="imageSrc" class="pane-img" />
          <div v-else class="pane-empty">이미지를 선택하세요</div>
        </div>
        <div class="result-pane">
          <div class="pane-title accent">이후</div>
          <img v-if="resultSrc" :src="resultSrc" class="pane-img" />
          <div v-else class="pane-empty">{{ busy ? '처리 중…' : '아직 결과가 없습니다' }}</div>
          <button v-if="resultSrc" class="chain-btn" @click="chain"><Icon name="rotate-cw" /> 이 결과로 이어서 Refine</button>
        </div>
      </div>

      <!-- 프롬프트 수술 결과 — 의도대로 됐는지 눈으로 확인 -->
      <div v-if="lastPrompt" class="prompt-trace">
        <div class="trace-row"><span class="trace-key">프롬프트</span><span class="trace-val">{{ lastPrompt }}</span></div>
        <div v-if="lastNegative" class="trace-row"><span class="trace-key danger">네거티브</span><span class="trace-val">{{ lastNegative }}</span></div>
      </div>
      <div v-if="errorText" class="refine-error">{{ errorText }}</div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, onUnmounted } from 'vue'
import { requestAction } from '../stores/widgetStore.js'
import { onBackendEvent } from '../bridge.js'
import { mediaUrl } from '../utils/media.js'
import { sam3CnDefaults, sam3CnSettings } from '../utils/sam3ControlNet'
import CustomSelect from './CustomSelect.vue'
import ToggleSwitch from './ToggleSwitch.vue'
import Sam3ControlNetPanel from './Sam3ControlNetPanel.vue'

/**
 * SAM3 Refine 패널 (sam-extra 워크플로 2).
 *
 * 확장의 Refine 패널은 Gradio 전용이라 HTTP로 호출할 수 없다. 그래서 같은 결과가
 * 나오도록 앱이 직접 img2img + alwayson_scripts["SAM3 Mask"]를 만든다.
 * Target 프롬프트 수술 로직은 core/refine_prompt.py (확장과 1:1, 테스트로 고정).
 */
const props = withDefaults(defineProps<{
  imagePath?: string
  samplerOptions?: string[]
  schedulerOptions?: string[]
  checkpointOptions?: string[]
}>(), {
  imagePath: '',
  samplerOptions: () => ['Use same sampler'],
  schedulerOptions: () => ['Use same scheduler'],
  checkpointOptions: () => ['sam3.pt'],
})

const emit = defineEmits<{
  'pick-image': []
  'image-changed': [path: string]
}>()

const imagePath = ref(props.imagePath)
const imageSrc = ref(props.imagePath ? mediaUrl(props.imagePath) : '')
const resultSrc = ref('')
const resultPath = ref('')
const busy = ref(false)
const errorText = ref('')
const lastPrompt = ref('')
const lastNegative = ref('')

// Target/Replacement
const target = ref('')
const exclude = ref('')
const replacement = ref('')
const negative = ref('')
const inheritMain = ref(true)
const inheritNegative = ref(true)

// 마스크
const threshold = ref(0.4)
const maskDilation = ref(0)
const maskBlur = ref(4)
const maskOutline = ref(0)
const maskHull = ref(false)
const maskMode = ref('Individual')

// 인페인트
const denoise = ref(0.4)
const padding = ref(32)
const onlyMasked = ref(true)
const fill = ref('original')
const steps = ref(28)
const cfg = ref(7)
const sampler = ref('Use same sampler')
const scheduler = ref('Use same scheduler')
const seed = ref('-1')
const checkpoint = ref('sam3.pt')
const unloadAfter = ref(true)

// ControlNet 13필드 — widget id 키의 로컬 상태 (utils/sam3ControlNet). 예전엔 10필드만
// 있었고(override_external·threshold_a/b 누락) 전처리기 목록을 손으로 복제해 두었다.
const cn = reactive(sam3CnDefaults())

function setImage(path: string) {
  const normalized = (path || '').replace(/\\/g, '/')
  imagePath.value = normalized
  imageSrc.value = normalized ? mediaUrl(normalized) : ''
  resultSrc.value = ''
  resultPath.value = ''
  errorText.value = ''
  emit('image-changed', normalized)
}

function chain() {
  if (!resultPath.value) return
  setImage(resultPath.value)
}

function run() {
  if (!imagePath.value || busy.value) return
  busy.value = true
  errorText.value = ''
  requestAction('run_refine', {
    path: imagePath.value,
    settings: {
      // 프롬프트 수술 (main_prompt/main_negative는 백엔드가 t2i에서 채운다)
      target: target.value,
      replacement: replacement.value,
      negative: negative.value,
      inherit_main: inheritMain.value,
      inherit_negative: inheritNegative.value,
      // SAM3 — 키 이름은 core/sam3_args.SAM3_SPEC 과 동일해야 한다
      sam3_exclude_prompt: exclude.value,
      sam3_threshold: threshold.value,
      sam3_mask_dilation: maskDilation.value,
      sam3_mask_blur: maskBlur.value,
      sam3_mask_outline_px: maskOutline.value,
      sam3_mask_hull: maskHull.value,
      sam3_mask_mode: maskMode.value,
      sam3_denoising_strength: denoise.value,
      sam3_inpaint_only_masked_padding: padding.value,
      sam3_inpaint_only_masked: onlyMasked.value,
      sam3_inpainting_fill: fill.value,
      sam3_checkpoint: checkpoint.value,
      sam3_unload_after: unloadAfter.value,
      sam3_use_steps: true,
      sam3_steps: steps.value,
      sam3_use_cfg_scale: true,
      sam3_cfg_scale: cfg.value,
      sam3_use_sampler: sampler.value !== 'Use same sampler',
      sam3_sampler: sampler.value,
      sam3_use_scheduler: scheduler.value !== 'Use same scheduler',
      sam3_scheduler: scheduler.value,
      sam3_use_seed: seed.value !== '-1',
      sam3_seed: parseInt(seed.value) || -1,
      // ControlNet 13필드 (sam3_cn_*)
      ...sam3CnSettings(cn),
      // 부모 i2i 샘플링 파라미터
      steps: steps.value,
      cfg_scale: cfg.value,
      seed: parseInt(seed.value) || -1,
      sampler: sampler.value,
      scheduler: scheduler.value,
    },
  })
}

// I2IView 가 서브탭을 v-if 로 바꿀 때마다 이 패널이 마운트·언마운트된다 — 구독을 풀지 않으면
// 전환마다 죽은 콜백(과 그 setup 스코프)이 bridge 구독 목록에 쌓인다.
let offRefineResult: (() => void) | null = null
onMounted(() => {
  offRefineResult = onBackendEvent('refineResult', (json: string) => {
    busy.value = false
    try {
      const r = JSON.parse(json)
      if (r.error) { errorText.value = r.error; return }
      if (r.after) {
        resultPath.value = r.after
        resultSrc.value = mediaUrl(r.after, true)
      }
      lastPrompt.value = r.prompt || ''
      lastNegative.value = r.negative_prompt || ''
    } catch {
      errorText.value = 'Refine 결과를 해석하지 못했습니다'
    }
  })
})
onUnmounted(() => {
  offRefineResult?.()
  offRefineResult = null
})

defineExpose({ setImage })
</script>

<style scoped>
.refine-workspace { height: 100%; display: flex; background: var(--bg-primary); }
.sidebar {
  width: 340px; display: flex; flex-direction: column;
  background: var(--bg-secondary); border-right: 1px solid var(--border);
}
.sidebar-scroll { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 16px; }
.sidebar-footer { padding: 16px; background: var(--bg-card); border-top: 1px solid var(--border); }

.glass-card label { display: block; font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-muted); margin-bottom: 4px; }
.glass-card label.accent { color: var(--accent); }
.glass-card label.danger { color: var(--state-alert-fg); }
.glass-card .sub { font-weight: var(--fw-bold); opacity: 0.7; }
.glass-card input[type=text], .glass-card input[type=number], .glass-card textarea {
  width: 100%; box-sizing: border-box; padding: 8px 10px;
  background: var(--bg-input); border: 1px solid var(--border);
  border-radius: 6px; color: var(--text-primary); font-size: 12px;
  outline: none; font-family: inherit;
}
.glass-card input:focus, .glass-card textarea:focus { border-color: var(--accent); }
.ext-check-row { display: flex; align-items: center; gap: 8px; margin: 6px 0; }
.ext-check-row span { font-size: 11px; color: var(--text-secondary); font-weight: var(--fw-bold); }
.hint { font-size: var(--fs-label); line-height: 1.5; color: var(--text-muted); }

.source-thumb {
  width: 100%; aspect-ratio: 16/9; background: var(--bg-input); border-radius: var(--radius-base);
  margin-top: 8px; overflow: hidden; position: relative; cursor: pointer; border: 1px solid var(--border);
}
.source-thumb img { width: 100%; height: 100%; object-fit: cover; }
.upload-hint { height: 100%; display: flex; align-items: center; justify-content: center; color: var(--text-muted); font-size: var(--fs-label); font-weight: var(--fw-bold); letter-spacing: 0; }
.edit-overlay {
  position: absolute; inset: 0; background: rgba(0,0,0,0.6); color: var(--accent);
  display: flex; align-items: center; justify-content: center; font-size: var(--fs-label); font-weight: var(--fw-bold);
  opacity: 0; transition: var(--transition);
}
.source-thumb:hover .edit-overlay { opacity: 1; }

.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.mt-8 { margin-top: 8px; }
.mt-12 { margin-top: 12px; }
.seed-row { display: flex; gap: 6px; align-items: center; }
.seed-input { flex: 1; font-family: 'Consolas', monospace; }
.seed-btn {
  padding: 8px 12px; background: var(--bg-button); border: 1px solid var(--border);
  border-radius: 6px; color: var(--accent); font-size: 14px; cursor: pointer;
}

.btn-generate {
  /* 주 버튼: 글자를 얹는 면이라 --accent 가 아니라 --accent-fill + --on-accent */
  width: 100%; height: 46px; background: var(--accent-fill); border: none;
  border-radius: var(--radius-pill); color: var(--on-accent); font-weight: var(--fw-bold);
  font-size: 12px; letter-spacing: 0; cursor: pointer; transition: var(--transition);
}
.btn-generate:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-generate:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(250, 204, 21, 0.3); }

.canvas-area { flex: 1; padding: 24px; display: flex; flex-direction: column; gap: 12px; min-width: 0; }
.result-grid { flex: 1; display: grid; grid-template-columns: 1fr 1fr; gap: 16px; min-height: 0; }
.result-pane {
  background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  padding: 12px; gap: 8px; min-height: 0; position: relative;
}
.pane-title { font-size: var(--fs-label); font-weight: var(--fw-bold); letter-spacing: 0; color: var(--text-muted); }
.pane-title.accent { color: var(--accent); }
.pane-img { max-width: 100%; max-height: calc(100% - 40px); object-fit: contain; border-radius: 8px; }
.pane-empty { flex: 1; display: flex; align-items: center; justify-content: center; color: var(--text-muted); font-size: 12px; }
.chain-btn {
  padding: 6px 12px; font-size: var(--fs-label); font-weight: var(--fw-bold);
  background: var(--bg-button); border: 1px solid var(--border);
  border-radius: 999px; color: var(--text-primary); cursor: pointer;
}
.chain-btn:hover { border-color: var(--accent); color: var(--accent); }

.prompt-trace {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 10px; padding: 10px 12px; font-size: 11px;
  display: flex; flex-direction: column; gap: 6px; max-height: 120px; overflow-y: auto;
}
.trace-row { display: flex; gap: 8px; }
.trace-key { flex-shrink: 0; font-weight: var(--fw-bold); font-size: var(--fs-label); color: var(--accent); letter-spacing: 0; padding-top: 1px; }
.trace-key.danger { color: var(--state-alert-fg); }
.trace-val { color: var(--text-secondary); word-break: break-word; font-family: 'Consolas', monospace; }
.refine-error {
  background: rgba(248, 113, 113, 0.1); border: 1px solid var(--state-alert-fg);
  border-radius: 8px; padding: 10px 12px; color: var(--state-alert-fg); font-size: 11px;
}
</style>
