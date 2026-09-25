<template>
  <div class="viewer" :class="{ generating }">
    <!-- 이미지 표시 영역 -->
    <div class="image-area">
      <!-- 생성 중 + 중간 그림이 오면 그걸 보여 준다 — 보던 옛 그림이 아니라 지금 만들어지는 것 -->
      <template v-if="generating && previewUrl">
        <img :src="previewUrl" alt="생성 중 미리보기" class="generated-image preview" />
      </template>
      <template v-else-if="imageUrl">
        <img :key="imageSrc" :src="imageSrc" alt="Generated" class="generated-image" :class="{ dimmed: generating }" />
      </template>
      <template v-else>
        <div class="placeholder">
          <div v-if="generating" class="generating-block">
            <div class="spinner" />
            <div class="gen-text">{{ statusLine }}</div>
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: progressPct + '%' }" />
            </div>
          </div>
          <template v-else>
            <div class="placeholder-icon"><Icon name="image" /></div>
            <div class="placeholder-text">{{ status || '이미지를 생성하세요' }}</div>
            <div class="placeholder-sub">좌측에서 프롬프트를 입력하고 생성 버튼을 클릭하세요</div>
          </template>
        </div>
      </template>

      <!-- 진행 카드 — 그림이 떠 있어도 항상 위에. 예전엔 그림이 있으면 창 맨 위 3px 선뿐이라
           생성 중인지 알 수 없었다. -->
      <div v-if="generating && (imageUrl || previewUrl)" class="gen-overlay" role="status" aria-live="polite">
        <div class="spinner small" />
        <div class="gen-overlay-text">
          <span class="gen-overlay-title">{{ previewUrl ? '생성 중 · 미리보기' : '생성 중' }}</span>
          <span class="gen-overlay-sub">{{ statusLine }}</span>
        </div>
        <div class="gen-overlay-bar"><div class="progress-fill" :style="{ width: progressPct + '%' }" /></div>
      </div>
    </div>

    <!-- 하단 정보 바 -->
    <div class="info-bar" v-if="imageUrl">
      <span class="info-item">해상도 {{ displayInfo(resolution) }}</span>
      <span class="info-item">시드 {{ displayInfo(seed) }}</span>
      <button class="explore-btn" @click="exploreSeed" v-if="seed"><Icon name="search" /> 시드 탐색</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { requestAction } from '../stores/widgetStore.js'
import { mediaUrl } from '../utils/media.js'
import { displayInfo } from '../utils/imageInfo'

const props = withDefaults(defineProps<{
  imageUrl?: string
  resolution?: string
  seed?: string
  status?: string
  /** 생성 중인가 — App 의 isGenerating. status 문자열을 파싱하던 예전 방식보다 확실하다. */
  generating?: boolean
  /** 0~100 */
  progress?: number
  /** "ETA 12s" 같은 문자열 (없으면 빈 값) */
  eta?: string
  /** 생성 중 중간 그림 (data URL). 없으면 옛 그림을 흐리게 두고 카드만 띄운다. */
  previewUrl?: string
}>(), {
  imageUrl: '',
  resolution: '',
  seed: '',
  status: '',
  generating: false,
  progress: 0,
  eta: '',
  previewUrl: '',
})

function exploreSeed() {
  requestAction('explore_seed', { seed: props.seed })
}

const imageNonce = ref(Date.now())
watch(() => props.imageUrl, () => {
  imageNonce.value = Date.now()
})

const imageSrc = computed(() => {
  if (!props.imageUrl) return ''
  const base = mediaUrl(props.imageUrl)
  return base + (base.includes('?') ? '&' : '?') + 't=' + imageNonce.value
})

const progressPct = computed(() => {
  if (typeof props.progress === 'number' && props.progress > 0) return Math.max(0, Math.min(100, Math.round(props.progress)))
  const m = props.status?.match(/(\d+)\/(\d+)/)
  if (m) return Math.round(parseInt(m[1]) / parseInt(m[2]) * 100)
  return 0
})

const statusLine = computed(() => {
  const m = props.status?.match(/(\d+)\/(\d+)/)
  const steps = m ? `${m[1]} / ${m[2]} 스텝` : (props.status || '준비 중…')
  const pct = progressPct.value ? ` · ${progressPct.value}%` : ''
  const eta = props.eta ? ` · ${props.eta}` : ''
  return `${steps}${pct}${eta}`
})
</script>

<style scoped>
.viewer {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  background: var(--bg-primary);
}

.image-area {
  position: relative;
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  padding: 16px;
}

.generated-image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  border-radius: 4px;
  transition: opacity .25s ease, filter .25s ease;
}
/* 생성 중에 남아 있는 옛 그림 — '지금 결과' 로 읽히지 않게 한 단계 물린다 */
.generated-image.dimmed { opacity: .45; filter: saturate(.6); }
/* Forge 의 중간 그림은 저해상도다 — 원래 크기로 두면 손톱만 하게 뜬다. 결과가 뜰 자리를 그대로 채운다. */
.generated-image.preview { width: 100%; height: 100%; opacity: .95; }

.gen-overlay {
  position: absolute; top: 18px; left: 50%; transform: translateX(-50%);
  display: grid; grid-template-columns: auto 1fr; column-gap: 12px; row-gap: 8px; align-items: center;
  min-width: 260px; max-width: min(520px, calc(100% - 40px));
  padding: 10px 14px; border-radius: 12px;
  background: color-mix(in srgb, var(--bg-card) 88%, transparent);
  border: 1px solid var(--border-strong);
  box-shadow: 0 10px 30px rgba(0,0,0,.35);
  backdrop-filter: blur(6px);
  pointer-events: none;
}
.gen-overlay-text { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.gen-overlay-title { font-size: var(--fs-body); font-weight: var(--fw-bold); color: var(--accent); }
.gen-overlay-sub { font-size: var(--fs-meta); color: var(--text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.gen-overlay-bar { grid-column: 1 / -1; height: 4px; border-radius: 2px; background: var(--rule); overflow: hidden; }

.placeholder {
  text-align: center;
  user-select: none;
}

.placeholder-icon {
  font-size: 64px;
  opacity: 0.15;
  margin-bottom: 16px;
}

.placeholder-text {
  font-size: 18px;
  color: var(--text-muted);
  font-weight: var(--fw-bold);
  margin-bottom: 8px;
}

.placeholder-sub {
  font-size: 13px;
  color: var(--text-muted);
  margin-bottom: 4px;
}

.info-bar {
  display: flex;
  justify-content: center;
  gap: 24px;
  padding: 8px 16px;
  background: var(--bg-secondary);
  border-top: 1px solid var(--rule);
}

.info-item { font-size: 12px; color: var(--text-muted); }
.explore-btn {
  padding: 3px 12px; background: var(--bg-button); border: 1px solid var(--border);
  border-radius: 4px; color: var(--accent); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer;
}
.explore-btn:hover { background: var(--accent-dim); border-color: var(--accent); }
.generating-block { text-align: center; }
.spinner {
  width: 40px; height: 40px; margin: 0 auto 12px;
  /* 진행 표시의 '빈 부분'은 구분선 역할이라 --rule (라이트에서도 보이는 값) */
  border: 3px solid var(--rule); border-top: 3px solid var(--accent);
  border-radius: 50%; animation: spin 0.8s linear infinite;
}
.spinner.small { width: 22px; height: 22px; margin: 0; border-width: 2.5px; }
@keyframes spin { to { transform: rotate(360deg); } }
.gen-text { color: var(--accent); font-size: 14px; font-weight: var(--fw-bold); }
.progress-bar {
  width: 200px; height: 4px; background: var(--rule); border-radius: 2px;
  margin: 12px auto 0; overflow: hidden;
}
.progress-fill {
  height: 100%; background: var(--accent); border-radius: 2px;
  transition: width 0.3s ease;
}
</style>
