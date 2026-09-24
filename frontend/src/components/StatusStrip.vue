<template>
  <!-- C1c 계기 스트립 — 화면 맨 아래 한 줄. 왼쪽부터 백엔드 · VRAM · 모델.
       큐/알림은 여기 없다: 사용자가 "큐는 떠다니고 알림은 아이콘 배지가 낫다"고 해서
       우하단 플로팅 핀(QueuePanel)과 우상단 종 배지로 이미 나뉘어 있다. -->
  <footer class="status-strip">
    <!-- 백엔드 -->
    <div class="ss-item" :title="backendTitle">
      <span class="ss-name">{{ backendName }}</span>
      <span class="ss-host">{{ backendHost }}</span>
      <!-- 상태는 글자가 본체다. 점은 거들 뿐 — 색맹/저대비 화면에서 색만 남으면 뜻이 사라진다. -->
      <span class="ss-state" :class="backendLevel"><i class="ss-dot" aria-hidden="true"></i>{{ backendText }}</span>
    </div>

    <span class="ss-sep" aria-hidden="true"></span>

    <!-- VRAM — 클릭하면 백엔드에 모델 unload 요청(부모가 확인 대화상자까지 맡는다) -->
    <div class="ss-item ss-vram" :class="{ live: hasVram }" role="button" :tabindex="hasVram ? 0 : -1"
      :aria-disabled="!hasVram" :title="hasVram ? vramTooltip : 'VRAM 정보 없음 — 백엔드가 값을 보내면 채워진다'"
      @click="clickVram" @keydown.enter.prevent="clickVram" @keydown.space.prevent="clickVram">
      <span class="ss-key">VRAM</span>
      <span class="ss-meter">
        <span class="ss-meter-fill" :class="vramLevel" :style="{ width: meterWidth }"></span>
      </span>
      <span class="ss-val">{{ vramText }}</span>
      <span v-if="vramWarnText" class="ss-state" :class="vramLevel === 'critical' ? 'bad' : 'warn'">{{ vramWarnText }}</span>
    </div>

    <span class="ss-sep" aria-hidden="true"></span>

    <!-- 모델 -->
    <div class="ss-item" :title="modelTitle">
      <span class="ss-key">모델</span>
      <span class="ss-val ss-model">{{ modelText }}</span>
    </div>

    <!-- 상태 한 줄 — 파이썬 show_status(진행·완료·실패). 토스트가 아니라 덮어쓰는 한 줄이라
         스텝마다 오는 진행 문구도 소란스럽지 않다. 남는 폭을 쓰고, 넘치면 말줄임 + title. -->
    <template v-if="statusMessage">
      <span class="ss-sep" aria-hidden="true"></span>
      <div class="ss-item ss-status" :class="statusMessage.level" :title="statusMessage.text"
        role="status" aria-live="polite">
        <span class="ss-status-text">{{ statusMessage.text }}</span>
      </div>
    </template>

    <!-- 오른쪽 — 다음 생성이 어떤 값으로 나갈지. 왼쪽에만 몰아 두면 넓은 화면에서
         바의 95% 가 빈 채로 남아 '넣다 만 줄' 처럼 보인다. 여기 값은 전부
         이미 위젯 스토어에 있는 것이라 새로 물어오지 않는다. -->
    <div class="ss-right">
      <div class="ss-item"><span class="ss-key">출력</span><span class="ss-val">{{ sizeText }}</span></div>
      <span class="ss-sep" aria-hidden="true"></span>
      <div class="ss-item"><span class="ss-key">스텝</span><span class="ss-val">{{ stepsText }}</span></div>
      <span class="ss-sep" aria-hidden="true"></span>
      <div class="ss-item"><span class="ss-key">CFG</span><span class="ss-val">{{ cfgText }}</span></div>
    </div>
  </footer>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useWidgetStore } from '../stores/widgetStore.js'
import { useStatusMessage } from '../composables/useStatusMessage'

// 파이썬 show_status 한 줄(statusMessage 이벤트 + 마운트 때 getStatusMessage 한 번)
const { message: statusMessage } = useStatusMessage()

/**
 * backendStatus 시그널의 페이로드. 필드가 전부 optional 인 이유는 이 값이
 * 파이썬이 보낸 JSON 을 파싱한 결과라서다 — 한 필드가 비어도 줄 전체가
 * 사라지면 안 되므로 자리마다 '—' 로 떨어지게 두고 예외를 만들지 않는다.
 */
interface BackendStatusInfo {
  kind?: string
  label?: string
  url?: string
  connected?: boolean
  error?: string
}

const props = withDefaults(defineProps<{
  /** null = 아직 신호를 못 받음(= '연결 안 됨'). 파이썬이 짧게 몇 번 되풀이해 보낸다. */
  backend?: BackendStatusInfo | null
  vram?: { used: number; total: number; pct: number }
  vramLevel?: 'ok' | 'warn' | 'critical'
  vramTooltip?: string
}>(), {
  backend: null,
  vram: () => ({ used: 0, total: 0, pct: 0 }),
  vramLevel: 'ok',
  vramTooltip: '',
})

const emit = defineEmits<{ 'vram-click': [] }>()

const wStore = useWidgetStore()
const storeWidgets = wStore.widgets

/** 위젯 값 하나를 문자열로. 비어 있으면 '—' — 빈 칸이 보이는 것보다 낫다. */
function widgetText(id: string): string {
  const raw = (storeWidgets as Record<string, unknown>)[id]
  const value = String(raw ?? '').trim()
  return value || '—'
}

// ── 다음 생성 파라미터 (오른쪽) ────────────────────────────────────────────
const sizeText = computed(() => {
  const w = widgetText('width_input')
  const h = widgetText('height_input')
  return w === '—' || h === '—' ? '—' : `${w}×${h}`
})
const stepsText = computed(() => widgetText('steps_input'))
const cfgText = computed(() => widgetText('cfg_input'))

// ── 백엔드 ────────────────────────────────────────────────────────────────
const backendName = computed(() => {
  const b = props.backend
  if (b?.label) return b.label
  // label 이 비어도 kind 는 어느 쪽 백엔드인지 알려 준다. 둘 다 없으면 '—'.
  if (b?.kind === 'comfyui') return 'ComfyUI'
  if (b?.kind === 'webui') return 'WebUI'
  return '—'
})

/** `http://127.0.0.1:7860/` → `127.0.0.1:7860`. 스킴/슬래시는 매 순간 읽을 정보가 아니다. */
const backendHost = computed(() => {
  const url = (props.backend?.url || '').trim()
  if (!url) return '—'
  return url.replace(/^[a-z][a-z0-9+.-]*:\/\//i, '').replace(/\/+$/, '') || '—'
})

/**
 * 세 상태를 나누는 기준:
 *  · 신호 자체가 없음  → '연결 안 됨' (아직 백엔드를 고르지 않았거나 파이썬이 못 보냄)
 *  · connected=false  → '응답 없음'   (주소는 있는데 그쪽이 대답하지 않음)
 *  · connected=true   → '연결됨'
 */
const backendLevel = computed<'ok' | 'bad' | 'idle'>(() => {
  if (!props.backend) return 'idle'
  return props.backend.connected ? 'ok' : 'bad'
})
const backendText = computed(() => (
  backendLevel.value === 'ok' ? '연결됨' : backendLevel.value === 'bad' ? '응답 없음' : '연결 안 됨'
))
const backendTitle = computed(() => {
  const b = props.backend
  if (!b) return '백엔드 연결 안 됨 — 시작 게이트에서 백엔드를 고른다'
  const head = `${backendName.value} · ${backendHost.value} · ${backendText.value}`
  // 실패 사유는 스트립에 못 넣는다(한 줄이라 잘린다) — 툴팁이 그 자리다.
  return b.error ? `${head}\n${b.error}` : head
})

// ── VRAM ──────────────────────────────────────────────────────────────────
const hasVram = computed(() => (props.vram?.total || 0) > 0)
const meterWidth = computed(() => `${Math.max(0, Math.min(100, props.vram?.pct || 0))}%`)
const vramText = computed(() => (
  hasVram.value ? `${props.vram.used} / ${props.vram.total} GB (${props.vram.pct}%)` : '—'
))
// 등급 문구도 색과 함께 글자로 남긴다. ok 일 땐 붙일 말이 없다.
const vramWarnText = computed(() => {
  if (!hasVram.value) return ''
  return props.vramLevel === 'critical' ? '위험' : props.vramLevel === 'warn' ? '주의' : ''
})

function clickVram() {
  if (!hasVram.value) return   // 값도 없는데 unload 를 쏘면 사용자는 무슨 일이 났는지 모른다
  emit('vram-click')
}

// ── 모델 ──────────────────────────────────────────────────────────────────
/** 체크포인트 콤보의 현재 값. App.vue 와 같은 store 접근 방식(widgets 프록시)이다. */
const modelRaw = computed(() => String(storeWidgets.model_combo ?? '').trim())
/** `SDXL\animaPencil_v5.safetensors [abc123]` → `animaPencil_v5`. 전체는 title 에 남는다. */
const modelText = computed(() => {
  const raw = modelRaw.value
  if (!raw) return '—'
  const base = raw.split(/[\\/]/).pop() || raw
  return base.replace(/\s*\[[0-9a-f]{6,}\]\s*$/i, '').replace(/\.(safetensors|ckpt|pt|pth|sft)$/i, '').trim() || raw
})
const modelTitle = computed(() => modelRaw.value || '체크포인트 미선택')
</script>

<style scoped>
/* 높이 28px 은 `.main-workspace { padding-bottom: 28px }` 와 짝이다 — 한쪽만 바꾸면
   워크스페이스 맨 아랫줄이 스트립 뒤로 숨는다. 우하단 큐 핀은 bottom:32px 이라
   이 24px 위로 8px 떠 있다(겹치지 않음). */
.status-strip {
  position: fixed; bottom: 0; left: 0; right: 0; height: 28px; z-index: 500;
  display: flex; align-items: center; gap: var(--sp-5); padding: 0 var(--sp-4);
  background: var(--bg-primary); border-top: 1px solid var(--rule);
  font-size: var(--fs-label); line-height: 1; white-space: nowrap; overflow: hidden;
}
.ss-item { display: flex; align-items: center; gap: 8px; min-width: 0; }
.ss-sep { width: 1px; height: 12px; background: var(--rule); flex-shrink: 0; }
/* 남은 폭을 전부 먹고 오른쪽 끝에 붙는다 — 두 묶음이 바의 양끝을 잡아야
   줄이 '채워진 것' 으로 보인다. 좁아지면 이쪽이 먼저 잘린다(왼쪽이 더 중요). */
.ss-right { margin-left: auto; display: flex; align-items: center; gap: var(--sp-4); min-width: 0; overflow: hidden; }

.ss-name { color: var(--text-primary); font-weight: var(--fw-bold); }
.ss-host { color: var(--text-muted); }
.ss-key { color: var(--text-muted); }
.ss-val { color: var(--text-secondary); font-size: var(--fs-meta); }

/* 상태 문구 — 글자색은 -fg(글자용), 점은 같은 색을 재사용한다. */
.ss-state { display: flex; align-items: center; gap: 4px; font-weight: var(--fw-bold); color: var(--text-muted); }
.ss-state.ok { color: var(--state-ok-fg); }
.ss-state.bad { color: var(--state-alert-fg); }
.ss-state.warn { color: var(--state-warn-fg); }
.ss-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex-shrink: 0; }

/* VRAM 은 값이 있을 때만 누를 수 있다 — 빈 줄을 눌러도 아무 일이 없으면 고장처럼 보인다. */
.ss-vram { cursor: default; }
.ss-vram.live { cursor: pointer; }
.ss-vram.live:hover .ss-meter-fill { filter: brightness(1.2); }
.ss-vram:focus-visible { outline: 1px solid var(--accent); outline-offset: 2px; border-radius: 4px; }
.ss-meter {
  width: 64px; height: 4px; border-radius: 2px; overflow: hidden;
  background: var(--bg-button); flex-shrink: 0;
}
/* 막대는 '면'이라 채움 토큰(--state-*), 옆의 글자는 -fg 토큰을 쓴다. */
.ss-meter-fill { display: block; height: 100%; width: 0; background: var(--state-ok); transition: width 1s ease; }
.ss-meter-fill.warn { background: var(--state-warn); }
.ss-meter-fill.critical { background: var(--state-alert); }

/* 체크포인트 이름은 길다. 줄을 밀어내느니 잘라내고 전체는 title 로 준다. */
.ss-model { max-width: 220px; overflow: hidden; text-overflow: ellipsis; }

/* 상태 한 줄 — 남는 폭을 쓰고 넘치면 말줄임. 색은 수준(level)마다 글자용 -fg 토큰. */
.ss-status { flex: 1 1 auto; min-width: 0; color: var(--text-secondary); font-size: var(--fs-meta); }
.ss-status-text { min-width: 0; overflow: hidden; text-overflow: ellipsis; }
.ss-status.success { color: var(--state-ok-fg); }
.ss-status.warning { color: var(--state-warn-fg); }
.ss-status.error { color: var(--state-alert-fg); font-weight: var(--fw-medium); }
</style>
