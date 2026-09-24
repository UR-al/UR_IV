<template>
  <!-- 앱 전체를 덮는 층. 별도 창(QDialog)이 아니라 앱 안의 화면이라, 시작할 때
       사용자가 보는 창은 언제나 하나다. -->
  <div v-if="open" class="gate" :aria-busy="busy || undefined">
    <div class="gate-scroll" @mousedown.self="onBackdrop">
      <div
        ref="rootEl"
        class="gate-col"
        role="dialog"
        aria-modal="true"
        :aria-labelledby="idTitle"
        @keydown="onKeydown"
      >
        <!-- ── 머리말 ── -->
        <header class="gate-head">
          <div class="gate-head-text">
            <h1 :id="idTitle" class="gate-title">AI STUDIO PRO</h1>
            <p class="gate-sub">이미지 생성에 쓸 백엔드를 고르세요. 설정에서 언제든 바꿀 수 있습니다.</p>
          </div>
          <!-- 닫기는 dismissible 일 때만 존재한다. 시작 게이트에는 버튼 자체가 없어야
               '닫을 수 없다'가 화면에서도 사실이 된다(비활성 버튼은 거짓말이다). -->
          <button v-if="dismissible" type="button" class="btn-quiet" @click="emit('dismiss')">
            닫기
          </button>
        </header>

        <div class="rule"></div>

        <!-- 실패는 사라지지 않는다. 부모가 error 를 비울 때까지 남는다. -->
        <div v-if="error" class="gate-error" role="alert">
          <span class="gate-error-tag">연결 실패</span>
          <span class="gate-error-msg selectable">{{ error }}</span>
        </div>

        <!-- ── 두 선택지 ── -->
        <div class="gate-opts">
          <section class="gate-opt">
            <div class="opt-head">
              <h2 class="opt-name">WebUI</h2>
              <span class="opt-status" :class="`is-${webui.tone}`" role="status">
                {{ webui.label }}
              </span>
            </div>
            <p class="opt-note">A1111 · Forge 호환</p>

            <div class="fld">
              <span class="fld-label" :id="idWebuiUrl">주소</span>
              <input
                ref="webuiInputEl"
                v-model="webuiUrlLocal"
                class="fld-input"
                type="text"
                spellcheck="false"
                placeholder="http://127.0.0.1:7860"
                :aria-labelledby="idWebuiUrl"
                @keydown.enter.prevent="choose('webui')"
              />
            </div>

            <p v-if="webui.tone === 'alert'" class="opt-hint">
              응답이 없어도 고를 수 있습니다. 백엔드를 나중에 켜면 그때 이어집니다.
            </p>

            <button
              type="button"
              class="opt-go"
              :class="{ ready: webui.tone === 'ok' }"
              :disabled="busy"
              @click="choose('webui')"
            >
              WebUI 로 시작
            </button>
          </section>

          <section class="gate-opt">
            <div class="opt-head">
              <h2 class="opt-name">ComfyUI</h2>
              <span class="opt-status" :class="`is-${comfy.tone}`" role="status">
                {{ comfy.label }}
              </span>
            </div>
            <p class="opt-note">워크플로 JSON 으로 실행</p>

            <div class="fld">
              <span class="fld-label" :id="idComfyUrl">주소</span>
              <input
                v-model="comfyUrlLocal"
                class="fld-input"
                type="text"
                spellcheck="false"
                placeholder="http://127.0.0.1:8188"
                :aria-labelledby="idComfyUrl"
                @keydown.enter.prevent="choose('comfyui')"
              />
            </div>

            <div class="fld">
              <span class="fld-label" :id="idWorkflow">워크플로</span>
              <input
                v-model="workflowPathLocal"
                class="fld-input"
                type="text"
                spellcheck="false"
                placeholder="JSON 파일 경로 (API Format)"
                :aria-labelledby="idWorkflow"
                @keydown.enter.prevent="choose('comfyui')"
              />
              <button type="button" class="fld-btn" :disabled="busy" @click="emit('pick-workflow')">찾아보기</button>
            </div>

            <!-- 파일을 열기 전에 뭐가 들었는지 한 줄로. 자물쇠 아이콘이 아니라 글자로. -->
            <p v-if="workflowLine" class="opt-wf" :class="`is-${workflowLine.tone}`">
              {{ workflowLine.text }}
            </p>

            <p v-if="comfy.tone === 'alert'" class="opt-hint">
              응답이 없어도 고를 수 있습니다. 백엔드를 나중에 켜면 그때 이어집니다.
            </p>

            <button
              type="button"
              class="opt-go"
              :class="{ ready: comfy.tone === 'ok' }"
              :disabled="busy"
              @click="choose('comfyui')"
            >
              ComfyUI 로 시작
            </button>
          </section>
        </div>

        <div class="rule"></div>

        <details v-if="dismissible" class="gate-workflow-tools">
          <summary>현재 ComfyUI 워크플로 도구</summary>
          <ComfyWorkflowControls />
        </details>

        <!-- ── 꼬리말 ── -->
        <footer class="gate-foot">
          <button type="button" class="btn-quiet" :disabled="busy" @click="probeNow">
            다시 확인
          </button>
          <span v-if="busy" class="gate-progress" role="status">연결하는 중…</span>
        </footer>
        <div v-if="busy" class="gate-bar" aria-hidden="true"><span></span></div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 백엔드 선택 게이트 — 예전 PyQt QDialog 를 대신하는 앱 내부 화면.
 *
 * **왜 창이 아니라 오버레이인가**: 시작할 때 별도 창이 뜨면 작업표시줄에 항목이
 * 둘 생기고 사용자에겐 앱이 두 개로 보인다. 색·글자 스케일도 QSS 와 CSS 두 벌을
 * 맞춰야 했다(테마가 갈라지던 자리).
 *
 * 이 컴포넌트는 **표시와 입력만** 한다 — 감지·연결·파일 다이얼로그는 전부 부모가
 * 이벤트를 받아 처리한다. 그래야 파이썬 쪽 계약이 이 파일 하나에 갇히지 않는다.
 */
import { computed, nextTick, ref, useId, watch } from 'vue'
import ComfyWorkflowControls from './ComfyWorkflowControls.vue'
import { describeWorkflowLine, type GateWorkflowInfo } from '../utils/gateWorkflowInfo'

/** 감지 결과. 'checking' 은 요청이 나갔지만 답이 안 온 상태. */
type ProbeState = 'ok' | 'fail' | 'checking'

/** `backends/comfyui_backend.analyze_workflow()` 결과를 프론트 이름으로 옮긴 것. */
type WorkflowInfo = GateWorkflowInfo

const props = withDefaults(defineProps<{
  /** 게이트를 보일지. */
  open: boolean
  /** 초기 WebUI 주소. */
  webuiUrl: string
  /** 초기 ComfyUI 주소. */
  comfyUrl: string
  /** 초기 ComfyUI 워크플로 경로. */
  workflowPath: string
  /** 자동 감지 결과. 없으면 '확인 중'으로 본다(열자마자 probe 를 보내므로). */
  probe?: { webui?: ProbeState; comfy?: ProbeState }
  /** 워크플로 분석 결과. 경로가 있는데 없으면 아직 분석 전이다. */
  workflowInfo?: WorkflowInfo
  /** 연결 시도 중 — 버튼을 잠그고 진행을 보인다. */
  busy?: boolean
  /** 연결 실패 메시지. 빈 문자열이면 감춘다. */
  error?: string
  /** 시작 때는 false(닫을 수 없음), 설정에서 열면 true. */
  dismissible?: boolean
}>(), {
  busy: false,
  error: '',
  dismissible: false,
})

const emit = defineEmits<{
  probe: [payload: { webuiUrl: string; comfyUrl: string }]
  select: [payload: { type: 'webui' | 'comfyui'; url: string; workflowPath: string }]
  'pick-workflow': []
  dismiss: []
}>()

// ── 입력값 ────────────────────────────────────────────────────────────────
// props 는 '초기값'이라 지역 복사본을 둔다. 부모가 값을 바꿔 보낼 때만(파일
// 선택 후 workflowPath 등) 덮어쓴다 — watch 는 prop 이 실제로 변할 때만 돌므로
// 사용자가 타이핑하는 중에 되돌아가지 않는다.
const webuiUrlLocal = ref(props.webuiUrl)
const comfyUrlLocal = ref(props.comfyUrl)
const workflowPathLocal = ref(props.workflowPath)

watch(() => props.webuiUrl, (v) => { webuiUrlLocal.value = v })
watch(() => props.comfyUrl, (v) => { comfyUrlLocal.value = v })
watch(() => props.workflowPath, (v) => { workflowPathLocal.value = v })

// id 는 인스턴스마다 달라야 한다 — 시작 게이트와 설정에서 연 게이트가 잠깐이라도
// 같이 떠 있으면 중복 id 때문에 라벨이 엉뚱한 입력을 가리킨다.
const uid = useId()
const idTitle = `${uid}-title`
const idWebuiUrl = `${uid}-webui-url`
const idComfyUrl = `${uid}-comfy-url`
const idWorkflow = `${uid}-workflow`

const rootEl = ref<HTMLElement | null>(null)
const webuiInputEl = ref<HTMLInputElement | null>(null)

// ── 상태 문구 ─────────────────────────────────────────────────────────────
// 뜻은 **글자**가 나른다. 색은 거들 뿐이라 색맹·흑백 화면에서도 읽힌다.
const STATUS_LABEL: Record<ProbeState, string> = {
  ok: '연결됨',
  fail: '응답 없음',
  checking: '확인 중',
}
const STATUS_TONE: Record<ProbeState, 'ok' | 'alert' | 'muted'> = {
  ok: 'ok',
  fail: 'alert',
  checking: 'muted',
}

function view(state: ProbeState | undefined) {
  // 미정의를 'checking' 으로 보는 이유: 열리는 순간 probe 를 보내므로 우리 입장에선
  // 이미 확인 중이다. '확인 전' 을 잠깐 보여줬다 바꾸면 글자만 깜빡인다.
  const s: ProbeState = state ?? 'checking'
  return { label: STATUS_LABEL[s], tone: STATUS_TONE[s] }
}

const webui = computed(() => view(props.probe?.webui))
const comfy = computed(() => view(props.probe?.comfy))

// ── 워크플로 한 줄 ────────────────────────────────────────────────────────
// 문구 규칙(분류 이름, 모델 고정, '생성 불가' 경고)은 utils/gateWorkflowInfo.ts.
const workflowLine = computed(() =>
  describeWorkflowLine(workflowPathLocal.value, props.workflowInfo),
)

// ── 동작 ──────────────────────────────────────────────────────────────────
function probeNow() {
  emit('probe', { webuiUrl: webuiUrlLocal.value.trim(), comfyUrl: comfyUrlLocal.value.trim() })
}

function choose(type: 'webui' | 'comfyui') {
  if (props.busy) return
  // 연결이 안 된 백엔드도 고를 수 있다 — 앱을 먼저 켜고 백엔드를 나중에 띄우는
  // 순서가 흔하다. 대신 상태 문구와 안내로 그 사실을 분명히 해 둔다.
  emit('select', {
    type,
    url: (type === 'webui' ? webuiUrlLocal.value : comfyUrlLocal.value).trim(),
    // WebUI 로 시작해도 경로를 같이 보낸다. 설정 화면과 값이 어긋나지 않게 하려면
    // 사용자가 이 화면에서 만진 값이 그대로 저장돼야 한다.
    workflowPath: workflowPathLocal.value.trim(),
  })
}

function onBackdrop() {
  if (props.dismissible) emit('dismiss')
}

/** 게이트 안에서 실제로 포커스를 받을 수 있는 것들. */
function focusables(): HTMLElement[] {
  const root = rootEl.value
  if (!root) return []
  const selector = 'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])'
  return Array.from(root.querySelectorAll<HTMLElement>(selector))
    .filter((el) => el.offsetParent !== null)
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape') {
    // 뒤쪽(App)의 전역 ESC 처리가 이 키를 가로채면 안 된다. 닫을 수 없는
    // 게이트에서 ESC 가 다른 패널을 닫아 버리던 사고를 막는다.
    e.preventDefault()
    e.stopPropagation()
    if (props.dismissible) emit('dismiss')
    return
  }
  if (e.key !== 'Tab') return
  // 포커스 가둠 — 게이트가 떠 있는 동안 탭이 뒤쪽 화면으로 새면 보이지 않는
  // 컨트롤에 입력하게 된다.
  const items = focusables()
  if (items.length === 0) return
  const first = items[0] as HTMLElement
  const last = items[items.length - 1] as HTMLElement
  const active = document.activeElement
  if (e.shiftKey && (active === first || !rootEl.value?.contains(active))) {
    e.preventDefault()
    last.focus()
  } else if (!e.shiftKey && active === last) {
    e.preventDefault()
    first.focus()
  }
}

// 열릴 때 한 번: 감지 요청 + 첫 입력으로 포커스. 마우스 없이도 바로 쓸 수 있어야 한다.
watch(() => props.open, (isOpen) => {
  if (!isOpen) return
  probeNow()
  nextTick(() => { webuiInputEl.value?.focus() })
}, { immediate: true })
</script>

<style scoped>
/* 층 자체 — 스크림이 아니라 불투명한 화면이다. 뒤가 비쳐 보이면 '창 위의 창'
   처럼 읽혀서, 창을 없앤 이유가 사라진다. */
.gate {
  position: fixed;
  inset: 0;
  z-index: 900;
  background: var(--bg-primary);
  overflow-y: auto;
}
/* 창이 짧을 때 위가 잘리지 않게: 가운데 정렬은 안쪽 래퍼가 맡는다. */
.gate-scroll {
  min-height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: var(--sp-6);
}

/* 카드가 아니라 여백이 넓은 단일 열. 상자 대신 헤어라인으로 가른다. */
.gate-col {
  width: 100%;
  max-width: 640px;
  display: flex;
  flex-direction: column;
  gap: var(--sp-6);
}

.rule { height: 1px; background: var(--rule); }

/* ── 머리말 ── */
.gate-head { display: flex; align-items: flex-start; justify-content: space-between; gap: var(--sp-4); }
.gate-head-text { display: flex; flex-direction: column; gap: var(--sp-1); }
.gate-title {
  font-size: var(--fs-title);
  font-weight: var(--fw-bold);
  color: var(--text-primary);
  /* 대문자 영문 제목이라 트래킹을 준다. px 로 박으면 --fs-title 이 바뀔 때 비율이
     깨지므로 em 으로 — tests/test_ui_copy_contract.py 가 px 자간을 막는 이유다.
     (0.125em × 16px = 기존 2px 와 같은 그림) */
  letter-spacing: 0.125em;
}
.gate-sub { font-size: var(--fs-meta); color: var(--text-secondary); }

/* ── 실패 ── */
.gate-error {
  display: flex;
  align-items: baseline;
  gap: var(--sp-2);
  padding-left: var(--sp-3);
  /* 상자로 감싸지 않는다 — 왼쪽 2px 선 하나면 눈이 먼저 여기로 온다. */
  border-left: 2px solid var(--state-alert);
}
.gate-error-tag {
  flex-shrink: 0;
  font-size: var(--fs-label);
  font-weight: var(--fw-bold);
  color: var(--state-alert-fg);
}
.gate-error-msg { font-size: var(--fs-meta); color: var(--text-primary); line-height: 1.5; }

/* ── 두 선택지 ── */
/* 같은 무게다. 한쪽만 색으로 꾸미지 않는다 — 다른 점은 '연결됐는가' 뿐이고
   그건 상태 문구와 버튼 강조로 말한다. */
.gate-opts {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--sp-6);
  align-items: start;
}
.gate-opt + .gate-opt { border-left: 1px solid var(--rule); padding-left: var(--sp-6); }
.gate-opt { display: flex; flex-direction: column; gap: var(--sp-3); }

.opt-head { display: flex; align-items: baseline; justify-content: space-between; gap: var(--sp-2); }
.opt-name { font-size: var(--fs-body); font-weight: var(--fw-bold); color: var(--text-primary); }
.opt-status { font-size: var(--fs-meta); font-weight: var(--fw-medium); white-space: nowrap; }
.opt-status.is-ok { color: var(--state-ok-fg); }
.opt-status.is-alert { color: var(--state-alert-fg); }
.opt-status.is-muted { color: var(--text-muted); }
/* 이름 바로 아래 붙는 부제라 위 간격만 지운다. */
.opt-note { font-size: var(--fs-label); color: var(--text-muted); margin-top: calc(var(--sp-2) * -1); }

.opt-hint { font-size: var(--fs-label); color: var(--text-secondary); line-height: 1.5; }
.opt-wf { font-size: var(--fs-label); line-height: 1.5; word-break: break-all; }
.opt-wf.is-muted { color: var(--text-muted); }
.opt-wf.is-warn { color: var(--state-warn-fg); }
.opt-wf.is-alert { color: var(--state-alert-fg); }

/* ── 입력 ── */
.fld { display: flex; align-items: center; gap: var(--sp-2); }
.fld-label {
  flex: 0 0 44px;
  font-size: var(--fs-label);
  font-weight: var(--fw-medium);
  color: var(--text-muted);
}
/* 전역 input 규칙(14px/10px 14px)은 이 화면의 4단계 스케일 밖이라 덮는다. */
.fld-input {
  flex: 1 1 auto;
  min-width: 0;
  padding: var(--sp-2) var(--sp-3);
  font-size: var(--fs-meta);
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: var(--radius-base);
  color: var(--text-primary);
}
.fld-input:focus { border-color: var(--accent); outline: none; background: var(--bg-input); }
.fld-input::placeholder { color: var(--text-muted); }

.fld-btn {
  flex: 0 0 auto;
  padding: var(--sp-2) var(--sp-3);
  background: var(--bg-button);
  border: 1px solid var(--border);
  border-radius: var(--radius-base);
  color: var(--text-secondary);
  font-size: var(--fs-label);
  font-weight: var(--fw-medium);
  cursor: pointer;
  white-space: nowrap;
  transition: var(--transition);
}
.fld-btn:hover:not(:disabled) { background: var(--bg-button-hover); color: var(--text-primary); border-color: var(--edge); }
.fld-btn:disabled { cursor: default; }

/* ── 시작 버튼 ── */
/* 기본은 중립면. 감지에 성공한 쪽만 강조면으로 올라온다 — 정체성이 아니라
   '지금 켜져 있다'는 상태라, 둘 다 켜져 있으면 둘 다 강조된다. */
.opt-go {
  margin-top: var(--sp-1);
  min-height: 40px;
  padding: 0 var(--sp-4);
  background: var(--bg-button);
  border: 1px solid var(--border);
  border-radius: var(--radius-base);
  color: var(--text-primary);
  font-size: var(--fs-body);
  font-weight: var(--fw-medium);
  cursor: pointer;
  transition: var(--transition);
}
.opt-go:hover:not(:disabled) { background: var(--bg-button-hover); border-color: var(--edge); }
.opt-go.ready {
  background: var(--accent-fill);
  border-color: var(--accent);
  color: var(--on-accent);
  font-weight: var(--fw-bold);
}
.opt-go.ready:hover:not(:disabled) { background: var(--accent-fill-hover); border-color: var(--accent-fill-hover); }
.opt-go:disabled { cursor: default; }

/* ── 꼬리말 ── */
.gate-foot { display: flex; align-items: center; justify-content: space-between; gap: var(--sp-3); }
.gate-progress { font-size: var(--fs-label); color: var(--text-secondary); }

.btn-quiet {
  background: transparent;
  border: none;
  padding: var(--sp-1) 0;
  color: var(--text-muted);
  font-size: var(--fs-meta);
  cursor: pointer;
  transition: var(--transition);
}
.btn-quiet:hover:not(:disabled) { color: var(--text-primary); }
.btn-quiet:disabled { cursor: default; }

/* 진행 표시 — 글자('연결하는 중…')가 본체고 이 선은 보조다. */
.gate-bar { height: 2px; background: var(--rule); overflow: hidden; }
.gate-bar span {
  display: block;
  width: 40%;
  height: 100%;
  background: var(--accent);
  animation: gate-slide 1.1s ease-in-out infinite;
}
@keyframes gate-slide {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(250%); }
}
/* 움직임에 민감한 사용자에게도 '진행 중'은 보여야 하므로, 애니메이션만 끄고
   선은 가득 채워 남긴다. */
@media (prefers-reduced-motion: reduce) {
  .gate-bar span { animation: none; width: 100%; }
}

/* 좁은 창에서는 세로 헤어라인이 가로선이 된다. 두 선택지의 무게는 그대로. */
@media (max-width: 720px) {
  .gate-opts { grid-template-columns: 1fr; }
  .gate-opt + .gate-opt {
    border-left: none;
    padding-left: 0;
    border-top: 1px solid var(--rule);
    padding-top: var(--sp-6);
  }
}
</style>
