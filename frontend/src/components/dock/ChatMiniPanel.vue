<template>
  <transition name="dock-panel">
    <aside v-show="open" class="dock-panel chat-mini" role="dialog" aria-label="대화" aria-modal="false" @keydown="onPanelKey">
      <header class="dp-head">
        <Icon name="message" />
        <select class="cmini-thread" :value="activeId" aria-label="대화 고르기" :title="active?.title || '새 대화'"
          @change="pickThread">
          <option v-for="t in sortedThreads" :key="t.id" :value="t.id">{{ t.title || '새 대화' }}</option>
        </select>
        <button type="button" class="dp-icon" title="새 대화" aria-label="새 대화" @click="newThread"><Icon name="plus" /></button>
        <button type="button" class="dp-icon" title="닫기 (Esc)" aria-label="대화 패널 닫기" @click="emit('close')"><Icon name="close" /></button>
      </header>
      <div class="cmini-sub" role="status">
        <span class="cmini-dot" :class="{ busy: !!busyId, off: !models.length }"></span>
        <span class="cmini-model" :title="model">{{ model || '모델 없음' }}</span>
        <span>· {{ busyId ? '작업 중' : (models.length ? '준비됨' : '채팅 모델 없음 — 대화 탭 설정에서 연결') }}</span>
      </div>

      <div ref="scrollRef" class="cmini-scroll" @scroll="onScroll" @wheel="onWheel">
        <p v-if="!active || !active.messages.length" class="cmini-empty">
          대화 탭과 같은 대화 목록을 씁니다. 모델 · 지침 · 요청 방식은 대화 탭의 설정을 따릅니다.
        </p>
        <article v-for="m in active?.messages || []" :key="m.id" class="cmini-msg"
          :class="[m.role, { pending: m.pending, error: !!m.error }]">
          <div class="cmini-role">{{ m.role === 'user' ? '나' : (m.model || model || 'AI') }}</div>
          <div v-if="m.images?.length" class="cmini-note">이미지 {{ m.images.length }}장 첨부</div>
          <pre v-if="m.role === 'assistant' && m.structured" class="cmini-content cmini-json">{{ m.content }}</pre>
          <div v-else-if="m.role === 'assistant'" class="cmini-content md" v-html="markdownOf(m)"></div>
          <div v-else class="cmini-content plain">{{ m.content }}</div>
          <div v-if="m.generation && m.pending" class="cmini-note">
            {{ m.generation.kind === 'video' ? '영상 생성' : '이미지 생성' }} · {{ m.generation.message || '생성 중' }}
          </div>
          <div v-if="m.artifacts?.length" class="cmini-artifacts">
            <template v-for="a in m.artifacts" :key="a.path">
              <img v-if="a.kind === 'image' || a.kind === 'animated'" :src="mediaUrl(a.path)" alt="생성 결과" :title="a.path" @load="onMediaLoad" />
              <span v-else class="cmini-note" :title="a.path">{{ a.kind === 'video' ? '영상' : '오디오' }} · {{ a.filename || '생성 결과' }} — 대화 탭에서 재생</span>
            </template>
          </div>
          <div v-if="m.pending && !m.content && !m.generation" class="cmini-note">{{ m.thinking ? '생각하는 중…' : '답을 기다리는 중…' }}</div>
          <div v-if="m.error" class="cmini-error">{{ m.error }}</div>
        </article>
      </div>

      <div class="cmini-composer">
        <textarea ref="composerRef" v-model="draft" class="cmini-input" rows="1" spellcheck="false"
          placeholder="메시지 — Enter 보내기 · Shift+Enter 줄바꿈" aria-label="대화 입력"
          @keydown="onComposerKey" @input="autoGrow"></textarea>
        <button v-if="busyId" type="button" class="cmini-send stop" title="중지 (Esc)" aria-label="중지" @click="stop"><Icon name="stop" /></button>
        <button v-else type="button" class="cmini-send" title="보내기 (Enter)" aria-label="보내기" :disabled="!canSend" @click="send"><Icon name="arrow-up" /></button>
      </div>
    </aside>
  </transition>
</template>

<script setup lang="ts">
/**
 * 도크의 작은 대화 패널 — 대화 탭(views/ChatView.vue)과 **같은 대화 목록**을 쓴다.
 *
 * 대화 목록 · 보내기 · 스트리밍 · 저장은 composables/useChatSession, 모델 · 지침 · 스키마는
 * composables/useChatPreferences — 둘 다 대화 탭과 나눠 쓴다(여기서 한 줄 보내면 대화 탭에도 흐른다).
 * 기능은 일부러 적다: 대화 고르기 · 새 대화 · 메시지 보기 · 입력 · 보내기/중지. 첨부 · 설정 ·
 * 이름 바꾸기 · 내보내기는 대화 탭에서 한다.
 */
import { computed, nextTick, onUnmounted, ref, watch } from 'vue'
import { requestAction } from '../../stores/widgetStore.js'
import { mediaUrl } from '../../utils/media.js'
import { createMarkdownMemo } from '../../utils/chatMarkdown'
import { isImeComposing } from '../../utils/imeComposition'
import type { GenerationRequest } from '../../utils/chatGeneration'
import { useChatPreferences } from '../../composables/useChatPreferences'
import { useChatSession } from '../../composables/useChatSession'
import { handleDockPanelKeydown } from '../../utils/dockPanelKeys'

const props = defineProps<{ open: boolean }>()
const emit = defineEmits<{ close: [] }>()

const { models, model, generationRequest, schemaProblem } = useChatPreferences()
const session = useChatSession()
const { activeId, busyId, active, sortedThreads, stop } = session

// 모달이 아니다 — 패널 안에서 누른 Esc 로 닫고 ↑/↓ 는 패널 것으로 둔다(utils/dockPanelKeys). 답 받는 중의
// 입력칸 Esc 는 먼저 중지에 쓰인다(onComposerKey 가 preventDefault → 닫지 않는다)
function onPanelKey(e: KeyboardEvent) { handleDockPanelKeydown(e, () => emit('close')) }

const draft = ref('')
const followBottom = ref(true)
const composerRef = ref<HTMLTextAreaElement | null>(null)
const scrollRef = ref<HTMLElement | null>(null)
const canSend = computed(() => !busyId.value && draft.value.trim().length > 0)
const markdownOf = createMarkdownMemo()

function pickThread(e: Event) {
  const id = (e.target as HTMLSelectElement).value
  if (id) activeId.value = id
}
function newThread() {
  session.newThread()
  focusComposer()
}
function checkSchema(request: GenerationRequest = generationRequest.value) {
  const problem = schemaProblem(request)
  if (!problem) return true
  requestAction('show_toast', { type: 'error', msg: `${problem} — 대화 탭의 설정에서 JSON 스키마를 고치세요` })
  return false
}
function send() {
  if (!canSend.value) return
  if (!active.value) session.newThread()
  const thread = active.value
  if (!thread || !checkSchema()) return
  session.sendMessage(thread, draft.value.trim(), [], generationRequest.value)
  draft.value = ''
  nextTick(autoGrow)
}
function onComposerKey(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey && !isImeComposing(e)) { e.preventDefault(); send() }
  else if (e.key === 'Escape' && busyId.value) { e.preventDefault(); stop() }
}
function autoGrow() {
  const el = composerRef.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 140) + 'px'
}
function focusComposer() { nextTick(() => composerRef.value?.focus()) }

// ── 스크롤 — 맨 아래를 보고 있으면 흐르는 답을 따라간다 ──
function onScroll() {
  const el = scrollRef.value
  if (!el) return
  followBottom.value = el.scrollHeight - el.scrollTop - el.clientHeight < 60
}
function onWheel(e: WheelEvent) { if (e.deltaY < 0) followBottom.value = false }
function onMediaLoad() { if (followBottom.value) scrollToBottom() }
function scrollToBottom() {
  const el = scrollRef.value
  if (!el) return
  el.scrollTop = el.scrollHeight
  followBottom.value = true
}
const offActivity = session.onChatActivity((kind) => {
  if (kind === 'asked') { followBottom.value = true; nextTick(scrollToBottom) }
  else if (kind === 'stream' && followBottom.value) nextTick(scrollToBottom)
})
watch(activeId, () => { followBottom.value = true; nextTick(scrollToBottom) })
watch(() => props.open, (open) => {
  if (!open) return
  followBottom.value = true
  nextTick(() => { scrollToBottom(); composerRef.value?.focus() })
}, { immediate: true })
onUnmounted(offActivity)
</script>

<style scoped src="./dockPanel.css"></style>
<style scoped>
.chat-mini { width: 420px; }
.cmini-thread {
  flex: 1; min-width: 0; height: 28px; padding: 0 var(--sp-2);
  border: 1px solid var(--border); border-radius: var(--radius-base);
  background: var(--bg-input); color: var(--text-primary); font-size: var(--fs-meta);
}
.cmini-sub {
  flex-shrink: 0; display: flex; align-items: center; gap: var(--sp-1);
  padding: var(--sp-1) var(--sp-3); border-bottom: 1px solid var(--rule);
  color: var(--text-muted); font-size: var(--fs-label); min-width: 0;
}
.cmini-model { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 55%; color: var(--text-secondary); }
.cmini-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--state-ok-fg); flex-shrink: 0; }
.cmini-dot.busy { background: var(--state-warn-fg); }
.cmini-dot.off { background: var(--text-muted); }

.cmini-scroll { flex: 1; min-height: 0; overflow-y: auto; padding: var(--sp-3); display: flex; flex-direction: column; gap: var(--sp-3); }
.cmini-empty { color: var(--text-muted); font-size: var(--fs-meta); line-height: 1.6; text-align: center; padding: var(--sp-6) var(--sp-2); }
.cmini-msg { display: flex; flex-direction: column; gap: var(--sp-1); min-width: 0; }
.cmini-msg.user { align-items: flex-end; }
.cmini-role { font-size: var(--fs-label); color: var(--text-muted); font-weight: var(--fw-medium); }
.cmini-content {
  max-width: 100%; font-size: var(--fs-body); line-height: 1.6; color: var(--text-primary);
  overflow-wrap: anywhere; user-select: text;
}
.cmini-msg.user .cmini-content {
  padding: var(--sp-2) var(--sp-3); border-radius: var(--radius-card);
  background: var(--bg-card); border: 1px solid var(--border); white-space: pre-wrap;
}
.cmini-json {
  margin: 0; padding: var(--sp-2); white-space: pre-wrap; border: 1px solid var(--border);
  border-radius: var(--radius-base); background: var(--bg-input); font-family: ui-monospace, Consolas, monospace; font-size: var(--fs-meta);
}
.cmini-note { font-size: var(--fs-label); color: var(--text-muted); }
.cmini-error { font-size: var(--fs-meta); color: var(--state-alert-fg); overflow-wrap: anywhere; }
.cmini-artifacts { display: flex; flex-wrap: wrap; gap: var(--sp-2); }
.cmini-artifacts img { max-width: 160px; max-height: 160px; border-radius: var(--radius-base); border: 1px solid var(--border); object-fit: contain; background: var(--bg-input); }
.md :deep(p) { margin: 0 0 var(--sp-2); }
.md :deep(p:last-child) { margin-bottom: 0; }
.md :deep(ul), .md :deep(ol) { margin: 0 0 var(--sp-2); padding-left: var(--sp-5); }
.md :deep(code) { padding: 1px var(--sp-1); border-radius: 4px; background: var(--bg-input); font-family: Consolas, 'JetBrains Mono', monospace; font-size: var(--fs-meta); }
.md :deep(pre) { margin: 0 0 var(--sp-2); padding: var(--sp-2); border-radius: var(--radius-base); background: var(--bg-input); border: 1px solid var(--border); overflow-x: auto; }
.md :deep(pre code) { padding: 0; background: transparent; white-space: pre; }
.md :deep(a) { color: var(--accent); text-decoration: underline; }
.md :deep(strong) { font-weight: var(--fw-bold); }

.cmini-composer {
  flex-shrink: 0; display: flex; align-items: flex-end; gap: var(--sp-2);
  padding: var(--sp-2) var(--sp-3) var(--sp-3); border-top: 1px solid var(--rule);
}
.cmini-input {
  flex: 1; min-width: 0; min-height: 36px; max-height: 140px; resize: none;
  padding: var(--sp-2) var(--sp-3); border: 1px solid var(--border); border-radius: var(--radius-base);
  background: var(--bg-input); color: var(--text-primary); font-size: var(--fs-body); line-height: 1.5; font-family: inherit;
}
.cmini-input:focus { outline: none; border-color: var(--state-info); }
.cmini-send {
  flex-shrink: 0; width: 36px; height: 36px; display: inline-flex; align-items: center; justify-content: center;
  border: 0; border-radius: 50%; background: var(--accent-fill); color: var(--on-accent); font-size: var(--fs-title); cursor: pointer;
}
.cmini-send:hover:not(:disabled) { background: var(--accent-fill-hover); }
.cmini-send:disabled { opacity: 0.35; cursor: not-allowed; }
.cmini-send.stop { background: var(--state-alert-fg); color: var(--bg-primary); }
</style>
