<template>
  <transition name="dock-panel">
    <aside v-show="open" class="dock-panel memo-panel" role="dialog" aria-label="메모장" aria-modal="false" @keydown="onPanelKey">
      <header class="dp-head">
        <Icon name="note" />
        <span class="dp-title">메모장</span>
        <span class="mp-sync" :class="syncLabel.tone" :title="syncLabel.detail || syncLabel.text" role="status">{{ syncLabel.text }}</span>
        <span class="dp-spacer"></span>
        <button type="button" class="dp-icon" title="Forge 와 지금 동기화" aria-label="Forge 와 지금 동기화"
          :disabled="!!sync?.syncing" :aria-busy="sync?.syncing ? 'true' : 'false'" @click="requestSync(true)"><Icon name="refresh" /></button>
        <button type="button" class="dp-icon" title="닫기 (Esc)" aria-label="메모장 닫기" @click="emit('close')"><Icon name="close" /></button>
      </header>

      <div class="mp-body">
        <nav class="mp-list-col" aria-label="메모 목록">
          <button type="button" class="mp-new" @click="startNew"><Icon name="plus" /> 새 메모</button>
          <ul class="mp-list">
            <li v-for="m in listed" :key="m.id">
              <button type="button" class="mp-item" :class="{ on: m.id === selectedId, conflict: isConflictCopy(m.title) }"
                :aria-current="m.id === selectedId ? 'true' : undefined" :title="memoDisplayTitle(m)" @click="pick(m.id)">
                <span class="mp-item-title">{{ memoDisplayTitle(m) }}</span>
                <span v-if="isConflictCopy(m.title)" class="mp-badge" title="Forge 와 앱에서 따로 고쳐져 둘 다 남긴 사본">충돌 사본</span>
              </button>
            </li>
          </ul>
          <p v-if="!listed.length" class="mp-list-empty">메모가 없습니다</p>
        </nav>

        <section v-if="draft" class="mp-editor">
          <input ref="titleRef" class="mp-title" :value="draft.title" :maxlength="MEMO_TITLE_MAX" placeholder="제목"
            aria-label="메모 제목" spellcheck="false" @input="edit({ title: inputValue($event) })" />
          <textarea ref="textRef" class="mp-text" :value="draft.text" :maxlength="MEMO_TEXT_MAX"
            placeholder="메모를 적으세요 — 자동으로 저장됩니다" aria-label="메모 내용" spellcheck="false"
            @input="edit({ text: inputValue($event) })"></textarea>
          <footer class="mp-foot">
            <span class="mp-state" role="status">{{ saveState }}</span>
            <span class="dp-spacer"></span>
            <template v-if="confirmingDelete">
              <span class="mp-confirm-text">이 메모를 지울까요?</span>
              <button type="button" class="mp-btn" @click="confirmingDelete = false">취소</button>
              <button type="button" class="mp-btn danger" @click="deleteSelected"><Icon name="trash" /> 삭제 확인</button>
            </template>
            <button v-else type="button" class="mp-btn" title="이 메모 삭제" @click="confirmingDelete = true"><Icon name="trash" /> 삭제</button>
          </footer>
        </section>
        <section v-else class="mp-editor mp-editor-empty">
          <p>왼쪽에서 메모를 고르거나 새 메모를 만드세요.</p>
          <button type="button" class="mp-btn" @click="startNew"><Icon name="plus" /> 새 메모</button>
        </section>
      </div>
    </aside>
  </transition>
</template>

<script setup lang="ts">
/**
 * 도크의 메모장 — 목록 + 새로 만들기/지우기 + 제목 + 큰 입력칸. 입력이 멈추면 ~600ms 뒤 자동 저장.
 *
 * 메모는 앱(user_data/memos.json)에 저장되고, 메모 라우트가 있는 Forge(sam-extra Notebook)에 닿으면
 * 백엔드가 합친다 — Forge 가 없거나 옛 확장이면 '로컬에만 저장됨'. 양쪽에서 따로 고친 메모는 백엔드가
 * 둘 다 남기고 로컬 것을 '(충돌 사본)' 으로 따로 둔다 — 여기선 별도 메모로 보이고, 쓰던 초안은 그 사본으로
 * 옮겨 가 이어서 쓴다(다음 저장이 사본을 고친다 — 사본이 저장마다 늘지 않게).
 * 목록 상태는 composables/useMemos(모듈 싱글턴), 초안 · 자동 저장은 composables/useMemoDraft.
 */
import { computed, nextTick, onUnmounted, ref, watch } from 'vue'
import { useMemos } from '../../composables/useMemos'
import { useMemoDraft } from '../../composables/useMemoDraft'
import { handleDockPanelKeydown } from '../../utils/dockPanelKeys'
import {
  MEMO_TEXT_MAX, MEMO_TITLE_MAX, isConflictCopy, memoDisplayTitle, memoSyncLabel, mergeDraftIntoList,
} from '../../utils/memos'

const props = defineProps<{ open: boolean }>()
const emit = defineEmits<{ close: [] }>()

const store = useMemos()
const { memos, sync, saved, revision } = store
const { draft, parked, selectedId, edit, select, createNew, removeSelected, flushNow } = useMemoDraft({
  memos, revision, saved, save: store.save, remove: store.remove, refresh: store.refresh,
})

// 모달이 아니다 — 패널 안에서 누른 Esc 로 닫고 ↑/↓ 는 패널 것으로 둔다. 바깥 포커스의 앱 단축키
// (히스토리 ↑/↓ · Esc 로 파라미터 열 닫기)는 막지 않는다(utils/dockPanelKeys)
function onPanelKey(e: KeyboardEvent) { handleDockPanelKeydown(e, () => emit('close')) }

// 저장이 끝나지 않은 다른 메모의 편집도 목록에 겹쳐 보인다(확인 뒤에 저장된다)
const listed = computed(() => mergeDraftIntoList(memos.value, [draft.value, ...parked.value]))
const confirmingDelete = ref(false)
const titleRef = ref<HTMLInputElement | null>(null)
const textRef = ref<HTMLTextAreaElement | null>(null)

// '몇 분 전' 이 굳지 않게 열려 있는 동안 30초마다 다시 계산한다
const now = ref(Date.now())
let clock: ReturnType<typeof setInterval> | null = null
const syncLabel = computed(() => memoSyncLabel(sync.value, now.value))

const saveState = computed(() => {
  const d = draft.value
  if (!d) return ''
  if (d.dirty) return '입력 중 — 곧 저장'
  if (d.sent) return d.sent.tries ? '저장 확인이 늦어 다시 보내는 중…' : '저장 중…'
  return d.base ? '저장됨' : '새 메모 — 입력하면 저장됩니다'
})

function inputValue(e: Event) { return (e.target as HTMLInputElement | HTMLTextAreaElement).value }
function pick(id: string) {
  confirmingDelete.value = false
  select(id)
}
function startNew() {
  confirmingDelete.value = false
  createNew()
  nextTick(() => titleRef.value?.focus())
}
function deleteSelected() {
  confirmingDelete.value = false
  removeSelected()
}

// Forge 와 합치기 — 열 때마다 당기되 30초 안에 다시 열면 건너뛴다(여닫기로 Forge 를 두드리지 않게)
let lastSyncAt = 0
function requestSync(force = false) {
  const t = Date.now()
  if (!force && t - lastSyncAt < 30_000) return
  lastSyncAt = t
  store.syncNow()
}

// 고르고 있던 메모가 사라지면(다른 곳에서 삭제 · 여기서 삭제) 맨 위 것을 연다
watch(listed, (list) => {
  if (!props.open || draft.value || !list.length) return
  select(list[0].id)
})

let openedBefore = false
watch(() => props.open, (open) => {
  if (clock) { clearInterval(clock); clock = null }
  if (!open) { confirmingDelete.value = false; flushNow(); return }
  now.value = Date.now()
  clock = setInterval(() => { now.value = Date.now() }, 30_000)
  // 처음 열 때는 useMemos 가 붙자마자 이미 목록을 당겼다
  if (openedBefore) store.refresh()
  openedBefore = true
  requestSync()
  if (!draft.value && memos.value.length) select(memos.value[0].id)
  nextTick(() => (draft.value ? textRef.value : null)?.focus())
}, { immediate: true })

onUnmounted(() => { if (clock) clearInterval(clock) })
</script>

<style scoped src="./dockPanel.css"></style>
<style scoped>
.memo-panel { width: 560px; }
.mp-sync {
  min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  padding: 2px var(--sp-2); border-radius: 10px; font-size: var(--fs-label);
  color: var(--text-muted); background: var(--bg-button);
}
.mp-sync.ok { color: var(--state-ok-fg); }
.mp-sync.busy { color: var(--state-info); }
.mp-sync.warn { color: var(--state-warn-fg); }

.mp-body { flex: 1; min-height: 0; display: flex; }
.mp-list-col {
  width: 176px; flex-shrink: 0; display: flex; flex-direction: column; gap: var(--sp-2);
  padding: var(--sp-2); border-right: 1px solid var(--rule); min-height: 0;
}
.mp-new {
  display: flex; align-items: center; justify-content: center; gap: var(--sp-1);
  height: 32px; border: 1px solid var(--border); border-radius: var(--radius-base);
  background: var(--bg-button); color: var(--text-primary); font-size: var(--fs-meta); font-weight: var(--fw-medium); cursor: pointer;
}
.mp-new:hover { border-color: var(--edge); background: var(--bg-button-hover); }
.mp-list { list-style: none; flex: 1; min-height: 0; overflow-y: auto; display: flex; flex-direction: column; gap: 2px; }
.mp-item {
  width: 100%; min-height: 32px; display: flex; flex-direction: column; align-items: flex-start; gap: 2px;
  padding: var(--sp-1) var(--sp-2); border: 1px solid transparent; border-radius: var(--radius-base);
  background: transparent; color: var(--text-secondary); font-size: var(--fs-meta); text-align: left; cursor: pointer;
}
.mp-item:hover { background: var(--bg-button); color: var(--text-primary); }
.mp-item.on { background: var(--bg-card); border-color: var(--state-info); color: var(--text-primary); }
.mp-item-title { max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.mp-badge {
  font-size: var(--fs-label); padding: 0 var(--sp-1); border-radius: 4px;
  color: var(--state-warn-fg); border: 1px solid var(--state-warn-fg);
}
.mp-list-empty { color: var(--text-muted); font-size: var(--fs-meta); text-align: center; padding: var(--sp-4) 0; }

.mp-editor { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: var(--sp-2); padding: var(--sp-3); }
.mp-title, .mp-text {
  width: 100%; border: 1px solid var(--border); border-radius: var(--radius-base);
  background: var(--bg-input); color: var(--text-primary); font-family: inherit;
}
.mp-title { height: 36px; padding: 0 var(--sp-3); font-size: var(--fs-body); font-weight: var(--fw-bold); }
.mp-text { flex: 1; min-height: 120px; padding: var(--sp-3); font-size: var(--fs-body); line-height: 1.6; resize: none; }
.mp-title:focus, .mp-text:focus { outline: none; border-color: var(--state-info); }
.mp-foot { flex-shrink: 0; display: flex; align-items: center; gap: var(--sp-2); min-height: 28px; }
.mp-state, .mp-confirm-text { font-size: var(--fs-label); color: var(--text-muted); }
.mp-confirm-text { color: var(--text-secondary); }
.mp-btn {
  display: inline-flex; align-items: center; gap: var(--sp-1); height: 28px; padding: 0 var(--sp-3);
  border: 1px solid var(--border); border-radius: var(--radius-base); background: var(--bg-button);
  color: var(--text-secondary); font-size: var(--fs-meta); cursor: pointer;
}
.mp-btn:hover { color: var(--text-primary); border-color: var(--edge); }
.mp-btn.danger { color: var(--state-alert-fg); border-color: var(--state-alert-fg); }
.mp-editor-empty { align-items: center; justify-content: center; color: var(--text-muted); font-size: var(--fs-meta); text-align: center; }

@media (max-width: 640px) {
  .mp-body { flex-direction: column; }
  .mp-list-col { width: auto; max-height: 38%; border-right: 0; border-bottom: 1px solid var(--rule); }
}
</style>
