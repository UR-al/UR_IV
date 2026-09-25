<template>
  <div class="tbf">
    <!-- 모두 비우기 버튼: 평소 왼쪽으로 반달처럼 살짝 튀어나옴, hover 시 원형+X -->
    <button v-if="blocks.length > 0"
      class="tbf-clear-all"
      @click.stop="clearAllBlocks"
      :title="`블록 ${blocks.length}개 모두 비우기 (Ctrl+Z로 복구 가능)`">
      <span class="tbf-clear-x"><Icon name="close" /></span>
    </button>
    <div class="tbf-blocks" @dragover.prevent="onDragOver" @drop="onDrop" @dragleave="dropIdx = -1">
      <template v-for="(tb, ti) in blocks" :key="ti">
        <!-- 드롭 위치 미리보기 -->
        <div class="tbf-drop-marker" v-if="dropIdx === ti && draggingFrom !== ti"></div>
        <!-- 편집 모드 -->
        <input v-if="editIdx === ti" class="tbf-edit" v-model="editText"
          @blur="finishEdit(ti)" @keydown.enter="onEditEnter($event, ti)" @keydown.escape="editIdx = -1"
          ref="editInputRef" />
        <!-- 블록 — 클릭: 편집(와일드카드는 관리자 열기) · 우클릭: 삭제 · 끌기: 순서 변경.
             (옛 '더블클릭 비활성화'는 일반 더블클릭으로 닿지 않고 수식키+더블클릭이면 태그가
              지워지는 상태라 제거했다.) -->
        <button v-else class="tbf-block" draggable="true"
          :class="[colorClass(tb.text), { wildcard: isWc(tb.text) }]"
          @click.exact="startEdit(ti)"
          @contextmenu.prevent="removeBlock(ti)"
          @dragstart="onDragStart(ti)" @dragend="draggingFrom = -1">
          <span class="wc-ico" v-if="isWc(tb.text)"><Icon name="dice" /></span>
          <span class="tbf-text">{{ tb.text }}</span>
        </button>
      </template>
      <!-- 끝에 드롭 -->
      <div class="tbf-drop-marker" v-if="dropIdx === blocks.length"></div>
      <!-- 추가 입력 -->
      <div class="tbf-add-wrap">
        <input ref="addInputRef" class="tbf-add" v-model="newTag" :placeholder="placeholder"
          @input="onAddInput"
          @keydown="onAddKey"
          @blur="ac.close()" />
        <div class="ac-popup-block" v-if="acItems.length > 0">
          <div v-for="(item, i) in acItems" :key="item.tag" class="ac-item"
            :class="{ selected: acIdx === i }"
            @mousedown.prevent="acceptSuggestion(item.tag)">{{ item.tag.replace(/_/g, ' ') }}<span v-if="item.ko" class="ac-ko">{{ item.ko }}</span></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, watch, onUnmounted } from 'vue'
import { useTagAutocomplete } from '../composables/useTagAutocomplete'
import { isImeComposing } from '../utils/imeComposition'

interface TagBlock {
  text: string
}

const props = withDefaults(defineProps<{
  modelValue?: string
  colorFn?: (text: string) => string
  placeholder?: string
  /** 텍스트 → 블록 경계. 없으면 프롬프트 태그 규칙(괄호·와일드카드 안 쉼표 보호).
   *  제외 규칙 칸은 적용 쪽과 같은 나누기(utils/excludeRules.splitExcludeRules)를 넘긴다. */
  split?: (text: string) => string[]
  /** 블록 목록 → 텍스트. 없으면 ', ' 로 잇는다. 이전 텍스트(지금 modelValue)를 받으므로 바뀌지 않은
   *  블록 사이의 구분자·줄바꿈을 살릴 수 있다 — 제외 규칙 칸은 utils/excludeRules.rewriteExcludeRules. */
  join?: (previous: string, parts: string[]) => string
}>(), {
  modelValue: '',
  colorFn: () => '',
  placeholder: '추가...',
  split: undefined,
  join: undefined,
})
const emit = defineEmits<{
  'update:modelValue': [value: string]
  'open-wildcard': [name: string | undefined]
}>()

// 블록 데이터
const blocks = ref<TagBlock[]>([])
const newTag = ref('')
const editIdx = ref(-1)
const editText = ref('')
const editInputRef = ref<HTMLInputElement[] | null>(null)
const draggingFrom = ref(-1)
const dropIdx = ref(-1)

// 초기화 + 동기화
function parseText(text: string): TagBlock[] {
  if (!text) return []
  if (props.split) return props.split(text).map(t => ({ text: t }))
  let depth = 0; let p = ''
  for (const ch of text) {
    if ('([{'.includes(ch)) depth++
    else if (')]}'.includes(ch)) depth = Math.max(0, depth - 1)
    p += (ch === ',' && depth > 0) ? '\x01' : ch
  }
  p = p.replace(/__([^_]+)__/g, m => m.replace(/,/g, '\x01'))
  return p.split(',').map(t => t.trim().replace(/\x01/g, ',')).filter(Boolean).map(t => ({ text: t }))
}

function syncToModel() {
  const parts = blocks.value.map(b => b.text)
  // 블록은 modelValue 에서 나눠 만들었고 부모가 아직 새 값을 내려주기 전이다 — modelValue 가 이전 텍스트
  emit('update:modelValue', props.join ? props.join(props.modelValue || '', parts) : parts.join(', '))
}

watch(() => props.modelValue, (v: string) => {
  blocks.value = parseText(v)
}, { immediate: true })

// 블록 조작
function clearAllBlocks() {
  // 전체 블록 삭제 — Ctrl+Z로 복구 가능 (PromptPanel의 undoStack이 widget 변경 추적).
  // 이 버튼은 v-if 로 사라져 포커스가 body 로 떨어지는데, 그것도 패널 Undo 범위다
  // (utils/promptUndoKeys 'unfocused'). 500ms debounce 전에 눌러도 Undo 가 먼저 확정한다(usePromptUndo).
  blocks.value = []
  editIdx.value = -1
  newTag.value = ''
  ac.close()
  syncToModel()
}

function removeBlock(idx: number) {
  blocks.value.splice(idx, 1)
  syncToModel()
}

function addBlock() {
  // 추가 뒤 250ms 에 이전 입력의 후보가 빈 칸 아래 되살아나지 않게 — 대기 중 요청까지 무효화
  ac.close()
  const tag = newTag.value.trim()
  if (tag) { blocks.value.push({ text: tag }); newTag.value = ''; syncToModel() }
}

function startEdit(idx: number) {
  if (isWc(blocks.value[idx].text)) { emit('open-wildcard', blocks.value[idx].text.match(/__(.+?)__/)?.[1]); return }
  editIdx.value = idx
  editText.value = blocks.value[idx].text
  nextTick(() => { if (editInputRef.value?.[0]) editInputRef.value[0].focus() })
}

/** 편집 칸 Enter — IME 조합 확정 Enter 로 편집을 끝내면 마지막 음절이 잘린다 */
function onEditEnter(e: KeyboardEvent, idx: number) {
  if (isImeComposing(e)) return
  finishEdit(idx)
}

function finishEdit(idx: number) {
  if (editIdx.value !== idx) return
  const t = editText.value.trim()
  if (t) blocks.value[idx].text = t
  else blocks.value.splice(idx, 1)
  editIdx.value = -1
  syncToModel()
}

// 자동완성 (블록 모드) — 항목은 {tag, ko}; 한글을 치면 한국어 키워드로 검색된다("장발" → long hair).
// 요청 수명주기(디바운스 취소·늦은 응답 버리기)는 useTagAutocomplete — blur·추가·Escape 뒤에
// 예전처럼 250ms 늦게 팝업이 되살아나 Enter/Tab 이 원치 않는 블록을 넣지 않는다.
const ac = useTagAutocomplete({ delay: 250 })
const acItems = ac.items
const acIdx = ac.index
const addInputRef = ref<HTMLInputElement | null>(null)
onUnmounted(() => ac.close())

function onAddInput(e: Event) {
  // v-model(newTag)은 IME 조합 중 갱신되지 않는다 — '장' 한 음절만 쳐도 검색되도록 요소 값을 읽는다
  const el = e.target as HTMLInputElement | null
  ac.request(el ? el.value : newTag.value)
}

function onAddKey(e: KeyboardEvent) {
  // IME 조합 중 키(확정 Enter 등)는 추가·선택이 아니다 — 조합이 끝난 뒤의 Enter 가 처리한다
  if (isImeComposing(e)) return
  // 자동완성 활성일 때 키 처리
  if (acItems.value.length > 0) {
    if (e.key === 'ArrowDown') { e.preventDefault(); ac.move(1); return }
    if (e.key === 'ArrowUp')   { e.preventDefault(); ac.move(-1); return }
    if (e.key === 'Tab')       { e.preventDefault(); acceptSuggestion(ac.selected()?.tag); return }
    if (e.key === 'Escape')    { ac.close(); return }
    // Enter는 자동완성 항목 선택 (위쪽으로 이동 안 했어도 첫 번째 선택).
    // 한글 검색이면 선택하지 않고 팝업만 닫은 뒤 평소처럼 입력한 글자를 블록으로 추가 — Tab/클릭으로 선택.
    if (e.key === 'Enter') {
      if (ac.queryHangul.value) { addBlock(); return }
      e.preventDefault(); acceptSuggestion(ac.selected()?.tag); return
    }
  }
  // 자동완성 없으면 기존 addBlock 동작
  if (e.key === 'Enter') addBlock()
}

function acceptSuggestion(tag: string | undefined) {
  ac.close()
  if (!tag) return
  newTag.value = tag.replace(/_/g, ' ')  // 표시는 공백으로
  addBlock()
  // IME 조합 중(클릭 수락)이면 v-model 이 DOM 을 비우지 않는다 — 조합 확정 때 옛 글자가 되살아나지 않게 직접 비운다
  const el = addInputRef.value as (HTMLInputElement & { composing?: boolean }) | null
  if (el && el.composing) el.value = ''
}

// 드래그 — 다중 행(wrap) 인식 + 행 내 X 기준 위치 판정
function onDragStart(idx: number) { draggingFrom.value = idx }
function onDragOver(e: DragEvent) {
  const container = e.currentTarget as HTMLElement
  // 실제 블록만 추출 (드롭 마커/입력 칸 제외)
  const els = Array.from(container.querySelectorAll('.tbf-block, .tbf-edit'))
  if (els.length === 0) { dropIdx.value = 0; return }

  const cx = e.clientX
  const cy = e.clientY

  // 각 블록의 위치/크기 수집
  const items = els.map((el, i) => {
    const box = el.getBoundingClientRect()
    return {
      idx: i,
      top: box.top,
      bottom: box.bottom,
      left: box.left,
      mid: box.left + box.width / 2,
      yCenter: (box.top + box.bottom) / 2,
    }
  })

  // 1) 커서 Y가 어느 행에 있는지 판정 — 행 Y 범위에 직접 들어있으면 그 행
  let inRow = items.filter(c => cy >= c.top && cy <= c.bottom)

  if (inRow.length === 0) {
    // 행 사이/위/아래에 있으면 가장 가까운 행 사용
    let closestY = items[0].yCenter
    let minDist = Math.abs(cy - closestY)
    for (const c of items) {
      const d = Math.abs(cy - c.yCenter)
      if (d < minDist) { minDist = d; closestY = c.yCenter }
    }
    // 그 yCenter와 비슷한 모든 항목 = 같은 행 (행 높이는 보통 ~30px이므로 8px 허용)
    inRow = items.filter(c => Math.abs(c.yCenter - closestY) < 8)
  }

  if (inRow.length === 0) {
    dropIdx.value = items.length
    return
  }

  // 2) 행 내에서 X 기준으로 삽입 위치 결정
  //    커서가 어떤 블록의 좌측 절반에 있으면 그 블록 앞,
  //    모두 통과(우측)하면 행의 마지막 블록 뒤
  let dropAt = inRow[inRow.length - 1].idx + 1
  for (const c of inRow) {
    if (cx < c.mid) {
      dropAt = c.idx
      break
    }
  }

  // 자기 자신 위에서는 마커 안 보이게 (시각적 안정)
  if (dropAt === draggingFrom.value || dropAt === draggingFrom.value + 1) {
    dropIdx.value = -1
  } else {
    dropIdx.value = dropAt
  }
}
function onDrop() {
  if (draggingFrom.value >= 0 && dropIdx.value >= 0 && draggingFrom.value !== dropIdx.value) {
    const [item] = blocks.value.splice(draggingFrom.value, 1)
    const target = dropIdx.value > draggingFrom.value ? dropIdx.value - 1 : dropIdx.value
    blocks.value.splice(target, 0, item)
    syncToModel()
  }
  draggingFrom.value = -1; dropIdx.value = -1
}

function colorClass(text: string) { return props.colorFn(text) }
function isWc(text: string) { return /__.+__/.test(text) }
</script>

<style scoped>
.tbf { position: relative; border: 1px solid var(--border); border-radius: var(--radius-base); padding: 6px; background: var(--bg-input); min-height: 36px; }

/* 모두 비우기 버튼 — 평소엔 반달, hover 시 원형 X */
.tbf-clear-all {
  position: absolute;
  left: -10px;             /* 왼쪽으로 튀어나오게 */
  top: 50%;
  transform: translateY(-50%);
  width: 14px;
  height: 28px;
  padding: 0;
  border: 1px solid rgba(248, 113, 113, 0.4);
  border-right: none;       /* 필드 경계와 매끄럽게 */
  border-radius: 50% 0 0 50%;  /* 반달 — 왼쪽이 둥글고 오른쪽이 직선 */
  background: rgba(248, 113, 113, 0.15);
  color: transparent;       /* X 숨김 */
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0;             /* 텍스트 안 보이게 */
  font-weight: var(--fw-bold);
  transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1);
  z-index: 10;
  overflow: hidden;
}
.tbf-clear-all:hover {
  left: -22px;              /* 더 튀어나오면서 */
  width: 26px;
  height: 26px;
  border-radius: 50%;       /* 원형 변형 */
  /* 전체 삭제는 '채움' 배지라 글자색(-fg)이 아니라 --state-alert.
     그 위 글자는 흰색 고정 — --state-alert 자체가 흰 글자와 4.6:1 을 맞춘 값이고,
     --text-primary 로 두면 라이트 모드에서 검정 글자가 얹혀 2.6:1 로 무너진다. */
  border: 1px solid var(--state-alert);
  background: var(--state-alert);
  color: #fff;
  font-size: 12px;
  box-shadow: 0 4px 14px rgba(248, 113, 113, 0.45), 0 0 0 2px rgba(248,113,113,0.15);
  overflow: visible;
}
.tbf-clear-all:active {
  transform: translateY(-50%) scale(0.9);
}
.tbf-clear-x {
  opacity: 0;
  transition: opacity 0.15s ease 0.05s;
}
.tbf-clear-all:hover .tbf-clear-x { opacity: 1; }
.tbf-blocks { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
.tbf-block {
  padding: 3px 10px; background: var(--bg-button); border: 1px solid var(--border);
  border-radius: 6px; color: var(--text-primary); font-size: 11px; cursor: pointer;
  transition: all 0.12s; user-select: none; position: relative;
}
.tbf-block:hover { border-color: var(--text-muted); }
.tbf-block[draggable="true"] { cursor: grab; }
.tbf-block[draggable="true"]:active { cursor: grabbing; opacity: 0.5; }
/* 색상 — 옛 11색을 태그 6색+중립으로 접었다. 11색은 5개 색상 무리에 뭉쳐 구분이
   안 됐고(인물수↔사물 2도), 토큰의 6색은 최소 이격 30도로 잡은 값이다.
   묶음은 토큰의 뜻 그대로: 인물·캐릭터(수·신체·특성) / 배경·구도 / 포즈·표정 /
   의상 / 효과·색 / NSFW / 중립(사물).
   테두리는 같은 토큰의 30% 틴트라 색이 갈라지지 않는다(하드코딩 rgba 는 안 따라온다). */
.tbf-block.bc-count { border-color: color-mix(in srgb, var(--tag-person) 30%, transparent); color: var(--tag-person); }
.tbf-block.bc-nsfw { border-color: color-mix(in srgb, var(--tag-nsfw) 30%, transparent); color: var(--tag-nsfw); }
.tbf-block.bc-body { border-color: color-mix(in srgb, var(--tag-person) 30%, transparent); color: var(--tag-person); }
.tbf-block.bc-clothing { border-color: color-mix(in srgb, var(--tag-wear) 30%, transparent); color: var(--tag-wear); }
.tbf-block.bc-action { border-color: color-mix(in srgb, var(--tag-pose) 30%, transparent); color: var(--tag-pose); }
.tbf-block.bc-expression { border-color: color-mix(in srgb, var(--tag-pose) 30%, transparent); color: var(--tag-pose); }
.tbf-block.bc-bg { border-color: color-mix(in srgb, var(--tag-scene) 30%, transparent); color: var(--tag-scene); }
.tbf-block.bc-effect { border-color: color-mix(in srgb, var(--tag-fx) 30%, transparent); color: var(--tag-fx); }
.tbf-block.bc-objects { border-color: color-mix(in srgb, var(--tag-neutral) 30%, transparent); color: var(--tag-neutral); }
.tbf-block.bc-color { border-color: color-mix(in srgb, var(--tag-fx) 30%, transparent); color: var(--tag-fx); }
.tbf-block.bc-trait { border-color: color-mix(in srgb, var(--tag-person) 30%, transparent); color: var(--tag-person); }
.tbf-block.wc-block { border-color: rgba(250,204,21,0.4); background: rgba(250,204,21,0.08); color: var(--accent); border-style: dashed; }
.wc-ico { margin-right: 2px; font-size: var(--fs-label); }
/* 드롭 마커 */
.tbf-drop-marker {
  width: 4px; height: 24px; background: var(--accent); border-radius: 3px; flex-shrink: 0;
  box-shadow: 0 0 8px var(--accent), 0 0 4px var(--accent);
  animation: tbf-marker-pulse 0.8s ease-in-out infinite;
}
@keyframes tbf-marker-pulse {
  0%, 100% { opacity: 1; transform: scaleY(1); }
  50% { opacity: 0.6; transform: scaleY(0.85); }
}
/* 편집 */
.tbf-edit { padding: 3px 8px; font-size: 11px; background: var(--bg-card); border: 1px solid var(--accent); border-radius: 4px; color: var(--text-primary); width: 120px; }
/* 추가 */
.tbf-add-wrap { position: relative; display: inline-block; }
.tbf-add { padding: 3px 8px; font-size: var(--fs-label); background: transparent; border: 1px dashed var(--border); border-radius: 4px; color: var(--text-muted); width: 80px; min-width: 60px; }
.tbf-add:focus { border-color: var(--accent); color: var(--text-primary); width: 140px; }
.ac-popup-block {
  position: absolute; top: 100%; left: 0; z-index: 100;
  margin-top: 2px; min-width: 180px; max-height: 200px; overflow-y: auto;
  background: var(--bg-secondary); border: 1px solid var(--accent);
  border-radius: 6px; padding: 2px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.5);
}
.ac-popup-block .ac-item {
  padding: 5px 10px; font-size: 11px; color: var(--text-primary);
  cursor: pointer; border-radius: 4px;
}
.ac-popup-block .ac-item:hover, .ac-popup-block .ac-item.selected {
  background: rgba(96,165,250,0.2); color: var(--state-info-fg);
}
.ac-popup-block .ac-item .ac-ko { margin-left: 8px; font-size: 10px; color: var(--text-muted); }
.ac-popup-block .ac-item.selected .ac-ko { color: inherit; opacity: .8; }
/* neg */
.neg .tbf-block { border-color: rgba(248,113,113,0.2); color: var(--state-alert-fg); font-size: var(--fs-label); }
</style>
