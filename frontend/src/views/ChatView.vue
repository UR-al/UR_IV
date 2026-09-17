<template>
  <div class="chat-view" @dragover.prevent="dragOver = true" @dragleave="dragOver = false" @drop.prevent="onDrop">
    <!-- 대화 목록 — GemmaStudio 처럼 왼쪽 열. 최근 것이 위. -->
    <button v-if="showThreads" class="cm-thread-backdrop" type="button" aria-label="대화 목록 닫기" @click="showThreads = false"></button>
    <aside id="chat-threads" class="chat-threads" :class="{ 'mobile-open': showThreads }">
      <div class="ct-head">
        <button class="ct-new" type="button" @click="newThread" title="새 대화 (Ctrl+N)">
          <Icon name="plus" size="15" /> 새 대화
        </button>
        <div class="ct-search">
          <Icon name="search" size="13" />
          <input v-model="search" placeholder="대화 찾기" spellcheck="false" />
        </div>
      </div>
      <div class="ct-list">
        <div v-if="!visibleThreads.length" class="ct-empty">{{ search ? '맞는 대화가 없습니다' : '아직 대화가 없습니다' }}</div>
        <div v-for="t in visibleThreads" :key="t.id" class="ct-row">
        <button type="button" class="ct-item" :class="{ on: t.id === activeId }" @click="activeId = t.id; showThreads = false" :title="t.title || '새 대화'">
          <Icon name="message" size="14" />
          <span class="ct-title">{{ t.title || '새 대화' }}</span>
        </button>
        <button class="ct-del" type="button" :aria-label="`${t.title || '새 대화'} 삭제`" title="대화 삭제" @click="deleteThread(t.id)"><Icon name="close" size="12" /></button>
        </div>
      </div>
      <div class="ct-foot">
        <span class="ct-foot-label">모델 · {{ provider === 'lmstudio' ? 'LM Studio' : 'Ollama' }}</span>
        <CustomSelect v-if="models.length" v-model="model" :options="models" placeholder="모델 선택..." @update:modelValue="saveModel" />
        <span v-else class="ct-foot-none" :title="url">모델 없음 — 대화 설정에서 연결 확인</span>
      </div>
    </aside>

    <!-- 대화 -->
    <section class="chat-main">
      <header class="cm-head">
        <button class="cm-tool cm-mobile-threads" type="button" aria-controls="chat-threads" :aria-expanded="showThreads" @click="showThreads = !showThreads">대화</button>
        <div class="cm-title-wrap">
          <input v-if="renaming" ref="renameRef" v-model="renameDraft" class="cm-rename" @keydown.enter="finishRename" @keydown.esc="renaming = false" @blur="finishRename" />
          <h2 v-else class="cm-title" @dblclick="startRename" :title="'더블클릭해서 이름 바꾸기'">{{ active?.title || '새 대화' }}</h2>
          <span class="cm-sub">{{ model || '모델 없음' }}<template v-if="active?.messages.length"> · {{ active.messages.length }}개 메시지</template></span>
        </div>
        <div class="cm-tools">
          <span class="cm-status" :class="{ busy: !!busyId, off: !models.length }">
            <span class="dot"></span>{{ busyId ? '작업 중' : (models.length ? '준비됨' : '생성 가능 · 채팅 모델 없음') }}
          </span>
          <button class="cm-tool" type="button" title="대화 설정 — 지침 · 답변 최대 토큰 · 문맥 창 · 온도" @click="showSystem = !showSystem" :class="{ on: showSystem }"><Icon name="settings" size="14" /></button>
          <button class="cm-tool" type="button" title="Markdown 으로 내보내기" :disabled="!active?.messages.length" @click="exportMarkdown"><Icon name="download" size="14" /></button>
          <button class="cm-tool" type="button" title="대화 내용 비우기" :disabled="!active?.messages.length" @click="clearMessages"><Icon name="eraser" size="14" /></button>
        </div>
      </header>

      <div v-show="showSystem" class="cm-system">
        <div class="cm-provider-row">
          <label>대화 서버 <select v-model="provider" :disabled="!!busyId" @change="changeProvider"><option value="ollama">Ollama</option><option value="lmstudio">LM Studio</option></select></label>
          <label>서버 주소 <input v-model.trim="url" :disabled="!!busyId" aria-label="대화 서버 주소" placeholder="http://localhost:1234" @input="markPreferencesEdited" @change="saveConnection" /></label>
          <button type="button" :disabled="modelsLoading || !!busyId" @click="requestModels">모델 목록 새로고침</button>
        </div>
        <p v-if="modelsError" class="cm-schema-error" role="alert">{{ modelsError }}</p>
        <p v-if="provider === 'lmstudio'" class="cm-settings-hint">LM Studio의 Developer에서 서버를 시작하세요 (기본 포트 1234). 주소는 /v1 포함·생략 모두 가능합니다. 인증 없는 로컬 서버 연결을 지원하며, 문맥 길이·모델 로딩 설정은 LM Studio에서 변경합니다. AI 어시스트의 Ollama 설정은 변경하지 않습니다.</p>
        <label for="chat-system-prompt">지침 — 모든 대화의 맨 앞에 붙습니다</label>
        <div class="cm-preset-row">
          <select v-model="systemPreset" aria-label="지침 프리셋">
            <option value="">프리셋 선택…</option>
            <option v-for="preset in CHAT_SYSTEM_PRESETS" :key="preset.id" :value="preset.id">{{ preset.label }}</option>
            <option v-if="personalSystemPrompt !== null" value="personal">저장된 개인 지침 복원</option>
          </select>
          <button type="button" :disabled="!systemPreset" @click="applySystemPreset">선택한 지침 적용</button>
          <small>직접 적용할 때만 변경됩니다. 개인 지침은 복원할 수 있습니다.</small>
        </div>
        <textarea id="chat-system-prompt" v-model="systemPrompt" rows="3" spellcheck="false" @input="systemPreset = ''; markPreferencesEdited()" @change="saveSystemPrompt"></textarea>
        <InstructionPresets scope="chat" :instructions="systemPrompt" @apply="applyCustomSystemPreset" />
        <details class="cm-structured" :open="structuredEnabled">
          <summary>구조화된 출력 · JSON 스키마</summary>
          <label class="cm-schema-toggle"><input v-model="structuredEnabled" type="checkbox" :disabled="!!busyId" @change="saveStructured" /> JSON 스키마로 대화 응답 형식 제한</label>
          <p class="cm-settings-hint">켜면 ‘대화만’ 모드로 동작하며 이미지·영상 생성은 실행하지 않습니다. Ollama는 format, LM Studio는 response_format으로 전달합니다. 모델에 따라 지원 범위가 다르며, 구조는 제한해도 내용의 정확성까지 보장하지는 않습니다.</p>
          <label for="chat-json-schema">JSON Schema (Draft 2020-12 · 참조 없는 인라인 스키마)</label>
          <textarea id="chat-json-schema" v-model="schemaText" :disabled="!!busyId" rows="9" spellcheck="false" maxlength="128000" :aria-invalid="!!schemaError" aria-describedby="chat-schema-help" @input="markPreferencesEdited" @change="savePreferences"></textarea>
          <button type="button" :disabled="!!busyId" @click="schemaExamplePending = !schemaExamplePending">태그·자연어·한국어 설명 예제</button>
          <span v-if="schemaExamplePending">현재 스키마를 예제로 바꿀까요? <button type="button" @click="useSchemaExample">예제로 교체</button><button type="button" @click="schemaExamplePending = false">취소</button></span>
          <p id="chat-schema-help" class="cm-settings-hint">최대 64,000자. $ref/$id 참조는 지원하지 않습니다. 답변 최대 토큰이 ‘제한 없음’이면 JSON 출력에만 4,096 토큰을 적용합니다. 완료된 응답은 스키마를 다시 검증하며 중지·잘림·불일치는 오류로 표시합니다. 형식과 충돌하는 대화 지침도 함께 조정하세요.</p>
          <p v-if="schemaError" class="cm-schema-error" role="alert">{{ schemaError }}</p>
        </details>
        <div class="cm-opts">
          <label class="cm-opt">
            <span>답변 최대 토큰</span>
            <select v-model.number="chatOptions.numPredict" @change="saveChatOptions">
              <option :value="-1">제한 없음</option>
              <option v-for="n in PREDICT_CHOICES" :key="n" :value="n">{{ n.toLocaleString() }}</option>
            </select>
          </label>
          <label v-if="provider === 'ollama'" class="cm-opt">
            <span>문맥 창 (num_ctx)</span>
            <select v-model.number="chatOptions.numCtx" @change="saveChatOptions">
              <option :value="0">모델 기본</option>
              <option v-for="n in CTX_CHOICES" :key="n" :value="n">{{ n.toLocaleString() }}</option>
            </select>
          </label>
          <label class="cm-opt cm-opt-range">
            <span>온도 {{ chatOptions.temperature.toFixed(1) }}</span>
            <input type="range" min="0" max="2" step="0.1" v-model.number="chatOptions.temperature" aria-describedby="chat-temperature-help" @change="saveChatOptions" />
          </label>
          <span class="cm-opt-hint">답이 중간에 끊기면 답변 최대 토큰과 문맥 창을 같이 늘리세요. 문맥 창이 클수록 VRAM 을 더 씁니다 (이미지 한 장이 수백 토큰).</span>
        </div>
        <p id="chat-temperature-help" class="cm-settings-hint">온도는 표현의 다양성입니다. 낮을수록 보수적이고 반복 가능한 답, 높을수록 다양한 답을 만듭니다. 태그·사실 설명은 낮게, 아이디어 탐색은 높게 조절하세요. 정확도를 보장하는 설정은 아닙니다.</p>
        <div class="cm-model-info" aria-live="polite">
          <div class="cm-model-info-title"><strong>선택 모델의 실제 지원 범위</strong><button type="button" :disabled="modelInfoLoading || !model" @click="requestModelInfo">다시 확인</button></div>
          <span v-if="modelInfoLoading">모델 정보를 확인하는 중…</span>
          <span v-else-if="modelInfoError">{{ modelInfoError }} — 추론 설정은 모델 기본값으로 전달합니다.</span>
          <template v-else-if="modelInfo">
            <span>{{ modelInfo.architecture || '구조 미제공' }} · {{ modelInfo.parameterSize || '크기 미제공' }} · {{ modelInfo.quantization || '양자화 미제공' }}</span>
            <span>MoE: {{ modelInfo.moe === true ? `확인됨 · 전문가 ${modelInfo.experts}개` : modelInfo.moe === false ? '아님' : '메타데이터에 없어 확인 불가' }}<template v-if="modelInfo.activeExperts"> · 토큰당 활성 {{ modelInfo.activeExperts }}개</template></span>
            <span>이미지 이해: {{ modelInfo.vision === true ? '지원' : modelInfo.vision === false ? '미지원' : '확인 불가' }} · 추론: {{ modelInfo.thinkingMode === 'levels' ? '강도 선택 (OFF 불가)' : modelInfo.thinkingMode === 'boolean' ? '켜기/끄기 지원' : modelInfo.thinkingMode === 'none' ? '미지원' : '확인 불가' }}<template v-if="modelInfo.contextLength"> · 모델 최대 문맥 {{ modelInfo.contextLength.toLocaleString() }} 토큰</template></span>
          </template>
          <label v-if="modelInfo?.thinkingMode === 'levels'" class="cm-thinking-level">추론 강도 <select v-model="thinkingLevel" @change="saveThinkingLevel"><option value="low">낮음 · 빠르게</option><option value="medium">중간</option><option value="high">높음 · 깊게</option></select></label>
          <p class="cm-settings-hint">MoE는 모델 내부 구조로 자동 사용됩니다. Dense↔MoE 전환이나 전문가 수 변경은 Ollama 채팅 옵션이 아닙니다. 다른 구조를 쓰려면 모델을 선택하세요. 도구 사용은 모델 지원 여부와 별개로 이 채팅에서는 실행하지 않습니다.</p>
        </div>
        <details class="cm-assist-link">
          <summary>Settings와 연동된 AI 어시스트 지침 편집 · 대화에는 적용 안 됨</summary>
          <AiAssistInstructionsSettings id-prefix="chat-ai-assist" />
        </details>
      </div>

      <div class="cm-scroll" ref="scrollRef" @scroll="onScroll" @wheel="onWheel">
        <div class="cm-col">
          <!-- 빈 상태 -->
          <div v-if="!active || !active.messages.length" class="cm-hero">
            <div class="cm-hero-mark"><Icon name="sparkles" size="22" /></div>
            <h3>무엇을 도와드릴까요?</h3>
            <p>로컬 모델과 대화하거나 이미지·영상을 실제로 생성합니다. 생성에는 Ollama가 필요하지 않습니다. 참조 이미지 한 장을 첨부하면 편집·이미지 기반 영상으로 이어집니다.</p>
            <div class="cm-suggest">
              <button v-for="s in SUGGESTIONS" :key="s" type="button" @click="draft = s; focusComposer()">{{ s }}</button>
            </div>
          </div>

          <!-- 메시지 — 말풍선이 아니라 한 열. GemmaStudio 와 같은 이유: 긴 답이 읽힌다. -->
          <article v-for="m in active?.messages || []" :key="m.id" class="msg" :class="[m.role, { pending: m.pending, error: !!m.error }]">
            <div class="msg-avatar" aria-hidden="true">{{ m.role === 'user' ? '나' : 'AI' }}</div>
            <div class="msg-body">
              <div class="msg-role">{{ m.role === 'user' ? '나' : (m.model || model || 'AI') }}<span v-if="m.evalCount" class="msg-meta"> · {{ m.evalCount.toLocaleString() }} 토큰<template v-if="m.durationMs"> · {{ (m.durationMs / 1000).toFixed(1) }}초</template></span></div>
              <div v-if="m.images?.length" class="msg-images">
                <img v-for="(src, i) in m.images" :key="i" :src="imageSrc(src)" alt="첨부 이미지" @load="onMediaLoad" />
              </div>
              <details v-if="m.thinking" class="msg-think" :open="!!m.pending">
                <summary><Icon name="bulb" size="12" /> {{ m.pending && !m.content ? '생각하는 중…' : `생각 (${m.thinking.length}자)` }}</summary>
                <pre>{{ m.thinking }}</pre>
              </details>
              <pre v-if="m.role === 'assistant' && m.structured" class="msg-content msg-json">{{ m.content }}</pre>
              <div v-else-if="m.role === 'assistant'" class="msg-content md" v-html="renderMarkdown(m.content)"></div>
              <div v-else class="msg-content plain">{{ m.content }}</div>
              <div v-if="m.generation && m.pending" class="msg-generation" role="status" aria-live="polite">
                <strong>{{ m.generation.kind === 'video' ? '영상 생성' : '이미지 생성' }}</strong>
                <span>{{ m.generation.message || '생성 중' }}</span>
                <progress v-if="m.generation.progress !== undefined" max="100" :value="m.generation.progress"></progress>
              </div>
              <div v-if="m.artifacts?.length" class="msg-artifacts">
                <figure v-for="a in m.artifacts" :key="a.path">
                  <video v-if="a.kind === 'video'" :src="mediaUrl(a.path)" controls preload="metadata" @loadedmetadata="onMediaLoad"></video>
                  <audio v-else-if="a.kind === 'audio'" :src="mediaUrl(a.path)" controls preload="metadata"></audio>
                  <img v-else :src="mediaUrl(a.path)" alt="생성 결과" @load="onMediaLoad" />
                  <figcaption><span :title="a.path">{{ a.filename || '생성 결과' }}</span>
                    <button type="button" @click="copyText(a.path)">경로 복사</button>
                    <button v-if="a.kind === 'image'" type="button" @click="attachments = [a.path]; focusComposer()">참조로 사용</button>
                  </figcaption>
                </figure>
              </div>
              <div v-if="m.pending && !m.content && !m.thinking && !m.generation" class="msg-thinking"><span></span><span></span><span></span></div>
              <div v-if="m.error" class="msg-error">{{ m.error }}</div>
              <div v-else-if="m.doneReason === 'length' && !m.pending" class="msg-cut">답변 최대 토큰에 걸려 잘렸습니다 — 톱니(설정)에서 늘릴 수 있습니다</div>
              <div class="msg-actions" v-if="!m.pending">
                <button type="button" title="복사" @click="copyText(m.content)"><Icon name="clipboard" size="13" /></button>
                <button v-if="m.role === 'assistant' && isLast(m)" type="button" title="같은 요청 · 현재 모델 설정으로 다시 생성" :disabled="!!busyId" @click="regenerate"><Icon name="refresh" size="13" /></button>
              </div>
            </div>
          </article>
        </div>
      </div>

      <!-- 컴포저 — 스크롤 영역 *아래*, 흐름 안에 둔다. 위에 띄우면 마지막 메시지가 그 뒤로 숨는다. -->
      <div class="cm-bottom" ref="bottomRef">
        <button v-if="!followBottom" class="cm-jump" type="button" @click="scrollToBottom(true)" title="맨 아래로"><Icon name="arrow-down" size="14" /></button>
         <div class="cm-composer" :class="{ drag: dragOver }">
         <div class="cmp-generation-options">
           <label>요청 <select v-model="generationRequest.mode" :disabled="!!busyId || structuredEnabled" aria-label="채팅 또는 생성 모드">
             <option value="auto">자동</option><option value="chat">대화만</option>
             <option value="image">이미지 생성</option><option value="video">영상 · H3 기본 품질</option>
           </select></label>
           <label v-if="generationRequest.mode === 'auto' || generationRequest.mode === 'image'">이미지 모델
             <select v-model="generationRequest.family" :disabled="!!busyId" aria-label="이미지 생성 모델">
               <option value="current">현재 선택 모델 · Anima 등</option><option value="krea2">Krea2</option>
             </select>
           </label>
           <label v-if="generationRequest.mode === 'video'">길이
             <select v-model.number="generationRequest.duration" :disabled="!!busyId" aria-label="영상 길이">
               <option :value="3">3초</option><option :value="5">5초</option><option :value="10">10초</option>
             </select>
           </label>
           <label v-if="attachments.length && generationRequest.family === 'current' && generationRequest.mode !== 'video'">변화량
             <input v-model.number="generationRequest.denoise" type="number" min="0.01" max="1" step="0.05" :disabled="!!busyId" aria-label="이미지 편집 변화량" />
           </label>
           <small>{{ structuredEnabled ? 'JSON 스키마 출력 켜짐 · 대화만 전송' : generationRequest.mode === 'chat' ? '대화 모델에 질문만 전달합니다' : '자동은 명확한 생성 요청만 실행 · 현재 모델은 T2I 설정 사용 · Krea2/H3는 ComfyUI 필요' }}</small>
         </div>
        <div v-if="attachments.length" class="cmp-attach">
          <div v-for="(a, i) in attachments" :key="i" class="cmp-thumb">
            <img :src="imageSrc(a)" alt="" />
            <button type="button" title="첨부 제거" @click="attachments.splice(i, 1)"><Icon name="close" size="11" /></button>
          </div>
        </div>
        <textarea ref="composerRef" v-model="draft" class="cmp-input" rows="1" spellcheck="false"
          placeholder="질문하거나 생성할 장면을 입력하세요 — Enter 보내기 · Shift+Enter 줄바꿈"
          @keydown="onComposerKey" @input="autoGrow" @paste="onPaste"></textarea>
        <div class="cmp-bar">
          <span class="cmp-hint"><Icon name="image" size="13" /> 이미지·텍스트 파일을 끌어 놓거나 붙여 넣기 · 히스토리 카드도 됩니다</span>
          <span class="cmp-spacer"></span>
          <button v-if="modelInfo?.thinkingMode === 'boolean'" type="button" class="cmp-think" :class="{ on: deepThink }" @click="toggleThink"
            :title="deepThink ? '깊은 추론 켜짐 — 답하기 전에 생각합니다 (느리고, 생각이 접힌 블록으로 보입니다)' : '깊은 추론 꺼짐 — 바로 답합니다 (빠름)'">
            <Icon name="bulb" size="13" /> 깊은 추론
          </button>
          <span v-else-if="modelInfo?.thinkingMode === 'levels'" class="cmp-think" title="대화 설정에서 강도를 선택합니다. 이 모델은 추론을 끌 수 없습니다.">추론 {{ thinkingLevel === 'low' ? '낮음' : thinkingLevel === 'high' ? '높음' : '중간' }}</span>
          <button v-if="busyId" type="button" class="cmp-send stop" title="중지 (Esc)" @click="stop"><Icon name="stop" size="14" /></button>
          <button v-else type="button" class="cmp-send" title="보내기 (Enter)" :disabled="!canSend" @click="send"><Icon name="arrow-up" size="15" /></button>
        </div>
        </div>
      </div>
    </section>
    <dialog ref="confirmRef" class="cm-confirm" aria-labelledby="chat-confirm-title" aria-describedby="chat-confirm-description" @cancel.prevent="dismissConfirmation">
      <form @submit.prevent="confirmDeletion">
        <h3 id="chat-confirm-title">{{ confirmation?.kind === 'clear' ? '대화 내용 비우기' : '대화 삭제' }}</h3>
        <p id="chat-confirm-description">“{{ confirmation?.title || '새 대화' }}”{{ confirmation?.kind === 'clear' ? '의 메시지를 모두 비울까요?' : '를 삭제할까요?' }} 이 작업은 되돌릴 수 없습니다.</p>
        <div><button ref="confirmCancelRef" type="button" @click="dismissConfirmation">취소</button><button type="submit" class="danger">{{ confirmation?.kind === 'clear' ? '모두 비우기' : '삭제' }}</button></div>
      </form>
    </dialog>
  </div>
</template>

<script setup lang="ts">
/**
 * 대화 탭 — 로컬 Ollama 와 스트리밍으로 대화한다.
 *
 * GemmaStudio 의 골격을 따랐다: 왼쪽에 대화 목록, 가운데에 **말풍선이 아닌 한 열**의
 * 메시지, 아래에 카드형 컴포저. 다른 점 하나 — 여기서는 이미지를 **모델에게 픽셀로**
 * 보낸다(GemmaStudio 는 OCR 글자만 보냈다). 그래서 Settings 의 추천 모델이 전부
 * 비전 모델이다.
 *
 * 상태는 Python 파일(config/chat_threads.json)에 저장한다 — localStorage 는 이미지가
 * 붙는 순간 5MB 를 넘겨 조용히 실패한다. 브리지: chat_load/chat_save/chat_send/chat_stop,
 * 시그널 chatThreads/chatToken/chatDone (tests/test_bridge_contract.py 가 이름을 지킨다).
 */
import { computed, nextTick, onActivated, onMounted, onUnmounted, ref, watch } from 'vue'
import { getBackend, onBackendEvent } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'
import { mediaUrl } from '../utils/media.js'
import { renderMarkdown } from '../utils/chatMarkdown'
import { copyTextToClipboard } from '../utils/clipboard'
import { CHAT_SYSTEM_PRESETS, selectSystemPreset, thinkingValue, type ChatModelInfo } from '../utils/chatSettings'
import { applyGenerationEvent, artifactMarkdown, type ChatArtifact, type GenerationRequest, type GenerationState } from '../utils/chatGeneration'
import CustomSelect from '../components/CustomSelect.vue'
import AiAssistInstructionsSettings from '../components/AiAssistInstructionsSettings.vue'
import InstructionPresets from '../components/InstructionPresets.vue'
import { PROMPT_JSON_SCHEMA, parseChatSchema } from '../utils/chatStructuredOutput'
import type { AiAssistInstructions } from '../types/bridge'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  createdAt: number
  images?: string[]
  model?: string
  thinking?: string
  evalCount?: number
  durationMs?: number
  doneReason?: string
  pending?: boolean
  requestId?: string
  error?: string
  structured?: boolean
  artifacts?: ChatArtifact[]
  generationRequest?: GenerationRequest
  generation?: GenerationState
}
interface ChatThread {
  id: string
  title: string
  model: string
  createdAt: number
  updatedAt: number
  messages: ChatMessage[]
}

const SUGGESTIONS = [
  '이 프롬프트를 더 자연스럽게 다듬어 줘',
  '첨부한 이미지를 자세히 설명해 줘',
  '이 장면에 어울리는 Danbooru 태그를 추천해 줘',
  '눈밭에 앉아 있는 고양이 이미지 만들어줘',
]
const DEFAULT_SYSTEM = CHAT_SYSTEM_PRESETS[0].prompt
const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8)

// ── 상태 ──
const threads = ref<ChatThread[]>([])
const activeId = ref('')
const search = ref('')
const draft = ref('')
const attachments = ref<string[]>([])
const generationRequest = ref<GenerationRequest>({ mode: 'auto', family: 'current', duration: 5, denoise: 0.65 })
const models = ref<string[]>([])
const provider = ref<'ollama' | 'lmstudio'>(localStorage.getItem('chatProvider') === 'lmstudio' ? 'lmstudio' : 'ollama')
const model = ref(localStorage.getItem(provider.value === 'lmstudio' ? 'chatLmStudioModel' : 'ollamaModel') || '')
const url = ref(localStorage.getItem(provider.value === 'lmstudio' ? 'chatLmStudioUrl' : 'ollamaUrl') || (provider.value === 'lmstudio' ? 'http://localhost:1234' : 'http://localhost:11434'))
const modelsError = ref('')
const modelsLoading = ref(false)
let modelsRequestId = ''
let modelsTimer: ReturnType<typeof setTimeout> | undefined
const structuredEnabled = ref(localStorage.getItem('chatStructuredEnabled') === '1')
const schemaText = ref(localStorage.getItem('chatJsonSchema') ?? PROMPT_JSON_SCHEMA)
const schemaExamplePending = ref(false)
const schemaError = computed(() => { try { parseChatSchema(schemaText.value); return '' } catch (error) { return (error as Error).message } })
if (structuredEnabled.value) generationRequest.value.mode = 'chat'
let preferencesEdited = false
function markPreferencesEdited() { preferencesEdited = true }
let disposed = false
const systemPrompt = ref(localStorage.getItem('chatSystemPrompt') ?? DEFAULT_SYSTEM)
const personalSystemPrompt = ref<string | null>(localStorage.getItem('chatPersonalSystemPrompt'))
const systemPreset = ref('')
const showSystem = ref(false)
const showThreads = ref(false)
const modelInfo = ref<ChatModelInfo | null>(null)
const modelInfoError = ref('')
const modelInfoLoading = ref(false)
let modelInfoRequestId = ''
let modelInfoTimer: ReturnType<typeof setTimeout> | null = null
const thinkingLevel = ref(['low', 'medium', 'high'].includes(localStorage.getItem('chatThinkingLevel') || '') ? localStorage.getItem('chatThinkingLevel')! : 'medium')
function saveThinkingLevel() { localStorage.setItem('chatThinkingLevel', thinkingLevel.value) }
const confirmation = ref<{ kind: 'clear' | 'delete'; id: string; title: string } | null>(null)
const confirmRef = ref<HTMLDialogElement | null>(null)
const confirmCancelRef = ref<HTMLButtonElement | null>(null)
let confirmationFocus: HTMLElement | null = null
// 생성 옵션 — Ollama 기본 문맥 창(4096)은 지침 + 이미지 + 24턴 기록이면 금방 찬다. 답이 잘리면 여기서 늘린다.
interface ChatOptions { temperature: number; numPredict: number; numCtx: number }
const DEFAULT_OPTIONS: ChatOptions = { temperature: 0.7, numPredict: -1, numCtx: 8192 }
const PREDICT_CHOICES = [512, 1024, 2048, 4096, 8192, 16384]
const CTX_CHOICES = [4096, 8192, 16384, 32768, 65536, 131072]
function loadChatOptions(): ChatOptions {
  const o = { ...DEFAULT_OPTIONS }
  try {
    const raw = JSON.parse(localStorage.getItem('chatOptions.v1') || 'null')
    if (raw && typeof raw === 'object') {
      // 목록에 없는 값(구버전·손으로 고친 것)은 select 가 빈칸으로 보인다 — 기본값으로 되돌린다
      if (typeof raw.temperature === 'number' && raw.temperature >= 0 && raw.temperature <= 2) o.temperature = raw.temperature
      if (raw.numPredict === -1 || PREDICT_CHOICES.includes(raw.numPredict)) o.numPredict = raw.numPredict
      if (raw.numCtx === 0 || CTX_CHOICES.includes(raw.numCtx)) o.numCtx = raw.numCtx
    }
  } catch {}
  return o
}
const chatOptions = ref<ChatOptions>(loadChatOptions())
function saveChatOptions() { localStorage.setItem('chatOptions.v1', JSON.stringify(chatOptions.value)); savePreferences() }
function ollamaOptions(): Record<string, number> {
  const o: Record<string, number> = { temperature: chatOptions.value.temperature }
  // '제한 없음' 도 -1 로 명시한다 — 모델파일이 num_predict 를 박아 둔 모델이 있다
  o.num_predict = chatOptions.value.numPredict > 0 ? chatOptions.value.numPredict : -1
  if (provider.value === 'ollama' && chatOptions.value.numCtx > 0) o.num_ctx = chatOptions.value.numCtx
  return o
}
// 깊은 추론 — Gemma 4 / Qwen3.x 같은 thinking 모델은 기본으로 생각부터 하느라 첫 글자가 1분 뒤에 온다.
// GemmaStudio 처럼 빠른 답변이 기본, 원할 때만 켠다. 켜면 생각이 접힌 블록으로 같이 흐른다.
const deepThink = ref(localStorage.getItem('chatThink') === '1')
function toggleThink() { deepThink.value = !deepThink.value; localStorage.setItem('chatThink', deepThink.value ? '1' : '0') }
const busyId = ref('')
const dragOver = ref(false)
const followBottom = ref(true)
const renaming = ref(false)
const renameDraft = ref('')
const renameRef = ref<HTMLInputElement | null>(null)
const composerRef = ref<HTMLTextAreaElement | null>(null)
const scrollRef = ref<HTMLElement | null>(null)
const bottomRef = ref<HTMLElement | null>(null)

const active = computed(() => threads.value.find((t) => t.id === activeId.value) || null)
const visibleThreads = computed(() => {
  const q = search.value.trim().toLowerCase()
  const list = [...threads.value].sort((a, b) => b.updatedAt - a.updatedAt)
  return q ? list.filter((t) => (t.title || '').toLowerCase().includes(q)) : list
})
const canSend = computed(() => !busyId.value && (draft.value.trim().length > 0 || attachments.value.length > 0))

// ── 저장 — 파일로, 지연해서 ──
let saveTimer: ReturnType<typeof setTimeout> | null = null
function scheduleSave() {
  if (saveTimer) clearTimeout(saveTimer)
  saveTimer = setTimeout(() => {
    saveTimer = null
    const snapshot = threads.value.map((t) => ({
      ...t,
      messages: t.messages.filter((m) => !m.pending).map(({ pending, requestId, ...rest }) => rest),
    }))
    requestAction('chat_save', { threads: snapshot })
  }, 600)
}
watch(threads, scheduleSave, { deep: true })

// ── 대화 목록 ──
function newThread() {
  // 이미 빈 대화가 있으면 그걸 쓴다 — 빈 것이 쌓이지 않게
  const empty = threads.value.find((t) => !t.messages.length)
  if (empty) { activeId.value = empty.id; focusComposer(); return }
  const t: ChatThread = { id: uid(), title: '', model: model.value, createdAt: Date.now(), updatedAt: Date.now(), messages: [] }
  threads.value.unshift(t)
  activeId.value = t.id
  focusComposer()
}
function deleteThread(id: string) {
  const t = threads.value.find((x) => x.id === id)
  if (t) openConfirmation('delete', t)
}
function clearMessages() {
  if (active.value) openConfirmation('clear', active.value)
}
function openConfirmation(kind: 'clear' | 'delete', thread: ChatThread) {
  confirmationFocus = document.activeElement as HTMLElement | null
  confirmation.value = { kind, id: thread.id, title: thread.title }
  nextTick(() => { confirmRef.value?.showModal(); confirmCancelRef.value?.focus() })
}
function dismissConfirmation() {
  confirmRef.value?.close()
  confirmation.value = null
  nextTick(() => { if (confirmationFocus?.isConnected) confirmationFocus.focus(); else focusComposer() })
}
function confirmDeletion() {
  const action = confirmation.value
  if (!action) return
  const thread = threads.value.find(t => t.id === action.id)
  // The target is captured when the dialog opens; another thread stays untouched.
  if (busyId.value && thread?.messages.some(m => m.requestId === busyId.value)) stop()
  if (thread && action.kind === 'clear') {
    thread.messages = []; thread.title = ''; thread.updatedAt = Date.now()
  } else if (thread) {
    threads.value = threads.value.filter(t => t.id !== thread.id)
    if (activeId.value === thread.id) activeId.value = threads.value[0]?.id || ''
    if (!threads.value.length) newThread()
  }
  dismissConfirmation()
}
function exportMarkdown() {
  const t = active.value
  if (!t || !t.messages.length) return
  const title = t.title || '새 대화'
  const parts = [`# ${title}`, '', `_${new Date(t.createdAt).toLocaleString()} · ${t.model || model.value || ''}_`, '']
  for (const m of t.messages) {
    if (m.pending) continue   // 아직 흐르는 답은 반쪽이다
    parts.push(`## ${m.role === 'user' ? '나' : (m.model || 'AI')}`, '')
    if (m.images?.length) parts.push(`(이미지 ${m.images.length}장 첨부)`, '')
    parts.push(m.content || '', '')
    if (m.artifacts?.length) parts.push(...m.artifacts.map(artifactMarkdown), '')
  }
  requestAction('chat_export', { title, markdown: parts.join('\n') })
}
function startRename() {
  if (!active.value) return
  renameDraft.value = active.value.title || ''
  renaming.value = true
  nextTick(() => renameRef.value?.select())
}
function finishRename() {
  if (!renaming.value) return
  renaming.value = false
  if (active.value) { active.value.title = renameDraft.value.trim().slice(0, 80); active.value.updatedAt = Date.now() }
}

// ── 보내기 · 받기 ──
function send() {
  if (!canSend.value || !active.value) return
  if (!checkSchema()) return
  const thread = active.value
  const text = draft.value.trim()
  const user: ChatMessage = { id: uid(), role: 'user', content: text, createdAt: Date.now() }
  user.generationRequest = { ...generationRequest.value, hadImage: attachments.value.length > 0 }
  if (attachments.value.length) user.images = [...attachments.value]
  thread.messages.push(user)
  if (!thread.title) thread.title = (text || '이미지').replace(/\s+/g, ' ').slice(0, 36)
  draft.value = ''
  attachments.value = []
  nextTick(autoGrow)   // v-model 이 DOM 을 비운 *뒤에* 재야 컴포저가 한 줄로 돌아온다
  ask(thread)
}
function ask(thread: ChatThread, retryRequest?: GenerationRequest) {
  const requestId = uid()
  const latestUser = [...thread.messages].reverse().find(m => m.role === 'user')
  const request = { ...(retryRequest || latestUser?.generationRequest || generationRequest.value) }
  if (structuredEnabled.value) request.mode = 'chat'
  const assistant: ChatMessage = { id: uid(), role: 'assistant', content: '', createdAt: Date.now(), pending: true, requestId, model: model.value }
  assistant.generationRequest = request
  assistant.structured = structuredEnabled.value
  thread.messages.push(assistant)
  thread.model = model.value
  thread.updatedAt = Date.now()
  busyId.value = requestId
  followBottom.value = true
  nextTick(() => scrollToBottom(false))
  requestAction('chat_send', {
    id: requestId,
    provider: provider.value,
    schema: structuredEnabled.value ? parseChatSchema(schemaText.value) : undefined,
    url: url.value,
    model: model.value,
    system: systemPrompt.value,
    think: thinkingValue(modelInfo.value, deepThink.value, thinkingLevel.value),
    options: ollamaOptions(),
    generation: request,
    messages: thread.messages
      .filter((m) => !m.pending && !m.error)
      .map((m) => ({ role: m.role, content: m.content, images: m.images })),
  })
}
function stop() {
  if (!busyId.value) return
  requestAction('chat_stop', { id: busyId.value })
}
function regenerate() {
  const thread = active.value
  if (!thread || busyId.value) return
  if (!checkSchema()) return
  const last = thread.messages[thread.messages.length - 1]
  if (last?.role === 'assistant') thread.messages.pop()
  ask(thread, last?.generationRequest)
}
function findPending(requestId: string): ChatMessage | null {
  for (const t of threads.value) {
    const m = t.messages.find((x) => x.requestId === requestId)
    if (m) return m
  }
  return null
}
function onToken(json: string) {
  try {
    const { id, text, thinking } = JSON.parse(json)
    const m = findPending(id)
    if (!m) return
    if (thinking) m.thinking = (m.thinking || '') + thinking
    if (!text) { if (thinking && followBottom.value) nextTick(() => scrollToBottom(false)); return }
    m.content += text
    if (followBottom.value) nextTick(() => scrollToBottom(false))
  } catch {}
}
function onDone(json: string) {
  try {
    const d = JSON.parse(json)
    const m = findPending(d.id)
    if (m) {
      if (typeof d.content === 'string' && d.content.length >= m.content.length) m.content = d.content
      if (!d.ok) m.error = d.error || '응답을 받지 못했습니다'
      else if (d.stopped) m.error = m.content ? '' : '중지됨'
      if (typeof d.evalCount === 'number') m.evalCount = d.evalCount
      if (typeof d.durationMs === 'number') m.durationMs = d.durationMs
      if (typeof d.doneReason === 'string' && d.doneReason) m.doneReason = d.doneReason
      m.pending = false
      delete m.requestId
      const t = threads.value.find((x) => x.messages.includes(m))
      if (t) t.updatedAt = Date.now()
      // 끝나면 액션 줄·메타가 생기고 생각 블록이 접혀 높이가 바뀐다 — 보고 있던 맨 아래를 지킨다
      if (followBottom.value) nextTick(() => scrollToBottom(false))
    }
    if (busyId.value === d.id) busyId.value = ''
  } catch { busyId.value = '' }
}
function onGeneration(json: string) {
  try {
    const event = JSON.parse(json)
    const message = findPending(event.id)
    if (!message || !applyGenerationEvent(message, event)) return
    if (event.model) message.model = String(event.model)
    const thread = threads.value.find(t => t.messages.includes(message))
    if (thread) thread.updatedAt = Date.now()
    if (event.done && busyId.value === event.id) busyId.value = ''
    if (followBottom.value) nextTick(() => scrollToBottom(false))
  } catch { /* unrelated or malformed events do not release another request */ }
}
function isLast(m: ChatMessage) {
  const msgs = active.value?.messages || []
  return msgs[msgs.length - 1] === m
}

// ── 첨부 — 이미지는 픽셀로 보낸다. 큰 사진은 1024px 로 줄여 base64 (모델·저장 둘 다 가볍게) ──
async function addImageFile(file: File) {
  if (!file.type.startsWith('image/')) return
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const r = new FileReader(); r.onload = () => resolve(String(r.result)); r.onerror = reject; r.readAsDataURL(file)
  })
  attachments.value.push(await downscale(dataUrl))
}
function downscale(dataUrl: string, max = 1024): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => {
      const scale = Math.min(1, max / Math.max(img.width, img.height))
      if (scale >= 1) { resolve(dataUrl); return }
      const c = document.createElement('canvas')
      c.width = Math.round(img.width * scale); c.height = Math.round(img.height * scale)
      c.getContext('2d')!.drawImage(img, 0, 0, c.width, c.height)
      resolve(c.toDataURL('image/jpeg', 0.9))
    }
    img.onerror = () => resolve(dataUrl)
    img.src = dataUrl
  })
}
const TEXT_EXT = /\.(txt|md|json|ya?ml|toml|csv|py|js|ts|vue|html?|css|xml|ini|cfg|log|sh|ps1|bat)$/i
async function addTextFile(file: File) {
  // 텍스트·코드는 본문에 펜스로 넣는다 — 모델이 파일을 "읽는" 가장 확실한 길
  const text = (await file.text()).slice(0, 100_000)
  const lang = (file.name.match(/\.(\w+)$/)?.[1] || '').toLowerCase()
  draft.value = (draft.value ? draft.value + '\n\n' : '') + `파일 \`${file.name}\`:\n\`\`\`${lang}\n${text}\n\`\`\`\n`
  nextTick(autoGrow)
}
function onDrop(e: DragEvent) {
  dragOver.value = false
  const files = [...(e.dataTransfer?.files || [])]
  if (files.length) {
    files.slice(0, 8).forEach((f) => {
      if (f.type.startsWith('image/')) addImageFile(f)
      else if (f.type.startsWith('text/') || TEXT_EXT.test(f.name)) addTextFile(f)
    })
    return
  }
  // 히스토리·갤러리 카드는 경로 텍스트로 온다 — Python 이 파일을 읽어 base64 로 넣는다
  const path = e.dataTransfer?.getData('text/plain') || ''
  if (path && /[\\/]/.test(path) && attachments.value.length < 8) attachments.value.push(path.replace(/\\/g, '/'))
}
function onPaste(e: ClipboardEvent) {
  const items = [...(e.clipboardData?.items || [])].filter((it) => it.type.startsWith('image/'))
  if (!items.length) return
  e.preventDefault()
  items.forEach((it) => { const f = it.getAsFile(); if (f) addImageFile(f) })
}
function imageSrc(v: string) { return v.startsWith('data:') ? v : mediaUrl(v) }

// ── 입력 ──
function onComposerKey(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send() }
  else if (e.key === 'Escape' && busyId.value) { e.preventDefault(); stop() }
}
function autoGrow() {
  const el = composerRef.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 180) + 'px'
}
function focusComposer() { nextTick(() => composerRef.value?.focus()) }
function onGlobalKey(e: KeyboardEvent) {
  if (confirmation.value) return
  if (e.ctrlKey && (e.key === 'n' || e.key === 'N') && document.querySelector('.chat-view')) { e.preventDefault(); newThread() }
}

// ── 스크롤 ──
let _settleUntil = 0   // 부드러운 점프 중에는 중간 scroll 이벤트가 followBottom 을 끄지 않게
function onScroll() {
  const el = scrollRef.value
  if (!el || performance.now() < _settleUntil) return
  followBottom.value = el.scrollHeight - el.scrollTop - el.clientHeight < 80
}
function onWheel(e: WheelEvent) {
  // 스트리밍 중 위로 굴리면 즉시 놓아준다 — scroll 이벤트보다 먼저 토큰이 와서 도로 끌려가지 않게
  if (e.deltaY < 0) followBottom.value = false
}
function onMediaLoad() {
  // 이미지가 늦게 떠서 내용이 자라면 맨 아래를 지킨다
  if (followBottom.value) scrollToBottom(false)
}
function scrollToBottom(smooth: boolean) {
  const el = scrollRef.value
  if (!el) return
  if (smooth) _settleUntil = performance.now() + 700
  el.scrollTo({ top: el.scrollHeight, behavior: smooth ? 'smooth' : 'auto' })
  followBottom.value = true
}
watch(activeId, () => { followBottom.value = true; nextTick(() => scrollToBottom(false)) })
// 스크롤 영역이 줄어들면(컴포저가 자람 · 설정 패널 열림 · 창 세로 축소) 맨 아래를 보고 있었으면 계속 맨 아래.
// 위에서 줄어들 땐 scrollTop 이 그대로라 scroll 이벤트도 안 오므로, 스크롤 영역 자체를 관찰해야 한다.
let _bottomObserver: ResizeObserver | null = null
onMounted(() => {
  if (typeof ResizeObserver === 'undefined' || !bottomRef.value) return
  _bottomObserver = new ResizeObserver(() => { if (followBottom.value) scrollToBottom(false) })
  _bottomObserver.observe(bottomRef.value)
  if (scrollRef.value) _bottomObserver.observe(scrollRef.value)
})
onUnmounted(() => { _bottomObserver?.disconnect(); _bottomObserver = null })

// ── 모델 · 설정 ──
function saveModel() {
  localStorage.setItem(provider.value === 'lmstudio' ? 'chatLmStudioModel' : 'ollamaModel', model.value)
  if (provider.value === 'ollama') requestAction('save_ui_prefs', { ollamaModel: model.value, ollamaUrl: url.value })
  savePreferences()
}
function saveSystemPrompt() {
  localStorage.setItem('chatSystemPrompt', systemPrompt.value)
  if (!CHAT_SYSTEM_PRESETS.some(preset => preset.prompt === systemPrompt.value)) {
    personalSystemPrompt.value = systemPrompt.value
    localStorage.setItem('chatPersonalSystemPrompt', systemPrompt.value)
  }
  savePreferences()
}
function applySystemPreset() {
  const selected = selectSystemPreset(systemPreset.value, systemPrompt.value, personalSystemPrompt.value)
  systemPrompt.value = selected.prompt
  personalSystemPrompt.value = selected.personal
  if (selected.personal !== null) localStorage.setItem('chatPersonalSystemPrompt', selected.personal)
  localStorage.setItem('chatSystemPrompt', selected.prompt)
  savePreferences()
}
function applyCustomSystemPreset(value: string | AiAssistInstructions) {
  if (typeof value !== 'string') return
  systemPrompt.value = value; systemPreset.value = ''; saveSystemPrompt()
}
function checkSchema() {
  if (!structuredEnabled.value || !schemaError.value) return true
  showSystem.value = true
  requestAction('show_toast', { type: 'error', msg: schemaError.value })
  return false
}
function saveStructured() {
  if (structuredEnabled.value) generationRequest.value.mode = 'chat'
  savePreferences()
}
function useSchemaExample() { schemaText.value = PROMPT_JSON_SCHEMA; schemaExamplePending.value = false; savePreferences() }
function savePreferences() {
  preferencesEdited = true
  localStorage.setItem('chatProvider', provider.value)
  localStorage.setItem('chatStructuredEnabled', structuredEnabled.value ? '1' : '0')
  localStorage.setItem('chatJsonSchema', schemaText.value)
  requestAction('save_ui_prefs', { chatSettingsV2: {
    provider: provider.value, lmStudioUrl: localStorage.getItem('chatLmStudioUrl') || 'http://localhost:1234',
    lmStudioModel: localStorage.getItem('chatLmStudioModel') || '',
    structuredEnabled: structuredEnabled.value, schemaText: schemaText.value,
    systemPrompt: systemPrompt.value, personalSystemPrompt: personalSystemPrompt.value,
    options: chatOptions.value,
  } })
}
async function restorePreferences() {
  const backend = await getBackend()
  if (disposed || preferencesEdited || !backend?.getUiPrefs) return
  backend.getUiPrefs((raw: string) => {
    if (disposed || preferencesEdited) return
    try {
      const prefs = JSON.parse(raw), saved = prefs.chatSettingsV2
      if (!saved || typeof saved !== 'object') return
      if (typeof saved.lmStudioUrl === 'string') localStorage.setItem('chatLmStudioUrl', saved.lmStudioUrl)
      if (typeof saved.lmStudioModel === 'string') localStorage.setItem('chatLmStudioModel', saved.lmStudioModel)
      provider.value = saved.provider === 'lmstudio' ? 'lmstudio' : 'ollama'
      localStorage.setItem('chatProvider', provider.value)
      model.value = provider.value === 'lmstudio' ? saved.lmStudioModel || '' : prefs.ollamaModel || model.value
      url.value = provider.value === 'lmstudio' ? saved.lmStudioUrl || 'http://localhost:1234' : prefs.ollamaUrl || url.value
      if (provider.value === 'ollama') {
        localStorage.setItem('ollamaUrl', url.value)
        localStorage.setItem('ollamaModel', model.value)
      }
      if (typeof saved.systemPrompt === 'string') systemPrompt.value = saved.systemPrompt
      if (typeof saved.personalSystemPrompt === 'string') personalSystemPrompt.value = saved.personalSystemPrompt
      if (typeof saved.schemaText === 'string') schemaText.value = saved.schemaText
      if (saved.options && typeof saved.options === 'object') {
        localStorage.setItem('chatOptions.v1', JSON.stringify(saved.options))
        chatOptions.value = loadChatOptions()
      }
      structuredEnabled.value = saved.structuredEnabled === true
      if (structuredEnabled.value) generationRequest.value.mode = 'chat'
      requestModels()
    } catch { /* do not replace local edits on invalid saved settings */ }
  })
}
function changeProvider() {
  models.value = []; modelsError.value = ''
  model.value = localStorage.getItem(provider.value === 'lmstudio' ? 'chatLmStudioModel' : 'ollamaModel') || ''
  url.value = localStorage.getItem(provider.value === 'lmstudio' ? 'chatLmStudioUrl' : 'ollamaUrl') || (provider.value === 'lmstudio' ? 'http://localhost:1234' : 'http://localhost:11434')
  savePreferences(); requestModels()
}
function saveConnection() {
  localStorage.setItem(provider.value === 'lmstudio' ? 'chatLmStudioUrl' : 'ollamaUrl', url.value)
  models.value = []; saveModel(); requestModels()
}
function requestModelInfo() {
  modelInfo.value = null
  modelInfoError.value = ''
  if (modelInfoTimer) clearTimeout(modelInfoTimer)
  modelInfoRequestId = uid()
  modelInfoLoading.value = !!model.value
  if (!model.value) return
  const id = modelInfoRequestId
  requestAction('chat_model_info', { id, url: url.value, model: model.value, provider: provider.value })
  modelInfoTimer = setTimeout(() => {
    if (modelInfoRequestId === id && modelInfoLoading.value) {
      modelInfoLoading.value = false
      modelInfoError.value = '모델 정보를 받지 못했습니다. 연결 후 다시 확인하세요'
    }
  }, 15000)
}
function onModelInfo(raw: string) {
  try {
    const event = JSON.parse(raw)
    if (event.id !== modelInfoRequestId || event.model !== model.value) return
    if (modelInfoTimer) clearTimeout(modelInfoTimer)
    modelInfoLoading.value = false
    if (event.ok && event.info) {
      modelInfo.value = event.info
      modelInfoError.value = ''
    }
    else modelInfoError.value = event.error || '모델 정보를 확인할 수 없습니다'
  } catch { /* stale/malformed metadata never changes the selected model */ }
}
watch([model, url, provider], requestModelInfo)
async function requestModels() {
  modelsRequestId = uid()
  const id = modelsRequestId
  clearTimeout(modelsTimer)
  modelsLoading.value = true; modelsError.value = ''
  modelsTimer = setTimeout(() => {
    if (id !== modelsRequestId) return
    modelsLoading.value = false; modelsError.value = '모델 목록 응답이 없습니다. 서버 실행과 주소를 확인하세요.'
  }, 12000)
  if (provider.value === 'lmstudio') {
    requestAction('chat_models', { id, url: url.value })
    return
  }
  try {
    const bk: any = await getBackend()
    if (disposed || id !== modelsRequestId || provider.value !== 'ollama') return
    url.value = localStorage.getItem('ollamaUrl') || url.value
    if (bk?.requestOllamaModels) bk.requestOllamaModels(url.value)
  } catch {}
}
function onModels(json: string) {
  if (provider.value !== 'ollama') return
  try {
    const p = JSON.parse(json)
    const list = Array.isArray(p) ? p : p.models
    if (!Array.isArray(list)) return
    clearTimeout(modelsTimer); modelsLoading.value = false
    models.value = list
    if (list.length && !list.includes(model.value)) {
      const base = (s: string) => (s || '').split(':')[0].toLowerCase()
      model.value = list.find((m: string) => base(m) === base(model.value)) || list[0]
    }
  } catch {}
}
function onChatModels(raw: string) {
  try {
    const event = JSON.parse(raw)
    if (provider.value !== 'lmstudio' || event.id !== modelsRequestId) return
    clearTimeout(modelsTimer); modelsLoading.value = false
    if (!event.ok) { models.value = []; modelsError.value = event.error || '모델 목록을 받지 못했습니다'; return }
    if (!Array.isArray(event.models)) return
    models.value = event.models.filter((item: unknown) => typeof item === 'string')
    if (!models.value.includes(model.value)) {
      model.value = models.value[0] || ''
      localStorage.setItem('chatLmStudioModel', model.value)
    }
    if (!models.value.length) modelsError.value = '서버에 모델이 없습니다. LM Studio에서 모델을 준비하세요.'
  } catch { /* obsolete/malformed model lists do not alter the selection */ }
}
async function copyText(text: string) {
  const ok = await copyTextToClipboard(text)
  requestAction('show_toast', { type: ok ? 'success' : 'error',
    msg: ok ? '복사됨' : '복사하지 못했습니다. 내용을 선택하고 Ctrl+C를 눌러 주세요.' })
}

const unsubs: Array<() => void> = []
onMounted(() => {
  unsubs.push(onBackendEvent('chatThreads', (json: string) => {
    try {
      const list = JSON.parse(json)
      if (Array.isArray(list)) threads.value = list
    } catch {}
    // GemmaStudio 처럼 열 때마다 빈 새 대화에서 시작한다 — 기존 기록은 목록에 남는다
    if (!activeId.value) newThread()
  }))
  unsubs.push(onBackendEvent('chatToken', onToken))
  unsubs.push(onBackendEvent('chatDone', onDone))
  unsubs.push(onBackendEvent('chatGenerationEvent', onGeneration))
  unsubs.push(onBackendEvent('ollamaModelsReady', onModels))
  unsubs.push(onBackendEvent('chatModelInfo', onModelInfo))
  unsubs.push(onBackendEvent('chatModelsReady', onChatModels))
  window.addEventListener('keydown', onGlobalKey)
  requestAction('chat_load')
  requestModels()
  restorePreferences().catch(() => {})
  requestModelInfo()
  // 백엔드가 목록을 안 돌려줘도(웹 모드·개발 서버) 빈 대화 하나는 있어야 입력이 된다
  setTimeout(() => { if (!threads.value.length) newThread() }, 1500)
})
onActivated(() => { requestModels(); focusComposer() })
onUnmounted(() => {
  disposed = true; clearTimeout(modelsTimer)
  unsubs.forEach((u) => { try { u() } catch {} })
  window.removeEventListener('keydown', onGlobalKey)
  if (saveTimer) { clearTimeout(saveTimer); saveTimer = null }
  if (modelInfoTimer) { clearTimeout(modelInfoTimer); modelInfoTimer = null }
})
</script>

<style scoped>
.chat-view { height: 100%; display: flex; position: relative; background: var(--bg-primary); overflow: hidden; }
.cm-provider-row { display: flex; flex-wrap: wrap; align-items: end; gap: 8px; }
.cm-provider-row label { display: flex; flex: 1 1 180px; min-width: 0; flex-direction: column; gap: 4px; }
.cm-provider-row input, .cm-provider-row select { min-width: 0; width: 100%; box-sizing: border-box; }
.cm-provider-row input, .cm-provider-row select, .cm-provider-row button, .cm-structured button { padding: 7px 9px; color: var(--text-primary); background: var(--bg-input); border: 1px solid var(--border); border-radius: var(--radius-base); font: inherit; font-size: 12px; }
.cm-structured, .cm-assist-link { border-top: 1px solid var(--border); padding: 10px 0; min-width: 0; color: var(--text-primary); }
.cm-structured summary, .cm-assist-link summary { cursor: pointer; font-size: 12px; line-height: 1.7; }
.cm-schema-toggle { display: flex; align-items: center; gap: 8px; margin: 12px 0; }
.cm-system #chat-json-schema, .msg-json { font-family: ui-monospace, Consolas, monospace; tab-size: 2; }
.cm-schema-error { color: var(--state-alert-fg); font-size: 12px; overflow-wrap: anywhere; }
.cm-system :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.msg-json { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; padding: 12px; border: 1px solid var(--border); border-radius: 8px; background: var(--bg-input); }
.cm-tool.cm-mobile-threads, .cm-thread-backdrop { display: none; }

/* ── 대화 목록 ── */
.chat-threads { width: 268px; flex-shrink: 0; display: flex; flex-direction: column; border-right: 1px solid var(--border); background: var(--bg-secondary); }
.ct-head { padding: 12px 12px 8px; display: flex; flex-direction: column; gap: 8px; }
.ct-new { height: 34px; display: flex; align-items: center; justify-content: center; gap: 6px; border-radius: var(--radius-base); border: 1px solid var(--border); background: var(--bg-button); color: var(--text-primary); font-size: var(--fs-body); font-weight: var(--fw-medium); cursor: pointer; }
.ct-new:hover { border-color: var(--accent); color: var(--accent); }
.ct-search { display: flex; align-items: center; gap: 6px; height: 30px; padding: 0 10px; border-radius: var(--radius-base); background: var(--bg-input); border: 1px solid var(--border); color: var(--text-muted); }
.ct-search input { flex: 1; min-width: 0; background: transparent; border: 0; outline: 0; color: var(--text-primary); font-size: var(--fs-meta); }
.ct-list { flex: 1; min-height: 0; overflow-y: auto; padding: 0 8px 8px; display: flex; flex-direction: column; gap: 2px; }
.ct-empty { padding: 18px 8px; color: var(--text-muted); font-size: var(--fs-meta); text-align: center; }
.ct-row { position: relative; flex-shrink: 0; }
.ct-item { position: relative; display: flex; align-items: center; gap: 8px; height: 32px; width: 100%; padding: 0 30px 0 8px; border: 0; border-radius: var(--radius-base); background: transparent; color: var(--text-secondary); font-size: var(--fs-body); text-align: left; cursor: pointer; }
.ct-item:hover { background: var(--bg-button); color: var(--text-primary); }
.ct-item.on { background: var(--accent-dim); color: var(--text-primary); }
.ct-title { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ct-del { position: absolute; right: 4px; top: 6px; display: flex; opacity: 0; width: 20px; height: 20px; align-items: center; justify-content: center; border: 0; background: transparent; border-radius: 4px; color: var(--text-muted); cursor: pointer; }
.ct-row:hover .ct-del, .ct-row:focus-within .ct-del { opacity: 1; }
.ct-del:hover { background: rgba(248,113,113,.12); color: var(--state-alert-fg); }
.ct-foot { padding: 10px 12px 12px; border-top: 1px solid var(--border); display: flex; flex-direction: column; gap: 6px; }
.ct-foot-label { font-size: var(--fs-label); font-weight: var(--fw-medium); color: var(--text-muted); }
.ct-foot-none { font-size: var(--fs-label); color: var(--state-alert-fg); }

/* ── 대화 본문 ── */
.chat-main { flex: 1; min-width: 0; display: flex; flex-direction: column; position: relative; }
/* 오른쪽 여백은 알림 종 자리다 (style.css --notif-gutter) */
.cm-head { height: 52px; flex-shrink: 0; display: flex; align-items: center; gap: 12px; padding: 0 var(--notif-gutter) 0 20px; border-bottom: 1px solid var(--border); }
.cm-title-wrap { flex: 1; min-width: 0; display: flex; flex-direction: column; }
.cm-title { margin: 0; font-size: 14px; font-weight: var(--fw-bold); color: var(--text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: text; }
.cm-rename { font: inherit; font-size: 14px; font-weight: var(--fw-bold); color: var(--text-primary); background: var(--bg-input); border: 1px solid var(--accent); border-radius: 4px; padding: 2px 6px; outline: 0; }
.cm-sub { font-size: var(--fs-label); color: var(--text-muted); }
.cm-tools { display: flex; align-items: center; gap: 6px; }
.cm-status { display: flex; align-items: center; gap: 6px; height: 24px; padding: 0 10px; border-radius: var(--radius-pill); background: var(--bg-button); color: var(--text-secondary); font-size: var(--fs-label); }
.cm-status .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--state-ok-fg); }
.cm-status.busy .dot { background: var(--accent); animation: chat-pulse 1s ease-in-out infinite; }
.cm-status.off .dot { background: var(--state-alert-fg); }
@keyframes chat-pulse { 50% { opacity: .35; } }
.cm-tool { width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; border: 1px solid var(--border); border-radius: var(--radius-base); background: var(--bg-button); color: var(--text-secondary); cursor: pointer; }
.cm-tool:hover { color: var(--text-primary); border-color: var(--text-muted); }
.cm-tool.on { color: var(--accent); border-color: var(--accent); }
.cm-tool:disabled { opacity: .35; cursor: default; }
.cm-system { padding: 10px 20px; border-bottom: 1px solid var(--border); background: var(--bg-secondary); display: flex; flex-direction: column; gap: 6px; max-height: 45%; min-height: 0; overflow-y: auto; flex-shrink: 1; }
.cm-system > * { flex-shrink: 0; }
.cm-system label { font-size: var(--fs-label); font-weight: var(--fw-medium); color: var(--text-muted); }
.cm-opts { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 14px; padding-top: 2px; }
.cm-opt { display: flex; flex-direction: column; gap: 4px; font-size: var(--fs-label); color: var(--text-muted); }
.cm-opt select { height: 28px; padding: 0 8px; border-radius: var(--radius-base); border: 1px solid var(--border); background: var(--bg-input); color: var(--text-primary); font: inherit; font-size: var(--fs-meta); outline: 0; }
.cm-opt-range input { width: 140px; accent-color: var(--accent); }
.cm-opt-hint { flex-basis: 100%; font-size: var(--fs-label); color: var(--text-muted); }
.cm-settings-hint { margin: 3px 0; color: var(--text-secondary); font-size: var(--fs-label); line-height: 1.6; }
.cm-preset-row { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; }
.cm-preset-row small { color: var(--text-muted); font-size: var(--fs-label); }
.cm-preset-row select, .cm-thinking-level select { max-width: 100%; height: 28px; padding: 0 8px; color: var(--text-primary); background: var(--bg-input); border: 1px solid var(--border); border-radius: var(--radius-base); font: inherit; }
.cm-preset-row button, .cm-model-info button { padding: 5px 9px; color: var(--text-primary); background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-base); font-size: var(--fs-label); cursor: pointer; }
.cm-model-info { display: flex; flex-direction: column; gap: 5px; padding-top: 8px; border-top: 1px solid var(--border); color: var(--text-secondary); font-size: var(--fs-label); }
.cm-model-info-title { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.cm-thinking-level { display: flex; align-items: center; gap: 8px; }
.cm-confirm { width: min(420px, calc(100vw - 32px)); margin: auto; max-height: calc(100dvh - 32px); overflow: auto; padding: 22px; box-sizing: border-box; color: var(--text-primary); background: var(--bg-secondary); border: 1px solid var(--border); border-radius: var(--radius-lg, 12px); box-shadow: 0 16px 60px rgba(0,0,0,.3); }
.cm-confirm::backdrop { background: rgba(0,0,0,.45); }
.cm-confirm h3 { margin: 0 0 12px; font-size: 17px; }
.cm-confirm p { margin: 0 0 20px; line-height: 1.6; overflow-wrap: anywhere; }
.cm-confirm form > div { display: flex; justify-content: flex-end; gap: 8px; }
.cm-confirm button { padding: 8px 16px; background: var(--bg-button); color: var(--text-primary); border: 1px solid var(--border); border-radius: var(--radius-base); font: inherit; cursor: pointer; }
.cm-confirm .danger { color: var(--state-alert-fg); }
.cm-confirm button:focus-visible, .ct-del:focus-visible, .cm-system select:focus-visible, .cm-system textarea:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
@media (hover: none) { .ct-del { opacity: 1; } }
.msg-meta { font-weight: normal; }
.msg-cut { font-size: var(--fs-meta); color: var(--state-alert-fg); }
.cm-system textarea { width: 100%; resize: vertical; padding: 8px 10px; border-radius: var(--radius-base); border: 1px solid var(--border); background: var(--bg-input); color: var(--text-primary); font: inherit; font-size: var(--fs-meta); line-height: 1.5; outline: 0; user-select: text; }

.cm-scroll { flex: 1; min-height: 0; overflow-y: auto; padding: 20px 20px 24px; }
.cm-col { width: min(790px, 100%); margin: 0 auto; display: flex; flex-direction: column; gap: 22px; }

.cm-hero { margin: 60px auto 0; max-width: 560px; text-align: center; display: flex; flex-direction: column; align-items: center; gap: 10px; }
.cm-hero-mark { width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; border-radius: 12px; background: var(--accent-dim); color: var(--accent); }
.cm-hero h3 { margin: 4px 0 0; font-size: 18px; font-weight: var(--fw-bold); color: var(--text-primary); }
.cm-hero p { margin: 0; color: var(--text-muted); font-size: var(--fs-body); line-height: 1.6; }
.cm-suggest { margin-top: 10px; display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; }
.cm-suggest button { padding: 8px 12px; border-radius: var(--radius-pill); border: 1px solid var(--border); background: var(--bg-card); color: var(--text-secondary); font-size: var(--fs-meta); cursor: pointer; }
.cm-suggest button:hover { border-color: var(--accent); color: var(--text-primary); }

.msg { display: grid; grid-template-columns: 30px minmax(0, 1fr); gap: 12px; }
.msg-avatar { width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; border-radius: 8px; font-size: 11px; font-weight: var(--fw-bold); background: var(--bg-button); color: var(--text-secondary); }
.msg.assistant .msg-avatar { background: var(--accent-fill); color: var(--on-accent); }
.msg-body { min-width: 0; display: flex; flex-direction: column; gap: 6px; position: relative; }
.msg-role { font-size: var(--fs-label); font-weight: var(--fw-medium); color: var(--text-muted); }
.msg-images { display: flex; flex-wrap: wrap; gap: 6px; }
.msg-images img { max-width: 220px; max-height: 220px; border-radius: 8px; border: 1px solid var(--border); object-fit: cover; }
.msg-content { color: var(--text-primary); font-size: var(--fs-body); line-height: 1.65; user-select: text; }
.msg-content.plain { white-space: pre-wrap; word-break: break-word; }
.msg-error { font-size: var(--fs-meta); color: var(--state-alert-fg); }
.msg-thinking { display: flex; gap: 4px; padding: 4px 0; }
.msg-thinking span { width: 6px; height: 6px; border-radius: 50%; background: var(--text-muted); animation: chat-dots 1.2s ease-in-out infinite; }
.msg-thinking span:nth-child(2) { animation-delay: .2s; } .msg-thinking span:nth-child(3) { animation-delay: .4s; }
@keyframes chat-dots { 0%, 80%, 100% { opacity: .25; transform: translateY(0); } 40% { opacity: 1; transform: translateY(-3px); } }
.msg-actions { display: flex; gap: 4px; opacity: 0; transition: opacity .15s; }
.msg:hover .msg-actions { opacity: 1; }
.msg-actions button { width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; border: 1px solid var(--border); border-radius: 4px; background: var(--bg-button); color: var(--text-muted); cursor: pointer; }
.msg-actions button:hover { color: var(--text-primary); border-color: var(--text-muted); }

/* 마크다운 — v-html 로 들어오는 태그들 (chatMarkdown.ts 가 만드는 것만) */
.md :deep(p) { margin: 0 0 8px; }
.md :deep(p:last-child) { margin-bottom: 0; }
.md :deep(h1), .md :deep(h2), .md :deep(h3) { margin: 10px 0 6px; font-weight: var(--fw-bold); color: var(--text-primary); }
.md :deep(h1) { font-size: 16px; } .md :deep(h2) { font-size: 15px; } .md :deep(h3) { font-size: 14px; }
.md :deep(ul), .md :deep(ol) { margin: 0 0 8px; padding-left: 22px; }
.md :deep(li) { margin: 2px 0; }
.md :deep(blockquote) { margin: 0 0 8px; padding: 4px 12px; border-left: 3px solid var(--accent); color: var(--text-secondary); }
.md :deep(hr) { border: 0; border-top: 1px solid var(--border); margin: 10px 0; }
.md :deep(code) { padding: 1px 5px; border-radius: 4px; background: var(--bg-input); font-family: Consolas, 'JetBrains Mono', monospace; font-size: 12px; }
.md :deep(pre) { position: relative; margin: 0 0 8px; padding: 10px 12px; border-radius: 8px; background: var(--bg-input); border: 1px solid var(--border); overflow-x: auto; }
.md :deep(pre code) { padding: 0; background: transparent; white-space: pre; }
.md :deep(pre[data-lang]:not([data-lang=""]))::before { content: attr(data-lang); position: absolute; top: 6px; right: 10px; font-size: var(--fs-label); color: var(--text-muted); }
.md :deep(a) { color: var(--accent); text-decoration: underline; }
.md :deep(strong) { font-weight: var(--fw-bold); }

.cm-jump { position: absolute; right: 24px; top: -40px; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; border-radius: 50%; border: 1px solid var(--border); background: var(--bg-card); color: var(--text-secondary); cursor: pointer; box-shadow: 0 4px 14px rgba(0,0,0,.3); }
.cm-jump:hover { color: var(--accent); border-color: var(--accent); }

/* ── 컴포저 — 스크롤 영역 아래, 흐름 안의 카드 (메시지를 가리지 않는다) ── */
.cm-bottom { position: relative; flex-shrink: 0; display: flex; justify-content: center; padding: 6px 20px 16px; }
.cm-composer { position: relative; width: min(790px, 100%); display: flex; flex-direction: column; gap: 6px; padding: 10px 12px 8px; border-radius: 14px; background: var(--bg-card); border: 1px solid var(--border-strong); box-shadow: 0 10px 40px rgba(0,0,0,.35); }
.cm-composer.drag { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-dim); }
.cmp-attach { display: flex; flex-wrap: wrap; gap: 6px; }
.cmp-thumb { position: relative; width: 56px; height: 56px; }
.cmp-thumb img { width: 100%; height: 100%; object-fit: cover; border-radius: 6px; border: 1px solid var(--border); }
.cmp-thumb button { position: absolute; top: -6px; right: -6px; width: 18px; height: 18px; display: flex; align-items: center; justify-content: center; border-radius: 50%; border: 1px solid var(--border-strong); background: var(--bg-card); color: var(--text-secondary); cursor: pointer; }
.cmp-input { width: 100%; min-height: 40px; max-height: 180px; resize: none; padding: 8px 4px; background: transparent; border: 0; outline: 0; color: var(--text-primary); font: inherit; font-size: var(--fs-body); line-height: 1.5; user-select: text; }
.cmp-bar { display: flex; align-items: center; gap: 8px; }
.cmp-hint { display: flex; align-items: center; gap: 5px; color: var(--text-muted); font-size: var(--fs-label); }
.cmp-spacer { flex: 1; }
.cmp-think { display: flex; align-items: center; gap: 5px; height: 28px; padding: 0 10px; border-radius: var(--radius-pill); border: 1px solid var(--border); background: transparent; color: var(--text-muted); font-size: var(--fs-label); cursor: pointer; }
.cmp-think:hover { color: var(--text-primary); border-color: var(--text-muted); }
.cmp-think.on { color: var(--accent); border-color: var(--accent); background: var(--accent-dim); }
.msg-think { border: 1px solid var(--border); border-radius: 8px; background: var(--bg-secondary); font-size: var(--fs-meta); }
.msg-think summary { display: flex; align-items: center; gap: 6px; padding: 6px 10px; color: var(--text-muted); cursor: pointer; user-select: none; list-style: none; }
.msg-think summary::-webkit-details-marker { display: none; }
.msg-think[open] summary { border-bottom: 1px solid var(--border); }
.msg-think pre { margin: 0; padding: 8px 10px; max-height: 220px; overflow: auto; white-space: pre-wrap; word-break: break-word; color: var(--text-secondary); font: inherit; font-size: var(--fs-meta); line-height: 1.5; user-select: text; }
.cmp-send { width: 34px; height: 34px; display: flex; align-items: center; justify-content: center; border-radius: 50%; border: 0; background: var(--accent-fill); color: var(--on-accent); cursor: pointer; }
.cmp-send:disabled { opacity: .3; cursor: default; }
.cmp-send.stop { background: var(--state-alert); color: var(--state-alert-fg); }
.cmp-generation-options { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; }
.cmp-generation-options label { display: flex; align-items: center; gap: 6px; color: var(--text-muted); font-size: var(--fs-label); white-space: nowrap; }
.cmp-generation-options select, .cmp-generation-options input { max-width: 230px; min-height: 27px; border: 1px solid var(--border); border-radius: 5px; background: var(--bg-input); color: var(--text-primary); font: inherit; }
.cmp-generation-options input { width: 64px; }
.cmp-generation-options small { flex-basis: 100%; color: var(--text-muted); font-size: var(--fs-label); }
.msg-generation { display: flex; flex-direction: column; gap: 6px; padding: 12px; border: 1px solid var(--border); border-radius: 8px; background: var(--bg-secondary); font-size: var(--fs-meta); color: var(--text-secondary); }
.msg-generation progress { width: 100%; height: 6px; accent-color: var(--accent); }
.msg-artifacts { display: flex; flex-wrap: wrap; gap: 10px; }
.msg-artifacts figure { margin: 0; max-width: 100%; min-width: 0; }
.msg-artifacts img, .msg-artifacts video { display: block; max-width: 100%; max-height: 560px; object-fit: contain; border-radius: 8px; }
.msg-artifacts figcaption { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 6px 0; color: var(--text-muted); font-size: var(--fs-label); }
.msg-artifacts figcaption span { max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.msg-artifacts button { background: var(--bg-button); border: 1px solid var(--border); border-radius: 4px; padding: 3px 7px; color: var(--text-secondary); cursor: pointer; }

@media (max-width: 1100px) {
  .chat-threads { width: 220px; }
}
@media (max-width: 760px) {
  .chat-threads { display: none; }
  .chat-threads.mobile-open { display: flex; position: absolute; inset: 0 auto 0 0; z-index: 5; width: min(280px, 85%); }
  .cm-thread-backdrop { display: block; position: absolute; inset: 0; z-index: 4; border: 0; background: rgb(0 0 0 / .35); }
  .cm-tool.cm-mobile-threads { display: flex; width: 40px; flex-shrink: 0; }
  .cm-head { padding: 0 10px; gap: 8px; }
  .cm-status, .cmp-hint { display: none; }
  .cm-scroll { padding: 14px 10px; }
  .cm-bottom { padding: 6px 8px 10px; }
  .cm-composer { padding: 8px; }
  .cmp-generation-options { gap: 5px 8px; }
  .cmp-generation-options select { max-width: 185px; }
  .msg { grid-template-columns: 24px minmax(0, 1fr); gap: 8px; }
  .msg-avatar { width: 24px; height: 24px; }
}
</style>
