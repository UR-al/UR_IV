<template>
  <div ref="rootRef" class="prompt-panel">
    <!-- data-prompt-undo: 이 요소 안의 입력칸에서 Ctrl+Z/Y 가 패널 Undo/Redo 가 된다(패널 버튼·
         포커스 없음도 포함 — utils/promptUndoKeys.ts). 표시한 칸은 PROMPT_UNDO_KEYS 로 추적해야
         한다(PromptPanel.undo.test.ts). 그 밖의 입력칸은 브라우저 기본 실행 취소 그대로. -->
    <!-- 1. FINAL OUTPUT PROMPT -->
    <!-- id 는 세로 레일 서랍의 스크롤 대상이다 — `utils/navSections.ts` 와 글자 그대로
         같아야 하고, tests/test_nav_rail_contract.py 가 그걸 지킨다. -->
    <div id="sec-prompt" class="glass-card highlight">
      <div class="card-header">
        최종 프롬프트
        <span class="token-info">토큰 {{ tokenCount }}</span>
      </div>
      <TagBlockField v-if="tagBlockMode" data-prompt-undo :model-value="widgets.total_prompt_display" :color-fn="blockColorClass" placeholder=""
        @update:model-value="onTotalBlockChange" @open-wildcard="forwardWildcard" />
      <!-- 블록 모드 최종 프롬프트는 onTotalBlockChange 로 추적 칸에만 되쓰므로 스냅숏 범위다.
           텍스트 모드 최종 프롬프트는 스냅숏 대상(PROMPT_UNDO_KEYS)이 아니라 네이티브 실행 취소를 둔다 -->
      <textarea v-else ref="totalPromptRef" v-model="widgets.total_prompt_display"
        class="total-prompt auto-grow" placeholder="최종 프롬프트" @input="autoGrow($event.target)"></textarea>
      <div class="prompt-actions">
        <button class="optimize-btn" @click="optimizePrompt" title="메인 태그의 중복(다른 칸과 겹친 태그 포함)을 지우고 순서를 정리합니다"><Icon name="wand" /> 최적화</button>
        <button class="optimize-btn" @click="toggleSeparate" title="표정/배경/포즈/사물/메타 태그를 분류해서 제거하거나 추출"><Icon name="tag" /> 분류</button>
        <span class="opt-result" v-if="optResult">{{ optResult }}</span>
      </div>
      <!-- 🧹 OPTIMIZE before/after 미리보기 → [적용] 클릭 시 반영 -->
      <div class="opt-preview" v-if="optPreview">
        <div class="opt-prev-head">
          <span><Icon name="wand" /> 메인 태그 최적화 미리보기</span>
          <span class="opt-prev-stat">중복 {{ optPreview.removed }}개 제거<template v-if="optPreview.fromContext"> (다른 칸과 겹침 {{ optPreview.fromContext }})</template> · {{ optPreview.tagCount }}개 태그</span>
        </div>
        <div class="opt-prev-cols">
          <div class="opt-prev-col"><label>이전</label><div class="opt-prev-text before">{{ optPreview.before }}</div></div>
          <div class="opt-prev-col"><label>이후</label><div class="opt-prev-text after">{{ optPreview.after }}</div></div>
        </div>
        <div class="opt-prev-conf" v-if="optPreview.conflicts.length"><Icon name="alert" /> 충돌: <span v-for="c in optPreview.conflicts" :key="c.group">{{ c.group }}({{ c.tags.join('/') }}) </span></div>
        <div class="opt-prev-btns">
          <button class="opt-apply" @click="applyOptimize"><Icon name="check" /> 적용</button>
          <button class="opt-cancel" @click="cancelOptimize">취소</button>
        </div>
      </div>

      <!-- ⑤ 카테고리 분리 토글 -->
      <div class="separate-panel" v-if="showSeparate">
        <div class="sep-chips">
          <button v-for="c in SEP_CATS" :key="c.key" class="sep-chip" :class="{ on: sepCats[c.key] }"
            @click="sepCats[c.key] = !sepCats[c.key]">
            {{ c.label }}<span class="sep-n" v-if="sepCounts[c.key]">{{ sepCounts[c.key] }}</span>
          </button>
        </div>
        <div class="sep-actions">
          <button class="sep-act remove" @click="applySeparate('remove')"><Icon name="trash" /> 선택 제거</button>
          <button class="sep-act extract" @click="applySeparate('extract')"><Icon name="scissors" /> 선택만 남기기</button>
          <span class="sep-hint" v-if="sepResult">{{ sepResult }}</span>
        </div>
      </div>
      <div class="conflicts" v-if="promptConflicts.length > 0">
        <div v-for="c in promptConflicts" :key="c.group" class="conflict-item"><Icon name="alert" /> {{ c.group }}: {{ c.tags.join(', ') }}</div>
      </div>
      <details class="neg-section">
        <summary class="danger-label neg-toggle">
          네거티브 <Icon name="chevron-down" size="12" />
          <button class="ai-btn neg-ai" @click.prevent.stop="runSmartNegative()" :disabled="ollamaLoading" title="AI 네거티브 자동 생성"><Icon name="cpu" /></button>
        </summary>
        <TagBlockField v-if="tagBlockMode" data-prompt-undo v-model="widgets.neg_prompt_text" :color-fn="() => 'neg'" class="neg" placeholder="네거티브 추가..." />
        <textarea v-else ref="negRef" data-prompt-undo v-model="widgets.neg_prompt_text" class="neg-prompt auto-grow" placeholder="Negative prompt..." @input="autoGrow($event.target)"></textarea>
      </details>
    </div>

    <!-- 2. CHARACTER & MODEL -->
    <div id="sec-character" class="glass-card">
      <div class="card-header">캐릭터 · 모델</div>
      <div class="input-group">
        <label>인물 수</label>
        <TagBlockField v-if="tagBlockMode" data-prompt-undo v-model="widgets.char_count_input" :color-fn="() => 'bc-count'" placeholder="인물수..." />
        <input v-else type="text" data-prompt-undo v-model="widgets.char_count_input" placeholder="e.g. 1girl, 2girls..." />
      </div>
      <div class="input-group autocomplete-wrap">
        <div class="row label-row">
          <label>캐릭터 <span v-if="sectionTokens.character" class="tk-badge" :class="tokenBadgeClass(sectionTokens.character)">{{ sectionTokens.character }}t</span></label>
          <button class="small-btn" @click="openCharPresetModal(); loadCharTags()">프리셋</button>
        </div>
        <TagBlockField v-if="tagBlockMode" data-prompt-undo v-model="widgets.character_input" :color-fn="() => 'bc-count'" placeholder="캐릭터..." @open-wildcard="forwardWildcard" />
        <input v-else type="text" data-prompt-undo v-model="widgets.character_input" placeholder="e.g. hatsune miku"
          @input="onFieldInput($event, 'character_input')" @keydown="onFieldKey($event, 'character_input')"
          @click="closeFieldAc" @blur="closeFieldAc(); loadCharTags()" />
        <div class="char-insight" v-if="charInsight.tags.length > 0">
          <div class="insight-header">
            <span class="insight-label"><Icon name="book" /> 공식 태그</span>
            <button class="insight-apply" @click="applyOfficialTags">전체 적용</button>
          </div>
          <div class="insight-tags">
            <button v-for="tag in charInsight.tags" :key="tag" class="char-tag-chip" @click="insertCharTag(tag)">{{ tag.replace(/_/g, ' ') }}</button>
          </div>
        </div>
        <div class="ac-popup" v-if="!tagBlockMode && ac.isOpenFor('character_input')">
          <div v-for="(item, i) in acItems" :key="item.tag" class="ac-item" :class="{ selected: acIdx === i }" @mousedown.prevent="acceptFieldSuggestion(item.tag, 'character_input')">{{ item.tag }}<span v-if="item.ko" class="ac-ko">{{ item.ko }}</span></div>
        </div>
      </div>
      <div class="input-group autocomplete-wrap">
        <label>작품 <span v-if="sectionTokens.copyright" class="tk-badge" :class="tokenBadgeClass(sectionTokens.copyright)">{{ sectionTokens.copyright }}t</span></label>
        <TagBlockField v-if="tagBlockMode" data-prompt-undo v-model="widgets.copyright_input" :color-fn="() => ''" placeholder="작품..." />
        <input v-else type="text" data-prompt-undo v-model="widgets.copyright_input" placeholder="Copyright / Series..."
          @input="onFieldInput($event, 'copyright_input')" @keydown="onFieldKey($event, 'copyright_input')"
          @click="closeFieldAc" @blur="closeFieldAc" />
        <div class="ac-popup" v-if="!tagBlockMode && ac.isOpenFor('copyright_input')">
          <div v-for="(item, i) in acItems" :key="item.tag" class="ac-item" :class="{ selected: acIdx === i }" @mousedown.prevent="acceptFieldSuggestion(item.tag, 'copyright_input')">{{ item.tag }}<span v-if="item.ko" class="ac-ko">{{ item.ko }}</span></div>
        </div>
      </div>
      <div class="input-group">
        <div class="row label-row">
          <label>작가 <span v-if="sectionTokens.artist" class="tk-badge" :class="tokenBadgeClass(sectionTokens.artist)">{{ sectionTokens.artist }}t</span></label>
          <button class="lock-btn" :class="{ locked: artistLocked }" @click="toggleArtistLock"><Icon :name="artistLocked ? 'lock' : 'unlock'" /></button>
        </div>
        <TagBlockField v-if="tagBlockMode" data-prompt-undo v-model="widgets.artist_input" :color-fn="() => ''" placeholder="작가..." />
        <textarea v-else ref="artistRef" data-prompt-undo v-model="widgets.artist_input" class="auto-grow" placeholder="Artist tags..." @input="autoGrow($event.target)"></textarea>
      </div>
      <div v-if="showGenerationFamily" class="input-group generation-family">
        <label>생성 엔진</label>
        <CustomSelect v-model="widgets.generation_family_combo" :options="generationFamilyItems" />
        <span class="engine-hint">T2I와 I2I에 공통 적용됩니다.</span>
      </div>
      <div v-if="!isKrea2Generation" class="input-group">
        <label>Checkpoint</label>
        <CustomSelect v-model="widgets.model_combo" :options="modelItems"
          :option-groups="modelOptionGroups" placeholder="모델 선택..." />
      </div>
      <div v-if="!isKrea2Generation" class="input-group">
        <label>VAE <span class="hint">(ANIMA 등 외부 VAE)</span></label>
        <CustomSelect v-model="widgets.vae_main_combo" :options="vaeItems" placeholder="(체크포인트 기본 사용)" />
      </div>
      <div v-if="!isKrea2Generation" class="input-group">
        <label>텍스트 인코더 <span class="hint">(드롭다운에서 추가 · 칩 클릭으로 제거)</span></label>
        <CustomSelect :modelValue="''" @update:modelValue="addTeFile"
          :options="teUnselectedItems"
          :placeholder="teItems.length === 0 ? '(text_encoder 폴더 비어있음)' : '+ TE 파일 추가...'" />
        <div v-if="teSelectedList.length > 0" class="te-chips">
          <button v-for="f in teSelectedList" :key="f" type="button"
            class="te-chip active" @click="removeTeFile(f)">
            {{ f }} <span class="te-chip-x">×</span>
          </button>
        </div>
      </div>
      <div v-else class="krea-engine-note">
        <strong>Krea 2 Turbo</strong>
        <span>전용 UNET · Qwen3-VL · VAE 워크플로를 실행하며 ComfyUI 백엔드가 필요합니다.</span>
        <span>T2I 전환 시 8 steps / CFG 1을 설정합니다. Sampler는 Euler 권장, Scheduler는 Simple 고정입니다.</span>
        <span v-if="isI2IRoute">I2I Identity Edit에는 Identity Edit · TextFusion LoRA와 Krea2 Edit custom node가 추가로 필요합니다.</span>
        <span>일반 Checkpoint · VAE · TE · LoRA Stack · Forge 확장과 Negative는 이 모드에서 적용되지 않습니다.</span>
      </div>
    </div>

    <!-- 3. PROMPT BLOCKS -->
    <details class="glass-card" open>
      <summary class="card-header">
        <span>프롬프트 블록</span>
        <span class="undo-btns" @click.stop>
          <button class="undo-btn" :disabled="undoStack.length < 2" @click="performUndo" title="실행 취소 (Ctrl+Z)"><Icon name="undo" /></button>
          <button class="undo-btn" :disabled="redoStack.length === 0" @click="performRedo" title="다시 실행 (Ctrl+Y)"><Icon name="redo" /></button>
        </span>
      </summary>
      <div class="input-group autocomplete-wrap">
        <div class="row label-row">
          <label>메인 태그 <span v-if="sectionTokens.main" class="tk-badge" :class="tokenBadgeClass(sectionTokens.main)">{{ sectionTokens.main }}t</span></label>
          <div class="ai-btns">
            <!-- 태그형 출력 AI -->
            <div class="ai-menu-wrap">
              <button class="ai-btn ai-menu-btn" @click.stop="toggleAiMenu('tag')" :disabled="ollamaLoading" title="태그를 만드는 AI"><Icon name="sparkles" /> 태그 AI <Icon name="chevron-down" size="12" /></button>
              <div class="ai-menu" v-if="aiMenu === 'tag'">
                <button @click="runAi('expand')">태그 확장 <em>연관 태그 추가</em></button>
                <button @click="runAi('suggest')">유사 태그 <em>비슷한 태그 추천</em></button>
                <button @click="openNlInput('nl2tags')">자연어 → 태그 <em>문장을 태그로</em></button>
              </div>
            </div>
            <!-- 자연어형 출력 AI -->
            <div class="ai-menu-wrap">
              <button class="ai-btn ai-menu-btn" @click.stop="toggleAiMenu('nl')" :disabled="ollamaLoading" title="자연어/문장을 만드는 AI"><Icon name="message" /> 자연어 AI <Icon name="chevron-down" size="12" /></button>
              <div class="ai-menu" v-if="aiMenu === 'nl'">
                <button @click="runAi('nl_caption')">자연어 캡션 <em>태그 → 영어 문장</em></button>
                <button @click="openNlInput('nl_scene')">영문 장면묘사 <em>키워드 → 장면</em></button>
                <button @click="runAi('translate')">한↔영 번역</button>
                <button @click="runAi('creative')">창의 생성 <em>캐릭터 기반 창작</em></button>
              </div>
            </div>
          </div>
        </div>
        <div v-if="aiMenu" class="ai-menu-backdrop" @click="aiMenu = ''"></div>
        <div class="nl-input-row" v-if="showNlInput">
          <input v-model="nlPrompt" ref="nlInputRef" :placeholder="pendingNlMode === 'nl2tags' ? '자연어 설명을 입력...' : '키워드를 입력...'" @keydown.enter="onNlEnter" class="nl-input" />
          <button class="ai-btn go" @click="runPendingNl()" :disabled="ollamaLoading">실행</button>
          <button class="ai-btn" @click="showNlInput = false" title="닫기"><Icon name="close" /></button>
        </div>
        <div class="ai-loading" v-if="ollamaLoading"><Icon name="cpu" /> AI 처리 중...</div>
        <div class="nl-result" v-if="nlResult">
          <textarea :value="nlResult" readonly class="nl-result-text" rows="4"></textarea>
          <div class="nl-result-btns">
            <button class="ai-btn go" @click="copyNlResult" title="복사">복사</button>
            <button class="ai-btn go" @click="useNlAsMain" title="메인 프롬프트에 넣기">메인에 넣기</button>
            <button v-if="nlRes" class="ai-btn go" @click="applyNlRes" :title="`추천 해상도 ${nlRes.w}×${nlRes.h} 적용`"><Icon name="crop" /> {{ nlRes.w }}×{{ nlRes.h }}</button>
            <button class="ai-btn" @click="nlResult = ''; nlRes = null" title="닫기"><Icon name="close" /></button>
          </div>
        </div>
        <TagBlockField v-if="tagBlockMode" data-prompt-undo v-model="widgets.main_prompt_text" :color-fn="blockColorClass" placeholder="태그 추가..." @open-wildcard="forwardWildcard" />
        <textarea v-else ref="mainRef" data-prompt-undo v-model="widgets.main_prompt_text" class="auto-grow" placeholder="메인 태그..."
          @input="onMainInput($event)" @keydown="onAutoKey($event)" @click="closeFieldAc" @blur="closeFieldAc" rows="3"></textarea>
        <div class="ac-popup" v-if="!tagBlockMode && ac.isOpenFor('main_prompt_text')">
          <div v-for="(item, i) in acItems" :key="item.tag" class="ac-item" :class="{ selected: acIdx === i }" @mousedown.prevent="acceptFieldSuggestion(item.tag, 'main_prompt_text')">{{ item.tag }}<span v-if="item.ko" class="ac-ko">{{ item.ko }}</span></div>
        </div>
      </div>
      <CompositionControl :model-value="widgets.main_prompt_text || ''" :other-prompts="compositionOtherPrompts" @append="appendComposition" />
      <div class="input-group">
        <label>접두 <span v-if="sectionTokens.prefix" class="tk-badge" :class="tokenBadgeClass(sectionTokens.prefix)">{{ sectionTokens.prefix }}t</span></label>
        <TagBlockField v-if="tagBlockMode" data-prompt-undo v-model="widgets.prefix_prompt_text" :color-fn="blockColorClass" placeholder="선행 추가..." @open-wildcard="forwardWildcard" />
        <textarea v-else ref="prefixRef" data-prompt-undo v-model="widgets.prefix_prompt_text" class="auto-grow" placeholder="선행..." @input="autoGrow($event.target)"></textarea>
      </div>
      <div class="input-group">
        <label>접미 <span v-if="sectionTokens.suffix" class="tk-badge" :class="tokenBadgeClass(sectionTokens.suffix)">{{ sectionTokens.suffix }}t</span></label>
        <TagBlockField v-if="tagBlockMode" data-prompt-undo v-model="widgets.suffix_prompt_text" :color-fn="blockColorClass" placeholder="후행 추가..." @open-wildcard="forwardWildcard" />
        <textarea v-else ref="suffixRef" data-prompt-undo v-model="widgets.suffix_prompt_text" class="auto-grow" placeholder="후행..." @input="autoGrow($event.target)"></textarea>
      </div>
      <details class="input-group exclude-section">
        <summary class="exclude-toggle">제외 (로컬)
          <span v-if="excludeRuleCount" class="excl-badge">{{ excludeRuleCount }}</span><Icon name="chevron-down" /><button class="excl-mgr-btn" @click.prevent.stop="showExcludeManager = true"><Icon name="search" /> 관리</button></summary>
        <div class="exclude-help">
          <span>단어 → 포함하는 모든 태그 제외 (short → short hair, very short hair)</span>
          <span>*단어 → 완전 일치만 제외 (*blue hair → blue hair만)</span>
          <span>_단어 → 앞에 붙는 태그 제외 (_short → very short, too short)</span>
          <span>단어_ → 뒤에 붙는 태그 제외 (short_ → short hair, short pants)</span>
          <span>_단어_ → 포함 (단어와 동일, 명시적)</span>
          <span>~단어 → 예외 완전일치 유지</span>
          <span>~_단어 → 예외 접미 유지 (~_tank top → tank top 유지)</span>
          <span>~단어_ → 예외 접두 유지 (~tank_ → tank top 유지)</span>
          <span>~_단어_ → 예외 포함 유지 (~_tank top_ → blue tank top 등 유지)</span>
          <span>규칙은 쉼표·줄바꿈으로 나눈다 — 공백은 규칙 안 글자 (long hair 는 규칙 하나)</span>
        </div>
        <TagBlockField v-if="tagBlockMode" data-prompt-undo v-model="widgets.exclude_prompt_local_input" :color-fn="excludeColorFn" :split="splitExcludeRules" :join="rewriteExcludeRules" placeholder="제외 규칙 추가..." />
        <textarea v-else data-prompt-undo v-model="widgets.exclude_prompt_local_input" class="auto-grow exclude-textarea" placeholder="제외 규칙 (쉼표·줄바꿈 구분)..." rows="2"></textarea>
      </details>
    </details>

    <!-- Exclude Manager Modal -->
    <transition name="fade">
      <div v-if="showExcludeManager" class="em-overlay" @mousedown.self="showExcludeManager = false">
        <div class="em-modal">
          <div class="em-header">
            <h3>제외어 관리</h3>
            <span class="em-desc">제외 규칙별 매칭 태그 미리보기</span>
            <button class="close-btn" @click="showExcludeManager = false"><Icon name="close" /></button>
          </div>
          <div class="em-body">
            <!-- 좌측: 규칙 목록 + 추가 -->
            <div class="em-rules">
              <input v-model="exRuleSearch" class="em-search" placeholder="규칙 검색..." />
              <div v-for="item in filteredExRules" :key="item.i" class="em-rule-item"
                :class="[excludeColorFn(item.rule), { active: selectedExRule === item.i }]"
                @click="selectedExRule = item.i; loadExcludeMatches(item.rule)">
                <!-- 편집 모드 -->
                <input v-if="editingExRule === item.i" class="em-rule-edit" v-model="editExRuleText"
                  @blur="finishEditExRule(item.i)" @keydown.enter="finishEditExRule(item.i)" @keydown.escape="editingExRule = -1"
                  @click.stop ref="exRuleEditRef" />
                <!-- 표시 모드 -->
                <span v-else class="em-rule-text" @dblclick.stop="startEditExRule(item.i)">{{ item.rule }}</span>
                <span class="em-match-count">{{ excludeMatches[item.rule]?.length || '...' }}</span>
                <button class="em-rule-rm" @click.stop="removeExcludeRule(item.i)"><Icon name="close" /></button>
              </div>
              <div v-if="excludeRules.length === 0" class="em-empty-sm">제외 규칙 없음</div>
              <div v-else-if="filteredExRules.length === 0" class="em-empty-sm">검색 결과 없음</div>
              <div class="em-add-row">
                <input v-model="newExcludeRule" placeholder="규칙 추가..." class="em-add-input" @keydown.enter="addExcludeRule" />
                <button class="em-add-btn" @click="addExcludeRule">+</button>
              </div>
            </div>
            <!-- 우측: 매칭 태그 목록 -->
            <div class="em-matches">
              <template v-if="selectedExRule >= 0 && currentExMatches.length > 0">
                <div class="em-match-header">
                  "{{ excludeRules[selectedExRule] }}" — {{ currentExMatches.length }}개 매칭
                </div>
                <div class="em-match-list">
                  <button v-for="item in currentExMatchItems" :key="item.tag" class="em-tag"
                    :class="{ excepted: item.excepted }" :title="item.hint"
                    @click="toggleException(item.tag)"
                    @contextmenu.prevent="addExactExclude(item.tag)">
                    {{ item.label }}
                  </button>
                </div>
              </template>
              <div v-else class="em-empty">좌측에서 규칙을 선택하세요</div>
            </div>
          </div>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useWidgetStore, requestAction, whenWidgetValuesLoaded } from '../stores/widgetStore.js'
import { openCharPresetModal } from '../composables/uiModals.js'
import { tagBlockMode } from '../composables/uiPrefs'
import { getBackend, onBackendEvent } from '../bridge.js'
import { isCaretMoveKey, isSameTagQuery, replaceTagToken, tagQueryAt, tagTokenAt, type TagQuery } from '../utils/tagSuggest'
import { useTagAutocomplete } from '../composables/useTagAutocomplete'
import { usePromptUndo } from '../composables/usePromptUndo'
import { isImeComposing } from '../utils/imeComposition'
import { isPanelOnTop, promptUndoCommand } from '../utils/promptUndoKeys'
import { createExcludeMatchRequests, parseExcludeMatches } from '../utils/excludeMatches'
import {
  addExactExcludeRule, appendExcludeRule, buildExcludeKeepIndex, excludeRuleColorClass, keepToggleHint,
  rewriteExcludeRules, splitExcludeRules, toggleKeepExact,
} from '../utils/excludeRules'
import { GENERATION_FAMILY_ITEMS, familyRestoreAction, isKrea2Family, storedFamilyLabel } from '../utils/generationFamily'
import { storedOllamaModel, storedOllamaUrl } from '../utils/ollamaPrefs'
import {
  isOptimizePreviewStale, optimizeContextPayload, snapshotOptimizeInputs, type OptimizeInputs,
} from '../utils/optimizePreview'
import CustomSelect from './CustomSelect.vue'
import TagBlockField from './TagBlockField.vue'
import CompositionControl from './CompositionControl.vue'

// 타입 문법 — 배열 문법은 emit 배선 가드(tests/test_component_emit_contract.py)가 읽지 못해
// 아무도 안 쏘는 'toggle-extend' 선언과 App 의 죽은 리스너가 남아 있었다.
const emit = defineEmits<{ 'open-wildcard': [name: string] }>()
/** TagBlockField 의 와일드카드 블록 클릭 → App 의 와일드카드 관리자로 전달 */
function forwardWildcard(name: string | undefined) {
  if (name) emit('open-wildcard', name)
}
const store = useWidgetStore()
const widgets = store.widgets
const route = useRoute()
const compositionOtherPrompts = computed(() => [
  widgets.char_count_input, widgets.character_input, widgets.copyright_input,
  widgets.artist_input, widgets.prefix_prompt_text, widgets.suffix_prompt_text,
].filter(Boolean).join(', '))

function appendComposition(text: string) {
  // Main-tag writes use the same Python proxy/recomposition path as text and blocks.
  // Save both sides immediately so even a rapid Ctrl+Z restores the full old prompt
  // (focus stays on the panel button or falls back to body — both are in the undo scope,
  //  utils/promptUndoKeys.promptUndoScope).
  commitUndoSnapshot()
  widgets.main_prompt_text = text
  commitUndoSnapshot()
}

// 옵션 라벨 = Python ComboBoxProxy 항목(대소문자 구분) — utils/generationFamily.ts 참고.
// 저장값(대문자)과 라벨 비교는 전부 대소문자 무시(isKrea2Family/toFamilyLabel).
const generationFamilyItems: string[] = [...GENERATION_FAMILY_ITEMS]
const generationFamilyStorageKey = 'generationFamily'
const standardSnapshotStorageKey = 'generationFamily.standardParams'
const savedGenerationFamily = storedFamilyLabel(window.localStorage.getItem(generationFamilyStorageKey))
const showGenerationFamily = computed(() => ['t2i', 'i2i'].includes(String(route.name || '')))
const isI2IRoute = computed(() => String(route.name || '') === 'i2i')
const isKrea2Generation = computed(
  () => showGenerationFamily.value && isKrea2Family(widgets.generation_family_combo))
function loadStandardGenerationSnapshot(): { steps: string; cfg: string } | null {
  try {
    const value = JSON.parse(window.localStorage.getItem(standardSnapshotStorageKey) || 'null')
    if (value && typeof value.steps === 'string' && typeof value.cfg === 'string') return value
  } catch {}
  return null
}
let standardGenerationSnapshot: { steps: string; cfg: string } | null = loadStandardGenerationSnapshot()
let generationFamilyRestored = false
let generationFamilyRestoreTimer: ReturnType<typeof setTimeout> | null = null

function saveStandardGenerationSnapshot() {
  if (!standardGenerationSnapshot) return
  try { window.localStorage.setItem(standardSnapshotStorageKey, JSON.stringify(standardGenerationSnapshot)) } catch {}
}

function clearStandardGenerationSnapshot() {
  standardGenerationSnapshot = null
  try { window.localStorage.removeItem(standardSnapshotStorageKey) } catch {}
}

watch(() => widgets.generation_family_combo, (value: any, previous: any) => {
  if (!generationFamilyRestored) return
  const family = String(value || 'STANDARD').toUpperCase()
  try { window.localStorage.setItem(generationFamilyStorageKey, family) } catch {}
  if (isKrea2Family(family) && !isKrea2Family(previous)) {
    if (!standardGenerationSnapshot) {
      standardGenerationSnapshot = {
        steps: String(widgets.steps_input || '25'),
        cfg: String(widgets.cfg_input || '7'),
      }
      saveStandardGenerationSnapshot()
    }
    widgets.steps_input = '8'
    widgets.cfg_input = '1'
  } else if (!isKrea2Family(family) && isKrea2Family(previous) && standardGenerationSnapshot) {
    widgets.steps_input = standardGenerationSnapshot.steps
    widgets.cfg_input = standardGenerationSnapshot.cfg
    clearStandardGenerationSnapshot()
  }
})

onMounted(() => {
  getBackend().then(() => {
    let attempts = 0
    const restore = () => {
      // QWebChannel의 초기 getAllWidgetValues 콜백이 끝난 뒤 복원해야
      // Python 기본값이 localStorage 선택을 다시 덮어쓰지 않는다.
      if (widgets.generation_family_combo === undefined && attempts++ < 40) {
        generationFamilyRestoreTimer = setTimeout(restore, 50)
        return
      }
      const family = savedGenerationFamily
      generationFamilyRestored = true
      // 이 브라우저에 저장된 선택이 없으면 Python 현재값을 그대로 둔다(웹 모드 새 브라우저가
      // 기본값 Standard 로 공유 콤보를 바꾸지 않게).
      if (family === null) return
      // savedGenerationFamily 는 라벨('Standard'/'Krea2')이다 — 대문자 상수와 직접 비교하면
      // 영원히 거짓이라 Standard 스냅샷 복원이 죽어 있었다 (utils/generationFamily.ts).
      // 콤보가 이미 Krea2 면(Python 이 새로고침을 넘어 상태 유지) steps/CFG 는 사용자가 맞춘
      // 현재값이라 건드리지 않는다 — Krea2 기본값(8/1)은 처음 전환할 때 watch 가 적용했다.
      const restoreAction = familyRestoreAction(
        family, widgets.generation_family_combo, standardGenerationSnapshot !== null)
      if (restoreAction === 'restore-standard' && standardGenerationSnapshot) {
        widgets.steps_input = standardGenerationSnapshot.steps
        widgets.cfg_input = standardGenerationSnapshot.cfg
        clearStandardGenerationSnapshot()
      }
      widgets.generation_family_combo = family
    }
    restore()
  })
})

onUnmounted(() => {
  if (generationFamilyRestoreTimer) clearTimeout(generationFamilyRestoreTimer)
})

// 블록 모드 (Settings에서 제어) — composables/uiPrefs 의 모듈 전역 ref 를 Settings 와 같이 본다.
// 예전엔 300ms setInterval 로 localStorage 를 폴링했다(같은 문서의 변경은 storage 이벤트가 안 온다).
// 파일 값은 App.vue uiPrefsLoaded → restoreUiFlagsFromPrefs 가 넣는다.

let _stopInitialUndoBaseline: (() => void) | null = null
const rootRef = ref<HTMLElement | null>(null)
const _backendUnsubs: Array<() => void> = []   // onBackendEvent 해제 함수 (탭 전환 시 누수 방지)
onMounted(() => {
  // Undo/Redo 키보드 단축키 — window 에 둔다(document 로 옮기면 EditorView 의 document 리스너가
  // stopImmediatePropagation 으로 먼저 가져가는 우선권이 사라진다). 판정은 onKeyDownGlobal 참고.
  window.addEventListener('keydown', onKeyDownGlobal)
  // Undo 기준점 = 첫 초기값(getAllWidgetValues)이 스토어에 들어간 직후의 상태.
  // 예전 '800ms 뒤 스냅' 은 초기값보다 먼저 찍히면 빈 칸이 기준점이 되어 Ctrl+Z 가 전 칸을 비웠다.
  _stopInitialUndoBaseline = whenWidgetValuesLoaded(resetUndoBaseline)
})
onUnmounted(() => {
  window.removeEventListener('keydown', onKeyDownGlobal)
  if (_stopInitialUndoBaseline) { _stopInitialUndoBaseline(); _stopInitialUndoBaseline = null }
  ac.close()
  for (const off of _backendUnsubs) { try { off() } catch {} }
  _backendUnsubs.length = 0
  // Undo 기록의 debounce 타이머와 watch 일괄 해제
  promptUndo.dispose()
})

const artistLocked = computed({
  get: () => widgets.btn_lock_artist === 'true',
  set: (v) => { widgets.btn_lock_artist = v ? 'true' : 'false' }
})
function toggleArtistLock() { artistLocked.value = !artistLocked.value; requestAction('set_artist_locked', { locked: artistLocked.value }) }

const totalPromptRef = ref<HTMLTextAreaElement | null>(null)
const negRef = ref<HTMLTextAreaElement | null>(null)
const artistRef = ref<HTMLTextAreaElement | null>(null)
const prefixRef = ref<HTMLTextAreaElement | null>(null)
const mainRef = ref<HTMLTextAreaElement | null>(null)
const suffixRef = ref<HTMLTextAreaElement | null>(null)

const modelItems = computed(() => store.getProperty('model_combo', 'items') || [])
const modelOptionGroups = computed(() => store.getProperty('model_combo', 'optionGroups') || [])
const vaeItems = computed(() => store.getProperty('vae_main_combo', 'items') || [])
const teItems = computed(() => store.getProperty('te_main_input', 'items') || [])
const teSelectedList = computed(() => {
  const raw = widgets.te_main_input || ''
  return raw.split(',').map((s: string) => s.trim()).filter(Boolean)
})
const teUnselectedItems = computed(() => {
  const selected = new Set(teSelectedList.value)
  return teItems.value.filter((f: string) => !selected.has(f))
})
function _serializeTe(list: string[]) {
  widgets.te_main_input = list.join(', ')
}
function addTeFile(name: any) {
  if (!name) return
  const cur = teSelectedList.value
  if (cur.includes(name)) return
  // teItems 정렬 순서 유지
  const next = teItems.value.filter((f: string) => cur.includes(f) || f === name)
  _serializeTe(next)
}
function removeTeFile(name: string) {
  _serializeTe(teSelectedList.value.filter((f: string) => f !== name))
}
// 토큰 카운트 헬퍼 — SD/CLIP 근사 (실제 토크나이저보다 약간 보수적)
// 규칙: 각 태그(쉼표 구분) → 단어 수 합산 + 쉼표 수
function approxTokens(text: string) {
  if (!text || !String(text).trim()) return 0
  const chunks = String(text).split(',').map(s => s.trim()).filter(Boolean)
  let total = 0
  for (const c of chunks) {
    const words = c.split(/\s+/).filter(Boolean)
    total += words.length
  }
  return total + Math.max(0, chunks.length - 1) // 쉼표
}
const tokenCount = computed(() => approxTokens(widgets.total_prompt_display))

// 섹션별 토큰 — 블록모드에서도 동작 (widgets 값은 join된 문자열)
const sectionTokens = computed(() => ({
  character: approxTokens(widgets.character_input),
  copyright: approxTokens(widgets.copyright_input),
  artist:    approxTokens(widgets.artist_input),
  main:      approxTokens(widgets.main_prompt_text),
  prefix:    approxTokens(widgets.prefix_prompt_text),
  suffix:    approxTokens(widgets.suffix_prompt_text),
}))
// 75 = CLIP 1청크 한계, 150 = 2청크(BREAK), 225 = 3청크
// 0=숨김, <75=녹색, <150=노랑, ≥150=빨강
function tokenBadgeClass(n: number) {
  if (!n) return ''
  if (n < 75) return 'tk-ok'
  if (n < 150) return 'tk-warn'
  return 'tk-over'
}

// ── Undo/Redo ──
// 기록(스냅숏 스택·debounce·대기 변경 확정)은 composables/usePromptUndo, 추적 키는
// utils/promptUndoKeys.PROMPT_UNDO_KEYS (인물수·캐릭터·작품·작가·메인·접두·접미·네거티브·제외).
const promptUndo = usePromptUndo(widgets)
const undoStack = promptUndo.undoStack    // 템플릿: Undo/Redo 버튼 disabled
const redoStack = promptUndo.redoStack
const performUndo = promptUndo.undo
const performRedo = promptUndo.redo
/** 프로그램 편집(최적화 적용·구도 추가) 직전·직후 — debounce 없이 지금 상태를 스냅숏으로 */
const commitUndoSnapshot = promptUndo.commit
/** 초기값이 들어온 지금 상태를 Undo 의 출발점으로 — 그 전(빈 칸)으로는 되돌아가지 않는다 */
const resetUndoBaseline = promptUndo.resetBaseline

// 키보드 단축키 (Ctrl+Z / Ctrl+Y / Ctrl+Shift+Z)
// 이 패널은 v-show 로 늘 마운트돼 있다. 패널이 보이고 포커스가 프롬프트 필드([data-prompt-undo])
// · 패널 버튼 · 어디에도 없음(body, 단 패널이 모달에 가려지지 않았을 때) 중 하나일 때만 스냅숏
// Undo/Redo 로 가로챈다 — 채팅·설정·검색·모달 등 다른 입력칸의 네이티브 실행 취소는 그대로 두고,
// 보이지 않는 T2I 프롬프트가 몰래 되돌아가지도 않는다. 판정: utils/promptUndoKeys.promptUndoCommand.
// (프롬프트 필드 안에서는 네이티브 textarea undo 대신 이 스냅숏이 쓰인다 — 값이 Python 과
//  같이 움직여야 해서다.)
function onKeyDownGlobal(e: KeyboardEvent) {
  const root = rootRef.value
  const command = promptUndoCommand(e, {
    target: e.target,
    root,
    visible: !!root && root.offsetParent !== null,
    unobscured: () => isPanelOnTop(root, document, { width: window.innerWidth, height: window.innerHeight }),
  })
  if (!command) return
  e.preventDefault()
  // 팝업 후보는 되돌리기 전 텍스트의 조각을 가리킨다 — 남겨 두면 수락이 엉뚱한 곳을 바꾼다
  closeFieldAc()
  if (command === 'undo') performUndo()
  else performRedo()
}

// 블록 색상 분류
const countPattern = /^(\d+)?(girl|boy|other)s?$|^solo$|^multiple_/
const blockColorCache = ref<Record<string, string>>({})

// ── Exclude Manager ──
const showExcludeManager = ref(false)
const selectedExRule = ref<any>(-1)
const excludeMatches = ref<Record<string, string[]>>({})  // {규칙텍스트: [tags]} — 인덱스 아닌 규칙 문자열로 키잉(중간 삭제/순서변경에도 안 어긋남)

// 규칙 나누기는 적용 쪽(core/exclude_rules)과 같은 정의 — 쉼표·줄바꿈만, 공백으로는 나누지 않는다
const excludeRules = computed(() => splitExcludeRules(widgets.exclude_prompt_local_input || ''))
const excludeRuleCount = computed(() => excludeRules.value.length)

// 매니저 규칙 검색 — 원본 인덱스(i)를 함께 들고 다녀 selectedExRule/remove가 정확히 동작
const exRuleSearch = ref('')
const filteredExRules = computed(() => {
  const q = exRuleSearch.value.trim().toLowerCase()
  const items = excludeRules.value.map((rule: string, i: number) => ({ rule, i }))
  return q ? items.filter((x: { rule: string; i: number }) => x.rule.toLowerCase().includes(q)) : items
})

const currentExMatches = computed(() => {
  const r = excludeRules.value[selectedExRule.value]
  return r ? (excludeMatches.value[r] || []) : []
})

// 규칙마다 한 번만 조회 — 백엔드는 GUI 스레드에서 태그 사전 전체를 훑는다(클릭마다 100ms+ 멈춤).
const excludeMatchRequests = createExcludeMatchRequests()

async function loadExcludeMatches(rule: string) {
  if (!excludeMatchRequests.begin(rule, excludeMatches.value)) return
  let sent = false
  try {
    const backend: any = await getBackend()
    if (!backend?.getExcludeMatches) return
    backend.getExcludeMatches(rule, (json: string) => {
      excludeMatchRequests.end(rule)
      const tags = parseExcludeMatches(json)
      if (tags) excludeMatches.value = { ...excludeMatches.value, [rule]: tags }
    })
    sent = true
  } catch {
    // 브리지 준비 실패 — 아래에서 조회 중 표시를 풀어 다음 클릭에 다시 시도한다
  } finally {
    if (!sent) excludeMatchRequests.end(rule)
  }
}

const newExcludeRule = ref('')
const editingExRule = ref(-1)
const editExRuleText = ref('')
const exRuleEditRef = ref<any>(null)

function startEditExRule(idx: any) {
  editingExRule.value = idx
  editExRuleText.value = excludeRules.value[idx]
  nextTick(() => { if (exRuleEditRef.value?.[0]) exRuleEditRef.value[0].focus() })
}

function finishEditExRule(idx: any) {
  if (editingExRule.value !== idx) return
  const newText = editExRuleText.value.trim()
  editingExRule.value = -1
  if (!newText) { removeExcludeRule(idx); return }
  const rules = excludeRules.value.slice()
  rules[idx] = newText
  // 칸을 다시 쓸 때는 바뀐 규칙 자리만 고친다 — 한 줄에 한 규칙·카테고리별 줄 배치가 남는다
  widgets.exclude_prompt_local_input = rewriteExcludeRules(widgets.exclude_prompt_local_input || '', rules)
  // 매칭 갱신
  selectedExRule.value = idx
  loadExcludeMatches(newText)
}

function addExcludeRule() {
  const rule = newExcludeRule.value.trim()
  if (!rule) return
  widgets.exclude_prompt_local_input = appendExcludeRule(widgets.exclude_prompt_local_input || '', rule)
  newExcludeRule.value = ''
}

function removeExcludeRule(idx: any) {
  const rules = excludeRules.value.slice()
  rules.splice(idx, 1)
  widgets.exclude_prompt_local_input = rewriteExcludeRules(widgets.exclude_prompt_local_input || '', rules)
  if (selectedExRule.value >= rules.length) selectedExRule.value = -1
}

// 예외 표시 — 적용과 같은 판정(~완전일치뿐 아니라 ~_x·~x_·~_x_ 패턴 예외도)으로 칠한다.
// 색인은 칸 텍스트가 바뀔 때만, 태그별 판정은 매칭 목록이 바뀔 때만 만든다 — 렌더마다 태그마다
// 규칙 전체를 다시 나누면 규칙 검색·추가 칸 한 글자마다 매칭 수천 개 × 규칙 수백 개를 돌았다.
const excludeKeepIndex = computed(() => buildExcludeKeepIndex(widgets.exclude_prompt_local_input || ''))
const currentExMatchItems = computed(() => {
  const keep = excludeKeepIndex.value
  return currentExMatches.value.map((tag: string) => {
    const keptBy = keep.keptBy(tag)
    return { tag, label: tag.replace(/_/g, ' '), excepted: keptBy !== null, hint: keepToggleHint(keptBy) }
  })
})

function addExactExclude(tag: string) {
  // 우클릭: *완전일치 제외 규칙 추가 (같은 태그의 완전 일치 규칙이 있으면 그대로)
  const cur = widgets.exclude_prompt_local_input || ''
  const next = addExactExcludeRule(cur, tag)
  if (next !== cur) widgets.exclude_prompt_local_input = next
}

function toggleException(tag: string) {
  // ~태그 예외 추가 / 해제 (해제는 같은 태그를 가리키는 ~완전일치 규칙 모두,
  // 패턴 예외로만 유지되는 태그는 그대로 — utils/excludeRules.toggleKeepExact)
  const cur = widgets.exclude_prompt_local_input || ''
  const next = toggleKeepExact(cur, tag)
  if (next !== cur) widgets.exclude_prompt_local_input = next
}

// 최종 프롬프트 블록 변경 시 → 원본 필드에서 태그 제거
function onTotalBlockChange(newVal: string) {
  // FINAL 블록의 새 순서 — 태그(소문자) → 위치 인덱스
  const order = newVal.split(',').map((t: string) => t.trim()).filter(Boolean)
  const orderIdx = new Map<string, number>()
  order.forEach((t: string, i: number) => { const k = t.toLowerCase(); if (!orderIdx.has(k)) orderIdx.set(k, i) })
  const newTags = new Set(order.map((t: string) => t.toLowerCase()))
  // 각 필드: (1) FINAL에서 사라진 태그 제거 (FINAL에서 지우면 인물수 칸 등에 남던 버그 수정)
  //          (2) 남은 태그를 FINAL 순서로 재정렬 → FINAL에서 블록을 옮기면 해당 필드 내부 순서가 따라감.
  //   (필드 순서 자체는 구조상 고정이므로 필드 경계를 넘는 이동은 각 필드 영역 내 재정렬로만 반영됨)
  for (const key of ['char_count_input', 'character_input', 'copyright_input', 'artist_input',
                     'main_prompt_text', 'prefix_prompt_text', 'suffix_prompt_text']) {
    const cur = widgets[key] || ''
    const kept = cur.split(',').map((t: string) => t.trim()).filter((t: string) => t && newTags.has(t.toLowerCase()))
    kept.sort((a: string, b: string) => (orderIdx.get(a.toLowerCase()) ?? 0) - (orderIdx.get(b.toLowerCase()) ?? 0))
    const result = kept.join(', ')
    if (result !== cur.trim()) widgets[key] = result
  }
}

// 규칙 종류별 색 — 예외(초록)·완전 일치(노랑)·포함(bc-nsfw)·접미(주황)·접두(보라). 분류는 파서와 같다.
function excludeColorFn(text: string) {
  return excludeRuleColorClass(text)
}

function blockColorClass(text: string) {
  if (text.includes('__') && /__(.+?)__/.test(text)) return 'wc-block'
  let t = text.trim().toLowerCase().replace(/ /g, '_').replace(/^\(+/, '').replace(/[\):.\d]+$/, '').trim()
  if (countPattern.test(t)) return 'bc-count'
  return blockColorCache.value[t] ? 'bc-' + blockColorCache.value[t] : ''
}

// 태그 분류 요청
async function classifyVisibleTags() {
  const allTags = new Set<string>()
  for (const key of ['main_prompt_text', 'prefix_prompt_text', 'suffix_prompt_text', 'total_prompt_display']) {
    const text: string = widgets[key] || ''
    for (const t of text.split(',')) {
      let tag = t.trim().replace(/ /g, '_').replace(/^\(+/, '').replace(/[\):.\d]+$/, '').trim()
      if (tag && !countPattern.test(tag.toLowerCase()) && !blockColorCache.value[tag.toLowerCase()] && !/__(.+?)__/.test(tag)) allTags.add(tag)
    }
  }
  if (allTags.size === 0) return
  const backend: any = await getBackend()
  if (backend.classifyTags) {
    backend.classifyTags(JSON.stringify([...allTags]), (json: string) => {
      try {
        const r = JSON.parse(json)
        if (!r.error) {
          const m: Record<string, string> = { sexual:'nsfw', body_parts:'body', clothing:'clothing', pose:'action', expression:'expression', background:'bg', composition:'bg', effect:'effect', objects:'objects', color:'color', character_trait:'trait' }
          for (const [tag, cat] of Object.entries(r)) blockColorCache.value[tag.toLowerCase()] = m[cat as string] || ''
        }
      } catch {}
    })
  }
}
watch(tagBlockMode, v => { if (v) setTimeout(classifyVisibleTags, 200) })

// 딥 프롬프트 클리너
interface Conflict { group: string; tags: string[]; [k: string]: any }
interface OptPreview {
  before: string; after: string; removed: number; fromContext: number; tagCount: number; conflicts: Conflict[]
  /** 미리보기를 만든 입력(메인 + 6칸) — [적용] 때 하나라도 바뀌었으면 다시 최적화한다 */
  inputs: OptimizeInputs
  [k: string]: any
}
const optResult = ref('')
const promptConflicts = ref<Conflict[]>([])
const optPreview = ref<OptPreview | null>(null)   // {before, after, removed, fromContext, tagCount, conflicts, inputs}
// 최적화 대상은 메인 태그뿐이다. 예전엔 7칸 합본(최종 프롬프트)을 정리해 메인 칸에 통째로 써서
// 인물수·캐릭터·작품·작가·접두·접미 태그가 메인에 영구 복제됐다(캐릭터를 바꿔도 옛 태그가 남음).
// 나머지 6칸(OPTIMIZE_CONTEXT_KEYS)은 context 로 보내 '다른 칸에 이미 있는 태그'를 메인에서 빼는 데와
// 충돌 검사에만 쓴다 — 그래서 결과는 6칸에도 달려 있다(utils/optimizePreview).
async function optimizePrompt() {
  const backend: any = await getBackend()
  if (!backend.deepCleanPrompt) return
  const inputs = snapshotOptimizeInputs(widgets)
  const before = inputs.main
  const context = optimizeContextPayload(inputs)
  backend.deepCleanPrompt(JSON.stringify({ prompt: before, context }), (json: string) => {
    try {
      const d = JSON.parse(json)
      if (d.error) { optResult.value = d.error; return }
      // 즉시 적용하지 않고 before/after 비교 후 [적용]으로 반영
      optPreview.value = {
        before,
        // 메인 태그가 전부 다른 칸과 겹치면 결과는 빈 문자열이다 — `||` 로 원문에 되돌리면 안 된다
        after: typeof d.optimized === 'string' ? d.optimized : before,
        removed: d.removed || 0,
        fromContext: d.removed_from_context || 0,
        tagCount: d.tag_count || 0,
        conflicts: d.conflicts || [],
        inputs,
      }
    } catch {}
  })
}
function applyOptimize() {
  const p = optPreview.value
  if (!p) return
  if (isOptimizePreviewStale(p.inputs, widgets)) {
    // 미리보기 뒤에 메인 태그나 다른 칸(캐릭터·접두 등)이 바뀌었다 — 옛 결과로 덮으면 그 사이 편집이
    // 사라지거나, 다른 칸에서 지운 태그가 메인에서도 빠져 프롬프트에서 통째로 없어진다
    optPreview.value = null
    requestAction('show_toast', { type: 'info', msg: '프롬프트 칸이 바뀌어 다시 최적화했습니다 — 미리보기를 확인하세요' })
    optimizePrompt()
    return
  }
  // 적용 전후를 바로 스냅숏으로 남겨, 곧바로 Ctrl+Z 해도 최적화 전으로 돌아간다
  // ([적용] 버튼은 미리보기와 함께 사라져 포커스가 body 로 떨어진다 — promptUndoKeys 의 'unfocused')
  commitUndoSnapshot()
  widgets.main_prompt_text = p.after
  commitUndoSnapshot()
  promptConflicts.value = p.conflicts || []
  optResult.value = `${p.removed}개 중복 제거, ${p.tagCount}개 태그`
  optPreview.value = null
  nextTick(() => { if (mainRef.value) autoGrow(mainRef.value) })
  setTimeout(() => { optResult.value = '' }, 5000)
}
function cancelOptimize() { optPreview.value = null }

// ⑤ 카테고리 분리 토글
const SEP_CATS = [
  { key: 'expression', label: '표정' },
  { key: 'location', label: '배경·장소' },
  { key: 'pose', label: '포즈·동작' },
  { key: 'object', label: '사물' },
  { key: 'meta', label: '메타' },
]
const showSeparate = ref(false)
const sepCats = reactive<Record<string, boolean>>({ expression: false, location: false, pose: false, object: false, meta: false })
const sepCounts = ref<Record<string, number>>({})
const sepResult = ref('')
async function refreshSepCounts() {
  const backend: any = await getBackend()
  if (!backend || !backend.separateTags) return
  const allKeys = SEP_CATS.map(c => c.key)
  backend.separateTags(widgets.main_prompt_text || '', JSON.stringify(allKeys), (json: string) => {
    try { const d = JSON.parse(json); if (!d.error) sepCounts.value = d.counts || {} } catch {}
  })
}
function toggleSeparate() {
  showSeparate.value = !showSeparate.value
  if (showSeparate.value) refreshSepCounts()
}
async function applySeparate(mode: string) {
  const selected = SEP_CATS.map(c => c.key).filter(k => sepCats[k])
  if (!selected.length) { requestAction('show_toast', { type: 'info', msg: '분류할 카테고리를 선택하세요' }); return }
  const backend: any = await getBackend()
  if (!backend || !backend.separateTags) return
  backend.separateTags(widgets.main_prompt_text || '', JSON.stringify(selected), (json: string) => {
    try {
      const d = JSON.parse(json)
      if (d.error) { requestAction('show_toast', { type: 'error', msg: '분류 실패: ' + d.error }); return }
      const groups = d.groups || {}
      const picked = selected.flatMap(k => groups[k] || [])
      if (mode === 'remove') {
        widgets.main_prompt_text = d.rest || ''
        sepResult.value = `${picked.length}개 제거`
      } else {
        widgets.main_prompt_text = picked.join(', ')
        sepResult.value = `${picked.length}개만 남김`
      }
      refreshSepCounts()
      nextTick(() => { if (mainRef.value) autoGrow(mainRef.value) })
      setTimeout(() => { sepResult.value = '' }, 4000)
    } catch {}
  })
}

// ② color 페어링 결합
// (🎯 구체화 버튼 제거 — ADVANCED의 '프롬프트 집중' 토글로 대체됨)

// 캐릭터 인사이트
const charInsight = ref<{ tags: string[]; raw: string }>({ tags: [], raw: '' })
async function loadCharTags() {
  const char = widgets.character_input
  if (!char || char.length < 2) { charInsight.value = { tags: [], raw: '' }; return }
  const backend: any = await getBackend()
  if (backend.getCharacterInsight) {
    backend.getCharacterInsight(char, (json: string) => {
      try { const d = JSON.parse(json); charInsight.value = { tags: d.tags || [], raw: d.raw || '' } } catch { charInsight.value = { tags: [], raw: '' } }
    })
  }
}
function insertCharTag(tag: string) {
  const cur = widgets.main_prompt_text || ''
  if (!cur.toLowerCase().includes(tag.toLowerCase())) widgets.main_prompt_text = cur ? cur.replace(/,?\s*$/, '') + ', ' + tag + ', ' : tag + ', '
}
function applyOfficialTags() { if (charInsight.value.raw) { widgets.main_prompt_text = charInsight.value.raw; nextTick(() => { if (mainRef.value) autoGrow(mainRef.value) }) } }

// Ollama
const ollamaLoading = ref(false)
const ollamaMode = ref('expand')
const showNlInput = ref(false)
const nlPrompt = ref('')
// LLM 버튼 2개(태그 AI / 자연어 AI) 드롭다운
const aiMenu = ref('')           // '' | 'tag' | 'nl'
const pendingNlMode = ref('nl2tags')
const nlInputRef = ref<HTMLInputElement | null>(null)
function toggleAiMenu(which: string) { aiMenu.value = aiMenu.value === which ? '' : which }
function runAi(mode: string) { aiMenu.value = ''; ollamaMode.value = mode; runOllama() }
function openNlInput(mode: string) {
  aiMenu.value = ''
  pendingNlMode.value = mode
  showNlInput.value = true
  nextTick(() => { try { nlInputRef.value?.focus() } catch {} })
}
function runPendingNl() {
  if (ollamaLoading.value) return
  ollamaMode.value = pendingNlMode.value || 'nl2tags'
  runOllama()
}
/** 자연어 입력 Enter — IME 조합 확정 Enter 는 실행이 아니다(마지막 음절이 잘린 채 나간다) */
function onNlEnter(e: KeyboardEvent) {
  if (isImeComposing(e)) return
  runPendingNl()
}
const nlResult = ref('')
const nlRes = ref<{ w: string; h: string } | null>(null)   // 창의 모드 추천 해상도 {w, h}
let ollamaTimer: ReturnType<typeof setTimeout> | null = null
async function runOllama() {
  const mode = ollamaMode.value
  const main = widgets.main_prompt_text || ''
  let contentArg = main
  let extraPrompt = ''
  let creativeChar = ''
  if (mode === 'nl2tags') {
    if (!nlPrompt.value.trim()) { requestAction('show_toast', { type: 'info', msg: '자연어 설명을 입력하세요' }); return }
    extraPrompt = nlPrompt.value
  } else if (mode === 'nl_scene') {
    if (!nlPrompt.value.trim()) { requestAction('show_toast', { type: 'info', msg: '키워드를 입력하세요' }); return }
    contentArg = nlPrompt.value
  } else if (mode === 'creative') {
    // 캐릭터는 별도 전달(외견 DB 조회용), 메인 태그는 추가 힌트로. 완전 비어도 생성 허용.
    creativeChar = (widgets.character_input || '').trim()
    contentArg = main.trim()
  } else {
    if (!main.trim()) { requestAction('show_toast', { type: 'info', msg: '프롬프트를 먼저 입력하세요' }); return }
  }
  ollamaLoading.value = true
  // 타임아웃 안전장치
  if (ollamaTimer) clearTimeout(ollamaTimer)
  ollamaTimer = setTimeout(() => {
    if (ollamaLoading.value) {
      ollamaLoading.value = false
      requestAction('show_toast', { type: 'error', msg: 'AI 응답 시간 초과 — Ollama 서버 상태를 확인하세요' })
    }
  }, 65000)
  const backend: any = await getBackend()
  if (!backend.ollamaEnhance) { ollamaLoading.value = false; if (ollamaTimer) clearTimeout(ollamaTimer); return }
  const url = storedOllamaUrl()
  const model = storedOllamaModel()   // 비면 백엔드 워커가 설치 모델로 정한다
  backend.ollamaEnhance(contentArg, mode, JSON.stringify({ prompt: extraPrompt, character: creativeChar, url, model }))
}

async function runSmartNegative() {
  const positivePrompt = widgets.total_prompt_display || widgets.main_prompt_text || ''
  if (!positivePrompt.trim()) {
    requestAction('show_toast', { type: 'info', msg: '포지티브 프롬프트를 먼저 입력하세요' })
    return
  }
  ollamaLoading.value = true
  if (ollamaTimer) clearTimeout(ollamaTimer)
  ollamaTimer = setTimeout(() => {
    if (ollamaLoading.value) {
      ollamaLoading.value = false
      requestAction('show_toast', { type: 'error', msg: 'AI 응답 시간 초과' })
    }
  }, 65000)
  const backend: any = await getBackend()
  if (!backend.ollamaEnhance) { ollamaLoading.value = false; if (ollamaTimer) clearTimeout(ollamaTimer); return }
  backend.ollamaEnhance(positivePrompt, 'negative', JSON.stringify({ url: storedOllamaUrl(), model: storedOllamaModel() }))
}

function copyNlResult() {
  try { navigator.clipboard?.writeText(nlResult.value) } catch {}
  requestAction('show_toast', { type: 'info', msg: '복사됨' })
}
function useNlAsMain() {
  let add = nlResult.value.trim()
  add = add.replace(/^\s*Resolution:.*$/im, '').trim()   // 해상도 줄은 별도 버튼이라 제외
  const firstBlock = add.split(/\n\s*\n/)[0].trim()        // 다중 블록(창의)이면 첫 블록(태그)만
  if (firstBlock) add = firstBlock
  // 앞의 태그는 override하지 않고 맨 뒤에 추가.
  // 앞에 내용(태그)이 있으면 ', '로 구분해서 자연어를 이어붙임.
  // 예: "muscular" 뒤에 자연어 → "muscular, A muscular man ..."
  const cur = (widgets.main_prompt_text || '').trim().replace(/[,\s]+$/, '')
  widgets.main_prompt_text = cur ? (cur + ', ' + add) : add
  nlResult.value = ''
  nlRes.value = null
  nextTick(() => { if (mainRef.value) autoGrow(mainRef.value) })
  requestAction('show_toast', { type: 'success', msg: '메인 프롬프트 끝에 추가' })
}

function applyNlRes() {
  if (!nlRes.value) return
  widgets.width_input = String(nlRes.value.w)
  widgets.height_input = String(nlRes.value.h)
  requestAction('show_toast', { type: 'success', msg: `해상도 ${nlRes.value.w}×${nlRes.value.h} 적용` })
}

// 자동완성 — 항목은 {tag, ko} (ko = 한국어 이름/설명, 한글 입력 시 한국어 키워드 검색)
// 요청 수명주기(디바운스 취소·늦은 응답 버리기·팝업 주인)는 useTagAutocomplete 가 맡고,
// 여기서는 '커서가 놓인 태그 조각'(utils/tagSuggest.tagTokenAt)을 질의하고 그 조각만 바꾼다.
// 수락 표기는 예전 그대로 원문(밑줄) — 블록 모드(TagBlockField)는 공백 표기다.
const ac = useTagAutocomplete({ delay: 300 })
const acItems = ac.items
const acIdx = ac.index
/**
 * 팝업(또는 응답 대기 중인 요청)이 가리키는 입력 요소와, 후보를 요청한 순간의 텍스트·커서 조각.
 * 수락은 커서가 **그 조각 그대로**일 때만 그 조각을 바꾼다(utils/tagSuggest.isSameTagQuery).
 */
let acField: { fieldId: string; el: HTMLInputElement | HTMLTextAreaElement; query: TagQuery } | null = null
/** 방금 수락으로 써 넣은 값 — 그 값으로 들어오는 input(IME 확정 등)은 새 검색을 띄우지 않는다 */
let acAcceptedText: string | null = null

function closeFieldAc() {
  ac.close()
  acField = null
}
watch(tagBlockMode, closeFieldAc)

function onFieldInput(e: Event, fieldId: string) {
  const el = e.target as HTMLInputElement | HTMLTextAreaElement | null
  if (!el) return
  // v-model 은 IME 조합 중 값을 갱신하지 않는다 — 요소 값을 직접 읽는다
  const text = el.value
  if (acAcceptedText !== null && text === acAcceptedText) { acAcceptedText = null; return }
  acAcceptedText = null
  const query = tagQueryAt(text, el.selectionStart)
  acField = { fieldId, el, query }
  ac.request(query.query, fieldId)
}
/** 떠 있는 후보가 아직 지금 텍스트·커서 조각의 것인가 */
function fieldAcStillValid(fieldId: string): boolean {
  const target = acField
  return !!target && target.fieldId === fieldId
    && isSameTagQuery(target.query, target.el.value, target.el.selectionStart)
}
function onFieldKey(e: KeyboardEvent, fieldId: string) {
  if (isImeComposing(e)) return          // 조합 확정 키는 후보 선택이 아니다
  if (!ac.isOpenFor(fieldId)) {
    // 응답을 기다리는 중(디바운스 300ms + 브리지 왕복)에 키보드로 커서만 옮기면 그 요청은 옛
    // 조각의 것이다 — 도착해 팝업이 뜨면 Enter/Tab 이 엉뚱한 태그를 바꾼다. 지금 무효로 만든다.
    if (acField?.fieldId === fieldId && isCaretMoveKey(e.key)) closeFieldAc()
    return
  }
  switch (e.key) {
    case 'ArrowDown': e.preventDefault(); ac.move(1); return
    case 'ArrowUp': e.preventDefault(); ac.move(-1); return
    case 'Tab':
    case 'Enter':
      // 한글 검색 팝업에서는 Enter 를 평소대로(줄바꿈) 두고 팝업만 닫는다 — Tab/클릭으로 선택
      if (e.key === 'Enter' && ac.queryHangul.value) { closeFieldAc(); return }
      // 요청 뒤 커서·값이 바뀌었으면(Undo·백엔드 갱신 등) 옛 조각의 후보다 — 키는 가로채지 않고
      // (Tab 은 포커스 이동, Enter 는 줄바꿈 그대로) 팝업만 닫는다
      if (!fieldAcStillValid(fieldId)) { closeFieldAc(); return }
      e.preventDefault(); acceptFieldSuggestion(ac.selected()?.tag, fieldId); return
    case 'Escape': closeFieldAc(); return
    default:
      // 커서가 다른 조각으로 옮겨 간다 — 옛 조각의 후보를 남겨 두면 엉뚱한 곳이 바뀐다
      if (isCaretMoveKey(e.key)) closeFieldAc()
  }
}
function acceptFieldSuggestion(tag: string | undefined, fieldId: string) {
  const target = acField
  closeFieldAc()
  if (!tag || !target || target.fieldId !== fieldId) return
  const el = target.el
  const text = el.value
  // 후보를 요청한 조각에서 커서가 떠났거나 값이 바뀌었으면 바꾸지 않는다(팝업만 닫힘)
  if (!isSameTagQuery(target.query, text, el.selectionStart)) return
  const next = replaceTagToken(text, tagTokenAt(text, el.selectionStart), tag)
  // IME 조합 중이면 v-model 이 DOM 값을 고치지 않는다(조합 보호) — 요소에도 직접 써 둔다.
  // 조합이 끝나며 오는 input 은 acAcceptedText 로 걸러 새 검색을 띄우지 않는다.
  if ((el as any).composing) el.value = next.text
  acAcceptedText = next.text
  widgets[fieldId] = next.text
  nextTick(() => {
    try {
      el.focus()
      el.setSelectionRange(next.caret, next.caret)
    } catch { /* 요소가 사라졌으면 무시 */ }
    if (el.tagName === 'TEXTAREA') autoGrow(el)
  })
}
function onMainInput(e: Event) { autoGrow(e.target); onFieldInput(e, 'main_prompt_text') }
function onAutoKey(e: KeyboardEvent) { onFieldKey(e, 'main_prompt_text') }

function autoGrow(el: any) { if (!el) return; el.style.height = 'auto'; el.style.height = el.scrollHeight + 'px' }
function growAll() { nextTick(() => { ;[totalPromptRef, negRef, artistRef, prefixRef, mainRef, suffixRef].forEach(r => { if (r.value) autoGrow(r.value) }) }) }

onMounted(() => {
  setTimeout(growAll, 500); setTimeout(growAll, 1500)
  // 블록 모드의 재시작 복원·localStorage 미러는 App.vue uiPrefsLoaded 한 곳이 맡는다(composables/uiPrefs).
  _backendUnsubs.push(onBackendEvent('ollamaResult', (json: string) => {
    ollamaLoading.value = false
    if (ollamaTimer) clearTimeout(ollamaTimer)
    try {
      const d = JSON.parse(json)
      if (d.error) {
        const msg = d.error.includes('연결') ? 'Ollama 서버에 연결할 수 없습니다' :
                    d.error.includes('시간') || d.error.includes('Timeout') ? 'AI 응답 시간 초과' :
                    `AI 오류: ${d.error}`
        requestAction('show_toast', { type: 'error', msg })
        return
      }
      if (d.tags) {
        const NL = ['nl_caption', 'nl_scene', 'translate', 'creative']
        if (NL.includes(d.mode)) {
          nlResult.value = d.tags
          // 추천 해상도 파싱 (창의 모드)
          nlRes.value = null
          const rm = d.tags.match(/Resolution:\s*(\d{3,4})\s*[x×]\s*(\d{3,4})/i)
          if (rm) nlRes.value = { w: rm[1], h: rm[2] }
          const label = d.mode === 'translate' ? '번역' : d.mode === 'nl_caption' ? '캡션'
                      : d.mode === 'creative' ? '창의 생성' : '장면묘사'
          requestAction('show_toast', { type: 'success', msg: `AI ${label} 완료` })
        } else if (d.mode === 'negative') {
          const existing = (widgets.neg_prompt_text || '').trim()
          widgets.neg_prompt_text = existing ? existing.replace(/,?\s*$/, '') + ', ' + d.tags : d.tags
          requestAction('show_toast', { type: 'success', msg: 'AI 네거티브 생성 완료' })
        } else {
          widgets.main_prompt_text = d.tags
          showNlInput.value = false
          nlPrompt.value = ''
          requestAction('show_toast', { type: 'success', msg: `AI ${d.mode === 'nl2tags' ? '변환' : d.mode === 'suggest' ? '추천' : '확장'} 완료` })
        }
      }
    } catch {}
  }))
})

watch(() => widgets.total_prompt_display, () => nextTick(() => { if (totalPromptRef.value) autoGrow(totalPromptRef.value) }))
watch(() => widgets.neg_prompt_text, () => nextTick(() => { if (negRef.value) autoGrow(negRef.value) }))
watch(() => widgets.artist_input, () => nextTick(() => { if (artistRef.value) autoGrow(artistRef.value) }))
watch(() => widgets.main_prompt_text, () => { nextTick(() => { if (mainRef.value) autoGrow(mainRef.value) }); if (tagBlockMode.value) classifyVisibleTags() })
</script>

<style scoped>
.prompt-panel { display: flex; flex-direction: column; }
summary { list-style: none; outline: none; }
summary::-webkit-details-marker { display: none; }
.input-group { margin-bottom: 10px; }
.row { display: flex; gap: 6px; }
.label-row { align-items: center; margin-bottom: 4px; }
.small-btn { height: 28px; padding: 0 12px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-base); color: var(--text-secondary); font-size: var(--fs-meta); font-weight: var(--fw-bold); cursor: pointer; white-space: nowrap; }
.lock-btn { width: 28px; height: 28px; display: inline-flex; align-items: center; justify-content: center; background: none; border: none; cursor: pointer; padding: 0; opacity: 0.55; }
.lock-btn.locked { opacity: 1; }
.total-prompt { min-height: 60px; font-family: 'Consolas', monospace; font-size: 12px; line-height: 1.5; color: var(--accent); border-color: var(--accent-dim); }
.neg-section { margin-top: 8px; }
.neg-toggle { cursor: pointer; list-style: none; display: flex; align-items: center; justify-content: space-between; }
.neg-toggle::-webkit-details-marker { display: none; }
.neg-ai { font-size: 12px !important; padding: 2px 6px !important; opacity: 0.6; }
.neg-ai:hover { opacity: 1; }
.danger-label { color: var(--state-alert-fg); font-size: var(--fs-label); font-weight: var(--fw-bold); letter-spacing: 0; }
.neg-prompt { min-height: 30px; color: var(--state-alert-fg); border-color: rgba(248,113,113,0.2); }
.auto-grow { resize: none; overflow: hidden; min-height: 32px; }
.token-info { font-size: var(--fs-label); color: var(--text-muted); }
.prompt-actions { display: flex; align-items: center; gap: 8px; margin-top: 4px; }
.optimize-btn { height: 28px; padding: 0 12px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-base); color: var(--text-secondary); font-size: var(--fs-meta); font-weight: var(--fw-bold); cursor: pointer; display: inline-flex; align-items: center; gap: 5px; }
.optimize-btn:hover { border-color: var(--accent); color: var(--accent); }
.opt-result { font-size: var(--fs-label); color: var(--state-ok-fg); }
.opt-preview { margin-top: 6px; background: var(--bg-secondary); border: 1px solid var(--accent); border-radius: 8px; padding: 9px; }
.opt-prev-head { display: flex; justify-content: space-between; align-items: center; font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--accent); margin-bottom: 7px; }
.opt-prev-stat { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-muted); }
.opt-prev-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.opt-prev-col label { display: block; font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-muted); margin-bottom: 3px; letter-spacing: 0; }
.opt-prev-text { max-height: 110px; overflow-y: auto; background: var(--bg-input); border: 1px solid var(--border); border-radius: 5px; padding: 6px 8px; font-size: var(--fs-label); line-height: 1.4; color: var(--text-secondary); word-break: break-word; white-space: pre-wrap; }
.opt-prev-text.after { color: var(--text-primary); border-color: rgba(74,222,128,0.4); }
.opt-prev-conf { margin-top: 6px; font-size: var(--fs-label); color: var(--state-warn-fg); }
.opt-prev-btns { display: flex; gap: 6px; margin-top: 8px; }
.opt-apply { flex: 1; padding: 6px; background: var(--accent-fill); color: var(--on-accent); border: none; border-radius: 5px; font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer; }
.opt-apply:hover { background: var(--accent-fill-hover); }
.opt-cancel { padding: 6px 12px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 5px; color: var(--text-secondary); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer; }

/* ⑤ 카테고리 분리 패널 */
.separate-panel { margin-top: 6px; padding: 8px; background: var(--bg-primary); border: 1px solid var(--border); border-radius: 6px; }
.sep-chips { display: flex; flex-wrap: wrap; gap: 5px; }
.sep-chip { display: inline-flex; align-items: center; gap: 5px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 12px; color: var(--text-secondary); font-size: var(--fs-label); font-weight: var(--fw-bold); padding: 3px 11px; cursor: pointer; }
.sep-chip:hover { color: var(--text-primary); }
.sep-chip.on { background: var(--accent-dim); border-color: var(--accent); color: var(--accent); }
.sep-n { font-size: var(--fs-label); font-weight: var(--fw-bold); padding: 0 5px; border-radius: 7px; background: rgba(0,0,0,0.3); }
.sep-actions { display: flex; align-items: center; gap: 6px; margin-top: 7px; }
.sep-act { border: none; border-radius: 5px; font-size: var(--fs-label); font-weight: var(--fw-bold); padding: 4px 10px; cursor: pointer; }
.sep-act.remove { background: rgba(248,113,113,0.14); border: 1px solid var(--state-alert-fg); color: var(--state-alert-fg); }
.sep-act.extract { background: rgba(96,165,250,0.14); border: 1px solid var(--state-info-fg); color: var(--state-info-fg); }
.sep-hint { font-size: var(--fs-label); color: var(--state-ok-fg); }
.conflicts { margin-top: 4px; }
.conflict-item { font-size: var(--fs-label); color: var(--state-warn-fg); padding: 2px 0; }
/* 캐릭터 태그를 다루는 블록이라 태그 6색의 person(인물·캐릭터)을 쓴다.
   청록 틴트만 남기면 글자와 면의 색이 갈라져서 틴트도 같은 토큰에서 뽑는다 */
.char-insight { margin-top: 6px; background: color-mix(in srgb, var(--tag-person) 3%, transparent); border: 1px solid color-mix(in srgb, var(--tag-person) 10%, transparent); border-radius: 6px; padding: 8px; }
.insight-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
.insight-label { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--tag-person); }
.insight-apply { padding: 2px 8px; background: var(--tag-person); border: none; border-radius: 3px; color: var(--bg-primary); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer; }
.insight-tags { display: flex; flex-wrap: wrap; gap: 3px; max-height: 80px; overflow-y: auto; }
.char-tag-chip { padding: 2px 8px; background: color-mix(in srgb, var(--tag-person) 8%, transparent); border: 1px solid color-mix(in srgb, var(--tag-person) 20%, transparent); border-radius: 4px; color: var(--tag-person); font-size: var(--fs-label); cursor: pointer; }
.char-tag-chip:hover { background: color-mix(in srgb, var(--tag-person) 15%, transparent); border-color: var(--tag-person); }
.ai-btns { display: flex; gap: 3px; }
.ai-btn { width: 28px; height: 28px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 4px; color: var(--text-muted); font-size: 11px; cursor: pointer; display: flex; align-items: center; justify-content: center; }
.ai-btn:hover { border-color: var(--accent); color: var(--accent); }
.ai-btn:disabled { opacity: 0.3; }
.ai-btn.go { width: auto; padding: 0 8px; background: var(--accent-fill); color: var(--on-accent); border: none; font-weight: var(--fw-bold); font-size: var(--fs-label); }
/* 2개 LLM 드롭다운 (태그 AI / 자연어 AI) */
.ai-menu-wrap { position: relative; display: inline-block; }
.ai-menu-btn { width: auto; padding: 0 9px; font-size: var(--fs-label); font-weight: var(--fw-bold); white-space: nowrap; }
.ai-menu { position: absolute; top: 100%; right: 0; margin-top: 4px; z-index: 60; min-width: 180px; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 4px; box-shadow: 0 8px 24px rgba(0,0,0,0.5); display: flex; flex-direction: column; gap: 1px; }
.ai-menu button { width: 100%; height: auto; text-align: left; background: transparent; border: none; color: var(--text-secondary); font-size: 11px; font-weight: var(--fw-bold); padding: 7px 10px; border-radius: 5px; cursor: pointer; white-space: nowrap; display: flex; justify-content: space-between; align-items: center; gap: 10px; }
.ai-menu button:hover { background: var(--bg-button); color: var(--accent); }
.ai-menu button em { font-style: normal; font-weight: var(--fw-normal); font-size: var(--fs-label); color: var(--text-muted); }
.ai-menu-backdrop { position: fixed; inset: 0; z-index: 55; }
.nl-input-row { display: flex; gap: 4px; margin-bottom: 6px; }
.nl-input { flex: 1; padding: 6px 10px; font-size: 11px; }
.ai-loading { font-size: var(--fs-label); color: var(--accent); margin-bottom: 4px; animation: pulse 1.5s infinite; }
.nl-result { margin: 4px 0; }
.nl-result-text { width: 100%; background: var(--bg-input); border: 1px solid var(--accent); border-radius: 6px; padding: 7px 9px; color: var(--text-primary); font-size: 12px; resize: vertical; line-height: 1.5; font-family: inherit; }
.nl-result-btns { display: flex; gap: 4px; margin-top: 4px; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
.autocomplete-wrap { position: relative; }
.ac-popup { position: absolute; left: 0; right: 0; top: 100%; z-index: 100; background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; max-height: 200px; overflow-y: auto; box-shadow: 0 8px 24px rgba(0,0,0,0.6); }
.ac-item { padding: 6px 12px; font-size: 11px; color: var(--text-secondary); cursor: pointer; border-bottom: 1px solid rgba(255,255,255,0.03); }
.ac-item:hover, .ac-item.selected { background: var(--accent-dim); color: var(--accent); }
.ac-item .ac-ko { margin-left: 8px; font-size: 10px; color: var(--text-muted); }
.ac-item.selected .ac-ko { color: inherit; opacity: .8; }
label.danger { color: var(--state-alert-fg); }
.exclude-section { margin-bottom: 0; }
.exclude-toggle { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--state-alert-fg); letter-spacing: 0; cursor: pointer; list-style: none; }
.excl-badge { display: inline-block; min-width: 14px; padding: 0 5px; border-radius: 7px; background: rgba(248,113,113,0.2); color: var(--state-alert-fg); font-size: var(--fs-label); font-weight: var(--fw-bold); text-align: center; }
.exclude-toggle::-webkit-details-marker { display: none; }
.exclude-help { display: flex; flex-direction: column; gap: 2px; margin: 6px 0; padding: 6px 8px; background: rgba(248,113,113,0.03); border: 1px solid rgba(248,113,113,0.1); border-radius: 4px; }
.exclude-help span { font-size: var(--fs-label); color: var(--text-muted); font-family: 'Consolas', monospace; }
.exclude-textarea { color: var(--state-alert-fg); border-color: rgba(248,113,113,0.2); font-size: 11px; }
.excl-mgr-btn { height: 24px; padding: 0 10px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-base); color: var(--text-secondary); font-size: var(--fs-label); cursor: pointer; margin-left: 8px; }
.excl-mgr-btn:hover { border-color: var(--state-alert-fg); color: var(--state-alert-fg); }

/* Exclude Manager Modal */
.em-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 3000; display: flex; align-items: center; justify-content: center; }
.em-modal { width: min(94vw, 1080px); height: min(88vh, 760px); background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px; display: flex; flex-direction: column; overflow: hidden; }
.em-header { display: flex; align-items: center; gap: 10px; padding: 12px 16px; border-bottom: 1px solid var(--border); }
.em-header h3 { font-size: 12px; letter-spacing: 0; color: var(--state-alert-fg); }
.em-desc { font-size: var(--fs-label); color: var(--text-muted); flex: 1; }
.em-body { flex: 1; display: flex; overflow: hidden; }
.em-rules { width: 280px; overflow-y: auto; border-right: 1px solid var(--border); padding: 8px; }
.em-search { width: 100%; box-sizing: border-box; padding: 6px 10px; margin-bottom: 8px; font-size: 11px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px; color: var(--text-primary); position: sticky; top: -8px; z-index: 1; }
.em-search:focus { outline: none; border-color: var(--state-alert-fg); }
.em-rule-item { display: flex; justify-content: space-between; align-items: center; padding: 6px 10px; font-size: 11px; cursor: pointer; border-radius: 4px; margin-bottom: 2px; border: 1px solid transparent; }
.em-rule-item:hover { background: var(--bg-input); }
.em-rule-item.active { border-color: var(--state-alert-fg); background: rgba(248,113,113,0.05); }
.em-rule-text { font-family: 'Consolas', monospace; }
.em-match-count { font-size: var(--fs-label); color: var(--text-muted); background: var(--bg-button); padding: 1px 6px; border-radius: 8px; }
.em-matches { flex: 1; overflow-y: auto; padding: 12px; }
.em-match-header { font-size: var(--fs-label); color: var(--text-muted); margin-bottom: 8px; }
.em-match-list { display: flex; flex-wrap: wrap; gap: 4px; }
.em-tag { padding: 3px 10px; background: rgba(248,113,113,0.05); border: 1px solid rgba(248,113,113,0.2); border-radius: 4px; color: var(--state-alert-fg); font-size: var(--fs-label); cursor: pointer; transition: all 0.12s; }
.em-tag:hover { border-color: rgba(248,113,113,0.5); }
.em-tag.excepted { background: rgba(74,222,128,0.1); border-color: rgba(74,222,128,0.3); color: var(--state-ok-fg); text-decoration: line-through; opacity: 0.6; }
.em-tag.excepted:hover { opacity: 1; }
.em-rule-rm { background: none; border: none; color: var(--state-alert-fg); cursor: pointer; font-size: 11px; flex-shrink: 0; }
.em-rule-edit { flex: 1; padding: 2px 6px; font-size: 11px; background: var(--bg-card); border: 1px solid var(--accent); border-radius: 3px; color: var(--text-primary); font-family: 'Consolas', monospace; }
.em-rule-text { cursor: default; }
.em-add-row { display: flex; gap: 4px; margin-top: 6px; padding-top: 6px; border-top: 1px solid var(--border); }
.em-add-input { flex: 1; padding: 4px 8px; font-size: var(--fs-label); background: var(--bg-input); border: 1px solid var(--border); border-radius: 3px; color: var(--text-primary); }
.em-add-btn { width: 28px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 3px; color: var(--accent); font-weight: var(--fw-bold); cursor: pointer; }
.em-empty-sm { padding: 8px; text-align: center; color: var(--text-muted); font-size: var(--fs-label); }
.em-empty { display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-muted); font-size: 12px; }
.fade-enter-active, .fade-leave-active { transition: opacity 0.2s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
input[type="number"] { -moz-appearance: textfield; }
input::-webkit-outer-spin-button, input::-webkit-inner-spin-button { -webkit-appearance: none; }
/* neg block field */
.neg :deep(.tbf) { border-color: rgba(248,113,113,0.15); }
.neg :deep(.tbf-block) { border-color: rgba(248,113,113,0.2); color: var(--state-alert-fg); font-size: var(--fs-label); }
.hint { font-size: var(--fs-label); color: var(--text-muted); font-weight: normal; margin-left: 4px; }
.generation-family { padding-bottom: 10px; border-bottom: 1px solid var(--border); }
.engine-hint { display: block; margin-top: 5px; font-size: var(--fs-label); color: var(--text-muted); }
.krea-engine-note {
  display: flex; flex-direction: column; gap: 4px; margin-bottom: 10px; padding: 10px;
  border: 1px solid rgba(167,139,250,0.35); border-radius: 7px;
  background: linear-gradient(135deg, rgba(124,58,237,0.12), rgba(45,212,191,0.05));
}
.krea-engine-note strong { font-size: 11px; letter-spacing: 0; color: var(--state-info-fg); }
.krea-engine-note span { font-size: var(--fs-label); line-height: 1.45; color: var(--text-muted); }
.tk-badge {
  display: inline-block; padding: 1px 6px; margin-left: 6px; font-size: var(--fs-label);
  font-weight: var(--fw-bold); border-radius: 8px; vertical-align: middle;
  border: 1px solid transparent;
}
.tk-badge.tk-ok    { background: rgba(74,222,128,0.12); color: var(--state-ok-fg); border-color: rgba(74,222,128,0.3); }
.tk-badge.tk-warn  { background: rgba(251,191,36,0.12); color: var(--state-warn-fg); border-color: rgba(251,191,36,0.4); }
.tk-badge.tk-over  { background: rgba(248,113,113,0.18); color: var(--state-alert-fg); border-color: rgba(248,113,113,0.5); }

/* Undo/Redo 버튼 */
.undo-btns { float: right; display: inline-flex; gap: 4px; margin-left: auto; }
.undo-btn {
  width: 28px; height: 28px; padding: 0; font-size: var(--fs-body); cursor: pointer;
  background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
  border-radius: 4px; color: var(--text-primary);
}
.undo-btn:hover:not(:disabled) { background: rgba(96,165,250,0.18); color: var(--state-info-fg); border-color: var(--state-info-fg); }
.undo-btn:disabled { opacity: 0.3; cursor: not-allowed; }
summary.card-header { display: flex; align-items: center; }
.te-chips { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
.te-chip {
  font-size: var(--fs-label);
  padding: 3px 8px;
  border-radius: 10px;
  border: 1px solid rgba(255,255,255,0.15);
  background: rgba(255,255,255,0.04);
  color: rgba(255,255,255,0.7);
  cursor: pointer;
  transition: all 0.15s;
}
.te-chip:hover { border-color: rgba(96,165,250,0.5); color: var(--text-primary); }
.te-chip.active {
  background: rgba(96,165,250,0.2);
  border-color: var(--state-info-fg);
  color: var(--state-info-fg);
}
.te-chip-x { margin-left: 4px; opacity: 0.6; }
.te-chip:hover .te-chip-x { opacity: 1; color: var(--state-alert-fg); }
</style>
