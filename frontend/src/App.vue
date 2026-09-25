<template>
  <div class="app-container">
    <!-- 시작 백엔드 게이트 — 앱 전체를 덮는 불투명 화면(z-index 900 > 레일 100).
         예전의 별도 QDialog 를 대신하므로, 시작할 때 사용자가 보는 창은 하나뿐이다.
         맨 위에 두는 이유: 이걸 통과하기 전엔 뒤의 레일·패널이 쓸 수 없는 상태다. -->
    <BackendGate
      :open="gateOpen"
      :webui-url="gateWebuiUrl"
      :comfy-url="gateComfyUrl"
      :workflow-path="gateWorkflowPath"
      :probe="gateProbe"
      :workflow-info="gateWorkflowInfo"
      :busy="gateBusy"
      :error="gateError"
      :dismissible="gateDismissible"
      @probe="onGateProbe"
      @select="onGateSelect"
      @pick-workflow="onGatePickWorkflow"
      @dismiss="onGateDismiss"
    />

    <NavRail @tab-changed="onTabChanged" />

    <main class="main-workspace" :data-tab="activeTabName">
      <!-- Left Panel -->
      <aside class="side-panel left" v-show="showLeftPanel">
        <div class="panel-scroll" v-show="panelMode === 'prompt'" v-scroll-memory="'leftPanel'">
          <PromptPanel @open-wildcard="openWildcardByName" />
          <!-- 워크플로우 프로파일 -->
          <div class="tool-card profile-card">
            <div class="profile-row">
              <label class="profile-label">프로파일</label>
              <CustomSelect :modelValue="''" @update:modelValue="loadWorkflowProfile"
                :options="profileNames"
                :placeholder="profileNames.length === 0 ? '(저장된 프로파일 없음)' : '선택하여 적용...'" />
              <button class="profile-mini-btn" @click="saveCurrentAsProfile" title="현재 세팅을 새 프로파일로 저장">+</button>
              <button class="profile-mini-btn" @click="openProfileManager" title="프로파일 관리 (삭제/이름변경)"><Icon name="settings" /></button>
            </div>
          </div>

          <!-- 스튜디오 도구 — 매니저 모달의 상태·동작은 composables/use*Manager 등(모듈 싱글턴), 모달은 표시만 -->
          <div class="tool-card">
            <label>스튜디오 도구</label>
            <div class="tool-grid">
              <button class="tool-btn" @click="syncLoraStack(); action('save_settings')">저장</button>
              <button class="tool-btn" @click="openPresetManager">프리셋</button>
              <button class="tool-btn" @click="openWeightManager">가중치</button>
              <button class="tool-btn" @click="openWcManager">와일드카드</button>
              <button class="tool-btn" @click="openInstantWcManager" title="JSON 기반 인라인 와일드카드 ($$name$$)">즉석 WC</button>
              <button class="tool-btn" @click="openOrderManager" title="최종 프롬프트의 섹션 순서를 직접 지정">순서</button>
              <button class="tool-btn" @click="openAbTestModal()">A/B 테스트</button>
              <button class="tool-btn" @click="openStatsModal">통계</button>
              <button class="tool-btn" :class="{ 'tool-btn-on': condEnabled }" @click="showCondModal = true" :title="`태그 조건부 프롬프트 (IF→THEN) 관리 — 현재 ${condEnabled ? 'ON' : 'OFF'}`">조건부<span v-if="condEnabled" class="tool-dot"></span></button>
            </div>
          </div>

        </div>
        <!-- 파라미터 — 프롬프트와 같은 열을 번갈아 쓴다. 예전엔 오른쪽으로 열리는
             오버레이였다: 값을 바꾸며 결과를 보는 데는 좋았지만, 왼쪽 열 옆에 열이
             하나 더 서서 무대를 가렸다. v-show 라 `#sec-params` 는 늘 DOM 에 있다. -->
        <div class="extend-overlay" v-show="panelMode === 'params'">
          <div class="extend-header">
            <h3>파라미터</h3>
            <button class="close-btn" @click="showExtendPanel = false" title="프롬프트로 (ESC)"><Icon name="close" /></button>
          </div>
          <div class="extend-scroll" v-scroll-memory="'extendPanel'">
            <!-- 카드마다 components/params/*.vue — 위젯 키·클래스·문구는 예전 그대로 옮겼다(App.vue 분할 ④).
                 .ext-* 카드 모양은 styles/panels.css 의 `:where(.extend-overlay)` 규칙이 이 열 안에서만 입힌다. -->
            <ParamsBasicCard :high-res="highRes" :random-res="randomRes" />
            <HiresFixCard />
            <PromptFilterCard :rating="rating" />
            <AdetailerCard :model-items="adModelItems" />
            <Sam3MaskCard />

            <!-- Anima Guidance Suite — SAM3와 완전히 분리된 독립 기능.
                 인자 계약(위치 기반 62/7/13개)은 core/anima_guidance.py 참조. -->
            <AnimaGuidancePanel :widgets="storeWidgets" />

            <!-- NegPiP 상시 적용 / 조건부 프롬프트는 STUDIO TOOLS '조건부' 모달로 이동 -->

            <LoraStackCard :lora="lora" />
          </div>
        </div>
        <div class="gen-footer">
          <div class="gen-actions">
            <button class="action-btn" :class="{ active: autoMode }" @click="onAutoModeClick">
              <Icon :name="autoMode ? 'refresh' : 'pause'" /> {{ autoMode ? '자동 켬' : '자동 꺼짐' }}
            </button>
            <button class="action-btn highlight" @click="action('random_prompt')"><Icon name="dice" /> 무작위</button>
          </div>
          <!-- 자동화 — 멈춰 있으면 설정, 돌면 조종석. 한 자리를 두 모드가 번갈아 쓴다.
               (예전엔 설정·상태·덱이 서로 다른 조건으로 겹쳐 그려져 푸터가 두 배가 됐다) -->
          <AutomationPanel
            v-if="autoMode"
            :settings="autoSettings"
            :running="isAutomating"
            :paused="autoPaused"
            :count="autoGenCount"
            :waiting="autoWaiting"
            :wait-remaining-ms="waitRemainingMs"
            :wait-total-ms="waitTotalMs"
            :deck-total="deckTotal"
            :deck-remaining="deckRemaining"
            :deck-used="deckUsed"
            :deck-allow-dup="deckAllowDup"
            :next-prompt="autoNextPrompt"
            :prompt-is-next="autoPromptIsNext"
            @update:settings="applyAutoSettings"
            @reset-deck="resetDeck"
            @pause="pauseAutomation"
            @resume="resumeAutomation"
            @stop="stopAutomation"
            @override="overrideNextPrompt"
          />
          <label class="auto-nl-toggle" :class="{ on: autoNlGen }" title="생성 시 메인 프롬프트의 태그를 자연어 문장으로 자동 변환한 뒤 생성합니다 (Flux/SD3/NAI 등 자연어 모델용). Ollama 필요.">
            <ToggleSwitch v-model="autoNlGen" size="sm" />
            <span>생성 시 태그를 자연어로 변환</span>
          </label>
          <!-- 자동화 중엔 멈추기가 조종석 안에 있다 — 같은 일을 하는 버튼을 두 개 두지 않는다. -->
          <div class="generate-row" v-if="!isAutomating">
            <button class="btn-generate" :class="{ converting: nlConverting }" @click="doGenerate" :disabled="isGenerating || nlConverting">
              <Icon v-if="autoMode && !isGenerating && !nlConverting" name="play" />
              {{ nlConverting ? '자연어 변환 중…' : isGenerating ? '생성 중…' : autoMode ? '자동 생성 시작' : 'GENERATE IMAGE' }}
            </button>
            <button v-if="isGenerating" class="btn-cancel" @click="cancelGeneration" title="생성 취소"><Icon name="close" /></button>
          </div>
          <div class="gen-eta" v-if="isGenerating && genEta">{{ genEta }}</div>
        </div>
      </aside>


      <!-- Center: Viewport + EXIF Bar -->
      <section class="viewport-area">
        <div class="viewport-main">
          <router-view v-slot="{ Component, route }">
            <!-- 탭이 바뀌었다는 걸 무대도 말한다 — 레일 표시만 바뀌면 같은 화면으로 읽힌다.
                 **들어올 때만** 움직인다. 나가는 전환(out-in)을 두면 새 화면이 그 전환이
                 끝날 때까지 마운트되지 않는데, 창이 가려져 프레임이 멈추면 그 '끝'이
                 오지 않는다 — 화면이 통째로 비는 것보다 살짝 늦게 떠오르는 편이 낫다. -->
            <transition name="stage">
            <keep-alive>
              <component :is="Component"
                :key="route.name"
                :image-url="currentImage"
                :resolution="resolution"
                :seed="seed"
                :status="status"
                v-bind="route.name === 't2i' ? { generating: isGenerating, progress: progressVal, eta: genEta, previewUrl: livePreview } : {}"
              />
            </keep-alive>
            </transition>
          </router-view>
        </div>
        <!-- EXIF Info Bar (Positive / Negative / Parameters 3탭) -->
        <div class="exif-bar" v-if="showLeftPanel && currentImage">
          <div class="exif-tabs">
            <button v-for="tab in exifTabs" :key="tab.id" class="exif-tab"
              :class="{ active: activeExifTab === tab.id }" @click="activeExifTab = tab.id">
              {{ tab.label }}
            </button>
          </div>
          <div class="exif-content" v-if="activeExifTab !== 'params'">{{ exifContent }}</div>
          <div class="exif-params" v-else-if="currentExif.params">
            <div class="param-line" v-if="currentExif.params.generation"><span class="pl">생성</span>{{ currentExif.params.generation }}</div>
            <div class="param-line" v-if="currentExif.params.core"><span class="pl">기본</span>{{ currentExif.params.core }}</div>
            <div class="param-line" v-if="currentExif.params.model"><span class="pl">모델</span>{{ currentExif.params.model }}</div>
            <div class="param-line" v-if="currentExif.params.hires"><span class="pl">고해상도</span>{{ currentExif.params.hires }}</div>
            <div class="param-line" v-if="currentExif.params.extensions"><span class="pl">확장</span>{{ currentExif.params.extensions }}</div>
            <div class="param-line other" v-if="currentExif.params.other"><span class="pl">기타</span>{{ currentExif.params.other }}</div>
          </div>
          <div class="exif-content" v-else>{{ currentExif.params_line || currentExif.raw || 'No parameters' }}</div>
        </div>
      </section>

      <!-- Right: History -->
      <aside class="side-panel right" v-show="showLeftPanel">
        <div class="hist-header">
          <h3>히스토리</h3>
          <span class="count-badge">{{ historyImages.length }}</span>
        </div>
        <button class="hist-nav-btn" @click="histPage = Math.max(0, histPage - 1)" :disabled="histPage <= 0"><Icon name="chevron-up" /></button>
        <div class="hist-scroll" v-scroll-memory="'history'">
          <div v-for="img in visibleHistory" :key="img" class="hist-card"
            @click="selectHistoryImage(img)"
            @contextmenu.prevent="showHistoryMenu($event, img)"
            :class="{ selected: currentImage === img, blink: historyBlink && currentImage === img }"
            draggable="true" @dragstart="onDragStart($event, img)"
          >
            <img :key="historyImageSrc(img)" :src="historyImageSrc(img)" decoding="async" />
          </div>
        </div>
        <button class="hist-nav-btn" @click="histPage++" :disabled="(histPage + 1) * histPerPage >= historyImages.length"><Icon name="chevron-down" /></button>

        <transition name="pop">
          <div v-if="ctxMenu.show" class="modern-ctx-menu" :style="ctxMenuStyle">
            <div class="ctx-item" @click="ctxAddFavorite"><Icon name="star" /> 즐겨찾기 추가</div>
            <div class="ctx-item" @click="ctxSendI2I"><Icon name="image" /> I2I로 보내기</div>
            <div class="ctx-item" @click="ctxSendInpaint"><Icon name="palette" /> 인페인트로 보내기</div>
            <div class="ctx-item" @click="ctxSendEditor"><Icon name="pencil" /> 에디터로 보내기</div>
            <div class="ctx-item" @click="ctxCompare('before')"><Icon name="search" /> 비교 (이전)</div>
            <div class="ctx-item" @click="ctxCompare('after')"><Icon name="search" /> 비교 (이후)</div>
            <div class="ctx-item" @click="ctxRunAdetailer"><Icon name="target" /> ADetailer</div>
            <div class="ctx-separator"></div>
            <div class="ctx-item" @click="ctxPullPrompt"><Icon name="download" /> 프롬프트 당겨오기</div>
            <div class="ctx-item" @click="ctxAddToQueue"><Icon name="plus" /> 다음 큐에 추가</div>
            <div class="ctx-separator"></div>
            <div class="ctx-item" @click="ctxCopyPath"><Icon name="clipboard" /> 경로 복사</div>
            <div class="ctx-item delete" @click="ctxDelete"><Icon name="trash" /> 휴지통으로 이동</div>
          </div>
        </transition>
      </aside>
    </main>

    <div class="global-progress" v-if="isGenerating">
      <div class="progress-fill" :style="{ width: progressVal + '%' }"></div>
    </div>

    <QueuePanel />

    <!-- 하단 계기 스트립 (백엔드 · VRAM · 모델) — 값이 없어도 줄은 남는다.
         예전 VRAM 바는 `v-if="vramInfo.total > 0"` 이라 백엔드가 없으면 통째로
         사라졌고, 그래서 개발 서버에서는 아무것도 안 보였다. -->
    <StatusStrip :backend="backendStatus" :vram="vramInfo" :vram-level="vramClass"
      :vram-tooltip="vramTooltip" @vram-click="onVramClick" />

    <!-- 스튜디오 도구 매니저 모달 7종 — components/managers/*.vue 는 표시만 한다. 상태와 백엔드 리스너는
         composables(모듈 싱글턴)에 있어 닫혀 있는 동안에도 목록·가중치·프로파일을 받는다.
         `<transition name="fade">` 은 여기 두어 예전과 같은 fade 규칙(이 파일의 scoped)이 모달 루트에 걸린다. -->
    <transition name="fade">
      <PresetManagerModal v-if="showPresetManager" />
    </transition>

    <!-- 캐릭터 특징 프리셋 / A/B 테스트 / override 모달 (Vue) -->
    <CharacterPresetModal v-if="uiModals.charPreset" @close="closeCharPresetModal" />
    <ABTestModal v-if="uiModals.abTest" @close="closeAbTestModal" />
    <CharFeatureOverrideModal v-if="uiModals.charOverride" @close="closeCharOverrideModal" />
    <LoraManagerModal v-if="showLoraModal" @close="showLoraModal = false" @add="onLoraAdd" />
    <CondPromptModal v-if="showCondModal" @close="showCondModal = false" />

    <!-- 세션 복구 배너 (크래시/OOM 후) -->
    <!-- z-index 9999 라 게이트(900) 위로 뜬다. 시작 직후 1.2초 뒤에 나타나므로
         게이트와 겹치기 딱 좋은 타이밍 — 백엔드를 고른 뒤에 묻는 게 순서다. -->
    <div v-if="sessionRestore && !gateOpen" class="session-restore">
      <span class="sr-msg"><Icon name="history" /> 이전 세션의 작업이 남아 있습니다. 복원할까요?</span>
      <button class="sr-apply" @click="applySessionRestore">복원</button>
      <button class="sr-dismiss" @click="dismissSessionRestore">닫기</button>
    </div>

    <transition name="fade">
      <WeightManagerModal v-if="showWeightManager" />
    </transition>
    <transition name="fade">
      <WildcardManagerModal v-if="showWcManager" />
    </transition>
    <transition name="fade">
      <WorkflowProfileModal v-if="showProfileManager" />
    </transition>
    <transition name="fade">
      <PromptOrderModal v-if="showOrderManager" />
    </transition>
    <transition name="fade">
      <InstantWildcardModal v-if="showInstantWcManager" />
    </transition>
    <transition name="fade">
      <GenStatsModal v-if="showStatsModal" />
    </transition>

    <!-- 알림 벨 + 기록 패널(components/NotificationCenter.vue) — 벨은 게이트 동안 숨긴다(z-index 2003 이라 게이트를 뚫고 뜬다) -->
    <NotificationCenter :show-bell="!gateOpen" />

    <AppTooltip />

    <!-- 전역 토스트(components/ToastLayer.vue, body 로 Teleport) — 목록은 composables/useToasts.ts -->
    <ToastLayer />
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, watch, onMounted, nextTick } from 'vue'
import { initBridge, onBackendEvent, getBackend } from './bridge.js'
import { requestAction, useWidgetStore } from './stores/widgetStore.js'
import { initialiseAppUpdates } from './stores/appUpdateStore'
import { mediaUrl } from './utils/media.js'
import { bumpMediaVersion, recordListedMediaVersions } from './utils/mediaVersions'
import { useLoraStack } from './composables/useLoraStack.js'
import { useHighRes } from './composables/useHighRes.js'
import { useRatingFilter } from './composables/useRatingFilter.js'
import { useRandomResolutions } from './composables/useRandomResolutions'
import { useToasts } from './composables/useToasts'
import { usePresetManager } from './composables/usePresetManager'
import { useGlobalWeights } from './composables/useGlobalWeights'
import { useWildcardManager } from './composables/useWildcardManager'
import { useWorkflowProfiles } from './composables/useWorkflowProfiles'
import { usePromptOrder } from './composables/usePromptOrder'
import { useInstantWildcards } from './composables/useInstantWildcards'
import { useGenStats } from './composables/useGenStats'
import { vScrollMemory } from './directives/vScrollMemory'
import { appModalStack } from './utils/modalStack'
import { createAppKeydownHandler } from './utils/appShortcuts'
import { formatResolution, formatSeed } from './utils/imageInfo'
import { vramLevel, vramTooltipText, type VramInfo } from './utils/vramStatus'
import { formatEta, previewDataUrl, progressPercent } from './utils/generationProgress'
import { exifFromPayload, exifTabContent, type ExifData } from './utils/exifPayload'
import { edgeHistoryIndex, historyPageOf, historyPageSlice, stepHistoryIndex } from './utils/historyNav'
import { normalizeUiScale } from './utils/uiScale'
import { useHistoryThumbs, normalizePreviewThumbWidth } from './composables/useHistoryThumbs'
import { useBackendGate } from './composables/useBackendGate'
import { reconcileTheme } from './theme/applyTheme'
import { resolveInstalledModel, storedOllamaModel, storedOllamaUrl } from './utils/ollamaPrefs'
import {
  AUTOMATION_KEYS, automationPatchFromServer, automationSyncPayload, hydrateAutomationSettings, markAutomationKeys,
  type AutomationKey,
} from './utils/automationSettings'
import { mirrorPrefsToStorage } from './utils/uiPrefMirror'
import { persistUiPrefs, restoreUiFlagsFromPrefs } from './composables/uiPrefs'
import { useSessionRestore } from './composables/useSessionRestore'
import { createPendingRequest } from './utils/pendingRequest'
import { applyDeleteToHistory, parseImageDeleteResult } from './utils/imageDeleteResult'
import type { ActionName, ActionPayload, AutomationSettings, AutomationStatusEvent } from './types/bridge'

const wStore = useWidgetStore()
const storeWidgets = wStore.widgets
// 전역 토스트 — composables/useToasts.ts(모듈 싱글턴). 화면은 ToastLayer · NotificationCenter 가 그린다.
// composable 들(useLoraStack · useSessionRestore …)에도 이 addToast 를 주입한다.
const { addToast } = useToasts()
// ADetailer 모델 목록 — adetailerModelsReady(부팅 때 requestADetailerModels 로 요청)가 채우고 AdetailerCard 가 그린다
const adModelItems = ref<string[]>([])

// ── 파라미터 열(components/params/*) — 상태의 주인은 App 이다: 부팅 복원(uiPrefsLoaded)·로드 시점을
//    여기서 정하고, 카드는 composable 결과 객체를 prop 으로 받아 그린다(App.vue 분할 ④).
// 고해상도(단일 패스) 모드 — composables/useHighRes.js
const highRes = useHighRes({ storeWidgets, saveUiPrefs })
const { restoreFromPrefs: restoreHighResFromPrefs } = highRes
// 랜덤 해상도 — composables/useRandomResolutions.ts (목록은 onMounted 에서 받는다)
const randomRes = useRandomResolutions({ storeWidgets, getBackend, requestAction })

// 공유 헬퍼 — 여러 composable이 주입받아 사용(함수 선언이라 hoisting으로 위에서도 호출됨).
// 파일(save_ui_prefs)과 localStorage 캐시(utils/uiPrefMirror 표)를 한 경로로 쓴다. 즉시 전송 —
// 고빈도 발생원(LoRA·고해상도 슬라이더)은 각 composable 이 스스로 디바운스한다.
function saveUiPrefs(payload: any) {
  persistUiPrefs(payload)
}
// Rating 필터 — composables/useRatingFilter.js (PromptFilterCard 가 그린다)
const rating = useRatingFilter({ saveUiPrefs })
const { restoreFromPrefs: restoreRatingFromPrefs } = rating

// 시작 백엔드 게이트 — composables/useBackendGate.ts (App.vue 분할 ④)
// 템플릿이 쓰는 이름은 하나도 빠짐없이 여기서 꺼내야 한다(누락 시 빌드는 통과해도 런타임에 깨짐 —
// App.templateBindings.test.ts 가 정적으로 막는다).
const {
  gateOpen, gateWebuiUrl, gateComfyUrl, gateWorkflowPath,
  gateProbe, gateWorkflowInfo, gateBusy, gateError, gateDismissible,
  bindBackendGate,
  onGateProbe, onGateSelect, onGatePickWorkflow, onGateDismiss,
} = useBackendGate()

import PromptPanel from './components/PromptPanel.vue'
import CustomSelect from './components/CustomSelect.vue'
import NavRail from './components/NavRail.vue'
import BackendGate from './components/BackendGate.vue'
import { viewMode, setViewMode } from './composables/useViewMode'
import QueuePanel from './components/QueuePanel.vue'
import StatusStrip from './components/StatusStrip.vue'
import CharacterPresetModal from './components/CharacterPresetModal.vue'
import ABTestModal from './components/ABTestModal.vue'
import CharFeatureOverrideModal from './components/CharFeatureOverrideModal.vue'
import LoraManagerModal from './components/LoraManagerModal.vue'
import CondPromptModal from './components/CondPromptModal.vue'
import ToggleSwitch from './components/ToggleSwitch.vue'
import AutomationPanel from './components/AutomationPanel.vue'
import AnimaGuidancePanel from './components/AnimaGuidancePanel.vue'
import AppTooltip from './components/AppTooltip.vue'
import NotificationCenter from './components/NotificationCenter.vue'
import ToastLayer from './components/ToastLayer.vue'
import ParamsBasicCard from './components/params/ParamsBasicCard.vue'
import HiresFixCard from './components/params/HiresFixCard.vue'
import PromptFilterCard from './components/params/PromptFilterCard.vue'
import AdetailerCard from './components/params/AdetailerCard.vue'
import Sam3MaskCard from './components/params/Sam3MaskCard.vue'
import LoraStackCard from './components/params/LoraStackCard.vue'
import PresetManagerModal from './components/managers/PresetManagerModal.vue'
import WeightManagerModal from './components/managers/WeightManagerModal.vue'
import WildcardManagerModal from './components/managers/WildcardManagerModal.vue'
import WorkflowProfileModal from './components/managers/WorkflowProfileModal.vue'
import PromptOrderModal from './components/managers/PromptOrderModal.vue'
import InstantWildcardModal from './components/managers/InstantWildcardModal.vue'
import GenStatsModal from './components/managers/GenStatsModal.vue'
import { loadCondRules, condEnabled } from './composables/condRules.js'
import { clampMenuPosition, menuPositionStyle } from './utils/ctxMenuPosition'
import { uiModals, closeCharPresetModal, openAbTestModal, closeAbTestModal,
         closeCharOverrideModal } from './composables/uiModals.js'

const currentImage = ref('')
const imageVersions = reactive<Record<string, number>>({})
// HISTORY 선택 이미지 테두리 깜빡임 (Settings에서 on/off)
const historyBlink = ref(window.localStorage.getItem('historyBlinkSelected') !== 'false')
const resolution = ref('')
const seed = ref('')
const status = ref('')
const isGenerating = ref(false)
const progressVal = ref(0)
// 생성 중 중간 그림(data URL) — Forge live preview. 그림을 보고 있어도 생성 중임이 보이게 뷰어가 이걸 띄운다.
const livePreview = ref('')
const genStartTime = ref(0)
const genEta = ref('')
const showLeftPanel = ref(true)

// 스크롤 위치 기억(v-scroll-memory) — directives/vScrollMemory.ts
/**
 * 왼쪽 열의 모드 — 프롬프트 / 파라미터. 레일 서랍이 정하고 여기선 같은 ref 를 본다.
 * `showExtendPanel` 은 옛 이름을 지키는 **쓰기 가능한 computed** — ESC · 닫기 버튼 같은
 * 기존 호출부가 `showExtendPanel.value = …` 그대로 산다.
 */
const panelMode = viewMode('panel')
const showExtendPanel = computed({
  get: () => panelMode.value === 'params',
  set: (open: boolean) => setViewMode('panel', open ? 'params' : 'prompt'),
})
const historyImages = ref<string[]>([])
const histPage = ref(0)
const histPerPage = 5
// 히스토리 카드 썸네일 — 목록 전체를 백엔드 캐시로 미리 만들어 위아래 페이지 넘김에 로딩이 없게 (useHistoryThumbs)
const {
  srcFor: historyThumbSrc, ensure: ensureHistoryThumbs, setWidth: setPreviewThumbWidth,
  invalidate: invalidateHistoryThumb,
  bind: bindHistoryThumbs,
} = useHistoryThumbs({ getBackend, onBackendEvent, mediaUrl })
// deep: 1 — 배열 교체뿐 아니라 unshift/splice(새 생성 결과) 같은 변이도 잡는다(deep:false 는 못 잡았다).
// 요소가 문자열이라 한 단계만 훑는다.
watch(historyImages, (list) => { void ensureHistoryThumbs(list) }, { deep: 1 })
function onPreviewThumbWidthChanged(e: Event) {
  setPreviewThumbWidth((e as CustomEvent).detail?.value, historyImages.value)
}

// EXIF
const activeExifTab = ref('positive')
const exifTabs = [
  { id: 'positive', label: 'Positive' },
  { id: 'negative', label: 'Negative' },
  { id: 'params', label: 'Parameters' },
]
const currentExif = ref<ExifData>({ prompt: '', negative: '', raw: '' })
const exifContent = computed(() => exifTabContent(currentExif.value, activeExifTab.value))

const autoMode = ref(false)
const isAutomating = ref(false)
function onAutoModeClick() {
  // 생성 중엔 여기서 막는다 — Python 도 거절하지만, 화면의 켬/꺼짐이 실제와 어긋나지 않게 왕복 자체를 안 한다
  if (isGenerating.value) { addToast('warning', '이미지 생성 중에는 자동화 모드를 바꿀 수 없습니다'); return }
  autoMode.value = !autoMode.value
  action('toggle_automation', { checked: autoMode.value })
}
// 생성 시 태그→자연어 자동 변환 토글
const autoNlGen = ref(window.localStorage.getItem('autoNlGen') === 'true')
const nlConverting = ref(false)
const _lastAutoNl = ref('')   // 마지막 변환 결과(NL→NL 재변환 방지)
// 변환 한 건의 대기 — 자기 타이머를 소유하고 끝날 때 해제한다(utils/pendingRequest)
const _nlGen = createPendingRequest<string>({
  timeoutMs: 65000,
  onPendingChange: (pending) => { nlConverting.value = pending },
  onTimeout: () => requestAction('show_toast', { type: 'error', msg: 'AI 자연어 변환 시간 초과 — 태그 그대로 생성' }),
})
watch(autoNlGen, (v) => {
  saveUiPrefs({ autoNlGen: v })   // 파일 + localStorage 캐시(uiPrefMirror)
  syncAutomationSettings()   // 자동화 백엔드에도 즉시 반영 (실행 중 토글 대비)
})
const autoGenCount = ref(0)
const autoWaiting = ref(false)
// 일시정지는 '멈추기'와 다르다 — 덱 진행을 유지한 채 대기/생성 사이에서만 선다.
const autoPaused = ref(false)
// 다음 생성에 나갈 프롬프트 전문. 덱·와일드카드·조건식이 매번 바꾸는데
// 지금까지는 화면에 나올 자리가 없었다 — 조종석이 이걸 태그 칩으로 그린다.
const autoNextPrompt = ref('')
// 그 프롬프트가 다음 덱 장에 그대로 쓰이는가 — false 면 생성 중인 프롬프트라 조종석이 편집을 막는다
const autoPromptIsNext = ref(true)
const deckRemaining = ref(0)
const deckTotal = ref(0)
const deckUsed = ref(0)
const deckAllowDup = ref(false)
function resetDeck() {
  requestAction('reset_prompt_deck')   // 백엔드: filtered_results에서 덱 가득 채우고 셔플 + 상태 재전송
  addToast('success', '덱을 초기화했습니다 (사용 0)')
}
// 자동화 대기 카운트다운 (다음 생성까지) — 진행 표시는 AutomationPanel 이 그린다.
const waitRemainingMs = ref(0)
const waitTotalMs = ref(0)

// 패널은 바뀐 항목만 올려보낸다. 여기서 합치면 기존 deep watch 가 그대로
// syncAutomationSettings 를 태워, 백엔드 계약(set_automation_settings)은 안 바뀐다.
// 고친 키는 기록한다 — 부팅 hydrate 응답이 늦게 와도 그 키는 되돌리지 않고, 동기화는 아는 키만 보낸다.
function applyAutoSettings(patch: Partial<typeof autoSettings>) {
  markAutomationKeys(_autoEditedKeys, patch)
  markAutomationKeys(_autoKnownKeys, patch)
  Object.assign(autoSettings, patch)
}
function pauseAutomation() {
  autoPaused.value = true   // 백엔드 status 를 기다리면 버튼이 한 박자 늦게 바뀐다
  action('pause_automation')
}
function resumeAutomation() {
  autoPaused.value = false
  action('resume_automation')
}
function stopAutomation() {
  action('stop_automation')
  isAutomating.value = false
  autoWaiting.value = false
  autoPaused.value = false
}
/** 다음 한 장에만 쓸 프롬프트. 빈 문자열이면 덮어쓰기 취소(원래 프롬프트로). */
function overrideNextPrompt(prompt: string) {
  requestAction('automation_override_next', { prompt })
}
// 하단 계기 스트립이 그리는 백엔드 상태. null = 아직 신호 없음(= '연결 안 됨').
// 파이썬이 backendStatus 를 짧게 몇 번 되풀이 보내므로 늦게 붙어도 곧 채워진다.
const backendStatus = ref<{ kind?: string; label?: string; url?: string; connected?: boolean; error?: string } | null>(null)
// source: 'nvml' | 'nvidia-smi' = GPU 전체(모든 프로세스, 작업 관리자와 같은 숫자) / 'backend' = 백엔드 자기 메모리만
const vramInfo = ref<VramInfo>({ used: 0, total: 0, pct: 0, source: '' })
const vramClass = computed(() => vramLevel(vramInfo.value.pct))
const vramTooltip = computed(() => vramTooltipText(vramInfo.value))   // utils/vramStatus
function onVramClick() {
  // 백엔드에 unload 요청 — 사용자가 명시적으로 메모리 정리하고 싶을 때
  if (vramClass.value === 'critical') {
    if (!confirm(`VRAM 사용량이 ${vramInfo.value.pct}%입니다. 백엔드에서 모델을 unload 할까요?`)) return
  }
  requestAction('unload_model_request', {})
  requestAction('show_toast', { type: 'info', msg: '백엔드에 모델 unload 요청 전송됨' })
}
const autoSettings = reactive<AutomationSettings>({ mode: 'count', limit: 10, repeat: 1, delay: 1.0, allowDupes: false, autoResetDeck: false, maxRetries: 2, cleanupEveryN: 0 })
// 화면이 실제로 아는 자동화 키(서버=파일에서 받았거나 사용자가 고친 것) — 동기화는 이 키만 보낸다.
// 파일 값을 받기 전의 하드코딩 기본값이 Python 에서 파일로 저장되지 않게(R2b#1, utils/automationSettings).
const _autoKnownKeys = new Set<AutomationKey>()
// 사용자가 고친 키 — 부팅 hydrate 응답이 늦게 와도 이 키들은 파일 값으로 되돌리지 않는다
const _autoEditedKeys = new Set<AutomationKey>()

function syncAutomationSettings() {
  action('set_automation_settings', {
    ...automationSyncPayload(autoSettings, _autoKnownKeys),
    // 자동화 중 태그→자연어 자동 변환 (백엔드 루프가 nl_caption 적용)
    autoNl: autoNlGen.value,
    ollamaUrl: storedOllamaUrl(),
    ollamaModel: storedOllamaModel(),   // 비면 백엔드 워커가 설치 모델로 정한다
  })
}

// PR 9: 서버 → Vue 푸시 중에는 watch가 다시 서버로 보내지 않도록 가드
let _applyingAutoFromServer = false

watch(autoSettings, () => {
  if (_applyingAutoFromServer) return
  syncAutomationSettings()
}, { deep: true })

/** 서버(파일) 값 → 화면. watch 가 되돌려 보내지 않게 가드한다(같은 값이면 Python 도 다시 쓰지 않는다). */
function applyAutomationSettingsFromServer(patch: Partial<AutomationSettings>) {
  markAutomationKeys(_autoKnownKeys, patch)
  _applyingAutoFromServer = true
  Object.assign(autoSettings, patch)
  // watch 콜백이 큐잉 처리되도록 microtask 끝난 후 가드 해제
  Promise.resolve().then(() => { _applyingAutoFromServer = false })
}

// ── LoRA 스택 — composables/useLoraStack.js (App.vue 분할 ④). 카드(LoraStackCard)는 이 객체를 prop 으로 받고,
//    App 은 생성 직전 전송·LoRA 매니저 모달·부팅 복원에 쓰는 이름만 꺼낸다.
const lora = useLoraStack({ storeWidgets, addToast, saveUiPrefs })
const { syncLoraStack, showLoraModal, onLoraAdd, restoreFromPrefs: restoreLoraFromPrefs } = lora
// 시작 시 Ollama 모델 검증 — 저장된 모델이 설치 목록에 없으면(또는 비어있으면)
// 첫 번째 설치 모델로 자동 교체 + 영속. Settings를 열어 모델을 바꾸지 않아도
// 프롬프트 강화/자연어가 바로 동작하게 한다. (실패해도 조용히 무시)
async function ensureOllamaModel() {
  try {
    const bk = await getBackend()
    const url = storedOllamaUrl()
    // 결과는 ollamaModelsReady → applyOllamaModels. (GUI 스레드를 막던 동기 ollamaListModels 는 없앴다)
    if (bk?.requestOllamaModels) bk.requestOllamaModels(url)
  } catch {}
}
function applyOllamaModels(json: string) {
  try {
    const payload = JSON.parse(json)
    const models = Array.isArray(payload) ? payload : payload.models
    const url = storedOllamaUrl()
    if (!Array.isArray(payload) && payload.url && payload.url !== url) return
    if (!Array.isArray(models) || models.length === 0) return
    const cur = storedOllamaModel()
    // 백엔드 resolve_model 과 같은 규칙(utils/ollamaPrefs): 같은 모델(:latest 무시) →
    // 같은 계열의 설치된 태그 → 첫 설치 모델. 직접 추가한 커스텀 모델도 설치돼 있으면 유지.
    const next = resolveInstalledModel(cur, models)
    if (next && next !== cur) {
      window.localStorage.setItem('ollamaModel', next)
      requestAction('save_ui_prefs', { ollamaModel: next, ollamaUrl: url })
    }
  } catch {}
}

const showCondModal = ref(false)   // 조건부 프롬프트 모달 (showLoraModal/onLoraAdd/세트/드래그는 useLoraStack)

// History pagination
const visibleHistory = computed(() => historyPageSlice(historyImages.value, histPage.value, histPerPage))

// Context menu (화면 밖 방지)
const ctxMenu = ref({ show: false, x: 0, y: 0, path: '' })
const ctxMenuStyle = computed(() => {
  // menuH는 실제 메뉴 높이(항목 13행 ≈ 400px) 이상으로 잡아야 하단 flip이
  // 충분히 동작. 250이면 항목 추가 후 실제 높이보다 작아 하단 잘림 발생했음.
  // 보정 규칙은 Gallery·Favorites 와 같은 utils/ctxMenuPosition 한 곳.
  return menuPositionStyle(clampMenuPosition(
    { x: ctxMenu.value.x, y: ctxMenu.value.y },
    { width: 210, height: 420 },
    { width: window.innerWidth, height: window.innerHeight },
  ))
})

// ── 스튜디오 도구 매니저 — 상태·동작·백엔드 리스너는 composables(모듈 싱글턴), 모달(components/managers)은 표시만.
//    모달은 v-if 로 여닫지만 목록은 닫혀 있어도 받아야 한다: 글로벌 가중치(생성 경로), 프로파일 드롭다운(헤더),
//    와일드카드 열기(PromptPanel). 리스너는 onMounted 의 예전 자리에서 bind*() 로 붙인다.
const { showPresetManager, openPresetManager } = usePresetManager()
const { showWeightManager, openWeightManager, bind: bindGlobalWeights, weightedPrompt } = useGlobalWeights()
const { showWcManager, openWcManager, openWildcardByName, loadWildcardTree } = useWildcardManager()
const {
  showProfileManager, profileNames, openProfileManager,
  loadWorkflowProfilesList, loadWorkflowProfile, saveCurrentAsProfile, bind: bindWorkflowProfiles,
} = useWorkflowProfiles()
const { showOrderManager, openOrderManager, bind: bindPromptOrder } = usePromptOrder()
const { showInstantWcManager, openInstantWcManager, bind: bindInstantWildcards } = useInstantWildcards()
const { showStatsModal, openStatsModal } = useGenStats()

// 제네릭 — bridge.d.ts ActionPayloads 에 적힌 액션은 페이로드 모양까지 type-check 된다.
function action<K extends ActionName>(name: K, payload?: ActionPayload<K>) { requestAction(name, payload) }

// 태그→자연어 변환 (전용 채널 genNlResult로 결과 수신 — PromptPanel 리스너와 분리)
function _convertTagsToNl(tags: string) {
  const waiting = _nlGen.start()
  const url = storedOllamaUrl()
  const model = storedOllamaModel()   // 비면 백엔드가 설치 모델로 정한다
  getBackend().then(b => {
    if (!b || !b.convertPromptToNl) { requestAction('show_toast', { type: 'error', msg: 'AI 변환 불가 — 태그 그대로 생성' }); _finishNlGen(null); return }
    b.convertPromptToNl(tags, JSON.stringify({ url, model }))
  }).catch(() => _finishNlGen(null))
  return waiting
}
function _finishNlGen(result: any) {
  _nlGen.finish(result)
}

async function doGenerate() {
  // 자동화 중이면 중지 (버튼은 조종석으로 옮겼지만 단축키·외부 호출 경로가 남아 있다)
  if (isAutomating.value) { stopAutomation(); return }
  // Ctrl+G 도 버튼의 disabled(isGenerating || nlConverting) 조건을 따른다
  if (isGenerating.value) return
  if (nlConverting.value) return
  // 생성 시 태그→자연어 자동 변환 (단일 생성에서만 — 자동화는 프롬프트마다 변환하면 너무 느림)
  if (autoNlGen.value && !autoMode.value) {
    const tags = (storeWidgets.main_prompt_text || '').trim()
    if (tags && tags !== _lastAutoNl.value) {
      const nl = await _convertTagsToNl(tags)
      if (nl && nl.trim()) {
        const combined = tags + ', ' + nl.trim()   // 태그 뒤에 자연어 추가 ('main에 넣기'와 동일)
        storeWidgets.main_prompt_text = combined
        _lastAutoNl.value = combined
        await nextTick()
        await new Promise(r => setTimeout(r, 60))   // 위젯이 백엔드로 동기화될 여유
      }
      // 변환 실패/빈응답 시 원본 태그 그대로 진행 (fallback)
    }
  }
  _doGenerateNow()
}

function _doGenerateNow() {
  // 글로벌 가중치 적용(composables/useGlobalWeights) — 실패해도 생성은 원문으로 진행한다
  // (예전엔 예외가 생성 자체를 막았다). 가중치가 없거나 바뀐 게 없으면 null.
  try {
    const weighted = weightedPrompt(storeWidgets.main_prompt_text || '')
    if (weighted !== null) storeWidgets.main_prompt_text = weighted
  } catch (e) {
    console.error('[GlobalWeights] apply failed:', e)
    addToast('error', '글로벌 가중치를 적용하지 못해 원래 프롬프트로 생성합니다')
  }
  // LoRA Stack — 생성 직전에 스택 전체를 다시 보낸다(빈 스택·전부 꺼짐도 반드시 전송).
  // Python 은 이 _vue_lora_entries 하나에서 LoRA 텍스트를 파생한다 (useLoraStack / core/lora_stack.py).
  syncLoraStack()
  syncAutomationSettings()
  genStartTime.value = Date.now()
  genEta.value = ''
  action('generate')
}

function cancelGeneration() {
  action('cancel_generation')
  // 취소는 backend가 imageGenerated/error를 안 쏠 수 있으므로 즉시 상태 복구
  isGenerating.value = false
  genEta.value = ''
}

function showHistoryMenu(e: MouseEvent, path: string) { ctxMenu.value = { show: true, x: e.clientX, y: e.clientY, path } }
function hideCtxMenu() { ctxMenu.value.show = false }
const ctxAddFavorite = () => { action('add_favorite', { path: ctxMenu.value.path }); hideCtxMenu() }
const ctxSendI2I = () => { action('send_to_i2i', { path: ctxMenu.value.path }); hideCtxMenu() }
const ctxSendInpaint = () => { action('send_to_inpaint', { path: ctxMenu.value.path }); hideCtxMenu() }
const ctxSendEditor = () => { action('send_to_editor', { path: ctxMenu.value.path }); hideCtxMenu() }
const ctxCompare = (slot: any) => { action('send_to_compare', { path: ctxMenu.value.path, slot }); hideCtxMenu() }
const ctxRunAdetailer = () => { action('run_adetailer_single', { path: ctxMenu.value.path, settings: { ad_model: 'face_yolov8n.pt', ad_confidence: 0.3, ad_denoise: 0.4 } }); hideCtxMenu() }
const ctxCopyPath = () => { navigator.clipboard?.writeText(ctxMenu.value.path); hideCtxMenu() }
const ctxPullPrompt = () => { action('pull_prompt_from_image', { path: ctxMenu.value.path }); hideCtxMenu() }
const ctxAddToQueue = () => { action('add_image_to_queue', { path: ctxMenu.value.path }); hideCtxMenu() }
const ctxDelete = () => {
  // 히스토리에서 바로 지우지 않는다 — 휴지통 이동이 실패하면 파일은 남는데 목록에서만
  // 사라진다. 백엔드의 imageDeleteResult(removed) 를 받은 뒤 applyImageDeleteToHistory 가 뺀다.
  action('delete_image', { path: ctxMenu.value.path })
  hideCtxMenu()
}
/** 삭제 결과 반영 — 파일이 실제로 없어졌을 때만 히스토리에서 빼고, 보던 그림이면 첫 장으로. */
function applyImageDeleteToHistory(raw: unknown) {
  const next = applyDeleteToHistory(historyImages.value, currentImage.value, parseImageDeleteResult(raw))
  if (!next) return
  historyImages.value = next.history
  if (next.current !== currentImage.value) currentImage.value = next.current
}

async function selectHistoryImage(path: string) {
  // (선택 시엔 이미지 내용이 안 바뀌므로 cache-bust 하지 않음 — 매 선택마다 썸네일이
  //  재로드되어 점멸하던 버그 수정. 재생성 시 cache-bust는 imageGenerated 핸들러가 처리.)
  currentImage.value = path
  // 해상도는 원본 픽셀 기준이어야 한다. 스트립은 썸네일을 쓰므로(historyImageSrc)
  // 거기서 재면 '175 × 256' 같은 썸네일 크기가 뜬다 — 메타데이터의 실제 크기를 쓴다.
  const measureOriginal = () => {
    const img = new Image()
    img.onload = () => {
      if (currentImage.value === path) resolution.value = `${img.naturalWidth} × ${img.naturalHeight}`
    }
    img.src = mediaUrl(path)
  }
  // EXIF 로드
  const backend = await getBackend()
  if (backend.getImageExif) {
    backend.getImageExif(path, (json: string) => {
      if (currentImage.value !== path) return  // 방향키로 빠르게 넘길 때 늦은 응답 무시
      try {
        const d = JSON.parse(json)
        currentExif.value = exifFromPayload(d)
        if (d.size) resolution.value = d.size
        else measureOriginal()
      } catch { measureOriginal() }
    })
  } else {
    measureOriginal()
  }
}

function historyImageSrc(path: string) {
  // 썸네일이 준비됐으면 썸네일, 아니면 원본. 편집돼 버전이 오른 이미지는 원본(캐시 무효화).
  return historyThumbSrc(path, imageVersions[path] || 0)
}

// History 키보드 상하 네비게이션 — 전역 ↑/↓로 이전/다음 이미지 선택(인덱스 계산은 utils/historyNav).
// 선택이 없으면 첫 이미지부터. 페이지 경계를 넘으면 histPage도 따라 이동.
function goToHistoryIndex(idx: number | null) {
  if (idx === null) return
  histPage.value = historyPageOf(idx, histPerPage)
  selectHistoryImage(historyImages.value[idx])
}
function navigateHistory(dir: number) {
  goToHistoryIndex(stepHistoryIndex(historyImages.value, currentImage.value, dir))
}

// 최상단(top=최신)/최하단(bottom=가장 오래됨)으로 바로 이동 (Shift+화살표 등)
function navigateHistoryEdge(edge: string) {
  goToHistoryIndex(edgeHistoryIndex(historyImages.value, edge))
}

// 드래그 앤 드롭 지원
function onDragStart(e: DragEvent, path: string) {
  e.dataTransfer!.setData('text/plain', path)
  e.dataTransfer!.effectAllowed = 'copy'
}

/** 지금 탭 — `.main-workspace[data-tab]` 로 노출돼 화면·테스트가 '어느 탭인가'를 읽는다. */
const activeTabName = ref('t2i')

function onTabChanged(tabName: string) {
  activeTabName.value = tabName
  showLeftPanel.value = ['t2i', 'i2i', 'inpaint'].includes(tabName)
  showExtendPanel.value = false
  hideCtxMenu()
}


async function loadHistory() {
  const backend = await getBackend()
  // 결과는 galleryImagesReady → applyHistoryImages. 목 모드(개발 서버)엔 슬롯이 없어 건너뛴다.
  if (backend.requestGalleryImages) backend.requestGalleryImages('')
}
function applyHistoryImages(arr: unknown) {
  if (!Array.isArray(arr)) return
  historyImages.value = arr as string[]   // 캐시 저장은 아래 watch 한 곳에서
}
// 목록 교체·생성 결과 추가(unshift)·삭제(splice) 때마다 부팅용 캐시 동기화 — deep: 1 이라 변이도 잡는다
watch(historyImages, (arr) => {
  try { localStorage.setItem('historyImagesCache', JSON.stringify(arr.slice(0, 50))) } catch {}
}, { deep: 1 })

import { useRouter, useRoute } from 'vue-router'
const router = useRouter()
const route = useRoute()

// ── 세션 복구 (크래시/VRAM OOM 후 작업 이어가기) — composables/useSessionRestore.ts ──
// 템플릿이 쓰는 이름(sessionRestore·applySessionRestore·dismissSessionRestore)은 전부 여기서 꺼낸다.
const {
  sessionRestore, applySessionRestore, dismissSessionRestore, startSessionBackup,
} = useSessionRestore({
  storeWidgets,
  currentTab: () => (route && route.name) || 't2i',
  goToTab: (tab: string) => { void router.push({ name: tab }) },
  getBackend,
  addToast,
})

// UI 크기 — Chromium zoom 으로 전역 확대 (폰트/아이콘/패딩 비례)
// 변경 시 즉시 반영, localStorage 영속, 다른 탭(Settings)에서 변경하면 storage event로 동기화
const _applyUiScale = (val: any) => {
  const scale = normalizeUiScale(val)   // 0.7~2.0 밖·숫자 아님 → 1.0 (utils/uiScale)
  try { document.documentElement.style.zoom = String(scale) } catch {}
}
_applyUiScale(localStorage.getItem('ui.scale') || '1.0')

onMounted(async () => {
  // 히스토리 썸네일: thumbnailReady 수신 + 부팅 폴백 폭(localStorage; uiPrefsLoaded 가 override) + 설정 즉시 반영
  bindHistoryThumbs()
  setPreviewThumbWidth(normalizePreviewThumbWidth(window.localStorage.getItem('previewThumbWidth')), historyImages.value)
  window.addEventListener('previewThumbWidthChanged', onPreviewThumbWidthChanged)
  await initBridge()
  // 게이트를 가장 먼저 붙인다. 파이썬은 Vue 로드 직후 backendSelectionRequired 를
  // 보내기 시작하므로(0.3초 뒤 첫 재전송), 다른 구독보다 늦으면 첫 신호를 흘린다.
  bindBackendGate()
  // 계기 스트립의 백엔드 칸. 게이트와 같은 이유로 일찍 붙인다 — 파이썬이 되풀이해
  // 보내긴 하지만, 늦게 붙을수록 첫 몇 번을 흘려 스트립이 '연결 안 됨'으로 머문다.
  onBackendEvent('backendStatus', (json: string) => {
    try { backendStatus.value = JSON.parse(json || 'null') } catch { /* 깨진 페이로드로 줄을 잃지 않는다 */ }
  })
  onBackendEvent('ollamaModelsReady', applyOllamaModels)
  onBackendEvent('adetailerModelsReady', (json: string) => { try { adModelItems.value = JSON.parse(json) } catch {} })
  storeWidgets.negpip_group = 'true'   // NegPiP 상시 적용 (UI 토글 제거) — Python 도 켠 채로 유지한다(generator_ui_setup)
  loadCondRules()                      // 조건부 프롬프트 규칙 로드 (모달/Search 공유)
  setTimeout(ensureOllamaModel, 3000)  // 폴백: uiPrefsLoaded가 안 와도 AI 모델 검증
  // 세션 복구 제안(1.2초 뒤, 프롬프트 로드된 뒤) + 편집 2.5초 뒤·30초 주기 백업 — 결정 전엔 백업을 덮지 않는다
  startSessionBackup()
  // Settings 등 다른 곳에서 ui.scale 변경 시 즉시 반영
  window.addEventListener('storage', (e) => {
    if (e.key === 'ui.scale') _applyUiScale(e.newValue)
  })
  // 같은 창에서의 변경 (Settings 슬라이더)도 즉시 — 커스텀 이벤트
  window.addEventListener('uiScaleChanged', (e) => _applyUiScale((e as CustomEvent).detail?.value))
  // HISTORY 선택 깜빡임 토글 — Settings에서 변경 시 즉시 반영
  window.addEventListener('historyBlinkChanged', (e) => { historyBlink.value = !!(e as CustomEvent).detail?.value })
  // localStorage 캐시로 즉시 표시 (응답 빠르게), 그 후 backend에서 최신 가져옴
  try {
    const cached = localStorage.getItem('historyImagesCache')
    if (cached) {
      const arr = JSON.parse(cached)
      if (Array.isArray(arr) && arr.length > 0) historyImages.value = arr
    }
  } catch {}
  // 시작 시 히스토리 자동 로드 — 사용자가 F5 안 눌러도 이전 이미지들 보이게
  onBackendEvent('galleryImagesReady', (json: string) => {
    try {
      const payload = JSON.parse(json)
      if (payload.folder === '') {
        applyHistoryImages(payload.files)
        // 출력 폴더의 원본 서명 — 즐겨찾기 카드도 같은 파일의 버전을 쓴다(utils/mediaVersions)
        recordListedMediaVersions(payload.files, payload.versions)
      }
    } catch {}
  })
  loadHistory()
  document.addEventListener('click', hideCtxMenu)
  document.addEventListener('wheel', (e) => { if (e.ctrlKey) e.preventDefault() }, { passive: false })
  // 브라우저 기본 우클릭 메뉴 전역 차단
  document.addEventListener('contextmenu', (e) => { e.preventDefault() })
  // 전역 단축키 — 판단(게이트 · ESC 는 모달 스택 맨 위부터 · 모달이 떠 있으면 ↑/↓ 히스토리 이동 금지)은
  // utils/appShortcuts 에 있고, 여기선 ref 와 동작만 잇는다. 모달은 열릴 때 스스로 appModalStack 에
  // 오른다(composables/useModalLayer) — App.modalLayer.test.ts 가 이 배선을 지킨다.
  document.addEventListener('keydown', createAppKeydownHandler({
    modalStack: appModalStack,
    isGateOpen: () => gateOpen.value,
    generate: () => { void doGenerate() },
    saveSettings: () => action('save_settings'),
    reloadHistory: () => { void loadHistory() },
    navigateTabs: (direction) => {
      try { window.dispatchEvent(new CustomEvent('navRailNavigate', { detail: { direction } })) } catch {}
    },
    isParamsPanelOpen: () => showExtendPanel.value,
    closeParamsPanel: () => { showExtendPanel.value = false },
    hasHistory: () => historyImages.value.length > 0,
    navigateHistory,
    navigateHistoryEdge,
    activeElement: () => document.activeElement as HTMLElement | null,
    jumpModifier: () => localStorage.getItem('historyJumpModifier'),
  }))

  // rating 필터는 마운트 때 보내지 않는다 — uiPrefsLoaded → restoreRatingFromPrefs 가 파일 값을 적용한
  // 직후 보낸다(localStorage 캐시가 파일·공유 필터를 덮지 않게, 감사 #107).

  // 워크플로우 프로파일 목록 미리 로드 (드롭다운에 즉시 보이도록)
  setTimeout(loadWorkflowProfilesList, 800)

  onBackendEvent('tabChanged', (tabId: string) => {
    const targetPath = tabId === 't2i' ? '/' : `/${tabId}`
    router.push(targetPath)
    onTabChanged(tabId)
  })

  onBackendEvent('imageGenerated', async (data: string) => {
    const parsed = JSON.parse(data)
    // 이미 목록에 있는 경로 = 같은 파일에 덮어쓴 결과 — 그 카드의 옛 썸네일·원본 캐시만 버린다.
    // (새 경로는 캐시가 없으니 버전을 올리지 않는다 — 올리면 세션 내 생성분이 늘 원본을 읽었다.)
    const overwritten = !!parsed.path && historyImages.value.includes(parsed.path)
    if (overwritten) {
      imageVersions[parsed.path] = Date.now()
      invalidateHistoryThumb(parsed.path)
      bumpMediaVersion(parsed.path)   // 갤러리·즐겨찾기 카드도 새로 읽게
    }
    isGenerating.value = false
    livePreview.value = ''
    genEta.value = ''
    status.value = ''
    // History에서 옛 이미지를 보던 중(브라우징)이면 뷰를 뺏지 않음 — '최신'(또는
    // 미선택)을 보고 있을 때만 새 이미지로 따라간다. 새 이미지는 History 맨 앞에
    // 추가되어 t2i 결과 목록에 표시됨. (선택 중이던 이미지/위치는 그대로 유지)
    const wasViewingLatest = !currentImage.value || currentImage.value === historyImages.value[0]
    if (parsed.path) {
      // 덮어쓴 경로는 맨 앞으로 옮긴다(같은 경로가 두 번 있으면 v-for key 가 겹친다)
      if (overwritten) historyImages.value = historyImages.value.filter(p => p !== parsed.path)
      historyImages.value.unshift(parsed.path)  // 무제한 — 갯수 캡 제거
      if (wasViewingLatest) {
        histPage.value = 0
      } else {
        // 보던 옛 이미지를 계속 보여주도록 그 이미지가 있는 페이지로 유지
        const ci = historyImages.value.indexOf(currentImage.value)
        if (ci >= 0) histPage.value = Math.floor(ci / histPerPage)
      }
    }
    if (wasViewingLatest && parsed.path) {
      currentImage.value = parsed.path
      // 크기·시드가 없는 결과(외부 워크플로 등)도 'undefined' 를 찍지 않는다(utils/imageInfo)
      resolution.value = formatResolution(parsed.width, parsed.height)
      seed.value = formatSeed(parsed.seed)
      // 생성 직후 EXIF 자동 로드 (추종 중일 때만 현재 EXIF 갱신)
      const bk = await getBackend()
      if (bk.getImageExif) {
        bk.getImageExif(parsed.path, (json: string) => {
          try { currentExif.value = exifFromPayload(JSON.parse(json)) } catch {}
        })
      }
    }
  })
  onBackendEvent('generationStarted', () => { isGenerating.value = true; autoWaiting.value = false; progressVal.value = 0; genStartTime.value = Date.now(); genEta.value = ''; livePreview.value = '' })
  onBackendEvent('generationPreview', (b64: string) => {
    if (!isGenerating.value) return
    const url = previewDataUrl(b64)   // JPEG/WebP/PNG 를 머리 바이트로 고른다(utils/generationProgress)
    if (url) livePreview.value = url
  })
  onBackendEvent('automationStatus', (json: string) => {
    try {
      // 필드 이름은 bridge.d.ts AutomationStatusEvent 계약 — Partial 인 이유: 옛 백엔드가 일부를 빠뜨린다.
      const d = JSON.parse(json) as Partial<AutomationStatusEvent>
      isAutomating.value = d.running || false
      autoPaused.value = d.paused || false
      autoGenCount.value = d.count || 0
      autoWaiting.value = d.waiting || false
      // 다음에 나갈 프롬프트 전문 — 조종석의 태그 칩이 이걸 그린다.
      autoNextPrompt.value = typeof d.prompt === 'string' ? d.prompt : ''
      autoPromptIsNext.value = d.prompt_is_next !== false   // 옛 백엔드는 안 보낸다 — 편집 가능으로 본다
      deckRemaining.value = d.deck_remaining || 0
      deckTotal.value = d.deck_total || 0
      deckUsed.value = d.deck_used || 0
      deckAllowDup.value = d.allow_duplicates || false
      waitRemainingMs.value = d.wait_remaining_ms || 0
      waitTotalMs.value = d.wait_total_ms || 0
      if (!d.running) { isGenerating.value = false }
    } catch {}
  })
  // 워크플로우 프로파일 목록 수신(workflowProfilesList → useWorkflowProfiles)
  bindWorkflowProfiles()
  // 프롬프트 섹션 순서 수신(promptOrderLoaded → usePromptOrder)
  bindPromptOrder()
  // PR 8: 인스턴트 와일드카드 목록 수신(instantWildcardsList → useInstantWildcards)
  bindInstantWildcards()
  // PR 9: 백엔드가 모드별 자동화 설정을 푸시 (백엔드 모드 전환 시). 부팅 값은 아래 hydrate 가 당겨 온다.
  onBackendEvent('automationSettingsLoaded', (json: string) => {
    const patch = automationPatchFromServer(json)
    if (Object.keys(patch).length) applyAutomationSettingsFromServer(patch)
  })
  onBackendEvent('generationProgress', (step: number, total: number) => {
    progressVal.value = progressPercent(step, total)
    status.value = `Generating... ${step}/${total}`
    // 간단 ETA: 시작 시각 기준(utils/generationProgress.formatEta)
    if (step > 0 && genStartTime.value) {
      const eta = formatEta((Date.now() - genStartTime.value) / 1000, step, total)
      if (eta !== null) genEta.value = eta
    }
  })
  // 생성 실패 — 진행 상태를 되돌리고 같은 메시지를 토스트로도 알린다(예전엔 두 번 구독했다).
  onBackendEvent('generationError', (msg: string) => {
    isGenerating.value = false; livePreview.value = ''; genEta.value = ''; status.value = `Error: ${msg}`
    addToast('error', msg)
  })

  // 글로벌 가중치 로드(globalWeightsLoaded → useGlobalWeights — 생성 경로가 모달 없이도 쓴다)
  bindGlobalWeights()

  // 랜덤 해상도 로드
  randomRes.loadRandomResList()

  // ADetailer 모델 로드
  const _bk = await getBackend()
  // 자동화 설정: 파일(현재 모드) 값으로 먼저 채우고 그 뒤에 동기화한다 — 하드코딩 기본값이 파일을
  // 덮지 않게(감사 #42). 응답이 없어도 시간 초과 뒤 동기화는 한다(autoNl·Ollama 전달 경로) — 그때도
  // 모르는 키는 보내지 않고(_autoKnownKeys), 늦게 온 응답은 고치지 않은 키에 적용된다(R2b#1).
  // 서버가 답하면 모든 키가 '아는 값'이다(파일에서 빠졌거나 잘못된 키는 화면의 안전값이 기준).
  hydrateAutomationSettings({
    request: typeof _bk.getAutomationSettings === 'function' ? (cb) => _bk.getAutomationSettings(cb) : null,
    editedKeys: () => _autoEditedKeys,
    apply: applyAutomationSettingsFromServer,
    onSettled: (hydrated) => {
      if (hydrated) for (const key of AUTOMATION_KEYS) _autoKnownKeys.add(key)
      syncAutomationSettings()
    },
  })
  // 결과는 adetailerModelsReady(위 구독). 동기 getADetailerModels 는 GUI 스레드를 막아 없앴다.
  if (_bk.requestADetailerModels) _bk.requestADetailerModels()

  // 와일드카드 로드(useWildcardManager — PromptPanel 의 와일드카드 칩이 모달 밖에서도 연다)
  loadWildcardTree(_bk)

  // VRAM 실시간 업데이트
  onBackendEvent('vramUpdated', (json: string) => { try { vramInfo.value = JSON.parse(json) } catch {} })

  // Global Toast 알림 (Python → Vue)
  onBackendEvent('showNotification', (type: string, msg: string) => { addToast(type, msg) })
  // 삭제 결과(경로별) — 실제로 파일이 없어졌을 때만 히스토리에서 뺀다(갤러리 삭제도 여기 반영).
  onBackendEvent('imageDeleteResult', applyImageDeleteToHistory)
  // 업데이트 알림도 같은 전역 토스트를 사용하므로 listener 등록 뒤 시작한다.
  void initialiseAppUpdates(true)

  onBackendEvent('uiPrefsLoaded', (json: string) => {
    try {
      const prefs = JSON.parse(json)
      // 파일 값 → localStorage 캐시: 미러는 여기 한 곳(utils/uiPrefMirror 표). 부팅 때 동기로 읽는 곳
      // (UI 배율·탭 순서·블록 모드·Ollama 등)의 첫 렌더 캐시일 뿐이고 파일 값이 이긴다.
      const mirrored = new Set(mirrorPrefsToStorage(prefs))
      // 테마: 디스크가 단일 소스 — 다른 기기/프로필에서 바꾼 값이 부팅 캐시
      // (localStorage)보다 우선한다. 부팅 값과 같으면 다시 칠하지 않는다.
      reconcileTheme(prefs)
      // AI Assistant 모델/URL — 캐시(위)를 설치 목록과 대조해 유효하지 않으면 자동 교체 →
      // Settings를 안 열어도 바로 동작.
      ensureOllamaModel()
      restoreRatingFromPrefs(prefs)   // rating 필터 단일 소스(ui_prefs) 복원 (useRatingFilter)
      restoreHighResFromPrefs(prefs)   // 고해상도: 단일 소스(ui_prefs)가 localStorage override (useHighRes)
      restoreLoraFromPrefs(prefs)   // 단일 소스(ui_prefs.loraStack) 복원 (useLoraStack)
      restoreUiFlagsFromPrefs(prefs)   // 블록 모드·갤러리 메타 — PromptPanel·Gallery 가 같은 ref 를 본다
      // tabOrder 복원 (Settings 탭 미방문 시에도 적용) — 항상 마운트돼 있는 NavRail은 같은 창
      // setItem을 못 받으므로 커스텀 이벤트로 재읽기 유도
      if (mirrored.has('tabOrder')) {
        try { window.dispatchEvent(new CustomEvent('tabOrderChanged')) } catch {}
      }
      if (typeof prefs.historyBlinkSelected === 'boolean') historyBlink.value = prefs.historyBlinkSelected
      // 히스토리 미리보기 품질(썸네일 폭) — 단일 소스 ui_prefs, localStorage 는 부팅 폴백
      if (mirrored.has('previewThumbWidth')) {
        setPreviewThumbWidth(normalizePreviewThumbWidth(prefs.previewThumbWidth), historyImages.value)
      }
      if (mirrored.has('uiScale')) _applyUiScale(Number(prefs.uiScale))
      if (mirrored.has('editorSidePanelWidth')) {
        try { window.dispatchEvent(new CustomEvent('editorSidePanelWidthChanged')) } catch {}
      }
      if (typeof prefs.autoNlGen === 'boolean') autoNlGen.value = prefs.autoNlGen
    } catch {}
  })

  // 생성용 태그→자연어 변환 결과 (전용 채널 — PromptPanel의 ollamaResult와 분리)
  onBackendEvent('genNlResult', (json: string) => {
    try {
      const d = JSON.parse(json)
      if (d && d.tags && !d.error) { _finishNlGen(d.tags) }
      else { if (d && d.error) requestAction('show_toast', { type: 'error', msg: 'AI 변환 실패 — 태그 그대로 생성' }); _finishNlGen(null) }
    } catch { _finishNlGen(null) }
  })

  // loraStackLoaded 핸들러는 useLoraStack 내부에서 등록됨 (App.vue 분할 ④)
  // 마운트 시점엔 LoRA 스택을 Python 에 보내지 않는다 — 이 시점 스택은 이 브라우저의 localStorage
  // 폴백이라, 웹 모드(ui_prefs 가 늦게 옴)에선 낡은/빈 스택이 공유 _vue_lora_entries 를 덮었다.
  // 전송은 uiPrefsLoaded → restoreLoraFromPrefs 가 ui_prefs 스택을 적용한 직후에 한다.
})
</script>

<style scoped>
/* 가로 배치 — 왼쪽에 세로 탭 레일(NavRail), 그 오른쪽이 작업 공간.
   상단 60px 헤더(가로 탭 바)를 없앤 자리가 그대로 무대 세로로 간다.
   형제인 진행바·VRAM 바·모달은 전부 position:fixed 라 흐름에서 빠져 있어
   가로 배치로 바꿔도 옆으로 늘어서지 않는다. */
.app-container { width: 100%; height: 100vh; display: flex; flex-direction: row; background: var(--bg-primary); }

/* 하단 고정 VRAM 바(22px)에 콘텐츠가 가리지 않도록 여백 확보.
   (큐가 하단 도크였을 땐 그 도크가 스페이서 역할을 했으나, 우측 드로어로 옮기며
   콘텐츠가 화면 맨 아래까지 내려와 GENERATE 버튼이 VRAM 바에 잘리던 문제 수정) */
/* min-width:0 이 없으면 안쪽 패널의 내용 폭이 flex 기본값(auto)을 밀어 올려
   레일이 눌리거나 가로 스크롤이 생긴다. */
.main-workspace { flex: 1; min-width: 0; display: flex; overflow: hidden; position: relative; padding-bottom: 28px; }

.side-panel { width: 360px; display: flex; flex-direction: column; background: var(--bg-secondary); border-right: 1px solid var(--border); z-index: 10; }
.side-panel.right { width: 220px; border-right: none; border-left: 1px solid var(--border); }
.panel-scroll { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 16px; }

/* 파라미터 — 왼쪽 열 안에서 .panel-scroll 과 자리를 번갈아 쓴다.
   (예전엔 left:360px 에 440px 폭으로 무대 위에 떠 있었다) */
.extend-overlay { flex: 1; min-height: 0; display: flex; flex-direction: column; }
.extend-header { display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; border-bottom: 1px solid var(--border); }
.extend-header h3 { font-size: 11px; letter-spacing: 0; color: var(--text-muted); }
.close-btn { width: 28px; height: 28px; background: var(--bg-button); border: 1px solid var(--border-strong); border-radius: 6px; color: var(--text-secondary); font-size: 16px; cursor: pointer; }
.close-btn:hover { border-color: var(--state-alert-fg); color: var(--state-alert-fg); background: rgba(248, 113, 113, 0.08); }
.extend-scroll { flex: 1; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 12px; }

/* .ext-card · .ext-title · .ext-field · .ext-row · .ext-check-row · .ext-sub · .ext-sub-title · .ext-toggle-grid 는
   styles/panels.css(전역)에 있다 — scoped 로 두면 자식 패널(AnimaGuidancePanel 등) 안쪽에 닿지 않는다.
   그 규칙은 `:where(.extend-overlay)` 안의 카드에만 걸리므로 카드는 이 파라미터 열 안에 둔다
   (RefinePanel·BatchView 의 Sam3ControlNetPanel 은 카드 모양을 받지 않는다 — tests/test_ext_card_styles.py).
   카드마다 제 모양(해상도·고해상도·랜덤 해상도·등급 필터·LoRA 블록)은 components/params/*.vue 의 scoped 에 있다. */

/* 무대 전환 — 탭이 바뀌면 새 화면이 살짝 떠오른다. 길면 매번 기다리게 되고 없으면 못 알아챈다.
   나가는 쪽 규칙은 일부러 없다 (위 주석). */
.stage-enter-active { transition: opacity .16s ease, transform .16s ease; }
.stage-enter-from { opacity: 0; transform: translateY(6px); }
@media (prefers-reduced-motion: reduce) {
  .stage-enter-active { transition: none; }
}

/* Tool Card */
.tool-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-card); padding: 16px; }
.tool-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
.tool-btn { position: relative; padding: 8px 4px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-base); color: var(--text-secondary); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer; transition: var(--transition); }
.tool-btn-on { color: var(--state-ok-fg); border-color: var(--state-ok-fg); box-shadow: 0 0 0 1px rgba(74,222,128,0.25) inset; }
.tool-dot { position: absolute; top: 3px; right: 4px; width: 6px; height: 6px; border-radius: 50%; background: var(--state-ok-fg); box-shadow: 0 0 5px var(--state-ok-fg); }
/* 세션 복구 배너 */
.session-restore { position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%); z-index: 9999; display: flex; align-items: center; gap: 12px; background: var(--bg-secondary); border: 1px solid var(--accent); border-radius: 10px; padding: 10px 16px; box-shadow: 0 8px 28px rgba(0,0,0,0.5); }
.sr-msg { font-size: 12px; color: var(--text-primary); }
.sr-apply { background: var(--accent-fill); color: var(--on-accent); border: none; border-radius: 6px; font-size: 11px; font-weight: var(--fw-bold); padding: 6px 14px; cursor: pointer; }
.sr-dismiss { background: var(--bg-button); border: 1px solid var(--border); border-radius: 6px; color: var(--text-secondary); font-size: 11px; font-weight: var(--fw-bold); padding: 6px 12px; cursor: pointer; }
.tool-btn:hover { border-color: var(--text-muted); color: var(--text-primary); }
.tool-btn.highlight { color: var(--accent); border-color: var(--accent-dim); }

/* 매니저 모달(프리셋·가중치·통계·와일드카드·프로파일·순서·즉석 WC)의 모양은 components/managers/ 로 옮겼다.
   fade 는 여기 남는다 — 모달의 `<transition name="fade">` 가 이 파일 템플릿에 있고, 이 규칙이 자식 모달의
   루트(오버레이)에 걸린다. */
.fade-enter-active, .fade-leave-active { transition: opacity 0.2s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }

.gen-footer { padding: 12px 16px; background: var(--bg-card); border-top: 1px solid var(--border); display: flex; flex-direction: column; gap: 8px; }
.gen-actions { display: flex; gap: 6px; }
.action-btn { flex: 1; padding: 7px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-base); color: var(--text-secondary); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer; transition: var(--transition); }
.action-btn.active { border-color: var(--state-ok-fg); color: var(--state-ok-fg); background: rgba(74,222,128,0.05); }
.action-btn.highlight { border-color: var(--accent-dim); color: var(--accent); }
.action-btn:hover { border-color: var(--text-muted); }
/* 자동화 설정·상태의 모양은 `components/AutomationPanel.vue` 로 옮겼다 —
   여기 있던 .auto-settings / .auto-status / .auto-deck-pre 는 같은 내용을
   조건만 달리해 두 번 그리던 자리였다. */

/* 워크플로우 프로파일 */
.profile-card { background: rgba(96,165,250,0.04); border: 1px solid rgba(96,165,250,0.15);
  border-radius: 6px; padding: 8px; margin-bottom: 8px; }
.profile-row { display: flex; align-items: center; gap: 6px; }
.profile-label { font-size: var(--fs-label); color: var(--state-info-fg); font-weight: var(--fw-bold); flex-shrink: 0;
  letter-spacing: 0; }
.profile-row :deep(.csel) { flex: 1; min-width: 0; }
.profile-mini-btn { width: 28px; height: 28px; flex-shrink: 0; cursor: pointer;
  background: rgba(96,165,250,0.15); color: var(--state-info-fg);
  border: 1px solid rgba(96,165,250,0.3); border-radius: 4px; font-weight: var(--fw-bold); }
.profile-mini-btn:hover { background: rgba(96,165,250,0.3); border-color: var(--state-info-fg); color: var(--text-primary); }

.generate-row { display: flex; gap: 6px; align-items: stretch; }
.btn-cancel { width: 50px; height: 50px; background: transparent; border: 2px solid var(--state-alert-fg); border-radius: var(--radius-pill); color: var(--state-alert-fg); font-size: 18px; font-weight: var(--fw-bold); cursor: pointer; transition: var(--transition); }
/* 채움 위의 흰 글자는 토큰이 아니다 — --state-alert 는 "흰 글자와 4.5:1"
   을 맞춘 면 색이고(라이트에선 #B3261E, 7.5:1), --text-primary 를 쓰면
   라이트에서 검은 글자가 빨간 면에 얹혀 2.2:1 로 무너진다. */
.btn-cancel:hover { background: var(--state-alert); color: #fff; }
.gen-eta { font-size: var(--fs-label); color: var(--text-muted); text-align: center; letter-spacing: 0; }
.btn-generate { width: 100%; height: 50px; background: var(--accent-fill); border: none; border-radius: var(--radius-pill); color: var(--on-accent); font-weight: var(--fw-bold); font-size: 14px; letter-spacing: 0; cursor: pointer; transition: var(--transition); }
.btn-generate:hover:not(:disabled) { background: var(--accent-fill-hover); transform: translateY(-2px); box-shadow: 0 8px 24px rgba(250, 204, 21, 0.3); }
.btn-generate:disabled { opacity: 0.5; cursor: wait; }
/* 자동화 중지 버튼(.automating)은 사라졌다 — 멈추기는 조종석 안에 있다. */
.btn-generate.converting { background: var(--state-info); color: #fff; cursor: wait; }
/* margin-bottom 은 .gen-footer 의 gap 과 겹쳐 16px 짜리 빈 줄을 만들고 있었다.
   자동화를 켜면 프롬프트 영역이 그만큼 더 깎인다 — 간격은 부모의 gap 하나로 둔다. */
.auto-nl-toggle { display: flex; align-items: center; gap: var(--sp-2); min-height: 28px; padding: var(--sp-1) var(--sp-2); background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-base); font-size: var(--fs-label); font-weight: var(--fw-medium); color: var(--text-secondary); cursor: pointer; transition: var(--transition); user-select: none; width: fit-content; max-width: 100%; }
.auto-nl-toggle.on { border-color: var(--accent); color: var(--accent); background: var(--accent-dim); }

/* Viewport */
.viewport-area { flex: 1; display: flex; flex-direction: column; overflow: hidden; background: var(--bg-primary); }
.viewport-main { flex: 1; position: relative; overflow: hidden; }

/* EXIF Bar */
.exif-bar { flex-shrink: 0; background: var(--bg-secondary); border-top: 1px solid var(--border); }
.exif-tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); }
.exif-tab { flex: 1; padding: 6px; background: transparent; border: none; color: var(--text-muted); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer; text-align: center; border-bottom: 2px solid transparent; }
.exif-tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.exif-content { padding: 6px 12px; font-size: 11px; color: var(--text-secondary); max-height: 80px; overflow-y: auto; line-height: 1.5; font-family: 'Consolas', monospace; white-space: pre-wrap; word-break: break-all; }
.exif-params { padding: 4px 12px; max-height: 80px; overflow-y: auto; }
.exif-params .param-line { display: flex; align-items: baseline; gap: 6px; padding: 2px 0; font-size: var(--fs-label); color: var(--text-secondary); font-family: 'Consolas', monospace; }
.exif-params .pl { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--accent); letter-spacing: 0; min-width: 40px; flex-shrink: 0; }
.exif-params .param-line.other { color: var(--text-muted); }

/* History */
.hist-header { padding: 16px; display: flex; justify-content: space-between; align-items: center; }
.hist-header h3 { font-size: 12px; letter-spacing: 0; color: var(--text-muted); }
.count-badge { background: var(--border); padding: 2px 8px; border-radius: 10px; font-size: var(--fs-label); color: var(--text-secondary); }
.hist-nav-btn { width: 100%; height: 28px; padding: 0; background: var(--bg-secondary); border: none; color: var(--text-secondary); font-size: var(--fs-meta); cursor: pointer; flex-shrink: 0; display: flex; align-items: center; justify-content: center; }
.hist-nav-btn:hover { background: var(--bg-card); color: var(--text-primary); }
.hist-nav-btn:disabled { opacity: 0.3; cursor: default; }
.hist-scroll { flex: 1; min-height: 0; overflow-y: auto; padding: 8px; display: flex; flex-direction: column; gap: 8px; }
/* flex:1 1 0 — 5개 카드가 컬럼 높이를 균등 분배 → 모든 간격(8px)이 일정.
   기존 aspect-ratio:1(정사각)은 5개가 컬럼을 못 채워 하단에 큰 빈 공간이 남아
   간격이 불균일해 보였음. */
.hist-card { position: relative; flex: 1 1 0; min-height: 0; border-radius: var(--radius-card); overflow: hidden; border: 2px solid transparent; cursor: pointer; transition: border-color 0.15s; }
.hist-card:hover { border-color: var(--border); }
.hist-card.selected { border-color: var(--accent); box-shadow: 0 0 12px var(--accent-dim); }
.hist-card.selected.blink { animation: histBlink 1s ease-in-out infinite; }
@keyframes histBlink {
  0%, 100% { border-color: var(--accent); box-shadow: 0 0 6px var(--accent-dim); }
  50% { border-color: var(--text-primary); box-shadow: 0 0 20px var(--accent); }
}
.hist-card img { width: 100%; height: 100%; object-fit: cover; display: block; }

/* Context Menu */
.modern-ctx-menu { position: fixed; background: var(--bg-input); border: 1px solid var(--border); border-radius: 10px; padding: 6px; z-index: 1000; min-width: 200px; box-shadow: 0 12px 32px rgba(0,0,0,0.8); max-height: calc(100vh - 16px); overflow-y: auto; }
.ctx-item { padding: 10px 14px; font-size: 11px; font-weight: var(--fw-bold); color: var(--text-muted); cursor: pointer; border-radius: 6px; transition: var(--transition); }
.ctx-item:hover { background: var(--bg-button-hover); color: var(--text-primary); }
.ctx-item.delete { color: var(--state-alert-fg); }
.ctx-item.delete:hover { background: rgba(248, 113, 113, 0.1); }
.ctx-separator { height: 1px; background: var(--rule); margin: 4px 0; }

.pop-enter-active { animation: pop 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275); }
@keyframes pop { from { opacity: 0; transform: scale(0.9); } to { opacity: 1; transform: scale(1); } }

/* 하단 VRAM 바는 components/StatusStrip.vue 로 옮겼다 (백엔드·VRAM·모델 한 줄). */

.global-progress { position: fixed; top: 0; left: 0; width: 100%; height: 3px; background: transparent; z-index: 1000; }
.progress-fill { height: 100%; background: var(--accent); transition: width 0.3s ease; }

/* 토스트 · 알림 벨/기록 패널은 components/ToastLayer.vue · NotificationCenter.vue 로 옮겼다. */
</style>
