/**
 * ui_prefs 쓰기 경로 한 곳 + 여러 화면이 같이 보는 설정 플래그 (감사 #41 · #145).
 *
 * - {@link persistUiPrefs}: localStorage 캐시(utils/uiPrefMirror 표)와 파일(save_ui_prefs)을 함께
 *   쓴다. SettingsView 가 키마다 'localStorage.setItem + save_ui_prefs' 를 손으로 하던 것을 대신한다.
 *   Python 쪽 즉시 효과(클리너·Anima Guard·Forge 저장 설정)가 있으므로 디바운스하지 않는다 —
 *   고빈도 발생원(LoRA 가중치·고해상도 배율 슬라이더)만 각 composable 이 스스로 디바운스한다.
 * - {@link tagBlockMode} · {@link galleryShowMetadata}: 모듈 전역 ref. 쓰는 곳(Settings)과 읽는 곳
 *   (PromptPanel·Gallery)이 같은 문서라 storage 이벤트가 안 와서 예전엔 300/500ms setInterval 로
 *   localStorage 를 폴링했다. 이제 같은 ref 를 공유하고, 다른 브라우저 탭(웹 모드)의 변경만 storage
 *   이벤트로 받는다. 부팅 시 파일 값은 App.vue uiPrefsLoaded → {@link restoreUiFlagsFromPrefs} 가 넣는다.
 */
import { ref } from 'vue'
import { requestAction } from '../stores/widgetStore.js'
import { mirrorPrefsToStorage, readStoredBool } from '../utils/uiPrefMirror'

/** ui_prefs 조각을 캐시와 파일에 함께 저장한다(값 검증은 캐시 쪽에만 — 파일은 Python 이 정규화). */
export function persistUiPrefs(payload: Record<string, unknown>): void {
  if (!payload || typeof payload !== 'object' || Object.keys(payload).length === 0) return
  mirrorPrefsToStorage(payload)
  requestAction('save_ui_prefs', payload)
}

/** 태그 블록 모드 — 프롬프트 칸을 태그 블록으로 그린다 (기본 꺼짐). */
export const tagBlockMode = ref(readStoredBool('tagBlockMode', false))
/** 갤러리 메타데이터 패널 (기본 켜짐). */
export const galleryShowMetadata = ref(readStoredBool('galleryShowMetadata', true))

export function setTagBlockMode(value: boolean): void {
  const next = !!value
  tagBlockMode.value = next
  persistUiPrefs({ tagBlockMode: next })
}

export function setGalleryShowMetadata(value: boolean): void {
  const next = !!value
  galleryShowMetadata.value = next
  persistUiPrefs({ galleryShowMetadata: next })
}

/** 파일 값(부팅 uiPrefsLoaded · Settings 의 getUiPrefs)을 화면에 — 저장은 다시 하지 않는다. */
export function restoreUiFlagsFromPrefs(prefs: Record<string, unknown> | null | undefined): void {
  if (!prefs || typeof prefs !== 'object') return
  if (typeof prefs.tagBlockMode === 'boolean') tagBlockMode.value = prefs.tagBlockMode
  if (typeof prefs.galleryShowMetadata === 'boolean') galleryShowMetadata.value = prefs.galleryShowMetadata
}

// 웹 모드: 같은 브라우저의 다른 탭이 바꾼 값은 storage 이벤트로 온다(같은 문서의 변경은 안 온다).
try {
  if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
    window.addEventListener('storage', (event: StorageEvent) => {
      if (event.key === 'tagBlockMode' && (event.newValue === 'true' || event.newValue === 'false')) {
        tagBlockMode.value = event.newValue === 'true'
      } else if (event.key === 'galleryShowMetadata' && (event.newValue === 'true' || event.newValue === 'false')) {
        galleryShowMetadata.value = event.newValue === 'true'
      }
    })
  }
} catch { /* 테스트·비브라우저 환경 */ }
