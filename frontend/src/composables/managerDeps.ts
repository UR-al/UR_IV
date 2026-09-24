/**
 * 스튜디오 도구 매니저(프리셋 · 가중치 · 와일드카드 · 프로파일 · 순서 · 즉석 WC · 통계)가 받는 의존성.
 *
 * 매니저 composable 은 **모듈 싱글턴**이다 — 모달은 v-if 로 열고 닫는데, 상태와 백엔드 리스너가
 * 모달 안에 있으면 닫혀 있는 동안 이벤트를 놓친다(글로벌 가중치 적용 · 프로파일 드롭다운 · PromptPanel
 * 의 와일드카드 열기가 모달 밖에서 이 상태를 쓴다). 그래서 상태는 여기, 모달은 표시만 한다.
 * 테스트는 `create*` 팩토리에 가짜 의존성을 넣고, 앱은 `use*()` 가 진짜 의존성으로 한 번 만든다.
 */
import { getBackend, onBackendEvent } from '../bridge.js'
import { requestAction, useWidgetStore } from '../stores/widgetStore.js'
import { useToasts } from './useToasts'
import type { ActionName, ActionPayload, BackendEvent } from '../types/bridge'

export type RequestActionFn = <K extends ActionName>(name: K, payload?: ActionPayload<K>) => void
export type OnBackendEventFn = (name: BackendEvent, cb: (...args: any[]) => void) => (() => void) | void
export type GetBackendFn = () => Promise<any>
export type AddToastFn = (type: string, msg: string) => void
/** 위젯 스토어(Python 프록시와 동기화되는 값) — 키마다 문자열 */
export type WidgetValues = Record<string, any>

export interface ManagerDeps {
  getBackend: GetBackendFn
  onBackendEvent: OnBackendEventFn
  requestAction: RequestActionFn
  addToast: AddToastFn
  storeWidgets: WidgetValues
}

/** 앱에서 쓰는 진짜 의존성 — 싱글턴을 처음 만들 때 한 번 읽는다. */
export function appManagerDeps(): ManagerDeps {
  return {
    getBackend,
    onBackendEvent,
    requestAction,
    addToast: useToasts().addToast,
    storeWidgets: useWidgetStore().widgets,
  }
}
