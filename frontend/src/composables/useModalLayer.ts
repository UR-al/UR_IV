import { onBeforeUnmount, onMounted, watch } from 'vue'
import { appModalStack, type ModalStack } from '../utils/modalStack'

export interface ModalLayerOptions {
  /**
   * 모달이 "열려 있는가". 없으면 마운트돼 있는 동안 열린 것으로 본다(v-if 로만 여닫는 모달).
   * `<transition>` 안의 모달은 닫힌 뒤에도 나가는 애니메이션 동안 마운트돼 있으므로 이 값을 준다 —
   * 플래그가 꺼지는 즉시 스택에서 내려야 ↑/↓ · ESC 가 한 박자 늦지 않는다.
   */
  isOpen?: () => boolean
  /** 전역 ESC(App.vue)가 이 모달을 닫을 때 부르는 함수. 없으면 모달이 제 ESC 를 직접 처리한다. */
  close?: () => void
  /** 테스트용 — 기본은 앱 전체 스택. */
  stack?: ModalStack
}

/**
 * 모달이 열려 있는 동안 앱 모달 스택(utils/modalStack)에 올라가 있게 한다.
 * 모달 컴포넌트의 setup 에서 한 번 부른다 — 여닫는 곳이 어디든(툴 버튼 · PromptPanel · 헤더)
 * App 의 전역 단축키가 이 모달을 알게 된다.
 */
export function useModalLayer(options: ModalLayerOptions = {}): void {
  const stack = options.stack ?? appModalStack
  let release: (() => void) | null = null
  let mounted = false

  function sync() {
    const open = mounted && (options.isOpen ? options.isOpen() : true)
    if (open && !release) {
      release = stack.open({ close: options.close })
    } else if (!open && release) {
      release()
      release = null
    }
  }

  onMounted(() => { mounted = true; sync() })
  // 동기 flush — 플래그를 끈 바로 다음 키 입력부터 스택이 맞아야 한다
  if (options.isOpen) watch(options.isOpen, sync, { flush: 'sync' })
  onBeforeUnmount(() => { mounted = false; sync() })
}
