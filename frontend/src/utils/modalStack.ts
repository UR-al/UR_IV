/**
 * 열린 모달의 스택 — 전역 ESC · ↑/↓ 히스토리 이동이 "지금 모달이 떠 있나"를 묻는 한 곳.
 *
 * 예전엔 App.vue 가 매니저 모달 플래그 7개를 ESC 처리와 ↑/↓ 가드(anyModal)에 **두 번** 나열했고,
 * 그 목록이 이미 어긋나 있었다 — LoRA · 조건부 · A/B · 캐릭터 프리셋 · override 모달은 목록에
 * 없어서, 그 모달을 띄운 채 ↑/↓ 를 누르면 뒤의 히스토리가 넘어갔다. 이제 모달이 열릴 때 스스로
 * 여기 올라오고(composables/useModalLayer) 닫힐 때 내려가므로, 새 모달이 생겨도 목록을 고칠 곳이 없다.
 *
 * `close` 가 있는 층은 전역 ESC 가 닫는다(맨 위 하나만). `close` 가 없는 층은 제 ESC 를 스스로
 * 처리하는 모달(window capture + stopPropagation)이라, 맨 위에 있으면 ESC 를 "먹기만" 한다 —
 * 뒤의 파라미터 열 같은 것을 대신 닫지 않게.
 */

export interface ModalLayer {
  /** 전역 ESC 로 닫을 때 부르는 함수. 없으면 모달이 제 ESC 를 직접 처리한다. */
  close?: () => void
}

export interface ModalStack {
  /** 층을 올린다. 돌려받은 함수로 내린다(두 번 불러도 안전). */
  open(layer: ModalLayer): () => void
  /** 전역 ESC — 모달이 하나라도 있으면 맨 위 층을 닫고(닫을 수 있으면) true. 없으면 false. */
  closeTop(): boolean
  /** ↑/↓ 히스토리 이동 같은 뒤쪽 단축키를 막아야 하는가. */
  isAnyOpen(): boolean
  size(): number
}

export function createModalStack(): ModalStack {
  const layers: ModalLayer[] = []
  return {
    open(layer) {
      const entry: ModalLayer = { close: layer.close }
      layers.push(entry)
      let released = false
      return () => {
        if (released) return
        released = true
        const index = layers.lastIndexOf(entry)
        if (index >= 0) layers.splice(index, 1)
      }
    },
    closeTop() {
      const top = layers[layers.length - 1]
      if (!top) return false
      top.close?.()
      return true
    },
    isAnyOpen: () => layers.length > 0,
    size: () => layers.length,
  }
}

/** 앱 전체가 함께 쓰는 스택 — App.vue 의 전역 keydown 이 읽는다. */
export const appModalStack = createModalStack()
