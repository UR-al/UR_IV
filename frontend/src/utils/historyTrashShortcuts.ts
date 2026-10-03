import { isEditableElement, type FocusedElementLike } from './appShortcuts'
import { isImeComposing } from './imeComposition'

export type HistoryTrashKeyEvent = Pick<KeyboardEvent,
  'key' | 'ctrlKey' | 'metaKey' | 'altKey' | 'shiftKey' | 'defaultPrevented' |
  'repeat' | 'isComposing' | 'preventDefault' | 'stopPropagation'> & {
    target?: unknown
    keyCode?: number
  }

export interface HistoryTrashShortcutDeps {
  isEnabled: () => boolean
  hasModal: () => boolean
  activeElement: () => FocusedElementLike | null
  deleteSelected: () => boolean
  undo: () => boolean
}

/** 히스토리 영역의 bubble keydown 전용. 전역/에디터/텍스트 입력의 Undo 는 가로채지 않는다. */
export function createHistoryTrashKeydownHandler(deps: HistoryTrashShortcutDeps): (event: HistoryTrashKeyEvent) => void {
  return (event) => {
    if (event.defaultPrevented || event.repeat || isImeComposing(event)) return
    if (!deps.isEnabled() || deps.hasModal()) return
    if (isEditableElement(deps.activeElement()) || isEditableElement(event.target as FocusedElementLike | null)) return
    const bareDelete = event.key === 'Delete' && !event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey
    const undo = event.key.toLowerCase() === 'z' && (event.ctrlKey || event.metaKey) && !event.altKey && !event.shiftKey
    const handled = bareDelete ? deps.deleteSelected() : undo ? deps.undo() : false
    if (handled) {
      event.preventDefault()
      event.stopPropagation()
    }
  }
}
