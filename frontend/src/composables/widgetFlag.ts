import { computed, type WritableComputedRef } from 'vue'

/**
 * 위젯 스토어의 'true'/'false' 문자열 값을 토글(v-model)용 boolean 으로 잇는다.
 *
 * Python 프록시(CheckBoxProxy · GroupBoxProxy)는 체크 상태를 문자열로 주고받는다. 읽을 때는
 * 정확히 'true' 만 켜짐, 쓸 때는 'true'/'false' — 파라미터 열의 모든 토글이 같은 규칙이라
 * 한 곳에 둔다. `onSet` 은 값을 쓴 **뒤** 부른다(랜덤 해상도를 켜면 목록을 받는 식).
 */
export function widgetFlag(
  widgets: Record<string, any>,
  key: string,
  onSet?: (value: boolean) => void,
): WritableComputedRef<boolean> {
  return computed({
    get: () => widgets[key] === 'true',
    set: (value: boolean) => {
      widgets[key] = value ? 'true' : 'false'
      onSet?.(value)
    },
  })
}
