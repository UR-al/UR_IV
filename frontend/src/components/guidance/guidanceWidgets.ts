/**
 * Anima 가이던스 섹션(components/guidance/*Section.vue) 공용 — 위젯 스토어 읽기·쓰기.
 *
 * 위젯 id 는 `_` + core/anima_guidance.py 의 스펙 키다(예: `_guid_enabled`). 실제 alwayson_scripts 인자
 * 배열은 백엔드가 그 스펙 순서대로 만든다 — 확장이 args 를 **위치로만** 읽으므로 순서를 프론트에서 다루지
 * 않는다. 섹션은 이 스토어 객체(reactive)의 키를 v-model 로 바로 쓴다.
 */
export type GuidanceWidgets = Record<string, any>

export function useGuidanceWidgets(props: { widgets: GuidanceWidgets }) {
  const w = props.widgets

  // 스토어는 값을 문자열로 들고 있다 ('true'/'false') — 백엔드 coercion 과 동일 규칙
  function b(key: string): boolean {
    return String(w[`_${key}`] ?? '') === 'true'
  }
  function setB(key: string, val: boolean) {
    w[`_${key}`] = val ? 'true' : 'false'
  }

  return { w, b, setB }
}
