// 이벤트 생성 — Parent 스텝의 외모·의상·배경 태그를 Child 스텝에 이어 붙이는(carry) 순수 로직.
//
// 스텝 카드 표시, '큐에 추가'/'지금 생성', 'T2I로 전송'이 모두 같은 프롬프트를 써야 한다.
// 예전엔 T2I 전송만 원본 step.prompt 를 보내서, 카드에 보이던 carry 태그가 T2I 에서 빠졌다.
// 그래서 carry 적용과 '실제로 쓸 프롬프트' 선택을 한곳(effectiveStepPrompt)으로 모은다.

export type EventTagCategory = 'appearance' | 'costume' | 'background' | 'other'

export interface EventCarryOptions {
  appearance: boolean
  costume: boolean
  background: boolean
}

export interface EventStepLike {
  prompt?: string
  displayPrompt?: string
  [k: string]: any
}

// 외모/의상/배경 태그 분류용 키워드 (부분 문자열 매칭, 앞 분류가 우선)
export const APPEARANCE_KEYS = ['hair', 'eyes', 'skin', 'ears', 'horns', 'tail', 'wings', 'fang', 'mole', 'scar', 'freckle', 'eyelash', 'pupil', 'iris', 'ahoge', 'bangs', 'sidelocks', 'ponytail', 'twintails', 'braid', 'bun', 'bob', 'short hair', 'long hair', 'medium hair']
export const COSTUME_KEYS = ['dress', 'shirt', 'skirt', 'pants', 'uniform', 'armor', 'suit', 'coat', 'jacket', 'hat', 'ribbon', 'bow', 'gloves', 'boots', 'shoes', 'socks', 'stockings', 'thighhighs', 'pantyhose', 'bikini', 'swimsuit', 'cape', 'scarf', 'necktie', 'collar', 'headband', 'hairclip', 'earrings', 'necklace', 'bracelet', 'belt', 'glasses', 'mask', 'hood', 'apron', 'maid', 'school uniform', 'sailor', 'kimono', 'yukata']
export const BG_KEYS = ['background', 'outdoors', 'indoors', 'sky', 'cloud', 'tree', 'grass', 'water', 'ocean', 'beach', 'mountain', 'city', 'room', 'bed', 'floor', 'wall', 'window', 'night', 'day', 'sunset', 'sunrise', 'rain', 'snow', 'forest', 'garden', 'street', 'school', 'classroom', 'library', 'kitchen', 'bathroom', 'rooftop', 'bridge', 'castle', 'temple', 'church']

export function classifyEventTag(tag: string): EventTagCategory {
  const t = tag.toLowerCase()
  if (APPEARANCE_KEYS.some(k => t.includes(k))) return 'appearance'
  if (COSTUME_KEYS.some(k => t.includes(k))) return 'costume'
  if (BG_KEYS.some(k => t.includes(k))) return 'background'
  return 'other'
}

function splitTags(prompt: string | undefined): string[] {
  return (prompt || '').split(',').map(t => t.trim()).filter(Boolean)
}

/**
 * carry 옵션을 적용해 각 스텝에 displayPrompt 를 붙인다. 첫 스텝(Parent)은 그대로다.
 * Child 는 자기 태그 뒤에, 켜진 분류의 Parent 태그 중 아직 없는 것(대소문자 무시)을 붙인다.
 */
export function applyEventCarry<T extends EventStepLike>(
  steps: readonly T[],
  opts: EventCarryOptions,
): Array<T & { displayPrompt: string }> {
  if (!steps.length) return []
  const parentByType: Record<EventTagCategory, string[]> = {
    appearance: [], costume: [], background: [], other: [],
  }
  for (const t of splitTags(steps[0]?.prompt)) parentByType[classifyEventTag(t)].push(t)

  const carried: string[] = [
    ...(opts.appearance ? parentByType.appearance : []),
    ...(opts.costume ? parentByType.costume : []),
    ...(opts.background ? parentByType.background : []),
  ]

  return steps.map((step, i) => {
    if (i === 0) return { ...step, displayPrompt: step.prompt || '' }
    const tags = splitTags(step.prompt)
    const seen = new Set(tags.map(t => t.toLowerCase()))
    for (const t of carried) {
      const key = t.toLowerCase()
      if (seen.has(key)) continue
      seen.add(key)
      tags.push(t)
    }
    return { ...step, displayPrompt: tags.join(', ') }
  })
}

/** 화면·큐·T2I 전송이 모두 쓰는 '실제 프롬프트' — carry 적용본 우선, 없으면 원본. */
export function effectiveStepPrompt(step: EventStepLike | null | undefined): string {
  if (!step) return ''
  return step.displayPrompt || step.prompt || ''
}
