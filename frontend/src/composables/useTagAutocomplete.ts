import { ref } from 'vue'
import { getBackend } from '../bridge.js'
import { fetchTagSuggestions, hasHangul, minSuggestPrefix, type TagSuggestion } from '../utils/tagSuggest'

/**
 * 태그 자동완성 요청 수명주기 — PromptPanel(텍스트 모드 3칸)과 TagBlockField(블록 모드)가 같이 쓴다.
 *
 * 예전엔 두 컴포넌트가 같은 로직을 복제해 들고 있었고, 둘 다
 *  - 닫는 경로(최소 길이 미만·수락·Escape·blur·블록 추가)에서 후보만 비우고 디바운스 타이머는
 *    그대로 둬서 250~300ms 뒤 **빈 칸이나 포커스 없는 칸 아래에 팝업이 되살아났고**,
 *  - 콜백이 '아직 유효한 요청인가'를 보지 않아 'lo'→'l' 로 지워도 'lo' 의 후보가 떴다.
 *    (그 상태에서 Enter/Tab 이 첫 후보를 끼워 넣었다.)
 * 여기서는 요청마다 번호(seq)를 매기고, 닫을 때 타이머 취소 + 번호 증가로 날아오는 중인
 * 응답까지 무효로 만든다. 팝업의 주인(owner)도 **응답이 도착할 때** 정해져, 다른 칸의 후보가
 * 지금 칸에 뜨지 않는다.
 * 새 질의가 시작되면 **이미 떠 있는 옛 질의의 후보**도 새 질의에 맞는 것(영문 접두 일치)만 남기고
 * 나머지는 내린다 — 예전엔 'bl' 후보(black_hair)가 떠 있는 채로 'blue' 를 치고 곧바로 Enter/Tab 을
 * 누르면 새 응답이 오기 전이라 black_hair 가 들어갔다(텍스트 모드·블록 모드 모두).
 *
 * 수락 표기(밑줄/공백)는 필드마다 다르게 유지하므로 여기서 다루지 않는다 — 각 컴포넌트 몫.
 */

export type TagSuggestionFetcher = (prefix: string, cb: (items: TagSuggestion[]) => void) => void | Promise<void>

export interface TagAutocompleteOptions {
  /** 후보 조회 — 기본은 브리지(getTagSuggestionsRich). 테스트는 가짜를 주입한다. */
  fetch?: TagSuggestionFetcher
  /** 입력 디바운스(ms) */
  delay?: number
}

/** 접두 비교 키 — 대소문자·밑줄/공백 표기 차이를 무시한다('long h' 는 long_hair 의 접두). */
function matchKey(s: string): string {
  return String(s ?? '').toLowerCase().replace(/_/g, ' ')
}

/** 브리지로 후보를 묻는다 — 브리지가 없거나 실패하면 빈 목록. */
export const fetchTagSuggestionsViaBridge: TagSuggestionFetcher = async (prefix, cb) => {
  let backend: any = null
  try { backend = await getBackend() } catch { backend = null }
  fetchTagSuggestions(backend, prefix, cb)
}

export function useTagAutocomplete(options: TagAutocompleteOptions = {}) {
  const fetcher = options.fetch || fetchTagSuggestionsViaBridge
  const delay = options.delay ?? 250
  const items = ref<TagSuggestion[]>([])
  const index = ref(0)
  /** 지금 떠 있는 후보의 질의가 한글인가 — 한글 검색이면 Enter 는 후보를 고르지 않는다 */
  const queryHangul = ref(false)
  /** 지금 떠 있는 팝업의 주인(필드 id). 응답이 도착한 요청의 owner 로 정해진다. */
  const owner = ref('')
  /** 지금 떠 있는 후보가 어느 질의의 것인가 — 새 질의가 오면 이것과 비교해 옛 후보를 거른다 */
  let shownPrefix = ''
  let seq = 0
  let timer: ReturnType<typeof setTimeout> | null = null

  function clearTimer() {
    if (timer !== null) { clearTimeout(timer); timer = null }
  }

  /** 떠 있는 후보만 내린다(대기 중인 요청은 그대로). */
  function hideShown() {
    items.value = []
    index.value = 0
    queryHangul.value = false
    owner.value = ''
    shownPrefix = ''
  }

  /** 팝업을 닫고 대기 중인 디바운스·날아오는 응답을 모두 무효로 만든다. */
  function close() {
    clearTimer()
    seq++
    hideShown()
  }

  /**
   * 새 질의가 시작됐다 — 떠 있는 후보 중 새 질의에도 맞는 것만 남긴다(새 응답이 올 때까지).
   * 같은 칸·같은 질의면(뒤 공백·IME 무변화 입력) 그대로 둔다. 다른 칸이거나 한글 질의(번역 후보라
   * 접두로 거를 수 없다)면 모두 내린다. 남는 게 없으면 팝업이 닫혀 Enter/Tab 이 기본 동작을 한다.
   */
  function narrowShown(prefix: string, ownerId: string) {
    if (!items.value.length) return
    if (ownerId === owner.value && prefix === shownPrefix) return
    const want = matchKey(prefix)
    const keep = ownerId === owner.value && !queryHangul.value && !hasHangul(prefix)
      ? items.value.filter(s => matchKey(s.tag).startsWith(want))
      : []
    if (!keep.length) { hideShown(); return }
    const current = items.value[index.value]
    items.value = keep
    index.value = Math.max(0, current ? keep.indexOf(current) : 0)
    shownPrefix = prefix
  }

  /**
   * ``query`` 로 후보를 요청한다(디바운스). 최소 길이 미만이면 팝업을 닫는다.
   * 새 요청은 앞선 요청을 무효로 만든다 — 늦게 온 옛 응답은 버려지고, 이미 떠 있는 옛 후보도
   * 새 질의에 맞는 것만 남는다(narrowShown).
   */
  function request(query: string, ownerId = '') {
    const prefix = String(query ?? '').trim()
    if (prefix.length < minSuggestPrefix(prefix)) { close(); return }
    clearTimer()
    narrowShown(prefix, ownerId)
    const my = ++seq
    timer = setTimeout(() => {
      timer = null
      if (my !== seq) return
      const deliver = (list: TagSuggestion[]) => {
        if (my !== seq) return          // 그 사이 닫혔거나 새 입력이 들어왔다
        items.value = Array.isArray(list) ? list : []
        index.value = 0
        queryHangul.value = hasHangul(prefix)
        owner.value = items.value.length ? ownerId : ''
        shownPrefix = items.value.length ? prefix : ''
      }
      try {
        const done = fetcher(prefix, deliver)
        if (done && typeof (done as Promise<void>).catch === 'function') {
          (done as Promise<void>).catch(() => deliver([]))
        }
      } catch {
        deliver([])
      }
    }, delay)
  }

  function isOpenFor(ownerId = ''): boolean {
    return items.value.length > 0 && owner.value === ownerId
  }

  /** 선택 이동 — 목록 끝에서 멈춘다 */
  function move(delta: number) {
    if (!items.value.length) return
    index.value = Math.max(0, Math.min(items.value.length - 1, index.value + delta))
  }

  function selected(): TagSuggestion | undefined {
    return items.value[index.value]
  }

  return { items, index, queryHangul, owner, request, close, isOpenFor, move, selected }
}
