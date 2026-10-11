/**
 * 연관 태그 패널의 순수 로직 — 질의 페이로드 · 넣을 글자 · 이벤트 검사 · 표시 문구.
 *
 * 브리지: related_tags_query / related_tags_build / related_tags_cancel → relatedTagsEvent
 * (ui/related_tags_actions.py · core/related_tags_service.py). 표시는 components/RelatedTagsControl.vue,
 * 상태는 composables/useRelatedTags.ts.
 */
import type {
  RelatedTagsBuildEvent, RelatedTagsBuildProgress, RelatedTagsBuildStatus, RelatedTagsEvent, RelatedTagsFields,
  RelatedTagsQueryEvent, RelatedTagsQueryPayload, RelatedTagsQueryState, RelatedTagsShard, RelatedTagSuggestion,
} from '../types/bridge'
import { planCompositionAppend } from './compositionPrompt'
import { underscoresToSpaces } from './promptComments'

/** 페이로드 칸 → 위젯 id(Python 프록시와 같은 이름). */
export const RELATED_TAG_FIELDS: Readonly<Record<keyof RelatedTagsFields, string>> = Object.freeze({
  main: 'main_prompt_text',
  character: 'character_input',
  copyright: 'copyright_input',
  count: 'char_count_input',
  prefix: 'prefix_prompt_text',
  suffix: 'suffix_prompt_text',
})
export const RELATED_NEGATIVE_FIELD = 'neg_prompt_text'
export const RELATED_EXCLUDE_FIELD = 'exclude_prompt_local_input'
export const RELATED_TAGS_LIMIT = 16

const SHARDS: readonly RelatedTagsShard[] = ['g', 's', 'q', 'e']
const QUERY_STATES: readonly RelatedTagsQueryState[] = ['ready', 'missing', 'unavailable', 'building', 'error']
const BUILD_STATES = ['building', 'done', 'canceled', 'error'] as const

function text(value: unknown): string {
  return typeof value === 'string' ? value : value == null ? '' : String(value)
}

/** 위젯 값 → 질의 페이로드(requestId 는 보낼 때 붙인다). */
export function buildRelatedTagsQuery(
  widgets: Record<string, unknown>, requestId: string, limit = RELATED_TAGS_LIMIT,
): RelatedTagsQueryPayload {
  const fields = {} as RelatedTagsFields
  for (const key of Object.keys(RELATED_TAG_FIELDS) as (keyof RelatedTagsFields)[]) {
    fields[key] = text(widgets[RELATED_TAG_FIELDS[key]])
  }
  return {
    requestId, fields, negative: text(widgets[RELATED_NEGATIVE_FIELD]),
    exclude: text(widgets[RELATED_EXCLUDE_FIELD]), limit,
  }
}

/** 같은 내용의 질의인가 — requestId 는 빼고 비교한다. */
export function querySignature(payload: RelatedTagsQueryPayload): string {
  const { requestId: _ignored, ...rest } = payload
  return JSON.stringify(rest)
}

/**
 * 넣을 글자 — 블록 모드는 공백, 텍스트 모드는 밑줄(각 모드의 자동완성과 같다). 괄호도 자동완성처럼 그대로 넣는다
 * (이스케이프는 정리 설정 '괄호 이스케이프'를 따른다). 밑줄→공백이 주석을 여는 태그(`a_//_b`)는 밑줄 그대로.
 */
export function relatedInsertText(tag: string, blockMode: boolean): string {
  return blockMode ? underscoresToSpaces(tag) : tag
}

/** 메인 태그 끝에 붙인 결과. 이미 어느 칸에 있으면 added=false(원문 그대로). */
export function planRelatedTagAppend(current: string, tag: string, blockMode: boolean, otherSections = '') {
  const plan = planCompositionAppend(current || '', [relatedInsertText(tag, blockMode)], otherSections)
  return { text: plan.text, added: plan.additions.length > 0 }
}

function isShard(value: unknown): value is RelatedTagsShard {
  return typeof value === 'string' && (SHARDS as readonly string[]).includes(value)
}

function shardList(value: unknown): RelatedTagsShard[] | null {
  if (!Array.isArray(value) || !value.every(isShard)) return null
  return [...value]
}

function stringList(value: unknown): string[] | null {
  return Array.isArray(value) && value.every(v => typeof v === 'string') ? [...value] : null
}

function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function parseProgress(value: unknown): RelatedTagsBuildProgress | null {
  if (!value || typeof value !== 'object') return null
  const p = value as Record<string, unknown>
  if (!isShard(p.shard) || !finite(p.index) || !finite(p.count) || !finite(p.progress) || !finite(p.overall)) return null
  return { shard: p.shard, index: p.index, count: p.count, progress: p.progress, overall: p.overall }
}

function parseSuggestion(value: unknown): RelatedTagSuggestion | null {
  if (!value || typeof value !== 'object') return null
  const s = value as Record<string, unknown>
  const via = stringList(s.via)
  if (typeof s.tag !== 'string' || !s.tag || typeof s.ko !== 'string' || !finite(s.count) || !via) return null
  return { tag: s.tag, ko: s.ko, count: s.count, via: via.slice(0, 3) }
}

/** relatedTagsEvent JSON → 이벤트. 모양이 하나라도 어긋나면 null(화면을 깨뜨리지 않고 버린다). */
export function parseRelatedTagsEvent(raw: unknown): RelatedTagsEvent | null {
  let value: unknown = raw
  if (typeof raw === 'string') {
    try { value = JSON.parse(raw) } catch { return null }
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const e = value as Record<string, unknown>
  if (typeof e.requestId !== 'string') return null
  const error = typeof e.error === 'string' ? e.error : undefined
  if (e.action === 'related_tags_build') {
    if (!(BUILD_STATES as readonly unknown[]).includes(e.state)) return null
    const event: RelatedTagsBuildEvent = { action: 'related_tags_build', requestId: e.requestId,
      state: e.state as RelatedTagsBuildEvent['state'] }
    if (e.progress !== undefined) {
      const progress = parseProgress(e.progress)
      if (!progress) return null
      event.progress = progress
    }
    if (e.shards !== undefined) {
      const shards = shardList(e.shards)
      if (!shards) return null
      event.shards = shards
    }
    if (error !== undefined) event.error = error
    return event
  }
  if (e.action !== 'related_tags_query') return null
  if (!(QUERY_STATES as readonly unknown[]).includes(e.state)) return null
  const ratings = shardList(e.ratings)
  const missing = shardList(e.missing)
  const stale = shardList(e.stale)
  const unavailable = e.unavailable === undefined ? [] : shardList(e.unavailable)
  const context = stringList(e.context)
  if (!ratings || !missing || !stale || !unavailable || !context || !Array.isArray(e.tags)) return null
  const tags = e.tags.map(parseSuggestion)
  if (tags.some(t => t === null)) return null
  let build: RelatedTagsBuildStatus | null = null
  if (e.build !== null && e.build !== undefined) {
    const progress = parseProgress(e.build)
    const id = (e.build as Record<string, unknown>).requestId
    if (!progress || typeof id !== 'string') return null
    build = { ...progress, requestId: id }
  }
  const event: RelatedTagsQueryEvent = {
    action: 'related_tags_query', requestId: e.requestId, state: e.state as RelatedTagsQueryState,
    tags: tags as RelatedTagSuggestion[], context, ratings, missing, stale, unavailable,
    estimateSec: finite(e.estimateSec) && e.estimateSec > 0 ? e.estimateSec : 0, build,
  }
  if (error !== undefined) event.error = error
  return event
}

/** 칩 글자 — 밑줄을 공백으로. */
export function relatedTagLabel(tag: string): string {
  return tag.replace(/_/g, ' ')
}

/** 칩 툴팁 — 게시물 수와 함께 쓰인 기준 태그. */
export function relatedTagTitle(item: RelatedTagSuggestion): string {
  const parts = [`${relatedTagLabel(item.tag)} · 게시물 ${item.count.toLocaleString('ko-KR')}개`]
  if (item.via.length) parts.push(`함께 쓰인 태그: ${item.via.map(relatedTagLabel).join(', ')}`)
  return parts.join('\n')
}

/** ['g','s'] → 'G · S' */
export function formatShards(shards: readonly string[]): string {
  return shards.map(s => s.toUpperCase()).join(' · ')
}

/** 대략 시간 — 1분 미만은 10초 단위 '약 30초', 그 이상은 올린 분 '약 3분'. 0 이하는 ''. */
export function formatEstimate(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return ''
  if (seconds < 60) return `약 ${Math.max(10, Math.round(seconds / 10) * 10)}초`
  return `약 ${Math.ceil(seconds / 60)}분`
}

/** '표 만드는 중… S 42% (2/2)' */
export function formatBuildProgress(progress: RelatedTagsBuildProgress): string {
  const percent = Math.max(0, Math.min(100, Math.floor(progress.progress * 100)))
  return `표 만드는 중… ${progress.shard.toUpperCase()} ${percent}% (${progress.index + 1}/${Math.max(1, progress.count)})`
}
