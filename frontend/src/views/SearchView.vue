<template>
  <div class="search-view">
    <!-- 검색 결과 0건: 안내 + 메인으로 돌아가기 -->
    <div v-if="results.length === 0 && !searching && hasSearched" class="no-results">
      <div class="nr-icon"><Icon name="search" /></div>
      <h2>검색 결과가 없습니다</h2>
      <p class="nr-hint">
        조건을 조금 더 느슨하게 하거나, <strong class="nr-accent">∪ OR 모드</strong>로 전환해보세요.<br>
        정확 매칭(<code>*word</code>)을 풀거나 제외 검색을 비워보는 것도 도움이 됩니다.
      </p>
      <div class="nr-actions">
        <button class="nr-btn primary" @click="hasSearched = false">
          ← 검색 화면으로
        </button>
        <button class="nr-btn" v-if="combineMode === 'and'"
          @click="combineMode = 'or'; search()">
          ∪ OR 모드로 재검색
        </button>
      </div>
      <div class="nr-summary" v-if="lastQuerySummary">
        <span class="nr-label">방금 검색:</span>
        <code>{{ lastQuerySummary }}</code>
      </div>
    </div>

    <!-- 검색 전: 중앙 검색 폼 -->
    <div v-else-if="results.length === 0 && !searching" class="welcome">
      <div class="ws-header">
        <div class="ws-icon"><Icon name="search" /></div>
        <h1>TAG EXPLORER</h1>
        <p>Danbooru 데이터베이스에서 프롬프트를 검색합니다</p>
      </div>

      <div class="search-form">
        <!-- Include -->
        <div class="form-section">
          <div class="form-row">
            <div class="form-field wide">
              <label>캐릭터</label>
              <input v-model="fields[1].include" placeholder="e.g. hatsune_miku, raiden_shogun" @keydown.enter="search" />
            </div>
            <div class="form-field">
              <label>작품</label>
              <input v-model="fields[0].include" placeholder="e.g. genshin_impact" @keydown.enter="search" />
            </div>
          </div>
          <div class="form-row">
            <div class="form-field wide">
              <label>일반 태그</label>
              <input v-model="fields[3].include" placeholder="e.g. 1girl, blue_hair, sword, outdoors" @keydown.enter="search" />
            </div>
            <div class="form-field">
              <label>작가</label>
              <input v-model="fields[2].include" placeholder="Artist name..." @keydown.enter="search" />
            </div>
          </div>
        </div>

        <!-- Exclude (접이식) -->
        <details class="form-section exclude-section" open>
          <summary class="exclude-toggle">제외 태그 <Icon name="chevron-down" size="12" /></summary>
          <div class="form-row">
            <div class="form-field"><label class="danger">캐릭터</label><input v-model="fields[1].exclude" placeholder="제외..." @keydown.enter="search" /></div>
            <div class="form-field"><label class="danger">작품</label><input v-model="fields[0].exclude" placeholder="제외..." @keydown.enter="search" /></div>
            <div class="form-field"><label class="danger">태그</label><input v-model="fields[3].exclude" placeholder="제외..." @keydown.enter="search" /></div>
            <div class="form-field"><label class="danger">작가</label><input v-model="fields[2].exclude" placeholder="제외..." @keydown.enter="search" /></div>
          </div>
        </details>

        <!-- Rating + 필드 결합 모드 + Go -->
        <div class="form-footer">
          <div class="rating-row">
            <button v-for="r in ratings" :key="r.key" class="rating-chip"
              :class="{ active: r.checked }" @click="r.checked = !r.checked">{{ r.label }}</button>
          </div>
          <!-- 결과 cap 토글 — UI 부하 방지 / 무제한 -->
          <div class="combine-row">
            <span class="combine-label">결과 제한:</span>
            <button class="combine-chip"
              :class="{ active: resultCapMode === 'capped' }"
              @click="resultCapMode = 'capped'"
              title="50만 건 cap (권장) — 그 이상은 무작위 샘플링&#10;UI 안정성 보장">
              50만 건
            </button>
            <button class="combine-chip warn-chip"
              :class="{ active: resultCapMode === 'unlimited' }"
              @click="resultCapMode = 'unlimited'"
              title="모든 결과를 받음 — JSON 직렬화 + Vue 메모리가 수 GB 가능&#10;⚠ 대규모 결과 시 화면 멈춤/렉 가능">
              무제한 ⚠
            </button>
          </div>

          <!-- 검색 결합 모드: AND (교집합) / OR (합집합) -->
          <!-- 모드는 필드 간 결합 + 필드 내 콤마 토큰 + 제외 검색에 모두 적용됨 -->
          <div class="combine-row">
            <span class="combine-label">검색 결합:</span>
            <button class="combine-chip" :class="{ active: combineMode === 'and' }"
              @click="combineMode = 'and'"
              title="AND 모드 — 모든 조건 동시 만족 (교집합)&#10;&#10;콤마(,) = AND&#10;필드 간 = AND&#10;예) char=raiden_shogun + copy=genshin → 라이덴 쇼군만&#10;예) char=1girl, blue_hair → 1girl 이면서 blue_hair인 행">
              ∩ AND
            </button>
            <button class="combine-chip" :class="{ active: combineMode === 'or' }"
              @click="combineMode = 'or'"
              title="OR 모드 — 하나라도 만족하면 통과 (합집합)&#10;&#10;콤마(,) = OR (모드 따라 자동 전환)&#10;필드 간 = OR&#10;예) char=ike + copy=arknights → Ike 또는 Arknights 작품 전체&#10;예) char=char1, char2 → char1 또는 char2 일러스트&#10;&#10;💡 명시적 [A,B] 그룹은 모드 무관 항상 AND&#10;💡 정확 매칭은 *ike 처럼 별표 사용">
              ∪ OR
            </button>
          </div>
          <div class="io-row">
            <button class="io-btn" v-host-dialog="'import_search_results'" @click="importResults"><Icon name="upload" /> .parquet 가져오기</button>
          </div>
          <button class="go-btn" @click="search" :disabled="searching"><Icon name="rocket" /> 검색</button>
        </div>
      </div>

      <div class="hints">
        <span>쉼표(,) = {{ combineMode === 'or' ? 'OR (모드 따라)' : 'AND (모드 따라)' }}</span>
        <span>[A|B] = OR (명시)</span>
        <span>[A,B] = AND (명시)</span>
        <span :title="'마지막 | 다음을 비워두면 → 이 필드 조건은 매칭 안 되어도 통과\n예: copyright=[tots|alc|] → TOT/ALC 매치 또는 무관\n\n⚠ 중간 ||(이중 파이프)는 wildcard 아님 — 그냥 빈 토큰 무시\n예: character=[A||B] → A 또는 B만 (필드 조건 면제 안 됨)'">[A|B|<b>↵</b>] = 끝에 비우면 필드 면제</span>
        <span>Exclude로 제외 조건 설정</span>
        <span class="hint-mode" :class="combineMode">
          현재: <strong>{{ combineMode === 'and' ? 'AND' : 'OR' }}</strong> · <strong>최신 {{ activeDatasetDisplay }}</strong>
          · <strong>{{ resultCapMode === 'capped' ? 'cap 50만' : 'cap OFF ⚠' }}</strong>
        </span>
      </div>

      <!-- 이전 검색 결과 있으면 바로 보기/랜덤 뽑기 -->
      <div class="prev-results" v-if="lastResults.length > 0">
        <span class="prev-label">이전 검색 결과 {{ lastResults.length }}건</span>
        <button class="prev-btn" @click="restoreLastResults"><Icon name="clipboard" /> 목록 보기</button>
        <button class="prev-btn gold" @click="restoreAndRandom"><Icon name="dice" /> 랜덤 뽑기</button>
      </div>
    </div>

    <!-- 검색 중 -->
    <div v-else-if="results.length === 0 && searching" class="loading">
      <!-- 원형 진행률 — 중앙에 % 표시 -->
      <div class="ring-loader">
        <svg viewBox="0 0 100 100">
          <!-- 배경 트랙 -->
          <circle cx="50" cy="50" r="42" class="ring-track" />
          <!-- 진행 호 — stroke-dasharray로 진행률 시각화 -->
          <circle cx="50" cy="50" r="42" class="ring-progress"
            :stroke-dashoffset="ringOffset" />
        </svg>
        <div class="ring-center">
          <!-- 표시값은 항상 [0, 100] 범위로 클램프 — 가짜 진행률 오차 흡수 -->
          <div class="ring-pct">{{ Math.min(100, Math.max(0, Math.round(searchProgress))) }}%</div>
        </div>
      </div>
      <h2>검색 중...</h2>
      <p>{{ statusText }}</p>
    </div>

    <!-- 검색 후: 결과 -->
    <template v-else>
      <!-- 상단 바 -->
      <div class="result-bar">
        <div class="bar-left">
          <button class="bar-btn" @click="viewMode = 'single'" :class="{ active: viewMode === 'single' }"><Icon name="file" /> Single</button>
          <button class="bar-btn" @click="viewMode = 'list'" :class="{ active: viewMode === 'list' }"><Icon name="clipboard" /> List</button>
          <span class="bar-sep">|</span>
          <button class="bar-btn" @click="prevResult" :disabled="previewIdx <= 0"><Icon name="chevron-left" /></button>
          <span class="bar-idx"><b>{{ previewIdx + 1 }}</b>/{{ filteredResults.length }}</span>
          <button class="bar-btn" @click="nextResult" :disabled="previewIdx >= filteredResults.length - 1"><Icon name="play" /></button>
          <button class="bar-btn gold" @click="randomResult"><Icon name="dice" /> Random</button>
        </div>
        <div class="bar-right">
          <button class="bar-btn parquet" v-host-dialog="'export_search_results'" @click="exportResults"><Icon name="download" /> .parquet 내보내기</button>
          <button class="bar-btn" @click="newSearch"><Icon name="search" /> 새 검색</button>
        </div>
      </div>
      <!-- 심층검색 바 + 분기 -->
      <div class="deep-bar">
        <span class="deep-label">정밀 검색</span>
        <input v-model="deepInclude" placeholder="포함 태그 (쉼표 구분)" @keydown.enter="applyDeepSearch" class="deep-input" />
        <input v-model="deepExclude" placeholder="제외 태그 (쉼표 구분)" @keydown.enter="applyDeepSearch" class="deep-input neg" />
        <button class="bar-btn" @click="applyDeepSearch">필터</button>
        <template v-if="filterHistory.length > 0">
          <span class="bar-sep">|</span>
          <span class="branch-label">분기:</span>
          <button v-for="(fh, fi) in filterHistory" :key="fi" class="branch-btn"
            @click="restoreBranch(fi)">{{ fh.label }} ({{ fh.count }})</button>
        </template>
        <button class="bar-btn" @click="resetDeepSearch" v-if="isFiltered"><Icon name="close" /> 초기화</button>
        <button class="bar-btn filter-btn" :class="{ active: hasActiveFilters }" @click="showFilterManager = true">
          <Icon name="filter" /> FILTER{{ hasActiveFilters ? ` (${activeFilterCount})` : '' }}
        </button>
        <button v-if="hasActiveFilters" class="bar-btn" @click="clearAllFilters"><Icon name="close" /> 필터 해제</button>
        <span class="bar-count">{{ filteredResults.length }} / {{ results.length }}</span>
      </div>

      <!-- Single View -->
      <div v-if="viewMode === 'single'" class="single-view">
        <div class="detail-card" v-if="currentResult">
          <div class="detail-meta">
            <div class="meta-pill"><span class="ml">프로젝트</span>{{ currentResult.copyright || 'ORIGINAL' }}</div>
            <div class="meta-pill artist"><span class="ml">작가</span>{{ currentResult.artist || 'UNKNOWN' }}</div>
            <div class="meta-pill character"><span class="ml">캐릭터</span>{{ currentResult.character || 'GENERIC' }}</div>
            <div class="meta-pill mini"><span class="ml">등급</span>{{ currentResult.rating || '?' }}</div>
            <div class="meta-pill mini" v-if="currentResult.image_width && currentResult.image_height"><span class="ml">해상도</span>{{ currentResult.image_width }}×{{ currentResult.image_height }}</div>
            <div class="meta-pill mini"><span class="ml">태그</span>{{ currentTags.length }}</div>
          </div>
          <div class="tag-section">
            <label>General Tags ({{ currentTags.length }})</label>
            <div class="tag-cloud">
              <span v-for="tag in currentTags" :key="tag" class="tag" :class="tagColor(tag)">{{ tag }}</span>
            </div>
          </div>
          <!-- 태그 색상 범례 -->
          <div class="tag-legend">
            <span class="legend-item count">인물수</span>
            <span class="legend-item clothing">의상</span>
            <span class="legend-item body">신체</span>
            <span class="legend-item nsfw">NSFW</span>
            <span class="legend-item action">포즈</span>
            <span class="legend-item expression">표정</span>
            <span class="legend-item bg">배경</span>
            <span class="legend-item objects">사물</span>
            <span class="legend-item effect">효과</span>
            <span class="legend-item color-tag">색상</span>
            <span class="legend-item trait">특성</span>
          </div>
          <div class="detail-actions">
            <button class="primary-btn" @click="applyResult">프롬프트로 사용</button>
            <button class="secondary-btn" @click="addToQueue">대기열에 추가</button>
            <button class="secondary-btn" @click="randomResult"><Icon name="dice" /> 다음 무작위</button>
          </div>
        </div>
      </div>

      <!-- List View -->
      <div v-else class="list-view">
        <!-- Sort + Pagination Toolbar -->
        <div class="list-toolbar">
          <div class="list-sort">
            <span class="sort-label">정렬:</span>
            <select v-model="sortBy" class="sort-select">
              <option value="default">기본순</option>
              <option value="character">캐릭터</option>
              <option value="artist">작가</option>
              <option value="copyright">작품</option>
              <option value="rating">레이팅</option>
              <option value="random">랜덤</option>
            </select>
            <button class="sort-dir-btn" v-if="sortBy !== 'default' && sortBy !== 'random'"
              @click="sortDir = sortDir === 'asc' ? 'desc' : 'asc'"
              :title="sortDir === 'asc' ? '오름차순 → 클릭 시 내림차순' : '내림차순 → 클릭 시 오름차순'">
              <Icon :name="sortDir === 'asc' ? 'arrow-up' : 'arrow-down'" size="12" />
            </button>
          </div>
          <div class="list-pager" v-if="totalPages > 1">
            <button class="page-btn" @click="gotoPage(0)" :disabled="currentPage === 0">≪</button>
            <button class="page-btn" @click="gotoPage(currentPage - 1)" :disabled="currentPage === 0"><Icon name="chevron-left" /></button>
            <span class="page-info">{{ currentPage + 1 }} / {{ totalPages }} ({{ sortedResults.length }})</span>
            <button class="page-btn" @click="gotoPage(currentPage + 1)" :disabled="currentPage >= totalPages - 1"><Icon name="play" /></button>
            <button class="page-btn" @click="gotoPage(totalPages - 1)" :disabled="currentPage >= totalPages - 1">≫</button>
          </div>
        </div>
        <div class="list-header">
          <span class="lh idx">#</span><span class="lh char">캐릭터</span>
          <span class="lh copy">작품</span><span class="lh artist">작가</span>
          <span class="lh tags">태그</span><span class="lh act"></span>
        </div>
        <div class="list-scroll">
          <div v-for="(r, i) in pagedResults" :key="(currentPage * PAGE_SIZE) + i" class="list-row"
            :class="{ active: r === currentResult }"
            @click="onListRowClick(r)">
            <span class="lr idx">{{ (currentPage * PAGE_SIZE) + i + 1 }}</span>
            <span class="lr char">{{ r.character || '-' }}</span>
            <span class="lr copy">{{ r.copyright || '-' }}</span>
            <span class="lr artist">{{ r.artist || '-' }}</span>
            <span class="lr tags">{{ (r.general || '').substring(0, 80) }}</span>
            <span class="lr act"><button class="use-btn" @click.stop="onListRowUse(r)">사용</button></span>
          </div>
        </div>
      </div>
      <!-- Filter Manager Modal -->
      <transition name="fade">
        <div v-if="showFilterManager" class="fm-overlay" @mousedown.self="showFilterManager = false">
          <div class="fm-modal">
            <div class="fm-header">
              <h3>필터 관리</h3>
              <button class="close-btn" @click="showFilterManager = false"><Icon name="close" /></button>
            </div>
            <div class="fm-body">
              <!-- Rating -->
              <div class="fm-section">
                <div class="fm-label">등급</div>
                <div class="fm-chips">
                  <button v-for="r in filterOptions.ratings" :key="r" class="fm-chip"
                    :class="{ active: activeFilters.ratings.has(r) }"
                    @click="toggleFilter('ratings', r)">{{ r.toUpperCase() }}</button>
                </div>
              </div>
              <!-- Character -->
              <div class="fm-section" v-if="filterOptions.characters.length">
                <div class="fm-label">CHARACTER ({{ filterOptions.characters.length }})</div>
                <input class="fm-search" v-model="fmCharSearch" placeholder="검색..." />
                <div class="fm-chip-scroll">
                  <button v-for="c in filteredFmChars" :key="c" class="fm-chip"
                    :class="{ active: activeFilters.characters.has(c) }"
                    @click="toggleFilter('characters', c)">{{ c }}</button>
                </div>
              </div>
              <!-- Copyright -->
              <div class="fm-section" v-if="filterOptions.copyrights.length">
                <div class="fm-label">COPYRIGHT ({{ filterOptions.copyrights.length }})</div>
                <input class="fm-search" v-model="fmCopySearch" placeholder="검색..." />
                <div class="fm-chip-scroll">
                  <button v-for="c in filteredFmCopyrights" :key="c" class="fm-chip"
                    :class="{ active: activeFilters.copyrights.has(c) }"
                    @click="toggleFilter('copyrights', c)">{{ c }}</button>
                </div>
              </div>
              <!-- Artist -->
              <div class="fm-section" v-if="filterOptions.artists.length">
                <div class="fm-label">ARTIST ({{ filterOptions.artists.length }})</div>
                <input class="fm-search" v-model="fmArtistSearch" placeholder="검색..." />
                <div class="fm-chip-scroll">
                  <button v-for="a in filteredFmArtists" :key="a" class="fm-chip"
                    :class="{ active: activeFilters.artists.has(a) }"
                    @click="toggleFilter('artists', a)">{{ a }}</button>
                </div>
              </div>
            </div>
            <div class="fm-footer">
              <span class="fm-count">{{ filteredByManager.length }} / {{ results.length }} 결과</span>
              <button class="fm-apply" @click="applyFilterManager">필터 적용</button>
            </div>
          </div>
        </div>
      </transition>

      <!-- 조건부 프롬프트는 STUDIO TOOLS의 '조건부' 버튼(모달)에서 관리합니다 -->
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, shallowRef, markRaw, toRaw, reactive, computed, onMounted, onUnmounted, watch } from 'vue'
import { condRulesPayload } from '../composables/condRules.js'
import { getBackend, onBackendEvent } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'
import { vHostDialog } from '../utils/hostDialogs'
import {
  buildSearchDeckUpdate,
  parseSearchResultLineage,
  resolveSearchCacheRestore,
  sameSearchLineage,
  type SearchResultLineage,
} from '../utils/searchRestore'
import { cachedTagSplitter } from '../utils/searchTags'
import CustomSelect from '../components/CustomSelect.vue'

interface RatingChip { key: string; label: string; checked: boolean }
interface SearchField { key: string; label: string; placeholder: string; include: string; exclude: string }
interface SearchRow { copyright?: string; character?: string; artist?: string; general?: string; rating?: string; image_width?: number; image_height?: number; [k: string]: any }
interface FilterBranch { label: string; count: number; data: SearchRow[] }
type FilterCategory = 'ratings' | 'characters' | 'copyrights' | 'artists'

const ratings = reactive<RatingChip[]>([
  { key: 'g', label: 'GEN', checked: true },
  { key: 's', label: 'SENS', checked: false },
  { key: 'q', label: 'QUES', checked: false },
  { key: 'e', label: 'EXPL', checked: false },
])
const fields = reactive<SearchField[]>([
  { key: 'copyright', label: 'PROJECT', placeholder: 'e.g. genshin_impact', include: '', exclude: '' },
  { key: 'character', label: 'CHAR', placeholder: 'e.g. raiden_shogun', include: '', exclude: '' },
  { key: 'artist', label: 'ARTIST', placeholder: 'Artist name...', include: '', exclude: '' },
  { key: 'general', label: 'TAGS', placeholder: '1girl, blue_hair...', include: '', exclude: '' },
])

// 결과 배열은 shallowRef — 수만~50만 행을 깊은 reactive proxy 로 감싸지 않는다(정렬·필터·
// 직렬화가 행마다 proxy 를 거치던 비용 제거). 행은 불변이고 배열은 언제나 통째로 교체하므로
// 얕은 추적으로 충분하다. 제자리 변경(push/splice) 금지 — 반드시 새 배열을 대입한다.
const results = shallowRef<SearchRow[]>([])   // 필터 base — Python base 와 같은 순서(덱 인덱스 기준)
const filteredResults = shallowRef<SearchRow[]>([])
const deepBase = shallowRef<SearchRow[]>([])  // 심층검색 결과(매니저 필터 적용 전 베이스) — rating 등 매니저 필터와 합성
const lastResults = shallowRef<SearchRow[]>([])  // 검색 폼으로 돌아가도 보존
const lastResultsLineage = ref<SearchResultLineage | null>(null)  // lastResults 를 만든 검색 스냅숏
const previewIdx = ref(0)
const searching = ref(false)
const statusText = ref('READY')
const searchProgress = ref(0)
// 원형 진행률 — 반지름 42 → 둘레 = 2π·42 ≈ 263.89
// stroke-dashoffset = 둘레 × (1 - 진행률/100)
const RING_CIRCUMFERENCE = 263.89
const ringOffset = computed(() =>
  RING_CIRCUMFERENCE * (1 - Math.max(0, Math.min(100, searchProgress.value)) / 100)
)
const viewMode = ref('single')
const deepInclude = ref('')
const deepExclude = ref('')
const isFiltered = ref(false)
// 분기 스냅숏도 shallowRef — push 대신 새 배열 대입(data 는 불변 deepBase 배열 참조)
const filterHistory = shallowRef<FilterBranch[]>([])

// 정렬 + 페이지네이션
const sortBy = ref(localStorage.getItem('search.sortBy') || 'default')  // default | character | artist | copyright | random
const sortDir = ref(localStorage.getItem('search.sortDir') || 'asc')    // asc | desc
watch(sortBy, (v) => { try { localStorage.setItem('search.sortBy', v) } catch {} })
watch(sortDir, (v) => { try { localStorage.setItem('search.sortDir', v) } catch {} })

// 필드 결합 모드 — AND (모두 매칭) / OR (하나라도 매칭)
// 예: copyright='genshin', character='raiden_shogun'
//   AND → 라이덴 쇼군만
//   OR  → 라이덴 쇼군이거나 원신 캐릭터 누구든
const combineMode = ref(localStorage.getItem('search.combineMode') || 'and')
watch(combineMode, (v) => { try { localStorage.setItem('search.combineMode', v) } catch {} })

// 0건 결과 안내용 — 검색 한 번이라도 실행했는지
const hasSearched = ref(false)
// 마지막 쿼리 요약 (사용자가 무엇을 검색했는지 보여주기 위함)
const lastQuerySummary = ref('')

// 런타임 Search는 사용자 지정 데이터 경로의 manifest가 가리키는 단일 릴리스만 사용한다.
const activeDatasetLabel = ref('2026_07')
const activeDatasetDisplay = computed(() => activeDatasetLabel.value.replace(/^([0-9]{4})_([0-9]{2})$/, '$1.$2'))
const activeResultLineage = ref<SearchResultLineage | null>(null)
const pendingResultLineage = ref<SearchResultLineage | null>(null)

// 결과 cap — 기본 50만 (UI 부하 방지), 'unlimited' 선택 시 전체 결과 받음
// 무제한이면 JSON 직렬화 + Vue 메모리가 무거워질 수 있음 (수 GB 가능)
const resultCapMode = ref(localStorage.getItem('search.resultCapMode') || 'capped')
if (!['capped', 'unlimited'].includes(resultCapMode.value)) resultCapMode.value = 'capped'
watch(resultCapMode, (v) => { try { localStorage.setItem('search.resultCapMode', v) } catch {} })
const PAGE_SIZE = 50
const currentPage = ref(0)
// 결과/필터/정렬이 바뀌면 1페이지로
watch(() => filteredResults.value.length, () => { currentPage.value = 0 })
watch(sortBy, () => { currentPage.value = 0 })

const sortedResults = computed(() => {
  const arr = filteredResults.value
  if (sortBy.value === 'default') return arr
  if (sortBy.value === 'random') {
    // 시드 기반 셔플 — 결과 안정성 위해 서비스 호출마다 새로 생성 (단순 Fisher-Yates)
    const copy = [...arr]
    for (let i = copy.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [copy[i], copy[j]] = [copy[j], copy[i]]
    }
    return copy
  }
  const k = sortBy.value
  const dir = sortDir.value === 'desc' ? -1 : 1
  return [...arr].sort((a, b) => {
    const av = (a[k] || '').toString().toLowerCase()
    const bv = (b[k] || '').toString().toLowerCase()
    if (av < bv) return -1 * dir
    if (av > bv) return 1 * dir
    return 0
  })
})

const totalPages = computed(() => Math.max(1, Math.ceil(sortedResults.value.length / PAGE_SIZE)))
const pagedResults = computed(() => {
  const start = currentPage.value * PAGE_SIZE
  return sortedResults.value.slice(start, start + PAGE_SIZE)
})
function gotoPage(p: number) {
  currentPage.value = Math.max(0, Math.min(totalPages.value - 1, p))
}
function onListRowClick(r: SearchRow) {
  // 정렬된 화면에서 클릭 → filteredResults 내의 인덱스로 변환 (previewIdx는 filteredResults 기준)
  const idx = filteredResults.value.indexOf(r)
  if (idx >= 0) {
    previewIdx.value = idx
    viewMode.value = 'single'
  }
}
function onListRowUse(r: SearchRow) {
  const idx = filteredResults.value.indexOf(r)
  if (idx >= 0) {
    previewIdx.value = idx
    applyResult()
  }
}

// 조건부 프롬프트 — 상태/저장/로드는 composables/condRules.js + 조건부 모달이 담당
let progressTimer: ReturnType<typeof setInterval> | null = null

const currentResult = computed<SearchRow | null>(() => filteredResults.value[previewIdx.value] || null)
// 현재 결과는 공백 구분 + 언더스코어 보존 형식이다. .parquet 가져오기와
// 과거에 저장한 외부 결과를 위해 콤마 구분 legacy 포맷도 계속 읽는다.
// 콤마가 있으면 콤마로만 쪼개 'aqua gloves'를 통째로 유지하고,
// 없으면 공백으로 쪼갠 뒤 표시용으로만 _→공백 치환 → 'black_hair'가 안 쪼개짐.
// (콤마 split에 공백을 섞으면 2025/2026의 'aqua gloves'가 잘못 쪼개지므로 분기 필수.)
const currentTags = computed(() => {
  const raw = currentResult.value?.general || ''
  if (!raw.trim()) return []
  if (raw.includes(',')) {
    return raw.split(',').map(t => t.trim()).filter(Boolean)
  }
  return raw.split(/\s+/).map(t => t.trim()).filter(Boolean).map(t => t.replace(/_/g, ' '))
})

// 검색 입력 영속 — localStorage는 sync (앱 빨리 닫혀도 안 잃어버림)
// 백엔드 ui_prefs.json 쓰기는 디바운스 (키스트로크마다 파일 쓰지 않게)
let _persistBackendTimer: ReturnType<typeof setTimeout> | null = null
function persistSearchFields() {
  const payload = {
    fields: fields.map(f => ({ include: f.include, exclude: f.exclude })),
    ratings: ratings.map(r => ({ key: r.key, checked: r.checked })),
    combineMode: combineMode.value,
    resultCapMode: resultCapMode.value,
  }
  // ① localStorage 즉시 저장 — 한 글자 입력 후 바로 닫아도 보존
  try { window.localStorage.setItem('lastSearchFields', JSON.stringify(payload)) } catch {}
  // ② 백엔드 ui_prefs 쓰기는 600ms 디바운스 — 파일 IO 줄이기
  if (_persistBackendTimer) clearTimeout(_persistBackendTimer)
  _persistBackendTimer = setTimeout(() => {
    requestAction('save_ui_prefs', { searchState: payload })
  }, 600)
}

// 저장된 검색 상태 적용 — uiPrefsLoaded 이벤트 + mount 시 능동 fetch 양쪽에서 재사용.
// fields/ratings/combineMode/resultCapMode 전부 복원.
function applySearchState(state: any) {
  if (!state) return false
  let touched = false
  if (Array.isArray(state.fields)) {
    state.fields.forEach((f: any, i: number) => {
      if (fields[i]) { fields[i].include = f.include || ''; fields[i].exclude = f.exclude || '' }
    })
    touched = true
  }
  if (Array.isArray(state.ratings)) {
    state.ratings.forEach((r: any) => { const found = ratings.find(rt => rt.key === r.key); if (found) found.checked = !!r.checked })
  }
  if (state.combineMode === 'and' || state.combineMode === 'or') combineMode.value = state.combineMode
  if (state.resultCapMode && ['capped', 'unlimited'].includes(state.resultCapMode)) resultCapMode.value = state.resultCapMode
  return touched
}

async function search() {
  searching.value = true; statusText.value = 'EXPLORING...'
  searchProgress.value = 0
  hasSearched.value = true  // 0건 UI 분기용
  // 가짜 진행률 — 지수 곡선(asymptotic)으로 스르륵 99%에 수렴, 100%는 결과 도착 시 점프
  //   100ms tick = 10 fps 업데이트, CSS transition 0.4s 가 보간해서 더 부드러움
  //   곡선식: target = 99 * (1 - exp(-elapsed / TAU))
  //     TAU=3000 → 3초까지 63%, 6초까지 86%, 9초까지 95%, 12초까지 98%
  const startTime = Date.now()
  const TAU = 3000  // 시간 상수 (ms)
  if (progressTimer) clearInterval(progressTimer)   // 직전 타이머 정리(중복 누적 방지)
  progressTimer = setInterval(() => {
    const elapsed = Date.now() - startTime
    const target = 99 * (1 - Math.exp(-elapsed / TAU))
    searchProgress.value = Math.min(99, target)
  }, 100)
  const backend: any = await getBackend()
  const query = {
    ratings: ratings.filter(r => r.checked).map(r => r.key),
    queries: Object.fromEntries(fields.map(f => [f.key, f.include])),
    excludes: Object.fromEntries(fields.map(f => [f.key, f.exclude])),
    combine_mode: combineMode.value,  // 'and' | 'or' — 필드 간 결합 방식
    disable_result_cap: resultCapMode.value === 'unlimited',  // true면 50만 cap 무시
  }
  // 사용자에게 보여줄 쿼리 요약 (0건 시 활용)
  const includeParts = fields
    .filter(f => f.include?.trim())
    .map(f => `${f.key}: ${f.include}`)
  const excludeParts = fields
    .filter(f => f.exclude?.trim())
    .map(f => `~${f.key}: ${f.exclude}`)
  const ratingSel = ratings.filter(r => r.checked).map(r => r.label).join('+') || 'no rating'
  lastQuerySummary.value = `[${activeDatasetLabel.value}] [${combineMode.value.toUpperCase()}] [${ratingSel}] ${[...includeParts, ...excludeParts].join(' / ') || '(빈 검색)'}`
  if (backend.searchDanbooru) backend.searchDanbooru(JSON.stringify(query))
}

function newSearch() {
  results.value = []; deepBase.value = []; filteredResults.value = []; previewIdx.value = 0
  deepInclude.value = ''; deepExclude.value = ''; isFiltered.value = false
  // 이전 검색의 정밀검색 분기는 버린다 — 남겨 두면 새 검색 뒤에도 '분기:' 버튼이 보이고,
  // 누르면 옛 행이 현재 lineage 로 자동화 덱을 덮는다(옛 배열도 keep-alive 로 계속 남는다).
  filterHistory.value = []
  // lastResults는 보존 — 검색 폼에서 다시 볼 수 있음
}

// 이전 결과(= 그 검색의 base)를 다시 펼친다. 매니저 필터 칩은 그대로 적용해 표시·칩·자동화
// 덱이 같은 집합을 가리키게 하고, lastResults 가 현재 lineage 의 것일 때만 덱을 동기화한다.
function _reopenLastResults() {
  const base = lastResults.value
  results.value = base
  deepBase.value = base
  filterHistory.value = []
  isFiltered.value = false
  filteredResults.value = _applyManagerFilters(base)
  if (sameSearchLineage(lastResultsLineage.value, activeResultLineage.value)) {
    syncDeck()
  } else {
    requestAction('show_toast', {
      type: 'warning',
      msg: '이전 결과의 검색 출처가 현재와 달라 자동화 덱은 바꾸지 않았습니다.',
    })
  }
}

function restoreLastResults() {
  _reopenLastResults()
  previewIdx.value = 0
  viewMode.value = 'list'
  statusText.value = `${filteredResults.value.length} MATCHES (복원)`
}

function restoreAndRandom() {
  _reopenLastResults()
  previewIdx.value = Math.floor(Math.random() * filteredResults.value.length)
  viewMode.value = 'single'
  statusText.value = `${filteredResults.value.length} MATCHES (랜덤)`
}

// localStorage가 진실의 원천 — 복원 여부 추적해서 uiPrefsLoaded 덮어쓰기 방지
let _restoredFromLocalStorage = false

onUnmounted(() => {
  if (progressTimer) { clearInterval(progressTimer); progressTimer = null }
})

onMounted(() => {
  // 결과는 manifest fingerprint를 검증하는 backend cache만 사용한다. 구형 브라우저
  // 결과 cache는 사용자 지정 경로 전환 시 stale할 수 있어 1회 제거한다.
  try {
    for (const key of ['search.datasetYear', 'search.resultsDataset', 'lastSearchResults', 'lastFullResults']) {
      window.localStorage.removeItem(key)
    }
  } catch {}
  ;(async () => {
    try {
      const backend: any = await getBackend()
      if (backend.getActiveSearchDataset) {
        const raw = await new Promise<string>(resolve => backend.getActiveSearchDataset((s: string) => resolve(s)))
        const info = JSON.parse(raw)
        if (typeof info.label === 'string' && info.label) activeDatasetLabel.value = info.label
      }
    } catch (e) { console.warn('[Search] active dataset 확인 실패:', e) }
  })()
  // 이전 검색 입력 필드 복원
  try {
    const sf = window.localStorage.getItem('lastSearchFields')
    if (sf) {
      const d = JSON.parse(sf)
      if (d.fields) d.fields.forEach((f: any, i: number) => { if (fields[i]) { fields[i].include = f.include || ''; fields[i].exclude = f.exclude || '' } })
      if (d.ratings) d.ratings.forEach((r: any) => { const found = ratings.find(rt => rt.key === r.key); if (found) found.checked = r.checked })
      if (d.combineMode && (d.combineMode === 'and' || d.combineMode === 'or')) combineMode.value = d.combineMode
      if (d.resultCapMode && ['capped', 'unlimited'].includes(d.resultCapMode)) resultCapMode.value = d.resultCapMode
      _restoredFromLocalStorage = true
    }
  } catch {}
  // localStorage 캐시가 없을 때(새 프로필·지운 저장소·다른 기기의 웹 클라이언트) — ui_prefs.json(파일)
  // 에서 능동 복원. uiPrefsLoaded 는 부팅 때 당겨 온 시작 시점 스냅숏이라(bridge.js sticky 재생)
  // 그 뒤 저장된 검색 상태는 getter 로 직접 가져와야 최신이다.
  if (!_restoredFromLocalStorage) {
    ;(async () => {
      try {
        const backend: any = await getBackend()
        if (backend.getUiPrefs) {
          const json = await new Promise<string>(resolve => backend.getUiPrefs((s: string) => resolve(s)))
          if (applySearchState(JSON.parse(json).searchState)) {
            _restoredFromLocalStorage = true  // 이후 uiPrefsLoaded 이벤트 중복 적용 방지
          }
        }
      } catch (e) { console.warn('[Search] ui_prefs 능동 복원 실패:', e) }
    })()
  }
  // 이전 검색 결과는 manifest fingerprint를 검증하는 backend cache에서만 복원한다.
  ;(async () => {
    try {
      const backend: any = await getBackend()
      // active = 필터된(활성) 셋 → 표시 + 자동화 덱; full = 전체 셋 → '필터 해제' 베이스
      let active: any = null, full: any = null
      if (backend.loadLastSearchResults) {
        try { active = JSON.parse(await new Promise<string>(r => backend.loadLastSearchResults((s: string) => r(s)))) } catch {}
      }
      if (backend.loadFullResults) {
        try { full = JSON.parse(await new Promise<string>(r => backend.loadFullResults((s: string) => r(s)))) } catch {}
      }
      const restored = resolveSearchCacheRestore<SearchRow>(active, full)
      if (restored) {
        activeResultLineage.value = pendingResultLineage.value
        pendingResultLineage.value = null
        // 결과 배열은 proxy 로 감싸지 않는다(markRaw) — 행 수만큼 proxy 를 만들지 않게.
        const base = markRaw(restored.base)
        results.value = base; lastResults.value = base   // 전체(필터 해제 베이스) — Python base 와 같은 순서
        lastResultsLineage.value = activeResultLineage.value
        // deep-only 중간층은 별도 저장하지 않으므로 full을 복원 기준으로 삼는다.
        // 그래야 0건 필터를 포함해 재시작 후 필터 해제/RESET으로 전체를 회복할 수 있다.
        deepBase.value = base
        filterHistory.value = []
        filteredResults.value = markRaw(restored.active)           // 표시 = 필터된 셋
        isFiltered.value = restored.isFiltered
        statusText.value = `${restored.active.length.toLocaleString()} MATCHES (디스크 복원)`
        return
      }
    } catch (e) { console.warn('[Search] backend restore failed:', e) }
  })()

  onBackendEvent('searchResultLineage', (json: string) => {
    // 결과 배열과 별도 additive signal이므로 즉시 active로 바꾸지 않는다. 다음
    // searchResultsReady/디스크 복원과 같은 작업에서 승격해야 A rows+B lineage 틈이 없다.
    pendingResultLineage.value = parseSearchResultLineage(json)
    if (pendingResultLineage.value) {
      activeDatasetLabel.value = pendingResultLineage.value.label
    }
  })
  onBackendEvent('searchResultsReady', (json: string) => {
    try {
      const data = JSON.parse(json)
      if (Array.isArray(data)) {
        activeResultLineage.value = pendingResultLineage.value
        pendingResultLineage.value = null
        markRaw(data)   // 결과 배열은 proxy 로 감싸지 않는다(행 불변, 통째 교체)
        results.value = data; deepBase.value = data; filteredResults.value = data; previewIdx.value = 0
        lastResults.value = data
        lastResultsLineage.value = activeResultLineage.value
        // 새 결과 → 이전 검색의 정밀검색 분기·필터 상태를 버린다(옛 분기가 새 lineage 로 덱을 덮지 않게)
        filterHistory.value = []
        isFiltered.value = false
        statusText.value = `${data.length} MATCHES`
        // 새 검색 → 기존 필터 칩 초기화 (옛 필터가 stale하게 남지 않도록)
        activeFilters.ratings.clear(); activeFilters.characters.clear()
        activeFilters.copyrights.clear(); activeFilters.artists.clear()
        try { window.localStorage.removeItem('searchActiveFilters') } catch {}
        persistSearchFields()
      } else if (data && typeof data === 'object' && data.error) {
        // 백엔드가 명시적 에러 객체를 보낸 경우 — 사용자에게 토스트
        statusText.value = '검색 실패'
        requestAction('show_toast', { type: 'error', msg: `검색 실패: ${data.error}` })
      } else {
        statusText.value = '검색 결과 형식 오류 (배열 예상)'
        requestAction('show_toast', { type: 'error', msg: '검색 결과 형식 오류 — 배열이 아닙니다' })
      }
    } catch (e: any) {
      pendingResultLineage.value = null
      statusText.value = `JSON 파싱 실패: ${e?.message || 'unknown'}`
      requestAction('show_toast', { type: 'error', msg: `검색 결과 JSON 파싱 실패: ${e?.message || e}` })
    }
    searching.value = false; searchProgress.value = 100
    if (progressTimer) { clearInterval(progressTimer); progressTimer = null }
  })
  onBackendEvent('searchStatus', (msg: string) => {
    statusText.value = msg.toUpperCase()
    // 결과 이벤트(searchResultsReady)가 안 오는 '진짜 종료(❌ 로드실패/데이터없음)'만 잡아 정리.
    // (기존 /없|실패|완료|0/ 은 데이터 병합 상태나 '파일 없음'(파일별 경고),
    //  '검색 완료' 등 진행/정상 메시지까지 오매칭 → 검색 도중 로딩이 꺼지고 '결과 없음'이 먼저 떴음.
    //  정상 완료/0건은 searchResultsReady 핸들러가 searching=false 처리하므로 여기선 불필요.)
    if (/^\s*❌/.test(msg)) {
      if (progressTimer) { clearInterval(progressTimer); progressTimer = null }
      searching.value = false
    }
  })
  onBackendEvent('uiPrefsLoaded', (json: string) => {
    // localStorage/능동 fetch로 이미 복원했으면 무시 (중복 적용 방지)
    if (_restoredFromLocalStorage) return
    try { if (applySearchState(JSON.parse(json).searchState)) _restoredFromLocalStorage = true } catch {}
  })

  // (조건부 프롬프트 로드는 App.vue가 composables/condRules.js의 loadCondRules로 처리)
})

watch(fields, persistSearchFields, { deep: true })
watch(ratings, persistSearchFields, { deep: true })
// 모드/cap 토글도 통합 payload에 자동 반영 (개별 키와 별개로 묶음 백업)
watch(combineMode, persistSearchFields)
watch(resultCapMode, persistSearchFields)

function applyDeepSearch() {
  const inc = deepInclude.value.toLowerCase().trim()
  const exc = deepExclude.value.toLowerCase().trim()
  if (!inc && !exc) return
  // deepBase(매니저 필터 전) 스냅샷을 분기로 저장
  const label = [inc ? `+${inc.substring(0,15)}` : '', exc ? `-${exc.substring(0,15)}` : ''].filter(Boolean).join(' ')
  // shallowRef 라 push 대신 새 배열 대입. deepBase 배열은 제자리 변경되지 않으므로(늘 새 배열
  // 대입) 복사 없이 참조를 스냅숏으로 둔다.
  filterHistory.value = [...filterHistory.value, { label, count: deepBase.value.length, data: deepBase.value }]
  // deepBase 누적 필터 → 매니저 필터(rating 등) 재적용해 표시/덱 갱신
  // '태그 단위 정확 일치'로 매칭한다. 데이터는 "monkey_d._luffy roronoa_zoro"처럼 공백으로
  // 태그를 구분하고 언더스코어는 태그 '내부'에 둔다. 입력의 공백을 언더스코어로 바꿔 태그 형식과
  // 맞춘 뒤, 행의 태그 집합에 그 태그가 '정확히' 있는지로 판정한다.
  //   부분일치(includes)면 'girl'이 '1girl'을, 'other'가 'mother'를 잡아 과다 제외됨.
  //   포함=모두 가져야(AND), 제외=하나라도 가지면 버림(OR).
  const s2u = (s: string) => s.trim().replace(/\s+/g, '_')
  const incTerms = inc ? inc.split(',').map(s2u).filter(Boolean) : []
  const excTerms = exc ? exc.split(',').map(s2u).filter(Boolean) : []
  deepBase.value = deepBase.value.filter(r => {
    const tags = new Set(`${r.copyright} ${r.character} ${r.artist} ${r.general}`
      .toLowerCase().split(/[\s,]+/).filter(Boolean))
    if (incTerms.some(q => !tags.has(q))) return false
    if (excTerms.some(q => tags.has(q))) return false
    return true
  })
  filteredResults.value = _applyManagerFilters(deepBase.value)
  previewIdx.value = 0; isFiltered.value = true
  deepInclude.value = ''; deepExclude.value = ''
  statusText.value = `DEEP: ${filteredResults.value.length} / ${results.value.length}`
  syncDeck()
}
function restoreBranch(idx: number) {
  const branch = filterHistory.value[idx]
  if (!branch) return
  deepBase.value = branch.data
  filterHistory.value = filterHistory.value.slice(0, idx)
  filteredResults.value = _applyManagerFilters(deepBase.value)
  previewIdx.value = 0
  isFiltered.value = filterHistory.value.length > 0
  statusText.value = `BRANCH: ${filteredResults.value.length}`
  syncDeck()
}
function resetDeepSearch() {
  deepInclude.value = ''; deepExclude.value = ''
  deepBase.value = [...results.value]
  filterHistory.value = []
  clearAllFilters()   // 매니저 필터 해제 + filteredResults=deepBase + 덱 동기화
  isFiltered.value = false
  statusText.value = `${results.value.length} MATCHES`
}

// 태그 색상 분류 — Python TagClassifier (tags_db 기반)
// LRU 캐시: 최대 5000개 — 무한 증가 방지. 가장 오래된 항목부터 제거
const TAG_CACHE_MAX = 5000
const tagCategoryCache = ref<Record<string, string>>({})  // tag → category
const _tagCacheKeys: string[] = []  // 삽입 순서 추적
function _setTagCache(key: string, value: string) {
  if (key in tagCategoryCache.value) {
    // 기존 항목 갱신 — 키 순서만 뒤로 이동
    const idx = _tagCacheKeys.indexOf(key)
    if (idx >= 0) _tagCacheKeys.splice(idx, 1)
  } else if (_tagCacheKeys.length >= TAG_CACHE_MAX) {
    // 가장 오래된 항목 evict
    const evict = _tagCacheKeys.shift()
    if (evict !== undefined) delete tagCategoryCache.value[evict]
  }
  _tagCacheKeys.push(key)
  tagCategoryCache.value[key] = value
}

// 카테고리 → CSS 클래스 매핑 (세분화)
const catToColor: Record<string, string> = {
  'sexual': 'nsfw', 'body_parts': 'body', 'clothing': 'clothing',
  'pose': 'action', 'expression': 'expression', 'background': 'bg',
  'composition': 'composition', 'effect': 'effect', 'objects': 'objects',
  'character': 'char', 'copyright': 'copy', 'artist': 'artist-tag',
  'color': 'color-tag', 'character_trait': 'trait', 'animals': 'objects',
  'art_style': 'style', 'general': '',
}

// 인물수 태그는 프론트에서 직접 판단 (빠름)
const countPattern = /^(\d+)?(girl|boy|other)s?$|^solo$|^multiple_/

function tagColor(tag: string) {
  const t = tag.trim().toLowerCase().replace(/ /g, '_')
  if (countPattern.test(t)) return 'count'
  // 캐시 검색 (원본 + underscore 변환 둘 다)
  const cached = tagCategoryCache.value[t] || tagCategoryCache.value[tag.trim()]
  if (cached) return catToColor[cached] || ''
  return ''
}

// 현재 보고 있는 결과의 태그를 배치로 분류 요청
async function classifyCurrentTags() {
  if (!currentResult.value) return
  const tags = currentTags.value.map(t => t.trim().replace(/ /g, '_'))
  // 캐시에 없는 태그만 요청
  const uncached = tags.filter(t => !(t in tagCategoryCache.value))
  if (uncached.length === 0) return
  const backend: any = await getBackend()
  if (backend.classifyTags) {
    backend.classifyTags(JSON.stringify(uncached), (json: string) => {
      try {
        const result: any = JSON.parse(json)
        if (!result.error) {
          // LRU 캐시로 추가 — 무한 증가 방지
          for (const [k, v] of Object.entries<any>(result)) {
            _setTagCache(k, v)
          }
        }
      } catch {}
    })
  }
}

// previewIdx 변경 시 + 결과 로드 시 자동 분류
watch(previewIdx, () => { classifyCurrentTags() })
watch(() => filteredResults.value.length, () => { if (filteredResults.value.length > 0) classifyCurrentTags() })

// ── Filter Manager ──
const showFilterManager = ref(false)
const fmCharSearch = ref('')
const fmCopySearch = ref('')
const fmArtistSearch = ref('')
const activeFilters = reactive({
  ratings: new Set<string>(),
  characters: new Set<string>(),
  copyrights: new Set<string>(),
  artists: new Set<string>(),
})

// 재시작 시 필터 칩 복원 (Set 재구성) — 표시 결과는 이미 필터된 셋이지만,
// 어떤 필터가 적용됐는지 칩으로 보여주기 위해 복원.
try {
  const _af = JSON.parse(window.localStorage.getItem('searchActiveFilters') || 'null')
  if (_af) {
    (_af.ratings || []).forEach((v: string) => activeFilters.ratings.add(v))
    ;(_af.characters || []).forEach((v: string) => activeFilters.characters.add(v))
    ;(_af.copyrights || []).forEach((v: string) => activeFilters.copyrights.add(v))
    ;(_af.artists || []).forEach((v: string) => activeFilters.artists.add(v))
  }
} catch {}

// 칸 문자열 분해는 현재 base(results) 배열 범위로 메모이즈 — 필터 모달에서 칩을 토글할
// 때마다 전 행의 character/copyright/artist 를 다시 쪼개지 않는다(utils/searchTags).
function _tagSplitter() {
  return cachedTagSplitter(results.value)
}

// 검색 결과에서 필터 옵션 추출
const filterOptions = computed(() => {
  const chars = new Set<string>(), copys = new Set<string>(), arts = new Set<string>(), rats = new Set<string>()
  const split = _tagSplitter()
  for (const row of results.value) {
    if (row.rating) rats.add(row.rating)
    for (const c of split(row.character)) chars.add(c)
    for (const c of split(row.copyright)) copys.add(c)
    for (const a of split(row.artist)) arts.add(a)
  }
  return {
    ratings: [...rats].sort(),
    characters: [...chars].sort(),
    copyrights: [...copys].sort(),
    artists: [...arts].sort(),
  }
})

const filteredFmChars = computed(() => {
  const q = fmCharSearch.value.toLowerCase()
  return q ? filterOptions.value.characters.filter(c => c.toLowerCase().includes(q)) : filterOptions.value.characters
})
const filteredFmCopyrights = computed(() => {
  const q = fmCopySearch.value.toLowerCase()
  return q ? filterOptions.value.copyrights.filter(c => c.toLowerCase().includes(q)) : filterOptions.value.copyrights
})
const filteredFmArtists = computed(() => {
  const q = fmArtistSearch.value.toLowerCase()
  return q ? filterOptions.value.artists.filter(a => a.toLowerCase().includes(q)) : filterOptions.value.artists
})

function toggleFilter(category: FilterCategory, value: string) {
  const s = activeFilters[category]
  if (s.has(value)) s.delete(value)
  else s.add(value)
}

const hasActiveFilters = computed(() =>
  activeFilters.ratings.size + activeFilters.characters.size + activeFilters.copyrights.size + activeFilters.artists.size > 0
)
const activeFilterCount = computed(() =>
  activeFilters.ratings.size + activeFilters.characters.size + activeFilters.copyrights.size + activeFilters.artists.size
)

// 매니저 필터 로직 (주어진 배열에 적용) — deepBase 위에서 합성하므로 심층검색 유지
function _applyManagerFilters(arr: SearchRow[]): SearchRow[] {
  let next = arr
  const split = _tagSplitter()
  if (activeFilters.ratings.size) next = next.filter(r => activeFilters.ratings.has(r.rating as string))
  if (activeFilters.characters.size) next = next.filter(r => split(r.character).some(c => activeFilters.characters.has(c)))
  if (activeFilters.copyrights.size) next = next.filter(r => split(r.copyright).some(c => activeFilters.copyrights.has(c)))
  if (activeFilters.artists.size) next = next.filter(r => split(r.artist).some(a => activeFilters.artists.has(a)))
  return next
}
// 필터 적용 미리보기 — 전체(results)가 아니라 deepBase(심층검색 결과) 위에서 필터
const filteredByManager = computed(() => _applyManagerFilters(deepBase.value))
// filteredResults → 자동화 덱 동기화 (모든 필터 변경 시 호출 → 덱이 항상 표시와 일치)
// 행 전체가 아니라 base(results) 배열의 인덱스만 보낸다 — 4만 행이면 20MB → 수백 KB.
// Python 은 같은 순서의 base 를 들고 있고 lineage·base_size 가 맞을 때만 받아들인다.
function syncDeck() {
  const payload = buildSearchDeckUpdate(
    filteredResults.value,
    activeResultLineage.value,
    toRaw(results.value),
    row => toRaw(row),
  )
  if (!payload) {
    requestAction('show_toast', {
      type: 'error',
      msg: '검색 결과 출처를 확인할 수 없어 필터를 적용하지 않았습니다.',
    })
    return
  }
  requestAction('update_prompt_deck', payload)
}

// activeFilters(Set들)를 localStorage에 영속 → 재시작 시 필터 칩 복원
function _persistActiveFilters() {
  try {
    window.localStorage.setItem('searchActiveFilters', JSON.stringify({
      ratings: [...activeFilters.ratings],
      characters: [...activeFilters.characters],
      copyrights: [...activeFilters.copyrights],
      artists: [...activeFilters.artists],
    }))
  } catch {}
}

function applyFilterManager() {
  filteredResults.value = [...filteredByManager.value]
  previewIdx.value = 0
  showFilterManager.value = false
  syncDeck()   // 덱 동기화 + lastSearchResults 갱신
  _persistActiveFilters()
  requestAction('show_toast', { type: 'success', msg: `필터 적용: ${filteredResults.value.length}건` })
}

function clearAllFilters() {
  activeFilters.ratings.clear()
  activeFilters.characters.clear()
  activeFilters.copyrights.clear()
  activeFilters.artists.clear()
  filteredResults.value = [...deepBase.value]   // 심층검색은 유지, 매니저 필터만 해제
  previewIdx.value = 0
  syncDeck()
  _persistActiveFilters()
  requestAction('show_toast', { type: 'info', msg: '필터 해제됨' })
}

// (clearSortFilter: clearAllFilters의 단순 alias였음 — 미사용으로 제거)

function prevResult() { if (previewIdx.value > 0) previewIdx.value-- }
function nextResult() { if (previewIdx.value < filteredResults.value.length - 1) previewIdx.value++ }
function randomResult() { if (filteredResults.value.length > 1) previewIdx.value = Math.floor(Math.random() * filteredResults.value.length) }
function applyResult() {
  if (!currentResult.value) return
  // 조건부 프롬프트 규칙도 함께 전달
  const payload = {
    ...currentResult.value,
    ...condRulesPayload(),
  }
  requestAction('apply_search_result', payload)
}
function addToQueue() {
  if (!currentResult.value) return
  const payload = {
    ...currentResult.value,
    ...condRulesPayload(),
  }
  requestAction('add_search_to_queue', payload)
}
// (조건부 규칙 저장/자동저장은 composables/condRules.js가 담당)

function exportResults() { requestAction('export_search_results', { count: filteredResults.value.length, data: filteredResults.value }) }
// 가져오기도 검색과 같은 결과 상한(50만, 무작위 표본)을 따른다 — '무제한' 모드면 끈다
function importResults() {
  requestAction('import_search_results', { disable_result_cap: resultCapMode.value === 'unlimited' })
}
</script>

<style scoped>
.search-view { height: 100%; display: flex; flex-direction: column; background: var(--bg-primary); overflow: hidden; }

/* ═══ Welcome (검색 전) ═══ */
.welcome { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px; gap: 32px; }
.ws-header { text-align: center; }
.ws-icon { font-size: 48px; opacity: 0.5; margin-bottom: 10px; }
.ws-header h1 { letter-spacing: 0.08em; font-size: 22px; letter-spacing: 0; color: var(--text-secondary); font-weight: var(--fw-bold); }
.ws-header p { font-size: 13px; color: var(--text-muted); margin-top: 6px; }

.search-form { width: 100%; max-width: 720px; display: flex; flex-direction: column; gap: 12px; }
.form-section { display: flex; flex-direction: column; gap: 10px; }
.form-row { display: flex; gap: 10px; }
.form-field { flex: 1; display: flex; flex-direction: column; gap: 3px; }
.form-field.wide { flex: 2; }
.form-field label { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-muted); letter-spacing: 0; }
.form-field input { padding: 11px 14px; font-size: 13px; }

.exclude-section { border: 1px solid rgba(248,113,113,0.1); border-radius: var(--radius-card); padding: 12px; }
.exclude-toggle { font-size: 11px; font-weight: var(--fw-bold); color: var(--state-alert-fg); cursor: pointer; letter-spacing: 0; list-style: none; }
.exclude-toggle::-webkit-details-marker { display: none; }

.form-footer { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; justify-content: center; margin-top: 4px; }
.rating-row { display: flex; gap: 4px; }
.rating-chip { height: 30px; padding: 0 18px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-pill); color: var(--text-secondary); font-size: var(--fs-meta); font-weight: var(--fw-bold); cursor: pointer; }
.rating-chip.active { border-color: var(--accent); color: var(--accent); background: var(--accent-dim); }

/* 필드 결합 모드 토글 */
.combine-row {
  display: flex; align-items: center; gap: 6px;
  margin-top: 10px;
}
.combine-label {
  font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-muted);
  letter-spacing: 0; margin-right: 4px;
}
.combine-chip {
  height: 30px; padding: 0 14px; background: var(--bg-button);
  border: 1px solid var(--border); border-radius: var(--radius-pill);
  color: var(--text-secondary); font-size: var(--fs-meta); font-weight: var(--fw-bold);
  cursor: pointer; transition: all 0.15s;
  font-family: 'Consolas', monospace;
}
.combine-chip:hover { color: var(--text-secondary); border-color: var(--text-muted); }
.combine-chip.active {
  border-color: var(--accent); color: var(--accent);
  background: var(--accent-dim);
  box-shadow: 0 0 0 1px var(--accent-dim);
}
/* 무제한 cap 모드 — 활성 시 위험 색상 (orange/red) */
.warn-chip.active {
  border-color: var(--state-warn-fg) !important;
  color: var(--state-warn-fg) !important;
  background: rgba(251, 146, 60, 0.1) !important;
  box-shadow: 0 0 0 1px rgba(251, 146, 60, 0.2) !important;
}
.hint-mode {
  margin-left: 8px;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: var(--fs-label) !important;
}
.hint-mode.and { background: rgba(96, 165, 250, 0.08); color: var(--state-info-fg); }
.hint-mode.or  { background: rgba(250, 204, 21, 0.08); color: var(--accent); }
.hint-mode strong { font-weight: var(--fw-bold); }
.go-btn { padding: 12px 40px; background: var(--accent-fill); border: none; border-radius: var(--radius-pill); color: var(--on-accent); font-weight: var(--fw-bold); font-size: 14px; cursor: pointer; letter-spacing: 0; }
.go-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(250,204,21,0.3); }
.io-row { display: flex; gap: 4px; }
.io-btn { height: 30px; padding: 0 12px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-base); color: var(--text-secondary); font-size: var(--fs-meta); font-weight: var(--fw-bold); cursor: pointer; }
.prev-results {
  display: flex; align-items: center; gap: 10px; padding: 12px 20px;
  background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-card);
}
.prev-label { font-size: 11px; color: var(--text-muted); font-weight: var(--fw-bold); flex: 1; }
.prev-btn {
  padding: 7px 16px; background: var(--bg-button); border: 1px solid var(--border);
  border-radius: 6px; color: var(--text-secondary); font-size: 11px; font-weight: var(--fw-bold); cursor: pointer;
}
.prev-btn:hover { border-color: var(--text-muted); color: var(--text-primary); }
.prev-btn.gold { border-color: var(--accent-dim); color: var(--accent); }

.hints { display: flex; gap: 14px; }
.hints span { font-size: var(--fs-label); color: var(--text-muted); background: var(--bg-card); padding: 4px 12px; border-radius: 4px; }

/* ═══ No Results ═══ */
.no-results {
  flex: 1; display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  padding: 40px; gap: 16px; text-align: center;
}
.nr-icon { font-size: 64px; opacity: 0.3; filter: grayscale(0.5); }
.no-results h2 {
  color: var(--text-primary); font-size: 22px;
  font-weight: var(--fw-bold); letter-spacing: 0;
}
.nr-hint {
  color: var(--text-muted); font-size: 13px; line-height: 1.7;
  max-width: 520px;
}
.nr-hint strong.nr-accent { color: var(--accent); font-weight: var(--fw-bold); }
.nr-hint code {
  background: var(--bg-card); padding: 2px 6px;
  border-radius: 4px; color: var(--accent);
  font-family: 'Consolas', monospace; font-size: 12px;
}
.nr-actions { display: flex; gap: 10px; margin-top: 8px; }
.nr-btn {
  padding: 10px 22px; background: var(--bg-button);
  border: 1px solid var(--border); border-radius: var(--radius-pill);
  color: var(--text-secondary); font-size: 12px; font-weight: var(--fw-bold);
  cursor: pointer; transition: all 0.15s;
}
.nr-btn:hover {
  background: var(--bg-card); color: var(--text-primary);
  border-color: var(--text-muted);
}
.nr-btn.primary {
  background: var(--accent-fill); color: var(--on-accent); border-color: var(--accent-fill);
}
.nr-btn.primary:hover {
  background: var(--accent-fill-hover); transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(250, 204, 21, 0.25);
}
.nr-summary {
  margin-top: 12px; padding: 10px 16px;
  background: var(--bg-card); border-radius: 8px;
  display: flex; align-items: center; gap: 10px;
  max-width: 80%; overflow-x: auto;
}
.nr-label { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-muted); letter-spacing: 0; }
.nr-summary code {
  font-family: 'Consolas', monospace; font-size: 11px;
  color: var(--text-secondary);
  white-space: nowrap;
}

/* ═══ Loading ═══ */
.loading { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 14px; }
.loading h2 { color: var(--text-muted); letter-spacing: 0; }
.loading p { color: var(--text-muted); font-size: 12px; }

/* 원형 진행률 — 90% > 99% 식으로 안에 표시 */
.ring-loader {
  position: relative;
  width: 100px; height: 100px;
}
.ring-loader svg {
  width: 100%; height: 100%;
  /* -90도 회전: 12시 방향에서 시작 (기본은 3시) */
  transform: rotate(-90deg);
}
.ring-track {
  stroke: var(--bg-button);
  stroke-width: 6;
  fill: none;
}
.ring-progress {
  stroke: var(--accent);
  stroke-width: 6;
  stroke-linecap: round;
  fill: none;
  /* 둘레 (2π·42 ≈ 263.89) */
  stroke-dasharray: 263.89;
  /* dashoffset은 :stroke-dashoffset 바인딩으로 동적 — 부드러운 전환 */
  transition: stroke-dashoffset 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  /* 살아있는 느낌의 글로우 */
  filter: drop-shadow(0 0 4px rgba(250, 204, 21, 0.4));
  animation: ring-pulse 2s ease-in-out infinite;
}
@keyframes ring-pulse {
  0%, 100% { opacity: 0.85; }
  50% { opacity: 1; }
}
.ring-center {
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
  pointer-events: none;
}
.ring-pct {
  font-family: 'Consolas', 'JetBrains Mono', monospace;
  font-size: 22px;
  font-weight: var(--fw-bold);
  color: var(--accent);
  letter-spacing: -0.5px;
  text-shadow: 0 0 8px rgba(250, 204, 21, 0.3);
  /* 숫자 변경 시 살짝 튀게 */
  transition: transform 0.15s;
}
.ring-pct::after {
  content: ''; /* % 부호는 텍스트에 포함되어 있음 */
}

/* ═══ Result Bar ═══ */
.result-bar { display: flex; align-items: center; padding: 6px 12px; background: var(--bg-secondary); border-bottom: 1px solid var(--border); gap: 8px; flex-shrink: 0; }
.bar-left, .bar-right { display: flex; align-items: center; gap: 4px; flex-shrink: 0; }
.bar-center { flex: 1; display: flex; align-items: center; gap: 4px; justify-content: center; }
.bar-btn { padding: 4px 10px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 4px; color: var(--text-muted); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer; }
.bar-btn:hover:not(:disabled) { color: var(--text-primary); border-color: var(--text-muted); }
.bar-btn.active { background: var(--accent-dim); border-color: var(--accent); color: var(--accent); }
.bar-btn.gold { color: var(--accent); border-color: var(--accent-dim); }
.bar-btn:disabled { opacity: 0.3; }
.bar-sep { color: var(--rule); }
.bar-idx { font-family: monospace; font-size: 11px; color: var(--text-secondary); }
.bar-idx b { color: var(--accent); font-size: 14px; }
.bar-count { font-size: var(--fs-label); color: var(--text-muted); font-weight: var(--fw-bold); letter-spacing: 0; }
.filter-btn.active { color: var(--accent); border-color: var(--accent-dim); }

/* Filter Manager Modal */
.fm-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 2000; display: flex; align-items: center; justify-content: center; }
.fm-modal { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 16px; width: 560px; max-height: 80vh; display: flex; flex-direction: column; }
.fm-header { display: flex; align-items: center; justify-content: space-between; padding: 16px 20px; border-bottom: 1px solid var(--border); }
.fm-header h3 { font-size: 13px; font-weight: var(--fw-bold); letter-spacing: 0; color: var(--text-primary); }
.fm-body { flex: 1; overflow-y: auto; padding: 16px 20px; display: flex; flex-direction: column; gap: 16px; }
.fm-section { display: flex; flex-direction: column; gap: 6px; }
.fm-label { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-muted); letter-spacing: 0; }
.fm-search { padding: 6px 10px; font-size: 11px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px; color: var(--text-primary); }
.fm-chips { display: flex; flex-wrap: wrap; gap: 4px; }
.fm-chip-scroll { display: flex; flex-wrap: wrap; gap: 4px; max-height: 120px; overflow-y: auto; }
.fm-chip {
  padding: 4px 10px; background: var(--bg-button); border: 1px solid var(--border);
  border-radius: var(--radius-pill); color: var(--text-muted); font-size: var(--fs-label); font-weight: var(--fw-bold);
  cursor: pointer; transition: var(--transition); white-space: nowrap;
}
.fm-chip:hover { border-color: var(--text-muted); color: var(--text-primary); }
.fm-chip.active { border-color: var(--accent); color: var(--accent); background: var(--accent-dim); }
.fm-footer { display: flex; align-items: center; justify-content: space-between; padding: 12px 20px; border-top: 1px solid var(--border); }
.fm-count { font-size: 11px; color: var(--text-muted); font-family: monospace; }
.fm-apply { padding: 8px 24px; background: var(--accent-fill); border: none; border-radius: var(--radius-pill); color: var(--on-accent); font-weight: var(--fw-bold); font-size: 11px; cursor: pointer; letter-spacing: 0; }
.bar-btn.parquet { background: rgba(96,165,250,0.1); border-color: rgba(96,165,250,0.3); color: var(--state-info-fg); }

/* Deep bar */
.deep-bar {
  display: flex; align-items: center; gap: 6px; padding: 5px 12px;
  background: rgba(250,204,21,0.02); border-bottom: 1px solid var(--border); flex-shrink: 0; flex-wrap: wrap;
}
.deep-label { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--accent); letter-spacing: 0; }
.deep-input { width: 160px; padding: 4px 8px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary); font-size: var(--fs-label); }
.deep-input.neg { border-color: rgba(248,113,113,0.2); }
.branch-label { font-size: var(--fs-label); color: var(--text-muted); font-weight: var(--fw-bold); }
.branch-btn { padding: 3px 8px; background: var(--bg-button); border: 1px solid var(--accent-dim); border-radius: 4px; color: var(--accent); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer; }
.branch-btn:hover { background: var(--accent-dim); }

/* ═══ Single View ═══ */
.single-view { flex: 1; overflow-y: auto; padding: 24px; display: flex; justify-content: center; }
.detail-card { max-width: 720px; width: 100%; display: flex; flex-direction: column; gap: 20px; }
.detail-meta { display: flex; gap: 10px; flex-wrap: wrap; }
.meta-pill { padding: 12px 16px; background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 10px; font-size: 14px; font-weight: var(--fw-bold); color: var(--text-primary); display: flex; flex-direction: column; gap: 4px; flex: 1; min-width: 140px; overflow-wrap: anywhere; word-break: break-word; }
.meta-pill.artist { color: var(--accent); }
/* 캐릭터는 '성공'이 아니라 분류다 — 상태색(state-ok)을 쓰면 사용자가 설정에서 성공색을
   바꿨을 때 캐릭터 이름 색이 같이 변한다. 분류색인 tag-person 을 쓴다. */
.meta-pill.character { color: var(--tag-person); }
/* 짧은 값(등급·해상도·태그 수) 카드 — 내용 폭만큼 늘고 줄바꿈하지 않는다. flex: 0 이면
   '640×768' 이 글자마다 쪼개져 세 줄로 찍혔다(부모의 overflow-wrap: anywhere). */
.meta-pill.mini { min-width: 60px; flex: 0 0 auto; white-space: nowrap; overflow-wrap: normal; word-break: keep-all; }
.ml { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-muted); letter-spacing: 0; }

.tag-section { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
.tag-section label { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-muted); letter-spacing: 0; margin-bottom: 10px; display: block; }
.tag-cloud { display: flex; flex-wrap: wrap; gap: 5px; max-height: 300px; overflow-y: auto; }
.tag { padding: 4px 10px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 4px; color: var(--text-muted); font-size: var(--fs-label); overflow-wrap: anywhere; max-width: 100%; }
/* 아래 16개 태그 분류색은 아직 토큰화하지 않았다 — 테마의 태그 토큰은 6색(person/scene/
   pose/wear/fx/nsfw)뿐이라, 16개를 6개에 욱여넣으면 서로 다른 분류가 같은 색이 되어
   범례(.legend-item)가 구분 기능을 잃는다. 팔레트를 늘릴지 분류를 합칠지는 설계 결정이라
   여기서 임의로 정하지 않는다. (라이트 모드에서 이 색들은 흰 배경 대비가 부족하다.) */
.tag.count { color: #60a5fa; border-color: rgba(96,165,250,0.3); background: rgba(96,165,250,0.05); }
.tag.clothing { color: #a78bfa; border-color: rgba(167,139,250,0.3); background: rgba(167,139,250,0.05); }
.tag.body { color: #fb923c; border-color: rgba(251,146,60,0.3); background: rgba(251,146,60,0.05); }
.tag.nsfw { color: #f87171; border-color: rgba(248,113,113,0.3); background: rgba(248,113,113,0.05); }
.tag.action { color: #4ade80; border-color: rgba(74,222,128,0.3); background: rgba(74,222,128,0.05); }
.tag.expression { color: #fbbf24; border-color: rgba(251,191,36,0.3); background: rgba(251,191,36,0.05); }
.tag.bg { color: #38bdf8; border-color: rgba(56,189,248,0.3); background: rgba(56,189,248,0.05); }
.tag.objects { color: #94a3b8; border-color: rgba(148,163,184,0.3); background: rgba(148,163,184,0.05); }
.tag.effect { color: #c084fc; border-color: rgba(192,132,252,0.3); background: rgba(192,132,252,0.05); }
.tag.color-tag { color: #f472b6; border-color: rgba(244,114,182,0.3); background: rgba(244,114,182,0.05); }
.tag.trait { color: #34d399; border-color: rgba(52,211,153,0.3); background: rgba(52,211,153,0.05); }
.tag.composition { color: #818cf8; border-color: rgba(129,140,248,0.3); background: rgba(129,140,248,0.05); }
.tag.style { color: #e879f9; border-color: rgba(232,121,249,0.3); background: rgba(232,121,249,0.05); }
.tag.char { color: #2dd4bf; border-color: rgba(45,212,191,0.3); background: rgba(45,212,191,0.05); }
.tag.copy { color: #22d3ee; border-color: rgba(34,211,238,0.3); background: rgba(34,211,238,0.05); }
.tag.artist-tag { color: #facc15; border-color: rgba(250,204,21,0.3); background: rgba(250,204,21,0.05); }

.tag-legend { display: flex; gap: 6px; flex-wrap: wrap; }
.legend-item { font-size: var(--fs-label); font-weight: var(--fw-bold); padding: 2px 6px; border-radius: 3px; }
.legend-item.count { color: #60a5fa; background: rgba(96,165,250,0.1); }
.legend-item.clothing { color: #a78bfa; background: rgba(167,139,250,0.1); }
.legend-item.body { color: #fb923c; background: rgba(251,146,60,0.1); }
.legend-item.nsfw { color: #f87171; background: rgba(248,113,113,0.1); }
.legend-item.action { color: #4ade80; background: rgba(74,222,128,0.1); }
.legend-item.expression { color: #fbbf24; background: rgba(251,191,36,0.1); }
.legend-item.bg { color: #38bdf8; background: rgba(56,189,248,0.1); }
.legend-item.objects { color: #94a3b8; background: rgba(148,163,184,0.1); }
.legend-item.effect { color: #c084fc; background: rgba(192,132,252,0.1); }
.legend-item.color-tag { color: #f472b6; background: rgba(244,114,182,0.1); }
.legend-item.trait { color: #34d399; background: rgba(52,211,153,0.1); }

.detail-actions { display: flex; gap: 10px; }
.primary-btn { flex: 2; height: 48px; background: var(--accent-fill); border: none; border-radius: var(--radius-pill); color: var(--on-accent); font-weight: var(--fw-bold); font-size: 13px; letter-spacing: 0; cursor: pointer; }
.secondary-btn { flex: 1; height: 48px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-pill); color: var(--text-secondary); font-weight: var(--fw-bold); font-size: 11px; cursor: pointer; }
.primary-btn:hover, .secondary-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.3); }

/* ═══ List View ═══ */
.list-view { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.list-toolbar {
  display: flex; justify-content: space-between; align-items: center;
  padding: 8px 16px; background: var(--bg-primary); border-bottom: 1px solid var(--border);
  gap: 16px; flex-wrap: wrap;
}
.list-sort { display: flex; align-items: center; gap: 6px; }
.sort-label { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-muted); letter-spacing: 0; }
.sort-select {
  background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px;
  padding: 5px 10px; color: var(--text-primary); font-size: 11px; font-weight: var(--fw-bold);
  cursor: pointer; outline: none;
}
.sort-select:hover { border-color: var(--accent); }
.sort-dir-btn {
  background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px;
  padding: 5px 10px; color: var(--accent); font-size: 12px; font-weight: var(--fw-bold);
  cursor: pointer; min-width: 28px;
}
.sort-dir-btn:hover { background: var(--bg-button); }
.list-pager { display: flex; align-items: center; gap: 4px; }
.page-btn {
  background: var(--bg-input); border: 1px solid var(--border); border-radius: 4px;
  padding: 4px 10px; color: var(--text-secondary); font-size: 11px; font-weight: var(--fw-bold);
  cursor: pointer; min-width: 32px;
}
.page-btn:hover:not(:disabled) { background: var(--bg-button); color: var(--accent); }
.page-btn:disabled { opacity: 0.3; cursor: not-allowed; }
.page-info {
  font-size: 11px; font-weight: var(--fw-bold); color: var(--text-secondary);
  padding: 0 10px; min-width: 100px; text-align: center;
}
.list-header { display: flex; padding: 6px 16px; background: var(--bg-secondary); border-bottom: 1px solid var(--border); font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-muted); letter-spacing: 0; }
.lh.idx { width: 40px; } .lh.char { width: 160px; } .lh.copy { width: 120px; } .lh.artist { width: 100px; } .lh.tags { flex: 1; } .lh.act { width: 44px; }
.list-scroll { flex: 1; overflow-y: auto; }
.list-row { display: flex; align-items: center; padding: 6px 16px; font-size: 11px; border-bottom: 1px solid rgba(255,255,255,0.02); cursor: pointer; }
.list-row:hover { background: rgba(255,255,255,0.02); }
.list-row.active { background: var(--accent-dim); }
.lr.idx { width: 40px; color: var(--text-muted); font-family: monospace; font-size: var(--fs-label); }
.lr.char { width: 160px; color: var(--tag-person); font-weight: var(--fw-bold); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lr.copy { width: 120px; color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lr.artist { width: 100px; color: var(--accent); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lr.tags { flex: 1; color: var(--text-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: var(--fs-label); }
.lr.act { width: 44px; text-align: center; }
.use-btn { padding: 2px 8px; background: var(--accent-fill); border: none; border-radius: 3px; color: var(--on-accent); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer; }
.list-pager { display: flex; align-items: center; justify-content: center; gap: 12px; padding: 6px; border-top: 1px solid var(--border); }
.pager-info { font-size: 11px; color: var(--text-muted); font-family: monospace; }

label.danger { color: var(--state-alert-fg); }

/* 조건부 프롬프트 */
.cond-section { padding: 12px 16px; border-top: 1px solid var(--border); flex-shrink: 0; display: flex; flex-direction: column; gap: 8px; }
.cond-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-items: start; }
.cond-card { background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
.cond-card.neg { border-color: rgba(248,113,113,0.15); }
.cond-title { font-size: 12px; font-weight: var(--fw-bold); letter-spacing: 0; cursor: pointer; list-style: none; }
.cond-title::-webkit-details-marker { display: none; }
.cond-title.positive { color: var(--state-ok-fg); }
.cond-title.negative { color: var(--state-alert-fg); }
.cond-desc { font-size: var(--fs-label); color: var(--text-muted); margin: 4px 0 8px; }
.cond-rule { display: flex; align-items: center; gap: 4px; margin-bottom: 4px; flex-wrap: wrap; }
.cond-rule-block { border: 1px solid var(--border); border-radius: 6px; padding: 6px; margin-bottom: 4px; }
.cond-row1, .cond-row2 { display: flex; align-items: center; gap: 4px; }
.cond-row2 { margin-top: 3px; }
.cond-check input { accent-color: var(--accent); }
.cond-kw { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--accent); }
.cond-input { flex: 1; min-width: 80px; padding: 6px 10px; font-size: 12px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary); }
.cond-input.neg { border-color: rgba(248,113,113,0.2); }
.cond-sel { padding: 3px 4px; font-size: var(--fs-label); background: var(--bg-input); border: 1px solid var(--border); border-radius: 3px; color: var(--text-secondary); }
.cond-sel.sm { width: 55px; }
.cond-rm { background: none; border: none; color: var(--state-alert-fg); cursor: pointer; font-size: 12px; padding: 0 2px; }
.cond-save-row { display: flex; align-items: center; gap: 10px; margin-top: 12px; }
.cond-autosave {
  font-size: var(--fs-label); color: var(--state-ok-fg);
  background: rgba(74, 222, 128, 0.08); padding: 6px 10px;
  border-radius: 6px; font-weight: var(--fw-bold); letter-spacing: 0;
}
.cond-save-btn { flex: 1; padding: 8px; background: var(--accent-fill); border: none; border-radius: 8px; color: var(--on-accent); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer; }
.cond-add { width: 100%; padding: 5px; background: var(--bg-button); border: 1px dashed var(--border); border-radius: 4px; color: var(--text-muted); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer; margin-top: 4px; }
</style>
