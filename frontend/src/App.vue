<template>
  <div class="app-container">
    <header class="app-header">
      <TabBar @tab-changed="onTabChanged" />
    </header>

    <main class="main-workspace">
      <!-- Left Panel -->
      <aside class="side-panel left" v-if="showLeftPanel">
        <div class="panel-scroll">
          <PromptPanel @toggle-extend="showExtendPanel = !showExtendPanel"
            @open-wildcard="openWildcardByName" />
          <div class="tool-card">
            <label>Studio Tools</label>
            <div class="tool-grid">
              <button class="tool-btn" @click="action('save_settings')">SAVE</button>
              <button class="tool-btn" @click="showPresetManager = true; loadPresetList()">PRESET</button>
              <button class="tool-btn" @click="showWeightManager = true">WEIGHT</button>
              <button class="tool-btn" @click="showWcManager = true">WILDCARD</button>
              <button class="tool-btn" @click="action('ab_test')">A/B TEST</button>
              <button class="tool-btn" @click="showStatsModal = true; loadGenStats()">STATS</button>
            </div>
          </div>

        </div>
        <div class="gen-footer">
          <div class="gen-actions">
            <button class="action-btn" :class="{ active: autoMode }" @click="autoMode = !autoMode; action('toggle_automation', { checked: autoMode })">
              {{ autoMode ? '🔄 AUTO ON' : '⏸ AUTO OFF' }}
            </button>
            <button class="action-btn highlight" @click="action('random_prompt')">🎲 RANDOM</button>
          </div>
          <!-- 자동화 설정 (AUTO ON일 때만) -->
          <div class="auto-settings" v-if="autoMode">
            <div class="auto-row">
              <label>{{ autoSettings.mode === 'count' ? '횟수' : '시간(분)' }}</label>
              <input type="number" v-model.number="autoSettings.limit" min="1" class="auto-input" />
              <CustomSelect v-model="autoSettings.mode" :options="['count', 'timer']" placeholder="모드" class="auto-select" />
            </div>
            <div class="auto-row">
              <label>반복</label>
              <input type="number" v-model.number="autoSettings.repeat" min="1" max="100" class="auto-input" />
              <label>대기(초)</label>
              <input type="number" v-model.number="autoSettings.delay" min="0" step="0.5" class="auto-input" />
            </div>
            <label class="auto-check"><input type="checkbox" v-model="autoSettings.allowDupes" /><span>중복 허용</span></label>
          </div>
          <!-- 자동화 상태 표시 -->
          <div class="auto-status" v-if="isAutomating">
            <div class="auto-status-bar">
              <span class="auto-pulse"></span>
              <span>자동화 진행 중 — {{ autoGenCount }}장 완료</span>
            </div>
            <div class="auto-status-sub" v-if="autoWaiting">대기 중... ({{ autoSettings.delay }}초)</div>
          </div>
          <div class="generate-row">
            <button class="btn-generate" :class="{ automating: isAutomating }" @click="doGenerate" :disabled="isGenerating && !isAutomating">
              {{ isAutomating ? '⏹ STOP AUTOMATION' : isGenerating ? 'GENERATING...' : autoMode ? '▶ START AUTOMATION' : 'GENERATE IMAGE' }}
            </button>
          </div>
        </div>
      </aside>

      <!-- 반달 화살표 (좌측 패널 옆, 항상 표시) -->
      <div class="half-moon" v-if="showLeftPanel" @click="showExtendPanel = !showExtendPanel"
        :class="{ open: showExtendPanel }">
        <span>{{ showExtendPanel ? '◀' : '▶' }}</span>
      </div>

      <!-- Extended Panel — 뷰어 위에 오버레이 -->
      <transition name="slide">
        <aside class="extend-overlay" v-if="showExtendPanel && showLeftPanel">
          <div class="extend-header">
            <h3>ADVANCED SETTINGS</h3>
            <button class="close-btn" @click="showExtendPanel = false">✕</button>
          </div>
          <div class="extend-scroll">
            <!-- Parameters (기본) -->
            <div class="ext-card">
              <div class="ext-title">PARAMETERS</div>
              <div class="ext-field">
                <label>Resolution</label>
                <div class="ext-res-row">
                  <input type="number" v-model="storeWidgets.width_input" />
                  <span>×</span>
                  <input type="number" v-model="storeWidgets.height_input" />
                  <button class="ext-mini-btn" @click="action('swap_resolution')">↔</button>
                </div>
                <div class="ext-res-opts">
                  <label class="ext-check-sm"><input type="checkbox" v-model="randomResEnabled" /><span>랜덤</span></label>
                  <label class="ext-check-sm"><input type="checkbox" v-model="autoResEnabled" /><span>자동(Parquet)</span></label>
                </div>
                <!-- 랜덤 해상도 편집기 -->
                <div v-if="randomResEnabled" class="rand-res-editor">
                  <div class="rand-res-list">
                    <div v-for="(r, i) in randomResList" :key="i" class="rand-res-item">
                      <span class="rand-res-val">{{ r[0] }}×{{ r[1] }}</span>
                      <span class="rand-res-desc">{{ r[2] }}</span>
                      <button class="rand-res-del" @click="removeRandomRes(i)">✕</button>
                    </div>
                    <div v-if="!randomResList.length" class="rand-res-empty">해상도를 추가하세요</div>
                  </div>
                  <div class="rand-res-add">
                    <input type="number" v-model.number="newResW" placeholder="W" class="rand-res-input" />
                    <span>×</span>
                    <input type="number" v-model.number="newResH" placeholder="H" class="rand-res-input" />
                    <button class="rand-res-btn" @click="addRandomRes">+</button>
                  </div>
                </div>
              </div>
              <div class="ext-row">
                <div class="ext-field"><label>Sampler</label>
                  <CustomSelect v-model="storeWidgets.sampler_combo" :options="samplerItems" placeholder="Sampler..." />
                </div>
                <div class="ext-field"><label>Scheduler</label>
                  <CustomSelect v-model="storeWidgets.scheduler_combo" :options="schedulerItems" placeholder="Scheduler..." />
                </div>
              </div>
              <div class="ext-row">
                <div class="ext-field"><label>Steps</label><input type="number" v-model="storeWidgets.steps_input" min="1" max="150" /></div>
                <div class="ext-field"><label>CFG</label><input type="number" v-model="storeWidgets.cfg_input" step="0.5" /></div>
                <div class="ext-field"><label>Seed</label><input type="text" v-model="storeWidgets.seed_input" /></div>
              </div>
            </div>

            <!-- Hires.fix -->
            <details class="ext-card">
              <summary class="ext-title">HIRES.FIX</summary>
              <label class="ext-check-row"><input type="checkbox" v-model="hires_enabled" /><span>Hires.fix 활성화</span></label>
              <div class="ext-field">
                <label>Upscaler</label>
                <CustomSelect v-model="storeWidgets.upscaler_combo" :options="upscalerItems" placeholder="Upscaler..." />
              </div>
              <div class="ext-row">
                <div class="ext-field"><label>Steps</label><input type="number" v-model="storeWidgets.hires_steps_input" /></div>
                <div class="ext-field"><label>Denoise</label><input type="number" v-model="storeWidgets.hires_denoising_input" step="0.05" /></div>
              </div>
              <div class="ext-row">
                <div class="ext-field"><label>Scale</label><input type="number" v-model="storeWidgets.hires_scale_input" step="0.1" min="1" /></div>
                <div class="ext-field"><label>CFG (0=off)</label><input type="number" v-model="storeWidgets.hires_cfg_input" step="0.5" /></div>
              </div>
            </details>

            <details class="ext-card" open>
              <summary class="ext-title">PROMPT FILTERS</summary>
              <!-- Rating 토글 -->
              <div class="ext-sub-title">RATING FILTER</div>
              <div class="rating-toggle-row">
                <button v-for="r in ratingFilters" :key="r.key" class="rating-toggle"
                  :class="{ active: r.on }" @click="r.on = !r.on; saveRatingFilter()">{{ r.label }}</button>
              </div>
              <div class="ext-toggle-grid">
                <label class="ext-check-row"><input type="checkbox" v-model="removeCharacter" /><span>캐릭터 제거</span></label>
                <label class="ext-check-row"><input type="checkbox" v-model="removeCharacterFeatures" /><span>캐릭터 특징 제거</span></label>
                <label class="ext-check-row"><input type="checkbox" v-model="removeCopyright" /><span>작품 제거</span></label>
                <label class="ext-check-row"><input type="checkbox" v-model="removeArtist" /><span>작가 제거</span></label>
                <label class="ext-check-row"><input type="checkbox" v-model="removeMeta" /><span>메타 제거</span></label>
                <label class="ext-check-row"><input type="checkbox" v-model="removeCensorship" /><span>검열 제거</span></label>
                <label class="ext-check-row"><input type="checkbox" v-model="removeText" /><span>텍스트 제거</span></label>
                <label class="ext-check-row"><input type="checkbox" v-model="autoCharFeatures" /><span>특징 자동 추가</span></label>
              </div>
            </details>

            <!-- ADetailer -->
            <details class="ext-card">
              <summary class="ext-title">ADETAILER</summary>
              <label class="ext-check-row"><input type="checkbox" v-model="ad_enabled" /><span>ADetailer 활성화</span></label>
              <!-- Slot 1 -->
              <div class="ext-sub-title">Slot 1</div>
              <label class="ext-check-row"><input type="checkbox" v-model="ad_s1_enabled" /><span>Slot 1 활성화</span></label>
              <div class="ext-field"><label>Model</label>
                <CustomSelect v-model="storeWidgets._ad_s1_model" :options="adModelItems" placeholder="AD Model..." /></div>
              <div class="ext-field"><label>Prompt</label>
                <input type="text" v-model="storeWidgets._ad_s1_prompt" placeholder="ADetailer prompt..." /></div>
              <div class="ext-field"><label>Negative Prompt</label>
                <input type="text" v-model="storeWidgets._ad_s1_neg" placeholder="AD negative..." /></div>
              <div class="ext-row">
                <div class="ext-field"><label>Confidence</label><input type="number" v-model="storeWidgets._ad_s1_confidence" step="0.05" min="0" max="1" /></div>
                <div class="ext-field"><label>Denoise</label><input type="number" v-model="storeWidgets._ad_s1_denoise" step="0.05" min="0" max="1" /></div>
              </div>
              <div class="ext-row">
                <div class="ext-field"><label>Mask Blur</label><input type="number" v-model="storeWidgets._ad_s1_mask_blur" min="0" /></div>
                <div class="ext-field"><label>Padding</label><input type="number" v-model="storeWidgets._ad_s1_padding" min="0" /></div>
                <div class="ext-field"><label>Dilate/Erode</label><input type="number" v-model="storeWidgets._ad_s1_dilate_erode" /></div>
              </div>
              <div class="ext-field"><label>Mask Merge</label>
                <CustomSelect v-model="storeWidgets._ad_s1_mask_merge" :options="['None', 'Merge', 'Merge and Invert']" placeholder="None" /></div>
              <!-- Separate settings -->
              <label class="ext-check-row"><input type="checkbox" v-model="storeWidgets._ad_s1_use_inp_size" true-value="true" false-value="false" /><span>별도 Inpaint 크기</span></label>
              <div class="ext-row" v-if="storeWidgets._ad_s1_use_inp_size === 'true'">
                <div class="ext-field"><label>Width</label><input type="number" v-model="storeWidgets._ad_s1_inp_w" /></div>
                <div class="ext-field"><label>Height</label><input type="number" v-model="storeWidgets._ad_s1_inp_h" /></div>
              </div>
              <label class="ext-check-row"><input type="checkbox" v-model="storeWidgets._ad_s1_use_steps" true-value="true" false-value="false" /><span>별도 Steps</span></label>
              <div class="ext-row" v-if="storeWidgets._ad_s1_use_steps === 'true'">
                <div class="ext-field"><label>Steps</label><input type="number" v-model="storeWidgets._ad_s1_steps" min="1" /></div>
              </div>
              <label class="ext-check-row"><input type="checkbox" v-model="storeWidgets._ad_s1_use_cfg" true-value="true" false-value="false" /><span>별도 CFG</span></label>
              <div class="ext-row" v-if="storeWidgets._ad_s1_use_cfg === 'true'">
                <div class="ext-field"><label>CFG</label><input type="number" v-model="storeWidgets._ad_s1_cfg" step="0.5" /></div>
              </div>
              <label class="ext-check-row"><input type="checkbox" v-model="storeWidgets._ad_s1_use_sampler" true-value="true" false-value="false" /><span>별도 Sampler</span></label>
              <div class="ext-row" v-if="storeWidgets._ad_s1_use_sampler === 'true'">
                <div class="ext-field"><label>Sampler</label>
                  <CustomSelect v-model="storeWidgets._ad_s1_sampler" :options="samplerItems" placeholder="Sampler" /></div>
                <div class="ext-field"><label>Scheduler</label>
                  <CustomSelect v-model="storeWidgets._ad_s1_scheduler" :options="schedulerItems" placeholder="Scheduler" /></div>
              </div>
              <label class="ext-check-row"><input type="checkbox" v-model="storeWidgets._ad_s1_use_ckpt" true-value="true" false-value="false" /><span>별도 Checkpoint</span></label>
              <label class="ext-check-row"><input type="checkbox" v-model="storeWidgets._ad_s1_use_vae" true-value="true" false-value="false" /><span>별도 VAE</span></label>

              <!-- Slot 2 -->
              <details class="ext-sub">
                <summary>Slot 2</summary>
                <label class="ext-check-row"><input type="checkbox" v-model="ad_s2_enabled" /><span>Slot 2 활성화</span></label>
                <div class="ext-field"><label>Model</label>
                  <CustomSelect v-model="storeWidgets._ad_s2_model" :options="adModelItems" placeholder="AD Model..." /></div>
                <div class="ext-field"><label>Prompt</label>
                  <input type="text" v-model="storeWidgets._ad_s2_prompt" placeholder="Slot 2 prompt..." /></div>
                <div class="ext-field"><label>Negative Prompt</label>
                  <input type="text" v-model="storeWidgets._ad_s2_neg" placeholder="AD negative..." /></div>
                <div class="ext-row">
                  <div class="ext-field"><label>Confidence</label><input type="number" v-model="storeWidgets._ad_s2_confidence" step="0.05" min="0" max="1" /></div>
                  <div class="ext-field"><label>Denoise</label><input type="number" v-model="storeWidgets._ad_s2_denoise" step="0.05" min="0" max="1" /></div>
                </div>
                <div class="ext-row">
                  <div class="ext-field"><label>Mask Blur</label><input type="number" v-model="storeWidgets._ad_s2_mask_blur" min="0" /></div>
                  <div class="ext-field"><label>Padding</label><input type="number" v-model="storeWidgets._ad_s2_padding" min="0" /></div>
                  <div class="ext-field"><label>Dilate/Erode</label><input type="number" v-model="storeWidgets._ad_s2_dilate_erode" /></div>
                </div>
                <div class="ext-field"><label>Mask Merge</label>
                  <CustomSelect v-model="storeWidgets._ad_s2_mask_merge" :options="['None', 'Merge', 'Merge and Invert']" placeholder="None" /></div>
                <label class="ext-check-row"><input type="checkbox" v-model="storeWidgets._ad_s2_use_inp_size" true-value="true" false-value="false" /><span>별도 Inpaint 크기</span></label>
                <div class="ext-row" v-if="storeWidgets._ad_s2_use_inp_size === 'true'">
                  <div class="ext-field"><label>Width</label><input type="number" v-model="storeWidgets._ad_s2_inp_w" /></div>
                  <div class="ext-field"><label>Height</label><input type="number" v-model="storeWidgets._ad_s2_inp_h" /></div>
                </div>
                <label class="ext-check-row"><input type="checkbox" v-model="storeWidgets._ad_s2_use_steps" true-value="true" false-value="false" /><span>별도 Steps</span></label>
                <div class="ext-row" v-if="storeWidgets._ad_s2_use_steps === 'true'">
                  <div class="ext-field"><label>Steps</label><input type="number" v-model="storeWidgets._ad_s2_steps" min="1" /></div>
                </div>
                <label class="ext-check-row"><input type="checkbox" v-model="storeWidgets._ad_s2_use_cfg" true-value="true" false-value="false" /><span>별도 CFG</span></label>
                <div class="ext-row" v-if="storeWidgets._ad_s2_use_cfg === 'true'">
                  <div class="ext-field"><label>CFG</label><input type="number" v-model="storeWidgets._ad_s2_cfg" step="0.5" /></div>
                </div>
                <label class="ext-check-row"><input type="checkbox" v-model="storeWidgets._ad_s2_use_sampler" true-value="true" false-value="false" /><span>별도 Sampler</span></label>
                <div class="ext-row" v-if="storeWidgets._ad_s2_use_sampler === 'true'">
                  <div class="ext-field"><label>Sampler</label>
                    <CustomSelect v-model="storeWidgets._ad_s2_sampler" :options="samplerItems" placeholder="Sampler" /></div>
                  <div class="ext-field"><label>Scheduler</label>
                    <CustomSelect v-model="storeWidgets._ad_s2_scheduler" :options="schedulerItems" placeholder="Scheduler" /></div>
                </div>
                <label class="ext-check-row"><input type="checkbox" v-model="storeWidgets._ad_s2_use_ckpt" true-value="true" false-value="false" /><span>별도 Checkpoint</span></label>
                <label class="ext-check-row"><input type="checkbox" v-model="storeWidgets._ad_s2_use_vae" true-value="true" false-value="false" /><span>별도 VAE</span></label>
              </details>
            </details>

            <!-- NegPiP -->
            <div class="ext-card">
              <label class="ext-check-row">
                <input type="checkbox" v-model="negpipEnabled" />
                <span class="ext-title" style="margin:0">NegPiP 확장</span>
              </label>
              <div class="ext-hint">(keyword:-1.0) 네거티브 가중치 문법</div>
            </div>

            <!-- 조건부 프롬프트 → Search 탭에서 관리 -->
            <div class="ext-card">
              <div class="ext-title">CONDITIONAL PROMPTS</div>
              <div class="ext-hint">조건부 프롬프트는 Search 탭 하단에서 관리합니다</div>
              <label class="ext-check-row"><input type="checkbox" v-model="extWidgets.cond_prevent_dupe" /><span>중복 방지</span></label>
            </div>

            <!-- LoRA Stack -->
            <div class="ext-card">
              <div class="ext-title">LoRA STACK</div>
              <div class="lora-empty" v-if="loraStack.length === 0">
                LoRA Manager에서 추가하세요
              </div>
              <div v-for="(lora, i) in loraStack" :key="i" class="lora-block">
                <label class="lora-check"><input type="checkbox" v-model="lora.enabled" /></label>
                <div class="lora-info-col">
                  <div class="lora-name">{{ lora.name }}</div>
                  <div class="lora-triggers" v-if="lora.triggerWords && lora.triggerWords.length">
                    <button v-for="tw in lora.triggerWords" :key="tw" class="trigger-chip"
                      @click="insertTriggerWord(tw)" :title="'클릭하여 프롬프트에 삽입'">{{ tw }}</button>
                  </div>
                </div>
                <input type="range" min="-100" max="200" v-model.number="lora.weight" class="lora-slider" />
                <span class="lora-weight">{{ (lora.weight / 100).toFixed(2) }}</span>
                <button class="lora-remove" @click="loraStack.splice(i, 1)">✕</button>
              </div>
              <button class="ext-add-btn" @click="action('open_lora_manager')">+ ADD LoRA</button>
            </div>
          </div>
        </aside>
      </transition>

      <!-- Center: Viewport + EXIF Bar -->
      <section class="viewport-area">
        <div class="viewport-main">
          <router-view v-slot="{ Component }">
            <keep-alive>
              <component :is="Component"
                :image-url="currentImage"
                :resolution="resolution"
                :seed="seed"
                :status="status"
              />
            </keep-alive>
          </router-view>
        </div>
        <!-- EXIF Info Bar (Positive / Negative / Parameters 3탭) -->
        <div class="exif-bar" v-if="showLeftPanel && currentImage">
          <div class="exif-tabs">
            <button v-for="tab in exifTabs" :key="tab.id" class="exif-tab"
              :class="{ active: activeExifTab === tab.id }" @click="activeExifTab = tab.id">
              {{ tab.label }}
            </button>
          </div>
          <div class="exif-content" v-if="activeExifTab !== 'params'">{{ exifContent }}</div>
          <div class="exif-params" v-else-if="currentExif.params">
            <div class="param-line" v-if="currentExif.params.generation"><span class="pl">GEN</span>{{ currentExif.params.generation }}</div>
            <div class="param-line" v-if="currentExif.params.core"><span class="pl">CORE</span>{{ currentExif.params.core }}</div>
            <div class="param-line" v-if="currentExif.params.model"><span class="pl">MODEL</span>{{ currentExif.params.model }}</div>
            <div class="param-line" v-if="currentExif.params.hires"><span class="pl">HIRES</span>{{ currentExif.params.hires }}</div>
            <div class="param-line" v-if="currentExif.params.extensions"><span class="pl">EXT</span>{{ currentExif.params.extensions }}</div>
            <div class="param-line other" v-if="currentExif.params.other"><span class="pl">ETC</span>{{ currentExif.params.other }}</div>
          </div>
          <div class="exif-content" v-else>{{ currentExif.params_line || currentExif.raw || 'No parameters' }}</div>
        </div>
      </section>

      <!-- Right: History -->
      <aside class="side-panel right" v-if="showLeftPanel">
        <div class="hist-header">
          <h3>HISTORY</h3>
          <span class="count-badge">{{ historyImages.length }}</span>
        </div>
        <button class="hist-nav-btn" @click="histPage = Math.max(0, histPage - 1)" :disabled="histPage <= 0">▲</button>
        <div class="hist-scroll">
          <div v-for="img in visibleHistory" :key="img" class="hist-card"
            @click="selectHistoryImage(img)"
            @contextmenu.prevent="showHistoryMenu($event, img)"
            :class="{ selected: currentImage === img }"
            draggable="true" @dragstart="onDragStart($event, img)"
          >
            <img :src="'file:///' + img" loading="lazy" />
          </div>
        </div>
        <button class="hist-nav-btn" @click="histPage++" :disabled="(histPage + 1) * histPerPage >= historyImages.length">▼</button>

        <transition name="pop">
          <div v-if="ctxMenu.show" class="modern-ctx-menu" :style="ctxMenuStyle">
            <div class="ctx-item" @click="ctxAddFavorite">⭐ ADD TO FAVORITES</div>
            <div class="ctx-item" @click="ctxSendI2I">🖼️ SEND TO I2I</div>
            <div class="ctx-item" @click="ctxSendInpaint">🎨 SEND TO INPAINT</div>
            <div class="ctx-item" @click="ctxSendEditor">✏️ SEND TO EDITOR</div>
            <div class="ctx-item" @click="ctxCompare('before')">🔍 COMPARE (BEFORE)</div>
            <div class="ctx-item" @click="ctxCompare('after')">🔍 COMPARE (AFTER)</div>
            <div class="ctx-item" @click="ctxRunAdetailer">🎯 ADETAILER</div>
            <div class="ctx-separator"></div>
            <div class="ctx-item" @click="ctxCopyPath">📋 COPY PATH</div>
            <div class="ctx-item delete" @click="ctxDelete">🗑️ DELETE</div>
          </div>
        </transition>
      </aside>
    </main>

    <div class="global-progress" v-if="isGenerating">
      <div class="progress-fill" :style="{ width: progressVal + '%' }"></div>
    </div>

    <QueuePanel />

    <!-- VRAM 게이지 (하단 고정) -->
    <div class="vram-bar" v-if="vramInfo.total > 0">
      <div class="vram-fill" :style="{ width: vramInfo.pct + '%' }" :class="vramClass"></div>
      <span class="vram-text">VRAM {{ vramInfo.used }}GB / {{ vramInfo.total }}GB ({{ vramInfo.pct }}%)</span>
    </div>

    <!-- Preset Manager Modal -->
    <transition name="fade">
      <div v-if="showPresetManager" class="pm-overlay" @click.self="showPresetManager = false">
        <div class="pm-modal">
          <div class="pm-header">
            <h3>PRESET MANAGER</h3>
            <button class="close-btn" @click="showPresetManager = false">✕</button>
          </div>
          <div class="pm-body">
            <div class="pm-list">
              <div v-for="p in presetList" :key="p" class="pm-item"
                :class="{ active: selectedPreset === p }" @click="selectedPreset = p; loadPresetPreview(p)">
                {{ p }}
              </div>
              <div v-if="presetList.length === 0" class="pm-empty">저장된 프리셋 없음</div>
            </div>
            <div class="pm-preview">
              <div v-if="presetPreview" class="pm-detail">
                <div class="pm-field" v-for="(val, key) in presetPreview" :key="key">
                  <span class="pm-key">{{ key }}</span>
                  <span class="pm-val">{{ typeof val === 'string' ? val.substring(0, 100) : val }}</span>
                </div>
              </div>
              <div v-else class="pm-empty">프리셋을 선택하세요</div>
            </div>
          </div>
          <div class="pm-footer">
            <button class="pm-btn" @click="loadSelectedPreset" :disabled="!selectedPreset">📂 LOAD</button>
            <button class="pm-btn" @click="deleteSelectedPreset" :disabled="!selectedPreset">🗑 DELETE</button>
            <div class="pm-spacer"></div>
            <button class="pm-btn accent" @click="saveNewPreset">💾 SAVE CURRENT</button>
          </div>
        </div>
      </div>
    </transition>

    <!-- Weight Manager Modal -->
    <transition name="fade">
      <div v-if="showWeightManager" class="wm-overlay" @click.self="showWeightManager = false">
        <div class="wm-modal">
          <div class="wm-header">
            <h3>GLOBAL TAG WEIGHTS</h3>
            <span class="wm-desc">등록된 태그는 어디서든 자동으로 지정 가중치가 적용됩니다</span>
            <button class="close-btn" @click="showWeightManager = false">✕</button>
          </div>
          <div class="wm-body">
            <div v-for="(w, i) in globalWeights" :key="i" class="wm-row">
              <input v-model="w.tag" placeholder="태그명..." class="wm-tag-input" />
              <input type="range" min="50" max="200" v-model.number="w.weight" class="wm-slider" />
              <span class="wm-val">{{ (w.weight / 100).toFixed(2) }}</span>
              <button class="wm-rm" @click="globalWeights.splice(i, 1)">✕</button>
            </div>
            <button class="wm-add" @click="globalWeights.push({ tag: '', weight: 100 })">+ ADD TAG WEIGHT</button>
          </div>
          <div class="wm-footer">
            <button class="wm-save" @click="saveGlobalWeights">💾 SAVE</button>
          </div>
        </div>
      </div>
    </transition>

    <!-- Wildcard Manager Modal -->
    <transition name="fade">
      <div v-if="showWcManager" class="wc-overlay" @click.self="showWcManager = false">
        <div class="wc-modal">
          <div class="wc-modal-header">
            <h3>WILDCARD MANAGER</h3>
            <span class="wc-path">wildcards/</span>
            <button class="wc-new-btn" @click="createNewWildcard">+ NEW</button>
            <button class="close-btn" @click="showWcManager = false">✕</button>
          </div>
          <div class="wc-modal-body">
            <!-- 파일 목록 -->
            <div class="wc-sidebar">
              <div v-for="wc in wildcards" :key="wc.name" class="wc-file-item"
                :class="{ active: selectedWc === wc.name }" @click="selectWildcard(wc.name)">
                <span class="wc-fname" @dblclick.stop="renameWildcard(wc.name)">{{ wc.name }}</span>
                <span class="wc-file-count">{{ wc.tags.length }}</span>
                <button class="wc-del" @click.stop="deleteWildcard(wc.name)">✕</button>
              </div>
            </div>
            <!-- 편집 영역 -->
            <div class="wc-content">
              <template v-if="selectedWcData">
                <div class="wc-content-header">
                  <!-- 파일명 인라인 편집 -->
                  <h4 v-if="!wcRenaming" @dblclick="startWcRename">{{ selectedWc }}</h4>
                  <input v-else v-model="wcNewName" class="wc-rename-input" @blur="finishWcRename" @keydown.enter="finishWcRename" ref="wcRenameRef" />
                  <span class="wc-syntax">문법: <code>{{'__' + selectedWc + '__'}}</code></span>
                </div>
                <!-- 블록 편집 -->
                <div class="wc-blocks">
                  <div v-for="(line, li) in wcEditLines" :key="li" class="wc-block-row">
                    <span class="wc-block-idx">{{ li + 1 }}</span>
                    <input v-model="wcEditLines[li]" class="wc-block-input" @keydown.enter="addWcLine(li)" />
                    <button class="wc-block-rm" @click="wcEditLines.splice(li, 1)">✕</button>
                  </div>
                  <button class="wc-add-line" @click="wcEditLines.push('')">+ 줄 추가</button>
                </div>
                <!-- 하단: 삽입 + 저장 -->
                <div class="wc-bottom-bar">
                  <CustomSelect v-model="wcInsertTarget" :options="['main', 'prefix', 'suffix', 'clipboard']" placeholder="삽입 위치" class="wc-insert-sel" />
                  <button class="wc-use-btn" @click="useWcSyntax">USE</button>
                  <div class="wc-spacer"></div>
                  <button class="wc-save-btn" @click="saveCurrentWildcard">💾 SAVE</button>
                </div>
              </template>
              <div v-else class="wc-empty">좌측에서 와일드카드를 선택하거나 NEW를 클릭하세요</div>
            </div>
          </div>
        </div>
      </div>
    </transition>

    <!-- Generation Stats Modal -->
    <transition name="fade">
      <div v-if="showStatsModal" class="stats-overlay" @click.self="showStatsModal = false">
        <div class="stats-modal">
          <div class="stats-header">
            <h3>GENERATION STATISTICS</h3>
            <button class="close-btn" @click="showStatsModal = false">X</button>
          </div>
          <div class="stats-body" v-if="genStats.total > 0">
            <div class="stats-cards">
              <div class="stat-card">
                <div class="stat-val">{{ genStats.total }}</div>
                <div class="stat-label">Total</div>
              </div>
              <div class="stat-card accent">
                <div class="stat-val">{{ genStats.success_rate }}%</div>
                <div class="stat-label">Success</div>
              </div>
              <div class="stat-card">
                <div class="stat-val">{{ genStats.avg_time }}s</div>
                <div class="stat-label">Avg Time</div>
              </div>
              <div class="stat-card">
                <div class="stat-val">{{ formatTime(genStats.total_time) }}</div>
                <div class="stat-label">Total Time</div>
              </div>
            </div>

            <div class="stats-section" v-if="genStats.daily && genStats.daily.length">
              <h4>DAILY GENERATIONS (30 days)</h4>
              <div class="daily-chart">
                <div v-for="d in genStats.daily" :key="d.date" class="daily-bar-wrap"
                  :title="d.date + ': ' + d.count + '장'">
                  <div class="daily-bar" :style="{ height: genStats.daily_max ? (d.count / genStats.daily_max * 100) + '%' : '0%' }"></div>
                </div>
              </div>
              <div class="daily-labels">
                <span>30일 전</span><span>오늘</span>
              </div>
            </div>

            <div class="stats-two-col">
              <div class="stats-section" v-if="genStats.top_models && genStats.top_models.length">
                <h4>TOP MODELS</h4>
                <div v-for="m in genStats.top_models" :key="m.name" class="stats-bar-row">
                  <span class="bar-name">{{ m.name }}</span>
                  <div class="bar-track"><div class="bar-fill" :style="{ width: (m.count / genStats.total * 100) + '%' }"></div></div>
                  <span class="bar-count">{{ m.count }}</span>
                </div>
              </div>
              <div class="stats-section" v-if="genStats.top_resolutions && genStats.top_resolutions.length">
                <h4>TOP RESOLUTIONS</h4>
                <div v-for="r in genStats.top_resolutions" :key="r.res" class="stats-bar-row">
                  <span class="bar-name">{{ r.res }}</span>
                  <div class="bar-track"><div class="bar-fill" :style="{ width: (r.count / genStats.total * 100) + '%' }"></div></div>
                  <span class="bar-count">{{ r.count }}</span>
                </div>
              </div>
            </div>

            <div class="stats-section" v-if="genStats.recent && genStats.recent.length">
              <h4>RECENT GENERATIONS</h4>
              <div class="recent-table">
                <div v-for="r in genStats.recent" :key="r.timestamp" class="recent-row">
                  <span class="r-time">{{ r.timestamp?.slice(5, 16).replace('T', ' ') }}</span>
                  <span class="r-status" :class="r.success ? 'ok' : 'fail'">{{ r.success ? 'OK' : 'FAIL' }}</span>
                  <span class="r-dur">{{ r.duration_sec }}s</span>
                  <span class="r-res">{{ r.width }}x{{ r.height }}</span>
                  <span class="r-model">{{ (r.model || '').split('/').pop()?.slice(0, 20) }}</span>
                </div>
              </div>
            </div>
          </div>
          <div class="stats-body stats-empty" v-else>
            <div class="stats-empty-msg">아직 생성 기록이 없습니다</div>
          </div>
        </div>
      </div>
    </transition>

    <!-- Global Toast Notifications -->
    <transition-group name="toast" tag="div" class="toast-container">
      <div v-for="t in toasts" :key="t.id" class="toast" :class="t.type" @click="removeToast(t.id)">
        <span class="toast-icon">{{ t.type === 'error' ? '⚠' : t.type === 'success' ? '✓' : 'ℹ' }}</span>
        {{ t.msg }}
      </div>
    </transition-group>
  </div>
</template>

<script setup>
import { ref, reactive, computed, watch, onMounted, nextTick } from 'vue'
import { initBridge, onBackendEvent, getBackend } from './bridge.js'
import { requestAction, useWidgetStore } from './stores/widgetStore.js'

const wStore = useWidgetStore()
const storeWidgets = wStore.widgets
const samplerItems = computed(() => wStore.getProperty('sampler_combo', 'items') || [])
const schedulerItems = computed(() => wStore.getProperty('scheduler_combo', 'items') || [])
const upscalerItems = computed(() => wStore.getProperty('upscaler_combo', 'items') || [])
const adModelItems = ref([])

// Hires/ADetailer 체크박스 (proxy 연동)
const randomResEnabled = computed({ get: () => storeWidgets.random_res_check === 'true', set: v => { storeWidgets.random_res_check = v ? 'true' : 'false'; if (v) loadRandomResList() } })
const autoResEnabled = computed({ get: () => storeWidgets.auto_res_check === 'true', set: v => { storeWidgets.auto_res_check = v ? 'true' : 'false' } })

// 랜덤 해상도 관리
const randomResList = ref([])
const newResW = ref(832)
const newResH = ref(1216)
async function loadRandomResList() {
  const bk = await getBackend()
  if (bk.getRandomResolutions) {
    bk.getRandomResolutions((json) => {
      try { randomResList.value = JSON.parse(json) } catch {}
    })
  }
}
function addRandomRes() {
  const w = Math.round((newResW.value || 832) / 8) * 8
  const h = Math.round((newResH.value || 1216) / 8) * 8
  if (w < 256 || h < 256) return
  randomResList.value.push([w, h, `${w}x${h}`])
  requestAction('set_random_resolutions', { list: randomResList.value })
}
function removeRandomRes(i) {
  randomResList.value.splice(i, 1)
  requestAction('set_random_resolutions', { list: randomResList.value })
}
const negpipEnabled = computed({ get: () => storeWidgets.negpip_group === 'true', set: v => { storeWidgets.negpip_group = v ? 'true' : 'false' } })

const hires_enabled = computed({ get: () => storeWidgets.hires_options_group === 'true', set: v => { storeWidgets.hires_options_group = v ? 'true' : 'false' } })
// Rating 필터
const ratingFilters = reactive([
  { key: 'g', label: 'G', on: true },
  { key: 's', label: 'S', on: true },
  { key: 'q', label: 'Q', on: false },
  { key: 'e', label: 'E', on: false },
])
// localStorage에서 복원
try {
  const saved = JSON.parse(window.localStorage.getItem('ratingFilter') || '[]')
  if (saved.length === 4) saved.forEach((v, i) => { ratingFilters[i].on = v })
} catch {}
function saveUiPrefs(payload) {
  requestAction('save_ui_prefs', payload)
}
function saveRatingFilter() {
  window.localStorage.setItem('ratingFilter', JSON.stringify(ratingFilters.map(r => r.on)))
  saveUiPrefs({ ratingFilter: ratingFilters.map(r => r.on) })
  // Python에 전달
  requestAction('set_rating_filter', { ratings: ratingFilters.filter(r => r.on).map(r => r.key) })
}

const removeArtist = computed({ get: () => storeWidgets.chk_remove_artist === 'true', set: v => { storeWidgets.chk_remove_artist = v ? 'true' : 'false' } })
const removeCopyright = computed({ get: () => storeWidgets.chk_remove_copyright === 'true', set: v => { storeWidgets.chk_remove_copyright = v ? 'true' : 'false' } })
const removeCharacter = computed({ get: () => storeWidgets.chk_remove_character === 'true', set: v => { storeWidgets.chk_remove_character = v ? 'true' : 'false' } })
const removeCharacterFeatures = computed({ get: () => storeWidgets.chk_remove_character_features === 'true', set: v => { storeWidgets.chk_remove_character_features = v ? 'true' : 'false' } })
const removeMeta = computed({ get: () => storeWidgets.chk_remove_meta === 'true', set: v => { storeWidgets.chk_remove_meta = v ? 'true' : 'false' } })
const removeCensorship = computed({ get: () => storeWidgets.chk_remove_censorship === 'true', set: v => { storeWidgets.chk_remove_censorship = v ? 'true' : 'false' } })
const removeText = computed({ get: () => storeWidgets.chk_remove_text === 'true', set: v => { storeWidgets.chk_remove_text = v ? 'true' : 'false' } })
const autoCharFeatures = computed({ get: () => storeWidgets.chk_auto_char_features === 'true', set: v => { storeWidgets.chk_auto_char_features = v ? 'true' : 'false' } })
const ad_enabled = computed({ get: () => storeWidgets.adetailer_group === 'true', set: v => { storeWidgets.adetailer_group = v ? 'true' : 'false' } })
const ad_s1_enabled = computed({ get: () => storeWidgets.ad_slot1_group === 'true', set: v => { storeWidgets.ad_slot1_group = v ? 'true' : 'false' } })
const ad_s2_enabled = computed({ get: () => storeWidgets.ad_slot2_group === 'true', set: v => { storeWidgets.ad_slot2_group = v ? 'true' : 'false' } })
import PromptPanel from './components/PromptPanel.vue'
import CustomSelect from './components/CustomSelect.vue'
import TabBar from './components/TabBar.vue'
import QueuePanel from './components/QueuePanel.vue'

const currentImage = ref('')
const resolution = ref('')
const seed = ref('')
const status = ref('')
const isGenerating = ref(false)
const progressVal = ref(0)
const showLeftPanel = ref(true)
const showExtendPanel = ref(false)
const historyImages = ref([])
const histPage = ref(0)
const histPerPage = 5

// EXIF
const activeExifTab = ref('positive')
const exifTabs = [
  { id: 'positive', label: 'Positive' },
  { id: 'negative', label: 'Negative' },
  { id: 'params', label: 'Parameters' },
]
const currentExif = ref({ prompt: '', negative: '', raw: '' })
const exifContent = computed(() => {
  if (activeExifTab.value === 'positive') return currentExif.value.prompt || 'No EXIF data'
  if (activeExifTab.value === 'negative') return currentExif.value.negative || ''
  return currentExif.value.raw || ''
})

const autoMode = ref(false)
const isAutomating = ref(false)
const autoGenCount = ref(0)
const autoWaiting = ref(false)
const vramInfo = ref({ used: 0, total: 0, pct: 0 })
const vramClass = computed(() => vramInfo.value.pct > 90 ? 'critical' : vramInfo.value.pct > 70 ? 'warn' : 'ok')
const autoSettings = reactive({ mode: 'count', limit: 10, repeat: 1, delay: 1.0, allowDupes: false })

function syncAutomationSettings() {
  const limit = Number(autoSettings.limit)
  const repeat = Number(autoSettings.repeat)
  const delay = Number(autoSettings.delay)

  action('set_automation_settings', {
    mode: autoSettings.mode,
    limit: Number.isFinite(limit) && limit > 0 ? limit : 10,
    repeat: Number.isFinite(repeat) && repeat > 0 ? repeat : 1,
    delay: Number.isFinite(delay) && delay >= 0 ? delay : 1,
    allowDupes: !!autoSettings.allowDupes,
  })
}

watch(autoSettings, () => {
  syncAutomationSettings()
}, { deep: true })

// Toast 알림 시스템
const toasts = ref([])
let toastId = 0
function addToast(type, msg) {
  const id = toastId++
  toasts.value.push({ id, type, msg })
  setTimeout(() => removeToast(id), 5000)
}
function removeToast(id) { toasts.value = toasts.value.filter(t => t.id !== id) }

// Extended panel state (NegPiP/조건부만 로컬)
const extWidgets = reactive({
  negpip_enabled: false,
  cond_prevent_dupe: true,
  cond_pos_on: false, cond_neg_on: false,
  cond_pos_rules: '', cond_neg_rules: '',
})
const loraStack = reactive([])

// LoRA localStorage 복원
try {
  const saved = JSON.parse(window.localStorage.getItem('loraStack') || '[]')
  if (Array.isArray(saved)) saved.forEach(l => loraStack.push(l))
} catch {}

let _loraInitialized = false
function _saveLoraStack() {
  try { window.localStorage.setItem('loraStack', JSON.stringify(loraStack)) } catch {}
  // 초기화 완료 전 빈 배열로 덮어쓰기 방지, 이후에는 빈 배열도 정상 저장
  if (!_loraInitialized && loraStack.length === 0) return
  _loraInitialized = true
  saveUiPrefs({ loraStack: loraStack.map(l => ({ ...l })) })
}

function syncLoraStack() {
  requestAction('set_lora_stack', {
    entries: loraStack.map(l => ({
      name: l.name || '',
      weight: Number.isFinite(Number(l.weight)) ? Number(l.weight) / 100 : 0.8,
      enabled: l.enabled !== false,
      triggerWords: Array.isArray(l.triggerWords) ? l.triggerWords : [],
    })),
  })
}

// LoRA 추가 함수 (Python에서 호출)
function addLoraToStack(name, weight, triggerWords = []) {
  const existing = loraStack.find(l => l.name === name)
  if (existing) {
    existing.weight = Math.round(weight * 100); existing.enabled = true
    if (triggerWords.length) existing.triggerWords = triggerWords
  }
  else loraStack.push({ name, weight: Math.round(weight * 100), enabled: true, triggerWords })
  _saveLoraStack()
}

// LoRA 변경 감시 → 자동 저장 (복원 중에는 무시)
let _loraRestoring = false
watch(loraStack, () => {
  if (_loraRestoring) return
  _saveLoraStack()
  syncLoraStack()
}, { deep: true })
function insertTriggerWord(tw) {
  const cur = storeWidgets.main_prompt_text || ''
  if (!cur.toLowerCase().includes(tw.toLowerCase())) {
    storeWidgets.main_prompt_text = cur ? cur.replace(/,?\s*$/, '') + ', ' + tw + ', ' : tw + ', '
    addToast('info', `트리거 워드 삽입: ${tw}`)
  } else {
    addToast('info', `이미 포함된 태그: ${tw}`)
  }
}

// History pagination
const visibleHistory = computed(() => {
  const start = histPage.value * histPerPage
  return historyImages.value.slice(start, start + histPerPage)
})

// Context menu (화면 밖 방지)
const ctxMenu = ref({ show: false, x: 0, y: 0, path: '' })
const ctxMenuStyle = computed(() => {
  const menuW = 210, menuH = 250
  let x = ctxMenu.value.x, y = ctxMenu.value.y
  if (x + menuW > window.innerWidth) x = window.innerWidth - menuW - 10
  if (y + menuH > window.innerHeight) y = window.innerHeight - menuH - 10
  if (x < 0) x = 10
  if (y < 0) y = 10
  return { top: y + 'px', left: x + 'px' }
})

const wildcards = ref([])
const showPresetManager = ref(false)
const presetList = ref([])
const selectedPreset = ref('')
const presetPreview = ref(null)

async function loadPresetList() {
  const bk = await getBackend()
  if (bk.getPresetList) bk.getPresetList((json) => { try { presetList.value = JSON.parse(json) } catch {} })
}
async function loadPresetPreview(name) {
  const bk = await getBackend()
  if (bk.getPresetData) bk.getPresetData(name, (json) => { try { presetPreview.value = JSON.parse(json) } catch {} })
}
function loadSelectedPreset() {
  if (!selectedPreset.value) return
  action('load_preset_by_name', { name: selectedPreset.value })
  showPresetManager.value = false
}
function deleteSelectedPreset() {
  if (!selectedPreset.value || !confirm(`"${selectedPreset.value}" 삭제?`)) return
  action('delete_preset', { name: selectedPreset.value })
  presetList.value = presetList.value.filter(p => p !== selectedPreset.value)
  selectedPreset.value = ''; presetPreview.value = null
}
function saveNewPreset() {
  const name = prompt('프리셋 이름:')
  if (!name) return
  action('save_preset_by_name', { name })
  presetList.value.push(name)
  addToast('success', `프리셋 "${name}" 저장됨`)
}

const showWeightManager = ref(false)
const globalWeights = reactive([])  // [{tag, weight}]

function saveGlobalWeights() {
  const valid = globalWeights.filter(w => w.tag.trim())
  action('save_global_weights', { weights: valid })
  showWeightManager.value = false
}

// 프롬프트에 글로벌 가중치 적용
function applyGlobalWeights(text) {
  if (!globalWeights.length) return text
  let result = text
  for (const w of globalWeights) {
    if (!w.tag.trim()) continue
    const tag = w.tag.trim()
    const weight = (w.weight / 100).toFixed(2)
    if (weight === '1.00') continue
    // 이미 가중치가 있으면 교체, 없으면 추가
    const regex = new RegExp(`\\(${tag.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\$&')}:[\\d.]+\\)`, 'gi')
    if (regex.test(result)) {
      result = result.replace(regex, `(${tag}:${weight})`)
    } else {
      const plain = new RegExp(`(?<![:(])\\b${tag.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\$&')}\\b(?![:\\)])`, 'gi')
      result = result.replace(plain, `(${tag}:${weight})`)
    }
  }
  return result
}
// Generation Stats
const showStatsModal = ref(false)
const genStats = reactive({ total: 0, success: 0, fail: 0, success_rate: 0, avg_time: 0, total_time: 0, daily: [], daily_max: 0, top_models: [], top_resolutions: [], recent: [] })
async function loadGenStats() {
  const bk = await getBackend()
  if (bk.getGenStats) {
    bk.getGenStats((json) => {
      try { Object.assign(genStats, JSON.parse(json)) } catch {}
    })
  }
}
function formatTime(sec) {
  if (!sec) return '0s'
  if (sec < 60) return sec + 's'
  if (sec < 3600) return Math.floor(sec / 60) + 'm ' + Math.round(sec % 60) + 's'
  return Math.floor(sec / 3600) + 'h ' + Math.floor((sec % 3600) / 60) + 'm'
}

const showWcManager = ref(false)
const selectedWc = ref('')
const selectedWcData = computed(() => wildcards.value.find(w => w.name === selectedWc.value) || null)
const wcEditLines = ref([])
const wcInsertTarget = ref('main')

function selectWildcard(name) {
  selectedWc.value = name
  const wc = wildcards.value.find(w => w.name === name)
  wcEditLines.value = wc ? [...wc.tags] : []
}

async function saveCurrentWildcard() {
  if (!selectedWc.value) return
  const bk = await getBackend()
  const content = wcEditLines.value.join('\n')
  bk.saveWildcard(selectedWc.value + '.txt', content, (json) => {
    try { const r = JSON.parse(json); if (r.ok) addToast('success', '와일드카드 저장됨') } catch {}
  })
  // 로컬 업데이트
  const wc = wildcards.value.find(w => w.name === selectedWc.value)
  if (wc) wc.tags = wcEditLines.value.filter(l => l.trim())
}

async function createNewWildcard() {
  const name = prompt('새 와일드카드 이름:')
  if (!name) return
  const bk = await getBackend()
  bk.saveWildcard(name + '.txt', '', () => {
    wildcards.value.push({ name, file: name + '.txt', tags: [] })
    selectedWc.value = name; wcEditLines.value = []
  })
}

async function deleteWildcard(name) {
  if (!confirm(`"${name}" 와일드카드를 삭제할까요?`)) return
  const bk = await getBackend()
  bk.deleteWildcard(name, () => {
    wildcards.value = wildcards.value.filter(w => w.name !== name)
    if (selectedWc.value === name) { selectedWc.value = ''; wcEditLines.value = [] }
  })
}

async function renameWildcard(oldName) {
  const newName = prompt('새 이름:', oldName)
  if (!newName || newName === oldName) return
  const bk = await getBackend()
  bk.renameWildcard(oldName, newName, () => {
    const wc = wildcards.value.find(w => w.name === oldName)
    if (wc) { wc.name = newName; wc.file = newName + '.txt' }
    if (selectedWc.value === oldName) selectedWc.value = newName
  })
}

const wcRenaming = ref(false)
const wcNewName = ref('')
const wcRenameRef = ref(null)

function startWcRename() {
  wcRenaming.value = true
  wcNewName.value = selectedWc.value
  nextTick(() => { if (wcRenameRef.value) wcRenameRef.value.focus() })
}

async function finishWcRename() {
  wcRenaming.value = false
  const newName = wcNewName.value.trim()
  if (!newName || newName === selectedWc.value) return
  const bk = await getBackend()
  bk.renameWildcard(selectedWc.value, newName, () => {
    const wc = wildcards.value.find(w => w.name === selectedWc.value)
    if (wc) { wc.name = newName; wc.file = newName + '.txt' }
    selectedWc.value = newName
    addToast('success', '이름 변경됨')
  })
}

// USE = 와일드카드 사용 문법 삽입 (__name__)
function useWcSyntax() {
  if (!selectedWc.value) return
  const syntax = `__${selectedWc.value}__`
  if (wcInsertTarget.value === 'clipboard') {
    navigator.clipboard?.writeText(syntax)
    addToast('info', '문법 복사됨: ' + syntax)
  } else {
    const targetMap = { main: 'main_prompt_text', prefix: 'prefix_prompt_text', suffix: 'suffix_prompt_text' }
    const key = targetMap[wcInsertTarget.value] || 'main_prompt_text'
    const cur = storeWidgets[key] || ''
    storeWidgets[key] = cur ? cur.replace(/,?\s*$/, '') + ', ' + syntax + ', ' : syntax + ', '
    addToast('success', syntax + ' 삽입됨')
  }
}

function addWcLine(afterIdx) { wcEditLines.value.splice(afterIdx + 1, 0, '') }

function openWildcardByName(name) {
  showWcManager.value = true
  selectWildcard(name)
}

function action(name, payload = {}) { requestAction(name, payload) }

function insertWildcardTag(tag) {
  const cur = storeWidgets.main_prompt_text || ''
  storeWidgets.main_prompt_text = cur ? cur.replace(/,?\s*$/, '') + ', ' + tag + ', ' : tag + ', '
}

function doGenerate() {
  // 자동화 중이면 중지
  if (isAutomating.value) {
    action('stop_automation')
    isAutomating.value = false
    autoWaiting.value = false
    return
  }
  // 글로벌 가중치 적용
  if (globalWeights.length > 0) {
    const cur = storeWidgets.main_prompt_text || ''
    const weighted = applyGlobalWeights(cur)
    if (weighted !== cur) storeWidgets.main_prompt_text = weighted
  }
  // LoRA Stack
  const activeLoras = loraStack.filter(l => l.enabled)
  if (activeLoras.length > 0) {
    const loraText = activeLoras.map(l => `<lora:${l.name}:${(l.weight/100).toFixed(2)}>`).join(', ')
    requestAction('set_lora_text', { lora_text: loraText })
  }
  syncAutomationSettings()
  action('generate')
}

function showHistoryMenu(e, path) { ctxMenu.value = { show: true, x: e.clientX, y: e.clientY, path } }
function hideCtxMenu() { ctxMenu.value.show = false }
const ctxAddFavorite = () => { action('add_favorite', { path: ctxMenu.value.path }); hideCtxMenu() }
const ctxSendI2I = () => { action('send_to_i2i', { path: ctxMenu.value.path }); hideCtxMenu() }
const ctxSendInpaint = () => { action('send_to_inpaint', { path: ctxMenu.value.path }); hideCtxMenu() }
const ctxSendEditor = () => { action('send_to_editor', { path: ctxMenu.value.path }); hideCtxMenu() }
const ctxCompare = (slot) => { action('send_to_compare', { path: ctxMenu.value.path, slot }); hideCtxMenu() }
const ctxRunAdetailer = () => { action('run_adetailer_single', { path: ctxMenu.value.path, settings: { ad_model: 'face_yolov8n.pt', ad_confidence: 0.3, ad_denoise: 0.4 } }); hideCtxMenu() }
const ctxCopyPath = () => { navigator.clipboard?.writeText(ctxMenu.value.path); hideCtxMenu() }
const ctxDelete = () => {
  const path = ctxMenu.value.path
  action('delete_image', { path })
  // 히스토리에서 즉시 제거
  const idx = historyImages.value.indexOf(path)
  if (idx >= 0) historyImages.value.splice(idx, 1)
  // 현재 보고 있는 이미지였으면 다음 이미지로
  if (currentImage.value === path) {
    currentImage.value = historyImages.value[0] || ''
  }
  hideCtxMenu()
}

async function selectHistoryImage(path) {
  currentImage.value = path
  const img = new Image()
  img.onload = () => { resolution.value = `${img.naturalWidth} × ${img.naturalHeight}` }
  img.src = 'file:///' + path
  // EXIF 로드
  const backend = await getBackend()
  if (backend.getImageExif) {
    backend.getImageExif(path, (json) => {
      try {
        const d = JSON.parse(json)
        currentExif.value = { prompt: d.prompt || '', negative: d.negative || '', raw: d.raw || '', params: d.params || null, params_line: d.params_line || '' }
      } catch {}
    })
  }
}

// 드래그 앤 드롭 지원
function onDragStart(e, path) {
  e.dataTransfer.setData('text/plain', path)
  e.dataTransfer.effectAllowed = 'copy'
}

function onTabChanged(tabName) {
  showLeftPanel.value = ['t2i', 'i2i', 'inpaint'].includes(tabName)
  showExtendPanel.value = false
  hideCtxMenu()
}

async function loadHistory() {
  const backend = await getBackend()
  if (backend.getGalleryImages) {
    backend.getGalleryImages('', (json) => {
      try { historyImages.value = JSON.parse(json).slice(0, 100) } catch {}
    })
  }
}

import { useRouter } from 'vue-router'
const router = useRouter()

onMounted(async () => {
  await initBridge()
  // History는 앱 시작 시 비어있음 — 생성된 이미지만 추가됨
  document.addEventListener('click', hideCtxMenu)
  document.addEventListener('wheel', (e) => { if (e.ctrlKey) e.preventDefault() }, { passive: false })
  // 브라우저 기본 우클릭 메뉴 전역 차단
  document.addEventListener('contextmenu', (e) => { e.preventDefault() })
  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'g') { e.preventDefault(); doGenerate() }
    if (e.ctrlKey && e.key === 's') { e.preventDefault(); action('save_settings') }
    if (e.key === 'F5') { e.preventDefault(); loadHistory() }
  })

  // 초기 rating 필터 전달
  requestAction('set_rating_filter', { ratings: ratingFilters.filter(r => r.on).map(r => r.key) })

  onBackendEvent('tabChanged', (tabId) => {
    const targetPath = tabId === 't2i' ? '/' : `/${tabId}`
    router.push(targetPath)
    onTabChanged(tabId)
  })

  onBackendEvent('imageGenerated', async (data) => {
    const parsed = JSON.parse(data)
    currentImage.value = parsed.path
    resolution.value = `${parsed.width} × ${parsed.height}`
    seed.value = String(parsed.seed)
    isGenerating.value = false
    status.value = ''
    if (parsed.path) {
      historyImages.value.unshift(parsed.path)
      if (historyImages.value.length > 100) historyImages.value.pop()
      histPage.value = 0
      // 생성 직후 EXIF 자동 로드
      const bk = await getBackend()
      if (bk.getImageExif) {
        bk.getImageExif(parsed.path, (json) => {
          try {
            const d = JSON.parse(json)
            currentExif.value = { prompt: d.prompt || '', negative: d.negative || '', raw: d.raw || '', params: d.params || null, params_line: d.params_line || '' }
          } catch {}
        })
      }
    }
  })
  onBackendEvent('generationStarted', () => { isGenerating.value = true; autoWaiting.value = false; progressVal.value = 0 })
  onBackendEvent('automationStatus', (json) => {
    try {
      const d = JSON.parse(json)
      isAutomating.value = d.running || false
      autoGenCount.value = d.count || 0
      autoWaiting.value = d.waiting || false
      if (!d.running) { isGenerating.value = false }
    } catch {}
  })
  onBackendEvent('generationProgress', (step, total) => {
    progressVal.value = Math.round(step / total * 100)
    status.value = `Generating... ${step}/${total}`
  })
  onBackendEvent('generationError', (msg) => { isGenerating.value = false; status.value = `Error: ${msg}` })

  // 글로벌 가중치 로드
  onBackendEvent('globalWeightsLoaded', (json) => {
    try { const d = JSON.parse(json); globalWeights.splice(0); d.forEach(w => globalWeights.push(w)) } catch {}
  })

  // 랜덤 해상도 로드
  loadRandomResList()

  // ADetailer 모델 로드
  const _bk = await getBackend()
  syncAutomationSettings()
  if (_bk.getADetailerModels) _bk.getADetailerModels((json) => { try { adModelItems.value = JSON.parse(json) } catch {} })

  // 와일드카드 로드
  if (_bk.getWildcardTree) _bk.getWildcardTree((json) => { try { wildcards.value = JSON.parse(json) } catch {} })

  // VRAM 실시간 업데이트
  onBackendEvent('vramUpdated', (json) => { try { vramInfo.value = JSON.parse(json) } catch {} })

  // Global Toast 알림 (Python → Vue)
  onBackendEvent('showNotification', (type, msg) => { addToast(type, msg) })

  onBackendEvent('uiPrefsLoaded', (json) => {
    try {
      const prefs = JSON.parse(json)
      if (Array.isArray(prefs.ratingFilter) && prefs.ratingFilter.length === ratingFilters.length) {
        prefs.ratingFilter.forEach((v, i) => { ratingFilters[i].on = !!v })
        window.localStorage.setItem('ratingFilter', JSON.stringify(prefs.ratingFilter))
        requestAction('set_rating_filter', { ratings: ratingFilters.filter(r => r.on).map(r => r.key) })
      }
      if (Array.isArray(prefs.loraStack)) {
        _loraRestoring = true
        loraStack.splice(0, loraStack.length, ...prefs.loraStack.map(l => ({ ...l })))
        window.localStorage.setItem('loraStack', JSON.stringify(prefs.loraStack))
        nextTick(() => { _loraRestoring = false; _loraInitialized = true })
      }
      // tabOrder 복원 (Settings 탭 미방문 시에도 적용)
      if (Array.isArray(prefs.tabOrder) && prefs.tabOrder.length > 0) {
        window.localStorage.setItem('tabOrder', JSON.stringify(prefs.tabOrder))
      }
    } catch {}
  })

  // 에러도 Toast로 표시
  onBackendEvent('generationError', (msg) => { addToast('error', msg) })

  // LoRA 추가 이벤트 (Python lora_manager → Vue)
  onBackendEvent('loraInserted', (json) => {
    try {
      const d = JSON.parse(json)
      addLoraToStack(d.name, d.weight || 0.8, d.trigger_words || [])
      showExtendPanel.value = true
    } catch {}
  })

  onBackendEvent('loraStackLoaded', (json) => {
    try {
      const entries = JSON.parse(json)
      if (!Array.isArray(entries)) return
      _loraRestoring = true
      loraStack.splice(0, loraStack.length, ...entries.map(entry => ({
        name: entry.name || '',
        weight: Number.isFinite(Number(entry.weight)) ? Math.round(Number(entry.weight) * 100) : 80,
        enabled: entry.enabled !== false,
        triggerWords: Array.isArray(entry.triggerWords) ? entry.triggerWords : [],
      })))
      // localStorage만 업데이트, ui_prefs 재저장 안 함 (루프 방지)
      try { window.localStorage.setItem('loraStack', JSON.stringify(loraStack)) } catch {}
      nextTick(() => { _loraRestoring = false; _loraInitialized = true })
    } catch {}
  })

  syncLoraStack()
})
</script>

<style scoped>
.app-container { width: 100%; height: 100vh; display: flex; flex-direction: column; background: var(--bg-primary); }
.app-header { height: 60px; display: flex; align-items: center; justify-content: center; background: var(--bg-primary); border-bottom: 1px solid var(--border); z-index: 100; }
.main-workspace { flex: 1; display: flex; overflow: hidden; position: relative; }

.side-panel { width: 360px; display: flex; flex-direction: column; background: var(--bg-secondary); border-right: 1px solid var(--border); z-index: 10; }
.side-panel.right { width: 220px; border-right: none; border-left: 1px solid var(--border); }
.panel-scroll { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 16px; }

/* 반달 화살표 */
.half-moon {
  position: absolute; left: 360px; top: 50%; transform: translateY(-50%);
  width: 20px; height: 60px; background: var(--bg-secondary);
  border: 1px solid var(--border); border-left: none;
  border-radius: 0 30px 30px 0; cursor: pointer; z-index: 10;
  display: flex; align-items: center; justify-content: center;
  color: var(--text-muted); font-size: 10px; transition: all 0.2s;
}
.half-moon:hover { background: var(--bg-card); color: var(--accent); width: 24px; }
.half-moon.open { background: var(--accent-dim); color: var(--accent); left: 680px; }

/* Extended Panel Overlay */
.extend-overlay {
  position: absolute; left: 360px; top: 0; bottom: 0; width: 320px;
  background: var(--bg-secondary); border-right: 1px solid var(--border);
  z-index: 50; display: flex; flex-direction: column;
  box-shadow: 8px 0 32px rgba(0,0,0,0.5);
}
.extend-header { display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; border-bottom: 1px solid var(--border); }
.extend-header h3 { font-size: 11px; letter-spacing: 2px; color: var(--text-muted); }
.close-btn { background: none; border: none; color: var(--text-muted); font-size: 16px; cursor: pointer; }
.close-btn:hover { color: #f87171; }
.extend-scroll { flex: 1; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 12px; }

.ext-card { background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: var(--radius-card); padding: 12px; }
.ext-title { font-size: 10px; font-weight: 800; color: var(--text-muted); letter-spacing: 1px; margin-bottom: 10px; cursor: pointer; }
.ext-field { margin-bottom: 8px; }
.ext-field label { font-size: 9px; color: var(--text-muted); font-weight: 700; display: block; margin-bottom: 3px; }
.ext-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.rating-toggle-row { display: flex; gap: 4px; margin-bottom: 8px; }
.rating-toggle {
  flex: 1; padding: 4px; background: var(--bg-button); border: 1px solid var(--border);
  border-radius: 4px; color: var(--text-muted); font-size: 10px; font-weight: 900;
  cursor: pointer; text-align: center; transition: var(--transition);
}
.rating-toggle.active { border-color: var(--accent); color: var(--accent); background: var(--accent-dim); }
.ext-toggle-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; }
.ext-sub { margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border); }
.ext-sub summary { font-size: 10px; color: var(--text-secondary); cursor: pointer; }
.ext-sub-title { font-size: 10px; font-weight: 800; color: var(--accent); letter-spacing: 1px; margin: 8px 0 4px; }

/* LoRA block */
.lora-block { display: flex; align-items: center; gap: 6px; padding: 6px 8px; background: var(--bg-button); border-radius: 6px; margin-bottom: 4px; }
.lora-info-col { flex: 1; min-width: 0; }
.lora-name { font-size: 11px; color: var(--text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lora-triggers { display: flex; flex-wrap: wrap; gap: 3px; margin-top: 3px; }
.trigger-chip {
  padding: 1px 6px; font-size: 9px; font-weight: 700; color: var(--accent);
  background: var(--accent-dim); border: 1px solid rgba(250, 204, 21, 0.15);
  border-radius: 4px; cursor: pointer; transition: var(--transition);
}
.trigger-chip:hover { background: rgba(250, 204, 21, 0.2); border-color: var(--accent); }
.lora-slider { width: 60px; accent-color: var(--accent); }
.lora-weight { font-size: 10px; color: var(--accent); min-width: 30px; text-align: right; font-family: monospace; }
.lora-remove { background: none; border: none; color: #f87171; cursor: pointer; font-size: 12px; }
.ext-add-btn { width: 100%; padding: 8px; background: var(--bg-button); border: 1px dashed var(--border); border-radius: 6px; color: var(--text-secondary); font-size: 10px; font-weight: 700; cursor: pointer; margin-top: 4px; }
.ext-res-row { display: flex; align-items: center; gap: 6px; }
.ext-res-row input { text-align: center; flex: 1; }
.ext-res-row span { color: var(--text-muted); }
.ext-mini-btn { width: 32px; height: 32px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary); cursor: pointer; flex-shrink: 0; }
.ext-res-opts { display: flex; gap: 8px; margin-top: 4px; }
.ext-check-sm { display: flex; align-items: center; gap: 3px; font-size: 9px; color: var(--text-muted); cursor: pointer; white-space: nowrap; }
.ext-check-sm input { width: 12px; height: 12px; margin: 0; }

/* 랜덤 해상도 편집기 */
.rand-res-editor { margin-top: 6px; border: 1px solid var(--border); border-radius: 6px; padding: 6px; background: rgba(0,0,0,0.15); }
.rand-res-list { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 6px; }
.rand-res-item { display: flex; align-items: center; gap: 4px; padding: 2px 6px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 4px; font-size: 10px; }
.rand-res-val { color: var(--text-primary); font-weight: 700; font-family: monospace; }
.rand-res-desc { color: var(--text-muted); font-size: 9px; }
.rand-res-del { background: none; border: none; color: #f87171; cursor: pointer; font-size: 10px; padding: 0 2px; }
.rand-res-empty { font-size: 9px; color: var(--text-muted); padding: 4px; }
.rand-res-add { display: flex; align-items: center; gap: 4px; }
.rand-res-add span { color: var(--text-muted); font-size: 10px; }
.rand-res-input { width: 50px; padding: 3px 4px; font-size: 10px; text-align: center; }
.rand-res-btn { width: 24px; height: 24px; background: var(--accent); border: none; border-radius: 4px; color: #000; font-weight: 900; cursor: pointer; font-size: 14px; }
.ext-check-row { display: flex; align-items: center; gap: 4px; font-size: 10px; color: var(--text-secondary); cursor: pointer; margin-bottom: 4px; white-space: nowrap; }
.ext-check-row input[type="checkbox"] { flex-shrink: 0; margin: 0; }
.ext-check-row span { overflow: hidden; text-overflow: ellipsis; }
.ext-check-row input[type="checkbox"] { accent-color: var(--accent); }
.ext-hint { font-size: 10px; color: var(--text-muted); margin-top: 4px; }
.cond-textarea { min-height: 60px; font-size: 11px; line-height: 1.6; font-family: 'Consolas', monospace; }
.cond-block { margin-top: 6px; }
.cond-toggle-row { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
.cond-toggle { padding: 2px 10px; border: 1px solid var(--border); border-radius: 4px; background: var(--bg-button); color: var(--text-muted); font-size: 9px; font-weight: 800; cursor: pointer; }
.cond-toggle.on { background: rgba(74,222,128,0.15); border-color: #4ade80; color: #4ade80; }
.lora-check { flex-shrink: 0; }
.lora-check input { accent-color: var(--accent); }
.lora-empty { font-size: 11px; color: var(--text-muted); text-align: center; padding: 12px; }

.slide-enter-active, .slide-leave-active { transition: transform 0.25s ease, opacity 0.25s ease; }
.slide-enter-from, .slide-leave-to { transform: translateX(-20px); opacity: 0; }

/* Tool Card */
.tool-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-card); padding: 16px; }
.tool-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
.tool-btn { padding: 8px 4px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-base); color: var(--text-secondary); font-size: 10px; font-weight: 700; cursor: pointer; transition: var(--transition); }
.tool-btn:hover { border-color: var(--text-muted); color: var(--text-primary); }
.tool-btn.highlight { color: var(--accent); border-color: var(--accent-dim); }

/* Preset Manager */
.pm-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 2000; display: flex; align-items: center; justify-content: center; }
.pm-modal { width: 650px; height: 450px; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px; display: flex; flex-direction: column; overflow: hidden; }
.pm-header { display: flex; align-items: center; gap: 12px; padding: 12px 16px; border-bottom: 1px solid var(--border); }
.pm-header h3 { font-size: 12px; letter-spacing: 2px; color: var(--text-muted); flex: 1; }
.pm-body { flex: 1; display: flex; overflow: hidden; }
.pm-list { width: 180px; overflow-y: auto; border-right: 1px solid var(--border); padding: 8px; }
.pm-item { padding: 7px 10px; font-size: 11px; color: var(--text-secondary); cursor: pointer; border-radius: 4px; margin-bottom: 2px; }
.pm-item:hover { background: var(--bg-input); }
.pm-item.active { background: var(--accent-dim); color: var(--accent); }
.pm-preview { flex: 1; overflow-y: auto; padding: 12px; }
.pm-detail { display: flex; flex-direction: column; gap: 4px; }
.pm-field { display: flex; gap: 8px; font-size: 10px; border-bottom: 1px solid rgba(255,255,255,0.03); padding: 3px 0; }
.pm-key { color: var(--accent); font-weight: 700; min-width: 100px; }
.pm-val { color: var(--text-secondary); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pm-empty { display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-muted); }
.pm-footer { display: flex; align-items: center; gap: 6px; padding: 10px 16px; border-top: 1px solid var(--border); }
.pm-btn { padding: 6px 14px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 6px; color: var(--text-secondary); font-size: 10px; font-weight: 700; cursor: pointer; }
.pm-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
.pm-btn:disabled { opacity: 0.3; }
.pm-btn.accent { background: var(--accent); color: #000; border: none; }
.pm-spacer { flex: 1; }

/* Weight Manager */
.wm-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 2000; display: flex; align-items: center; justify-content: center; }
.wm-modal { width: 500px; max-height: 500px; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px; display: flex; flex-direction: column; overflow: hidden; }
.wm-header { padding: 12px 16px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.wm-header h3 { font-size: 12px; letter-spacing: 2px; color: var(--text-muted); }
.wm-desc { font-size: 9px; color: var(--text-muted); flex: 1; }
.wm-body { flex: 1; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 6px; }
.wm-row { display: flex; align-items: center; gap: 6px; }
.wm-tag-input { flex: 1; padding: 5px 8px; font-size: 11px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary); }
.wm-slider { width: 100px; accent-color: var(--accent); }
.wm-val { font-size: 11px; color: var(--accent); min-width: 35px; text-align: right; font-family: monospace; }
.wm-rm { background: none; border: none; color: #f87171; cursor: pointer; font-size: 14px; }
.wm-add { width: 100%; padding: 6px; background: var(--bg-button); border: 1px dashed var(--border); border-radius: 4px; color: var(--text-muted); font-size: 10px; cursor: pointer; }
.wm-footer { padding: 10px 16px; border-top: 1px solid var(--border); display: flex; justify-content: flex-end; }
.wm-save { padding: 7px 20px; background: var(--accent); border: none; border-radius: 6px; color: #000; font-size: 11px; font-weight: 800; cursor: pointer; }

/* Generation Stats Modal */
.stats-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 2000; display: flex; align-items: center; justify-content: center; }
.stats-modal { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 16px; width: 640px; max-height: 85vh; overflow-y: auto; }
.stats-header { display: flex; align-items: center; justify-content: space-between; padding: 20px 24px; border-bottom: 1px solid var(--border); }
.stats-header h3 { font-size: 13px; font-weight: 900; letter-spacing: 2px; color: var(--text-primary); }
.stats-body { padding: 24px; display: flex; flex-direction: column; gap: 24px; }
.stats-empty { display: flex; align-items: center; justify-content: center; min-height: 200px; }
.stats-empty-msg { color: var(--text-muted); font-size: 14px; }

.stats-cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.stat-card { background: var(--bg-input); border: 1px solid var(--border); border-radius: 12px; padding: 16px; text-align: center; }
.stat-card.accent { border-color: var(--accent-dim); }
.stat-val { font-size: 28px; font-weight: 900; color: var(--text-primary); font-family: monospace; }
.stat-card.accent .stat-val { color: var(--accent); }
.stat-label { font-size: 9px; font-weight: 800; color: var(--text-muted); letter-spacing: 1.5px; margin-top: 4px; }

.stats-section h4 { font-size: 10px; font-weight: 900; color: var(--text-muted); letter-spacing: 1.5px; margin-bottom: 12px; }
.stats-two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }

.daily-chart { display: flex; align-items: flex-end; gap: 2px; height: 80px; padding: 0 2px; }
.daily-bar-wrap { flex: 1; height: 100%; display: flex; align-items: flex-end; }
.daily-bar { width: 100%; background: var(--accent); border-radius: 2px 2px 0 0; min-height: 1px; transition: height 0.3s; }
.daily-labels { display: flex; justify-content: space-between; margin-top: 4px; font-size: 9px; color: var(--text-muted); }

.stats-bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.bar-name { font-size: 11px; color: var(--text-secondary); min-width: 80px; max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar-track { flex: 1; height: 6px; background: var(--bg-button); border-radius: 3px; overflow: hidden; }
.bar-fill { height: 100%; background: var(--accent); border-radius: 3px; transition: width 0.3s; }
.bar-count { font-size: 10px; color: var(--text-muted); min-width: 30px; text-align: right; font-family: monospace; }

.recent-table { display: flex; flex-direction: column; gap: 4px; }
.recent-row { display: flex; align-items: center; gap: 8px; padding: 6px 10px; background: var(--bg-input); border-radius: 6px; font-size: 11px; }
.r-time { color: var(--text-muted); min-width: 80px; font-family: monospace; }
.r-status { font-weight: 800; font-size: 10px; min-width: 30px; }
.r-status.ok { color: #4ade80; }
.r-status.fail { color: #f87171; }
.r-dur { color: var(--accent); min-width: 40px; font-family: monospace; }
.r-res { color: var(--text-secondary); min-width: 70px; }
.r-model { color: var(--text-muted); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* Wildcard Manager Modal */
.wc-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 2000; display: flex; align-items: center; justify-content: center; }
.wc-modal { width: 700px; height: 500px; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px; display: flex; flex-direction: column; overflow: hidden; }
.wc-modal-header { display: flex; align-items: center; gap: 12px; padding: 12px 16px; border-bottom: 1px solid var(--border); }
.wc-modal-header h3 { font-size: 12px; letter-spacing: 2px; color: var(--text-muted); }
.wc-path { font-size: 10px; color: var(--text-muted); font-family: monospace; flex: 1; }
.wc-modal-body { flex: 1; display: flex; overflow: hidden; }
.wc-sidebar { width: 180px; overflow-y: auto; border-right: 1px solid var(--border); padding: 8px; }
.wc-file-item { padding: 6px 10px; font-size: 11px; color: var(--text-secondary); cursor: pointer; border-radius: 4px; display: flex; justify-content: space-between; }
.wc-file-item:hover { background: var(--bg-input); }
.wc-file-item.active { background: var(--accent-dim); color: var(--accent); }
.wc-file-count { font-size: 9px; color: var(--text-muted); }
.wc-content { flex: 1; overflow-y: auto; padding: 16px; }
.wc-content-header { margin-bottom: 12px; }
.wc-content-header h4 { font-size: 14px; color: var(--text-primary); }
.wc-content-header span { font-size: 10px; color: var(--text-muted); }
.wc-content-header code { background: var(--bg-input); padding: 2px 6px; border-radius: 3px; font-size: 10px; color: var(--accent); }
.wc-new-btn { padding: 4px 12px; background: var(--accent); border: none; border-radius: 4px; color: #000; font-size: 10px; font-weight: 800; cursor: pointer; }
.wc-fname { flex: 1; overflow: hidden; text-overflow: ellipsis; cursor: pointer; }
.wc-del { background: none; border: none; color: #f87171; cursor: pointer; font-size: 11px; opacity: 0; transition: 0.15s; }
.wc-file-item:hover .wc-del { opacity: 1; }
.wc-syntax { font-size: 10px; color: var(--text-muted); }
.wc-syntax code { background: var(--bg-input); padding: 1px 6px; border-radius: 3px; color: var(--accent); }
.wc-copy-btn { padding: 2px 8px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 3px; color: var(--text-secondary); font-size: 9px; cursor: pointer; }
.wc-blocks { display: flex; flex-direction: column; gap: 3px; max-height: 300px; overflow-y: auto; }
.wc-block-row { display: flex; align-items: center; gap: 4px; }
.wc-block-idx { font-size: 9px; color: var(--text-muted); min-width: 20px; text-align: right; font-family: monospace; }
.wc-block-input { flex: 1; padding: 4px 8px; font-size: 11px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 3px; color: var(--text-primary); }
.wc-block-use { padding: 2px 6px; background: var(--accent-dim); border: 1px solid var(--accent); border-radius: 3px; color: var(--accent); font-size: 9px; font-weight: 700; cursor: pointer; }
.wc-block-rm { background: none; border: none; color: #f87171; cursor: pointer; font-size: 12px; }
.wc-add-line { width: 100%; padding: 4px; background: var(--bg-button); border: 1px dashed var(--border); border-radius: 3px; color: var(--text-muted); font-size: 9px; cursor: pointer; margin-top: 4px; }
.wc-rename-input { font-size: 14px; background: var(--bg-input); border: 1px solid var(--accent); border-radius: 4px; color: var(--text-primary); padding: 2px 8px; width: 200px; }
.wc-bottom-bar { display: flex; align-items: center; gap: 6px; margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border); }
.wc-insert-sel { padding: 4px 8px; font-size: 10px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 4px; color: var(--text-secondary); }
.wc-use-btn { padding: 5px 16px; background: var(--accent); border: none; border-radius: 4px; color: #000; font-size: 10px; font-weight: 800; cursor: pointer; }
.wc-spacer { flex: 1; }
.wc-save-btn { padding: 5px 14px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 4px; color: var(--text-secondary); font-size: 10px; font-weight: 700; cursor: pointer; }
.wc-save-btn:hover { border-color: var(--accent); color: var(--accent); }
.wc-empty { display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-muted); font-size: 13px; }
.fade-enter-active, .fade-leave-active { transition: opacity 0.2s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }

.gen-footer { padding: 12px 16px; background: var(--bg-card); border-top: 1px solid var(--border); display: flex; flex-direction: column; gap: 8px; }
.gen-actions { display: flex; gap: 6px; }
.action-btn { flex: 1; padding: 7px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-base); color: var(--text-secondary); font-size: 10px; font-weight: 700; cursor: pointer; transition: var(--transition); }
.action-btn.active { border-color: #4ade80; color: #4ade80; background: rgba(74,222,128,0.05); }
.action-btn.highlight { border-color: var(--accent-dim); color: var(--accent); }
.action-btn:hover { border-color: var(--text-muted); }
.auto-settings { display: flex; flex-direction: column; gap: 4px; padding: 8px; background: rgba(74,222,128,0.03); border: 1px solid rgba(74,222,128,0.1); border-radius: 8px; }
.auto-row { display: flex; align-items: center; gap: 4px; }
.auto-row label { font-size: 9px; color: var(--text-muted); font-weight: 700; min-width: 32px; }
.auto-input { width: 50px; padding: 4px 6px; font-size: 11px; text-align: center; }
.auto-select { min-width: 90px; padding: 4px; font-size: 10px; }

/* 자동화 상태 */
.auto-status { padding: 8px 12px; background: rgba(250, 204, 21, 0.05); border: 1px solid var(--accent-dim); border-radius: 8px; margin-bottom: 8px; }
.auto-status-bar { display: flex; align-items: center; gap: 8px; font-size: 11px; color: var(--accent); font-weight: 700; }
.auto-status-sub { font-size: 10px; color: var(--text-muted); margin-top: 4px; }
.auto-pulse { width: 8px; height: 8px; background: var(--accent); border-radius: 50%; animation: pulse 1.5s ease-in-out infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.generate-row { display: flex; gap: 6px; }
.auto-check { display: flex; align-items: center; gap: 4px; font-size: 10px; color: var(--text-secondary); cursor: pointer; }
.auto-check input { accent-color: #4ade80; }
.btn-generate { width: 100%; height: 50px; background: var(--accent); border: none; border-radius: var(--radius-pill); color: #000; font-weight: 800; font-size: 14px; letter-spacing: 1px; cursor: pointer; transition: var(--transition); }
.btn-generate:hover:not(:disabled) { background: var(--accent-hover); transform: translateY(-2px); box-shadow: 0 8px 24px rgba(250, 204, 21, 0.3); }
.btn-generate:disabled { opacity: 0.5; cursor: wait; }
.btn-generate.automating { background: #f87171; color: #fff; }
.btn-generate.automating:hover:not(:disabled) { background: #ef4444; box-shadow: 0 8px 24px rgba(248, 113, 113, 0.3); }

/* Viewport */
.viewport-area { flex: 1; display: flex; flex-direction: column; overflow: hidden; background: #050505; }
.viewport-main { flex: 1; position: relative; overflow: hidden; }

/* EXIF Bar */
.exif-bar { flex-shrink: 0; background: #0D0D0D; border-top: 1px solid var(--border); }
.exif-tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); }
.exif-tab { flex: 1; padding: 6px; background: transparent; border: none; color: var(--text-muted); font-size: 10px; font-weight: 700; cursor: pointer; text-align: center; border-bottom: 2px solid transparent; }
.exif-tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.exif-content { padding: 6px 12px; font-size: 11px; color: var(--text-secondary); max-height: 80px; overflow-y: auto; line-height: 1.5; font-family: 'Consolas', monospace; white-space: pre-wrap; word-break: break-all; }
.exif-params { padding: 4px 12px; max-height: 80px; overflow-y: auto; }
.exif-params .param-line { display: flex; align-items: baseline; gap: 6px; padding: 2px 0; font-size: 10px; color: var(--text-secondary); font-family: 'Consolas', monospace; }
.exif-params .pl { font-size: 8px; font-weight: 900; color: var(--accent); letter-spacing: 1px; min-width: 40px; flex-shrink: 0; }
.exif-params .param-line.other { color: var(--text-muted); }

/* History */
.hist-header { padding: 16px; display: flex; justify-content: space-between; align-items: center; }
.hist-header h3 { font-size: 12px; letter-spacing: 2px; color: var(--text-muted); }
.count-badge { background: var(--border); padding: 2px 8px; border-radius: 10px; font-size: 10px; color: var(--text-secondary); }
.hist-nav-btn { width: 100%; padding: 4px; background: #131313; border: none; color: #484848; font-size: 12px; cursor: pointer; flex-shrink: 0; }
.hist-nav-btn:hover { background: #1A1A1A; color: #E8E8E8; }
.hist-nav-btn:disabled { opacity: 0.3; cursor: default; }
.hist-scroll { flex: 1; overflow-y: auto; padding: 8px; display: flex; flex-direction: column; gap: 8px; }
.hist-card { position: relative; border-radius: var(--radius-card); overflow: hidden; border: 2px solid transparent; cursor: pointer; transition: border-color 0.15s; }
.hist-card:hover { border-color: #333; }
.hist-card.selected { border-color: var(--accent); box-shadow: 0 0 12px var(--accent-dim); }
.hist-card img { width: 100%; aspect-ratio: 1; object-fit: cover; display: block; }

/* Context Menu */
.modern-ctx-menu { position: fixed; background: #181818; border: 1px solid #222; border-radius: 10px; padding: 6px; z-index: 1000; min-width: 200px; box-shadow: 0 12px 32px rgba(0,0,0,0.8); }
.ctx-item { padding: 10px 14px; font-size: 11px; font-weight: 600; color: #909090; cursor: pointer; border-radius: 6px; transition: var(--transition); }
.ctx-item:hover { background: #252525; color: #FFF; }
.ctx-item.delete { color: #f87171; }
.ctx-item.delete:hover { background: rgba(248, 113, 113, 0.1); }
.ctx-separator { height: 1px; background: #222; margin: 4px 0; }

.pop-enter-active { animation: pop 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275); }
@keyframes pop { from { opacity: 0; transform: scale(0.9); } to { opacity: 1; transform: scale(1); } }

/* VRAM Bar */
.vram-bar {
  position: fixed; bottom: 0; left: 0; right: 0; height: 22px;
  background: #080808; border-top: 1px solid var(--border); z-index: 500;
  display: flex; align-items: center;
}
.vram-fill { height: 100%; transition: width 1s ease; }
.vram-fill.ok { background: rgba(74,222,128,0.4); }
.vram-fill.warn { background: rgba(251,191,36,0.5); }
.vram-fill.critical { background: rgba(248,113,113,0.6); }
.vram-text {
  position: absolute; left: 50%; transform: translateX(-50%);
  font-size: 11px; font-weight: 800; color: #B0B0B0; letter-spacing: 0.5px;
  text-shadow: 0 1px 3px rgba(0,0,0,0.8);
}

.global-progress { position: fixed; top: 0; left: 0; width: 100%; height: 3px; background: transparent; z-index: 1000; }
.progress-fill { height: 100%; background: var(--accent); transition: width 0.3s ease; }

/* Toast Notifications */
.toast-container { position: fixed; top: 70px; right: 20px; z-index: 2000; display: flex; flex-direction: column; gap: 8px; pointer-events: none; }
.toast {
  padding: 10px 20px; border-radius: 8px; font-size: 12px; font-weight: 600;
  color: #FFF; pointer-events: auto; cursor: pointer; min-width: 200px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.5); backdrop-filter: blur(8px);
}
.toast.success { background: rgba(74, 222, 128, 0.9); color: #000; }
.toast.error { background: rgba(248, 113, 113, 0.9); color: #FFF; }
.toast.info { background: rgba(96, 165, 250, 0.9); color: #FFF; }
.toast-icon { margin-right: 8px; }
.toast-enter-active { animation: slideIn 0.3s ease; }
.toast-leave-active { animation: slideOut 0.3s ease; }
@keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
@keyframes slideOut { from { opacity: 1; } to { transform: translateX(100%); opacity: 0; } }
</style>
