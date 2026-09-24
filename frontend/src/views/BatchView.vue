<template>
  <div class="batch-view">

    <!-- Batch 탭 — 듀얼 패널 (좌측 설정 / 우측 썸네일 그리드) -->
    <div v-if="subTab === 'batch'" class="tab-body ad-layout">
      <div class="ad-settings">
        <h3>일괄 처리</h3>
        <div class="file-drop compact" @dragover.prevent @drop.prevent="onDropBatch">
          <div class="drop-hint">
            이미지 드래그 또는
            <button class="link-btn" v-host-dialog="'open_batch_files'" @click="action('open_batch_files')">파일 선택</button>
          </div>
        </div>
        <div class="file-count" v-if="batchFiles.length">{{ batchFiles.length }}개 파일</div>
        <label class="s-label">작업</label>
        <CustomSelect v-model="batchOp" :options="['resize', 'format']" placeholder="작업 선택" />
        <div v-if="batchOp === 'resize'" class="op-settings">
          <div class="row"><input class="s-input" v-model="resizeW" placeholder="W" /><span>×</span><input class="s-input" v-model="resizeH" placeholder="H" /></div>
        </div>
        <div v-if="batchOp === 'format'" class="op-settings">
          <CustomSelect v-model="formatType" :options="['PNG', 'JPEG', 'WEBP']" placeholder="포맷" />
        </div>
        <div class="op-hint">결과는 출력 폴더의 batch 폴더에 새 파일로 저장됩니다 — 원본과 생성 정보는 그대로 둡니다.</div>
        <div class="ad-progress" v-if="batchJob && (batchJob.running || batchJob.done)">
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: jobPercent(batchJob) + '%' }"></div>
          </div>
          <span>{{ jobProgressText(batchJob) }}</span>
        </div>
        <div class="op-hint" v-if="batchJob && !batchJob.running && batchJob.output_dir" :title="batchJob.output_dir">
          저장 위치: {{ batchJob.output_dir }}
        </div>
        <button class="btn-start" @click="startBatch" :disabled="batchFiles.length === 0 || batchRunning">
          {{ batchRunning ? '배치 처리 중…' : `배치 시작 (${batchFiles.length}파일)` }}
        </button>
      </div>
      <!-- 우측: 썸네일 그리드 -->
      <div class="ad-compare">
        <div v-if="batchFiles.length === 0" class="grid-empty">
          <div class="grid-empty-ico"><Icon name="package" /></div>
          <div class="grid-empty-title">이미지를 드래그하여 추가</div>
          <div class="grid-empty-sub">여러 장을 한꺼번에 처리할 수 있습니다</div>
        </div>
        <div v-else class="thumb-grid">
          <div v-for="(f, i) in batchFiles" :key="f" class="thumb-card">
            <img :src="mediaUrl(f)" :alt="basename(f)" loading="lazy" />
            <div class="thumb-name" :title="f">{{ basename(f) }}</div>
            <button class="thumb-rm" @click="batchFiles.splice(i, 1)" title="제거">×</button>
          </div>
        </div>
      </div>
    </div>

    <!-- Upscale 탭 — 듀얼 패널 -->
    <div v-if="subTab === 'upscale'" class="tab-body ad-layout">
      <div class="ad-settings">
        <h3>업스케일</h3>
        <div class="file-drop compact" @dragover.prevent @drop.prevent="onDropUpscale">
          <div class="drop-hint">
            이미지 드래그 또는
            <button class="link-btn" v-host-dialog="'open_upscale_files'" @click="action('open_upscale_files')">파일 선택</button>
          </div>
        </div>
        <div class="file-count" v-if="upscaleFiles.length">{{ upscaleFiles.length }}개 파일</div>
        <label class="s-label">업스케일러</label>
        <CustomSelect v-model="upscaler" :options="upscalers" placeholder="업스케일러 선택..." />
        <label class="s-label">배율</label>
        <div class="slider-row">
          <input type="range" min="1" max="4" step="0.5" v-model.number="scaleFactor" />
          <span>{{ scaleFactor }}x</span>
        </div>
        <div class="op-hint">결과는 출력 폴더의 upscale 폴더에 새 파일로 저장됩니다.</div>
        <div class="ad-progress" v-if="upscaleJob && (upscaleJob.running || upscaleJob.done)">
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: jobPercent(upscaleJob) + '%' }"></div>
          </div>
          <span>{{ jobProgressText(upscaleJob) }}</span>
        </div>
        <div class="op-hint" v-if="upscaleJob && !upscaleJob.running && upscaleJob.output_dir" :title="upscaleJob.output_dir">
          저장 위치: {{ upscaleJob.output_dir }}
        </div>
        <button class="btn-start" @click="startUpscale" :disabled="upscaleFiles.length === 0 || upscaleRunning">
          {{ upscaleRunning ? '업스케일 중…' : `업스케일 시작 (${upscaleFiles.length}파일)` }}
        </button>
      </div>
      <!-- 우측: 썸네일 그리드 -->
      <div class="ad-compare">
        <div v-if="upscaleFiles.length === 0" class="grid-empty">
          <div class="grid-empty-ico"><Icon name="search" /></div>
          <div class="grid-empty-title">업스케일할 이미지를 드래그</div>
          <div class="grid-empty-sub">2~4배 해상도 향상</div>
        </div>
        <div v-else class="thumb-grid">
          <div v-for="(f, i) in upscaleFiles" :key="f" class="thumb-card">
            <img :src="mediaUrl(f)" :alt="basename(f)" loading="lazy" />
            <div class="thumb-name" :title="f">{{ basename(f) }}</div>
            <button class="thumb-rm" @click="upscaleFiles.splice(i, 1)" title="제거">×</button>
          </div>
        </div>
      </div>
    </div>

    <!-- ADetailer 탭 -->
    <div v-if="subTab === 'adetailer'" class="tab-body ad-layout">
      <!-- 좌측: 설정 -->
      <div class="ad-settings">
        <h3>ADetailer</h3>
        <div class="file-drop" @dragover.prevent @drop.prevent="onDropAd">
          <div v-if="adFiles.length === 0" class="drop-hint">
            이미지 드래그 또는
            <button class="link-btn" v-host-dialog="'open_ad_files'" @click="action('open_ad_files')">파일 선택</button>
            /
            <button class="link-btn" v-host-dialog="'open_ad_folder'" @click="action('open_ad_folder')">폴더 선택</button>
          </div>
          <div v-else class="file-list">
            <div v-for="(f, i) in adFiles" :key="f" class="file-item"
              :class="{ active: adCurrentIdx === i, done: adResults[i] }"
              @click="previewAdFile(i)">
              <span>{{ basename(f) }}</span>
              <span v-if="adResults[i]" class="done-badge"><Icon name="check" /></span>
              <button class="rm-btn" @click.stop="adFiles.splice(i, 1)">×</button>
            </div>
          </div>
        </div>
        <div class="file-count" v-if="adFiles.length">{{ adFiles.length }}개 파일</div>

        <label class="s-label">AD Model</label>
        <CustomSelect v-model="adModel" :options="adModelItems" placeholder="AD Model..." />
        <div class="ad-params">
          <div class="ad-param">
            <label>Confidence</label>
            <input type="number" v-model.number="adConfidence" step="0.05" min="0" max="1" />
          </div>
          <div class="ad-param">
            <label>Denoise</label>
            <input type="number" v-model.number="adDenoise" step="0.05" min="0" max="1" />
          </div>
          <div class="ad-param">
            <label>Prompt</label>
            <label class="ad-toggle"><input type="checkbox" v-model="adUseExifPrompt" /><span>EXIF 프롬프트 사용</span></label>
            <input v-if="!adUseExifPrompt" type="text" v-model="adPrompt" placeholder="(선택사항)" />
            <div v-else class="exif-prompt-hint">각 이미지의 EXIF에서 Positive/Negative를 자동으로 읽어 사용합니다</div>
          </div>
        </div>

        <!-- 프로그레스 -->
        <div class="ad-progress" v-if="adProcessing">
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: adProgressPct + '%' }"></div>
          </div>
          <span>
            {{ adProgressCur }}/{{ adProgressTotal }}
            <span v-if="adEtaText" class="ad-eta">· ETA {{ adEtaText }}</span>
          </span>
        </div>

        <div class="ad-actions">
          <button class="btn-start" @click="runAdSingle" :disabled="adFiles.length === 0 || adProcessing">
            현재 이미지 적용
          </button>
          <button class="btn-start batch" @click="runAdBatch" :disabled="adFiles.length === 0 || adProcessing">
            전체 배치 ({{ adFiles.length }}장)
          </button>
          <button class="btn-stop" v-if="adProcessing" @click="action('stop_adetailer_batch')">
            중지
          </button>
        </div>
      </div>

      <!-- 우측: Before/After 비교 -->
      <div class="ad-compare">
        <div v-if="adBefore && adAfter" class="compare-split">
          <div class="compare-col">
            <div class="compare-label">이전</div>
            <img :src="mediaUrl(adBefore)" />
          </div>
          <div class="compare-col">
            <div class="compare-label">이후</div>
            <img :src="mediaUrl(adAfter)" />
          </div>
        </div>
        <div v-else-if="adPreview" class="preview-single">
          <img :src="mediaUrl(adPreview)" />
        </div>
        <div v-else class="compare-empty">
          좌측에서 이미지를 선택하면 미리보기가 표시됩니다
        </div>
      </div>
    </div>

    <!-- SAM3 탭 -->
    <div v-if="subTab === 'sam3'" class="tab-body ad-layout">
      <div class="ad-settings">
        <h3>SAM3</h3>
        <div class="file-drop" @dragover.prevent @drop.prevent="onDropSam3">
          <div v-if="sam3Files.length === 0" class="drop-hint">
            이미지 드래그 또는
            <button class="link-btn" v-host-dialog="'open_ad_files'" @click="action('open_ad_files')">파일 선택</button>
            /
            <button class="link-btn" v-host-dialog="'open_ad_folder'" @click="action('open_ad_folder')">폴더 선택</button>
          </div>
          <div v-else class="file-list">
            <div v-for="(f, i) in sam3Files" :key="f" class="file-item"
              :class="{ active: sam3CurrentIdx === i, done: sam3Results[i] }"
              @click="previewSam3File(i)">
              <span>{{ basename(f) }}</span>
              <span v-if="sam3Results[i]" class="done-badge"><Icon name="check" /></span>
              <button class="rm-btn" @click.stop="sam3Files.splice(i, 1)">×</button>
            </div>
          </div>
        </div>
        <div class="file-count" v-if="sam3Files.length">{{ sam3Files.length }}개 파일</div>

        <div class="ad-params">
          <div class="ad-param">
            <label>검출 프롬프트</label>
            <input type="text" v-model="sam3Prompt" placeholder="face" />
          </div>
          <div class="ad-param">
            <label>인페인트 프롬프트</label>
            <label class="ad-toggle"><input type="checkbox" v-model="sam3UseExifPrompt" /><span>EXIF 프롬프트 사용</span></label>
            <input v-if="!sam3UseExifPrompt" type="text" v-model="sam3InpaintPrompt" placeholder="비워두면 메인 프롬프트 유지" />
            <div v-else class="exif-prompt-hint">각 이미지의 EXIF에서 Positive/Negative를 자동으로 읽어 사용합니다</div>
          </div>
          <div class="ad-param">
            <label>Exclude Prompt (보호할 영역)</label>
            <input type="text" v-model="sam3ExcludePrompt" placeholder="face, eyes, hand" />
          </div>
          <div class="ad-param">
            <label>Negative Prompt</label>
            <input type="text" v-model="sam3NegativePrompt" placeholder="(선택사항)" />
          </div>
          <div class="ad-param">
            <label>Mode</label>
            <CustomSelect v-model="sam3Mode" :options="['Inpaint', 'Mask only']" placeholder="Inpaint" />
          </div>
          <div class="ad-param">
            <label>Mask Mode</label>
            <CustomSelect v-model="sam3MaskMode" :options="['Individual', 'Combined']" placeholder="Mask Mode" />
          </div>
          <div class="ad-param">
            <label>Threshold</label>
            <input type="number" v-model.number="sam3Threshold" step="0.01" min="0" max="1" />
          </div>
          <div class="ad-param">
            <label>Denoise</label>
            <input type="number" v-model.number="sam3Denoise" step="0.01" min="0" max="1" />
          </div>
          <div class="ad-param">
            <label>Mask Blur</label>
            <input type="number" v-model.number="sam3MaskBlur" min="0" />
          </div>
          <div class="ad-param">
            <label>Padding</label>
            <input type="number" v-model.number="sam3Padding" min="0" />
          </div>
          <div class="ad-param">
            <label>Mask Dilation (px)</label>
            <input type="number" v-model.number="sam3MaskDilation" min="0" />
          </div>
          <div class="ad-param">
            <label>Outline expand (px)</label>
            <input type="number" v-model.number="sam3MaskOutline" min="0" />
          </div>
          <div class="ad-param">
            <label>Inpainting fill</label>
            <CustomSelect v-model="sam3Fill"
              :options="['fill', 'original', 'latent noise', 'latent nothing']" placeholder="original" />
          </div>
          <div class="ad-param">
            <label>스텝</label>
            <input type="number" v-model.number="sam3Steps" min="1" />
          </div>
          <div class="ad-param">
            <label>CFG</label>
            <input type="number" v-model.number="sam3Cfg" step="0.5" min="0" />
          </div>
          <div class="ad-param">
            <label>Seed (-1 = 랜덤)</label>
            <input type="number" v-model.number="sam3Seed" />
          </div>
          <div class="ad-param">
            <label>Checkpoint</label>
            <input type="text" v-model="sam3Checkpoint" placeholder="sam3.pt" />
          </div>
          <div class="ad-param">
            <label>Device</label>
            <CustomSelect v-model="sam3Device" :options="['cuda', 'auto', 'cpu']" placeholder="cuda" />
          </div>
          <label class="ad-toggle"><input type="checkbox" v-model="sam3MaskHull" /><span>Convex Hull (머리카락 감싸기)</span></label>
          <label class="ad-toggle"><input type="checkbox" v-model="sam3OnlyMasked" /><span>마스크된 영역만</span></label>
          <label class="ad-toggle"><input type="checkbox" v-model="sam3RestoreFace" /><span>Restore face</span></label>
          <label class="ad-toggle"><input type="checkbox" v-model="sam3PreviewOverlay" /><span>오버레이 미리보기</span></label>
          <label class="ad-toggle"><input type="checkbox" v-model="sam3SaveArtifacts" /><span>Artifacts 저장</span></label>
          <label class="ad-toggle" title="검출 직후 SAM3(~3.5GB) VRAM 회수 — 16GB GPU 권장">
            <input type="checkbox" v-model="sam3UnloadAfter" /><span>검출 후 SAM3 VRAM 해제</span></label>
          <!-- ControlNet 13필드 — T2I·Refine 과 같은 패널 (예전엔 배치 SAM3 에 CN 이 아예 없었다) -->
          <Sam3ControlNetPanel :widgets="sam3Cn" class="sam3-cn" />
        </div>

        <div class="ad-progress" v-if="sam3Processing">
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: sam3ProgressPct + '%' }"></div>
          </div>
          <span>{{ sam3ProgressCur }}/{{ sam3ProgressTotal }}</span>
        </div>

        <div class="ad-actions">
          <button class="btn-start" @click="runSam3Single" :disabled="sam3Files.length === 0 || sam3Processing">
            현재 이미지 적용
          </button>
          <button class="btn-start batch" @click="runSam3Batch" :disabled="sam3Files.length === 0 || sam3Processing">
            전체 배치 ({{ sam3Files.length }}장)
          </button>
        </div>
      </div>

      <div class="ad-compare">
        <div v-if="sam3Before && sam3After" class="compare-split">
          <div class="compare-col">
            <div class="compare-label">이전</div>
            <img :src="mediaUrl(sam3Before)" />
          </div>
          <div class="compare-col">
            <div class="compare-label">이후</div>
            <img :src="mediaUrl(sam3After)" />
          </div>
        </div>
        <div v-else-if="sam3Preview" class="preview-single">
          <img :src="mediaUrl(sam3Preview)" />
        </div>
        <div v-else class="compare-empty">
          좌측에서 이미지를 선택하면 미리보기가 표시됩니다
        </div>
      </div>
    </div>

    <!-- CAPTION 탭 — CAFormer 태그 + ToriiGate/Ollama 자연어 캡션 -->
    <div v-if="subTab === 'caption'" class="tab-body ad-layout">
      <div class="ad-settings caption-settings">
        <h3>이미지 캡션</h3>
        <fieldset class="cap-controls" :disabled="captionRunning">
        <label class="s-label" for="caption-engine">처리 방식</label>
        <select id="caption-engine" class="s-select" v-model="captionEngine" @change="onCaptionEngineChanged">
          <option value="caformer">CAFormer · Danbooru 태그</option>
          <option value="torii">ToriiGate · 자연어 캡션</option>
          <option value="combined">CAFormer + ToriiGate · 태그 + 자연어</option>
          <option value="ollama">기타 Ollama 비전 모델 · 자연어</option>
        </select>

        <div class="cap-runtime-card" :class="{ warning: captionRuntimeError }">
          <div class="cap-runtime-head">
            <span>{{ captionRuntimeSummary }}</span>
            <button class="cap-inline-btn" @click="loadCaptionRuntime" :disabled="captionRuntimeLoading">
              {{ captionRuntimeLoading ? '확인 중' : '다시 확인' }}
            </button>
          </div>
          <div v-if="needsCaformer" class="cap-runtime-path" :title="activeCaformerDir || '자동 탐색'">
            {{ activeCaformerDir || 'Hugging Face 캐시에서 자동 탐색' }}
          </div>
          <div v-if="captionRuntimeError" class="cap-runtime-error">{{ captionRuntimeError }}</div>
        </div>

        <template v-if="needsCaformer">
          <label class="s-label">CAFormer 모델 폴더</label>
          <div class="cap-outdir">
            <input class="s-input cap-path-input" v-model="captionCaformerDir" @change="saveCaptionPrefs"
              :placeholder="detectedCaformerDir || 'model.onnx 폴더 자동 탐색'" />
            <button class="cap-refresh" v-host-dialog="'caption_pick_caformer_dir'" @click="action('caption_pick_caformer_dir')" title="CAFormer 모델 폴더 선택"><Icon name="folder" /></button>
            <button v-if="captionCaformerDir" class="cap-refresh" @click="clearCaptionCaformerDir" title="자동 탐색 사용"><Icon name="rotate-ccw" /></button>
          </div>
          <div class="cap-opts cap-tag-opts">
            <label><input type="checkbox" v-model="captionIncludeCharacters" @change="saveCaptionPrefs" /> 캐릭터 태그</label>
            <label><input type="checkbox" v-model="captionIncludeRating" @change="saveCaptionPrefs" /> 등급 태그</label>
          </div>
          <label class="cap-best-toggle">
            <input type="checkbox" v-model="captionUseBestThresholds" @change="saveCaptionPrefs" />
            태그별 최적 임계값 사용
          </label>
          <div v-if="!captionUseBestThresholds" class="cap-threshold-grid">
            <label>일반 <input class="s-input" type="number" min="0.05" max="0.95" step="0.01" v-model.number="captionGeneralThreshold" @change="saveCaptionPrefs" /></label>
            <label>캐릭터 <input class="s-input" type="number" min="0.05" max="0.95" step="0.01" v-model.number="captionCharacterThreshold" @change="saveCaptionPrefs" /></label>
            <label v-if="captionIncludeRating">등급 <input class="s-input" type="number" min="0.05" max="0.95" step="0.01" v-model.number="captionRatingThreshold" @change="saveCaptionPrefs" /></label>
          </div>
        </template>

        <template v-if="needsNaturalCaption">
          <label class="s-label">{{ captionEngine === 'ollama' ? 'Ollama 비전 모델' : 'ToriiGate 모델 (Ollama)' }}</label>
          <div class="cap-model-row">
            <CustomSelect v-if="visibleCaptionModels.length" v-model="activeCaptionModel" :options="visibleCaptionModels"
              placeholder="모델 선택..." @update:modelValue="saveCaptionPrefs" />
            <input v-else class="s-input" v-model="activeCaptionModel" @change="saveCaptionPrefs" placeholder="모델 이름 입력..." />
            <button class="cap-refresh" @click="loadCaptionModels" title="Ollama 모델 목록 새로고침"><Icon name="refresh" /></button>
          </div>
          <label class="s-label">자연어 캡션 지시</label>
          <textarea class="s-textarea" v-model="captionPrompt" @change="saveCaptionPrefs" rows="4"></textarea>
          <div v-if="captionEngine === 'combined'" class="cap-combined-hint">
            CAFormer 태그를 ToriiGate에 시각 근거로 전달하고, 저장할 때 태그와 자연어 사이에 빈 줄을 넣습니다.
          </div>
        </template>

        <div class="cap-opts">
          <label><input type="checkbox" v-model="captionSave" @change="saveCaptionPrefs" /> .txt 저장</label>
          <label><input type="checkbox" v-model="captionOverwrite" @change="saveCaptionPrefs" /> 기존 덮어쓰기</label>
        </div>
        <label class="s-label">저장 위치</label>
        <div class="cap-outdir">
          <span class="cap-outdir-path" :title="captionOutDir || '이미지와 같은 폴더'">{{ captionOutDir || '이미지와 같은 폴더 (.txt 사이드카)' }}</span>
          <button class="cap-refresh" v-host-dialog="'caption_pick_outdir'" @click="action('caption_pick_outdir')" title="저장 폴더 선택"><Icon name="folder" /></button>
          <button v-if="captionOutDir" class="cap-refresh" @click="clearCaptionOutDir" title="기본값(이미지 옆)으로"><Icon name="rotate-ccw" /></button>
        </div>
        <div class="cap-pick">
          <button class="link-btn" v-host-dialog="'caption_pick_files'" @click="action('caption_pick_files')"><Icon name="file" /> 파일 선택</button>
          <button class="link-btn" v-host-dialog="'caption_pick_folder'" @click="action('caption_pick_folder')"><Icon name="folder" /> 폴더 선택</button>
        </div>
        </fieldset>
        <div class="file-count" v-if="captionItems.length">{{ captionItems.length }}개 이미지</div>
        <button class="btn-start" @click="captionAll" :disabled="!captionItems.length || captionRunning">
          {{ captionRunning ? `처리 중... ${captionCur}/${captionTotal}` : `전체 처리 (${captionItems.length})` }}
        </button>
        <button v-if="captionItems.length" class="link-btn cap-clear" @click="clearCaption" :disabled="captionRunning">목록 비우기</button>
      </div>
      <div class="ad-compare">
        <div v-if="!captionItems.length" class="grid-empty">
          <div class="grid-empty-ico"><Icon name="tag" /></div>
          <div class="grid-empty-title">파일 또는 폴더를 선택하세요</div>
          <div class="grid-empty-sub">{{ captionEmptyHint }}</div>
        </div>
        <div v-else class="cap-list">
          <div v-for="it in captionItems" :key="it.path" class="cap-item">
            <img :src="mediaUrl(it.path)" loading="lazy" class="cap-thumb" />
            <div class="cap-body">
              <div class="cap-name" :title="it.path">
                {{ basename(it.path) }}
                <span class="cap-status" :class="it.status">{{ statusLabel(it.status) }}</span>
              </div>
              <div class="cap-sidecar" :title="it.txtPath || resolvedCaptionSidecarPath(it.path)">
                TXT · {{ it.txtPath || resolvedCaptionSidecarPath(it.path) }}
              </div>
              <textarea class="cap-text" v-model="it.caption" placeholder="캡션 (편집 가능)..." rows="3" :disabled="captionRunning"></textarea>
              <div v-if="it.error" class="cap-item-error">{{ it.error }}</div>
              <div class="cap-actions">
                <button class="cap-btn" @click="captionSingle(it)" :disabled="captionRunning"><Icon name="tag" /> 처리</button>
                <button class="cap-btn" @click="saveCaptionItem(it)" :disabled="captionRunning"><Icon name="save" /> 저장</button>
                <button class="cap-btn t2i" @click="sendCaptionToT2I(it)" :disabled="captionRunning || !it.caption" title="이 캡션을 T2I 메인 프롬프트에 추가하고 이동">→ T2I</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onActivated, onUnmounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { getBackend, onBackendEvent } from '../bridge.js'
import { mediaUrl } from '../utils/media.js'
import {
  captionSidecarPath,
  hasUnresolvedCaptionItems,
  matchesCaptionIdentity,
} from '../utils/captionSession'
import { requestAction, useWidgetStore } from '../stores/widgetStore.js'
import { vHostDialog } from '../utils/hostDialogs'
import { useViewMode } from '../composables/useViewMode'
import { storedOllamaUrl } from '../utils/ollamaPrefs'
import { droppedImagePaths, newPaths } from '../utils/dropPaths'
import { jobPercent, jobProgressText, parseBatchJobState } from '../utils/batchJobState'
import { sam3CnDefaults, sam3CnSettings } from '../utils/sam3ControlNet'
import { createExifWarningNotice } from '../utils/exifWarningNotice'
import CustomSelect from '../components/CustomSelect.vue'
import Sam3ControlNetPanel from '../components/Sam3ControlNetPanel.vue'
import type {
  ActionName,
  ActionPayload,
  BatchJobStatePayload,
  CaptionDoneEvent,
  CaptionEngineMode,
  CaptionJobStatus,
  CaptionProgressEvent,
  CaptionRuntimeSnapshot,
  CaptionStartResponse,
} from '../types/bridge'

interface CaptionItem {
  path: string
  caption: string
  status: string
  error?: string
  txtPath?: string
  [k: string]: any
}

const router = useRouter()
const widgets = useWidgetStore()
/** 서브탭은 왼쪽 레일의 서랍이 정한다 — 여기선 읽기만 한다. `useViewMode` 참조. */
const { mode: subTab } = useViewMode('batch')
const action = <K extends ActionName>(name: K, payload?: ActionPayload<K>) => requestAction(name, payload)
const basename = (p: any) => typeof p === 'string' ? p.split('/').pop()!.split('\\').pop() : p.name || p

// ── Batch ──
const batchFiles = ref<string[]>([])
const batchOp = ref('resize')
const resizeW = ref('1024')
const resizeH = ref('1024')
const formatType = ref('PNG')

/**
 * 드롭한 이미지의 경로를 목록에 더한다. QtWebEngine 의 File 에는 경로가 없어서
 * (예전엔 `(f as any).path` 가 undefined 로 들어갔다) OS 에서 끌어 온 파일은 건너뛰고
 * 안내한다 — 앱 안 히스토리·갤러리 카드의 경로 드래그는 그대로 받는다.
 */
function addDroppedPaths(e: DragEvent, target: { value: string[] }) {
  const { paths, unresolved } = droppedImagePaths(e.dataTransfer)
  target.value.push(...newPaths(target.value, paths))
  if (unresolved) {
    requestAction('show_toast', {
      type: 'warning',
      msg: `OS 에서 끌어 온 파일은 경로를 알 수 없어 ${unresolved}개를 건너뛰었습니다 — '파일 선택'을 사용하세요`,
    })
  }
}

// 일괄 처리·업스케일 진행 (batchJobState) — 실행 중에는 시작 버튼을 막는다.
const batchJob = ref<BatchJobStatePayload | null>(null)
const upscaleJob = ref<BatchJobStatePayload | null>(null)
const batchRunning = computed(() => !!batchJob.value?.running)
const upscaleRunning = computed(() => !!upscaleJob.value?.running)

function onDropBatch(e: DragEvent) {
  addDroppedPaths(e, batchFiles)
}
function startBatch() {
  if (batchRunning.value) return
  action('start_batch', {
    files: batchFiles.value,
    operation: batchOp.value,
    settings: { width: resizeW.value, height: resizeH.value, format: formatType.value },
  })
}

// ── Upscale ──
const upscaleFiles = ref<string[]>([])
// 백엔드 목록이 오기 전 기본값 — Lanczos 는 Forge·ComfyUI 둘 다 모델 없이 된다.
const upscaler = ref('Lanczos')
const upscalers = ref<string[]>(['Lanczos', 'R-ESRGAN 4x+', 'R-ESRGAN 4x+ Anime6B'])
const scaleFactor = ref(2)

function onDropUpscale(e: DragEvent) {
  addDroppedPaths(e, upscaleFiles)
}
function startUpscale() {
  if (upscaleRunning.value) return
  action('start_upscale', {
    files: upscaleFiles.value,
    upscaler: upscaler.value,
    scale: scaleFactor.value,
  })
}

// ── Caption (CAFormer 태그 + ToriiGate/Ollama 자연어, .txt 사이드카) ──
const captionItems = ref<CaptionItem[]>([])   // [{path, caption, status}]
const captionEngine = ref<CaptionEngineMode>('combined')
const captionModel = ref(window.localStorage.getItem('ollamaCaptionModel')
  || window.localStorage.getItem('ollamaModel') || '')
const captionToriiModel = ref('hf.co/DraconicDragon/ToriiGate-0.5-GGUF:BF16')
const captionPrompt = ref(window.localStorage.getItem('captionPrompt')
  || 'Write a factual natural-language caption. Cover the main subject, appearance, clothing, pose, action, composition, and visible background without inventing details.')
const captionSave = ref(true)
const captionOverwrite = ref(false)
const captionOutDir = ref(window.localStorage.getItem('captionOutDir') || '')
const captionCaformerDir = ref('')
const captionIncludeCharacters = ref(true)
const captionIncludeRating = ref(false)
const captionUseBestThresholds = ref(true)
const captionGeneralThreshold = ref(0.35)
const captionCharacterThreshold = ref(0.43)
const captionRatingThreshold = ref(0.38)
const captionRuntime = ref<Partial<CaptionRuntimeSnapshot>>({})
const captionRuntimeLoading = ref(false)
let captionRuntimeRequestId = 0
const captionRunning = ref(false)
const captionCur = ref(0)
const captionTotal = ref(0)
const ollamaModels = ref<string[]>([])

const CAPTION_BACKEND_WAIT_MS = 8_000
const CAPTION_BRIDGE_CALL_MS = 10_000
const CAPTION_RUNTIME_WAIT_MS = 15_000
const CAPTION_RUNTIME_DEBOUNCE_MS = 300
const CAPTION_JOB_POLL_MS = 1_500
const CAPTION_JOB_LOST_MS = 45_000

function makeCaptionToken(scope: string) {
  try {
    if (typeof crypto?.randomUUID === 'function') return `${scope}-${crypto.randomUUID()}`
  } catch {}
  return `${scope}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
}

const captionClientToken = makeCaptionToken('caption-client')
let activeCaptionJobId = ''
let activeCaptionPaths = new Set<string>()
let captionRuntimeResponseTimer: ReturnType<typeof setTimeout> | null = null
let captionRuntimeDebounceTimer: ReturnType<typeof setTimeout> | null = null
let captionJobPollTimer: ReturnType<typeof setTimeout> | null = null
let captionJobPollInFlight = false
let captionJobLastContactAt = 0
let captionJobIdlePolls = 0
let captionSidecarGeneration = 0
let captionSidecarErrorGeneration = -1
let captionDisposed = false
const captionEventUnsubs: Array<() => void> = []

function captionToast(type: 'success' | 'error' | 'info' | 'warning', msg: string) {
  requestAction('show_toast', { type, msg })
}

function errorMessage(error: unknown, fallback: string) {
  if (error instanceof Error && error.message) return error.message
  const text = String(error || '').trim()
  return text || fallback
}

function waitForCaptionBackend(timeoutMs = CAPTION_BACKEND_WAIT_MS): Promise<any> {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => reject(new Error('백엔드 연결 대기 시간이 초과되었습니다.')), timeoutMs)
    getBackend().then(
      backend => { window.clearTimeout(timer); resolve(backend) },
      error => { window.clearTimeout(timer); reject(error) },
    )
  })
}

function invokeCaptionJson<T>(
  backend: any,
  method: string,
  payload: Record<string, unknown>,
  timeoutMs = CAPTION_BRIDGE_CALL_MS,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const fn = backend?.[method]
    if (typeof fn !== 'function') {
      reject(new Error(`${method} 기능을 사용할 수 없습니다.`))
      return
    }
    let settled = false
    const timer = window.setTimeout(() => {
      if (settled) return
      settled = true
      reject(new Error(`${method} 응답 시간이 초과되었습니다.`))
    }, timeoutMs)
    try {
      fn(JSON.stringify(payload), (raw: unknown) => {
        if (settled) return
        settled = true
        window.clearTimeout(timer)
        try {
          const parsed = typeof raw === 'string' ? JSON.parse(raw || '{}') : raw
          if (!parsed || typeof parsed !== 'object') throw new Error('잘못된 백엔드 응답입니다.')
          resolve(parsed as T)
        } catch (error) {
          reject(error)
        }
      })
    } catch (error) {
      settled = true
      window.clearTimeout(timer)
      reject(error)
    }
  })
}

const needsCaformer = computed(() => captionEngine.value === 'caformer' || captionEngine.value === 'combined')
const needsNaturalCaption = computed(() => captionEngine.value !== 'caformer')
const detectedCaformerDir = computed(() => captionRuntime.value.caformer?.modelDir || '')
const activeCaformerDir = computed(() => captionCaformerDir.value.trim() || detectedCaformerDir.value)
const toriiModels = computed(() => ollamaModels.value.filter(model => /toriigate/i.test(model)))
const visibleCaptionModels = computed(() => captionEngine.value === 'ollama' ? ollamaModels.value : toriiModels.value)
const activeCaptionModel = computed<string>({
  get: () => captionEngine.value === 'ollama' ? captionModel.value : captionToriiModel.value,
  set: value => {
    if (captionEngine.value === 'ollama') captionModel.value = String(value || '')
    else captionToriiModel.value = String(value || '')
  },
})
const captionRuntimeError = computed(() => {
  if (captionRuntime.value.error) return captionRuntime.value.error
  if (needsCaformer.value && captionRuntime.value.caformer?.error) return captionRuntime.value.caformer.error
  if (needsNaturalCaption.value && captionEngine.value !== 'ollama' && captionRuntime.value.torii?.error) return captionRuntime.value.torii.error
  return ''
})
const captionRuntimeSummary = computed(() => {
  const parts: string[] = []
  if (needsCaformer.value) parts.push(captionRuntime.value.caformer?.available ? 'CAFormer 준비됨' : 'CAFormer 확인 필요')
  if (needsNaturalCaption.value) {
    if (captionEngine.value === 'ollama') parts.push('Ollama 비전 모델 사용')
    else parts.push(captionRuntime.value.torii?.available ? 'ToriiGate 준비됨' : 'ToriiGate 확인 필요')
  }
  return parts.join(' · ') || '런타임 확인 전'
})
const captionEmptyHint = computed(() => {
  const action = captionSave.value ? '.txt 사이드카로 저장합니다' : '화면에서 검토할 수 있습니다'
  if (captionEngine.value === 'caformer') return `CAFormer로 Danbooru 태그를 만들고 ${action}`
  if (captionEngine.value === 'torii') return `ToriiGate로 자연어 캡션을 만들고 ${action}`
  if (captionEngine.value === 'ollama') return `선택한 Ollama 비전 모델로 자연어 캡션을 만들고 ${action}`
  return `CAFormer 태그와 ToriiGate 자연어 캡션을 만들고 ${action}`
})

const captionUrl = () => storedOllamaUrl()
function saveCaptionPrefs() {
  requestAction('save_ui_prefs', {
    captionEngine: captionEngine.value,
    captionModel: captionModel.value,
    captionToriiModel: captionToriiModel.value,
    captionPrompt: captionPrompt.value,
    captionSave: captionSave.value,
    captionOverwrite: captionOverwrite.value,
    captionOutDir: captionOutDir.value,
    captionCaformerDir: captionCaformerDir.value,
    captionIncludeCharacters: captionIncludeCharacters.value,
    captionIncludeRating: captionIncludeRating.value,
    captionUseBestThresholds: captionUseBestThresholds.value,
    captionGeneralThreshold: captionGeneralThreshold.value,
    captionCharacterThreshold: captionCharacterThreshold.value,
    captionRatingThreshold: captionRatingThreshold.value,
  })
}
function applyCaptionPrefs(prefs: Record<string, any>) {
  const previousOutDir = captionOutDir.value
  if (['caformer', 'torii', 'combined', 'ollama'].includes(prefs.captionEngine)) captionEngine.value = prefs.captionEngine
  if (typeof prefs.captionModel === 'string') captionModel.value = prefs.captionModel
  if (typeof prefs.captionToriiModel === 'string' && prefs.captionToriiModel) captionToriiModel.value = prefs.captionToriiModel
  if (typeof prefs.captionPrompt === 'string') captionPrompt.value = prefs.captionPrompt
  if (typeof prefs.captionSave === 'boolean') captionSave.value = prefs.captionSave
  if (typeof prefs.captionOverwrite === 'boolean') captionOverwrite.value = prefs.captionOverwrite
  if (typeof prefs.captionOutDir === 'string') captionOutDir.value = prefs.captionOutDir
  if (typeof prefs.captionCaformerDir === 'string') captionCaformerDir.value = prefs.captionCaformerDir
  if (typeof prefs.captionIncludeCharacters === 'boolean') captionIncludeCharacters.value = prefs.captionIncludeCharacters
  if (typeof prefs.captionIncludeRating === 'boolean') captionIncludeRating.value = prefs.captionIncludeRating
  if (typeof prefs.captionUseBestThresholds === 'boolean') captionUseBestThresholds.value = prefs.captionUseBestThresholds
  if (Number.isFinite(Number(prefs.captionGeneralThreshold))) captionGeneralThreshold.value = Number(prefs.captionGeneralThreshold)
  if (Number.isFinite(Number(prefs.captionCharacterThreshold))) captionCharacterThreshold.value = Number(prefs.captionCharacterThreshold)
  if (Number.isFinite(Number(prefs.captionRatingThreshold))) captionRatingThreshold.value = Number(prefs.captionRatingThreshold)
  if (captionItems.value.length && previousOutDir !== captionOutDir.value) reloadCaptionSidecars()
}
function onCaptionOutDirChanged(next: string) {
  if (captionRunning.value) return
  captionOutDir.value = next
  saveCaptionPrefs()
  reloadCaptionSidecars()
}
function clearCaptionOutDir() { onCaptionOutDirChanged('') }
function clearCaptionCaformerDir() { captionCaformerDir.value = ''; saveCaptionPrefs() }
function onCaptionEngineChanged() {
  choosePreferredToriiModel()
  saveCaptionPrefs()
}
function choosePreferredToriiModel() {
  if (captionEngine.value !== 'torii' && captionEngine.value !== 'combined') return
  const preferred = toriiModels.value.find(model => /:bf16$/i.test(model)) || toriiModels.value[0]
  if (preferred && !toriiModels.value.includes(captionToriiModel.value)) {
    captionToriiModel.value = preferred
    saveCaptionPrefs()
  }
}
async function loadCaptionModels() {
  try {
    const backend = await waitForCaptionBackend()
    // 결과는 ollamaModelsReady → applyCaptionModels (동기 ollamaListModels 는 없앴다)
    if (backend.requestOllamaModels) backend.requestOllamaModels(captionUrl())
  } catch (error) {
    captionToast('error', errorMessage(error, 'Ollama 모델 목록을 불러오지 못했습니다.'))
  }
}
function applyCaptionModels(json: string) {
  try {
    const payload = JSON.parse(json)
    const models = Array.isArray(payload) ? payload : payload.models
    if (!Array.isArray(models)) return
    if (!Array.isArray(payload) && payload.url && payload.url !== captionUrl()) return
    ollamaModels.value = models
    if (!captionModel.value && models.length) captionModel.value = models[0]
    choosePreferredToriiModel()
  } catch {}
}

function scheduleCaptionRuntimeProbe() {
  if (captionDisposed || subTab.value !== 'caption' || captionRunning.value) return
  if (captionRuntimeDebounceTimer) window.clearTimeout(captionRuntimeDebounceTimer)
  captionRuntimeDebounceTimer = window.setTimeout(() => {
    captionRuntimeDebounceTimer = null
    void loadCaptionRuntime()
  }, CAPTION_RUNTIME_DEBOUNCE_MS)
}

function invalidateCaptionRuntimeSnapshot() {
  // 입력이 바뀐 순간 기존 요청/결과를 폐기한다. 다음 debounce probe 전 실행해도
  // 이전 explicit 경로나 모델을 준비 완료로 표시하거나 payload에 재사용하지 않는다.
  captionRuntimeRequestId += 1
  if (captionRuntimeResponseTimer) window.clearTimeout(captionRuntimeResponseTimer)
  captionRuntimeResponseTimer = null
  captionRuntimeLoading.value = false
  captionRuntime.value = {}
}

async function loadCaptionRuntime() {
  if (captionDisposed) return
  if (captionRuntimeDebounceTimer) {
    window.clearTimeout(captionRuntimeDebounceTimer)
    captionRuntimeDebounceTimer = null
  }
  if (captionRuntimeResponseTimer) {
    window.clearTimeout(captionRuntimeResponseTimer)
    captionRuntimeResponseTimer = null
  }
  captionRuntimeLoading.value = true
  const requestId = ++captionRuntimeRequestId
  captionRuntime.value = {}
  try {
    const backend = await waitForCaptionBackend()
    if (captionDisposed || requestId !== captionRuntimeRequestId) return
    if (typeof backend.requestCaptionRuntime !== 'function') {
      throw new Error('캡션 런타임 확인 기능을 사용할 수 없습니다.')
    }
    backend.requestCaptionRuntime(JSON.stringify({
      clientToken: captionClientToken,
      requestId,
      url: captionUrl(),
      caformerModelDir: captionCaformerDir.value,
      toriiModel: captionToriiModel.value,
    }))
    captionRuntimeResponseTimer = window.setTimeout(() => {
      if (captionDisposed || requestId !== captionRuntimeRequestId) return
      captionRuntimeResponseTimer = null
      captionRuntimeLoading.value = false
      captionRuntime.value = {
        clientToken: captionClientToken,
        requestId,
        error: '캡션 런타임 확인 시간이 초과되었습니다.',
      }
    }, CAPTION_RUNTIME_WAIT_MS)
  } catch (error) {
    if (captionDisposed || requestId !== captionRuntimeRequestId) return
    captionRuntimeLoading.value = false
    captionRuntime.value = {
      clientToken: captionClientToken,
      requestId,
      error: errorMessage(error, '캡션 런타임을 확인하지 못했습니다.'),
    }
  }
}
function applyUpscalers(json: string) {
  try {
    const list = JSON.parse(json)
    if (Array.isArray(list) && list.length) {
      upscalers.value = list
      if (!list.includes(upscaler.value)) upscaler.value = list[0]
    }
  } catch {}
}
function applyADetailerModels(json: string) {
  try {
    const models = JSON.parse(json)
    if (models.length) { adModelItems.value = models; adModel.value = models[0] }
  } catch {}
}
function resolvedCaptionSidecarPath(imagePath: string) {
  return captionSidecarPath(imagePath, captionOutDir.value)
}

function invalidateCaptionSidecarLoads() {
  captionSidecarGeneration += 1
  captionSidecarErrorGeneration = -1
}

function clearCaption() {
  if (captionRunning.value) return
  invalidateCaptionSidecarLoads()
  captionItems.value = []
}
function statusLabel(s: string) {
  return ({ pending: '생성 중', done: '✓ 완료', error: '⚠ 실패', skip: '건너뜀' } as Record<string, string>)[s] || ''
}

function reloadCaptionSidecars() {
  if (captionRunning.value) return
  invalidateCaptionSidecarLoads()
  const generation = captionSidecarGeneration
  const outDir = captionOutDir.value
  for (const item of captionItems.value) {
    item.caption = ''
    item.status = ''
    item.error = ''
    item.txtPath = ''
    void loadCaptionFor(item, generation, outDir)
  }
}

function showSidecarLoadErrorOnce(generation: number, message: string) {
  if (generation !== captionSidecarGeneration || captionSidecarErrorGeneration === generation) return
  captionSidecarErrorGeneration = generation
  captionToast('error', `캡션 파일 불러오기 실패: ${message}`)
}

async function loadCaptionFor(
  item: CaptionItem,
  generation = captionSidecarGeneration,
  outDir = captionOutDir.value,
) {
  try {
    const backend = await waitForCaptionBackend()
    const data = await invokeCaptionJson<{ caption?: string; txtPath?: string; error?: string }>(
      backend,
      'loadCaption',
      { path: item.path, outDir },
    )
    if (
      generation !== captionSidecarGeneration
      || outDir !== captionOutDir.value
      || !captionItems.value.includes(item)
    ) return
    if (data.error) throw new Error(data.error)
    item.caption = typeof data.caption === 'string' ? data.caption : ''
    item.txtPath = typeof data.txtPath === 'string' ? data.txtPath : resolvedCaptionSidecarPath(item.path)
    item.error = ''
  } catch (error) {
    if (generation !== captionSidecarGeneration || !captionItems.value.includes(item)) return
    const message = errorMessage(error, '사이드카를 읽지 못했습니다.')
    item.error = message
    showSidecarLoadErrorOnce(generation, message)
  }
}

async function captionSingle(item: CaptionItem) {
  if (captionRunning.value) return
  if (!validateCaptionConfig()) return
  captionRunning.value = true; captionCur.value = 0; captionTotal.value = 1
  item.status = 'pending'
  item.error = ''
  await startCaptionJob([item.path], captionOverwrite.value)
}

async function captionAll() {
  if (captionRunning.value || !captionItems.value.length) return
  if (!validateCaptionConfig()) return
  captionRunning.value = true; captionCur.value = 0; captionTotal.value = captionItems.value.length
  for (const it of captionItems.value) { it.status = ''; it.error = '' }
  await startCaptionJob(captionItems.value.map(i => i.path), captionOverwrite.value)
}
function validateCaptionConfig() {
  if (needsNaturalCaption.value && !activeCaptionModel.value.trim()) {
    requestAction('show_toast', { type: 'error', msg: 'Ollama 비전 모델을 입력하세요' })
    return false
  }
  return true
}
function captionPayload(files: string[], overwrite: boolean, jobId: string) {
  return {
    clientToken: captionClientToken,
    jobId,
    files,
    engine: captionEngine.value,
    prompt: captionPrompt.value,
    model: activeCaptionModel.value,
    url: captionUrl(),
    save: captionSave.value,
    overwrite,
    outDir: captionOutDir.value,
    caformerModelDir: activeCaformerDir.value,
    includeCharacters: captionIncludeCharacters.value,
    includeRating: captionIncludeRating.value,
    useBestThresholds: captionUseBestThresholds.value,
    generalThreshold: captionGeneralThreshold.value,
    characterThreshold: captionCharacterThreshold.value,
    ratingThreshold: captionRatingThreshold.value,
    separator: '\n\n',
  }
}

function isActiveCaptionJob(payload: { clientToken?: string; jobId?: string }, expectedJobId = activeCaptionJobId) {
  return Boolean(
    matchesCaptionIdentity(payload, captionClientToken, expectedJobId)
    && activeCaptionJobId === expectedJobId,
  )
}

function clearCaptionJobPolling() {
  if (captionJobPollTimer) window.clearTimeout(captionJobPollTimer)
  captionJobPollTimer = null
}

function applyCaptionProgress(data: CaptionProgressEvent) {
  if (!isActiveCaptionJob(data)) return false
  captionJobLastContactAt = Date.now()
  captionJobIdlePolls = 0
  if (Number.isFinite(Number(data.index))) captionCur.value = Number(data.index) + 1
  if (Number.isFinite(Number(data.total))) captionTotal.value = Number(data.total)
  const item = captionItems.value.find(candidate => candidate.path === data.path)
  if (!item) return true
  if (data.error) {
    item.status = 'error'
    item.error = data.error
    return true
  }
  if (typeof data.caption === 'string') item.caption = data.caption
  if (typeof data.txtPath === 'string') item.txtPath = data.txtPath
  item.error = ''
  item.status = data.skipped ? 'skip' : 'done'
  return true
}

function failActiveCaptionJob(jobId: string, message: string) {
  if (jobId !== activeCaptionJobId) return
  clearCaptionJobPolling()
  captionRunning.value = false
  for (const item of captionItems.value) {
    if (!activeCaptionPaths.has(item.path) || (item.status && item.status !== 'pending')) continue
    item.status = 'error'
    item.error = message
  }
  activeCaptionPaths = new Set()
  activeCaptionJobId = ''
  captionToast('error', `캡션 실패: ${message}`)
  scheduleCaptionRuntimeProbe()
}

function finishCaptionJob(data: CaptionDoneEvent | CaptionJobStatus) {
  if (!isActiveCaptionJob(data)) return
  clearCaptionJobPolling()
  captionRunning.value = false
  if (Number.isFinite(Number(data.total))) captionTotal.value = Number(data.total)
  const finished = Number(data.ok || 0) + Number(data.failed || 0)
  if (finished > captionCur.value) captionCur.value = Math.min(captionTotal.value || finished, finished)
  activeCaptionPaths = new Set()
  activeCaptionJobId = ''

  const detail = data.error ? ` · ${data.error}` : ''
  if (data.error && !data.ok) {
    captionToast('error', `캡션 실패: ${data.error}`)
  } else {
    captionToast(
      data.failed || data.error ? 'warning' : 'success',
      `캡션 완료: ${data.ok}/${data.total}${data.failed ? ` (실패 ${data.failed})` : ''}${detail}`,
    )
  }
  scheduleCaptionRuntimeProbe()
}

function handleCaptionDone(data: CaptionDoneEvent) {
  if (!isActiveCaptionJob(data)) return
  captionJobLastContactAt = Date.now()
  if (!hasUnresolvedCaptionItems(captionItems.value, activeCaptionPaths)) {
    finishCaptionJob(data)
    return
  }
  // 재연결 중 progress만 놓치고 done을 받은 경우, 완료 신호만으로 닫으면
  // 항목 텍스트가 빈 채 남는다. 저널을 먼저 받아 applyCaptionProgress로 복원한다.
  scheduleCaptionJobPoll(0)
}

function scheduleCaptionJobPoll(delay = CAPTION_JOB_POLL_MS) {
  clearCaptionJobPolling()
  if (captionDisposed || !captionRunning.value || !activeCaptionJobId) return
  const jobId = activeCaptionJobId
  captionJobPollTimer = window.setTimeout(() => {
    captionJobPollTimer = null
    void pollCaptionJob(jobId)
  }, delay)
}

async function pollCaptionJob(jobId: string) {
  if (captionDisposed || !captionRunning.value || jobId !== activeCaptionJobId || captionJobPollInFlight) return
  captionJobPollInFlight = true
  try {
    const backend = await waitForCaptionBackend()
    if (captionDisposed || jobId !== activeCaptionJobId) return
    const status = await invokeCaptionJson<CaptionJobStatus>(
      backend,
      'getCaptionJobStatus',
      { clientToken: captionClientToken, jobId },
    )
    if (!isActiveCaptionJob(status, jobId)) {
      throw new Error('다른 캡션 작업의 상태 응답을 받았습니다.')
    }
    captionJobLastContactAt = Date.now()
    if (Number.isFinite(Number(status.total))) captionTotal.value = Number(status.total)
    if (Array.isArray(status.items)) {
      for (const item of status.items) {
        if (item) applyCaptionProgress(item)
      }
    }
    if (status.status === 'done') {
      finishCaptionJob(status)
      return
    }
    if (status.status === 'running') {
      captionJobIdlePolls = 0
      const current = Number(
        status.current
        ?? status.processed
        ?? (Number(status.ok || 0) + Number(status.failed || 0)),
      )
      if (Number.isFinite(current)) captionCur.value = Math.max(captionCur.value, current)
    } else if (status.status === 'idle') {
      captionJobIdlePolls += 1
      if (captionJobIdlePolls >= 2) {
        failActiveCaptionJob(jobId, status.error || '백엔드에서 실행 중인 캡션 작업을 찾지 못했습니다.')
        return
      }
    }
  } catch (error) {
    if (captionDisposed || jobId !== activeCaptionJobId) return
    if (Date.now() - captionJobLastContactAt >= CAPTION_JOB_LOST_MS) {
      failActiveCaptionJob(
        jobId,
        `${errorMessage(error, '작업 상태를 확인하지 못했습니다.')} 백엔드 작업은 계속 실행 중일 수 있습니다.`,
      )
      return
    }
  } finally {
    captionJobPollInFlight = false
    if (!captionDisposed && captionRunning.value && jobId === activeCaptionJobId) scheduleCaptionJobPoll()
  }
}

async function startCaptionJob(files: string[], overwrite: boolean) {
  invalidateCaptionSidecarLoads()
  const jobId = makeCaptionToken('caption-job')
  activeCaptionJobId = jobId
  activeCaptionPaths = new Set(files)
  captionJobLastContactAt = Date.now()
  captionJobIdlePolls = 0
  let backend: any
  try {
    backend = await waitForCaptionBackend()
  } catch (error) {
    failActiveCaptionJob(jobId, errorMessage(error, '캡션 백엔드에 연결하지 못했습니다.'))
    return
  }
  if (jobId !== activeCaptionJobId) return
  try {
    const response = await invokeCaptionJson<CaptionStartResponse>(
      backend,
      'startCaptionBatch',
      captionPayload(files, overwrite, jobId),
    )
    if (jobId !== activeCaptionJobId) return
    // 메서드 반환 콜백은 이 호출에만 귀속된다. 준비 단계에서 실패하면 백엔드가
    // 식별자를 정규화하기 전일 수 있으므로, 명시적 오류를 먼저 전달한다.
    if (response.error) {
      failActiveCaptionJob(jobId, response.error)
      return
    }
    if (!isActiveCaptionJob(response, jobId)) {
      failActiveCaptionJob(jobId, '다른 캡션 작업의 시작 응답을 받았습니다.')
      return
    }
    if (!response.started) {
      failActiveCaptionJob(jobId, '캡션 작업이 시작되지 않았습니다.')
      return
    }
    captionJobLastContactAt = Date.now()
    scheduleCaptionJobPoll(500)
  } catch (error) {
    if (jobId !== activeCaptionJobId) return
    // 콜백만 유실되고 작업은 시작됐을 수 있으므로 상태 조회를 통해 한 번 더 판정한다.
    captionJobLastContactAt = Math.min(
      captionJobLastContactAt,
      Date.now() - CAPTION_JOB_LOST_MS + CAPTION_JOB_POLL_MS,
    )
    scheduleCaptionJobPoll(0)
    console.warn('[caption] start response unavailable; reconciling job status', error)
  }
}

async function saveCaptionItem(item: CaptionItem) {
  if (captionRunning.value) return
  try {
    const backend = await waitForCaptionBackend()
    const data = await invokeCaptionJson<{ ok?: boolean; txtPath?: string; error?: string }>(
      backend,
      'saveCaption',
      { path: item.path, caption: item.caption, outDir: captionOutDir.value },
    )
    if (data.error) throw new Error(data.error)
    if (!data.ok) throw new Error('캡션 파일을 저장하지 못했습니다.')
    item.txtPath = data.txtPath || resolvedCaptionSidecarPath(item.path)
    item.error = ''
    captionToast('success', `캡션 저장됨: ${item.txtPath}`)
  } catch (error) {
    const message = errorMessage(error, '캡션 파일을 저장하지 못했습니다.')
    item.error = message
    captionToast('error', `캡션 저장 실패: ${message}`)
  }
}

// 캡션을 T2I 메인 프롬프트에 추가하고 T2I 탭으로 이동
function sendCaptionToT2I(item: CaptionItem) {
  const cap = (item.caption || '').trim()
  if (!cap) { requestAction('show_toast', { type: 'warning', msg: '캡션이 비어있습니다' }); return }
  const cur = ((widgets as any).main_prompt_text || '').trim().replace(/[,\s]+$/, '')
  ;(widgets as any).main_prompt_text = cur ? (cur + ', ' + cap) : cap
  router.push('/')   // T2I 탭 (path '/')
  requestAction('show_toast', { type: 'success', msg: 'T2I 프롬프트에 추가했습니다' })
}

// ── ADetailer ──
const adFiles = ref<string[]>([])
const adModel = ref('face_yolov8n.pt')
const adModelItems = ref<string[]>([])
const adConfidence = ref(0.3)
const adDenoise = ref(0.4)
const adPrompt = ref('')
const adUseExifPrompt = ref(false)
const adCurrentIdx = ref(-1)
const adPreview = ref('')
const adBefore = ref('')
const adAfter = ref('')
const adResults = ref<Record<number, boolean>>({})  // index → true
const adProcessing = ref(false)
const adProgressCur = ref(0)
const adProgressTotal = ref(0)
const adProgressPct = computed(() => adProgressTotal.value ? Math.round(adProgressCur.value / adProgressTotal.value * 100) : 0)
// ETA — 시작 시각과 진행 수를 기반으로 평균 처리 시간을 추정
const _adStartTime = ref(0)
const adEtaText = computed(() => {
  if (!adProcessing.value || _adStartTime.value === 0) return ''
  const done = adProgressCur.value
  const total = adProgressTotal.value
  const remaining = total - done
  if (done < 1 || remaining <= 0) return ''
  const elapsedSec = (Date.now() - _adStartTime.value) / 1000
  const avgPerItem = elapsedSec / done
  const sec = Math.round(avgPerItem * remaining)
  if (sec < 60) return `${sec}초`
  if (sec < 3600) return `${Math.floor(sec / 60)}분 ${sec % 60}초`
  return `${Math.floor(sec / 3600)}시 ${Math.floor((sec % 3600) / 60)}분`
})

function onDropAd(e: DragEvent) {
  addDroppedPaths(e, adFiles)
}

function previewAdFile(i: number) {
  adCurrentIdx.value = i
  adPreview.value = adFiles.value[i]
  adBefore.value = ''
  adAfter.value = ''
}

function _adSettings() {
  return {
    ad_model: adModel.value,
    ad_confidence: adConfidence.value,
    ad_denoise: adDenoise.value,
    ad_prompt: adUseExifPrompt.value ? '' : adPrompt.value,
    use_exif_prompt: adUseExifPrompt.value,
  }
}

// 'EXIF 프롬프트 사용' — 워커가 메타데이터를 못 읽으면 결과에 exif_warning 을 싣는다(실행마다 첫 경고 + 요약)
const adExifNotice = createExifWarningNotice('ADetailer')
const sam3ExifNotice = createExifWarningNotice('SAM3')

function runAdSingle() {
  const idx = adCurrentIdx.value >= 0 ? adCurrentIdx.value : 0
  const path = adFiles.value[idx]
  if (!path) return
  adExifNotice.reset()
  adProcessing.value = true
  adBefore.value = path
  adAfter.value = ''
  action('run_adetailer_single', { path, settings: _adSettings() })
}

function runAdBatch() {
  if (!adFiles.value.length) return
  adExifNotice.reset()
  adProcessing.value = true
  adResults.value = {}
  adProgressCur.value = 0
  adProgressTotal.value = adFiles.value.length
  _adStartTime.value = Date.now()  // ETA 계산용 시작 시각
  action('run_adetailer_batch', { paths: adFiles.value, settings: _adSettings() })
}

// ── SAM3 ──
const sam3Files = ref<string[]>([])
const sam3Prompt = ref('face')
const sam3InpaintPrompt = ref('')
const sam3NegativePrompt = ref('')
const sam3UseExifPrompt = ref(false)
const sam3MaskMode = ref('Individual')
const sam3Threshold = ref(0.4)
const sam3Denoise = ref(0.3)
const sam3MaskBlur = ref(8)
const sam3Padding = ref(32)
const sam3Checkpoint = ref('sam3.pt')
const sam3PreviewOverlay = ref(false)
const sam3SaveArtifacts = ref(true)
// t2i 패널과 동일 수준으로 노출 (예전엔 배치에 없어 확장 기본값으로만 돌았음)
const sam3Mode = ref('Inpaint')
const sam3ExcludePrompt = ref('')
const sam3MaskDilation = ref(0)
const sam3MaskHull = ref(false)
const sam3MaskOutline = ref(0)
const sam3Device = ref('cuda')
const sam3Fill = ref('original')
const sam3OnlyMasked = ref(true)
const sam3Steps = ref(28)
const sam3Cfg = ref(7)
const sam3Seed = ref(-1)
const sam3RestoreFace = ref(false)
const sam3UnloadAfter = ref(true)
// ControlNet 13필드 (widget id 키 로컬 상태 — utils/sam3ControlNet)
const sam3Cn = reactive(sam3CnDefaults())
const sam3CurrentIdx = ref(-1)
const sam3Preview = ref('')
const sam3Before = ref('')
const sam3After = ref('')
const sam3Results = ref<Record<number, boolean>>({})
const sam3Processing = ref(false)
const sam3ProgressCur = ref(0)
const sam3ProgressTotal = ref(0)
const sam3ProgressPct = computed(() => sam3ProgressTotal.value ? Math.round(sam3ProgressCur.value / sam3ProgressTotal.value * 100) : 0)

function onDropSam3(e: DragEvent) {
  addDroppedPaths(e, sam3Files)
}

function previewSam3File(i: number) {
  sam3CurrentIdx.value = i
  sam3Preview.value = sam3Files.value[i]
  sam3Before.value = ''
  sam3After.value = ''
}

function _sam3Settings() {
  return {
    sam3_mode: sam3Mode.value,
    sam3_mask_mode: sam3MaskMode.value,
    sam3_prompt: sam3Prompt.value || 'face',
    // 예전엔 배치 경로에 exclude/dilation/hull/outline/device 입력이 없어
    // 확장 기본값으로만 돌았다 — t2i 패널과 같은 수준으로 노출한다
    sam3_exclude_prompt: sam3ExcludePrompt.value,
    sam3_inpaint_prompt: sam3UseExifPrompt.value ? '' : sam3InpaintPrompt.value,
    sam3_negative_prompt: sam3NegativePrompt.value,
    sam3_threshold: sam3Threshold.value,
    sam3_mask_dilation: sam3MaskDilation.value,
    sam3_mask_hull: sam3MaskHull.value,
    sam3_mask_outline_px: sam3MaskOutline.value,
    sam3_checkpoint: sam3Checkpoint.value || 'sam3.pt',
    sam3_device: sam3Device.value,
    sam3_mask_blur: sam3MaskBlur.value,
    sam3_denoising_strength: sam3Denoise.value,
    sam3_inpainting_fill: sam3Fill.value,
    sam3_inpaint_only_masked: sam3OnlyMasked.value,
    sam3_inpaint_only_masked_padding: sam3Padding.value,
    sam3_use_inpaint_width_height: false,
    sam3_inpaint_width: 1024,
    sam3_inpaint_height: 1024,
    sam3_use_steps: true,
    sam3_steps: sam3Steps.value,
    sam3_use_cfg_scale: true,
    sam3_cfg_scale: sam3Cfg.value,
    sam3_use_seed: sam3Seed.value !== -1,
    sam3_seed: sam3Seed.value,
    sam3_restore_face: sam3RestoreFace.value,
    sam3_preview_overlay: sam3PreviewOverlay.value,
    sam3_save_artifacts: sam3SaveArtifacts.value,
    // 16GB GPU에서 인페인트 OOM 방지 — 확장 API 기본값이 False라 명시 전송 필수
    sam3_unload_after: sam3UnloadAfter.value,
    // ControlNet 주입 (sam3_cn_* 13필드)
    ...sam3CnSettings(sam3Cn),
    use_exif_prompt: sam3UseExifPrompt.value,
    // 부모 i2i 샘플링 파라미터 (Forge 현재 UI 값에 좌우되지 않게)
    steps: sam3Steps.value,
    cfg_scale: sam3Cfg.value,
    seed: sam3Seed.value,
  }
}

function runSam3Single() {
  const idx = sam3CurrentIdx.value >= 0 ? sam3CurrentIdx.value : 0
  const path = sam3Files.value[idx]
  if (!path) return
  sam3ExifNotice.reset()
  sam3Processing.value = true
  sam3Before.value = path
  sam3After.value = ''
  action('run_sam3_single', { path, settings: _sam3Settings() })
}

function runSam3Batch() {
  if (!sam3Files.value.length) return
  sam3ExifNotice.reset()
  sam3Processing.value = true
  sam3Results.value = {}
  sam3ProgressCur.value = 0
  sam3ProgressTotal.value = sam3Files.value.length
  action('run_sam3_batch', { paths: sam3Files.value, settings: _sam3Settings() })
}

// 캡션 화면/모델 경로가 바뀌면 마지막 값 하나만 검사한다.
watch(subTab, (value) => {
  if (value === 'caption' && !captionRunning.value) {
    void loadCaptionModels()
    scheduleCaptionRuntimeProbe()
  }
})
watch(
  [captionEngine, captionCaformerDir, captionToriiModel],
  () => {
    invalidateCaptionRuntimeSnapshot()
    scheduleCaptionRuntimeProbe()
  },
)

// BatchView는 keep-alive라 Settings에서 Ollama URL을 바꾼 뒤 돌아오는 경우도 새로 확인해야 한다.
onActivated(() => {
  if (subTab.value !== 'caption' || captionRunning.value) return
  void loadCaptionModels()
  scheduleCaptionRuntimeProbe()
})

onMounted(async () => {
  captionDisposed = false
  captionEventUnsubs.push(onBackendEvent('ollamaModelsReady', applyCaptionModels))
  // 일괄 처리·업스케일 진행 — 실행 중 버튼 잠금과 진행률 (Python ui/batch_jobs.py)
  captionEventUnsubs.push(onBackendEvent('batchJobState', (json: string) => {
    const state = parseBatchJobState(json)
    if (!state) return
    if (state.job === 'batch') batchJob.value = state
    else upscaleJob.value = state
  }))
  onBackendEvent('upscalersReady', applyUpscalers)
  onBackendEvent('adetailerModelsReady', applyADetailerModels)

  const backend: any = await getBackend()
  if (captionDisposed) return

  if (backend.getUiPrefs) {
    backend.getUiPrefs((json: string) => {
      try {
        const prefs = JSON.parse(json || '{}')
        const needsMigration = !Object.prototype.hasOwnProperty.call(prefs, 'captionEngine')
        applyCaptionPrefs(prefs)
        choosePreferredToriiModel()
        if (needsMigration) saveCaptionPrefs()
        scheduleCaptionRuntimeProbe()
      } catch {}
    })
  }

  // 캡션 모델 드롭다운 — UI 시작 시 자동 새로고침
  loadCaptionModels()

  // 업스케일러·AD 모델 로드 — 결과는 upscalersReady·adetailerModelsReady(위 구독).
  // GUI 스레드를 막던 동기 getUpscalers·getADetailerModels 는 없앴다.
  if (backend.requestUpscalers) backend.requestUpscalers()
  if (backend.requestADetailerModels) backend.requestADetailerModels()

  // 파일 선택 이벤트
  onBackendEvent('batchFilesSelected', (json: string) => {
    try {
      const parsed = JSON.parse(json)
      const paths: string[] = Array.isArray(parsed) ? parsed.filter((p: unknown): p is string => typeof p === 'string' && !!p) : []
      // 목록 항목은 경로가 v-for key 라 중복을 넣지 않는다
      const target = subTab.value === 'adetailer' ? adFiles
        : subTab.value === 'sam3' ? sam3Files
          : subTab.value === 'upscale' ? upscaleFiles
            : batchFiles
      target.value.push(...newPaths(target.value, paths))
    } catch {}
  })

  // ADetailer 결과 수신
  onBackendEvent('adetailerResult', (json: string) => {
    try {
      const d = JSON.parse(json)
      if (d.error) {
        requestAction('show_toast', { type: 'error', msg: `AD 오류: ${d.error}` })
        adProcessing.value = false
        return
      }
      adBefore.value = d.before
      adAfter.value = d.after
      const adWarn = adExifNotice.note(d)
      if (adWarn) requestAction('show_toast', { type: 'warning', msg: adWarn })
      if (typeof d.index === 'number') {
        adResults.value[d.index] = true
        adCurrentIdx.value = d.index
      }
      // 단일 처리 완료
      if (typeof d.index !== 'number') {
        adProcessing.value = false
        _adStartTime.value = 0
        requestAction('show_toast', { type: 'success', msg: 'ADetailer 완료' })
      }
      // 배치 전체 완료 — 마지막 인덱스가 total-1
      if (typeof d.index === 'number' && d.index === adProgressTotal.value - 1) {
        adProcessing.value = false
        _adStartTime.value = 0
        requestAction('show_toast', { type: 'success', msg: `ADetailer 배치 완료 (${adProgressTotal.value}장)` })
        const adSummary = adExifNotice.summary()
        if (adSummary) requestAction('show_toast', { type: 'warning', msg: adSummary })
      }
    } catch {}
  })

  // 배치 진행률
  onBackendEvent('adetailerProgress', (cur: number, total: number) => {
    adProgressCur.value = cur
    adProgressTotal.value = total
    if (cur >= total) adProcessing.value = false
  })

  onBackendEvent('sam3Result', (json: string) => {
    try {
      const d = JSON.parse(json)
      if (d.error) {
        requestAction('show_toast', { type: 'error', msg: `SAM3 오류: ${d.error}` })
        sam3Processing.value = false
        return
      }
      sam3Before.value = d.before
      sam3After.value = d.after
      const sam3Warn = sam3ExifNotice.note(d)
      if (sam3Warn) requestAction('show_toast', { type: 'warning', msg: sam3Warn })
      if (typeof d.index === 'number') {
        sam3Results.value[d.index] = true
        sam3CurrentIdx.value = d.index
      }
      if (typeof d.index !== 'number') {
        sam3Processing.value = false
        requestAction('show_toast', { type: 'success', msg: 'SAM3 완료' })
      }
    } catch {}
  })

  onBackendEvent('sam3Progress', (cur: number, total: number) => {
    sam3ProgressCur.value = cur
    sam3ProgressTotal.value = total
    if (cur >= total) {
      sam3Processing.value = false
      const sam3Summary = sam3ExifNotice.summary()
      if (sam3Summary) requestAction('show_toast', { type: 'warning', msg: sam3Summary })
    }
  })

  // 캡션 대상 선택 (파일/폴더)
  captionEventUnsubs.push(onBackendEvent('captionFilesSelected', (json: string) => {
    try {
      const paths = JSON.parse(json)
      if (captionRunning.value || !Array.isArray(paths)) return
      captionItems.value = paths.map((p: string) => ({ path: p, caption: '', status: '' }))
      reloadCaptionSidecars()
    } catch {}
  }))
  // 캡션 배치 진행
  captionEventUnsubs.push(onBackendEvent('captionProgress', (json: string) => {
    try {
      applyCaptionProgress(JSON.parse(json) as CaptionProgressEvent)
    } catch {}
  }))
  captionEventUnsubs.push(onBackendEvent('captionOutDirSelected', (p: string) => {
    if (!captionRunning.value) onCaptionOutDirChanged(p)
  }))
  captionEventUnsubs.push(onBackendEvent('captionModelDirSelected', (p: string) => {
    if (captionRunning.value) return
    captionCaformerDir.value = p
    saveCaptionPrefs()
  }))
  captionEventUnsubs.push(onBackendEvent('captionRuntimeReady', (json: string) => {
    try {
      const next = JSON.parse(json || '{}') as CaptionRuntimeSnapshot
      if (next.clientToken !== captionClientToken) return
      if (Number(next.requestId || 0) !== captionRuntimeRequestId) return
      if (captionRuntimeResponseTimer) window.clearTimeout(captionRuntimeResponseTimer)
      captionRuntimeResponseTimer = null
      captionRuntimeLoading.value = false
      captionRuntime.value = next
      const detectedTorii = captionRuntime.value.torii?.model
      if (detectedTorii && !captionToriiModel.value) captionToriiModel.value = detectedTorii
    } catch (error) {
      console.warn('[caption] malformed runtime response', error)
    }
  }))
  scheduleCaptionRuntimeProbe()
  captionEventUnsubs.push(onBackendEvent('captionDone', (json: string) => {
    try {
      handleCaptionDone(JSON.parse(json) as CaptionDoneEvent)
    } catch {}
  }))
})

onUnmounted(() => {
  captionDisposed = true
  captionRuntimeRequestId += 1
  for (const unsubscribe of captionEventUnsubs.splice(0)) unsubscribe()
  if (captionRuntimeResponseTimer) window.clearTimeout(captionRuntimeResponseTimer)
  if (captionRuntimeDebounceTimer) window.clearTimeout(captionRuntimeDebounceTimer)
  clearCaptionJobPolling()
  activeCaptionJobId = ''
  activeCaptionPaths = new Set()
  captionRunning.value = false
  invalidateCaptionSidecarLoads()
})
</script>

<style scoped>
.batch-view { width: 100%; height: 100%; display: flex; flex-direction: column; }

/* CAPTION 탭 */
.s-textarea { width: 100%; background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; color: var(--text-primary); font-size: 12px; resize: vertical; line-height: 1.4; }
.s-textarea:focus { outline: none; border-color: var(--accent); }
.caption-settings { width: 370px; }
.cap-controls { min-width: 0; margin: 0; padding: 0; border: 0; display: flex; flex-direction: column; gap: 10px; }
.cap-controls:disabled button, .cap-controls:disabled input, .cap-controls:disabled select,
.cap-controls:disabled textarea, .cap-controls:disabled :deep(.csel-display) { opacity: .55; cursor: not-allowed; }
.cap-model-row { display: flex; gap: 6px; align-items: center; }
.cap-model-row > :first-child { flex: 1; min-width: 0; }
.cap-refresh { flex-shrink: 0; width: 32px; height: 32px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 6px; color: var(--text-secondary); font-size: 13px; cursor: pointer; }
.cap-refresh:hover { color: var(--accent); border-color: var(--accent); }
.cap-opts { display: flex; gap: 14px; margin: 8px 0; }
.cap-runtime-card { padding: 8px 9px; border: 1px solid rgba(74,222,128,.28); border-radius: 7px; background: rgba(74,222,128,.07); }
.cap-runtime-card.warning { border-color: rgba(248,113,113,.32); background: rgba(248,113,113,.07); }
.cap-runtime-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; color: var(--text-secondary); font-size: 10px; font-weight: var(--fw-bold); }
.cap-inline-btn { padding: 2px 5px; border: 0; background: transparent; color: var(--accent); font-size: 9px; cursor: pointer; }
.cap-inline-btn:disabled { opacity: .5; cursor: default; }
.cap-runtime-path { margin-top: 5px; color: var(--text-muted); font-size: 9px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.cap-runtime-error { margin-top: 5px; color: var(--state-alert-fg); font-size: 9px; line-height: 1.35; }
.cap-path-input { flex: 1; min-width: 0; }
.cap-tag-opts { margin-bottom: 2px; }
.cap-best-toggle { display: flex; align-items: center; gap: 5px; color: var(--text-secondary); font-size: 10px; cursor: pointer; }
.cap-best-toggle input, .cap-tag-opts input { accent-color: var(--accent); }
.cap-threshold-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; }
.cap-threshold-grid label { display: grid; grid-template-columns: 42px minmax(0, 1fr); align-items: center; gap: 5px; color: var(--text-muted); font-size: 9px; white-space: nowrap; }
.cap-threshold-grid input { min-width: 0; width: 100%; padding: 5px 6px; }
.cap-combined-hint { padding: 7px 8px; border-radius: 6px; background: var(--accent-dim); color: var(--text-secondary); font-size: 9px; line-height: 1.4; }
.cap-outdir { display: flex; gap: 6px; align-items: center; }
.cap-outdir-path { flex: 1; min-width: 0; font-size: 11px; color: var(--text-secondary); background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px; padding: 7px 9px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cap-opts label { display: flex; align-items: center; gap: 5px; font-size: 11px; color: var(--text-secondary); cursor: pointer; }
.cap-pick { display: flex; gap: 8px; margin-top: 6px; }
.cap-clear { margin-top: 8px; align-self: flex-start; }
.cap-list { display: flex; flex-direction: column; gap: 10px; padding: 4px; }
.cap-item { display: flex; gap: 10px; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 8px; }
.cap-thumb { width: 110px; height: 110px; object-fit: cover; border-radius: 6px; flex-shrink: 0; background: var(--bg-input); }
.cap-body { flex: 1; display: flex; flex-direction: column; gap: 5px; min-width: 0; }
.cap-name { font-size: 11px; font-weight: var(--fw-bold); color: var(--text-secondary); display: flex; align-items: center; gap: 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cap-sidecar { color: var(--text-muted); font-size: 9px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cap-item-error { color: var(--state-alert-fg); font-size: 9px; line-height: 1.35; }
.cap-status { font-size: var(--fs-label); font-weight: var(--fw-bold); padding: 1px 6px; border-radius: 7px; flex-shrink: 0; }
/* 옅은 틴트 위의 '글자'라 채움용(--state-*)이 아니라 글자용(--state-*-fg) */
.cap-status.pending { background: rgba(251,191,36,0.18); color: var(--state-warn-fg); }
.cap-status.done { background: rgba(74,222,128,0.18); color: var(--state-ok-fg); }
.cap-status.error { background: rgba(248,113,113,0.18); color: var(--state-alert-fg); }
.cap-status.skip { background: var(--bg-button); color: var(--text-muted); }
.cap-text { flex: 1; min-height: 56px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px; padding: 7px 9px; color: var(--text-primary); font-size: 12px; resize: vertical; line-height: 1.45; }
.cap-text:focus { outline: none; border-color: var(--accent); }
.cap-text:disabled { opacity: .65; cursor: not-allowed; }
.cap-actions { display: flex; gap: 6px; }
.cap-btn { background: var(--bg-button); border: 1px solid var(--border); border-radius: 5px; color: var(--text-secondary); font-size: var(--fs-label); font-weight: var(--fw-bold); padding: 4px 10px; cursor: pointer; }
.cap-btn:hover:not(:disabled) { color: var(--accent); border-color: var(--accent); }
.cap-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.cap-btn.t2i { color: var(--accent); border-color: rgba(226,179,64,0.4); }
.cap-btn.t2i:hover:not(:disabled) { background: var(--accent-fill); color: var(--on-accent); }
.tab-body { flex: 1; overflow-y: auto; padding: 20px; }

.panel { max-width: 500px; margin: 0 auto; display: flex; flex-direction: column; gap: 10px; }
.panel h3 { color: var(--text-primary); font-size: 13px; font-weight: var(--fw-bold); letter-spacing: 0; margin: 0; }

.file-drop {
  border: 2px dashed var(--border); border-radius: 8px; min-height: 100px;
  display: flex; align-items: center; justify-content: center; padding: 12px;
}
.drop-hint { color: var(--text-muted); font-size: 12px; text-align: center; }
.link-btn { min-height: 28px; padding: 0 4px; background: none; border: none; color: var(--accent); cursor: pointer; text-decoration: underline; font-size: var(--fs-meta); }
.link-btn:disabled { opacity: .45; cursor: not-allowed; }
.file-list { width: 100%; max-height: 200px; overflow-y: auto; }
.file-item {
  display: flex; align-items: center; gap: 6px;
  padding: 4px 8px; font-size: 11px; color: var(--text-secondary); cursor: pointer;
  border-radius: 4px;
}
.file-item span { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.file-item.active { background: var(--accent-dim); color: var(--accent); }
.file-item.done { opacity: 0.6; }
.done-badge { color: var(--state-ok-fg); font-weight: var(--fw-bold); flex: 0 !important; }
.rm-btn { background: none; border: none; color: var(--state-alert-fg); cursor: pointer; font-size: 14px; flex-shrink: 0; }
.file-count { font-size: var(--fs-label); color: var(--text-muted); }

.s-label { color: var(--text-muted); font-size: var(--fs-label); font-weight: var(--fw-bold); letter-spacing: 0; }
.s-select, .s-input {
  background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px;
  padding: 8px 10px; color: var(--text-primary); font-size: 12px; outline: none;
}
.s-select:focus, .s-input:focus { border-color: var(--accent); }
.row { display: flex; align-items: center; gap: 6px; }
.row span { color: var(--text-muted); }
.slider-row { display: flex; align-items: center; gap: 8px; }
.slider-row input { flex: 1; accent-color: var(--accent); }
.slider-row span { color: var(--text-secondary); font-size: 12px; min-width: 30px; font-family: monospace; }
.op-settings { display: flex; flex-direction: column; gap: 4px; }
/* 결과 위치 안내 — 원본을 덮어쓰지 않는다는 걸 시작 전에 보여 준다 */
.op-hint {
  font-size: var(--fs-label); color: var(--text-muted); line-height: 1.45;
  overflow: hidden; text-overflow: ellipsis; word-break: break-all;
}
.sam3-cn { margin-top: 6px; }
/* 주 버튼: 면은 --accent-fill(글자가 4.5:1 로 읽히게 민 값), 글자는 --on-accent */
.btn-start {
  padding: 10px; background: var(--accent-fill); border: none; border-radius: 8px;
  color: var(--on-accent); font-weight: var(--fw-bold); font-size: 11px; cursor: pointer; letter-spacing: 0;
}
.btn-start:disabled { opacity: 0.3; cursor: not-allowed; }
.btn-start.batch { background: var(--bg-button); color: var(--accent); border: 1px solid var(--accent-dim); }
/* 정지 버튼은 '채움'이라 --state-alert(글자용 -fg 아님). 그 위 글자는 흰색 고정 —
   상태 채움색이 흰 글자와 4.5:1 을 맞춘 값이고, --text-primary 는 라이트에서 검정이 된다. */
.btn-stop { padding: 10px; background: var(--state-alert); border: none; border-radius: 8px; color: #FFFFFF; font-weight: var(--fw-bold); font-size: 11px; cursor: pointer; }

/* ADetailer Layout */
.ad-layout { display: flex; gap: 0; padding: 0 !important; }
.ad-settings { width: 320px; flex-shrink: 0; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; border-right: 1px solid var(--border); }
.ad-settings h3 { color: var(--text-primary); font-size: 13px; font-weight: var(--fw-bold); letter-spacing: 0; margin: 0; }
.ad-params { display: flex; flex-direction: column; gap: 8px; }
.ad-param { display: flex; flex-direction: column; gap: 2px; }
.ad-param label { font-size: var(--fs-label); color: var(--text-muted); font-weight: var(--fw-bold); }
.ad-param input { padding: 6px 8px; font-size: 12px; }
.ad-toggle { display: flex; align-items: center; gap: 4px; font-size: var(--fs-label); color: var(--text-muted); cursor: pointer; margin-bottom: 4px; }
.ad-toggle input { width: 14px; height: 14px; accent-color: var(--accent); }
.exif-prompt-hint { font-size: var(--fs-label); color: var(--accent); background: var(--accent-dim); padding: 6px 8px; border-radius: 4px; }
.ad-actions { display: flex; flex-direction: column; gap: 6px; margin-top: auto; }

.ad-progress { display: flex; align-items: center; gap: 8px; }
.progress-bar { flex: 1; height: 6px; background: var(--bg-button); border-radius: 3px; overflow: hidden; }
.progress-fill { height: 100%; background: var(--accent); border-radius: 3px; transition: width 0.3s; }
.ad-progress span { font-size: var(--fs-label); color: var(--text-muted); font-family: monospace; }

/* Compare */
.ad-compare { flex: 1; display: flex; align-items: center; justify-content: center; padding: 20px; overflow: hidden; }
/* 썸네일 그리드가 있으면 stretch — top-left 채움 */
.ad-compare:has(.thumb-grid) { align-items: stretch; justify-content: stretch; }
.compare-split { display: flex; gap: 12px; width: 100%; height: 100%; }
.compare-col { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 6px; min-width: 0; }
.compare-label { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-muted); letter-spacing: 0; }
.compare-col img { max-width: 100%; max-height: calc(100% - 24px); object-fit: contain; border-radius: 6px; }
.preview-single { display: flex; align-items: center; justify-content: center; }
.preview-single img { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 6px; }
.compare-empty { color: var(--text-muted); font-size: 13px; }

/* Batch/Upscale 우측 썸네일 그리드 */
.file-drop.compact { min-height: 80px; padding: 14px; }
.thumb-grid {
  width: 100%; height: 100%;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 10px; padding: 4px;
  align-content: start;
  overflow-y: auto;
}
.thumb-card {
  position: relative;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  display: flex; flex-direction: column;
  transition: all 0.15s;
  aspect-ratio: 1;
}
.thumb-card:hover {
  border-color: var(--text-muted);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
}
.thumb-card img {
  flex: 1; min-height: 0;
  width: 100%; object-fit: cover;
  background: var(--bg-primary);
}
.thumb-name {
  font-size: var(--fs-label); color: var(--text-secondary);
  padding: 5px 8px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  background: var(--bg-card);
  border-top: 1px solid var(--border);
}
/* 이 두 색은 토큰화하지 않는다: 바탕이 테마를 안 타는 검정 오버레이라
   --state-alert-fg 로 바꾸면 라이트에서 어두운 빨강이 검은 원 위에 얹혀 안 보인다.
   hover 도 채움(밝은 빨강)과 글자(검정)가 짝으로 맞춰진 값이다. */
.thumb-rm {
  position: absolute; top: 4px; right: 4px;
  width: 22px; height: 22px;
  background: rgba(0, 0, 0, 0.7); border: none;
  border-radius: 50%; color: #f87171;
  font-size: 16px; font-weight: var(--fw-bold);
  cursor: pointer; opacity: 0;
  transition: opacity 0.15s, background 0.15s;
  line-height: 1;
}
.thumb-card:hover .thumb-rm { opacity: 1; }
.thumb-rm:hover { background: rgba(248, 113, 113, 0.9); color: #000; }

/* 우측 빈 상태 */
.grid-empty {
  width: 100%; height: 100%;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 12px;
  color: var(--text-muted);
}
.grid-empty-ico { font-size: 56px; opacity: 0.3; line-height: 1; }
.grid-empty-title { font-size: 15px; font-weight: var(--fw-bold); color: var(--text-secondary); }
.grid-empty-sub { font-size: 12px; color: var(--text-muted); }
</style>
