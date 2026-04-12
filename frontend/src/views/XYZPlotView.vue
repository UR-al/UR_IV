<template>
  <div class="xyz-view">
    <div class="config-panel">
      <h3>XYZ Plot</h3>
      <div v-for="(axis, ai) in axes" :key="ai" class="axis-config">
        <label class="axis-label">{{ axis.name }} 축</label>
        <CustomSelect v-model="axis.type" :options="['', ...axisOptions]" placeholder="None" />
        <input v-if="axis.type" class="s-input" v-model="axis.values"
          :placeholder="axis.type === 'Prompt S/R' ? 'search, replace1, replace2' : '값1, 값2, 값3 또는 20-40:5'"
        />
      </div>
      <div class="combo-info">
        조합 수: <span class="accent">{{ comboCount }}</span>
      </div>
      <button class="btn-gen" @click="startPlot" :disabled="comboCount === 0">
        XYZ Plot 시작
      </button>
    </div>
    <div class="result-area">
      <div v-if="resultImages.length === 0" class="empty">
        축을 설정하고 시작하면 결과가 여기에 표시됩니다
      </div>
      <div v-else class="result-grid">
        <div v-for="(img, i) in resultImages" :key="i" class="result-item">
          <img :src="'file:///' + img.path" />
          <div class="result-label">{{ img.label }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed } from 'vue'
import { requestAction } from '../stores/widgetStore.js'
import CustomSelect from '../components/CustomSelect.vue'

const axisOptions = ['Prompt S/R', 'Negative S/R', 'Steps', 'CFG Scale', 'Sampler', 'Scheduler', 'Seed', 'Width', 'Height', 'Denoising']

const axes = reactive([
  { name: 'X', type: '', values: '' },
  { name: 'Y', type: '', values: '' },
  { name: 'Z', type: '', values: '' },
])
const resultImages = ref([])

const comboCount = computed(() => {
  let count = 1
  for (const a of axes) {
    if (a.type && a.values.trim()) {
      const vals = parseValues(a.values)
      if (vals.length > 0) count *= vals.length
    }
  }
  return axes.some(a => a.type && a.values.trim()) ? count : 0
})

function parseValues(str) {
  const s = str.trim()
  const m = s.match(/^(\d+)-(\d+):(\d+)$/)
  if (m) {
    const vals = []
    for (let v = parseInt(m[1]); v <= parseInt(m[2]); v += parseInt(m[3])) vals.push(String(v))
    return vals
  }
  return s.split(',').map(v => v.trim()).filter(Boolean)
}

async function startPlot() {
  const axisData = axes.filter(a => a.type && a.values.trim()).map(a => ({
    type: a.type, values: parseValues(a.values),
  }))
  // 조합 생성
  const backend = await getBackend()
  if (backend.generateXYZCombinations) {
    backend.generateXYZCombinations(JSON.stringify(axisData), (json) => {
      try {
        const data = JSON.parse(json)
        if (data.combinations) {
          // 대기열에 추가
          requestAction('start_xyz_plot', { axes: axisData, combinations: data.combinations })
        }
      } catch {}
    })
  }
}

import { getBackend } from '../bridge.js'
</script>

<style scoped>
.xyz-view { width: 100%; height: 100%; display: flex; }
.config-panel {
  width: 320px; padding: 16px; border-right: 1px solid #1A1A1A;
  display: flex; flex-direction: column; gap: 12px; overflow-y: auto;
}
.config-panel h3 { color: #E8E8E8; font-size: 14px; margin: 0; }
.axis-config { display: flex; flex-direction: column; gap: 4px; }
.axis-label { color: #E2B340; font-size: 12px; font-weight: 700; }
.s-select, .s-input {
  background: #131313; border: 1px solid #222; border-radius: 4px; padding: 6px 8px;
  color: #E8E8E8; font-size: 12px; outline: none; caret-color: #E2B340;
}
.s-input:focus { border-color: #E2B340; }
.s-input::selection { background: rgba(226, 179, 64, 0.3); }
.s-select:focus { border-color: #E2B340; }
.combo-info { color: #787878; font-size: 12px; text-align: center; }
.accent { color: #E2B340; font-weight: 700; }
.btn-gen {
  padding: 12px; background: #E2B340; border: none; border-radius: 6px;
  color: #000; font-weight: 700; cursor: pointer;
}
.btn-gen:disabled { opacity: 0.35; cursor: not-allowed; }
.result-area { flex: 1; overflow-y: auto; padding: 8px; }
.empty { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: #484848; font-size: 14px; }
.result-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 4px; align-content: start;
}
.result-item { border-radius: 4px; overflow: hidden; border: 1px solid #1A1A1A; }
.result-item img { width: 100%; aspect-ratio: 1; object-fit: cover; }
.result-label { font-size: 10px; color: #585858; padding: 4px 6px; text-align: center; }
</style>
