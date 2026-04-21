<template>
  <div class="min-h-screen bg-tc-bg">
    <NavBar />

    <main class="max-w-[1200px] mx-auto px-4 py-6 space-y-px">

      <!-- Configure panel -->
      <div class="panel space-y-4">
        <div class="panel-header">
          <span class="panel-title">CONFIGURE FORECAST</span>
          <span class="panel-meta">WALK-FORWARD · 5-FOLD · TEMPORAL SPLIT · NO LOOKAHEAD</span>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          <!-- Ticker -->
          <div class="relative lg:col-span-1">
            <label class="label-xxs block mb-1">TICKER</label>
            <input
              v-model="tickerQuery"
              @input="onTickerInput"
              @blur="closeSuggestions"
              class="input-field uppercase"
              placeholder="AAPL, TSLA…"
              autocomplete="off"
            />
            <ul v-if="suggestions.length"
                class="absolute z-[100] mt-px w-full bg-tc-surface border border-tc-border-hi overflow-hidden"
                style="top:100%">
              <li v-for="s in suggestions" :key="s.ticker"
                  @mousedown.prevent="selectTicker(s)"
                  class="flex justify-between items-center px-3 py-2 hover:bg-tc-hover cursor-pointer">
                <span class="font-mono font-bold text-tc-cyan text-sm">{{ s.ticker }}</span>
                <span class="text-tc-dim text-xs truncate ml-2">{{ s.name }}</span>
              </li>
            </ul>
          </div>

          <!-- Time range -->
          <div>
            <label class="label-xxs block mb-1">TIME RANGE</label>
            <select v-model="timeRange" class="input-field">
              <option value="1">1 YEAR</option>
              <option value="3">3 YEARS</option>
              <option value="5">5 YEARS</option>
            </select>
          </div>

          <!-- Horizon -->
          <div>
            <label class="label-xxs block mb-1">FORECAST HORIZON</label>
            <select v-model="horizon" class="input-field">
              <option :value="1">1 DAY</option>
              <option :value="5">5 DAYS</option>
              <option :value="20">20 DAYS</option>
            </select>
          </div>

          <!-- Train -->
          <div class="flex items-end">
            <button @click="trainModel" :disabled="!selectedTicker || store.loading" class="btn-primary">
              <span v-if="store.loading" class="flex items-center justify-center gap-2">
                <span class="blink text-base leading-none">▌</span>
                TRAINING...
              </span>
              <span v-else>RUN MODEL</span>
            </button>
          </div>
        </div>

        <p v-if="selectedTicker" class="label-xxs text-tc-dim">
          {{ selectedTicker }} · {{ dateRange.start }} → {{ dateRange.end }} · {{ horizon }}-DAY HORIZON
        </p>
      </div>

      <!-- Error -->
      <div v-if="store.error"
           class="panel border-tc-red"
           style="border-color: var(--accent-red)">
        <span class="label-xxs text-tc-red">ERR ·</span>
        <span class="text-sm text-tc-text ml-2">{{ store.error }}</span>
      </div>

      <!-- Loading -->
      <div v-if="store.loading" class="panel py-8 flex flex-col items-center gap-3">
        <div class="flex items-center gap-3">
          <span class="text-tc-cyan text-xl blink">▌</span>
          <span class="label-xs text-tc-meta">TRAINING MODEL...</span>
        </div>
        <span class="label-xxs text-tc-dim">WALK-FORWARD · 5-FOLD · TEMPORAL SPLIT</span>
      </div>

      <!-- Results -->
      <template v-if="store.result && !store.loading">

        <!-- Metrics row -->
        <div class="grid grid-cols-3 border border-tc-border">
          <div class="p-5">
            <div class="metric__label">MAPE</div>
            <div class="metric__value">{{ (store.result.metrics.mape * 100).toFixed(2) }}%</div>
            <div class="metric__delta">MEAN ABS % ERROR</div>
          </div>
          <div class="p-5 border-l border-tc-border">
            <div class="metric__label">RMSE</div>
            <div class="metric__value">{{ store.result.metrics.rmse.toFixed(4) }}</div>
            <div class="metric__delta">RETURN BASIS</div>
          </div>
          <div class="p-5 border-l border-tc-border">
            <div class="metric__label">DIR. ACC</div>
            <div class="metric__value"
                 :style="{ color: store.result.metrics.dirAcc >= 0.5 ? 'var(--accent-green)' : 'var(--accent-red)' }">
              {{ (store.result.metrics.dirAcc * 100).toFixed(1) }}%
            </div>
            <div class="metric__delta"
                 :style="{ color: store.result.metrics.dirAcc >= 0.5 ? 'var(--accent-green)' : 'var(--accent-red)' }">
              {{ ((store.result.metrics.dirAcc - 0.5) * 100 >= 0 ? '+' : '') }}{{ ((store.result.metrics.dirAcc - 0.5) * 100).toFixed(1) }}PP VS RANDOM
            </div>
          </div>
        </div>

        <!-- Backtest chart -->
        <div class="panel">
          <div class="panel-header">
            <span class="panel-title">BACKTEST — ACTUAL VS PREDICTED</span>
            <span class="panel-meta">WALK-FORWARD 5-FOLD · NO LOOKAHEAD</span>
          </div>
          <div class="chart-legend">
            <span class="chart-legend__item">
              <span class="chart-legend__line" style="background:#e0e0e0"></span>ACTUAL
            </span>
            <span class="chart-legend__item">
              <span class="chart-legend__line" style="background:#00e5ff"></span>PREDICTED
            </span>
          </div>
          <div class="h-72">
            <BacktestChart :predictions="store.result.predictions" />
          </div>
        </div>

        <!-- SHAP feature importance -->
        <div v-if="store.result.featureImportance?.length" class="panel">
          <div class="panel-header">
            <span class="panel-title">FEATURE IMPORTANCE — SHAP</span>
            <span class="panel-meta">MEAN |SHAP| · TOP 10 · NORMALISED</span>
          </div>
          <div class="h-72">
            <FeatureImportanceChart :importance="store.result.featureImportance" />
          </div>
        </div>

        <!-- Stock similarity -->
        <div class="panel">
          <div class="panel-header">
            <div>
              <span class="panel-title">SIMILAR STOCKS</span>
              <div class="panel-meta mt-1">PCA · 34-FEATURE BEHAVIOURAL PROFILES · COSINE SIMILARITY · 20-TICKER UNIVERSE</div>
            </div>
            <span v-if="similarityLoading" class="label-xxs text-tc-dim blink">COMPUTING...</span>
          </div>

          <div v-if="similarityData" class="flex flex-wrap gap-2 mb-4">
            <div v-for="s in similarityData.similar" :key="s.ticker"
                 class="flex items-center gap-2 border border-tc-border bg-tc-surface px-3 py-1">
              <span class="font-mono font-bold text-tc-purple text-sm">{{ s.ticker }}</span>
              <span class="label-xxs text-tc-dim">{{ (s.similarity * 100).toFixed(1) }}%</span>
            </div>
          </div>

          <div class="h-80">
            <SimilarityChart v-if="similarityData" :data="similarityData" />
            <div v-else-if="!similarityLoading && similarityError"
                 class="h-full flex items-center justify-center label-xs text-tc-dim">
              {{ similarityError }}
            </div>
            <div v-else class="h-full bg-tc-surface animate-pulse" />
          </div>
        </div>

        <!-- Next forecast -->
        <div class="panel">
          <div class="panel-header">
            <span class="panel-title">NEXT {{ horizon }}-DAY FORECAST — {{ selectedTicker }}</span>
          </div>
          <div class="h-48">
            <ForecastChart :data="store.result.nextForecast" />
          </div>
          <table class="w-full mt-4 text-sm">
            <thead>
              <tr class="border-b border-tc-border">
                <th class="pb-2 text-left label-xxs">DATE</th>
                <th class="pb-2 text-left label-xxs">PREDICTED PRICE</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(row, i) in store.result.nextForecast" :key="i"
                  class="border-b border-tc-border hover:bg-tc-hover transition-colors">
                <td class="py-2 text-tc-meta">{{ row.date }}</td>
                <td class="py-2 font-display font-bold text-tc-cyan">${{ row.y_pred.toFixed(2) }}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Footer -->
        <div class="py-3 text-center label-xxs text-tc-dim">
          {{ store.result.dataPoints }} DATA POINTS · {{ store.result.featureCount }} FEATURES · ID {{ store.result.forecastId }}
        </div>

      </template>

    </main>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useForecastStore } from '../stores/forecast'
import NavBar from '../components/NavBar.vue'
import BacktestChart from '../components/BacktestChart.vue'
import ForecastChart from '../components/ForecastChart.vue'
import FeatureImportanceChart from '../components/FeatureImportanceChart.vue'
import SimilarityChart from '../components/SimilarityChart.vue'
import axios from 'axios'

const store = useForecastStore()

const tickerQuery = ref('')
const selectedTicker = ref('')
const suggestions = ref([])
const timeRange = ref('5')
const horizon = ref(5)

const similarityData = ref(null)
const similarityLoading = ref(false)
const similarityError = ref(null)

let searchTimeout = null

function onTickerInput() {
  selectedTicker.value = ''
  clearTimeout(searchTimeout)
  searchTimeout = setTimeout(async () => {
    if (tickerQuery.value.length < 1) { suggestions.value = []; return }
    try {
      const res = await axios.get('/stocks/search', { params: { q: tickerQuery.value }, withCredentials: true })
      suggestions.value = res.data
    } catch { suggestions.value = [] }
  }, 250)
}

function selectTicker(s) {
  tickerQuery.value = s.ticker
  selectedTicker.value = s.ticker
  suggestions.value = []
}

function closeSuggestions() {
  setTimeout(() => { suggestions.value = [] }, 150)
  if (tickerQuery.value && !selectedTicker.value) {
    selectedTicker.value = tickerQuery.value.toUpperCase()
  }
}

const dateRange = computed(() => {
  const end = new Date()
  const start = new Date()
  start.setFullYear(end.getFullYear() - Number(timeRange.value))
  const fmt = d => d.toISOString().split('T')[0]
  return { start: fmt(start), end: fmt(end) }
})

async function fetchSimilarity(ticker) {
  similarityData.value = null
  similarityError.value = null
  similarityLoading.value = true
  try {
    const res = await axios.get(`/stocks/${ticker}/similarity`, { withCredentials: true })
    similarityData.value = res.data
  } catch (err) {
    similarityError.value = err.response?.data?.error || 'SIMILARITY UNAVAILABLE'
  } finally {
    similarityLoading.value = false
  }
}

async function trainModel() {
  if (!selectedTicker.value) return
  await store.train({
    ticker: selectedTicker.value,
    startDate: dateRange.value.start,
    endDate: dateRange.value.end,
    horizon: Number(horizon.value)
  })
  if (store.result) fetchSimilarity(selectedTicker.value)
}
</script>
