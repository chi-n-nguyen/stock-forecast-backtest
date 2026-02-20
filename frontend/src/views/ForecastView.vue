<template>
  <div class="min-h-screen bg-gray-950">
    <!-- Nav -->
    <NavBar />

    <main class="max-w-6xl mx-auto px-4 py-8 space-y-8">

      <!-- Stock Selector Card -->
      <div class="card space-y-5">
        <h2 class="text-lg font-semibold text-gray-200">Configure Forecast</h2>

        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <!-- Ticker search -->
          <div class="relative lg:col-span-1">
            <label class="block text-xs text-gray-400 mb-1">Ticker</label>
            <input
              v-model="tickerQuery"
              @input="onTickerInput"
              @blur="closeSuggestions"
              class="input-field uppercase"
              placeholder="AAPL, TSLA…"
              autocomplete="off"
            />
            <!-- Suggestions dropdown -->
            <ul v-if="suggestions.length"
                class="absolute z-10 mt-1 w-full bg-gray-800 border border-gray-700 rounded-lg overflow-hidden shadow-lg">
              <li v-for="s in suggestions" :key="s.ticker"
                  @mousedown.prevent="selectTicker(s)"
                  class="flex justify-between items-center px-3 py-2 hover:bg-gray-700 cursor-pointer text-sm">
                <span class="font-semibold text-sky-400">{{ s.ticker }}</span>
                <span class="text-gray-400 truncate ml-2 text-xs">{{ s.name }}</span>
              </li>
            </ul>
          </div>

          <!-- Time range -->
          <div>
            <label class="block text-xs text-gray-400 mb-1">Time Range</label>
            <select v-model="timeRange" class="input-field">
              <option value="1">1 Year</option>
              <option value="3">3 Years</option>
              <option value="5">5 Years</option>
            </select>
          </div>

          <!-- Horizon -->
          <div>
            <label class="block text-xs text-gray-400 mb-1">Forecast Horizon</label>
            <select v-model="horizon" class="input-field">
              <option :value="1">1 Day</option>
              <option :value="5">5 Days</option>
              <option :value="20">20 Days</option>
            </select>
          </div>

          <!-- Train button -->
          <div class="flex items-end">
            <button @click="trainModel" :disabled="!selectedTicker || store.loading" class="btn-primary w-full">
              <span v-if="store.loading" class="flex items-center justify-center gap-2">
                <svg class="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
                </svg>
                Training…
              </span>
              <span v-else>Train Model</span>
            </button>
          </div>
        </div>

        <!-- Training config summary -->
        <p v-if="selectedTicker" class="text-xs text-gray-500">
          {{ selectedTicker }} &bull; {{ dateRange.start }} → {{ dateRange.end }} &bull; {{ horizon }}-day horizon
        </p>
      </div>

      <!-- Error -->
      <div v-if="store.error" class="bg-red-900/40 border border-red-700 rounded-lg px-4 py-3 text-red-300 text-sm">
        {{ store.error }}
      </div>

      <!-- Loading skeleton -->
      <div v-if="store.loading" class="space-y-4">
        <div class="card h-16 animate-pulse bg-gray-800" />
        <div class="card h-72 animate-pulse bg-gray-800" />
      </div>

      <!-- Results -->
      <template v-if="store.result && !store.loading">
        <!-- Metrics row -->
        <div class="grid grid-cols-3 gap-4">
          <div class="metric-tile">
            <span class="text-xs text-gray-400 uppercase tracking-wide">MAPE</span>
            <span class="text-2xl font-bold text-sky-400">{{ (store.result.metrics.mape * 100).toFixed(2) }}%</span>
            <span class="text-xs text-gray-500">Mean Abs % Error</span>
          </div>
          <div class="metric-tile">
            <span class="text-xs text-gray-400 uppercase tracking-wide">RMSE</span>
            <span class="text-2xl font-bold text-emerald-400">${{ store.result.metrics.rmse.toFixed(2) }}</span>
            <span class="text-xs text-gray-500">Root Mean Sq Error</span>
          </div>
          <div class="metric-tile">
            <span class="text-xs text-gray-400 uppercase tracking-wide">Dir. Accuracy</span>
            <span class="text-2xl font-bold" :class="store.result.metrics.dirAcc >= 0.5 ? 'text-emerald-400' : 'text-red-400'">
              {{ (store.result.metrics.dirAcc * 100).toFixed(1) }}%
            </span>
            <span class="text-xs text-gray-500">Up/Down correct</span>
          </div>
        </div>

        <!-- Backtest Chart -->
        <div class="card">
          <div class="flex items-center justify-between mb-4">
            <h3 class="font-semibold text-gray-200">Backtest: Actual vs Predicted</h3>
            <span class="text-xs text-gray-500">Walk-forward 5-fold · No lookahead bias</span>
          </div>
          <div class="h-72">
            <BacktestChart :predictions="store.result.predictions" />
          </div>
        </div>

        <!-- Next Forecast -->
        <div class="card">
          <h3 class="font-semibold text-gray-200 mb-4">
            Next {{ horizon }}-Day Forecast — {{ selectedTicker }}
          </h3>
          <div class="h-52">
            <ForecastChart :data="store.result.nextForecast" />
          </div>
          <div class="mt-4 overflow-x-auto">
            <table class="w-full text-sm text-left">
              <thead>
                <tr class="text-gray-400 border-b border-gray-800">
                  <th class="pb-2 pr-6">Date</th>
                  <th class="pb-2">Predicted Price</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(row, i) in store.result.nextForecast" :key="i"
                    class="border-b border-gray-800/50 hover:bg-gray-800/30">
                  <td class="py-2 pr-6 text-gray-400">{{ row.date }}</td>
                  <td class="py-2 font-semibold text-sky-300">${{ row.y_pred.toFixed(2) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Info row -->
        <p class="text-xs text-gray-600 text-center">
          Model trained on {{ store.result.dataPoints }} data points · {{ store.result.featureCount }} features ·
          Forecast ID: {{ store.result.forecastId }}
        </p>
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
import axios from 'axios'

const store = useForecastStore()

const tickerQuery = ref('')
const selectedTicker = ref('')
const suggestions = ref([])
const timeRange = ref('5')
const horizon = ref(5)

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

async function trainModel() {
  if (!selectedTicker.value) return
  await store.train({
    ticker: selectedTicker.value,
    startDate: dateRange.value.start,
    endDate: dateRange.value.end,
    horizon: Number(horizon.value)
  })
}
</script>
