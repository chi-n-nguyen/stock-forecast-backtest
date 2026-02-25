<template>
  <div class="min-h-screen bg-gray-950">
    <NavBar />

    <main class="max-w-4xl mx-auto px-4 py-8 space-y-6">
      <h1 class="text-xl font-semibold text-gray-200">Forecast History</h1>

      <!-- Loading -->
      <div v-if="loading" class="space-y-3">
        <div v-for="i in 4" :key="i" class="card h-16 animate-pulse bg-gray-800" />
      </div>

      <!-- Error -->
      <div v-else-if="error" class="text-red-400 text-sm">{{ error }}</div>

      <!-- Empty -->
      <div v-else-if="forecasts.length === 0" class="card text-center py-12 text-gray-500">
        No forecasts yet. Go to <router-link to="/" class="text-sky-400 hover:underline">Forecast</router-link> to train your first model.
      </div>

      <!-- List -->
      <div v-else class="space-y-3">
        <div v-for="f in forecasts" :key="f._id"
             class="card hover:border-gray-700 transition-colors cursor-pointer"
             @click="loadDetail(f._id)">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-4">
              <span class="text-lg font-bold text-sky-400">{{ f.ticker }}</span>
              <span class="text-sm text-gray-400">{{ f.startDate }} → {{ f.endDate }}</span>
              <span class="text-xs bg-gray-800 text-gray-300 px-2 py-0.5 rounded">{{ f.horizon }}D horizon</span>
            </div>
            <div class="flex gap-6 text-sm text-right">
              <div>
                <div class="text-xs text-gray-500">MAPE</div>
                <div class="font-semibold text-gray-200">{{ (f.metrics.mape * 100).toFixed(2) }}%</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Dir. Acc</div>
                <div class="font-semibold" :class="f.metrics.dirAcc >= 0.5 ? 'text-emerald-400' : 'text-red-400'">
                  {{ (f.metrics.dirAcc * 100).toFixed(1) }}%
                </div>
              </div>
              <div class="hidden sm:block text-xs text-gray-600 self-end">
                {{ new Date(f.createdAt).toLocaleDateString() }}
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Detail Modal -->
      <div v-if="selected" class="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4"
           @click.self="selected = null">
        <div class="bg-gray-900 border border-gray-700 rounded-2xl w-full max-w-3xl max-h-[90vh] overflow-y-auto p-6 space-y-6">
          <div class="flex items-center justify-between">
            <h2 class="text-lg font-semibold">
              {{ selected.ticker }} · {{ selected.horizon }}-Day Forecast
            </h2>
            <button @click="selected = null" class="text-gray-400 hover:text-white text-xl leading-none">&times;</button>
          </div>

          <!-- Metrics -->
          <div class="grid grid-cols-3 gap-4">
            <div class="metric-tile">
              <span class="text-xs text-gray-400">MAPE</span>
              <span class="text-xl font-bold text-sky-400">{{ (selected.metrics.mape * 100).toFixed(2) }}%</span>
            </div>
            <div class="metric-tile">
              <span class="text-xs text-gray-400">RMSE</span>
              <span class="text-xl font-bold text-emerald-400">{{ selected.metrics.rmse.toFixed(4) }}</span>
            </div>
            <div class="metric-tile">
              <span class="text-xs text-gray-400">Dir. Acc</span>
              <span class="text-xl font-bold" :class="selected.metrics.dirAcc >= 0.5 ? 'text-emerald-400' : 'text-red-400'">
                {{ (selected.metrics.dirAcc * 100).toFixed(1) }}%
              </span>
              <span class="text-xs text-gray-500">
                +{{ ((selected.metrics.dirAcc - 0.5) * 100).toFixed(1) }}pp vs random
              </span>
            </div>
          </div>

          <!-- Backtest chart -->
          <div class="h-64">
            <BacktestChart :predictions="selected.predictions" />
          </div>

          <!-- Next forecast table -->
          <div>
            <h3 class="font-medium text-gray-300 mb-2">Next Forecast</h3>
            <table class="w-full text-sm">
              <thead>
                <tr class="text-gray-400 border-b border-gray-800">
                  <th class="pb-2 text-left">Date</th>
                  <th class="pb-2 text-left">Predicted</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(r, i) in selected.nextForecast" :key="i" class="border-b border-gray-800/40">
                  <td class="py-1.5 text-gray-400">{{ r.date }}</td>
                  <td class="py-1.5 text-sky-300 font-semibold">${{ r.y_pred.toFixed(2) }}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <p class="text-xs text-gray-600">
            ID: {{ selected._id }} · {{ new Date(selected.createdAt).toLocaleString() }}
          </p>
        </div>
      </div>

    </main>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import NavBar from '../components/NavBar.vue'
import BacktestChart from '../components/BacktestChart.vue'

const forecasts = ref([])
const loading = ref(true)
const error = ref(null)
const selected = ref(null)

onMounted(async () => {
  try {
    const res = await axios.get('/forecasts', { withCredentials: true })
    forecasts.value = res.data
  } catch (err) {
    error.value = err.response?.data?.error || err.message
  } finally {
    loading.value = false
  }
})

async function loadDetail(id) {
  try {
    const res = await axios.get(`/forecasts/${id}`, { withCredentials: true })
    selected.value = res.data
  } catch (err) {
    error.value = err.response?.data?.error || err.message
  }
}
</script>
