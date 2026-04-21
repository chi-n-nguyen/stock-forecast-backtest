<template>
  <div class="min-h-screen bg-tc-bg">
    <NavBar />

    <main class="max-w-[1200px] mx-auto px-4 py-6 space-y-px">

      <div class="panel">
        <div class="panel-header">
          <span class="panel-title">FORECAST HISTORY</span>
        </div>

        <!-- Loading -->
        <div v-if="loading" class="space-y-px">
          <div v-for="i in 4" :key="i" class="h-14 bg-tc-surface animate-pulse" />
        </div>

        <!-- Error -->
        <div v-else-if="error" class="label-xs text-tc-red py-4">{{ error }}</div>

        <!-- Empty -->
        <div v-else-if="forecasts.length === 0" class="py-12 text-center">
          <span class="label-xs text-tc-dim">NO FORECASTS YET ·
            <router-link to="/" class="text-tc-cyan hover:underline">RUN A MODEL</router-link>
          </span>
        </div>

        <!-- List -->
        <div v-else class="space-y-px">
          <div v-for="f in forecasts" :key="f._id"
               class="flex items-center justify-between px-3 py-3 bg-tc-surface border border-tc-border hover:bg-tc-hover hover:border-tc-border-hi transition-colors cursor-pointer"
               @click="loadDetail(f._id)">
            <div class="flex items-center gap-5">
              <span class="font-display font-bold text-tc-cyan text-base tracking-widest">{{ f.ticker }}</span>
              <span class="label-xxs text-tc-dim hidden sm:inline">{{ f.startDate }} → {{ f.endDate }}</span>
              <span class="label-xxs border border-tc-border px-2 py-0.5 text-tc-meta">{{ f.horizon }}D</span>
            </div>
            <div class="flex items-center gap-8">
              <div class="text-right">
                <div class="label-xxs">MAPE</div>
                <div class="font-display font-bold text-tc-text text-sm">{{ (f.metrics.mape * 100).toFixed(2) }}%</div>
              </div>
              <div class="text-right">
                <div class="label-xxs">DIR. ACC</div>
                <div class="font-display font-bold text-sm"
                     :style="{ color: f.metrics.dirAcc >= 0.5 ? 'var(--accent-green)' : 'var(--accent-red)' }">
                  {{ (f.metrics.dirAcc * 100).toFixed(1) }}%
                </div>
              </div>
              <span class="label-xxs text-tc-dim hidden md:inline">{{ new Date(f.createdAt).toLocaleDateString() }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Detail modal -->
      <div v-if="selected"
           class="fixed inset-0 flex items-center justify-center z-[300] p-4"
           style="background: rgba(0,0,0,0.75)"
           @click.self="selected = null">
        <div class="panel w-full max-w-3xl max-h-[90vh] overflow-y-auto space-y-5"
             style="border-color: var(--border-active)">

          <div class="flex items-center justify-between">
            <span class="panel-title">
              {{ selected.ticker }} · {{ selected.horizon }}-DAY FORECAST
            </span>
            <button @click="selected = null" class="text-tc-dim hover:text-tc-text text-xl leading-none">×</button>
          </div>

          <!-- Metrics -->
          <div class="grid grid-cols-3 border border-tc-border">
            <div class="p-4">
              <div class="metric__label">MAPE</div>
              <div class="metric__value text-lg">{{ (selected.metrics.mape * 100).toFixed(2) }}%</div>
            </div>
            <div class="p-4 border-l border-tc-border">
              <div class="metric__label">RMSE</div>
              <div class="metric__value text-lg">{{ selected.metrics.rmse.toFixed(4) }}</div>
            </div>
            <div class="p-4 border-l border-tc-border">
              <div class="metric__label">DIR. ACC</div>
              <div class="metric__value text-lg"
                   :style="{ color: selected.metrics.dirAcc >= 0.5 ? 'var(--accent-green)' : 'var(--accent-red)' }">
                {{ (selected.metrics.dirAcc * 100).toFixed(1) }}%
              </div>
              <div class="metric__delta"
                   :style="{ color: selected.metrics.dirAcc >= 0.5 ? 'var(--accent-green)' : 'var(--accent-red)' }">
                {{ ((selected.metrics.dirAcc - 0.5) * 100 >= 0 ? '+' : '') }}{{ ((selected.metrics.dirAcc - 0.5) * 100).toFixed(1) }}PP VS RANDOM
              </div>
            </div>
          </div>

          <!-- Chart -->
          <div>
            <div class="chart-legend mb-2">
              <span class="chart-legend__item">
                <span class="chart-legend__line" style="background:#e0e0e0"></span>ACTUAL
              </span>
              <span class="chart-legend__item">
                <span class="chart-legend__line" style="background:#00e5ff"></span>PREDICTED
              </span>
            </div>
            <div class="h-60">
              <BacktestChart :predictions="selected.predictions" />
            </div>
          </div>

          <!-- Next forecast table -->
          <div>
            <div class="label-xs text-tc-meta mb-2">NEXT FORECAST</div>
            <table class="w-full text-sm">
              <thead>
                <tr class="border-b border-tc-border">
                  <th class="pb-2 text-left label-xxs">DATE</th>
                  <th class="pb-2 text-left label-xxs">PREDICTED</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(r, i) in selected.nextForecast" :key="i"
                    class="border-b border-tc-border hover:bg-tc-hover transition-colors">
                  <td class="py-1.5 text-tc-meta">{{ r.date }}</td>
                  <td class="py-1.5 font-display font-bold text-tc-cyan">${{ r.y_pred.toFixed(2) }}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div class="label-xxs text-tc-dim">
            ID {{ selected._id }} · {{ new Date(selected.createdAt).toLocaleString() }}
          </div>

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
