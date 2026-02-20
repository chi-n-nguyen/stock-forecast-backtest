<template>
  <Line :data="chartData" :options="chartOptions" />
</template>

<script setup>
import { computed } from 'vue'
import { Line } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler)

const props = defineProps({
  predictions: { type: Array, required: true }
})

// Downsample to max 200 points for rendering performance
const sampled = computed(() => {
  const p = props.predictions
  if (p.length <= 200) return p
  const step = Math.floor(p.length / 200)
  return p.filter((_, i) => i % step === 0)
})

const chartData = computed(() => ({
  labels: sampled.value.map(p => p.date),
  datasets: [
    {
      label: 'Actual',
      data: sampled.value.map(p => p.y_true),
      borderColor: '#94a3b8',
      backgroundColor: 'transparent',
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.2
    },
    {
      label: 'Predicted',
      data: sampled.value.map(p => p.y_pred),
      borderColor: '#38bdf8',
      backgroundColor: 'rgba(56,189,248,0.08)',
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.2,
      fill: false
    }
  ]
}))

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: 'index', intersect: false },
  plugins: {
    legend: {
      labels: { color: '#9ca3af', font: { size: 11 }, boxWidth: 16 }
    },
    tooltip: {
      backgroundColor: '#1f2937',
      titleColor: '#e5e7eb',
      bodyColor: '#d1d5db',
      callbacks: {
        label: ctx => `${ctx.dataset.label}: $${ctx.parsed.y?.toFixed(2)}`
      }
    }
  },
  scales: {
    x: {
      ticks: { color: '#6b7280', maxTicksLimit: 8, maxRotation: 0 },
      grid: { color: '#1f2937' }
    },
    y: {
      ticks: { color: '#6b7280', callback: v => `$${v.toFixed(0)}` },
      grid: { color: '#1f2937' }
    }
  }
}
</script>
