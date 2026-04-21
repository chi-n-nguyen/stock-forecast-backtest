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
      borderColor: '#e0e0e0',
      backgroundColor: 'transparent',
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.1
    },
    {
      label: 'Predicted',
      data: sampled.value.map(p => p.y_pred),
      borderColor: '#00e5ff',
      backgroundColor: 'transparent',
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.1
    }
  ]
}))

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: 'index', intersect: false },
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: '#1a1a1a',
      borderColor: '#444444',
      borderWidth: 1,
      titleColor: '#888888',
      bodyColor: '#e0e0e0',
      padding: 10,
      cornerRadius: 0,
      titleFont: { family: "'IBM Plex Mono', monospace", size: 10 },
      bodyFont: { family: "'IBM Plex Mono', monospace", size: 11 },
      callbacks: {
        label: ctx => `${ctx.dataset.label.toUpperCase()}: $${ctx.parsed.y?.toFixed(2)}`
      }
    }
  },
  scales: {
    x: {
      ticks: { color: '#555555', maxTicksLimit: 8, maxRotation: 0, font: { size: 10 } },
      grid: { color: '#1e1e1e', drawBorder: false },
      border: { color: '#2a2a2a' }
    },
    y: {
      ticks: { color: '#555555', callback: v => `$${v.toFixed(0)}`, font: { size: 10 } },
      grid: { color: '#1e1e1e', drawBorder: false },
      border: { color: '#2a2a2a' }
    }
  }
}
</script>
