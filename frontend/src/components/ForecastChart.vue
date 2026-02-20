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
  Tooltip,
  Legend,
  Filler
} from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler)

const props = defineProps({
  data: { type: Array, required: true }
})

const chartData = computed(() => ({
  labels: props.data.map(d => d.date),
  datasets: [
    {
      label: 'Forecast',
      data: props.data.map(d => d.y_pred),
      borderColor: '#f59e0b',
      backgroundColor: 'rgba(245,158,11,0.12)',
      borderWidth: 2,
      pointBackgroundColor: '#f59e0b',
      pointRadius: 4,
      tension: 0.3,
      fill: true
    }
  ]
}))

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      labels: { color: '#9ca3af', font: { size: 11 }, boxWidth: 16 }
    },
    tooltip: {
      backgroundColor: '#1f2937',
      titleColor: '#e5e7eb',
      bodyColor: '#d1d5db',
      callbacks: {
        label: ctx => `Predicted: $${ctx.parsed.y?.toFixed(2)}`
      }
    }
  },
  scales: {
    x: {
      ticks: { color: '#6b7280', maxRotation: 0 },
      grid: { color: '#1f2937' }
    },
    y: {
      ticks: { color: '#6b7280', callback: v => `$${v.toFixed(0)}` },
      grid: { color: '#1f2937' }
    }
  }
}
</script>
