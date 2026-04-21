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
      borderColor: '#00e5ff',
      backgroundColor: 'rgba(0, 229, 255, 0.05)',
      borderWidth: 1.5,
      pointBackgroundColor: '#00e5ff',
      pointRadius: 3,
      pointHoverRadius: 5,
      tension: 0.1,
      fill: true
    }
  ]
}))

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
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
        label: ctx => `FORECAST: $${ctx.parsed.y?.toFixed(2)}`
      }
    }
  },
  scales: {
    x: {
      ticks: { color: '#555555', maxRotation: 0, font: { size: 10 } },
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
