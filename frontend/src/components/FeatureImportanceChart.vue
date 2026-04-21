<template>
  <Bar :data="chartData" :options="chartOptions" />
</template>

<script setup>
import { computed } from 'vue'
import { Bar } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend
} from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend)

const props = defineProps({
  importance: { type: Array, required: true }
})

const top10 = computed(() => props.importance.slice(0, 10))

const chartData = computed(() => ({
  labels: top10.value.map(d => d.feature),
  datasets: [
    {
      label: 'Mean |SHAP|',
      data: top10.value.map(d => d.importance),
      backgroundColor: '#00e5ff',
      borderRadius: 0,
      borderSkipped: false
    }
  ]
}))

const chartOptions = {
  indexAxis: 'y',
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
        label: ctx => `${(ctx.parsed.x * 100).toFixed(2)}% OF TOTAL SHAP`
      }
    }
  },
  scales: {
    x: {
      ticks: {
        color: '#555555',
        callback: v => `${(v * 100).toFixed(1)}%`,
        font: { size: 10 }
      },
      grid: { color: '#1e1e1e', drawBorder: false },
      border: { color: '#2a2a2a' }
    },
    y: {
      ticks: { color: '#888888', font: { size: 10 } },
      grid: { display: false },
      border: { color: '#2a2a2a' }
    }
  }
}
</script>
