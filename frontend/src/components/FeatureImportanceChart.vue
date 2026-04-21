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
  Title,
  Tooltip,
  Legend
} from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

const props = defineProps({
  importance: { type: Array, required: true }
})

// Top 10 features, already sorted descending by importance from backend
const top10 = computed(() => props.importance.slice(0, 10))

const chartData = computed(() => ({
  labels: top10.value.map(d => d.feature),
  datasets: [
    {
      label: 'Mean |SHAP|',
      data: top10.value.map(d => d.importance),
      backgroundColor: top10.value.map((_, i) =>
        i === 0 ? '#38bdf8' : i < 3 ? '#818cf8' : '#334155'
      ),
      borderRadius: 4,
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
      backgroundColor: '#1f2937',
      titleColor: '#e5e7eb',
      bodyColor: '#d1d5db',
      callbacks: {
        label: ctx => `${(ctx.parsed.x * 100).toFixed(2)}% of total SHAP`
      }
    }
  },
  scales: {
    x: {
      ticks: {
        color: '#6b7280',
        callback: v => `${(v * 100).toFixed(1)}%`
      },
      grid: { color: '#1f2937' }
    },
    y: {
      ticks: { color: '#9ca3af', font: { size: 11 } },
      grid: { display: false }
    }
  }
}
</script>
