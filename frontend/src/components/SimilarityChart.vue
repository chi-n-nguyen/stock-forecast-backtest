<template>
  <Scatter :data="chartData" :options="chartOptions" :plugins="[labelPlugin]" />
</template>

<script setup>
import { computed } from 'vue'
import { Scatter } from 'vue-chartjs'
import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  Tooltip,
  Legend
} from 'chart.js'

ChartJS.register(LinearScale, PointElement, Tooltip, Legend)

const props = defineProps({
  data: { type: Object, required: true }
})

// Set of top-5 similar ticker strings for fast lookup
const similarSet = computed(() => new Set(props.data.similar.map(s => s.ticker)))

const chartData = computed(() => {
  const all = props.data.all_tickers
  const query = props.data.query
  const top5 = props.data.similar

  const universe = all.filter(t => t.ticker !== query.ticker && !similarSet.value.has(t.ticker))

  return {
    datasets: [
      {
        label: 'Universe',
        data: universe.map(t => ({ x: t.x, y: t.y, ticker: t.ticker })),
        backgroundColor: '#334155',
        pointRadius: 6,
        pointHoverRadius: 8
      },
      {
        label: 'Similar',
        data: top5.map(t => ({ x: t.x, y: t.y, ticker: t.ticker, similarity: t.similarity })),
        backgroundColor: '#818cf8',
        pointRadius: 8,
        pointHoverRadius: 10
      },
      {
        label: props.data.ticker,
        data: [{ x: query.x, y: query.y, ticker: query.ticker }],
        backgroundColor: '#38bdf8',
        pointRadius: 11,
        pointHoverRadius: 13
      }
    ]
  }
})

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      labels: { color: '#9ca3af', font: { size: 11 }, boxWidth: 12 }
    },
    tooltip: {
      backgroundColor: '#1f2937',
      titleColor: '#e5e7eb',
      bodyColor: '#d1d5db',
      callbacks: {
        label: ctx => {
          const pt = ctx.raw
          const sim = pt.similarity != null ? ` · similarity ${(pt.similarity * 100).toFixed(1)}%` : ''
          return `${pt.ticker}${sim}`
        }
      }
    }
  },
  scales: {
    x: {
      ticks: { color: '#4b5563', maxTicksLimit: 5 },
      grid: { color: '#1f2937' },
      title: { display: true, text: 'PC 1', color: '#6b7280', font: { size: 11 } }
    },
    y: {
      ticks: { color: '#4b5563', maxTicksLimit: 5 },
      grid: { color: '#1f2937' },
      title: { display: true, text: 'PC 2', color: '#6b7280', font: { size: 11 } }
    }
  }
}

// Inline Chart.js plugin that draws ticker labels next to each point
const labelPlugin = {
  id: 'tickerLabels',
  afterDatasetsDraw(chart) {
    const ctx = chart.ctx
    ctx.save()
    ctx.font = '11px ui-monospace, monospace'
    ctx.textBaseline = 'middle'

    chart.data.datasets.forEach((dataset, di) => {
      const meta = chart.getDatasetMeta(di)
      meta.data.forEach((point, pi) => {
        const raw = dataset.data[pi]
        if (!raw?.ticker) return
        const isQuery = di === 2
        const isSimilar = di === 1
        ctx.fillStyle = isQuery ? '#38bdf8' : isSimilar ? '#a5b4fc' : '#6b7280'
        ctx.fillText(raw.ticker, point.x + 9, point.y)
      })
    })
    ctx.restore()
  }
}
</script>
