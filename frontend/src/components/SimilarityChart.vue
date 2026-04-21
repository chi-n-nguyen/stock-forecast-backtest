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
        backgroundColor: '#444444',
        pointRadius: 5,
        pointHoverRadius: 7
      },
      {
        label: 'Similar',
        data: top5.map(t => ({ x: t.x, y: t.y, ticker: t.ticker, similarity: t.similarity })),
        backgroundColor: '#b388ff',
        pointRadius: 7,
        pointHoverRadius: 9
      },
      {
        label: props.data.ticker,
        data: [{ x: query.x, y: query.y, ticker: query.ticker }],
        backgroundColor: '#00e5ff',
        pointRadius: 10,
        pointHoverRadius: 12
      }
    ]
  }
})

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      labels: {
        color: '#555555',
        font: { family: "'IBM Plex Mono', monospace", size: 10 },
        boxWidth: 8,
        padding: 16
      }
    },
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
        label: ctx => {
          const pt = ctx.raw
          const sim = pt.similarity != null ? ` · ${(pt.similarity * 100).toFixed(1)}% SIMILAR` : ''
          return `${pt.ticker}${sim}`
        }
      }
    }
  },
  scales: {
    x: {
      ticks: { color: '#555555', maxTicksLimit: 5, font: { size: 10 } },
      grid: { color: '#1e1e1e', drawBorder: false },
      border: { color: '#2a2a2a' },
      title: { display: true, text: 'PC 1', color: '#555555', font: { family: "'IBM Plex Mono', monospace", size: 10 } }
    },
    y: {
      ticks: { color: '#555555', maxTicksLimit: 5, font: { size: 10 } },
      grid: { color: '#1e1e1e', drawBorder: false },
      border: { color: '#2a2a2a' },
      title: { display: true, text: 'PC 2', color: '#555555', font: { family: "'IBM Plex Mono', monospace", size: 10 } }
    }
  }
}

const labelPlugin = {
  id: 'tickerLabels',
  afterDatasetsDraw(chart) {
    const ctx = chart.ctx
    ctx.save()
    ctx.font = "10px 'IBM Plex Mono', monospace"
    ctx.textBaseline = 'middle'

    chart.data.datasets.forEach((dataset, di) => {
      const meta = chart.getDatasetMeta(di)
      meta.data.forEach((point, pi) => {
        const raw = dataset.data[pi]
        if (!raw?.ticker) return
        ctx.fillStyle = di === 2 ? '#00e5ff' : di === 1 ? '#b388ff' : '#555555'
        ctx.fillText(raw.ticker, point.x + 9, point.y)
      })
    })
    ctx.restore()
  }
}
</script>
