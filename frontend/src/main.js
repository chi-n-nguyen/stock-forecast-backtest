import { createApp } from 'vue'
import { createPinia } from 'pinia'
import { Chart } from 'chart.js'
import router from './router'
import App from './App.vue'
import './style.css'

Chart.defaults.font.family = "'IBM Plex Mono', monospace"
Chart.defaults.font.size = 11
Chart.defaults.color = '#555555'

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.mount('#app')
