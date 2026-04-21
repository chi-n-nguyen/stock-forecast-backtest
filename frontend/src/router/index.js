import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'Forecast',
    component: () => import('../views/ForecastView.vue')
  },
  {
    path: '/history',
    name: 'History',
    component: () => import('../views/HistoryView.vue')
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
