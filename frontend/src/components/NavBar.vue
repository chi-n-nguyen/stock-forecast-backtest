<template>
  <nav class="border-b border-gray-800 bg-gray-950/80 backdrop-blur sticky top-0 z-40">
    <div class="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
      <!-- Logo -->
      <router-link to="/" class="text-sky-400 font-bold text-lg tracking-tight">StockCast</router-link>

      <!-- Links -->
      <div class="flex items-center gap-6">
        <router-link to="/"
          class="text-sm font-medium transition-colors"
          :class="route.path === '/' ? 'text-sky-400' : 'text-gray-400 hover:text-gray-200'">
          Forecast
        </router-link>
        <router-link to="/history"
          class="text-sm font-medium transition-colors"
          :class="route.path === '/history' ? 'text-sky-400' : 'text-gray-400 hover:text-gray-200'">
          History
        </router-link>
      </div>

      <!-- User -->
      <div class="flex items-center gap-3">
        <img v-if="auth.user?.avatar" :src="auth.user.avatar" class="w-7 h-7 rounded-full" />
        <span class="text-sm text-gray-400 hidden sm:inline">{{ auth.user?.name }}</span>
        <button @click="handleLogout" class="text-xs text-gray-500 hover:text-gray-300 transition-colors">
          Logout
        </button>
      </div>
    </div>
  </nav>
</template>

<script setup>
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()

async function handleLogout() {
  await auth.logout()
  router.push('/login')
}
</script>
