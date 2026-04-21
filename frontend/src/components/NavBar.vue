<template>
  <nav class="border-b border-gray-800 bg-gray-950/80 backdrop-blur sticky top-0 z-40">
    <div class="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
      <router-link to="/" class="text-sky-400 font-bold text-lg tracking-tight">StockCast</router-link>

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
    </div>
  </nav>
</template>

<script setup>
import { useRoute } from 'vue-router'
const route = useRoute()
</script>
