<template>
  <nav class="bg-tc-bg border-b border-tc-border sticky top-0 z-40" style="height:48px">
    <div class="max-w-[1200px] mx-auto px-4 h-full flex items-center justify-between">

      <router-link to="/" class="font-display font-bold text-tc-cyan tracking-wider text-base">
        STOCKCAST
      </router-link>

      <div class="flex items-center gap-6">
        <router-link to="/"
          class="label-xs transition-colors"
          :class="route.path === '/' ? 'text-tc-cyan border-b-2 border-tc-cyan pb-px' : 'text-tc-meta hover:text-tc-text'">
          FORECAST
        </router-link>
        <router-link to="/history"
          class="label-xs transition-colors"
          :class="route.path === '/history' ? 'text-tc-cyan border-b-2 border-tc-cyan pb-px' : 'text-tc-meta hover:text-tc-text'">
          HISTORY
        </router-link>
      </div>

    </div>
  </nav>
</template>

<script setup>
import { useRoute } from 'vue-router'
const route = useRoute()
</script>
