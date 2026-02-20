import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/auth': 'http://localhost:3000',
      '/stocks': 'http://localhost:3000',
      '/ml': 'http://localhost:3000',
      '/forecasts': 'http://localhost:3000',
      '/me': 'http://localhost:3000'
    }
  }
})
