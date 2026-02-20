import { defineStore } from 'pinia'
import { ref } from 'vue'
import axios from 'axios'

export const useAuthStore = defineStore('auth', () => {
  const user = ref(null)
  const checked = ref(false)

  async function fetchUser() {
    try {
      const res = await axios.get('/me', { withCredentials: true })
      user.value = res.data
    } catch {
      user.value = null
    } finally {
      checked.value = true
    }
  }

  async function logout() {
    await axios.post('/auth/logout', {}, { withCredentials: true })
    user.value = null
  }

  return { user, checked, fetchUser, logout }
})
