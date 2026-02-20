import { defineStore } from 'pinia'
import { ref } from 'vue'
import axios from 'axios'

export const useForecastStore = defineStore('forecast', () => {
  const result = ref(null)
  const loading = ref(false)
  const error = ref(null)

  async function train({ ticker, startDate, endDate, horizon }) {
    loading.value = true
    error.value = null
    result.value = null
    try {
      const res = await axios.post('/ml/train', { ticker, startDate, endDate, horizon }, { withCredentials: true })
      result.value = res.data
    } catch (err) {
      error.value = err.response?.data?.error || err.message
    } finally {
      loading.value = false
    }
  }

  function reset() {
    result.value = null
    error.value = null
  }

  return { result, loading, error, train, reset }
})
