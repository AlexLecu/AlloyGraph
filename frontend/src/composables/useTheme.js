import { ref, watch } from 'vue'

const STORAGE_KEY = 'alloygraph-theme'

function getSystemPreference() {
  if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
    return 'light'
  }
  return 'dark'
}

function getInitialTheme() {
  const stored = localStorage.getItem(STORAGE_KEY)
  if (stored === 'light' || stored === 'dark') return stored
  return getSystemPreference()
}

function applyTheme(value) {
  document.documentElement.setAttribute('data-theme', value)
}

// Shared singleton state
const theme = ref(
  document.documentElement.getAttribute('data-theme') || getInitialTheme()
)

// Register watcher once at module level (not per useTheme() call)
watch(theme, (val) => {
  applyTheme(val)
  localStorage.setItem(STORAGE_KEY, val)
})

export function useTheme() {
  const toggleTheme = () => {
    theme.value = theme.value === 'dark' ? 'light' : 'dark'
  }

  // Ensure DOM is in sync on first use
  applyTheme(theme.value)

  return { theme, toggleTheme }
}
