<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import ResearchChat from './components/ResearchChat.vue'
import AlloyDesigner from './components/AlloyDesigner.vue'
import axios from 'axios'
import { API_BASE_URL } from './config'

const activeTab = ref('chat')
const isBackendOnline = ref(false)
const designContext = ref(null) // Stores alloy data for design handoff
let healthCheckInterval = null

const handleDesign = (alloy) => {
  designContext.value = alloy
  activeTab.value = 'design'
}

// Check backend health
const checkBackendHealth = async () => {
  try {
    await axios.get(`${API_BASE_URL}/health`, { timeout: 2000 })
    isBackendOnline.value = true
  } catch (error) {
    isBackendOnline.value = false
  }
}

// Start health checks on mount
onMounted(() => {
  checkBackendHealth()
  healthCheckInterval = setInterval(checkBackendHealth, 5000) // Check every 5s
})

// Cleanup on unmount
onUnmounted(() => {
  if (healthCheckInterval) {
    clearInterval(healthCheckInterval)
  }
})
</script>

<template>
  <div class="app-container">
    <!-- Modern Header -->
    <header class="glass-header">
      <div class="header-content">
        <div class="logo">
          <span class="logo-icon">🧬</span>
          <h1>AlloyGraph</h1>
        </div>
        <div :class="['status-badge', { offline: !isBackendOnline }]">
          <span class="status-dot"></span>
          {{ isBackendOnline ? 'Backend Online' : 'Backend Offline' }}
        </div>
      </div>
    </header>

    <!-- Modern Tab Navigation -->
    <nav class="tab-nav">
      <button 
        :class="['tab-button', { active: activeTab === 'chat' }]" 
        @click="activeTab = 'chat'"
      >
        <span class="tab-icon">🔬</span>
        <span class="tab-label">Research</span>
      </button>
      <button 
        :class="['tab-button', { active: activeTab === 'design' }]" 
        @click="activeTab = 'design'"
      >
        <span class="tab-icon">🧪</span>
        <span class="tab-label">Evaluate & Design</span>
      </button>
    </nav>

    <!-- Content Area -->
    <main class="content-area">
      <transition name="fade" mode="out-in">
        <KeepAlive>
          <component 
            :is="activeTab === 'chat' ? ResearchChat : AlloyDesigner"
            v-bind="activeTab === 'design' ? { initialAlloy: designContext } : {}"
            @design="handleDesign"
          />
        </KeepAlive>
      </transition>
    </main>
  </div>
</template>

<style scoped>
/* === APP CONTAINER === */
.app-container {
  min-height: 100vh;
  width: 100%;
}

/* === HEADER === */
.glass-header {
  background: var(--bg-card);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-xl);
  padding: var(--space-lg);
  margin-bottom: var(--space-xl);
  box-shadow: var(--shadow-lg);
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.logo {
  display: flex;
  align-items: center;
  gap: var(--space-md);
}

.logo-icon {
  font-size: 2.5rem;
}

.logo h1 {
  margin: 0;
  font-size: var(--font-size-2xl);
  font-weight: var(--font-weight-bold);
  background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.status-badge {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: var(--space-sm) var(--space-md);
  background: rgba(6, 214, 160, 0.1);
  border: 1px solid var(--success);
  border-radius: var(--radius-lg);
  font-size: var(--font-size-sm);
  color: var(--success);
  font-weight: var(--font-weight-medium);
  transition: all var(--transition-base);
}

.status-badge.offline {
  background: rgba(239, 71, 111, 0.1);
  border-color: var(--danger);
  color: var(--danger);
}

.status-badge.offline .status-dot {
  background: var(--danger);
  animation: none;
}

.status-dot {
  width: 8px;
  height: 8px;
  background: var(--success);
  border-radius: 50%;
  animation: pulse 2s ease-in-out infinite;
}

/* === TAB NAVIGATION === */
.tab-nav {
  display: flex;
  gap: var(--space-md);
  margin-bottom: var(--space-xl);
  padding: var(--space-sm);
  background: var(--bg-card);
  backdrop-filter: blur(10px);
  border-radius: var(--radius-lg);
  border: 1px solid var(--border-subtle);
}

.tab-button {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-sm);
  padding: var(--space-md) var(--space-lg);
  background: transparent;
  border: none;
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  font-size: var(--font-size-md);
  font-weight: var(--font-weight-medium);
  font-family: var(--font-family);
  cursor: pointer;
  transition: all var(--transition-base);
  position: relative;
  overflow: hidden;
}

.tab-button::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  opacity: 0;
  transition: opacity var(--transition-base);
  z-index: -1;
}

.tab-button:hover:not(.active) {
  color: var(--text-primary);
  background: var(--bg-glass);
}

.tab-button.active {
  color: white;
  box-shadow: var(--shadow-glow);
}

.tab-button.active::before {
  opacity: 1;
}

.tab-icon {
  font-size: 1.25rem;
}

.tab-label {
  font-weight: var(--font-weight-semibold);
}

/* === CONTENT AREA === */
.content-area {
  animation: fadeIn var(--transition-slow) ease-out;
}

/* === TRANSITIONS === */
.fade-enter-active,
.fade-leave-active {
  transition: opacity var(--transition-base);
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
