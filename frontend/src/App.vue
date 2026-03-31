<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import ResearchChat from './components/ResearchChat.vue'
import AlloyDesigner from './components/AlloyDesigner.vue'
import ToastNotification from './components/ToastNotification.vue'
import axios from 'axios'
import { API_BASE_URL } from './config'
import { useTheme } from './composables/useTheme'
import { useTour } from './composables/useTour'

const { theme, toggleTheme } = useTheme()
const { tourCompleted, startTour } = useTour()

const activeTab = ref('chat')
const isBackendOnline = ref(false)
const designContext = ref(null)
const showInfo = ref(false)
const showTourHint = ref(false)
const openSection = ref(null)
const toggleSection = (id) => { openSection.value = openSection.value === id ? null : id }
let healthCheckInterval = null

const handleDesign = (alloy) => {
  designContext.value = alloy
  activeTab.value = 'design'
}

const tourOptions = { switchToTab: (tab) => { activeTab.value = tab } }

const dismissHint = () => { showTourHint.value = false }

const launchTour = () => {
  showInfo.value = false
  showTourHint.value = false
  setTimeout(() => startTour(tourOptions), 300)
}

const checkBackendHealth = async () => {
  try {
    await axios.get(`${API_BASE_URL}/health`, { timeout: 2000 })
    isBackendOnline.value = true
  } catch (error) {
    isBackendOnline.value = false
  }
}

onMounted(() => {
  checkBackendHealth()
  healthCheckInterval = setInterval(checkBackendHealth, 5000)

  if (!tourCompleted.value) {
    setTimeout(() => { showTourHint.value = true }, 800)
  }
})

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
        <div class="header-right">
          <button
            class="theme-toggle"
            @click="toggleTheme"
            :title="theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'"
            :aria-label="theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'"
            data-tour="theme-toggle"
          >
            <span v-if="theme === 'dark'" class="theme-icon">☀️</span>
            <span v-else class="theme-icon">🌙</span>
          </button>
          <div :class="['status-badge', { offline: !isBackendOnline }]">
            <span class="status-dot"></span>
            {{ isBackendOnline ? 'Backend Online' : 'Backend Offline' }}
          </div>
        </div>
      </div>
    </header>

    <!-- Modern Tab Navigation -->
    <nav class="tab-nav">
      <button
        :class="['tab-button', { active: activeTab === 'chat' }]"
        @click="activeTab = 'chat'"
        data-tour="research-tab"
      >
        <span class="tab-icon">🔬</span>
        <span class="tab-label">Research</span>
      </button>
      <button
        :class="['tab-button', { active: activeTab === 'design' }]"
        @click="activeTab = 'design'"
        data-tour="design-tab"
      >
        <span class="tab-icon">🧪</span>
        <span class="tab-label">Evaluate & Design</span>
      </button>
      <div class="tour-button-wrapper">
        <button
          class="nav-icon-button tour-button"
          @click="launchTour"
          title="Take a guided tour"
        >
          <span>?</span>
        </button>
        <transition name="hint-pop">
          <div v-if="showTourHint" class="tour-hint" @click="launchTour">
            <span class="tour-hint-text">New here? Take a quick tour!</span>
            <button class="tour-hint-close" @click.stop="dismissHint" aria-label="Dismiss">&times;</button>
          </div>
        </transition>
      </div>
      <button
        class="nav-icon-button"
        @click="showInfo = true"
        title="Help & Information"
        data-tour="info-button"
      >
        <span>ℹ️</span>
      </button>
    </nav>

    <!-- INFO MODAL -->
    <transition name="fade">
      <div v-if="showInfo" class="modal-overlay" @click.self="showInfo = false">
        <div class="modal-content">
          <div class="modal-header">
            <h3>AlloyGraph Guide</h3>
            <button class="close-btn" @click="showInfo = false">×</button>
          </div>
          <div class="modal-body">
            <div class="accordion">
              <div class="accordion-item">
                <button class="accordion-trigger" @click="toggleSection('design')" :class="{ open: openSection === 'design' }">
                  <span>🧬 Inverse Design Mode</span>
                  <span class="accordion-arrow">›</span>
                </button>
                <div v-if="openSection === 'design'" class="accordion-content">
                  <p><strong>Purpose:</strong> AI-driven composition synthesis to meet target mechanical properties using multi-agent optimization.</p>
                  <ul>
                    <li><strong>Target Properties:</strong> Specify minimum values for YS, UTS, Elongation, Elastic Modulus, or maximum Density. Set to <strong>0</strong> to exclude.</li>
                    <li><strong>Processing Route:</strong> Select <em>wrought</em> or <em>cast</em> based on your manufacturing method.</li>
                    <li><strong>Iterations:</strong> Higher values (5-10) explore more compositional space but increase runtime (~2-5 min per iteration).</li>
                  </ul>
                </div>
              </div>

              <div class="accordion-item">
                <button class="accordion-trigger" @click="toggleSection('predict')" :class="{ open: openSection === 'predict' }">
                  <span>🧪 Property Prediction Mode</span>
                  <span class="accordion-arrow">›</span>
                </button>
                <div v-if="openSection === 'predict'" class="accordion-content">
                  <p><strong>Purpose:</strong> ML/KG data fusion to predict properties for known compositions, validated against physics constraints.</p>
                  <ul>
                    <li><strong>Input:</strong> Enter weight percentages (should sum to ~100%).</li>
                    <li><strong>ML Models:</strong> Trained on Ni-based superalloy database with engineered metallurgical features (γ' fraction, Md parameter, lattice mismatch, VEC).</li>
                    <li><strong>Knowledge Graph Fusion:</strong> If composition closely matches known alloys, predictions are weighted toward experimental data.</li>
                  </ul>
                </div>
              </div>

              <div class="accordion-item">
                <button class="accordion-trigger" @click="toggleSection('physics')" :class="{ open: openSection === 'physics' }">
                  <span>📊 Physics Validation & Confidence</span>
                  <span class="accordion-arrow">›</span>
                </button>
                <div v-if="openSection === 'physics'" class="accordion-content">
                  <ul>
                    <li><span class="status-pass">PASS</span> No critical violations. Md &lt; 0.98, lattice mismatch &lt; 0.8%, properties within known ranges.</li>
                    <li><span class="status-reject">REJECT</span> Physics constraints violated (TCP phase risk, excessive lattice mismatch, γ' incoherence).</li>
                    <li><strong>Confidence:</strong> HIGH (database match), MEDIUM (interpolation), LOW (extrapolation).</li>
                    <li><strong>Intervals:</strong> Uncertainty ranges based on model confidence and nearest-neighbor distances.</li>
                  </ul>
                </div>
              </div>

              <div class="accordion-item">
                <button class="accordion-trigger" @click="toggleSection('chat')" :class="{ open: openSection === 'chat' }">
                  <span>🔬 Research & Chat Mode</span>
                  <span class="accordion-arrow">›</span>
                </button>
                <div v-if="openSection === 'chat'" class="accordion-content">
                  <p><strong>Purpose:</strong> Query the knowledge graph to find similar alloys, explore literature data, or ask metallurgical questions.</p>
                  <ul>
                    <li><strong>Search:</strong> "Find alloys similar to Inconel 718" or "Show me high-γ' superalloys"</li>
                    <li><strong>Properties:</strong> "What alloys have YS > 1000 MPa?" or "Compare Waspaloy and Inconel 718"</li>
                    <li><strong>Learn:</strong> Ask about phase stability, strengthening mechanisms, or processing effects.</li>
                  </ul>
                </div>
              </div>

              <div class="accordion-item">
                <button class="accordion-trigger" @click="toggleSection('limits')" :class="{ open: openSection === 'limits' }">
                  <span>⚙️ Known Limitations</span>
                  <span class="accordion-arrow">›</span>
                </button>
                <div v-if="openSection === 'limits'" class="accordion-content">
                  <ul>
                    <li>Predictions assume room temperature (20°C) unless otherwise specified.</li>
                    <li>γ' volume fraction uses a composition-based solubility model; accuracy may vary for non-standard compositions.</li>
                    <li>TCP risk is composition-based; actual phase formation depends on heat treatment and kinetics.</li>
                    <li>Experimental validation is recommended for novel alloy designs.</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </transition>

    <!-- Content Area -->
    <main>
      <Transition name="tab-fade" mode="out-in">
        <KeepAlive :include="['ResearchChat']">
          <ResearchChat v-if="activeTab === 'chat'" key="chat" @design="handleDesign" />
          <AlloyDesigner v-else key="design" :initialAlloy="designContext" />
        </KeepAlive>
      </Transition>
    </main>

    <!-- Global Toast Notifications -->
    <ToastNotification />
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

.header-right {
  display: flex;
  align-items: center;
  gap: var(--space-md);
}

.theme-toggle {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  background: var(--bg-glass);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all var(--transition-base);
  padding: 0;
}

.theme-toggle:hover {
  background: rgba(99, 102, 241, 0.15);
  border-color: var(--primary);
  transform: scale(1.05);
}

.theme-icon {
  font-size: 1.25rem;
  line-height: 1;
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


/* === NAV ICON BUTTONS === */
.nav-icon-button {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 44px;
  min-width: 44px;
  padding: var(--space-md);
  background: var(--bg-glass);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  font-size: 1.25rem;
  cursor: pointer;
  transition: all var(--transition-base);
}

.nav-icon-button:hover {
  background: rgba(99, 102, 241, 0.15);
  border-color: var(--primary);
  color: var(--primary);
}

.nav-icon-button.tour-button span {
  font-weight: var(--font-weight-bold);
  font-size: 1.1rem;
  font-family: var(--font-family);
}

/* === TOUR HINT BUBBLE === */
.tour-button-wrapper {
  position: relative;
}

.tour-hint {
  position: absolute;
  bottom: calc(100% + 10px);
  right: 0;
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  color: white;
  padding: var(--space-sm) var(--space-md);
  border-radius: var(--radius-md);
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-medium);
  white-space: nowrap;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  box-shadow: var(--shadow-lg);
  z-index: 10;
  animation: hint-bounce 2s ease-in-out infinite;
}

.tour-hint::before {
  content: '';
  position: absolute;
  bottom: -6px;
  right: 16px;
  width: 12px;
  height: 12px;
  background: var(--secondary);
  transform: rotate(45deg);
  border-radius: 2px;
}

.tour-hint-text {
  pointer-events: none;
}

.tour-hint-close {
  background: none;
  border: none;
  color: rgba(255, 255, 255, 0.7);
  font-size: 1.1rem;
  cursor: pointer;
  padding: 0;
  line-height: 1;
  transition: color var(--transition-fast);
}

.tour-hint-close:hover {
  color: white;
}

@keyframes hint-bounce {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-4px); }
}

.hint-pop-enter-active {
  transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.hint-pop-leave-active {
  transition: all 0.2s ease-in;
}
.hint-pop-enter-from {
  opacity: 0;
  transform: scale(0.8) translateY(-8px);
}
.hint-pop-leave-to {
  opacity: 0;
  transform: scale(0.9);
}

/* === INFO MODAL === */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: var(--space-xl);
}

.modal-content {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-xl);
  max-width: 700px;
  max-height: 80vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  box-shadow: var(--shadow-lg);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--space-lg);
  border-bottom: 1px solid var(--border-subtle);
}

.modal-header h3 {
  margin: 0;
  font-size: var(--font-size-lg);
  color: var(--text-primary);
}

.close-btn {
  background: none;
  border: none;
  font-size: 1.5rem;
  color: var(--text-muted);
  cursor: pointer;
  padding: 0;
  line-height: 1;
  transition: color var(--transition-base);
}

.close-btn:hover {
  color: var(--text-primary);
}

.modal-body {
  padding: var(--space-lg);
  overflow-y: auto;
  font-size: var(--font-size-sm);
  color: var(--text-secondary);
  line-height: 1.6;
}

/* === ACCORDION === */
.accordion {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.accordion-item {
  border-radius: var(--radius-md);
  overflow: hidden;
}

.accordion-trigger {
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--space-md);
  background: var(--bg-glass);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  color: var(--text-primary);
  font-size: var(--font-size-md);
  font-weight: var(--font-weight-semibold);
  font-family: var(--font-family);
  cursor: pointer;
  transition: all var(--transition-base);
}

.accordion-trigger:hover {
  background: var(--bg-elevated);
  border-color: var(--border-strong);
}

.accordion-trigger.open {
  background: var(--bg-elevated);
  border-color: var(--primary);
  border-radius: var(--radius-md) var(--radius-md) 0 0;
}

.accordion-arrow {
  font-size: 1.2rem;
  transition: transform var(--transition-base);
  color: var(--text-muted);
}

.accordion-trigger.open .accordion-arrow {
  transform: rotate(90deg);
  color: var(--primary);
}

.accordion-content {
  padding: var(--space-md) var(--space-lg);
  background: var(--bg-glass);
  border: 1px solid var(--border-subtle);
  border-top: none;
  border-radius: 0 0 var(--radius-md) var(--radius-md);
  animation: accordionOpen 0.2s ease-out;
}

@keyframes accordionOpen {
  from { opacity: 0; transform: translateY(-4px); }
  to { opacity: 1; transform: translateY(0); }
}

.accordion-content p {
  margin: 0 0 var(--space-sm);
  color: var(--text-secondary);
  font-size: var(--font-size-sm);
  line-height: 1.6;
}

.accordion-content ul {
  margin: 0;
  padding-left: var(--space-lg);
}

.accordion-content li {
  margin-bottom: var(--space-xs);
  color: var(--text-secondary);
  font-size: var(--font-size-sm);
  line-height: 1.6;
}

.status-pass {
  display: inline-block;
  background: rgba(6, 214, 160, 0.2);
  color: var(--success);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
}

.status-reject {
  display: inline-block;
  background: rgba(239, 71, 111, 0.2);
  color: var(--danger);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
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

/* === TAB TRANSITIONS === */
.tab-fade-enter-active,
.tab-fade-leave-active {
  transition: opacity var(--transition-slow), transform var(--transition-slow);
}

.tab-fade-enter-from {
  opacity: 0;
  transform: translateY(8px);
}

.tab-fade-leave-to {
  opacity: 0;
  transform: translateY(-8px);
}

/* === RESPONSIVE: TABLET (≤768px) === */
@media (max-width: 768px) {
  .glass-header {
    padding: var(--space-md);
    margin-bottom: var(--space-md);
    border-radius: var(--radius-md);
  }

  .logo-icon {
    font-size: 1.75rem;
  }

  .logo h1 {
    font-size: var(--font-size-xl);
  }

  .status-badge {
    font-size: 0;
    padding: var(--space-xs);
    min-width: 28px;
    justify-content: center;
  }

  .status-badge .status-dot {
    margin: 0;
  }

  .tab-nav {
    gap: var(--space-xs);
    margin-bottom: var(--space-md);
    padding: var(--space-xs);
  }

  .tab-button {
    padding: var(--space-sm) var(--space-md);
    font-size: var(--font-size-sm);
  }

  .modal-content {
    max-width: 90vw;
  }

  .modal-overlay {
    padding: var(--space-md);
  }

  .tour-hint {
    display: none;
  }
}

/* === RESPONSIVE: PHONE (≤480px) === */
@media (max-width: 480px) {
  .glass-header {
    padding: var(--space-sm);
    margin-bottom: var(--space-sm);
  }

  .logo-icon {
    font-size: 1.5rem;
  }

  .logo h1 {
    font-size: var(--font-size-lg);
  }

  .logo {
    gap: var(--space-xs);
  }

  .header-right {
    gap: var(--space-xs);
  }

  .theme-toggle {
    width: 36px;
    height: 36px;
  }

  .tab-nav {
    gap: 4px;
    padding: 4px;
    margin-bottom: var(--space-sm);
  }

  .tab-button {
    padding: var(--space-sm);
    gap: 0;
  }

  .tab-label {
    display: none;
  }

  .tab-icon {
    font-size: 1.1rem;
  }

  .nav-icon-button {
    width: 38px;
    min-width: 38px;
    padding: var(--space-sm);
    font-size: 1rem;
  }

  .modal-content {
    max-width: 95vw;
    max-height: 90vh;
  }

  .modal-overlay {
    padding: var(--space-sm);
  }

  .modal-body {
    padding: var(--space-md);
  }

  .modal-header {
    padding: var(--space-md);
  }
}
</style>
