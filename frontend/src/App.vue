<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import ResearchChat from './components/ResearchChat.vue'
import AlloyDesigner from './components/AlloyDesigner.vue'
import axios from 'axios'
import { API_BASE_URL } from './config'

const activeTab = ref('chat')
const isBackendOnline = ref(false)
const designContext = ref(null) // Stores alloy data for design handoff
const showInfo = ref(false)
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
      <button
        class="info-button"
        @click="showInfo = true"
        title="Help & Information"
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
            <h4>🧬 Inverse Design Mode (Auto)</h4>
            <p><strong>Purpose:</strong> AI-driven composition synthesis to meet target mechanical properties using multi-agent optimization.</p>
            <ul>
              <li><strong>Target Properties:</strong> Specify minimum values for YS (Yield Strength), UTS (Ultimate Tensile Strength), Elongation, Elastic Modulus, or maximum Density. Set to <strong>0</strong> to exclude from optimization.</li>
              <li><strong>Processing Route:</strong> Select <em>wrought</em> or <em>cast</em> based on your intended manufacturing method.</li>
              <li><strong>Iterations:</strong> Higher values (5-10) explore more compositional space but increase runtime (~2-5 min per iteration).</li>
            </ul>

            <h4>🧪 Property Prediction Mode (Manual)</h4>
            <p><strong>Purpose:</strong> ML/KG data fusion to predict properties for known compositions, validated against physics constraints.</p>
            <ul>
              <li><strong>Input:</strong> Enter weight percentages (should sum to ~100%).</li>
              <li><strong>ML Models:</strong> Trained on Ni-based superalloy database with engineered metallurgical features (γ' fraction, Md parameter, lattice mismatch, VEC).</li>
              <li><strong>Knowledge Graph Fusion:</strong> If composition closely matches known alloys in the database, predictions are weighted toward experimental data.</li>
            </ul>

            <h4>📊 Physics Validation & Confidence</h4>
            <ul>
              <li><span class="status-pass">PASS</span> No critical violations. Md (phase stability) &lt; 0.98, lattice mismatch &lt; 0.8%, properties within known ranges.</li>
              <li><span class="status-reject">REJECT</span> Physics constraints violated (e.g., TCP phase risk, excessive lattice mismatch, γ' incoherence).</li>
              <li><strong>Confidence Level:</strong> HIGH (database match + model agreement), MEDIUM (model interpolation), LOW (extrapolation beyond training data).</li>
              <li><strong>Prediction Intervals:</strong> Uncertainty ranges shown for each property based on model confidence and nearest-neighbor distances.</li>
            </ul>

            <h4>🔬 Research & Chat Mode</h4>
            <p><strong>Purpose:</strong> Query the knowledge graph to find similar alloys, explore literature data, or ask metallurgical questions.</p>
            <ul>
              <li><strong>Search by Composition:</strong> "Find alloys similar to Inconel 718" or "Show me high-γ' superalloys"</li>
              <li><strong>Property Queries:</strong> "What alloys have YS > 1000 MPa?" or "Compare Waspaloy and Inconel 718"</li>
              <li><strong>Metallurgical Questions:</strong> Ask about phase stability, strengthening mechanisms, or processing effects.</li>
            </ul>

            <h4>⚙️ Known Limitations</h4>
            <ul>
              <li>Predictions assume room temperature (20°C) unless otherwise specified.</li>
              <li>γ' (gamma prime) volume fraction uses a composition-based solubility model; accuracy may vary for non-standard compositions.</li>
              <li>TCP risk assessment is based on composition; actual phase formation depends on heat treatment and kinetics.</li>
              <li>For optimal accuracy, experimental validation is recommended for novel alloy designs.</li>
            </ul>
          </div>
        </div>
      </div>
    </transition>

    <!-- Content Area -->
    <main class="content-area">
      <KeepAlive>
        <ResearchChat
          v-if="activeTab === 'chat'"
          @design="handleDesign"
        />
      </KeepAlive>
      <AlloyDesigner
        v-if="activeTab === 'design'"
        :initialAlloy="designContext"
        @design="handleDesign"
      />
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

/* === INFO BUTTON === */
.info-button {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 44px;
  min-width: 44px;
  padding: var(--space-md);
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  font-size: 1.25rem;
  cursor: pointer;
  transition: all var(--transition-base);
}

.info-button:hover {
  background: rgba(99, 102, 241, 0.15);
  border-color: var(--primary);
  color: var(--primary);
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

.modal-body h4 {
  margin: var(--space-lg) 0 var(--space-sm);
  font-size: var(--font-size-md);
  color: var(--text-primary);
}

.modal-body h4:first-child {
  margin-top: 0;
}

.modal-body p {
  margin: 0 0 var(--space-sm);
}

.modal-body ul {
  margin: 0;
  padding-left: var(--space-lg);
}

.modal-body li {
  margin-bottom: var(--space-xs);
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
