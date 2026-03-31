<script setup>
import { computed, ref } from 'vue'

const props = defineProps({
  alloy: {
    type: Object,
    required: true
  }
})

defineEmits(['design'])

// Collapsed state - starts collapsed by default
const isExpanded = ref(false)

const toggleExpand = (event) => {
  // Prevent if clicking on buttons inside header
  if (event.target.closest('button')) return
  isExpanded.value = !isExpanded.value
}

// Composition view selector (for wt% / at%)
const compositionView = ref('wt')

// Check if we have phase compositions
const hasGammaComposition = computed(() =>
  props.alloy.gamma_composition && Object.keys(props.alloy.gamma_composition).length > 0
)
const hasGammaPrimeComposition = computed(() =>
  props.alloy.gamma_prime_composition && Object.keys(props.alloy.gamma_prime_composition).length > 0
)
const hasPhaseData = computed(() => hasGammaComposition.value || hasGammaPrimeComposition.value)

// Available composition views (wt% / at% only)
const availableViews = computed(() => {
  const views = []
  if (props.alloy.composition && Object.keys(props.alloy.composition).length > 0) {
    views.push({ key: 'wt', label: 'wt%' })
  }
  if (props.alloy.atomic_composition && Object.keys(props.alloy.atomic_composition).length > 0) {
    views.push({ key: 'at', label: 'at%' })
  }
  return views
})

// Get current composition based on selected view
const currentComposition = computed(() => {
  const data = compositionView.value === 'at'
    ? props.alloy.atomic_composition
    : props.alloy.composition
  if (!data) return []
  return Object.entries(data).sort(([, a], [, b]) => b - a)
})

const compositionLabel = computed(() =>
  compositionView.value === 'at' ? 'Composition (at%)' : 'Composition (wt%)'
)

// Phase compositions - full, sorted by value
const gammaComposition = computed(() => {
  if (!props.alloy.gamma_composition) return []
  return Object.entries(props.alloy.gamma_composition).sort(([, a], [, b]) => b - a)
})

const gammaPrimeComposition = computed(() => {
  if (!props.alloy.gamma_prime_composition) return []
  return Object.entries(props.alloy.gamma_prime_composition).sort(([, a], [, b]) => b - a)
})

// Temperature handling
const selectedTemp = ref(null)

// Get available temperatures from properties
const availableTemperatures = computed(() => {
  if (!props.alloy.properties || props.alloy.properties.length === 0) return []

  const temps = new Set()
  props.alloy.properties.forEach(p => {
    if (p.temperature_c !== null && p.temperature_c !== undefined) {
      temps.add(p.temperature_c)
    }
  })

  return Array.from(temps).sort((a, b) => a - b)
})

// Format temperature for display
const formatTemp = (t) => {
  if (t >= 20 && t <= 25) return 'Room'
  return `${t}°C`
}

// Initialize selected temperature to room temp or first available
const initSelectedTemp = () => {
  if (availableTemperatures.value.length === 0) return null
  const roomTemp = availableTemperatures.value.find(t => t >= 20 && t <= 25)
  if (roomTemp !== undefined) return roomTemp
  return availableTemperatures.value[0]
}

// Set initial temperature
if (selectedTemp.value === null) {
  selectedTemp.value = initSelectedTemp()
}

// Helper to normalize temperature for matching
const tempMatches = (propTemp, targetTemp) => {
  if (targetTemp === null) return true
  if (propTemp === null || propTemp === undefined) return false
  if (targetTemp >= 20 && targetTemp <= 25) {
    return propTemp >= 20 && propTemp <= 25
  }
  return propTemp === targetTemp
}

// TCP Risk styling
const tcpRiskClass = computed(() => {
  const risk = props.alloy.tcp_risk?.toLowerCase() || ''
  if (risk.includes('low')) return 'risk-low'
  if (risk.includes('medium') || risk.includes('moderate')) return 'risk-medium'
  if (risk.includes('high')) return 'risk-high'
  return ''
})

// Quick stats for collapsed view
const quickStats = computed(() => {
  const stats = []

  // Get yield strength at room temp
  if (props.alloy.properties) {
    const ys = props.alloy.properties.find(p => {
      const type = p.property_type?.toLowerCase() || ''
      const isYield = type.includes('yield') || type.includes('0.2%')
      const isRoom = p.temperature_c >= 20 && p.temperature_c <= 25
      return isYield && isRoom && p.value
    })
    if (ys) stats.push({ label: 'YS', value: `${ys.value.toFixed(0)} MPa` })
  }

  // Density
  if (props.alloy.density_gcm3) {
    stats.push({ label: 'Density', value: `${props.alloy.density_gcm3.toFixed(2)} g/cm³` })
  }

  // Gamma prime fraction
  if (props.alloy.gamma_prime_vol_pct) {
    stats.push({ label: "γ'", value: `${props.alloy.gamma_prime_vol_pct.toFixed(0)}%` })
  }

  return stats.slice(0, 3) // Max 3 quick stats
})

// Physical properties (temperature-independent)
const physicalProperties = computed(() => {
  const props_list = []

  if (props.alloy.density_gcm3) {
    props_list.push({ label: 'Density', value: `${props.alloy.density_gcm3.toFixed(2)} g/cm³` })
  }
  if (props.alloy.gamma_prime_vol_pct) {
    props_list.push({ label: 'Gamma Prime', value: `${props.alloy.gamma_prime_vol_pct.toFixed(1)}%` })
  }
  if (props.alloy.lattice_mismatch_pct !== null && props.alloy.lattice_mismatch_pct !== undefined) {
    props_list.push({ label: 'Lattice Mismatch', value: `${props.alloy.lattice_mismatch_pct.toFixed(2)}%` })
  }
  if (props.alloy.al_ti_ratio !== null && props.alloy.al_ti_ratio !== undefined) {
    props_list.push({ label: 'Al/Ti Ratio', value: props.alloy.al_ti_ratio.toFixed(2) })
  }

  return props_list
})

// Mechanical properties (temperature-dependent)
const mechanicalProperties = computed(() => {
  const props_list = []

  if (!props.alloy.properties || props.alloy.properties.length === 0) return props_list

  const findProp = (keywords) => {
    return props.alloy.properties.find(p => {
      if (!p.property_type) return false
      const type = p.property_type.toLowerCase()
      const matchesKeyword = keywords.some(kw => type.includes(kw))
      return matchesKeyword && tempMatches(p.temperature_c, selectedTemp.value)
    })
  }

  const ys = findProp(['yield', '0.2%', 'ys'])
  const uts = findProp(['ultimate', 'tensile', 'uts'])
  const el = findProp(['elongation', 'ductility'])
  const em = findProp(['elastic', 'modulus', "young's"])

  if (ys && ys.value) props_list.push({ label: 'Yield Strength', value: `${ys.value.toFixed(0)} MPa` })
  if (uts && uts.value) props_list.push({ label: 'Tensile Strength', value: `${uts.value.toFixed(0)} MPa` })
  if (el && el.value) props_list.push({ label: 'Elongation', value: `${el.value.toFixed(1)}%` })
  if (em && em.value) props_list.push({ label: 'Elastic Modulus', value: `${em.value.toFixed(0)} GPa` })

  return props_list
})
</script>

<template>
  <div :class="['alloy-card', { expanded: isExpanded }]">
    <!-- Collapsed Header (always visible) -->
    <div class="card-header" @click="toggleExpand">
      <div class="header-main">
        <div class="title-row">
          <span class="expand-icon">{{ isExpanded ? '▼' : '▶' }}</span>
          <h3>{{ alloy.name }}</h3>
          <span class="badge">{{ alloy.processing_method }}</span>
          <span v-if="alloy.tcp_risk" :class="['tcp-badge', tcpRiskClass]" :title="'TCP Risk: ' + alloy.tcp_risk">
            {{ alloy.tcp_risk }}
          </span>
        </div>
        <!-- Quick stats when collapsed -->
        <div v-if="!isExpanded && quickStats.length" class="quick-stats">
          <span v-for="stat in quickStats" :key="stat.label" class="quick-stat">
            <span class="qs-label">{{ stat.label }}:</span> {{ stat.value }}
          </span>
        </div>
      </div>
      <button @click.stop="$emit('design', alloy)" class="design-btn" title="Design variant">
        🧬 Design
      </button>
    </div>

    <!-- Expanded Content -->
    <div v-if="isExpanded" class="card-body">
      <!-- Composition -->
      <div class="section">
        <div class="section-header">
          <span class="section-title">{{ compositionLabel }}</span>
          <div v-if="availableViews.length > 1" class="view-tabs">
            <button
              v-for="view in availableViews"
              :key="view.key"
              :class="['tab', { active: compositionView === view.key }]"
              @click="compositionView = view.key"
            >{{ view.label }}</button>
          </div>
        </div>
        <div class="comp-grid">
          <span v-for="([el, val]) in currentComposition" :key="el" class="comp-tag">
            <b>{{ el }}</b> {{ typeof val === 'number' ? val.toFixed(1) : val }}%
          </span>
        </div>
      </div>

      <!-- Phase Compositions (stacked vertically, full data) -->
      <div v-if="hasPhaseData" class="section phases">
        <div v-if="hasGammaComposition" class="phase-row">
          <span class="phase-label">Gamma</span>
          <div class="phase-comp">
            <span v-for="([el, val]) in gammaComposition" :key="el" class="phase-tag">
              <b>{{ el }}</b> {{ val.toFixed(1) }}%
            </span>
          </div>
        </div>
        <div v-if="hasGammaPrimeComposition" class="phase-row">
          <span class="phase-label">Gamma Prime</span>
          <div class="phase-comp">
            <span v-for="([el, val]) in gammaPrimeComposition" :key="el" class="phase-tag">
              <b>{{ el }}</b> {{ val.toFixed(1) }}%
            </span>
          </div>
        </div>
      </div>

      <!-- Physical Properties (temperature-independent) -->
      <div v-if="physicalProperties.length > 0" class="section">
        <div class="section-header">
          <span class="section-title">Physical Properties</span>
        </div>
        <div class="props-row">
          <div v-for="prop in physicalProperties" :key="prop.label" class="prop-item">
            <span class="prop-label">{{ prop.label }}</span>
            <span class="prop-value">{{ prop.value }}</span>
          </div>
        </div>
      </div>

      <!-- Mechanical Properties (temperature-dependent) -->
      <div v-if="mechanicalProperties.length > 0" class="section">
        <div class="section-header">
          <span class="section-title">Mechanical Properties</span>
          <select v-if="availableTemperatures.length > 1" v-model="selectedTemp" class="temp-select">
            <option v-for="t in availableTemperatures" :key="t" :value="t">{{ formatTemp(t) }}</option>
          </select>
          <span v-else-if="selectedTemp !== null" class="temp-tag">@ {{ formatTemp(selectedTemp) }}</span>
        </div>
        <div class="props-row">
          <div v-for="prop in mechanicalProperties" :key="prop.label" class="prop-item">
            <span class="prop-label">{{ prop.label }}</span>
            <span class="prop-value">{{ prop.value }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.alloy-card {
  margin-top: 0.5rem;
  background: var(--bg-panel);
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  font-size: 0.8rem;
  transition: all 0.2s ease;
}

.alloy-card:hover {
  border-color: var(--border-strong);
}

.alloy-card.expanded {
  border-color: rgba(99, 102, 241, 0.3);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.6rem 0.75rem;
  cursor: pointer;
  user-select: none;
}

.card-header:hover {
  background: var(--bg-glass);
}

.header-main {
  flex: 1;
  min-width: 0;
}

.title-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.expand-icon {
  font-size: 0.6rem;
  color: var(--text-muted);
  width: 0.8rem;
  transition: transform 0.2s;
}

/* Quick stats in collapsed view */
.quick-stats {
  display: flex;
  gap: 0.75rem;
  margin-top: 0.3rem;
  padding-left: 1.3rem;
}

.quick-stat {
  font-size: 0.7rem;
  color: var(--text-secondary);
}

.qs-label {
  color: var(--text-muted);
}

/* Expanded body */
.card-body {
  padding: 0.5rem 0.75rem 0.75rem;
  border-top: 1px solid var(--border-subtle);
  animation: slideDown 0.2s ease;
}

@keyframes slideDown {
  from {
    opacity: 0;
    transform: translateY(-8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

h3 {
  margin: 0;
  font-size: 1rem;
  color: var(--primary-light);
  font-weight: 600;
}

.badge {
  font-size: 0.65rem;
  padding: 0.1rem 0.4rem;
  background: var(--bg-glass);
  border-radius: 4px;
  color: var(--text-muted);
  text-transform: lowercase;
}

.tcp-badge {
  font-size: 0.6rem;
  padding: 0.1rem 0.35rem;
  border-radius: 4px;
  font-weight: 600;
  text-transform: uppercase;
}

.tcp-badge.risk-low {
  background: rgba(34, 197, 94, 0.2);
  color: #22c55e;
}

.tcp-badge.risk-medium {
  background: rgba(234, 179, 8, 0.2);
  color: #eab308;
}

.tcp-badge.risk-high {
  background: rgba(239, 68, 68, 0.2);
  color: #ef4444;
}

.design-btn {
  background: linear-gradient(135deg, var(--secondary) 0%, #8b5cf6 100%);
  color: white;
  border: none;
  padding: 0.3rem 0.6rem;
  border-radius: 5px;
  font-size: 0.7rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  white-space: nowrap;
}

.design-btn:hover {
  transform: translateY(-1px);
  filter: brightness(1.1);
}

.section {
  margin-bottom: 0.5rem;
}

.section:last-child {
  margin-bottom: 0;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.3rem;
}

.section-title {
  font-size: 0.65rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-weight: 600;
}

.view-tabs {
  display: flex;
  gap: 0.2rem;
}

.tab {
  background: var(--bg-glass);
  border: 1px solid var(--border-subtle);
  color: var(--text-muted);
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
  cursor: pointer;
}

.tab.active {
  background: var(--primary);
  border-color: var(--primary);
  color: white;
}

.comp-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem;
}

.comp-tag {
  background: var(--bg-input);
  padding: 0.15rem 0.35rem;
  border-radius: 3px;
  font-size: 0.7rem;
  color: var(--text-secondary);
}

.comp-tag b {
  color: var(--primary-light);
  margin-right: 0.15rem;
}

/* Phase compositions - stacked */
.phases {
  background: var(--bg-glass);
  border-radius: 6px;
  padding: 0.4rem;
}

.phase-row {
  display: flex;
  align-items: flex-start;
  gap: 0.5rem;
  margin-bottom: 0.35rem;
}

.phase-row:last-child {
  margin-bottom: 0;
}

.phase-label {
  font-size: 0.6rem;
  font-weight: 600;
  color: var(--text-muted);
  min-width: 4.5rem;
  text-transform: uppercase;
  padding-top: 0.1rem;
}

.phase-comp {
  display: flex;
  flex-wrap: wrap;
  gap: 0.2rem;
  flex: 1;
}

.phase-tag {
  font-size: 0.65rem;
  color: var(--text-muted);
  background: var(--bg-glass);
  padding: 0.1rem 0.25rem;
  border-radius: 2px;
}

.phase-tag b {
  color: var(--primary-hover);
  margin-right: 0.1rem;
}

/* Temperature selector */
.temp-select {
  background: var(--bg-input);
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
  color: var(--text-primary);
  font-size: 0.75rem;
  padding: 0.25rem 0.4rem;
  cursor: pointer;
}

.temp-tag {
  font-size: 0.75rem;
  color: var(--text-muted);
}

/* Properties row */
.props-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
}

.prop-item {
  background: var(--bg-glass);
  padding: 0.25rem 0.4rem;
  border-radius: 4px;
  text-align: center;
  min-width: 70px;
}

.prop-label {
  display: block;
  font-size: 0.55rem;
  color: var(--text-muted);
  text-transform: uppercase;
  margin-bottom: 0.1rem;
}

.prop-value {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--text-primary);
}

/* Responsive */
@media (max-width: 480px) {
  .alloy-card { font-size: 0.85rem; }
  .card-header { padding: 0.5rem 0.6rem; }
  h3 { font-size: 0.95rem; }
  .quick-stats { gap: 0.5rem; padding-left: 0.8rem; }
  .quick-stat { font-size: 0.75rem; }
  .card-body { padding: 0.4rem 0.6rem 0.6rem; }
  .section-title { font-size: 0.7rem; }
  .comp-tag { font-size: 0.75rem; padding: 0.2rem 0.4rem; }
  .phase-label { font-size: 0.65rem; min-width: 3.5rem; }
  .phase-tag { font-size: 0.7rem; }
  .prop-label { font-size: 0.6rem; }
  .prop-value { font-size: 0.8rem; }
  .prop-item { min-width: 60px; padding: 0.3rem 0.4rem; }
  .phase-row { flex-wrap: wrap; }
  .badge { font-size: 0.7rem; }
  .design-btn { padding: 0.35rem 0.7rem; font-size: 0.75rem; }
}
</style>
