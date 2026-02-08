<script setup>
import { ref, computed, watch, onMounted } from 'vue'
import axios from 'axios'
import { API_BASE_URL } from '../config'

// --- PROPS ---
const props = defineProps({
  initialAlloy: {
    type: Object,
    default: null
  }
})

// --- STATE ---
const mode = ref('manual') // 'manual' (Composition -> Props) or 'auto' (Props -> Composition)
const loading = ref(false)
const logs = ref([])
const result = ref(null)

// --- ERROR STATE ---
const error = ref(null)
const errorType = ref(null) // 'network', 'timeout', 'validation', 'server', 'unknown'
const retryCount = ref(0)
const maxRetries = 3

// --- LOADING STATE WITH PROGRESS STEPS ---
const loadingMessage = ref('')
const startTime = ref(null)
const elapsedSeconds = ref(0)
let elapsedTimer = null
const loadingStep = ref(0)
let stepTimer = null

const evaluationSteps = [
  { icon: '🔬', label: 'Running ML predictions & physics models' },
  { icon: '📊', label: 'Analyst investigating alloy sources' },
  { icon: '🔗', label: 'Querying Knowledge Graph' },
  { icon: '⚖️', label: 'Triangulating ML, Physics, and KG data' },
  { icon: '🔎', label: 'Reviewer verifying metallurgical consistency' },
  { icon: '✍️', label: 'Generating metallurgical analysis' },
]

const designSteps = [
  { icon: '🎨', label: 'Designing alloy composition' },
  { icon: '🔬', label: 'Pre-computing ML & physics predictions' },
  { icon: '⚖️', label: 'Analyst investigating alloy sources' },
  { icon: '🔎', label: 'Reviewer verifying predictions' },
  { icon: '✍️', label: 'Generating analysis report' },
]

const currentSteps = computed(() => mode.value === 'manual' ? evaluationSteps : designSteps)

// --- DESIGN HISTORY STATE ---
const designHistory = ref([])
const showHistory = ref(false)
const maxHistoryItems = 20

// --- ERROR HANDLING HELPERS ---
const classifyError = (err) => {
  if (err.code === 'ECONNABORTED' || err.message.includes('timeout')) {
    return 'timeout'
  } else if (err.code === 'ERR_NETWORK' || err.message.includes('Network Error')) {
    return 'network'
  } else if (!err.response) {
    // No response received - could be CORS, network, or parsing error
    return 'no_response'
  } else if (err.response?.status >= 400 && err.response?.status < 500) {
    return 'validation'
  } else if (err.response?.status >= 500) {
    return 'server'
  }
  return 'unknown'
}

const getErrorMessage = (type, err) => {
  const messages = {
    network: 'Network error: Unable to connect to backend. Check if backend is running.',
    no_response: 'Connection error: Request sent but no response received. This may be a CORS issue, response parsing error, or network timeout.',
    timeout: 'Request timed out. The server took too long to respond.',
    validation: `Invalid request: ${err.response?.data?.error || err.message}`,
    server: `Server error: ${err.response?.data?.error || err.message}`,
    unknown: `Unexpected error: ${err.message} (Code: ${err.code || 'N/A'})`
  }
  return messages[type] || messages.unknown
}

const getErrorRecoveryActions = (type) => {
  const actions = {
    network: ['Check your internet connection', 'Verify backend is running on port 5001', 'Retry the operation'],
    no_response: ['Check browser console for detailed error', 'Verify backend is running', 'Check for CORS issues', 'Retry the operation'],
    timeout: ['Reduce max iterations', 'Try with simpler targets', 'Retry the operation'],
    validation: ['Increase max iterations (try 5-10)', 'Relax target constraints', 'Try different starting composition', 'Adjust gamma prime target if needed'],
    server: ['Wait a moment and retry', 'Check backend logs', 'Contact support if persists'],
    unknown: ['Retry the operation', 'Check browser console for details', 'Contact support']
  }
  return actions[type] || actions.unknown
}

const clearError = () => {
  error.value = null
  errorType.value = null
}

const getConfidenceClass = (confidence) => {
  if (confidence === undefined || confidence === null) return ''
  if (confidence >= 0.7) return 'confidence-high'
  if (confidence >= 0.5) return 'confidence-medium'
  return 'confidence-low'
}

// --- PREDICTION INFO HELPERS ---
const getTcpEmoji = (risk) => {
  if (!risk) return ''
  const r = risk.toLowerCase()
  if (r === 'moderate') return '🟡'
  if (r === 'elevated') return '🟠'
  if (r === 'critical') return '🔴'
  return ''
}

const hasUsefulPredictionInfo = (results) => {
  if (!results) return false
  const hasMatch = results.confidence?.matched_alloy && results.confidence.matched_alloy !== 'None'
  // Check TCP from metrics (key may be "TCP Risk" or "tcp_risk") or top-level tcpRisk
  const tcp = results.metallurgyMetrics?.['TCP Risk'] || results.metallurgyMetrics?.tcp_risk || results.tcpRisk
  const hasTcpWarning = tcp && tcp !== 'Low' && tcp !== 'UNKNOWN'
  return hasMatch || hasTcpWarning
}

// --- DESIGN HISTORY HELPERS ---
const loadHistory = () => {
  try {
    const stored = localStorage.getItem('alloyDesignHistory')
    if (stored) {
      designHistory.value = JSON.parse(stored)
    }
  } catch (e) {
    console.error('Failed to load history:', e)
  }
}

const saveToHistory = (design) => {
  const historyItem = {
    id: Date.now(),
    timestamp: new Date().toISOString(),
    mode: mode.value,
    design: { ...design },
    targets: mode.value === 'auto' ? { ...targets.value } : null,
    composition: design.composition || manualComp.value,
    properties: design.properties || {},
    temperature: mode.value === 'auto' ? autoTemp.value : manualTemp.value,
    processing: mode.value === 'auto' ? autoProcessing.value : manualProcessing.value
  }

  designHistory.value.unshift(historyItem)
  if (designHistory.value.length > maxHistoryItems) {
    designHistory.value = designHistory.value.slice(0, maxHistoryItems)
  }

  try {
    localStorage.setItem('alloyDesignHistory', JSON.stringify(designHistory.value))
  } catch (e) {
    console.error('Failed to save history:', e)
  }
}

const loadFromHistory = (item) => {
  // Force reactivity by clearing result first
  result.value = null
  
  // Use nextTick to ensure mode switch completes before setting result
  if (item.mode === 'manual') {
    mode.value = 'manual'
    manualComp.value = { ...item.composition }
    manualTemp.value = item.temperature
    manualProcessing.value = item.processing
  } else {
    mode.value = 'auto'
    if (item.targets) {
      targets.value = { ...item.targets }
    }
    autoTemp.value = item.temperature
    autoProcessing.value = item.processing
  }
  
  // Set result on next tick to ensure reactivity
  setTimeout(() => {
    result.value = { ...item.design }
    showHistory.value = false
    
    // Scroll to results after a brief delay
    setTimeout(() => {
      const resultSection = document.querySelector('.output-area')
      if (resultSection) {
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }
    }, 150)
  }, 10)
}

const clearHistory = () => {
  if (confirm('Are you sure you want to clear all design history?')) {
    designHistory.value = []
    localStorage.removeItem('alloyDesignHistory')
  }
}

const exportHistory = () => {
  const dataStr = JSON.stringify(designHistory.value, null, 2)
  const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr)
  const exportFileDefaultName = `alloy_history_${new Date().toISOString().split('T')[0]}.json`

  const linkElement = document.createElement('a')
  linkElement.setAttribute('href', dataUri)
  linkElement.setAttribute('download', exportFileDefaultName)
  linkElement.click()
}

// --- SIMPLE LOADING HELPERS ---
const startLoading = (message) => {
  loading.value = true
  loadingMessage.value = message
  startTime.value = Date.now()
  elapsedSeconds.value = 0
  loadingStep.value = 0
  error.value = null
  errorType.value = null

  // Update elapsed time every second + advance through steps with natural pacing
  if (elapsedTimer) clearInterval(elapsedTimer)
  elapsedTimer = setInterval(() => {
    elapsedSeconds.value = Math.floor((Date.now() - startTime.value) / 1000)
    // Accelerating thresholds: early deterministic steps are fast,
    // later agent steps take longer (matches actual pipeline rhythm)
    // Steps advance once and stay on the last step until response arrives
    const steps = currentSteps.value
    const thresholds = steps.map((_, i) => Math.round(2 + i * 1.5 + i * i * 0.8))
    let step = 0
    for (let i = thresholds.length - 1; i >= 0; i--) {
      if (elapsedSeconds.value >= thresholds[i]) {
        step = Math.min(i + 1, steps.length - 1)
        break
      }
    }
    loadingStep.value = step
  }, 1000)
}

const stopLoading = () => {
  loading.value = false
  loadingMessage.value = ''
  loadingStep.value = 0
  if (elapsedTimer) {
    clearInterval(elapsedTimer)
    elapsedTimer = null
  }
}

// Clear results when switching modes
watch(mode, () => {
  result.value = null
  logs.value = []
  clearError()
})

// Load history and custom presets on mount
onMounted(() => {
  loadHistory()
  loadCustomPresets()
})

// --- MANUAL MODE STATE ---
const manualComp = ref({ Ni: 60, Cr: 20, Al: 10, Ti: 5, Co: 5 })
const manualTemp = ref(20) // Room temperature
const manualProcessing = ref('cast')  // Processing type for Evaluate mode

// Initialize from prop if available
const initFromProp = (alloy) => {
  if (!alloy) return
  if (alloy.composition) {
    manualComp.value = { ...alloy.composition }
  }
  if (alloy.processing_method) {
    manualProcessing.value = alloy.processing_method.toLowerCase()
  }
  // Reset results when a new alloy is loaded
  result.value = null
  mode.value = 'manual'
}

// Initial check
if (props.initialAlloy) initFromProp(props.initialAlloy)

// Watch for changes (needed when component is kept alive)
watch(() => props.initialAlloy, (newVal) => {
  initFromProp(newVal)
})

const BUILTIN_PRESETS = {
  "Waspaloy": { composition: {"Ni": 58.0, "Cr": 19.5, "Co": 13.5, "Mo": 4.3, "Al": 1.3, "Ti": 3.0, "C": 0.08, "B": 0.006, "Zr": 0.06}, processing: "wrought", builtin: true },
  "Inconel 718": { composition: { "Ni": 52.5, "Cr": 19.0, "Fe": 19.0, "Nb": 5.1, "Mo": 3.0, "Ti": 0.9, "Al": 0.5 }, processing: "wrought", builtin: true },
  "Udimet 720": { composition: { "Ni": 55.0, "Cr": 16.0, "Co": 14.7, "Ti": 5.0, "Al": 2.5, "Mo": 3.0, "W": 1.25 }, processing: "wrought", builtin: true },
  "IN738LC": { composition: {"Ni": 61.5, "Cr": 16.0, "Co": 8.5, "Mo": 1.75, "W": 2.6, "Al": 3.4, "Ti": 3.4, "Ta": 1.75, "Nb": 0.9, "C": 0.11, "B": 0.01, "Zr": 0.05}, processing: "cast", builtin: true },
  "Udimet 500": { composition: { "Ni": 54.0, "Cr": 18.0, "Co": 18.5, "Mo": 4.0, "Al": 2.9, "Ti": 2.9, "C": 0.08, "B": 0.006, "Zr": 0.05 }, processing: "wrought", builtin: true },
  "Haynes 282": { composition: { "Ni": 57.0, "Cr": 19.5, "Co": 10.0, "Mo": 8.5, "Ti": 2.1, "Al": 1.5, "Fe": 1.0, "Mn": 0.15, "Si": 0.1, "C": 0.06, "B": 0.005 }, processing: "wrought", builtin: true },
  "CMSX-4": { composition: {"Ni": 61.7, "Cr": 6.5, "Co": 9.0, "Mo": 0.6, "W": 6.0, "Al": 5.6, "Ti": 1.0, "Ta": 6.5, "Re": 3.0, "Hf": 0.1}, processing: "cast", builtin: true },
  "René 65": { composition: {"Ni": 51.6, "Cr": 16.0, "Co": 13.0, "Mo": 4.0, "W": 4.0, "Al": 2.1, "Ti": 3.7, "Nb": 0.7, "Fe": 1.0, "B": 0.016, "Zr": 0.05, "C": 0.01}, processing: "wrought", builtin: true }
}

// Custom presets stored in localStorage
const customPresets = ref({})

// Load custom presets from localStorage
const loadCustomPresets = () => {
  try {
    const stored = localStorage.getItem('alloyCustomPresets')
    if (stored) {
      customPresets.value = JSON.parse(stored)
    }
  } catch (e) {
    console.error('Failed to load custom presets:', e)
  }
}

// Save custom presets to localStorage
const saveCustomPresets = () => {
  try {
    localStorage.setItem('alloyCustomPresets', JSON.stringify(customPresets.value))
  } catch (e) {
    console.error('Failed to save custom presets:', e)
  }
}

// All presets (builtin + custom)
const allPresets = computed(() => {
  return { ...BUILTIN_PRESETS, ...customPresets.value }
})

// Track selected preset
const selectedPreset = ref(null)

// Save current composition as a new preset
const showSavePreset = ref(false)
const newPresetName = ref('')

const openSavePreset = () => {
  newPresetName.value = ''
  showSavePreset.value = true
}

const saveAsPreset = () => {
  const name = newPresetName.value.trim()
  if (!name) return

  if (BUILTIN_PRESETS[name]) {
    alert('Cannot overwrite built-in presets. Choose a different name.')
    return
  }

  customPresets.value[name] = {
    composition: { ...manualComp.value },
    processing: manualProcessing.value,
    builtin: false
  }
  saveCustomPresets()
  selectedPreset.value = name
  showSavePreset.value = false
  newPresetName.value = ''
}

const deletePreset = (name) => {
  if (BUILTIN_PRESETS[name]) {
    alert('Cannot delete built-in presets.')
    return
  }

  if (confirm(`Delete preset "${name}"?`)) {
    delete customPresets.value[name]
    saveCustomPresets()
    if (selectedPreset.value === name) {
      selectedPreset.value = null
    }
  }
}

const editPreset = (name) => {
  if (BUILTIN_PRESETS[name]) {
    alert('Cannot edit built-in presets. Load it and save as a new preset instead.')
    return
  }

  // Load the preset for editing
  loadPreset(name)
}

const loadPreset = (name) => {
  const preset = allPresets.value[name]
  if (!preset) return
  manualComp.value = { ...preset.composition }
  manualProcessing.value = preset.processing
  selectedPreset.value = name
  result.value = null
}

// Clear all composition values
const clearComposition = () => {
  manualComp.value = {}
  selectedPreset.value = null
  result.value = null
}

// JSON Import Modal state
const showJsonImport = ref(false)
const jsonInput = ref('')
const jsonError = ref('')

const openJsonImport = () => {
  jsonInput.value = ''
  jsonError.value = ''
  showJsonImport.value = true
}

const closeJsonImport = () => {
  showJsonImport.value = false
  jsonInput.value = ''
  jsonError.value = ''
}

const importJsonComposition = () => {
  jsonError.value = ''

  if (!jsonInput.value.trim()) {
    jsonError.value = 'Please enter a JSON composition'
    return
  }

  try {
    // Try to parse the JSON
    let parsed = JSON.parse(jsonInput.value.trim())

    // Handle wrapped format like { "composition": { ... } }
    if (parsed.composition && typeof parsed.composition === 'object') {
      parsed = parsed.composition
    }

    // Validate it's an object with numeric values
    if (typeof parsed !== 'object' || Array.isArray(parsed)) {
      jsonError.value = 'Invalid format: Expected an object like {"Ni": 60, "Cr": 20}'
      return
    }

    // Filter to only valid element entries (string keys, numeric values)
    const cleaned = {}
    for (const [key, value] of Object.entries(parsed)) {
      const numVal = parseFloat(value)
      if (!isNaN(numVal) && numVal > 0) {
        // Capitalize first letter of element symbol
        const element = key.charAt(0).toUpperCase() + key.slice(1).toLowerCase()
        cleaned[element] = Math.round(numVal * 1000) / 1000  // Round to 3 decimals
      }
    }

    if (Object.keys(cleaned).length === 0) {
      jsonError.value = 'No valid elements found. Use format: {"Ni": 60, "Cr": 20}'
      return
    }

    // Success - update composition and close modal
    manualComp.value = cleaned
    result.value = null
    closeJsonImport()

  } catch (e) {
    jsonError.value = `Invalid JSON: ${e.message}`
  }
}

const copyToEvaluation = () => {
  if (!result.value || !result.value.composition) return
  
  // Copy the predicted composition to manual mode
  manualComp.value = { ...result.value.composition }
  
  // Switch to manual mode
  mode.value = 'manual'
  
  // Scroll to composition input
  setTimeout(() => {
    const compSection = document.querySelector('.composition-input')
    if (compSection) {
      compSection.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, 100)
}

const addElement = (el) => {
  if (el && !manualComp.value[el]) {
    manualComp.value[el] = 0.0
  }
}

const removeElement = (el) => {
  delete manualComp.value[el]
}

const runValidation = async (isRetry = false) => {
  startLoading('Analyzing composition with ML models and physics validation...')
  if (!isRetry) {
    logs.value = []
    result.value = null
    retryCount.value = 0
  }
  logs.value.push(`🧪 Validating Composition at ${manualTemp.value}°C (${manualProcessing.value})...`)

  try {
    const res = await axios.post(`${API_BASE_URL}/api/validate`, {
      composition: manualComp.value,
      temp: manualTemp.value,
      processing: manualProcessing.value
    })

    if (res.data?.result?.error) {
      stopLoading()
      errorType.value = 'validation'
      error.value = res.data.result.error
      logs.value.push("❌ Error: " + res.data.result.error)
      return
    }

    result.value = res.data.result
    logs.value.push("✅ Prediction Complete.")

    if (res.data.result?.properties) {
      saveToHistory(res.data.result)
    }

    stopLoading()
    retryCount.value = 0
  } catch (err) {
    stopLoading()

    // Log detailed error for debugging
    console.error('Validation error:', err)

    errorType.value = classifyError(err)
    error.value = getErrorMessage(errorType.value, err)
    logs.value.push("❌ Error: " + error.value)
  }
}

const retryValidation = async () => {
  if (retryCount.value < maxRetries) {
    retryCount.value++
    logs.value.push(`🔄 Retrying (${retryCount.value}/${maxRetries})...`)

    // Exponential backoff
    const delay = Math.min(1000 * Math.pow(2, retryCount.value - 1), 10000)
    await new Promise(resolve => setTimeout(resolve, delay))

    await runValidation(true)
  } else {
    logs.value.push("❌ Max retries reached. Please try again later.")
  }
}

// --- AUTO MODE STATE ---
const targets = ref({
  yield: 0,         // Realistic superalloy yield strength target
  tensile: 0,          // Optional - leave at 0
  elongation: 0,       // Optional - leave at 0
  elastic_modulus: 0,  // Optional - leave at 0
  density: 0,        // Realistic density target for Ni-based superalloys
  gamma_prime: 0       // Optional - leave at 0
})
const autoIterations = ref(3)
const autoTemp = ref(20)  // Room temperature
const autoProcessing = ref('cast')

const runDesign = async (isRetry = false) => {
  const iterMsg = autoIterations.value > 1 ? ` (${autoIterations.value} iterations)` : ''
  startLoading(`AI agents designing alloy composition${iterMsg}. This may take several minutes...`)

  if (!isRetry) {
    logs.value = []
    result.value = null
    retryCount.value = 0
  }

  logs.value.push("🚀 Starting Inverse Design Agent...")
  let targetLog = `Targets: Yield≥${targets.value.yield} MPa`
  if (targets.value.tensile > 0) targetLog += `, Tensile≥${targets.value.tensile} MPa`
  if (targets.value.elongation > 0) targetLog += `, Elongation≥${targets.value.elongation}%`
  if (targets.value.density < 99) targetLog += `, Density≤${targets.value.density} g/cm³`
  if (targets.value.gamma_prime > 0) targetLog += `, Gamma Prime≥${targets.value.gamma_prime}%`
  logs.value.push(targetLog)

  try {
    // Build target_props object - ONLY include non-zero/non-default values
    const target_props = {}

    // Always include yield strength (primary target)
    if (targets.value.yield > 0) {
      target_props['Yield Strength'] = targets.value.yield
    }

    // Only include optional targets if they're set
    if (targets.value.tensile > 0) {
      target_props['Tensile Strength'] = targets.value.tensile
    }
    if (targets.value.elongation > 0) {
      target_props['Elongation'] = targets.value.elongation
    }
    if (targets.value.elastic_modulus > 0) {
      target_props['Elastic Modulus'] = targets.value.elastic_modulus
    }
    if (targets.value.density < 99) {  // Density target is "less than"
      target_props['Density'] = targets.value.density
    }
    if (targets.value.gamma_prime > 0) {  // CRITICAL: Only send if set!
      target_props['Gamma Prime'] = targets.value.gamma_prime
    }

    const response = await axios.post(`${API_BASE_URL}/api/design`, {
      target_props: target_props,
      processing: autoProcessing.value,
      temp: autoTemp.value,
      max_iter: autoIterations.value
    })

    const designResult = response.data.result
    result.value = designResult

    if (designResult.design_status === "incomplete" && designResult.issues && designResult.issues.length > 0) {
      logs.value.push("⚠️ Design completed with issues:")
      designResult.issues.forEach(issue => {
        const icon = issue.severity === "High" ? "🔴" : issue.severity === "Medium" ? "🟡" : "🔵"
        logs.value.push(`${icon} ${issue.type}: ${issue.description}`)
      })
      if (designResult.recommendations && designResult.recommendations.length > 0) {
        logs.value.push("\n💡 Recommendations:")
        designResult.recommendations.forEach(rec => {
          logs.value.push(`   • ${rec}`)
        })
      }
    } else if (designResult.error) {
      logs.value.push("❌ Error: " + designResult.error)
    } else {
      logs.value.push("✅ Design Complete!")
    }

    if (designResult.composition) {
      saveToHistory(designResult)
    }

    stopLoading()
    retryCount.value = 0
  } catch (err) {
    stopLoading()

    // Log detailed error for debugging
    console.error('Design error:', err)

    errorType.value = classifyError(err)
    error.value = getErrorMessage(errorType.value, err)
    logs.value.push("❌ Error: " + error.value)
  }
}

const retryDesign = async () => {
  if (retryCount.value < maxRetries) {
    retryCount.value++
    logs.value.push(`🔄 Retrying (${retryCount.value}/${maxRetries})...`)

    // Exponential backoff
    const delay = Math.min(1000 * Math.pow(2, retryCount.value - 1), 10000)
    await new Promise(resolve => setTimeout(resolve, delay))

    await runDesign(true)
  } else {
    logs.value.push("❌ Max retries reached. Please try again later.")
  }
}
// Helper to parse value from potential string with units (e.g. "970.0 MPa @ 20C")
const parseVal = (v) => {
    if (typeof v === 'number') return v
    if (typeof v === 'string') {
        // Extract first number found
        const match = v.match(/[\d\.]+/)
        return match ? parseFloat(match[0]) : 0
    }
    return 0
}

// Helper to fuzzy find property (e.g. "Yield Strength" matches "Yield Strength (MPa)" or "YieldStrength")
const lookUpProp = (obj, keyPart) => {
    if (!obj) return undefined
    // Direct match
    if (obj[keyPart]) return obj[keyPart]

    // Fuzzy match keys
    const keyPartSimple = keyPart.replace(/\s+/g, '').toLowerCase() // e.g. "yieldstrength"
    const keys = Object.keys(obj)
    for (const k of keys) {
        // Normalize key k
        // e.g. "Yield Strength (MPa)" -> "yieldstrength(mpa)"
        // Check if k starts with keyPartSimple or contains it distinctly?
        // Simplest: remove spaces, lowercase, remove content in parens
        const kClean = k.replace(/\s+/g, '').toLowerCase().replace(/\(.*\)/, '')
        if (kClean === keyPartSimple) return obj[k]
        if (kClean.includes(keyPartSimple)) return obj[k]
    }
    return undefined
}

// Helper to format metric labels with human-readable names
const formatMetricLabel = (key) => {
  const labelMap = {
    'md_average': 'Md Temperature (avg)',
    'sss_wt_pct': 'Solid Solution Strengthening (wt%)',
    'tcp_risk': 'TCP Risk',
    'gamma_prime_fraction': 'γ\' Fraction',
    'lattice_misfit': 'Lattice Misfit',
    'partitioning_ni': 'Ni Partitioning',
    'partitioning_cr': 'Cr Partitioning',
    'partitioning_al': 'Al Partitioning',
    'partitioning_ti': 'Ti Partitioning'
  }
  return labelMap[key] || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
}

// Helper to detect if a property prediction might be overestimated
const getOverestimationWarning = (prop, actual, upper, confidence) => {
  if (!upper || !actual || upper <= actual) return null
  
  const overestimation = ((upper - actual) / actual) * 100
  
  // If confidence is low and interval is wide, warn about optimism
  if (confidence < 0.6 && overestimation > 15) {
    return `High uncertainty (±${overestimation.toFixed(0)}%) - prediction may be optimistic`
  }
  
  if (overestimation > 25) {
    return `Wide confidence interval (±${overestimation.toFixed(0)}%) - treat prediction with caution`
  }
  
  return null
}

// Helper to assess interval quality
const getIntervalQuality = (lower, upper, actual) => {
  if (!lower || !upper || !actual) return 'unknown'
  
  const intervalWidth = upper - lower
  const relativeWidth = (intervalWidth / actual) * 100
  
  if (relativeWidth < 10) return 'tight'  // Green
  if (relativeWidth < 20) return 'moderate'  // Yellow
  return 'wide'  // Orange/Red
}

// Computed property for visual comparisons in design mode
const propertyComparisons = computed(() => {
  if (!result.value || mode.value !== 'auto') return []

  const comparisons = []
  const actualProps = result.value.properties || {}
  const propertyIntervals = result.value.property_intervals || {}
  const confidence = result.value.confidence || {}

  const propMap = [
    { key: 'Yield Strength', target: targets.value.yield, unit: 'MPa', icon: '🏋️' },
    { key: 'Tensile Strength', target: targets.value.tensile, unit: 'MPa', icon: '⛓️' },
    { key: 'Elongation', target: targets.value.elongation, unit: '%', icon: '📏' },
    { key: 'Elastic Modulus', target: targets.value.elastic_modulus, unit: 'GPa', icon: '🔧' },
    { key: 'Density', target: targets.value.density, unit: 'g/cm³', icon: '🧱', isMax: true },
    { key: 'Gamma Prime', target: targets.value.gamma_prime, unit: '%', icon: '💎' }
  ]

  for (const prop of propMap) {
    if (prop.target > 0 || (prop.isMax && prop.target < 99)) {
      const actualVal = parseVal(lookUpProp(actualProps, prop.key))
      if (actualVal) {
        // Get confidence interval if available
        const interval = propertyIntervals[prop.key] || {}
        const lower = interval.lower
        const upper = interval.upper
        const confidenceScore = confidence.score

        // Calculate ±value for display
        let plusMinus = null
        if (lower !== undefined && upper !== undefined && actualVal !== undefined) {
          plusMinus = ((upper - lower) / 2).toFixed(prop.key.includes('Density') ? 2 : 1)
        }

        // Simplified status determination - always show percentage of target
        let met, status, exceeds = false
        let percentage

        // Always calculate percentage as (actual/target) × 100
        percentage = Math.round((actualVal / prop.target) * 100)

        if (prop.isMax) {
          // For MAX targets (e.g., Density ≤ 8.5)
          // Lower is better, so >100% is bad
          if (actualVal <= prop.target) {
            met = true
            exceeds = false
            status = 'In Range'
          } else {
            met = false
            exceeds = true
            status = 'Too High'
          }
        } else {
          // For MIN targets (e.g., YS ≥ 510)
          // Higher is better, but too high is suspicious

          if (actualVal < prop.target) {
            // Under target
            met = false
            exceeds = false
            status = 'Too Low'
          } else {
            // Meets or exceeds target - determine if it's reasonable
            let limit

            // Per-property overrides
            if (prop.key.includes('Strength')) {
              // Strength can vary 2-3x depending on heat treatment
              limit = prop.target * 2.0;  // Allow 2x overshoot
            } else if (prop.key.includes('Gamma Prime')) {
              // Gamma Prime is a TARGET, not minimum - use range-based logic
              const tolerance = Math.max(2.0, prop.target * 0.2);  // ±20% or ±2%
              const gp_max = prop.target + tolerance;

              if (actualVal > gp_max) {
                // Above target range
                met = false
                exceeds = true
                status = 'Above Range'
                // Show how far above the upper bound (inverted percentage)
                percentage = Math.round((gp_max / actualVal) * 100)
              } else {
                // Within range [target - tolerance, target + tolerance]
                met = true
                exceeds = false
                status = 'In Range'
                percentage = 100
              }
              // Skip the limit check below for Gamma Prime
              limit = null
            } else {
              // Default: 50% overshoot is Too High
              limit = prop.target * 1.5
            }

            // Only check limit if it was set (not Gamma Prime)
            if (limit !== null) {
              if (actualVal > limit) {
                // Exceeds reasonable limit
                met = false
                exceeds = true
                status = 'Too High'
              } else {
                // Within reasonable range
                met = true
                exceeds = false
                status = 'In Range'
              }
            }
          }
        }

        comparisons.push({
          ...prop,
          actual: actualVal,
          lower,
          upper,
          plusMinus,
          confidenceScore,
          percentage: Math.round(percentage),
          met,
          exceeds,  // Flag for max targets that are exceeded
          status
        })
      }
    }
  }

  return comparisons
})

const parsedResults = computed(() => {
  if (!result.value) return null

  const data = result.value
  let comp = null
  let props = {}
  let propertyIntervals = {}
  let confidence = {}
  let explanation = ""
  let auditPenalties = []
  let metallurgyMetrics = {}
  let status = "UNKNOWN"
  let tcpRisk = "UNKNOWN"
  let similar = []
  let summary = ""
  let issues = []
  let recommendations = []
  let designStatus = ""
  let reasoning = ""
  let analystReasoning = ""
  let reviewerAssessment = ""
  let investigationFindings = ""
  let sourceReliability = ""
  let correctionsApplied = []
  let correctionsExplanation = ""

  // CASE 1: Design Output - has composition at top level
  if (data.composition) {
    comp = data.composition
    props = data.properties || {}
    propertyIntervals = data.property_intervals || {}
    confidence = data.confidence || {}
    explanation = data.explanation || ""
    auditPenalties = data.audit_penalties || []
    metallurgyMetrics = data.metallurgy_metrics || {}
    status = data.status || "UNKNOWN"
    tcpRisk = data.tcp_risk || metallurgyMetrics['TCP Risk'] || "UNKNOWN"
    similar = data.similar_alloys || []
    summary = data.summary || ""
    issues = data.issues || []
    recommendations = data.recommendations || []
    designStatus = data.design_status || ""
    reasoning = data.reasoning || ""
    correctionsApplied = Array.isArray(data.corrections_applied)
      ? data.corrections_applied.filter(c => c && typeof c === 'object' && c.property_name)
      : []
    correctionsExplanation = data.corrections_explanation || ""
  } else {
    // CASE 2: Evaluate/Validation Output - we submitted the composition
    comp = manualComp.value
    props = data.properties || {}
    propertyIntervals = data.property_intervals || {}
    confidence = data.confidence || {}
    explanation = data.explanation || ""
    auditPenalties = data.audit_penalties || []
    metallurgyMetrics = data.metallurgy_metrics || {}
    status = data.status || "UNKNOWN"
    tcpRisk = data.tcp_risk || metallurgyMetrics['TCP Risk'] || "UNKNOWN"
    similar = data.similar_alloys || []
    summary = data.summary || ""
    issues = data.issues || []
    recommendations = data.recommendations || []
    designStatus = data.design_status || ""
    reasoning = data.reasoning || ""
    correctionsApplied = Array.isArray(data.corrections_applied)
      ? data.corrections_applied.filter(c => c && typeof c === 'object' && c.property_name)
      : []
    correctionsExplanation = data.corrections_explanation || ""
  }

  // Extract agent reasoning fields (available in both modes)
  analystReasoning = data.analyst_reasoning || ""
  reviewerAssessment = data.reviewer_assessment || ""
  investigationFindings = data.investigation_findings || ""
  sourceReliability = data.source_reliability || ""

  // Filter out useless audit violations (no description or generic messages)
  issues = issues.filter(issue => {
    // Remove audit violations with "No description"
    if (issue.type === 'Audit Violation' && issue.description && issue.description.includes('No description')) {
      return false
    }
    // Remove issues with generic "Review composition constraints" as only recommendation
    if (issue.recommendation === 'Review composition constraints.' && !issue.description) {
      return false
    }
    return true
  })

  // Get list of properties shown in target achievement (to avoid duplicates)
  const shownInTargets = new Set()
  if (mode.value === 'auto' && propertyComparisons.value.length > 0) {
    propertyComparisons.value.forEach(comp => shownInTargets.add(comp.key))
  }

  // Format properties for display (exclude those already shown in target achievement)
    const formattedProps = [
    { label: "Yield Strength", val: parseVal(lookUpProp(props, "Yield Strength")), unit: "MPa", icon: "🏋️", hasTarget: targets.value?.yield > 0 },
    { label: "Tensile Strength", val: parseVal(lookUpProp(props, "Tensile Strength") || lookUpProp(props, "Ultimate Tensile Strength")), unit: "MPa", icon: "⛓️", hasTarget: targets.value?.tensile > 0 },
    { label: "Elongation", val: parseVal(lookUpProp(props, "Elongation")), unit: "%", icon: "📏", hasTarget: targets.value?.elongation > 0 },
    { label: "Elastic Modulus", val: parseVal(lookUpProp(props, "Elastic Modulus")), unit: "GPa", icon: "🔧", hasTarget: targets.value?.elastic_modulus > 0 },
    { label: "Density", val: parseVal(lookUpProp(props, "Density")), unit: "g/cm³", icon: "🧱", hasTarget: targets.value?.density < 99 },
    { label: "Gamma Prime", val: parseVal(lookUpProp(props, "Gamma Prime")), unit: "%", icon: "💎", hasTarget: targets.value?.gamma_prime > 0 },
  ]
    .filter(p => p.val !== undefined && p.val !== null)
    .filter(p => !shownInTargets.has(p.label))  // Remove properties already in target achievement
    .filter(p => mode.value === 'manual' || !p.hasTarget)  // In design mode, only show properties WITHOUT targets

  // Add interval data, overestimation warnings, and quality assessment to each property
    formattedProps.forEach(prop => {
    const interval = propertyIntervals[prop.label] || {}
    prop.lower = interval.lower
    prop.upper = interval.upper
    prop.uncertainty = interval.uncertainty
    
    // Calculate interval display (+/- value)
    if (prop.lower !== undefined && prop.upper !== undefined) {
      // Use 1 decimal for standard properties, 2 for Density (consistent with Target Achievement section)
      prop.interval = ((prop.upper - prop.lower) / 2).toFixed(prop.label.includes('Density') ? 2 : 1)
      prop.intervalPercent = (((prop.upper - prop.lower) / (2 * prop.val)) * 100).toFixed(0)
      prop.intervalQuality = getIntervalQuality(prop.lower, prop.upper, prop.val)
    }
    
    // Add confidence score
    prop.confidence = confidence?.score || 0.5
    
    // Add overestimation warning
    prop.overestimationWarning = getOverestimationWarning(prop, prop.val, prop.upper, prop.confidence)
  })
  
  // Separate physics metrics from audit penalties for dedicated display
  // Skip tcp_risk since it's displayed in the dedicated prediction-info-panel
  // Filter to ONLY valid metrics from MetallurgyVerifierTool (prevent LLM hallucinations)
  const VALID_METRICS = new Set([
    // Core metrics from MetallurgyVerifierTool
    'Md (TCP Stability)', 'TCP Risk', 'γ/γ\' Misfit (%)', 'Refractory Content (wt%)',
    'Matrix + SSS Strength (MPa)', 'Al+Ti (weldability)', 'Cr (oxidation)',
    // Legacy/alternative formats that may appear
    'md_average', 'md_avg', 'gamma_prime_vol', 'gamma_prime_fraction',
    'sss_wt_pct', 'lattice_misfit', 'density_gcm3',
    // KG-derived metrics
    'kg_md_avg', 'kg_tcp_risk', 'kg_sss_wt_pct'
  ])
  const physicsMetrics = []
  if (metallurgyMetrics) {
    Object.entries(metallurgyMetrics).forEach(([key, value]) => {
      // Skip tcp_risk - already shown in dedicated TCP tag
      if (key === 'tcp_risk' || key === 'TCP Risk') return

      // Filter out hallucinated metrics (e.g., "Grain Size", "Inclusion Content")
      if (!VALID_METRICS.has(key)) {
        console.warn(`Filtered out unknown metric: ${key}`)
        return
      }

      let displayValue = typeof value === 'number' ? value.toFixed(2) : value
      let warning = null

      // Add contextual warnings for materials scientists
      if (key.includes('weldability') && typeof value === 'number') {
        if (value > 6.5) warning = '⚠️ Difficult to weld'
        else if (value > 5.0) warning = '⚡ Weld with care'
      }
      if (key.includes('oxidation') && typeof value === 'number') {
        if (value >= 18) displayValue = value.toFixed(1) + ' (excellent)'
        else if (value >= 15) displayValue = value.toFixed(1) + ' (good)'
        else if (value >= 10) displayValue = value.toFixed(1) + ' (moderate)'
        else displayValue = value.toFixed(1) + ' (limited)'
      }
      if (key.includes('Misfit') && typeof value === 'number') {
        if (Math.abs(value) < 0.3) displayValue = value.toFixed(3) + ' (optimal)'
        else if (Math.abs(value) < 0.5) displayValue = value.toFixed(3) + ' (good)'
        else displayValue = value.toFixed(3) + ' (high)'
      }

      physicsMetrics.push({
        label: formatMetricLabel(key),
        value: displayValue,
        warning,
        key
      })
    })
  }

  return {
    comp,
    formattedProps,
    propertyIntervals,
    confidence,
    explanation,
    auditPenalties,
    metallurgyMetrics,
    physicsMetrics,
    status,
    tcpRisk,
    similar,
    summary,
    issues,
    recommendations,
    designStatus,
    reasoning,
    analystReasoning,
    reviewerAssessment,
    investigationFindings,
    sourceReliability,
    correctionsApplied,
    correctionsExplanation
  }
})

</script>

<template>
  <div class="alloy-designer">
    <!-- Modern Mode Switcher with History Button -->
    <div class="mode-switcher glass-card">
      <button
        :class="['mode-btn', { active: mode === 'manual' }]"
        @click="mode = 'manual'"
      >
        <span class="mode-icon">🧪</span>
        <span class="mode-label">Evaluate</span>
      </button>
      <button
        :class="['mode-btn', { active: mode === 'auto' }]"
        @click="mode = 'auto'"
      >
        <span class="mode-icon">🧬</span>
        <span class="mode-label">Design</span>
      </button>
      <button
        class="history-toggle-btn"
        @click="showHistory = !showHistory"
        :title="showHistory ? 'Hide History' : 'Show History'"
      >
        <span class="mode-icon">📜</span>
        <span class="mode-label">History</span>
        <span v-if="designHistory.length > 0" class="history-badge">{{ designHistory.length }}</span>
      </button>
    </div>

    <!-- DESIGN HISTORY PANEL -->
    <transition name="slide-down">
      <div v-if="showHistory" class="history-panel glass-card">
        <div class="history-header">
          <h3>Design History ({{ designHistory.length }}/{{ maxHistoryItems }})</h3>
          <div class="history-actions">
            <button @click="exportHistory" class="small-btn" :disabled="designHistory.length === 0">
              📥 Export
            </button>
            <button @click="clearHistory" class="small-btn danger" :disabled="designHistory.length === 0">
              🗑️ Clear
            </button>
          </div>
        </div>
        <div v-if="designHistory.length === 0" class="empty-history">
          <p>No design history yet. Run a validation or design to start building your history.</p>
        </div>
        <div v-else class="history-list">
          <div
            v-for="item in designHistory"
            :key="item.id"
            class="history-item"
            @click="loadFromHistory(item)"
          >
            <div class="history-item-header">
              <span class="history-mode-badge" :class="item.mode">
                {{ item.mode === 'manual' ? '🧪 Evaluate' : '🧬 Design' }}
              </span>
              <span class="history-timestamp">
                {{ new Date(item.timestamp).toLocaleString() }}
              </span>
            </div>
            <div class="history-item-details">
              <div class="history-comp-preview">
                <span v-for="(val, key) in item.composition" :key="key" class="mini-tag">
                  {{ key }}: {{ Number(val).toFixed(1) }}%
                </span>
              </div>
              <div class="history-conditions">
                {{ item.temperature }}°C | {{ item.processing }}
              </div>
            </div>
          </div>
        </div>
      </div>
    </transition>

    <!-- MANUAL MODE: EVALUATE COMPOSITION -->
    <div v-if="mode === 'manual'" class="panel glass-card">
      <h3>Define Alloy Composition</h3>
      <p class="helper-text">Enter element percentages (should sum to ~100%). Use presets or add custom elements.</p>
      
      <!-- Quick Presets & Actions -->
      <div class="presets-section">
        <label class="section-label">Quick Start:</label>
        <div class="preset-buttons">
          <template v-for="(preset, name) in allPresets" :key="name">
            <div class="preset-item">
              <button @click="loadPreset(name)"
                      :class="['preset-btn', { 'preset-selected': selectedPreset === name, 'custom-preset': !preset.builtin }]">
                {{ name }}
              </button>
              <button v-if="!preset.builtin"
                      @click.stop="deletePreset(name)"
                      class="preset-delete-btn"
                      title="Delete preset">×</button>
            </div>
          </template>
          <span class="preset-divider">|</span>
          <button @click="openSavePreset" class="preset-btn action-btn" title="Save current composition as preset">
            💾 Save Preset
          </button>
          <button @click="openJsonImport" class="preset-btn action-btn" title="Import composition from JSON">
            📋 Import JSON
          </button>
          <button @click="clearComposition" class="preset-btn action-btn clear-btn" title="Clear all elements">
            🗑️ Clear
          </button>
        </div>
      </div>
      
      <!-- Composition Grid -->
      <div class="section-label">Composition (wt%):</div>
      <div class="comp-grid">
        <div v-for="(val, el) in manualComp" :key="el" class="element-box">
          <label>{{ el }}</label>
          <input type="number" v-model.number="manualComp[el]" step="0.1" />
          <span class="remove" @click="removeElement(el)">×</span>
        </div>
        <div class="add-box">
          <input type="text" placeholder="+ Add Element" @keyup.enter="addElement($event.target.value); $event.target.value=''" class="add-input" />
        </div>
      </div>

      <!-- Bottom Actions -->
      <div class="eval-controls">
        <div class="temp-inline">
          <label>Temperature:</label>
          <input type="number" v-model.number="manualTemp" class="temp-simple" />
          <span class="temp-unit">°C</span>
          
          <label style="margin-left: var(--space-lg);">Processing:</label>
          <select v-model="manualProcessing" class="temp-simple" style="width: 100px">
            <option value="cast">Cast</option>
            <option value="wrought">Wrought</option>
          </select>
        </div>
        
        <button @click="runValidation" :disabled="loading" class="primary-btn">
          {{ loading ? 'Running Models...' : '▶ Predict Properties' }}
        </button>
      </div>
    </div>

    <!-- AUTO MODE: ALLOY DESIGN -->
    <div v-if="mode === 'auto'" class="panel glass-card">
      <h3>Define Target Properties</h3>
      <p class="helper-text">Specify target values for desired properties. Set to 0 to skip any property.</p>
      
      <div class="target-grid">
        <!-- Yield Strength -->
        <div class="field">
          <label>Min Yield Strength <span class="default-hint">(MPa)</span></label>
          <input v-model.number="targets.yield" type="number" placeholder="1200" />
        </div>

        <!-- Tensile Strength -->
        <div class="field">
          <label>Min Tensile Strength <span class="default-hint">(MPa)</span></label>
          <input v-model.number="targets.tensile" type="number" placeholder="0" />
        </div>

        <!-- Elongation -->
        <div class="field">
          <label>Min Elongation <span class="default-hint">(%)</span></label>
          <input v-model.number="targets.elongation" type="number" step="0.1" placeholder="0" />
        </div>

        <!-- Elastic Modulus -->
        <div class="field">
          <label>Min Elastic Modulus <span class="default-hint">(GPa)</span></label>
          <input v-model.number="targets.elastic_modulus" type="number" step="1" placeholder="0" />
        </div>

        <!-- Density -->
        <div class="field">
          <label>Max Density <span class="default-hint">(g/cm³)</span></label>
          <input v-model.number="targets.density" type="number" step="0.1" placeholder="8.5" />
        </div>

        <!-- Gamma Prime -->
        <div class="field">
          <label>
            Target Gamma Prime
            <span class="default-hint">(vol%)</span>
          </label>
          <input v-model.number="targets.gamma_prime" type="number" step="0.1" placeholder="0 = not specified" />
        </div>
      </div>

      <!-- Bottom Controls -->
      <div class="design-controls">
        <div class="control-row">
          <div class="inline-control">
            <label>Temperature:</label>
            <input type="number" v-model.number="autoTemp" class="small-input" />
            <span class="unit">°C</span>
          </div>
          
          <div class="inline-control">
            <label>Processing:</label>
            <select v-model="autoProcessing" class="small-select">
              <option value="cast">Cast</option>
              <option value="wrought">Wrought</option>
            </select>
          </div>
          
          <div class="inline-control">
            <label>Max Iterations:</label>
            <input type="number" v-model.number="autoIterations" min="1" max="10" class="small-input" style="width: 60px" />
          </div>
        </div>
        
        <button @click="runDesign" :disabled="loading" class="primary-btn magic-btn">
          {{ loading ? 'Inventing Alloy...' : '✨ Auto-Design Alloy' }}
        </button>
      </div>
    </div>

    <!-- OUTPUT RESULTS DASHBOARD -->
    <div class="output-area">
      <!-- ERROR BOUNDARY -->
      <div v-if="error && !loading" class="error-boundary glass-card">
        <div class="error-header">
          <span class="error-icon">❌</span>
          <h3>{{ errorType === 'network' ? 'Connection Error' : errorType === 'timeout' ? 'Timeout Error' : errorType === 'validation' ? 'Validation Error' : errorType === 'server' ? 'Server Error' : 'Unexpected Error' }}</h3>
        </div>
        <div class="error-message">{{ error }}</div>
        <div class="error-recovery">
          <h4>Recovery Actions:</h4>
          <ul>
            <li v-for="(action, idx) in getErrorRecoveryActions(errorType)" :key="idx">
              {{ action }}
            </li>
          </ul>
        </div>
        <div class="error-actions">
          <button
            @click="mode === 'manual' ? retryValidation() : retryDesign()"
            :disabled="retryCount >= maxRetries"
            class="retry-btn"
          >
            🔄 Retry {{ retryCount > 0 ? `(${retryCount}/${maxRetries})` : '' }}
          </button>
          <button @click="clearError" class="dismiss-btn">
            ✖️ Dismiss
          </button>
        </div>
        <div v-if="logs.length > 0" class="error-logs-scroll">
          <div v-for="(log, i) in logs" :key="i" class="log-line">{{ log }}</div>
        </div>
      </div>

      <!-- LOADING STATE WITH PIPELINE INDICATOR -->
      <div v-if="loading" class="loading-state glass-card">
        <div class="pipeline-header">
          <div class="pipeline-spinner"></div>
          <span class="pipeline-title">{{ mode === 'manual' ? 'Evaluating Alloy' : 'Designing Alloy' }}</span>
        </div>
        <div class="pipeline-active-step">
          <span class="active-step-icon">{{ currentSteps[loadingStep]?.icon }}</span>
          <span class="active-step-label">{{ currentSteps[loadingStep]?.label }}</span>
          <span class="active-step-dots"><span class="dot-anim">...</span></span>
        </div>
        <div class="pipeline-track">
          <div
            v-for="(step, i) in currentSteps"
            :key="i"
            :class="['pipeline-dot', { active: i === loadingStep }]"
          ></div>
        </div>
        <div class="pipeline-footer">
          <span class="elapsed-time">{{ elapsedSeconds }}s elapsed</span>
        </div>
        <div v-if="logs.length > 0" class="logs-scroll">
          <div v-for="(log, i) in logs" :key="i" class="log-line">{{ log }}</div>
        </div>
      </div>

      <div v-if="parsedResults" class="results-dashboard animate-in">
        <div class="dashboard-header">
           <h3>🎉 Analysis Complete at {{ result.temperature || (mode === 'manual' ? manualTemp : autoTemp) }}°C</h3>
           <p class="summary-text">{{ parsedResults.summary }}</p>
        </div>

        <!-- STATUS BADGE (for both Design and Evaluate modes) -->
        <div v-if="parsedResults.status && parsedResults.status !== 'UNKNOWN'" class="status-badge" :class="parsedResults.status.toLowerCase()">
          {{ parsedResults.status === 'PASS' ? '✅ PASS' : parsedResults.status === 'REJECT' ? '⚠️ REJECT' : '❌ FAIL' }}
        </div>

        <!-- PREDICTION INFO PANEL - Only show when there's useful info -->
        <div v-if="hasUsefulPredictionInfo(parsedResults)" class="prediction-info-panel">
          <!-- Similar Alloy Match - only when found -->
          <span v-if="parsedResults.confidence?.matched_alloy && parsedResults.confidence.matched_alloy !== 'None'" class="info-tag similar-alloy">
            🔗 Similar to <strong>{{ parsedResults.confidence.matched_alloy }}</strong>
          </span>
          <!-- TCP Risk - only when elevated or higher -->
          <span v-if="parsedResults.tcpRisk && parsedResults.tcpRisk !== 'Low' && parsedResults.tcpRisk !== 'UNKNOWN'"
                :class="['info-tag', 'tcp-' + parsedResults.tcpRisk.toLowerCase()]">
            {{ getTcpEmoji(parsedResults.tcpRisk) }} TCP: {{ parsedResults.tcpRisk }}
          </span>
        </div>

        <!-- 1. COMPOSITION FIRST (for Design mode) -->
        <div v-if="parsedResults.comp && mode === 'auto'" class="final-comp-section">
             <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
               <h4 style="margin: 0;">🧬 Suggested Composition</h4>
               <button @click="copyToEvaluation" class="copy-btn" title="Copy to Evaluation Mode">
                 📋 Copy to Evaluation
               </button>
             </div>
             <div class="mini-comp-grid">
                <span v-for="(v, k) in parsedResults.comp" :key="k" class="mini-comp-tag">
                   <b>{{ k }}</b>: {{ v }}%
                </span>
             </div>
        </div>

        <!-- 1.5 TARGET VS ACTUAL COMPARISON (Design mode only) -->
        <div v-if="propertyComparisons.length > 0 && mode === 'auto'" class="comparison-section">
          <h4>📊 Target Achievement</h4>
          <div class="comparison-list">
            <div v-for="comp in propertyComparisons" :key="comp.key" class="comparison-item">
              <div class="comparison-header">
                <span class="comparison-icon">{{ comp.icon }}</span>
                <span class="comparison-label">{{ comp.key }}</span>
                <span :class="['comparison-status', { met: comp.met, unmet: !comp.met && !comp.exceeds, exceeds: comp.exceeds }]">
                  {{ comp.met ? '✓ ' + comp.status : (comp.exceeds ? '⚠ ' + comp.status : '✗ ' + comp.status) }}
                </span>
              </div>
              <div class="comparison-values">
                <span class="comparison-target">Target: {{ comp.isMax ? '≤' : '≥' }} {{ comp.target }} {{ comp.unit }}</span>
                <span class="comparison-actual">
                  Predicted: {{ comp.actual.toFixed(comp.key.includes('Density') ? 2 : 1) }} {{ comp.unit }}
                  <!-- Discrete confidence interval display as ±value -->
                  <span v-if="comp.plusMinus" class="comparison-interval-discrete">
                    ±{{ comp.plusMinus }}
                  </span>
                </span>
              </div>
              <div class="comparison-bar-container">
                <div
                  :class="['comparison-bar', { met: comp.met, unmet: !comp.met && !comp.exceeds, exceeds: comp.exceeds }]"
                  :style="{ width: Math.min(comp.percentage, 100) + '%' }"
                >
                  <span class="comparison-percentage">{{ comp.percentage }}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>

                <!-- 2. ADDITIONAL PROPERTIES (properties without targets) -->
        <div v-if="parsedResults.formattedProps && parsedResults.formattedProps.length > 0" class="props-grid">
            <div v-for="prop in parsedResults.formattedProps" :key="prop.label"
                 :class="['prop-card', getConfidenceClass(prop.confidence)]">
               <div class="prop-icon">{{ prop.icon }}</div>
               <div class="prop-info">
                  <div class="prop-label">{{ prop.label }}</div>
                  <div class="prop-val">
                    {{ Number(prop.val).toFixed(prop.label.includes('Density') ? 2 : 1) }}
                    <small>{{ prop.unit }}</small>
                  </div>
                  <!-- Discrete interval display -->
                  <div v-if="prop.interval" class="prop-interval-discrete">
                    ±{{ prop.interval }} {{ prop.unit }}
                  </div>
               </div>
            </div>
        </div>

                <!-- DESIGN ISSUES (from design logic only, no audit duplication) -->
        <div v-if="parsedResults.issues && parsedResults.issues.length > 0" class="issues-panel">
          <div class="panel-header">⚠️ Design Issues</div>
          <div class="issues-list">
            <div v-for="(issue, i) in parsedResults.issues" :key="'issue-'+i" :class="['issue-item', 'severity-' + issue.severity.toLowerCase()]">
              <div class="issue-header">
                <span class="issue-icon">
                  {{ issue.severity === 'High' ? '🔴' : issue.severity === 'Medium' ? '🟡' : '🔵' }}
                </span>
                <span class="issue-type">{{ issue.type }}</span>
                <span class="issue-severity">{{ issue.severity }}</span>
              </div>
              <div class="issue-description">{{ issue.description }}</div>
              <div class="issue-recommendation">💡 {{ issue.recommendation }}</div>
            </div>
          </div>
        </div>

        <!-- PHYSICS AUDIT VIOLATIONS (when status is REJECT/FAIL) -->
        <div v-if="parsedResults.auditPenalties && parsedResults.auditPenalties.length > 0 && (parsedResults.status === 'REJECT' || parsedResults.status === 'FAIL')" class="issues-panel">
          <div class="panel-header">⚠️ Physics Audit Violations</div>
          <div class="issues-list">
            <div v-for="(penalty, i) in parsedResults.auditPenalties" :key="'penalty-'+i" class="issue-item severity-high">
              <div class="issue-header">
                <span class="issue-icon">🔴</span>
                <span class="issue-type">{{ penalty.name }}</span>
                <span class="issue-severity">{{ penalty.value }}</span>
              </div>
              <div class="issue-description">{{ penalty.reason }}</div>
            </div>
          </div>
        </div>

        <!-- METALLURGICAL INDICATORS - always show when available -->
        <div v-if="parsedResults.physicsMetrics && parsedResults.physicsMetrics.length > 0" class="metrics-panel">
          <div class="panel-header">🔬 Metallurgical Indicators</div>
          <div class="metrics-grid">
            <div v-for="metric in parsedResults.physicsMetrics" :key="metric.key" class="metric-item">
              <span class="metric-label">{{ metric.label }}:</span>
              <span class="metric-value">{{ metric.value }}</span>
              <span v-if="metric.warning" class="metric-warning">{{ metric.warning }}</span>
            </div>
          </div>
        </div>

        <!-- CORRECTIONS APPLIED (when agents made physics-based corrections) -->
        <div v-if="parsedResults.correctionsApplied && parsedResults.correctionsApplied.length > 0" class="corrections-panel">
          <div class="panel-header">🔧 Physics Corrections Applied</div>
          <div class="corrections-list">
            <div v-for="(corr, i) in parsedResults.correctionsApplied" :key="'corr-'+i" class="correction-item">
              <div class="correction-header">
                <span class="correction-prop">{{ corr.property_name }}</span>
                <span class="correction-arrow">
                  {{ Number(corr.original_value).toFixed(1) }} → {{ Number(corr.corrected_value).toFixed(1) }}
                </span>
              </div>
              <div class="correction-reason">{{ corr.correction_reason }}</div>
              <div v-if="corr.physics_constraint" class="correction-constraint">{{ corr.physics_constraint }}</div>
            </div>
          </div>
          <div v-if="parsedResults.correctionsExplanation" class="corrections-summary">
            {{ parsedResults.correctionsExplanation }}
          </div>
        </div>

        <!-- AGENT INVESTIGATION (Analyst + Reviewer reasoning) -->
        <div v-if="parsedResults.analystReasoning || parsedResults.reviewerAssessment || parsedResults.investigationFindings || parsedResults.sourceReliability" class="agent-reasoning-section">
          <div class="panel-header">🧠 Agent Investigation</div>

          <!-- Source Reliability Badge -->
          <div v-if="parsedResults.sourceReliability" class="source-reliability-badge">
            <span class="reliability-label">Source Reliability:</span>
            <span class="reliability-value">{{ parsedResults.sourceReliability }}</span>
          </div>

          <!-- Investigation Findings -->
          <div v-if="parsedResults.investigationFindings" class="investigation-findings">
            <div class="sub-header">🔍 Investigation Findings</div>
            <div class="findings-text">{{ parsedResults.investigationFindings }}</div>
          </div>

          <!-- Analyst Reasoning -->
          <div v-if="parsedResults.analystReasoning" class="analyst-section">
            <div class="sub-header">📊 Analyst Reasoning</div>
            <div class="reasoning-text">{{ parsedResults.analystReasoning }}</div>
          </div>

          <!-- Reviewer Assessment -->
          <div v-if="parsedResults.reviewerAssessment" class="reviewer-section">
            <div class="sub-header">🔎 Reviewer Assessment</div>
            <div class="assessment-text">{{ parsedResults.reviewerAssessment }}</div>
          </div>
        </div>

        <!-- AI DESIGN REASONING (design mode only) -->
        <div v-if="parsedResults.reasoning" class="reasoning-panel">
          <div class="panel-header">🤖 AI Design Reasoning</div>
          <div class="reasoning-text">{{ parsedResults.reasoning }}</div>
        </div>

        <!-- METALLURGICAL ANALYSIS -->
        <div v-if="parsedResults.explanation" class="explanation-panel">
          <div class="panel-header">💬 Metallurgical Analysis</div>
          <div class="explanation-text">{{ parsedResults.explanation }}</div>
        </div>

        <!-- SIMILAR ALLOYS -->
        <div v-if="parsedResults.similar.length" class="similar-section">
           <h4>🔍 Similar Known Alloys</h4>
           <div class="similar-list">
              <div v-for="alloy in parsedResults.similar" :key="alloy.name" class="similar-item">
                 <strong>{{ alloy.name }}</strong>
                 <span v-if="alloy.similarity" class="similarity-badge">{{ alloy.similarity }}</span>
                 <p class="similar-notes">{{ alloy.notes || JSON.stringify(alloy) }}</p>
              </div>
           </div>
        </div>

      </div>
    </div>
  </div>

  <!-- JSON Import Modal -->
  <transition name="modal-fade">
    <div v-if="showJsonImport" class="modal-overlay" @click.self="closeJsonImport">
      <div class="modal-content json-import-modal">
        <div class="modal-header">
          <h3>📋 Import Composition from JSON</h3>
          <button class="modal-close" @click="closeJsonImport">×</button>
        </div>

        <div class="modal-body">
          <p class="modal-help">
            Paste a JSON object with element symbols and weight percentages.
          </p>

          <div class="json-examples">
            <span class="example-label">Examples:</span>
            <code>{"Ni": 60, "Cr": 20, "Al": 5}</code>
            <code>{"composition": {"Ni": 58, "Co": 13}}</code>
          </div>

          <textarea
            v-model="jsonInput"
            class="json-textarea"
            placeholder='{"Ni": 60, "Cr": 20, "Co": 10, "Al": 5, "Ti": 3}'
            rows="6"
            @keydown.ctrl.enter="importJsonComposition"
            @keydown.meta.enter="importJsonComposition"
          ></textarea>

          <p v-if="jsonError" class="json-error">{{ jsonError }}</p>
        </div>

        <div class="modal-footer">
          <button class="cancel-btn" @click="closeJsonImport">Cancel</button>
          <button class="import-btn" @click="importJsonComposition">Import Composition</button>
        </div>
      </div>
    </div>
  </transition>

  <!-- Save Preset Modal -->
  <transition name="modal-fade">
    <div v-if="showSavePreset" class="modal-overlay" @click.self="showSavePreset = false">
      <div class="modal-content save-preset-modal">
        <div class="modal-header">
          <h3>💾 Save as Preset</h3>
          <button class="modal-close" @click="showSavePreset = false">×</button>
        </div>

        <div class="modal-body">
          <p class="modal-help">
            Save the current composition as a custom preset for quick access later.
          </p>

          <div class="preset-name-input">
            <label>Preset Name:</label>
            <input
              type="text"
              v-model="newPresetName"
              placeholder="My Custom Alloy"
              @keydown.enter="saveAsPreset"
              autofocus
            />
          </div>

          <div class="preset-preview">
            <span class="preview-label">Composition:</span>
            <span class="preview-elements">
              {{ Object.entries(manualComp).map(([el, val]) => `${el}: ${val}%`).join(', ') }}
            </span>
          </div>
        </div>

        <div class="modal-footer">
          <button class="cancel-btn" @click="showSavePreset = false">Cancel</button>
          <button class="import-btn" @click="saveAsPreset" :disabled="!newPresetName.trim()">Save Preset</button>
        </div>
      </div>
    </div>
  </transition>
</template>

<style scoped>




.panel {
  background: #252526;
  padding: 1.5rem;
  border-radius: 8px;
}

/* === SECTION LABELS === */
.section-label {
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-semibold);
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: var(--space-sm);
  display: block;
}

/* === PRESETS SECTION === */
.presets-section {
  margin-bottom: var(--space-lg);
  padding: var(--space-md);
  background: rgba(255, 255, 255, 0.02);
  border-radius: var(--radius-md);
  border: 1px solid rgba(255, 255, 255, 0.05);
}

.preset-buttons {
  display: flex;
  gap: var(--space-sm);
  flex-wrap: wrap;
  margin-top: var(--space-sm);
}

.preset-btn {
  padding: var(--space-sm) var(--space-md);
  background: var(--bg-glass);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: var(--radius-md);
  color: var(--text-primary);
  cursor: pointer;
  transition: all var(--transition-base);
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-medium);
}

.preset-btn:hover {
  background: rgba(255, 255, 255, 0.08);
  border-color: rgba(255, 255, 255, 0.2);
  transform: translateY(-1px);
}

.preset-btn.preset-selected {
  background: rgba(0, 212, 255, 0.2);
  border-color: #00d4ff;
  color: #00d4ff;
}

.preset-btn.action-btn {
  background: var(--bg-glass);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.preset-btn.action-btn:hover {
  background: rgba(255, 255, 255, 0.08);
  border-color: rgba(255, 255, 255, 0.2);
  box-shadow: none;
}

/* Preset item with delete button */
.preset-item {
  position: relative;
  display: inline-flex;
  align-items: center;
}

.preset-delete-btn {
  position: absolute;
  top: -6px;
  right: -6px;
  width: 18px;
  height: 18px;
  padding: 0;
  background: #ff4757;
  border: none;
  border-radius: 50%;
  color: white;
  font-size: 12px;
  font-weight: bold;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.preset-item:hover .preset-delete-btn {
  opacity: 1;
}

.preset-delete-btn:hover {
  background: #ff2f4a;
  transform: scale(1.1);
}

.preset-btn.custom-preset {
  background: rgba(255, 215, 0, 0.1);
  border-color: rgba(255, 215, 0, 0.3);
}

.preset-btn.custom-preset:hover {
  background: rgba(255, 215, 0, 0.2);
  border-color: rgba(255, 215, 0, 0.5);
}

.preset-btn.custom-preset.preset-selected {
  background: rgba(255, 215, 0, 0.3);
  border-color: #ffd700;
  color: #ffd700;
}

/* Save Preset Modal */
.save-preset-modal {
  max-width: 400px;
}

.preset-name-input {
  margin-bottom: var(--space-md);
}

.preset-name-input label {
  display: block;
  margin-bottom: var(--space-xs);
  color: var(--text-secondary);
  font-size: var(--font-size-sm);
}

.preset-name-input input {
  width: 100%;
  padding: var(--space-sm) var(--space-md);
  background: var(--bg-glass);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: var(--radius-md);
  color: var(--text-primary);
  font-size: var(--font-size-base);
}

.preset-name-input input:focus {
  outline: none;
  border-color: #00d4ff;
}

.preset-preview {
  padding: var(--space-sm);
  background: rgba(255, 255, 255, 0.03);
  border-radius: var(--radius-sm);
  font-size: var(--font-size-sm);
}

.preset-preview .preview-label {
  color: var(--text-muted);
  margin-right: var(--space-xs);
}

.preset-preview .preview-elements {
  color: var(--text-secondary);
}

/* === EVAL CONTROLS === */
.eval-controls {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-lg);
  margin-top: var(--space-xl);
  padding-top: var(--space-lg);
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.temp-inline {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  font-size: var(--font-size-sm);
  color: var(--text-secondary);
}

.temp-simple {
  width: 80px;
  background: var(--bg-glass);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: var(--radius-md);
  color: var(--text-primary);
  padding: var(--space-xs) var(--space-sm);
  font-size: var(--font-size-sm);
}

.temp-inline .temp-unit {
  color: var(--text-muted);
  font-size: var(--font-size-sm);
}





.comp-grid, .target-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  gap: 10px;
  margin-bottom: 1.5rem;
}

.element-box, .field {
  background: #1e1e1e;
  padding: 8px;
  border-radius: 6px;
  border: 1px solid #333;
  position: relative;
}
.element-box label, .field label {
  display: block;
  font-size: 0.8rem;
  color: #aaa;
  margin-bottom: 4px;
}
.element-box input, .field input {
  width: 100%;
  background: transparent;
  border: none;
  color: white;
  font-size: 1.1rem;
  font-weight: bold;
}
.remove {
  position: absolute;
  top: 2px;
  right: 6px;
  color: #555;
  cursor: pointer;
}
.remove:hover { color: red; }

.add-input {
  width: 100%;
  height: 100%;
  background: #1e1e1e;
  border: 1px dashed #555;
  color: #fff;
  text-align: center;
  border-radius: 6px;
}



.primary-btn {
  background: #007bff;
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 6px;
  font-size: 1rem;
  font-weight: bold;
  cursor: pointer;
  transition: transform 0.1s;
}
.primary-btn:hover { background: #0056b3; transform: scale(1.02); }
.primary-btn:disabled { background: #444; cursor: not-allowed; }

.magic-btn {
  background: linear-gradient(135deg, #6610f2, #d63384);
}

.copy-btn {
  background: #28a745;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 6px;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.copy-btn:hover {
  background: #218838;
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(40, 167, 69, 0.3);
}

/* MODAL STYLES */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
  backdrop-filter: blur(5px);
}

.modal-content {
  width: 90%;
  max-width: 600px;
  max-height: 85vh;
  overflow-y: auto;
  padding: 0;
  border: 1px solid #444;
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.5rem;
  border-bottom: 1px solid #333;
  background: rgba(255, 255, 255, 0.05);
}

.modal-header h3 {
  margin: 0;
  color: #00d4ff;
}

.close-btn {
  background: none;
  border: none;
  color: #888;
  font-size: 2rem;
  cursor: pointer;
  line-height: 1;
}

.close-btn:hover {
  color: #fff;
}

.modal-body {
  padding: 2rem;
  color: #ddd;
  line-height: 1.6;
}

.modal-body h4 {
  color: #ffd700;
  margin-top: 1.5rem;
  margin-bottom: 0.8rem;
  border-bottom: 1px solid #333;
  padding-bottom: 0.5rem;
}
.modal-body h4:first-child { margin-top: 0; }

.modal-body ul {
  padding-left: 1.5rem;
  margin-bottom: 1rem;
}

.modal-body li {
  margin-bottom: 0.5rem;
}

/* OUTPUT DASHBOARD STYLES */
.output-area {
  background: #111;
  padding: 1.5rem;
  border-radius: 12px;
  flex: 1;
  min-height: 300px;
  border: 1px solid #333;
}

.loading-state {
  text-align: center;
  padding: 2rem;
}
.spinner { 
  font-size: 1.5rem; 
  color: #ffd700; 
  margin-bottom: 1rem;
  animation: pulse 1s infinite;
}
@keyframes pulse { 0% { opacity: 0.5; } 50% { opacity: 1; } 100% { opacity: 0.5; } }

.logs-scroll {
  max-height: 150px;
  overflow-y: auto;
  text-align: left;
  font-family: monospace;
  font-size: 0.8rem;
  color: #666;
  border-top: 1px solid #222;
  padding-top: 10px;
}

.results-dashboard {
  color: #fff;
}
.dashboard-header {
  border-bottom: 1px solid #333;
  padding-bottom: 1rem;
  margin-bottom: 1.5rem;
}
.summary-text {
  font-style: italic;
  color: #aaa;
  margin-top: 0.5rem;
  font-size: 1.1rem;
  line-height: 1.4;
}

.props-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}
.prop-card {
  background: #222;
  padding: 1rem;
  border-radius: 10px;
  display: flex;
  align-items: center;
  gap: 1rem;
  border: 1px solid #333;
}

.prop-interval-discrete {
  color: var(--text-muted);
  font-size: 0.75rem;
  opacity: 0.7;
  font-style: italic;
  margin-top: 0.25rem;
}

.mode-icon {
  font-size: 1.5rem;
}

.mode-label {
  font-size: var(--font-size-md);
  font-weight: var(--font-weight-semibold);
}

/* === MODERN MODE SWITCHER === */
.mode-switcher {
  display: flex;
  flex-direction: row;
  gap: var(--space-md);
  padding: var(--space-sm);
  margin-bottom: var(--space-xl);
}

/* === HELPER TEXT === */
.helper-text {
  color: var(--text-secondary);
  font-size: var(--font-size-sm);
  margin-bottom: var(--space-md);
  line-height: 1.5;
}

/* === REQUIRED/OPTIONAL INDICATORS === */
.required {
  color: var(--danger);
  font-weight: var(--font-weight-bold);
  margin-right: var(--space-xs);
}

.optional {
  color: var(--text-muted);
  font-size: var(--font-size-xs);
  font-style: italic;
  margin-right: var(--space-xs);
}

/* === DEFAULT HINTS === */
.default-hint {
  color: var(--text-muted);
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-normal);
  font-style: italic;
}

/* === DESIGN CONTROLS === */
.design-controls {
  margin-top: var(--space-xl);
  padding-top: var(--space-lg);
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
}

.control-row {
  display: flex;
  align-items: center;
  gap: var(--space-lg);
  flex-wrap: wrap;
}

.inline-control {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  font-size: var(--font-size-sm);
  color: var(--text-secondary);
}

.small-input, .small-select {
  width: 100px;
  background: var(--bg-glass);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: var(--radius-md);
  color: var(--text-primary);
  padding: var(--space-xs) var(--space-sm);
  font-size: var(--font-size-sm);
}

.inline-control .unit {
  color: var(--text-muted);
  font-size: var(--font-size-sm);
}

/* === RANGE VALUE DISPLAY === */
.range-value {
  display: inline-block;
  min-width: 2rem;
  text-align: center;
  color: var(--primary);
  font-weight: var(--font-weight-semibold);
  margin-left: var(--space-sm);
}
.prop-icon { font-size: 2rem; }
.prop-val { font-size: 1.4rem; font-weight: bold; color: #00d4ff; }
.prop-val small { font-size: 0.8rem; color: #666; font-weight: normal; }
.prop-label { font-size: 0.9rem; color: #888; }
.prop-interval {
  font-size: 0.75rem;
  color: #999;
  margin-top: 4px;
}
.prop-uncertainty {
  font-size: 0.7rem;
  color: #aaa;
  font-style: italic;
}

/* Status Badge */
.status-badge {
  display: inline-block;
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  font-weight: 900;
  font-size: 1.3rem;
  margin-bottom: 1.5rem;
  text-transform: uppercase;
  letter-spacing: 1px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  border: 2px solid transparent;
}
.status-badge.pass {
  background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
  color: white;
  border-color: #20c997;
  animation: pulse-green 2s infinite;
}
.status-badge.reject {
  background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%);
  color: #1a1a1a;
  border-color: #ff9800;
  font-weight: 900;
  animation: pulse-orange 2s infinite;
}
.status-badge.fail {
  background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
  color: white;
  border-color: #c82333;
  animation: pulse-red 2s infinite;
}

@keyframes pulse-green {
  0%, 100% { box-shadow: 0 4px 12px rgba(40, 167, 69, 0.4); }
  50% { box-shadow: 0 4px 20px rgba(40, 167, 69, 0.7); }
}
@keyframes pulse-orange {
  0%, 100% { box-shadow: 0 4px 12px rgba(255, 193, 7, 0.5); }
  50% { box-shadow: 0 4px 20px rgba(255, 193, 7, 0.8); }
}
@keyframes pulse-red {
  0%, 100% { box-shadow: 0 4px 12px rgba(220, 53, 69, 0.4); }
  50% { box-shadow: 0 4px 20px rgba(220, 53, 69, 0.7); }
}

/* Panels */
.explanation-panel,
.audit-panel,
.metrics-panel {
  background: #2a2a2a;
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 1.5rem;
  border-left: 3px solid #007bff;
}
.panel-header {
  font-size: 1rem;
  font-weight: bold;
  color: #00d4ff;
  margin-bottom: 0.75rem;
}
.confidence-details {
  font-size: 0.9rem;
  color: #ccc;
  line-height: 1.6;
}

/* Prediction Info Panel - compact inline tags */
.prediction-info-panel {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-bottom: 1rem;
}
.info-tag {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.4rem 0.75rem;
  border-radius: 6px;
  font-size: 0.85rem;
  background: rgba(255, 255, 255, 0.08);
  color: #ddd;
}
.info-tag.similar-alloy {
  background: rgba(0, 212, 255, 0.15);
  color: #00d4ff;
}
.info-tag.tcp-moderate {
  background: rgba(255, 193, 7, 0.15);
  color: #ffc107;
}
.info-tag.tcp-elevated {
  background: rgba(255, 152, 0, 0.15);
  color: #ff9800;
}
.info-tag.tcp-critical {
  background: rgba(220, 53, 69, 0.15);
  color: #dc3545;
}
.explanation-text {
  font-size: 0.95rem;
  color: #ddd;
  line-height: 1.7;
  white-space: pre-wrap;
}
.mode-btn {
  flex: 1;
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  gap: var(--space-sm);
  padding: 12px 24px;
  background: var(--bg-glass);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: var(--radius-lg);
  color: var(--text-secondary);
  cursor: pointer;
  transition: all var(--transition-base);
  position: relative;
  font-family: var(--font-family);
  font-size: var(--font-size-md);
}

.mode-btn:hover:not(.active) {
  border-color: rgba(255, 255, 255, 0.2);
  background: rgba(255, 255, 255, 0.05);
  transform: translateY(-2px);
}

.mode-btn.active {
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  border-color: transparent;
  color: white;
  box-shadow: var(--shadow-glow);
}
.audit-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.audit-item {
  background: #1e1e1e;
  padding: 0.6rem;
  border-radius: 4px;
  font-size: 0.85rem;
  color: #ccc;
  border-left: 2px solid #ffc107;
}
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 0.5rem;
}
.metric-item {
  background: #1e1e1e;
  padding: 0.5rem;
  border-radius: 4px;
  font-size: 0.85rem;
}
.metric-label {
  color: #888;
  margin-right: 0.5rem;
}
.metric-value {
  color: #00d4ff;
  font-weight: bold;
}
.metric-warning {
  display: block;
  margin-top: 0.25rem;
  font-size: 0.75rem;
  color: #ff9800;
}

.final-comp-section {
  background: #2a2a2a;
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 2rem;
}
.mini-comp-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 0.5rem;
}
.mini-comp-tag {
  background: #111;
  padding: 4px 8px;
  border-radius: 4px;
  border: 1px solid #444;
  font-family: monospace;
}

.similar-item {
  background: #1e1e1e;
  padding: 10px;
  border-radius: 6px;
  margin-bottom: 8px;
  border-left: 3px solid #666;
}
.similar-notes {
  font-size: 0.8rem;
  color: #888;
  margin-top: 4px;
}

/* === DATA VISUALIZATION: TARGET VS ACTUAL COMPARISON === */
.comparison-section {
  background: rgba(255, 255, 255, 0.03);
  padding: var(--space-lg);
  border-radius: var(--radius-lg);
  margin-bottom: var(--space-xl);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.comparison-section h4 {
  margin: 0 0 var(--space-lg) 0;
  color: var(--text-primary);
  font-size: var(--font-size-lg);
}

.comparison-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
}

.comparison-item {
  background: rgba(0, 0, 0, 0.2);
  padding: var(--space-md);
  border-radius: var(--radius-md);
  border: 1px solid rgba(255, 255, 255, 0.05);
}

.comparison-header {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  margin-bottom: var(--space-sm);
}

.comparison-icon {
  font-size: 1.5rem;
}

.comparison-label {
  flex: 1;
  font-weight: var(--font-weight-semibold);
  color: var(--text-primary);
  font-size: var(--font-size-md);
}

.comparison-status {
  padding: var(--space-xs) var(--space-sm);
  border-radius: var(--radius-sm);
  font-size: 0.75rem;
  font-weight: var(--font-weight-bold);
}

.comparison-status.met {
  background: rgba(6, 214, 160, 0.2);
  color: var(--success);
}

.comparison-status.unmet {
  background: rgba(239, 71, 111, 0.2);
  color: var(--danger);
}

.comparison-status.uncertain {
  background: rgba(255, 152, 0, 0.2);
  color: #ff9800;
}

.comparison-status.exceeds {
  background: rgba(255, 193, 7, 0.2);
  color: #ffc107;
}

.comparison-values {
  display: flex;
  justify-content: space-between;
  margin-bottom: var(--space-sm);
  font-size: var(--font-size-sm);
  color: var(--text-secondary);
}

.comparison-target {
  color: var(--text-muted);
}

.comparison-actual {
  color: var(--primary);
  font-weight: var(--font-weight-semibold);
}

.comparison-interval-discrete {
  color: #888;
  font-size: 0.8rem;
  opacity: 0.9;
  font-style: italic;
  margin-left: 0.25rem;
  font-weight: 400;
}

.comparison-bar-container {
  height: 28px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: var(--radius-md);
  overflow: hidden;
  position: relative;
}

.comparison-bar {
  height: 100%;
  border-radius: var(--radius-md);
  transition: width 0.8s ease-out;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: var(--space-sm);
  position: relative;
  overflow: hidden;
}

.comparison-bar.met {
  background: linear-gradient(90deg, rgba(6, 214, 160, 0.6), var(--success));
}

.comparison-bar.unmet {
  background: linear-gradient(90deg, rgba(239, 71, 111, 0.6), var(--danger));
}

.comparison-bar.uncertain {
  background: linear-gradient(90deg, rgba(255, 152, 0, 0.6), #ff9800);
}

.comparison-bar.exceeds {
  background: linear-gradient(90deg, rgba(255, 193, 7, 0.6), #ffc107);
}

.comparison-bar::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(
    90deg,
    transparent,
    rgba(255, 255, 255, 0.2),
    transparent
  );
  animation: shimmer 2s ease-in-out infinite;
}

.comparison-percentage {
  position: relative;
  z-index: 1;
  color: white;
  font-weight: var(--font-weight-bold);
  font-size: 0.8rem;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
}

/* === ERROR BOUNDARY STYLES === */
.error-boundary {
  background: rgba(239, 71, 111, 0.1);
  border: 2px solid var(--danger);
  padding: var(--space-xl);
  border-radius: var(--radius-lg);
  margin-bottom: var(--space-lg);
  animation: shake 0.5s ease-in-out;
}

@keyframes shake {
  0%, 100% { transform: translateX(0); }
  10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
  20%, 40%, 60%, 80% { transform: translateX(5px); }
}

.error-header {
  display: flex;
  align-items: center;
  gap: var(--space-md);
  margin-bottom: var(--space-md);
}

.error-icon {
  font-size: 2rem;
}

.error-header h3 {
  margin: 0;
  color: var(--danger);
  font-size: var(--font-size-xl);
}

.error-message {
  background: rgba(0, 0, 0, 0.3);
  padding: var(--space-md);
  border-radius: var(--radius-md);
  color: var(--text-primary);
  margin-bottom: var(--space-md);
  font-family: monospace;
  font-size: var(--font-size-sm);
}

.error-recovery {
  background: rgba(255, 255, 255, 0.05);
  padding: var(--space-md);
  border-radius: var(--radius-md);
  margin-bottom: var(--space-md);
}

.error-recovery h4 {
  margin: 0 0 var(--space-sm) 0;
  color: var(--text-secondary);
  font-size: var(--font-size-md);
}

.error-recovery ul {
  margin: 0;
  padding-left: var(--space-lg);
  color: var(--text-primary);
  line-height: 1.8;
}

.error-actions {
  display: flex;
  gap: var(--space-md);
  margin-bottom: var(--space-md);
}

.retry-btn, .dismiss-btn {
  padding: var(--space-sm) var(--space-lg);
  border-radius: var(--radius-md);
  border: none;
  font-weight: var(--font-weight-semibold);
  cursor: pointer;
  transition: all var(--transition-base);
  font-family: var(--font-family);
}

.retry-btn {
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  color: white;
  flex: 1;
}

.retry-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}

.retry-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.dismiss-btn {
  background: var(--bg-glass);
  color: var(--text-primary);
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.dismiss-btn:hover {
  background: rgba(255, 255, 255, 0.1);
}

.error-logs-scroll {
  max-height: 120px;
  overflow-y: auto;
  background: rgba(0, 0, 0, 0.3);
  padding: var(--space-sm);
  border-radius: var(--radius-md);
  font-family: monospace;
  font-size: 0.75rem;
  color: #888;
}

/* === SIMPLE LOADING STATE STYLES === */
.loading-state {
  padding: var(--space-xl);
  text-align: center;
  animation: fadeIn 0.3s ease-in;
}

.spinner {
  width: 60px;
  height: 60px;
  border: 4px solid rgba(255, 255, 255, 0.1);
  border-top-color: var(--primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto var(--space-lg);
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.loading-message {
  font-size: var(--font-size-lg);
  color: var(--text-primary);
  margin-bottom: var(--space-md);
  font-weight: var(--font-weight-medium);
}

.elapsed-time {
  font-size: var(--font-size-sm);
  color: var(--text-muted);
  margin-bottom: var(--space-md);
}

/* === DESIGN HISTORY STYLES === */
.history-toggle-btn {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-sm);
  padding: 12px 24px;
  background: var(--bg-glass);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: var(--radius-lg);
  color: var(--text-secondary);
  cursor: pointer;
  transition: all var(--transition-base);
  position: relative;
  font-family: var(--font-family);
  font-size: var(--font-size-md);
}

.history-toggle-btn:hover {
  border-color: rgba(255, 255, 255, 0.2);
  background: rgba(255, 255, 255, 0.05);
  transform: translateY(-2px);
}

.history-badge {
  position: absolute;
  top: -8px;
  right: -8px;
  background: var(--danger);
  color: white;
  border-radius: 50%;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.7rem;
  font-weight: var(--font-weight-bold);
}

.history-panel {
  padding: var(--space-xl);
  margin-bottom: var(--space-xl);
  max-height: 400px;
  overflow-y: auto;
}

.history-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--space-lg);
  padding-bottom: var(--space-md);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.history-header h3 {
  margin: 0;
  font-size: var(--font-size-lg);
  color: var(--text-primary);
}

.history-actions {
  display: flex;
  gap: var(--space-sm);
}

.small-btn {
  padding: var(--space-xs) var(--space-md);
  background: var(--bg-glass);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: var(--radius-md);
  color: var(--text-primary);
  cursor: pointer;
  transition: all var(--transition-base);
  font-size: var(--font-size-sm);
  font-family: var(--font-family);
}

.small-btn:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.1);
  transform: translateY(-1px);
}

.small-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.small-btn.danger:hover:not(:disabled) {
  background: rgba(239, 71, 111, 0.2);
  border-color: var(--danger);
  color: var(--danger);
}

.empty-history {
  text-align: center;
  padding: var(--space-xl);
  color: var(--text-muted);
}

.history-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
}

.history-item {
  background: rgba(255, 255, 255, 0.03);
  padding: var(--space-md);
  border-radius: var(--radius-md);
  border: 1px solid rgba(255, 255, 255, 0.1);
  cursor: pointer;
  transition: all var(--transition-base);
}

.history-item:hover {
  background: rgba(255, 255, 255, 0.08);
  border-color: rgba(255, 255, 255, 0.2);
  transform: translateX(4px);
}

.history-item-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--space-sm);
}

.history-mode-badge {
  padding: var(--space-xs) var(--space-sm);
  border-radius: var(--radius-sm);
  font-size: 0.75rem;
  font-weight: var(--font-weight-semibold);
}

.history-mode-badge.manual {
  background: rgba(99, 102, 241, 0.2);
  color: var(--primary);
}

.history-mode-badge.auto {
  background: rgba(236, 72, 153, 0.2);
  color: var(--secondary);
}

.history-timestamp {
  font-size: 0.7rem;
  color: var(--text-muted);
}

.history-item-details {
  display: flex;
  flex-direction: column;
  gap: var(--space-sm);
}

.history-comp-preview {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-xs);
}

.mini-tag {
  background: rgba(0, 0, 0, 0.3);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 0.7rem;
  font-family: monospace;
  color: var(--text-secondary);
}

.history-conditions {
  font-size: 0.75rem;
  color: var(--text-muted);
  font-style: italic;
}

/* === SLIDE DOWN TRANSITION === */
.slide-down-enter-active,
.slide-down-leave-active {
  transition: all 0.3s ease-out;
}

.slide-down-enter-from {
  opacity: 0;
  transform: translateY(-20px);
}

.slide-down-leave-to {
  opacity: 0;
  transform: translateY(-20px);
}

/* === MOBILE RESPONSIVENESS === */
@media (max-width: 768px) {
  .mode-switcher {
    flex-direction: column;
  }

  .mode-btn, .history-toggle-btn {
    width: 100%;
  }

  .history-panel {
    max-height: 300px;
  }

  .comp-grid, .target-grid {
    grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
  }

  .eval-controls {
    flex-direction: column;
    align-items: stretch;
  }

  .temp-inline {
    flex-wrap: wrap;
    justify-content: center;
  }

  .control-row {
    flex-direction: column;
    align-items: stretch;
  }

  .inline-control {
    justify-content: space-between;
  }

  .props-grid {
    grid-template-columns: 1fr;
  }

  .error-actions {
    flex-direction: column;
  }

  .history-header {
    flex-direction: column;
    align-items: flex-start;
    gap: var(--space-md);
  }

  .preset-buttons {
    flex-direction: column;
  }

  .preset-btn {
    width: 100%;
  }

  .comparison-values {
    flex-direction: column;
    gap: var(--space-xs);
    align-items: flex-start;
  }
}

@media (max-width: 480px) {
  .history-comp-preview {
    flex-direction: column;
  }

  .mini-tag {
    width: 100%;
  }

  .spinner {
    width: 50px;
    height: 50px;
  }
}

/* === NEW STYLES FOR IMPROVEMENTS === */

/* Comparison interval (discrete display) */
.comparison-interval {
  color: var(--text-muted);
  font-size: 0.7rem;
  font-style: italic;
  opacity: 0.7;
}

/* Prominent Comparison Interval Badge with Quality Indicators */
.comparison-interval-badge {
  padding: 4px 8px;
  border-radius: var(--radius-sm);
  font-size: 0.7rem;
  font-weight: var(--font-weight-semibold);
  border: 1px solid;
  font-family: monospace;
}

.comparison-interval-badge.tight {
  background: rgba(6, 214, 160, 0.15);
  color: var(--success);
  border-color: var(--success);
}

.comparison-interval-badge.moderate {
  background: rgba(255, 193, 7, 0.15);
  color: #ffc107;
  border-color: #ffc107;
}

.comparison-interval-badge.wide {
  background: rgba(255, 152, 0, 0.15);
  color: #ff9800;
  border-color: #ff9800;
}

/* Issues Panel */
.issues-panel {
  background: rgba(255, 200, 0, 0.05);
  border: 1px solid rgba(255, 200, 0, 0.3);
  border-radius: var(--radius-lg);
  padding: var(--space-lg);
  margin-bottom: var(--space-xl);
}

.issues-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
  margin-top: var(--space-md);
}

.issue-item {
  background: rgba(0, 0, 0, 0.3);
  padding: var(--space-md);
  border-radius: var(--radius-md);
  border-left: 4px solid;
}

.issue-item.severity-high {
  border-left-color: var(--danger);
}

.issue-item.severity-medium {
  border-left-color: #f39c12;
}

.issue-item.severity-low {
  border-left-color: #3498db;
}

.issue-header {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  margin-bottom: var(--space-sm);
}

.issue-icon {
  font-size: 1.2rem;
}

.issue-type {
  flex: 1;
  font-weight: var(--font-weight-semibold);
  color: var(--text-primary);
}

.issue-severity {
  padding: 2px 8px;
  border-radius: var(--radius-sm);
  font-size: 0.7rem;
  font-weight: var(--font-weight-bold);
  background: rgba(255, 255, 255, 0.1);
  color: var(--text-secondary);
}

.issue-description {
  color: var(--text-secondary);
  margin-bottom: var(--space-sm);
  line-height: 1.6;
}

.issue-recommendation {
  color: var(--primary);
  font-size: var(--font-size-sm);
  padding: var(--space-xs) var(--space-sm);
  background: rgba(17, 153, 250, 0.1);
  border-radius: var(--radius-sm);
  border-left: 3px solid var(--primary);
}

/* Property Interval Badge with Quality Indicators */
.prop-interval-badge {
  display: inline-block;
  margin-top: 6px;
  padding: 4px 10px;
  border-radius: var(--radius-sm);
  font-size: 0.75rem;
  font-weight: var(--font-weight-semibold);
  font-family: monospace;
  border: 1px solid;
}

.prop-interval-badge.tight {
  background: rgba(6, 214, 160, 0.15);
  color: var(--success);
  border-color: var(--success);
}

.prop-interval-badge.moderate {
  background: rgba(255, 193, 7, 0.15);
  color: #ffc107;
  border-color: #ffc107;
}

.prop-interval-badge.wide {
  background: rgba(255, 152, 0, 0.15);
  color: #ff9800;
  border-color: #ff9800;
}

.prop-interval-badge.unknown {
  background: rgba(255, 255, 255, 0.05);
  color: var(--text-muted);
  border-color: rgba(255, 255, 255, 0.1);
}

/* Overestimation Warning */
.overestimation-warning {
  margin-top: 6px;
  padding: 6px 10px;
  background: rgba(255, 152, 0, 0.15);
  border: 1px solid #ff9800;
  border-radius: var(--radius-sm);
  font-size: 0.7rem;
  color: #ff9800;
  line-height: 1.4;
  display: flex;
  align-items: flex-start;
  gap: 4px;
}

.recommendations-section {
  margin-top: var(--space-lg);
  padding-top: var(--space-lg);
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.recommendations-header {
  font-weight: var(--font-weight-semibold);
  color: var(--text-primary);
  margin-bottom: var(--space-sm);
}

.recommendations-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.recommendations-list li {
  padding: var(--space-xs) 0;
  color: var(--text-secondary);
  padding-left: var(--space-md);
  position: relative;
}

.recommendations-list li::before {
  content: '→';
  position: absolute;
  left: 0;
  color: var(--primary);
}

/* Confidence indicators for property cards */
.prop-card.confidence-high {
  border-left: 3px solid rgba(76, 175, 80, 0.6);
}

.prop-card.confidence-medium {
  border-left: 3px solid rgba(255, 193, 7, 0.5);
}

.prop-card.confidence-low {
  border-left: 3px solid rgba(244, 67, 54, 0.5);
}

/* Reasoning Panel */
.reasoning-panel {
  background: rgba(17, 153, 250, 0.05);
  border: 1px solid rgba(17, 153, 250, 0.2);
  border-radius: var(--radius-lg);
  padding: var(--space-lg);
  margin-bottom: var(--space-xl);
}

.reasoning-text {
  color: var(--text-secondary);
  line-height: 1.8;
  white-space: pre-wrap;
}

/* === AGENT INVESTIGATION PANEL === */
.agent-reasoning-section {
  background: rgba(0, 212, 255, 0.04);
  border: 1px solid rgba(0, 212, 255, 0.15);
  border-radius: var(--radius-lg);
  padding: var(--space-lg);
  margin-bottom: var(--space-xl);
}

.agent-reasoning-section .sub-header {
  font-size: 0.9rem;
  font-weight: 600;
  color: #00d4ff;
  margin-bottom: 0.5rem;
  margin-top: 1rem;
}

.agent-reasoning-section .sub-header:first-of-type {
  margin-top: 0.5rem;
}

.source-reliability-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.4rem 0.75rem;
  background: rgba(0, 212, 255, 0.1);
  border-radius: 6px;
  font-size: 0.85rem;
  margin-bottom: 0.75rem;
}

.reliability-label {
  color: #aaa;
}

.reliability-value {
  color: #00d4ff;
  font-weight: 600;
}

.findings-text,
.assessment-text {
  color: var(--text-secondary);
  line-height: 1.7;
  white-space: pre-wrap;
  font-size: 0.92rem;
}

.analyst-section,
.reviewer-section,
.investigation-findings {
  padding: 0.75rem;
  background: rgba(255, 255, 255, 0.02);
  border-radius: var(--radius-md);
  margin-top: 0.5rem;
}

.reviewer-section {
  border-left: 3px solid rgba(123, 44, 191, 0.5);
}

.analyst-section {
  border-left: 3px solid rgba(0, 212, 255, 0.4);
}

.investigation-findings {
  border-left: 3px solid rgba(255, 193, 7, 0.4);
}

/* === CORRECTIONS PANEL === */
.corrections-panel {
  background: rgba(255, 193, 7, 0.05);
  border: 1px solid rgba(255, 193, 7, 0.2);
  border-radius: var(--radius-lg);
  padding: var(--space-lg);
  margin-bottom: var(--space-xl);
}

.corrections-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.correction-item {
  padding: 0.75rem;
  background: rgba(255, 255, 255, 0.03);
  border-radius: var(--radius-md);
  border-left: 3px solid rgba(255, 193, 7, 0.4);
}

.correction-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.4rem;
}

.correction-prop {
  font-weight: 600;
  color: #ffc107;
  font-size: 0.9rem;
}

.correction-arrow {
  font-family: monospace;
  color: #00d4ff;
  font-size: 0.85rem;
}

.correction-reason {
  color: var(--text-secondary);
  font-size: 0.85rem;
  line-height: 1.5;
}

.correction-constraint {
  color: var(--text-muted);
  font-size: 0.8rem;
  font-style: italic;
  margin-top: 0.25rem;
}

.corrections-summary {
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  color: var(--text-secondary);
  font-size: 0.9rem;
  line-height: 1.6;
}

/* === PIPELINE LOADING INDICATOR === */
.pipeline-header {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  margin-bottom: 1.5rem;
}

.pipeline-spinner {
  width: 22px;
  height: 22px;
  border: 2.5px solid rgba(255, 255, 255, 0.1);
  border-top-color: var(--primary, #00d4ff);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.pipeline-title {
  font-size: 1rem;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.7);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.pipeline-active-step {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.6rem;
  padding: 0.75rem 1.25rem;
  background: rgba(0, 212, 255, 0.06);
  border: 1px solid rgba(0, 212, 255, 0.15);
  border-radius: 12px;
  margin: 0 auto 1.25rem;
  max-width: 420px;
  animation: stepFadeIn 0.4s ease;
}

@keyframes stepFadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}

.active-step-icon {
  font-size: 1.3rem;
  flex-shrink: 0;
}

.active-step-label {
  font-size: 0.95rem;
  font-weight: 500;
  color: #00d4ff;
}

.active-step-dots {
  color: rgba(0, 212, 255, 0.5);
}

.dot-anim {
  display: inline-block;
  animation: dotPulse 1.4s ease-in-out infinite;
}

@keyframes dotPulse {
  0%, 100% { opacity: 0.2; }
  50% { opacity: 1; }
}

.pipeline-track {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.pipeline-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.15);
  transition: all 0.3s ease;
}

.pipeline-dot.active {
  width: 10px;
  height: 10px;
  background: var(--primary, #00d4ff);
  box-shadow: 0 0 8px rgba(0, 212, 255, 0.5);
}

.pipeline-footer {
  display: flex;
  align-items: center;
  justify-content: center;
  padding-top: 0.5rem;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
}

.pipeline-footer .elapsed-time {
  font-size: 0.8rem;
  color: #666;
  font-family: monospace;
}

/* Minimal interval display */
.prop-interval-minimal {
  font-size: 0.7rem;
  color: var(--text-muted);
  opacity: 0.6;
  font-style: italic;
  margin-top: 2px;
}

/* Preset divider and action buttons */
.preset-divider {
  color: var(--text-muted);
  margin: 0 var(--space-sm);
  opacity: 0.5;
}

/* Action buttons match regular preset buttons */
.action-btn {
  background: var(--bg-glass);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.action-btn:hover {
  background: rgba(255, 255, 255, 0.08);
  border-color: rgba(255, 255, 255, 0.2);
  box-shadow: none;
}

.clear-btn:hover {
  background: rgba(255, 80, 80, 0.15);
  border-color: rgba(255, 80, 80, 0.4);
  box-shadow: none;
}

/* JSON Import Modal */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  backdrop-filter: blur(4px);
}

.json-import-modal {
  background: var(--bg-secondary);
  border-radius: var(--radius-lg);
  width: 90%;
  max-width: 550px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
  border: 1px solid var(--border);
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--space-lg);
  border-bottom: 1px solid var(--border);
}

.modal-header h3 {
  margin: 0;
  font-size: 1.1rem;
  color: var(--text-primary);
}

.modal-close {
  background: none;
  border: none;
  color: var(--text-muted);
  font-size: 1.5rem;
  cursor: pointer;
  padding: 0;
  line-height: 1;
  transition: color 0.2s;
}

.modal-close:hover {
  color: var(--text-primary);
}

.modal-body {
  padding: var(--space-lg);
}

.modal-help {
  color: var(--text-secondary);
  margin-bottom: var(--space-md);
  font-size: 0.9rem;
}

.json-examples {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-sm);
  align-items: center;
  margin-bottom: var(--space-md);
  padding: var(--space-sm);
  background: rgba(0, 0, 0, 0.2);
  border-radius: var(--radius-sm);
}

.example-label {
  color: var(--text-muted);
  font-size: 0.8rem;
}

.json-examples code {
  background: rgba(17, 153, 250, 0.15);
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.75rem;
  color: var(--primary);
}

.json-textarea {
  width: 100%;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  color: var(--text-primary);
  padding: var(--space-md);
  font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
  font-size: 0.9rem;
  resize: vertical;
  min-height: 120px;
}

.json-textarea:focus {
  outline: none;
  border-color: var(--primary);
  box-shadow: 0 0 0 2px rgba(17, 153, 250, 0.2);
}

.json-textarea::placeholder {
  color: var(--text-muted);
  opacity: 0.6;
}

.json-error {
  color: #ff6b6b;
  font-size: 0.85rem;
  margin-top: var(--space-sm);
  padding: var(--space-sm);
  background: rgba(255, 80, 80, 0.1);
  border-radius: var(--radius-sm);
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: var(--space-md);
  padding: var(--space-lg);
  border-top: 1px solid var(--border);
}

.cancel-btn {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-secondary);
  padding: var(--space-sm) var(--space-lg);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all 0.2s;
}

.cancel-btn:hover {
  background: rgba(255, 255, 255, 0.05);
  color: var(--text-primary);
}

.import-btn {
  background: var(--primary);
  border: none;
  color: white;
  padding: var(--space-sm) var(--space-lg);
  border-radius: var(--radius-md);
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s;
}

.import-btn:hover {
  background: var(--primary-dark, #0d8ae0);
  transform: translateY(-1px);
}

/* Modal animation */
.modal-fade-enter-active,
.modal-fade-leave-active {
  transition: opacity 0.2s ease;
}

.modal-fade-enter-active .json-import-modal,
.modal-fade-leave-active .json-import-modal {
  transition: transform 0.2s ease;
}

.modal-fade-enter-from,
.modal-fade-leave-to {
  opacity: 0;
}

.modal-fade-enter-from .json-import-modal,
.modal-fade-leave-to .json-import-modal {
  transform: scale(0.95) translateY(-10px);
}
</style>
