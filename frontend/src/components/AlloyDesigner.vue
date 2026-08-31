<script setup>
defineOptions({ name: 'AlloyDesigner' })
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import axios from 'axios'
import { API_BASE_URL } from '../config'
import { useToast } from '../composables/useToast'
import { BUILTIN_PRESETS, DEFAULT_PRESET } from '../presets'
import CompositionEditor from './CompositionEditor.vue'
import TargetPropertyForm from './TargetPropertyForm.vue'
import ResultsDashboard from './ResultsDashboard.vue'
import DesignHistory from './DesignHistory.vue'
import ComparisonModal from './ComparisonModal.vue'

const { showToast } = useToast()

// --- PROPS ---
const props = defineProps({
  initialAlloy: { type: Object, default: null }
})

// --- CORE STATE ---
const mode = ref('manual')
const loading = ref(false)
const logs = ref([])
const result = ref(null)

// --- REQUEST LIFETIME ---
// Agent runs are slow but not unbounded. Without an explicit timeout axios waits
// forever, so a backend that stalls leaves the UI spinning with no way out.
// Design scales with the iteration count; evaluation is a single pass.
const VALIDATE_TIMEOUT_MS = 6 * 60 * 1000
const DESIGN_TIMEOUT_BASE_MS = 5 * 60 * 1000
const DESIGN_TIMEOUT_PER_ITER_MS = 5 * 60 * 1000
let inFlight = null

// Evaluation is now a background job: POST returns a job id in well under a
// second and we poll for the outcome. This exists because the CloudUT reverse
// proxy in front of the public site cuts connections at roughly 60 s, while a
// real evaluation takes 100-200 s -- so the old single long request returned
// 504 in the browser even though the backend was working fine. Nothing about
// the UI changes: the progress indicator is driven by an elapsed-time timer,
// not by the request, so it behaves identically while polling.
const POLL_INTERVAL_MS = 2500
// Individual HTTP calls are all short now, so they get short timeouts. The
// overall patience for a job is VALIDATE_TIMEOUT_MS, enforced across polls.
const SUBMIT_TIMEOUT_MS = 20 * 1000
const POLL_TIMEOUT_MS = 15 * 1000
// A transient network blip mid-poll should not kill a job that is still running
// server-side; give up only after this many consecutive failed polls.
const MAX_POLL_FAILURES = 5

// Set by cancelRun so an in-progress polling loop stops between requests, not
// just when a single HTTP call is aborted.
let pollCancelled = false

const cancelRun = () => {
  pollCancelled = true
  if (inFlight) { inFlight.abort(); inFlight = null }
}

const sleep = (ms) => new Promise(r => setTimeout(r, ms))

// --- ERROR STATE ---
const error = ref(null)
const errorType = ref(null)
const retryCount = ref(0)
const maxRetries = 3

// --- LOADING PROGRESS ---
const startTime = ref(null)
const elapsedSeconds = ref(0)
let elapsedTimer = null
const loadingStep = ref(0)

const evaluationSteps = [
  { icon: '\uD83D\uDD2C', label: 'Running ML predictions & physics models' },
  { icon: '\uD83D\uDCCA', label: 'Analyst investigating alloy sources' },
  { icon: '\uD83D\uDD17', label: 'Querying Knowledge Graph' },
  { icon: '\u2696\uFE0F', label: 'Triangulating ML, Physics, and KG data' },
  { icon: '\uD83D\uDD0E', label: 'Reviewer verifying metallurgical consistency' },
  { icon: '\u270D\uFE0F', label: 'Generating metallurgical analysis' },
]

const designSteps = [
  { icon: '\uD83C\uDFA8', label: 'Designing alloy composition' },
  { icon: '\uD83D\uDD2C', label: 'Pre-computing ML & physics predictions' },
  { icon: '\u2696\uFE0F', label: 'Analyst investigating alloy sources' },
  { icon: '\uD83D\uDD0E', label: 'Reviewer verifying predictions' },
  { icon: '\u270D\uFE0F', label: 'Generating analysis report' },
]

const currentSteps = computed(() => mode.value === 'manual' ? evaluationSteps : designSteps)

// --- MANUAL MODE STATE ---
// Open on a real alloy rather than a synthetic one. The previous default
// (Ni 60 / Cr 20 / Al 10 / Ti 5 / Co 5) summed to 100 but carried 15 wt% of
// gamma-prime formers -- above anything that can actually be produced -- so the
// first thing a visitor evaluated was a composition the system would reject.
const manualComp = ref({ ...BUILTIN_PRESETS[DEFAULT_PRESET].composition })
const manualTemp = ref(20)
const manualProcessing = ref(BUILTIN_PRESETS[DEFAULT_PRESET].processing)

// --- AUTO MODE STATE ---
const targets = ref({ yield: 0, tensile: 0, elongation: 0, elastic_modulus: 0, density: 0, gamma_prime: 0 })
const autoIterations = ref(3)
const autoTemp = ref(20)
const autoProcessing = ref('cast')

// --- DESIGN HISTORY ---
const designHistory = ref([])
const showHistory = ref(false)
const maxHistoryItems = 20

// --- COMPARISON ---
const showComparison = ref(false)
const comparisonItems = ref([])

const openComparison = (items) => {
  comparisonItems.value = items
  showComparison.value = true
}


// --- DRAFT PERSISTENCE ---
const DRAFT_KEY = 'alloygraph-draft'
let draftTimer = null

const saveDraft = () => {
  if (draftTimer) clearTimeout(draftTimer)
  draftTimer = setTimeout(() => {
    try {
      const draft = {
        mode: mode.value,
        manualComp: manualComp.value,
        manualTemp: manualTemp.value,
        manualProcessing: manualProcessing.value,
        targets: targets.value,
        autoIterations: autoIterations.value,
        autoTemp: autoTemp.value,
        autoProcessing: autoProcessing.value,
      }
      localStorage.setItem(DRAFT_KEY, JSON.stringify(draft))
    } catch (e) { /* quota exceeded — ignore */ }
  }, 500)
}

const restoreDraft = () => {
  try {
    const stored = localStorage.getItem(DRAFT_KEY)
    if (!stored) return false
    const draft = JSON.parse(stored)
    // Only restore if there's actual composition data
    if (draft.manualComp && Object.keys(draft.manualComp).length > 0) {
      manualComp.value = draft.manualComp
    }
    if (draft.manualTemp !== undefined) manualTemp.value = draft.manualTemp
    if (draft.manualProcessing) manualProcessing.value = draft.manualProcessing
    if (draft.targets) targets.value = { ...targets.value, ...draft.targets }
    if (draft.autoIterations) autoIterations.value = draft.autoIterations
    if (draft.autoTemp !== undefined) autoTemp.value = draft.autoTemp
    if (draft.autoProcessing) autoProcessing.value = draft.autoProcessing
    if (draft.mode) mode.value = draft.mode
    return true
  } catch (e) { return false }
}


// Watch all form state for auto-save
watch([manualComp, manualTemp, manualProcessing, targets, autoIterations, autoTemp, autoProcessing, mode], saveDraft, { deep: true })

// --- LOADING HELPERS ---
const startLoading = () => {
  loading.value = true
  startTime.value = Date.now()
  elapsedSeconds.value = 0
  loadingStep.value = 0
  error.value = null
  errorType.value = null

  if (elapsedTimer) clearInterval(elapsedTimer)
  elapsedTimer = setInterval(() => {
    elapsedSeconds.value = Math.floor((Date.now() - startTime.value) / 1000)
    const steps = currentSteps.value
    const thresholds = steps.map((_, i) => Math.round(2 + i * 1.5 + i * i * 0.8))
    let step = 0
    for (let i = thresholds.length - 1; i >= 0; i--) {
      if (elapsedSeconds.value >= thresholds[i]) { step = Math.min(i + 1, steps.length - 1); break }
    }
    loadingStep.value = step
  }, 1000)
}

const stopLoading = () => {
  loading.value = false
  loadingStep.value = 0
  inFlight = null
  if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null }
}

// --- ERROR HELPERS ---
const classifyError = (err) => {
  const msg = String(err?.message || '')
  if (err?.code === 'ERR_CANCELED' || err?.name === 'CanceledError' || msg === 'canceled') return 'cancelled'
  if (err?.code === 'ECONNABORTED' || msg.includes('timeout')) return 'timeout'
  if (err?.code === 'ERR_NETWORK' || msg.includes('Network Error')) return 'network'
  if (!err?.response) return 'no_response'
  if (err.response?.status >= 400 && err.response?.status < 500) return 'validation'
  if (err.response?.status >= 500) return 'server'
  return 'unknown'
}

// The backend returns str(exception) in its 500 handler, so `data.error` can be
// a raw Python message ("KeyError: 'Yield Strength'", a litellm traceback line).
// Surface a short technical detail for the curious, never the raw string as the
// primary message, and never anything that looks like a stack trace.
const MAX_DETAIL_CHARS = 100

const technicalDetail = (err) => {
  const raw = err?.response?.data?.error || err?.message || ''
  let first = String(raw).split('\n').find(l => l.trim()) || ''
  if (!first) return ''
  if (/Traceback|File "|  at /.test(first)) return ''
  // Upstream errors often append a serialised payload -- a Python dict or a
  // JSON body. The prose before it is the useful part; the structure is noise
  // that reads like a stack trace to anyone who is not debugging this.
  first = first.split(/[{[]/)[0].trim().replace(/[\s:,-]+$/, '')
  if (!first) return ''
  return first.length > MAX_DETAIL_CHARS ? `${first.slice(0, MAX_DETAIL_CHARS)}\u2026` : first
}

// A 200 response can still carry result.error: the agent pipeline reports an
// upstream failure (provider unreachable, model missing) in-band. That is a
// backend-side problem, not something wrong with the user's input, so it is
// classified as 'server' and put through the same sanitiser as a thrown error.
const messageFromResultError = (raw) => {
  const detail = technicalDetail({ message: String(raw || '') })
  const base = 'The backend hit an error while running this analysis.'
  return detail ? `${base} (${detail})` : base
}

const getErrorMessage = (type, err) => {
  const messages = {
    network: 'Could not reach the backend. Check that it is running, then try again.',
    no_response: 'The backend accepted the request but never replied. It may still be starting up.',
    timeout: 'This run took longer than expected and was stopped. Try fewer iterations, or run it again.',
    validation: 'The backend rejected this request. Check the composition and target values.',
    server: 'The backend hit an error while running this analysis.',
    cancelled: 'Run cancelled.',
    unknown: 'Something went wrong while running this analysis.'
  }
  const base = messages[type] || messages.unknown
  const detail = type === 'cancelled' ? '' : technicalDetail(err)
  return detail ? `${base} (${detail})` : base
}

const clearError = () => { error.value = null; errorType.value = null }

// --- HISTORY ---
const loadHistory = () => {
  try {
    const stored = localStorage.getItem('alloyDesignHistory')
    if (stored) designHistory.value = JSON.parse(stored)
  } catch (e) { console.error('Failed to load history:', e) }
}

const saveToHistory = (design) => {
  const item = {
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
  designHistory.value.unshift(item)
  if (designHistory.value.length > maxHistoryItems) designHistory.value = designHistory.value.slice(0, maxHistoryItems)
  try { localStorage.setItem('alloyDesignHistory', JSON.stringify(designHistory.value)) } catch (e) { console.error('Failed to save history:', e) }
}

const loadFromHistory = (item) => {
  result.value = null
  if (item.mode === 'manual') {
    mode.value = 'manual'
    manualComp.value = { ...item.composition }
    manualTemp.value = item.temperature
    manualProcessing.value = item.processing
  } else {
    mode.value = 'auto'
    if (item.targets) targets.value = { ...item.targets }
    autoTemp.value = item.temperature
    autoProcessing.value = item.processing
  }
  setTimeout(() => {
    result.value = { ...item.design }
    showHistory.value = false
    setTimeout(() => {
      const el = document.querySelector('.output-area')
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 150)
  }, 10)
}

const clearHistory = () => {
  designHistory.value = []
  localStorage.removeItem('alloyDesignHistory')
}

// --- API: VALIDATE (submit job, then poll) ---

// Thrown when the job itself reports failure, so the polling loop can hand the
// server's own message to the error panel instead of a generic HTTP message.
class JobFailedError extends Error {}

// Poll until the job reaches a terminal state. Returns the result object.
const pollForResult = async (jobId) => {
  const deadline = Date.now() + VALIDATE_TIMEOUT_MS
  let consecutiveFailures = 0
  let lastStatus = null

  while (Date.now() < deadline) {
    await sleep(POLL_INTERVAL_MS)
    if (pollCancelled) { const e = new Error('canceled'); e.code = 'ERR_CANCELED'; throw e }

    let res
    try {
      inFlight = new AbortController()
      res = await axios.get(`${API_BASE_URL}/api/validate/status/${jobId}`,
        { timeout: POLL_TIMEOUT_MS, signal: inFlight.signal })
      consecutiveFailures = 0
    } catch (err) {
      // A cancel or a genuine 404 is final; anything else may be transient.
      if (classifyError(err) === 'cancelled') throw err
      if (err?.response?.status === 404) {
        throw new JobFailedError(
          err.response.data?.error ||
          'The server lost track of this analysis. Please run it again.')
      }
      if (++consecutiveFailures >= MAX_POLL_FAILURES) throw err
      continue
    }

    const { status, result: jobResult, error: jobError } = res.data || {}

    if (status !== lastStatus) {
      lastStatus = status
      if (status === 'running') logs.value.push('Agents are analysing the composition...')
    }

    if (status === 'done') return jobResult
    if (status === 'failed') {
      throw new JobFailedError(jobError || 'The analysis failed on the server.')
    }
  }

  const e = new Error('timeout')
  e.code = 'ECONNABORTED'
  throw e
}

const runValidation = async (isRetry = false) => {
  startLoading()
  pollCancelled = false
  if (!isRetry) { logs.value = []; result.value = null; retryCount.value = 0 }
  logs.value.push(`Validating Composition at ${manualTemp.value}\u00B0C (${manualProcessing.value})...`)

  try {
    inFlight = new AbortController()
    const submit = await axios.post(`${API_BASE_URL}/api/validate`, {
      composition: manualComp.value, temp: manualTemp.value, processing: manualProcessing.value
    }, { timeout: SUBMIT_TIMEOUT_MS, signal: inFlight.signal })

    const jobId = submit.data?.job_id
    if (!jobId) throw new JobFailedError('The server did not return a job id.')
    logs.value.push('Analysis queued on the server.')

    const jobResult = await pollForResult(jobId)

    // A completed job can still carry an in-band error from the pipeline; this
    // is the same result.error contract the synchronous endpoint had.
    if (jobResult?.error) {
      stopLoading()
      errorType.value = 'server'
      error.value = messageFromResultError(jobResult.error)
      return
    }

    result.value = jobResult
    logs.value.push('Prediction Complete.')
    if (jobResult?.properties) saveToHistory(jobResult)
    stopLoading(); retryCount.value = 0
  } catch (err) {
    stopLoading()
    if (err instanceof JobFailedError) {
      console.error('Validation job failed:', err.message)
      errorType.value = 'server'
      error.value = messageFromResultError(err.message)
      return
    }
    const kind = classifyError(err)
    if (kind === 'cancelled') { logs.value.push('Evaluation cancelled.'); return }
    console.error('Validation error:', err)
    errorType.value = kind; error.value = getErrorMessage(kind, err)
  }
}

const retryValidation = async () => {
  if (retryCount.value < maxRetries) {
    retryCount.value++
    const delay = Math.min(1000 * Math.pow(2, retryCount.value - 1), 10000)
    await new Promise(r => setTimeout(r, delay))
    await runValidation(true)
  }
}

// --- API: DESIGN ---
const runDesign = async (isRetry = false) => {
  startLoading()
  if (!isRetry) { logs.value = []; result.value = null; retryCount.value = 0 }
  logs.value.push('Starting Inverse Design Agent...')

  try {
    const target_props = {}
    if (targets.value.yield > 0) target_props['Yield Strength'] = targets.value.yield
    if (targets.value.tensile > 0) target_props['Tensile Strength'] = targets.value.tensile
    if (targets.value.elongation > 0) target_props['Elongation'] = targets.value.elongation
    if (targets.value.elastic_modulus > 0) target_props['Elastic Modulus'] = targets.value.elastic_modulus
    // The form states "Set to 0 to skip any property", and every other target
    // honours that. Density used `< 99` alone, so the default 0 was sent as a
    // real ceiling of 0 g/cm3 -- a target nothing can meet.
    if (targets.value.density > 0 && targets.value.density < 99) {
      target_props['Density'] = targets.value.density
    }
    if (targets.value.gamma_prime > 0) target_props['Gamma Prime'] = targets.value.gamma_prime

    inFlight = new AbortController()
    const response = await axios.post(`${API_BASE_URL}/api/design`, {
      target_props, processing: autoProcessing.value, temp: autoTemp.value, max_iter: autoIterations.value
    }, {
      timeout: DESIGN_TIMEOUT_BASE_MS + DESIGN_TIMEOUT_PER_ITER_MS * Math.max(1, Number(autoIterations.value) || 1),
      signal: inFlight.signal,
    })
    const designResult = response.data.result
    result.value = designResult

    if (designResult.design_status === 'incomplete' && designResult.issues?.length > 0) {
      logs.value.push('Design completed with issues.')
    } else if (designResult.error) {
      logs.value.push('Error: ' + designResult.error)
    } else {
      logs.value.push('Design Complete!')
    }

    if (designResult.composition) saveToHistory(designResult)
    stopLoading(); retryCount.value = 0
  } catch (err) {
    stopLoading()
    const kind = classifyError(err)
    if (kind === 'cancelled') { logs.value.push('Design cancelled.'); return }
    console.error('Design error:', err)
    errorType.value = kind; error.value = getErrorMessage(kind, err)
  }
}

const retryDesign = async () => {
  if (retryCount.value < maxRetries) {
    retryCount.value++
    const delay = Math.min(1000 * Math.pow(2, retryCount.value - 1), 10000)
    await new Promise(r => setTimeout(r, delay))
    await runDesign(true)
  }
}

// --- COPY TO EVALUATION ---
const copyToEvaluation = () => {
  if (!result.value?.composition) return
  manualComp.value = { ...result.value.composition }
  mode.value = 'manual'
}

// --- WATCHERS & LIFECYCLE ---
watch(mode, () => { result.value = null; logs.value = []; clearError() })

const initFromProp = (alloy) => {
  if (!alloy) return
  if (alloy.composition) manualComp.value = { ...alloy.composition }
  if (alloy.processing_method) manualProcessing.value = alloy.processing_method.toLowerCase()
  result.value = null
  mode.value = 'manual'
}

if (props.initialAlloy) initFromProp(props.initialAlloy)
watch(() => props.initialAlloy, (v) => initFromProp(v))

onMounted(() => {
  loadHistory()
  // Restore draft unless an alloy was passed from chat
  if (!props.initialAlloy) {
    const restored = restoreDraft()
    if (restored) showToast('Previous draft restored', 'info')
  }
})

onUnmounted(() => {
  if (draftTimer) clearTimeout(draftTimer)
  if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null }
})
</script>

<template>
  <div class="alloy-designer">
    <!-- Mode Switcher -->
    <div class="mode-switcher glass-card" data-tour="mode-switcher">
      <button :class="['mode-btn', { active: mode === 'manual' }]" @click="mode = 'manual'" aria-label="Evaluate mode" data-tour="evaluate-btn">
        <span class="mode-icon">🧪</span>
        <span class="mode-label">Evaluate</span>
      </button>
      <button :class="['mode-btn', { active: mode === 'auto' }]" @click="mode = 'auto'" aria-label="Design mode" data-tour="design-btn">
        <span class="mode-icon">🧬</span>
        <span class="mode-label">Design</span>
      </button>
      <button class="history-toggle-btn" @click="showHistory = !showHistory"
              :aria-expanded="showHistory" aria-label="Toggle design history" data-tour="history-toggle">
        <span class="mode-icon">📜</span>
        <span class="mode-label">History</span>
        <span v-if="designHistory.length > 0" class="history-badge">{{ designHistory.length }}</span>
      </button>
    </div>

    <!-- History Panel -->
    <DesignHistory
      :history="designHistory"
      :maxItems="maxHistoryItems"
      :visible="showHistory"
      @load-item="loadFromHistory"
      @clear="clearHistory"
      @compare="openComparison"
    />

    <!-- Comparison Modal -->
    <ComparisonModal
      :visible="showComparison"
      :items="comparisonItems"
      @close="showComparison = false"
    />

    <!-- Manual Mode -->
    <CompositionEditor
      v-if="mode === 'manual'"
      v-model="manualComp"
      :temperature="manualTemp"
      :processing="manualProcessing"
      :loading="loading"
      @update:temperature="manualTemp = $event"
      @update:processing="manualProcessing = $event"
      @submit="runValidation"
    />

    <!-- Auto Mode -->
    <TargetPropertyForm
      v-if="mode === 'auto'"
      v-model="targets"
      :temperature="autoTemp"
      :processing="autoProcessing"
      :iterations="autoIterations"
      :loading="loading"
      @update:temperature="autoTemp = $event"
      @update:processing="autoProcessing = $event"
      @update:iterations="autoIterations = $event"
      @submit="runDesign"
    />

    <!-- Results Dashboard -->
    <ResultsDashboard
      :result="result"
      :error="error"
      :errorType="errorType"
      :loading="loading"
      :loadingStep="loadingStep"
      :currentSteps="currentSteps"
      :elapsedSeconds="elapsedSeconds"
      :logs="logs"
      :mode="mode"
      :targets="targets"
      :temperature="mode === 'manual' ? manualTemp : autoTemp"
      :manualComp="manualComp"
      :retryCount="retryCount"
      :maxRetries="maxRetries"
      @cancel="cancelRun"
      @retry="mode === 'manual' ? retryValidation() : retryDesign()"
      @dismiss-error="clearError"
      @copy-to-evaluation="copyToEvaluation"
    />
  </div>
</template>

<style scoped>
/* Mode Switcher */
.mode-switcher { display: flex; flex-direction: row; gap: var(--space-md); padding: var(--space-sm); margin-bottom: var(--space-xl); }

.mode-btn {
  flex: 1; display: flex; flex-direction: row; align-items: center; justify-content: center; gap: var(--space-sm);
  padding: 12px 24px; background: var(--bg-glass); border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg); color: var(--text-secondary); cursor: pointer;
  transition: all var(--transition-base); position: relative; font-family: var(--font-family); font-size: var(--font-size-md);
}
.mode-btn:hover:not(.active) { border-color: var(--border-strong); background: var(--bg-glass); transform: translateY(-2px); }
.mode-btn.active { background: linear-gradient(135deg, var(--primary), var(--secondary)); border-color: transparent; color: white; box-shadow: var(--shadow-glow); }
.mode-icon { font-size: 1.5rem; }
.mode-label { font-size: var(--font-size-md); font-weight: var(--font-weight-semibold); }

.history-toggle-btn {
  flex: 0 0 auto; display: flex; align-items: center; justify-content: center; gap: var(--space-sm);
  padding: 12px 24px; background: var(--bg-glass); border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg); color: var(--text-secondary); cursor: pointer;
  transition: all var(--transition-base); position: relative; font-family: var(--font-family); font-size: var(--font-size-md);
}
.history-toggle-btn:hover { border-color: var(--border-strong); background: var(--bg-glass); transform: translateY(-2px); }
.history-badge {
  position: absolute; top: -8px; right: -8px; background: var(--danger); color: white;
  border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center;
  font-size: 0.7rem; font-weight: var(--font-weight-bold);
}

@media (max-width: 768px) {
  .mode-switcher { flex-direction: column; margin-bottom: var(--space-md); }
  .mode-btn, .history-toggle-btn { width: 100%; }
}

@media (max-width: 480px) {
  .mode-switcher { gap: var(--space-xs); padding: var(--space-xs); margin-bottom: var(--space-sm); }
  .mode-btn { padding: 10px 16px; font-size: var(--font-size-sm); }
  .mode-icon { font-size: 1.25rem; }
  .history-toggle-btn { padding: 10px 16px; font-size: var(--font-size-sm); }
}
</style>
