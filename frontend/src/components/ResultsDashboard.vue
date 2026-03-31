<script setup>
import { computed } from 'vue'
import { useToast } from '../composables/useToast'
import AnimatedNumber from './AnimatedNumber.vue'

const { showToast } = useToast()

const props = defineProps({
  result: { type: Object, default: null },
  error: { type: String, default: null },
  errorType: { type: String, default: null },
  loading: { type: Boolean, default: false },
  loadingStep: { type: Number, default: 0 },
  currentSteps: { type: Array, default: () => [] },
  elapsedSeconds: { type: Number, default: 0 },
  logs: { type: Array, default: () => [] },
  mode: { type: String, default: 'manual' },
  targets: { type: Object, default: () => ({}) },
  temperature: { type: Number, default: 20 },
  manualComp: { type: Object, default: () => ({}) },
  retryCount: { type: Number, default: 0 },
  maxRetries: { type: Number, default: 3 },
})

const emit = defineEmits(['retry', 'dismiss-error', 'copy-to-evaluation'])

// --- HELPERS ---
const parseVal = (v) => {
  if (typeof v === 'number') return v
  if (typeof v === 'string') { const match = v.match(/[\d.]+/); return match ? parseFloat(match[0]) : null }
  return null
}

const lookUpProp = (obj, keyPart) => {
  if (!obj) return undefined
  if (obj[keyPart] !== undefined) return obj[keyPart]
  const keyPartSimple = keyPart.replace(/\s+/g, '').toLowerCase()
  for (const k of Object.keys(obj)) {
    const kClean = k.replace(/\s+/g, '').toLowerCase().replace(/\(.*\)/, '')
    if (kClean === keyPartSimple || kClean.includes(keyPartSimple)) return obj[k]
  }
  return undefined
}

const getConfidenceClass = (confidence) => {
  if (confidence === undefined || confidence === null) return ''
  if (confidence >= 0.7) return 'confidence-high'
  if (confidence >= 0.5) return 'confidence-medium'
  return 'confidence-low'
}

const getTcpEmoji = (risk) => {
  if (!risk) return ''
  const r = risk.toLowerCase()
  if (r === 'moderate') return '\uD83D\uDFE1'
  if (r === 'elevated') return '\uD83D\uDFE0'
  if (r === 'critical') return '\uD83D\uDD34'
  return ''
}

const hasUsefulPredictionInfo = (results) => {
  if (!results) return false
  const hasMatch = results.confidence?.matched_alloy && results.confidence.matched_alloy !== 'None'
  const tcp = results.metallurgy_metrics?.['TCP Risk'] || results.metallurgy_metrics?.tcp_risk || results.tcp_risk
  const hasTcpWarning = tcp && tcp !== 'Low' && tcp !== 'UNKNOWN'
  return hasMatch || hasTcpWarning
}

const formatMetricLabel = (key) => {
  const labelMap = {
    'md_average': 'Md Temperature (avg)', 'sss_wt_pct': 'Solid Solution Strengthening (wt%)',
    'tcp_risk': 'TCP Risk', 'gamma_prime_fraction': "\u03B3' Fraction",
    'lattice_misfit': 'Lattice Misfit', 'partitioning_ni': 'Ni Partitioning',
    'partitioning_cr': 'Cr Partitioning', 'partitioning_al': 'Al Partitioning',
    'partitioning_ti': 'Ti Partitioning'
  }
  return labelMap[key] || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
}

const getIntervalQuality = (lower, upper, actual) => {
  if (lower == null || upper == null || actual == null) return 'unknown'
  const relativeWidth = ((upper - lower) / actual) * 100
  if (relativeWidth < 10) return 'tight'
  if (relativeWidth < 20) return 'moderate'
  return 'wide'
}

const getOverestimationWarning = (prop, actual, upper, confidence) => {
  if (!upper || !actual || upper <= actual) return null
  const overestimation = ((upper - actual) / actual) * 100
  if (confidence < 0.6 && overestimation > 15) return `High uncertainty (\u00B1${overestimation.toFixed(0)}%) - prediction may be optimistic`
  if (overestimation > 25) return `Wide confidence interval (\u00B1${overestimation.toFixed(0)}%) - treat prediction with caution`
  return null
}

const getErrorTitle = (type) => {
  const titles = { network: 'Connection Error', timeout: 'Timeout Error', validation: 'Validation Error', server: 'Server Error' }
  return titles[type] || 'Unexpected Error'
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

// --- PROPERTY COMPARISONS (design mode) ---
const propertyComparisons = computed(() => {
  if (!props.result || props.mode !== 'auto') return []
  const comparisons = []
  const actualProps = props.result.properties || {}
  const propertyIntervals = props.result.property_intervals || {}
  const confidence = props.result.confidence || {}

  const propMap = [
    { key: 'Yield Strength', target: props.targets.yield, unit: 'MPa', icon: '\uD83C\uDFCB\uFE0F' },
    { key: 'Tensile Strength', target: props.targets.tensile, unit: 'MPa', icon: '\u26D3\uFE0F' },
    { key: 'Elongation', target: props.targets.elongation, unit: '%', icon: '\uD83D\uDCCF' },
    { key: 'Elastic Modulus', target: props.targets.elastic_modulus, unit: 'GPa', icon: '\uD83D\uDD27' },
    { key: 'Density', target: props.targets.density, unit: 'g/cm\u00B3', icon: '\uD83E\uDDF1', isMax: true },
    { key: 'Gamma Prime', target: props.targets.gamma_prime, unit: '%', icon: '\uD83D\uDC8E' }
  ]

  for (const prop of propMap) {
    if (prop.target > 0 || (prop.isMax && prop.target < 99)) {
      const actualVal = parseVal(lookUpProp(actualProps, prop.key))
      if (actualVal !== null) {
        const interval = propertyIntervals[prop.key] || {}
        let plusMinus = null
        if (interval.lower !== undefined && interval.upper !== undefined) {
          plusMinus = ((interval.upper - interval.lower) / 2).toFixed(prop.key.includes('Density') ? 2 : 1)
        }

        let met, status, exceeds = false
        let percentage = Math.round((actualVal / prop.target) * 100)

        if (prop.isMax) {
          met = actualVal <= prop.target; exceeds = !met; status = met ? 'In Range' : 'Too High'
        } else {
          if (actualVal < prop.target) { met = false; exceeds = false; status = 'Too Low' }
          else {
            let limit
            if (prop.key.includes('Strength')) { limit = prop.target * 2.0 }
            else if (prop.key.includes('Gamma Prime')) {
              const tolerance = Math.max(2.0, prop.target * 0.2)
              const gp_max = prop.target + tolerance
              if (actualVal > gp_max) { met = false; exceeds = true; status = 'Above Range'; percentage = Math.round((gp_max / actualVal) * 100) }
              else { met = true; exceeds = false; status = 'In Range'; percentage = 100 }
              limit = null
            } else { limit = prop.target * 1.5 }

            if (limit !== null) {
              if (actualVal > limit) { met = false; exceeds = true; status = 'Too High' }
              else { met = true; exceeds = false; status = 'In Range' }
            }
          }
        }

        comparisons.push({ ...prop, actual: actualVal, lower: interval.lower, upper: interval.upper, plusMinus, confidenceScore: confidence.score, percentage: Math.round(percentage), met, exceeds, status })
      }
    }
  }
  return comparisons
})

// --- PARSED RESULTS ---
const parsedResults = computed(() => {
  if (!props.result) return null
  const data = props.result

  let comp = data.composition || props.manualComp
  const rawProps = data.properties || {}
  const propertyIntervals = data.property_intervals || {}
  const confidence = data.confidence || {}
  const explanation = data.explanation || ''
  let auditPenalties = data.audit_penalties || []
  const metallurgyMetrics = data.metallurgy_metrics || {}
  const status = data.status || 'UNKNOWN'
  const tcpRisk = data.tcp_risk || metallurgyMetrics['TCP Risk'] || 'UNKNOWN'
  const similar = data.similar_alloys || []
  const summary = data.summary || ''
  const reasoning = data.reasoning || ''
  const analystReasoning = data.analyst_reasoning || ''
  const reviewerAssessment = data.reviewer_assessment || ''
  const investigationFindings = data.investigation_findings || ''
  const sourceReliability = data.source_reliability || ''
  const correctionsApplied = Array.isArray(data.corrections_applied)
    ? data.corrections_applied.filter(c => c && typeof c === 'object' && c.property_name) : []
  const correctionsExplanation = data.corrections_explanation || ''

  let issues = (data.issues || []).filter(issue => {
    if (issue.type === 'Audit Violation' && issue.description?.includes('No description')) return false
    if (issue.recommendation === 'Review composition constraints.' && !issue.description) return false
    return true
  })

  // Properties shown in target comparisons (avoid duplicates)
  const shownInTargets = new Set()
  if (props.mode === 'auto' && propertyComparisons.value.length > 0) {
    propertyComparisons.value.forEach(c => shownInTargets.add(c.key))
  }

  const formattedProps = [
    { label: 'Yield Strength', val: parseVal(lookUpProp(rawProps, 'Yield Strength')), unit: 'MPa', icon: '\uD83C\uDFCB\uFE0F', hasTarget: props.targets?.yield > 0 },
    { label: 'Tensile Strength', val: parseVal(lookUpProp(rawProps, 'Tensile Strength') || lookUpProp(rawProps, 'Ultimate Tensile Strength')), unit: 'MPa', icon: '\u26D3\uFE0F', hasTarget: props.targets?.tensile > 0 },
    { label: 'Elongation', val: parseVal(lookUpProp(rawProps, 'Elongation')), unit: '%', icon: '\uD83D\uDCCF', hasTarget: props.targets?.elongation > 0 },
    { label: 'Elastic Modulus', val: parseVal(lookUpProp(rawProps, 'Elastic Modulus')), unit: 'GPa', icon: '\uD83D\uDD27', hasTarget: props.targets?.elastic_modulus > 0 },
    { label: 'Density', val: parseVal(lookUpProp(rawProps, 'Density')), unit: 'g/cm\u00B3', icon: '\uD83E\uDDF1', hasTarget: props.targets?.density < 99 },
    { label: 'Gamma Prime', val: parseVal(lookUpProp(rawProps, 'Gamma Prime')), unit: '%', icon: '\uD83D\uDC8E', hasTarget: props.targets?.gamma_prime > 0 },
  ]
    .filter(p => p.val !== undefined && p.val !== null)
    .filter(p => !shownInTargets.has(p.label))
    .filter(p => props.mode === 'manual' || !p.hasTarget)

  formattedProps.forEach(prop => {
    const interval = propertyIntervals[prop.label] || {}
    prop.lower = interval.lower; prop.upper = interval.upper; prop.uncertainty = interval.uncertainty
    if (prop.lower !== undefined && prop.upper !== undefined) {
      prop.interval = ((prop.upper - prop.lower) / 2).toFixed(prop.label.includes('Density') ? 2 : 1)
      prop.intervalQuality = getIntervalQuality(prop.lower, prop.upper, prop.val)
    }
    prop.confidence = confidence?.score || 0.5
    prop.overestimationWarning = getOverestimationWarning(prop, prop.val, prop.upper, prop.confidence)
  })

  // Physics metrics
  const VALID_METRICS = new Set([
    'Md (TCP Stability)', 'TCP Risk', "\u03B3/\u03B3' Misfit (%)", 'Refractory Content (wt%)',
    'Matrix + SSS Strength (MPa)', 'Al+Ti (weldability)', 'Cr (oxidation)',
    'md_average', 'md_avg', 'gamma_prime_vol', 'gamma_prime_fraction',
    'sss_wt_pct', 'lattice_misfit', 'density_gcm3',
    'kg_md_avg', 'kg_tcp_risk', 'kg_sss_wt_pct'
  ])

  const physicsMetrics = []
  if (metallurgyMetrics) {
    Object.entries(metallurgyMetrics).forEach(([key, value]) => {
      if (key === 'tcp_risk' || key === 'TCP Risk') return
      if (!VALID_METRICS.has(key)) return

      let displayValue = typeof value === 'number' ? value.toFixed(2) : value
      let warning = null

      if (key.includes('weldability') && typeof value === 'number') {
        if (value > 6.5) warning = 'Difficult to weld'
        else if (value > 5.0) warning = 'Weld with care'
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

      physicsMetrics.push({ label: formatMetricLabel(key), value: displayValue, warning, key })
    })
  }

  return { comp, formattedProps, propertyIntervals, confidence, explanation, auditPenalties, metallurgyMetrics, physicsMetrics, status, tcpRisk, similar, summary, issues, reasoning, analystReasoning, reviewerAssessment, investigationFindings, sourceReliability, correctionsApplied, correctionsExplanation }
})

const copyToEvaluation = () => {
  emit('copy-to-evaluation')
  showToast('Copied to evaluation mode', 'success')
}
</script>

<template>
  <div class="output-area">
    <!-- ERROR BOUNDARY -->
    <div v-if="error && !loading" class="error-boundary glass-card" role="alert">
      <div class="error-header">
        <span class="error-icon">X</span>
        <h3>{{ getErrorTitle(errorType) }}</h3>
      </div>
      <div class="error-message">{{ error }}</div>
      <div class="error-recovery">
        <h4>Recovery Actions:</h4>
        <ul><li v-for="(action, idx) in getErrorRecoveryActions(errorType)" :key="idx">{{ action }}</li></ul>
      </div>
      <div class="error-actions">
        <button @click="$emit('retry')" :disabled="retryCount >= maxRetries" class="retry-btn">
          Retry {{ retryCount > 0 ? `(${retryCount}/${maxRetries})` : '' }}
        </button>
        <button @click="$emit('dismiss-error')" class="dismiss-btn">Dismiss</button>
      </div>
      <div v-if="logs.length > 0" class="error-logs-scroll">
        <div v-for="(log, i) in logs" :key="i" class="log-line">{{ log }}</div>
      </div>
    </div>

    <!-- LOADING STATE -->
    <div v-if="loading" class="loading-state glass-card" aria-live="polite">
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
        <div v-for="(step, i) in currentSteps" :key="i" :class="['pipeline-dot', { active: i === loadingStep }]"></div>
      </div>
      <div class="pipeline-footer">
        <span class="elapsed-time">{{ elapsedSeconds }}s elapsed</span>
      </div>
      <div v-if="logs.length > 0" class="logs-scroll">
        <div v-for="(log, i) in logs" :key="i" class="log-line">{{ log }}</div>
      </div>
    </div>

    <!-- RESULTS DASHBOARD -->
    <div v-if="parsedResults" class="results-dashboard">
      <div class="dashboard-header">
        <h3>Analysis Complete at {{ result.temperature || temperature }}°C</h3>
        <p class="summary-text">{{ parsedResults.summary }}</p>
      </div>

      <!-- Status Badge -->
      <div v-if="parsedResults.status && parsedResults.status !== 'UNKNOWN'" class="status-badge" :class="parsedResults.status.toLowerCase()">
        {{ parsedResults.status === 'PASS' ? 'PASS' : parsedResults.status === 'REJECT' ? 'REJECT' : 'FAIL' }}
      </div>

      <!-- Prediction Info Panel -->
      <div v-if="hasUsefulPredictionInfo(parsedResults)" class="prediction-info-panel">
        <span v-if="parsedResults.confidence?.matched_alloy && parsedResults.confidence.matched_alloy !== 'None'" class="info-tag similar-alloy">
          Similar to <strong>{{ parsedResults.confidence.matched_alloy }}</strong>
        </span>
        <span v-if="parsedResults.tcpRisk && parsedResults.tcpRisk !== 'Low' && parsedResults.tcpRisk !== 'UNKNOWN'"
              :class="['info-tag', 'tcp-' + parsedResults.tcpRisk.toLowerCase()]">
          {{ getTcpEmoji(parsedResults.tcpRisk) }} TCP: {{ parsedResults.tcpRisk }}
        </span>
      </div>

      <!-- Composition (design mode) -->
      <div v-if="parsedResults.comp && mode === 'auto'" class="final-comp-section">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
          <h4 style="margin: 0;">Suggested Composition</h4>
          <button @click="copyToEvaluation" class="copy-btn" title="Copy to Evaluation Mode">Copy to Evaluation</button>
        </div>
        <div class="mini-comp-grid">
          <span v-for="(v, k) in parsedResults.comp" :key="k" class="mini-comp-tag"><b>{{ k }}</b>: {{ v }}%</span>
        </div>
      </div>

      <!-- Target vs Actual Comparison -->
      <div v-if="propertyComparisons.length > 0 && mode === 'auto'" class="comparison-section">
        <h4>Target Achievement</h4>
        <div class="comparison-list">
          <div v-for="comp in propertyComparisons" :key="comp.key" class="comparison-item">
            <div class="comparison-header">
              <span class="comparison-icon">{{ comp.icon }}</span>
              <span class="comparison-label">{{ comp.key }}</span>
              <span :class="['comparison-status', { met: comp.met, unmet: !comp.met && !comp.exceeds, exceeds: comp.exceeds }]">
                {{ comp.met ? '\u2713 ' + comp.status : (comp.exceeds ? '! ' + comp.status : '\u2717 ' + comp.status) }}
              </span>
            </div>
            <div class="comparison-values">
              <span class="comparison-target">Target: {{ comp.isMax ? '\u2264' : '\u2265' }} {{ comp.target }} {{ comp.unit }}</span>
              <span class="comparison-actual">
                Predicted: <AnimatedNumber :value="comp.actual" :decimals="comp.key.includes('Density') ? 2 : 1" /> {{ comp.unit }}
                <span v-if="comp.plusMinus" class="comparison-interval-discrete">\u00B1{{ comp.plusMinus }}</span>
              </span>
            </div>
            <div class="comparison-bar-container">
              <div :class="['comparison-bar', { met: comp.met, unmet: !comp.met && !comp.exceeds, exceeds: comp.exceeds }]" :style="{ width: Math.min(comp.percentage, 100) + '%' }">
                <span class="comparison-percentage"><AnimatedNumber :value="comp.percentage" :decimals="0" />%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Additional Properties -->
      <div v-if="parsedResults.formattedProps.length > 0" class="props-grid">
        <div v-for="prop in parsedResults.formattedProps" :key="prop.label" :class="['prop-card', getConfidenceClass(prop.confidence)]">
          <div class="prop-icon">{{ prop.icon }}</div>
          <div class="prop-info">
            <div class="prop-label">{{ prop.label }}</div>
            <div class="prop-val">
              <AnimatedNumber :value="prop.val" :decimals="prop.label.includes('Density') ? 2 : 1" />
              <small>{{ prop.unit }}</small>
            </div>
            <div v-if="prop.interval" class="prop-interval-discrete">\u00B1{{ prop.interval }} {{ prop.unit }}</div>
          </div>
        </div>
      </div>

      <!-- Design Issues -->
      <div v-if="parsedResults.issues.length > 0" class="issues-panel">
        <div class="panel-header">Design Issues</div>
        <div class="issues-list">
          <div v-for="(issue, i) in parsedResults.issues" :key="'issue-'+i" :class="['issue-item', 'severity-' + (issue.severity || 'low').toLowerCase()]">
            <div class="issue-header">
              <span class="issue-icon">{{ issue.severity === 'High' ? '\uD83D\uDD34' : issue.severity === 'Medium' ? '\uD83D\uDFE1' : '\uD83D\uDD35' }}</span>
              <span class="issue-type">{{ issue.type }}</span>
              <span class="issue-severity">{{ issue.severity }}</span>
            </div>
            <div class="issue-description">{{ issue.description }}</div>
            <div class="issue-recommendation">{{ issue.recommendation }}</div>
          </div>
        </div>
      </div>

      <!-- Physics Audit Violations -->
      <div v-if="parsedResults.auditPenalties.length > 0 && (parsedResults.status === 'REJECT' || parsedResults.status === 'FAIL')" class="issues-panel">
        <div class="panel-header">Physics Audit Violations</div>
        <div class="issues-list">
          <div v-for="(penalty, i) in parsedResults.auditPenalties" :key="'penalty-'+i" class="issue-item severity-high">
            <div class="issue-header">
              <span class="issue-icon">\uD83D\uDD34</span>
              <span class="issue-type">{{ penalty.name }}</span>
              <span class="issue-severity">{{ penalty.value }}</span>
            </div>
            <div class="issue-description">{{ penalty.reason }}</div>
          </div>
        </div>
      </div>

      <!-- Metallurgical Indicators -->
      <div v-if="parsedResults.physicsMetrics.length > 0" class="metrics-panel">
        <div class="panel-header">Metallurgical Indicators</div>
        <div class="metrics-grid">
          <div v-for="metric in parsedResults.physicsMetrics" :key="metric.key" class="metric-item">
            <span class="metric-label">{{ metric.label }}:</span>
            <span class="metric-value">{{ metric.value }}</span>
            <span v-if="metric.warning" class="metric-warning">{{ metric.warning }}</span>
          </div>
        </div>
      </div>

      <!-- Corrections Applied -->
      <div v-if="parsedResults.correctionsApplied.length > 0" class="corrections-panel">
        <div class="panel-header">Physics Corrections Applied</div>
        <div class="corrections-list">
          <div v-for="(corr, i) in parsedResults.correctionsApplied" :key="'corr-'+i" class="correction-item">
            <div class="correction-header">
              <span class="correction-prop">{{ corr.property_name }}</span>
              <span class="correction-arrow">{{ (Number(corr.original_value) || 0).toFixed(1) }} \u2192 {{ (Number(corr.corrected_value) || 0).toFixed(1) }}</span>
            </div>
            <div class="correction-reason">{{ corr.correction_reason }}</div>
            <div v-if="corr.physics_constraint" class="correction-constraint">{{ corr.physics_constraint }}</div>
          </div>
        </div>
        <div v-if="parsedResults.correctionsExplanation" class="corrections-summary">{{ parsedResults.correctionsExplanation }}</div>
      </div>

      <!-- Agent Investigation -->
      <div v-if="parsedResults.analystReasoning || parsedResults.reviewerAssessment || parsedResults.investigationFindings || parsedResults.sourceReliability" class="agent-reasoning-section">
        <div class="panel-header">Agent Investigation</div>
        <div v-if="parsedResults.sourceReliability" class="source-reliability-badge">
          <span class="reliability-label">Source Reliability:</span>
          <span class="reliability-value">{{ parsedResults.sourceReliability }}</span>
        </div>
        <div v-if="parsedResults.investigationFindings" class="investigation-findings">
          <div class="sub-header">Investigation Findings</div>
          <div class="findings-text">{{ parsedResults.investigationFindings }}</div>
        </div>
        <div v-if="parsedResults.analystReasoning" class="analyst-section">
          <div class="sub-header">Analyst Reasoning</div>
          <div class="reasoning-text">{{ parsedResults.analystReasoning }}</div>
        </div>
        <div v-if="parsedResults.reviewerAssessment" class="reviewer-section">
          <div class="sub-header">Reviewer Assessment</div>
          <div class="assessment-text">{{ parsedResults.reviewerAssessment }}</div>
        </div>
      </div>

      <!-- AI Design Reasoning -->
      <div v-if="parsedResults.reasoning" class="reasoning-panel">
        <div class="panel-header">AI Design Reasoning</div>
        <div class="reasoning-text">{{ parsedResults.reasoning }}</div>
      </div>

      <!-- Metallurgical Analysis -->
      <div v-if="parsedResults.explanation" class="explanation-panel">
        <div class="panel-header">Metallurgical Analysis</div>
        <div class="explanation-text">{{ parsedResults.explanation }}</div>
      </div>

      <!-- Similar Alloys -->
      <div v-if="parsedResults.similar.length" class="similar-section">
        <h4>Similar Known Alloys</h4>
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
</template>

<style scoped>
/* Output Area */
.output-area { background: var(--bg-panel); padding: 1.5rem; border-radius: 12px; flex: 1; min-height: 300px; border: 1px solid var(--border-subtle); }

/* Error Boundary */
.error-boundary { background: rgba(239, 71, 111, 0.1); border: 2px solid var(--danger); padding: var(--space-xl); border-radius: var(--radius-lg); margin-bottom: var(--space-lg); animation: shake 0.5s ease-in-out; }
@keyframes shake { 0%, 100% { transform: translateX(0); } 10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); } 20%, 40%, 60%, 80% { transform: translateX(5px); } }
.error-header { display: flex; align-items: center; gap: var(--space-md); margin-bottom: var(--space-md); }
.error-icon { font-size: 2rem; color: var(--danger); font-weight: bold; }
.error-header h3 { margin: 0; color: var(--danger); font-size: var(--font-size-xl); }
.error-message { background: var(--bg-input); padding: var(--space-md); border-radius: var(--radius-md); color: var(--text-primary); margin-bottom: var(--space-md); font-family: monospace; font-size: var(--font-size-sm); }
.error-recovery { background: var(--bg-glass); padding: var(--space-md); border-radius: var(--radius-md); margin-bottom: var(--space-md); }
.error-recovery h4 { margin: 0 0 var(--space-sm) 0; color: var(--text-secondary); font-size: var(--font-size-md); }
.error-recovery ul { margin: 0; padding-left: var(--space-lg); color: var(--text-primary); line-height: 1.8; }
.error-actions { display: flex; gap: var(--space-md); margin-bottom: var(--space-md); }
.retry-btn, .dismiss-btn { padding: var(--space-sm) var(--space-lg); border-radius: var(--radius-md); border: none; font-weight: var(--font-weight-semibold); cursor: pointer; transition: all var(--transition-base); font-family: var(--font-family); }
.retry-btn { background: linear-gradient(135deg, var(--primary), var(--secondary)); color: white; flex: 1; }
.retry-btn:hover:not(:disabled) { transform: translateY(-2px); box-shadow: var(--shadow-lg); }
.retry-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.dismiss-btn { background: var(--bg-glass); color: var(--text-primary); border: 1px solid var(--border-strong); }
.dismiss-btn:hover { background: var(--bg-elevated); }
.error-logs-scroll { max-height: 120px; overflow-y: auto; background: var(--bg-input); padding: var(--space-sm); border-radius: var(--radius-md); font-family: monospace; font-size: 0.75rem; color: var(--text-muted); }

/* Loading State */
.loading-state { padding: var(--space-xl); text-align: center; animation: fadeIn 0.3s ease-in; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
.pipeline-header { display: flex; align-items: center; justify-content: center; gap: 0.75rem; margin-bottom: 1.5rem; }
.pipeline-spinner { width: 22px; height: 22px; border: 2.5px solid var(--border-subtle); border-top-color: var(--primary, #00d4ff); border-radius: 50%; animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.pipeline-title { font-size: 1rem; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.08em; }
.pipeline-active-step { display: flex; align-items: center; justify-content: center; gap: 0.6rem; padding: 0.75rem 1.25rem; background: rgba(0, 212, 255, 0.06); border: 1px solid rgba(0, 212, 255, 0.15); border-radius: 12px; margin: 0 auto 1.25rem; max-width: 420px; animation: stepFadeIn 0.4s ease; }
@keyframes stepFadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
.active-step-icon { font-size: 1.3rem; flex-shrink: 0; }
.active-step-label { font-size: 0.95rem; font-weight: 500; color: var(--primary); }
.active-step-dots { color: rgba(0, 212, 255, 0.5); }
.dot-anim { display: inline-block; animation: dotPulse 1.4s ease-in-out infinite; }
@keyframes dotPulse { 0%, 100% { opacity: 0.2; } 50% { opacity: 1; } }
.pipeline-track { display: flex; align-items: center; justify-content: center; gap: 0.5rem; margin-bottom: 1rem; }
.pipeline-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--border-subtle); transition: all 0.3s ease; }
.pipeline-dot.active { width: 10px; height: 10px; background: var(--primary, #00d4ff); box-shadow: 0 0 8px rgba(0, 212, 255, 0.5); }
.pipeline-footer { display: flex; align-items: center; justify-content: center; padding-top: 0.5rem; border-top: 1px solid var(--border-subtle); }
.pipeline-footer .elapsed-time { font-size: 0.8rem; color: var(--text-muted); font-family: monospace; }
.logs-scroll { max-height: 150px; overflow-y: auto; text-align: left; font-family: monospace; font-size: 0.8rem; color: var(--text-muted); border-top: 1px solid var(--border-subtle); padding-top: 10px; }

/* Results Dashboard */
.results-dashboard { color: var(--text-primary); }
.dashboard-header { border-bottom: 1px solid var(--border-subtle); padding-bottom: 1rem; margin-bottom: 1.5rem; }
.summary-text { font-style: italic; color: var(--text-secondary); margin-top: 0.5rem; font-size: 1.1rem; line-height: 1.4; }

/* Status Badge */
.status-badge { display: inline-block; padding: 0.75rem 1.5rem; border-radius: 8px; font-weight: 900; font-size: 1.3rem; margin-bottom: 1.5rem; text-transform: uppercase; letter-spacing: 1px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3); border: 2px solid transparent; }
.status-badge.pass { background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; border-color: #20c997; animation: pulse-green 2s infinite; }
.status-badge.reject { background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%); color: #1a1a1a; border-color: #ff9800; font-weight: 900; animation: pulse-orange 2s infinite; }
.status-badge.fail { background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); color: white; border-color: #c82333; animation: pulse-red 2s infinite; }
@keyframes pulse-green { 0%, 100% { box-shadow: 0 4px 12px rgba(40, 167, 69, 0.4); } 50% { box-shadow: 0 4px 20px rgba(40, 167, 69, 0.7); } }
@keyframes pulse-orange { 0%, 100% { box-shadow: 0 4px 12px rgba(255, 193, 7, 0.5); } 50% { box-shadow: 0 4px 20px rgba(255, 193, 7, 0.8); } }
@keyframes pulse-red { 0%, 100% { box-shadow: 0 4px 12px rgba(220, 53, 69, 0.4); } 50% { box-shadow: 0 4px 20px rgba(220, 53, 69, 0.7); } }

/* Prediction Info */
.prediction-info-panel { display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1rem; }
.info-tag { display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.4rem 0.75rem; border-radius: 6px; font-size: 0.85rem; background: var(--bg-glass); color: var(--text-secondary); }
.info-tag.similar-alloy { background: rgba(0, 212, 255, 0.15); color: #00d4ff; }
.info-tag.tcp-moderate { background: rgba(255, 193, 7, 0.15); color: #ffc107; }
.info-tag.tcp-elevated { background: rgba(255, 152, 0, 0.15); color: #ff9800; }
.info-tag.tcp-critical { background: rgba(220, 53, 69, 0.15); color: #dc3545; }

/* Composition */
.final-comp-section { background: var(--bg-elevated); padding: 1rem; border-radius: 8px; margin-bottom: 2rem; }
.mini-comp-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 0.5rem; }
.mini-comp-tag { background: var(--bg-input); padding: 4px 8px; border-radius: 4px; border: 1px solid var(--border-subtle); font-family: monospace; }
.copy-btn { background: var(--success); color: white; border: none; padding: 8px 16px; border-radius: 6px; font-size: 0.9rem; font-weight: 600; cursor: pointer; transition: all 0.2s; }
.copy-btn:hover { background: var(--success); transform: translateY(-2px); box-shadow: 0 4px 8px rgba(40, 167, 69, 0.3); }

/* Comparison */
.comparison-section { background: var(--bg-glass); padding: var(--space-lg); border-radius: var(--radius-lg); margin-bottom: var(--space-xl); border: 1px solid var(--border-subtle); }
.comparison-section h4 { margin: 0 0 var(--space-lg) 0; color: var(--text-primary); font-size: var(--font-size-lg); }
.comparison-list { display: flex; flex-direction: column; gap: var(--space-md); }
.comparison-item { background: var(--bg-input); padding: var(--space-md); border-radius: var(--radius-md); border: 1px solid var(--border-subtle); }
.comparison-header { display: flex; align-items: center; gap: var(--space-sm); margin-bottom: var(--space-sm); }
.comparison-icon { font-size: 1.5rem; }
.comparison-label { flex: 1; font-weight: var(--font-weight-semibold); color: var(--text-primary); font-size: var(--font-size-md); }
.comparison-status { padding: var(--space-xs) var(--space-sm); border-radius: var(--radius-sm); font-size: 0.75rem; font-weight: var(--font-weight-bold); }
.comparison-status.met { background: rgba(6, 214, 160, 0.2); color: var(--success); }
.comparison-status.unmet { background: rgba(239, 71, 111, 0.2); color: var(--danger); }
.comparison-status.exceeds { background: rgba(255, 193, 7, 0.2); color: #ffc107; }
.comparison-values { display: flex; justify-content: space-between; margin-bottom: var(--space-sm); font-size: var(--font-size-sm); color: var(--text-secondary); }
.comparison-target { color: var(--text-muted); }
.comparison-actual { color: var(--primary); font-weight: var(--font-weight-semibold); }
.comparison-interval-discrete { color: var(--text-muted); font-size: 0.8rem; opacity: 0.9; font-style: italic; margin-left: 0.25rem; font-weight: 400; }
.comparison-bar-container { height: 28px; background: var(--bg-glass); border-radius: var(--radius-md); overflow: hidden; position: relative; }
.comparison-bar { height: 100%; border-radius: var(--radius-md); transition: width 0.8s ease-out; display: flex; align-items: center; justify-content: flex-end; padding-right: var(--space-sm); position: relative; overflow: hidden; }
.comparison-bar.met { background: linear-gradient(90deg, rgba(6, 214, 160, 0.6), var(--success)); }
.comparison-bar.unmet { background: linear-gradient(90deg, rgba(239, 71, 111, 0.6), var(--danger)); }
.comparison-bar.exceeds { background: linear-gradient(90deg, rgba(255, 193, 7, 0.6), #ffc107); }
.comparison-bar::before { content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent); animation: bar-shimmer 2s ease-in-out infinite; }
@keyframes bar-shimmer { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }
.comparison-percentage { position: relative; z-index: 1; color: white; font-weight: var(--font-weight-bold); font-size: 0.8rem; text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5); }

/* Properties Grid */
.props-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
.prop-card { background: var(--bg-elevated); padding: 1rem; border-radius: 10px; display: flex; align-items: center; gap: 1rem; border: 1px solid var(--border-subtle); }
.prop-card.confidence-high { border-left: 3px solid rgba(76, 175, 80, 0.6); }
.prop-card.confidence-medium { border-left: 3px solid rgba(255, 193, 7, 0.5); }
.prop-card.confidence-low { border-left: 3px solid rgba(244, 67, 54, 0.5); }
.prop-icon { font-size: 2rem; }
.prop-val { font-size: 1.4rem; font-weight: bold; color: var(--primary); }
.prop-val small { font-size: 0.8rem; color: var(--text-muted); font-weight: normal; }
.prop-label { font-size: 0.9rem; color: var(--text-muted); }
.prop-interval-discrete { color: var(--text-muted); font-size: 0.75rem; opacity: 0.7; font-style: italic; margin-top: 0.25rem; }

/* Panels */
.explanation-panel, .metrics-panel { background: var(--bg-elevated); padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; border-left: 3px solid var(--primary); }
.panel-header { font-size: 1rem; font-weight: bold; color: var(--primary); margin-bottom: 0.75rem; }
.explanation-text { font-size: 0.95rem; color: var(--text-secondary); line-height: 1.7; white-space: pre-wrap; }
.metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 0.5rem; }
.metric-item { background: var(--bg-input); padding: 0.5rem; border-radius: 4px; font-size: 0.85rem; }
.metric-label { color: var(--text-muted); margin-right: 0.5rem; }
.metric-value { color: var(--primary); font-weight: bold; }
.metric-warning { display: block; margin-top: 0.25rem; font-size: 0.75rem; color: #ff9800; }

/* Issues */
.issues-panel { background: rgba(255, 200, 0, 0.05); border: 1px solid rgba(255, 200, 0, 0.3); border-radius: var(--radius-lg); padding: var(--space-lg); margin-bottom: var(--space-xl); }
.issues-list { display: flex; flex-direction: column; gap: var(--space-md); margin-top: var(--space-md); }
.issue-item { background: var(--bg-input); padding: var(--space-md); border-radius: var(--radius-md); border-left: 4px solid; }
.issue-item.severity-high { border-left-color: var(--danger); }
.issue-item.severity-medium { border-left-color: #f39c12; }
.issue-item.severity-low { border-left-color: #3498db; }
.issue-header { display: flex; align-items: center; gap: var(--space-sm); margin-bottom: var(--space-sm); }
.issue-icon { font-size: 1.2rem; }
.issue-type { flex: 1; font-weight: var(--font-weight-semibold); color: var(--text-primary); }
.issue-severity { padding: 2px 8px; border-radius: var(--radius-sm); font-size: 0.7rem; font-weight: var(--font-weight-bold); background: var(--bg-glass); color: var(--text-secondary); }
.issue-description { color: var(--text-secondary); margin-bottom: var(--space-sm); line-height: 1.6; }
.issue-recommendation { color: var(--primary); font-size: var(--font-size-sm); padding: var(--space-xs) var(--space-sm); background: rgba(17, 153, 250, 0.1); border-radius: var(--radius-sm); border-left: 3px solid var(--primary); }

/* Corrections */
.corrections-panel { background: rgba(255, 193, 7, 0.05); border: 1px solid rgba(255, 193, 7, 0.2); border-radius: var(--radius-lg); padding: var(--space-lg); margin-bottom: var(--space-xl); }
.corrections-list { display: flex; flex-direction: column; gap: 0.75rem; }
.correction-item { padding: 0.75rem; background: var(--bg-glass); border-radius: var(--radius-md); border-left: 3px solid rgba(255, 193, 7, 0.4); }
.correction-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem; }
.correction-prop { font-weight: 600; color: #ffc107; font-size: 0.9rem; }
.correction-arrow { font-family: monospace; color: var(--primary); font-size: 0.85rem; }
.correction-reason { color: var(--text-secondary); font-size: 0.85rem; line-height: 1.5; }
.correction-constraint { color: var(--text-muted); font-size: 0.8rem; font-style: italic; margin-top: 0.25rem; }
.corrections-summary { margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--border-subtle); color: var(--text-secondary); font-size: 0.9rem; line-height: 1.6; }

/* Agent Reasoning */
.agent-reasoning-section { background: rgba(0, 212, 255, 0.04); border: 1px solid rgba(0, 212, 255, 0.15); border-radius: var(--radius-lg); padding: var(--space-lg); margin-bottom: var(--space-xl); }
.agent-reasoning-section .sub-header { font-size: 0.9rem; font-weight: 600; color: var(--primary); margin-bottom: 0.5rem; margin-top: 1rem; }
.agent-reasoning-section .sub-header:first-of-type { margin-top: 0.5rem; }
.source-reliability-badge { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0.75rem; background: rgba(0, 212, 255, 0.1); border-radius: 6px; font-size: 0.85rem; margin-bottom: 0.75rem; }
.reliability-label { color: var(--text-secondary); }
.reliability-value { color: var(--primary); font-weight: 600; }
.findings-text, .assessment-text { color: var(--text-secondary); line-height: 1.7; white-space: pre-wrap; font-size: 0.92rem; }
.analyst-section, .reviewer-section, .investigation-findings { padding: 0.75rem; background: var(--bg-glass); border-radius: var(--radius-md); margin-top: 0.5rem; }
.reviewer-section { border-left: 3px solid rgba(123, 44, 191, 0.5); }
.analyst-section { border-left: 3px solid rgba(0, 212, 255, 0.4); }
.investigation-findings { border-left: 3px solid rgba(255, 193, 7, 0.4); }

/* Reasoning */
.reasoning-panel { background: rgba(17, 153, 250, 0.05); border: 1px solid rgba(17, 153, 250, 0.2); border-radius: var(--radius-lg); padding: var(--space-lg); margin-bottom: var(--space-xl); }
.reasoning-text { color: var(--text-secondary); line-height: 1.8; white-space: pre-wrap; }

/* Similar Alloys */
.similar-section { margin-bottom: var(--space-xl); }
.similar-section h4 { margin-bottom: var(--space-md); color: var(--text-primary); }
.similar-list { display: flex; flex-direction: column; gap: 0.5rem; }
.similar-item { background: var(--bg-input); padding: 10px; border-radius: 6px; border-left: 3px solid var(--text-muted); }
.similarity-badge { font-size: 0.75rem; padding: 2px 6px; background: rgba(17, 153, 250, 0.15); color: var(--primary); border-radius: var(--radius-sm); margin-left: 0.5rem; }
.similar-notes { font-size: 0.8rem; color: var(--text-muted); margin-top: 4px; }

/* Log Lines */
.log-line { font-size: 0.8rem; color: var(--text-muted); padding: 0.15rem 0; font-family: 'SF Mono', 'Fira Code', monospace; }

/* Responsive */
@media (max-width: 768px) {
  .output-area { padding: 1rem; min-height: 200px; }
  .props-grid { grid-template-columns: 1fr 1fr; }
  .metrics-grid { grid-template-columns: 1fr; }
  .error-actions { flex-direction: column; }
  .comparison-values { flex-direction: column; gap: var(--space-xs); align-items: flex-start; }
  .comparison-section { padding: var(--space-md); }
  .explanation-panel, .metrics-panel { padding: 0.75rem; }
  .issues-panel, .corrections-panel, .agent-reasoning-section, .reasoning-panel { padding: var(--space-md); }
  .pipeline-active-step { max-width: 100%; }
  .status-badge { font-size: 1.1rem; padding: 0.6rem 1.2rem; }
}

@media (max-width: 480px) {
  .output-area { padding: 0.75rem; min-height: 150px; border-radius: 8px; }
  .props-grid { grid-template-columns: 1fr 1fr; gap: 0.5rem; }
  .prop-card { padding: 0.75rem; gap: 0.5rem; }
  .prop-icon { font-size: 1.5rem; }
  .prop-val { font-size: 1.1rem; }
  .prop-label { font-size: 0.8rem; }
  .metrics-grid { gap: 0.35rem; }
  .metric-item { padding: 0.4rem; font-size: 0.8rem; }
  .summary-text { font-size: 0.95rem; }
  .status-badge { font-size: 1rem; padding: 0.5rem 1rem; letter-spacing: 0.5px; }
  .final-comp-section { padding: 0.75rem; }
  .mini-comp-tag { padding: 3px 6px; font-size: 0.8rem; }
  .loading-state { padding: var(--space-md); }
  .pipeline-active-step { padding: 0.5rem 0.75rem; }
  .active-step-label { font-size: 0.85rem; }
}
</style>
