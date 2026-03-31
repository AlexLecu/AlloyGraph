<script setup>
import { ref, computed, watch, nextTick, onMounted, onUnmounted } from 'vue'

const props = defineProps({
  visible: { type: Boolean, default: false },
  items: { type: Array, default: () => [] },
})

const emit = defineEmits(['close'])

const modalRef = ref(null)

// --- HELPERS (same as ResultsDashboard) ---
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

// --- PROPERTY DEFINITIONS ---
const PROPERTY_DEFS = [
  { key: 'Yield Strength', unit: 'MPa', higherIsBetter: true, decimals: 1 },
  { key: 'Tensile Strength', unit: 'MPa', higherIsBetter: true, decimals: 1 },
  { key: 'Elongation', unit: '%', higherIsBetter: true, decimals: 1 },
  { key: 'Elastic Modulus', unit: 'GPa', higherIsBetter: true, decimals: 1 },
  { key: 'Density', unit: 'g/cm\u00B3', higherIsBetter: false, decimals: 2 },
  { key: 'Gamma Prime', unit: '%', higherIsBetter: null, decimals: 1 },
]

// --- COMPUTED: UNION OF ALL ELEMENTS ---
const allElements = computed(() => {
  const elSet = new Set()
  props.items.forEach(item => {
    if (item.composition) Object.keys(item.composition).forEach(el => elSet.add(el))
  })
  return Array.from(elSet).sort((a, b) => {
    if (a === 'Ni') return -1
    if (b === 'Ni') return 1
    return a.localeCompare(b)
  })
})

// --- COMPUTED: PROPERTY ROWS WITH BEST/WORST ---
const propertyRows = computed(() => {
  return PROPERTY_DEFS.map(def => {
    const values = props.items.map(item => {
      const raw = lookUpProp(item.properties, def.key)
        ?? lookUpProp(item.design?.properties, def.key)
      return parseVal(raw)
    })

    let bestIdx = null
    let worstIdx = null
    if (def.higherIsBetter !== null) {
      const valid = values.map((v, i) => ({ v, i })).filter(e => e.v !== null)
      if (valid.length >= 2) {
        if (def.higherIsBetter) {
          bestIdx = valid.reduce((a, b) => a.v >= b.v ? a : b).i
          worstIdx = valid.reduce((a, b) => a.v <= b.v ? a : b).i
        } else {
          bestIdx = valid.reduce((a, b) => a.v <= b.v ? a : b).i
          worstIdx = valid.reduce((a, b) => a.v >= b.v ? a : b).i
        }
        if (values[bestIdx] === values[worstIdx]) { bestIdx = null; worstIdx = null }
      }
    }

    return { ...def, values, bestIdx, worstIdx }
  })
})

// --- FOCUS TRAP & KEYBOARD ---
const trapFocus = (e) => {
  const modal = modalRef.value
  if (!modal) return
  const focusable = modal.querySelectorAll('button, [tabindex]:not([tabindex="-1"])')
  if (focusable.length === 0) return
  const first = focusable[0]
  const last = focusable[focusable.length - 1]
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus() }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus() }
}

const onKeydown = (e) => {
  if (e.key === 'Escape' && props.visible) emit('close')
}

watch(() => props.visible, async (val) => {
  if (val) {
    await nextTick()
    const closeBtn = modalRef.value?.querySelector('.close-btn')
    if (closeBtn) closeBtn.focus()
  }
})

onMounted(() => document.addEventListener('keydown', onKeydown))
onUnmounted(() => document.removeEventListener('keydown', onKeydown))

const formatVal = (val, decimals) => val !== null ? val.toFixed(decimals) : '-'
</script>

<template>
  <transition name="fade">
    <div v-if="visible" class="modal-overlay" @click.self="$emit('close')"
         role="dialog" aria-modal="true" aria-label="History Comparison">
      <div class="comparison-modal" ref="modalRef" @keydown.tab="trapFocus($event)">

        <!-- Header -->
        <div class="modal-header">
          <h3>Compare {{ items.length }} Designs</h3>
          <button class="close-btn" @click="$emit('close')" aria-label="Close comparison">&times;</button>
        </div>

        <!-- Body -->
        <div class="modal-body">
          <div class="comparison-table" :style="{ '--col-count': items.length }">

            <!-- Run Info -->
            <div class="comparison-section">
              <div class="section-label-row">
                <div class="row-label">Run Info</div>
                <div v-for="item in items" :key="'info-' + item.id" class="col-value">
                  <span class="history-mode-badge" :class="item.mode">
                    {{ item.mode === 'manual' ? 'Evaluate' : 'Design' }}
                  </span>
                </div>
              </div>
              <div class="comparison-row">
                <div class="row-label">Date</div>
                <div v-for="item in items" :key="'ts-' + item.id" class="col-value">
                  {{ new Date(item.timestamp).toLocaleDateString() }}
                  <span class="sub-text">{{ new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }}</span>
                </div>
              </div>
              <div class="comparison-row">
                <div class="row-label">Temperature</div>
                <div v-for="item in items" :key="'temp-' + item.id" class="col-value">
                  {{ item.temperature }}&deg;C
                </div>
              </div>
              <div class="comparison-row">
                <div class="row-label">Processing</div>
                <div v-for="item in items" :key="'proc-' + item.id" class="col-value processing-tag">
                  {{ item.processing }}
                </div>
              </div>
            </div>

            <!-- Status -->
            <div class="comparison-section">
              <div class="comparison-row">
                <div class="row-label">Status</div>
                <div v-for="item in items" :key="'status-' + item.id" class="col-value">
                  <span :class="['status-mini', (item.design?.status || '').toLowerCase()]">
                    {{ item.design?.status || 'N/A' }}
                  </span>
                </div>
              </div>
              <div class="comparison-row">
                <div class="row-label">TCP Risk</div>
                <div v-for="item in items" :key="'tcp-' + item.id" class="col-value">
                  {{ item.design?.tcp_risk || item.design?.metallurgy_metrics?.['TCP Risk'] || 'N/A' }}
                </div>
              </div>
            </div>

            <!-- Composition -->
            <div class="comparison-section">
              <div class="section-label-row">
                <div class="row-label">Composition</div>
                <div v-for="item in items" :key="'comp-h-' + item.id" class="col-value col-header">wt%</div>
              </div>
              <div v-for="el in allElements" :key="'el-' + el" class="comparison-row">
                <div class="row-label element-label">{{ el }}</div>
                <div v-for="item in items" :key="'el-' + el + '-' + item.id" class="col-value mono">
                  {{ item.composition?.[el] != null ? Number(item.composition[el]).toFixed(1) : '-' }}
                </div>
              </div>
            </div>

            <!-- Properties -->
            <div class="comparison-section">
              <div class="section-label-row">
                <div class="row-label">Properties</div>
                <div v-for="item in items" :key="'prop-h-' + item.id" class="col-value col-header">Predicted</div>
              </div>
              <div v-for="row in propertyRows" :key="row.key" class="comparison-row">
                <div class="row-label">
                  {{ row.key }}
                  <span class="unit-label">{{ row.unit }}</span>
                </div>
                <div v-for="(val, idx) in row.values" :key="row.key + '-' + idx"
                     :class="['col-value', 'mono', {
                       'best-value': idx === row.bestIdx,
                       'worst-value': idx === row.worstIdx
                     }]">
                  {{ formatVal(val, row.decimals) }}
                </div>
              </div>
            </div>

          </div>
        </div>

        <!-- Footer -->
        <div class="modal-footer">
          <button class="close-action-btn" @click="$emit('close')">Close</button>
        </div>
      </div>
    </div>
  </transition>
</template>

<style scoped>
/* Modal Overlay */
.modal-overlay {
  position: fixed; inset: 0; background: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(4px); display: flex; align-items: center;
  justify-content: center; z-index: 1000; padding: var(--space-xl);
}

.comparison-modal {
  background: var(--bg-card); backdrop-filter: blur(10px);
  border: 1px solid var(--border-subtle); border-radius: var(--radius-xl);
  max-width: 900px; width: 95%; max-height: 85vh; overflow: hidden;
  display: flex; flex-direction: column; box-shadow: var(--shadow-lg);
}

.modal-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: var(--space-lg); border-bottom: 1px solid var(--border-subtle);
}
.modal-header h3 { margin: 0; font-size: var(--font-size-lg); color: var(--text-primary); }

.close-btn {
  background: none; border: none; font-size: 1.5rem; color: var(--text-muted);
  cursor: pointer; padding: 0; line-height: 1; transition: color var(--transition-base);
}
.close-btn:hover { color: var(--text-primary); }

.modal-body { padding: var(--space-lg); overflow-y: auto; flex: 1; }

.modal-footer {
  display: flex; justify-content: flex-end; padding: var(--space-md) var(--space-lg);
  border-top: 1px solid var(--border-subtle);
}
.close-action-btn {
  padding: var(--space-sm) var(--space-lg); background: var(--bg-glass);
  border: 1px solid var(--border-strong); border-radius: var(--radius-md);
  color: var(--text-primary); cursor: pointer; transition: all var(--transition-base);
  font-family: var(--font-family); font-size: var(--font-size-sm);
}
.close-action-btn:hover { background: var(--bg-elevated); }

/* Table Layout */
.comparison-section {
  margin-bottom: var(--space-lg); padding-bottom: var(--space-md);
  border-bottom: 1px solid var(--border-subtle);
}
.comparison-section:last-child { border-bottom: none; margin-bottom: 0; }

.section-label-row, .comparison-row {
  display: grid;
  grid-template-columns: 160px repeat(var(--col-count, 2), 1fr);
  gap: var(--space-sm); align-items: center; padding: var(--space-xs) 0;
}
.section-label-row {
  font-weight: var(--font-weight-semibold); color: var(--primary);
  font-size: var(--font-size-sm); text-transform: uppercase; letter-spacing: 0.5px;
  padding-bottom: var(--space-sm); border-bottom: 1px solid var(--border-subtle);
  margin-bottom: var(--space-xs);
}
.comparison-row:hover { background: var(--bg-glass); border-radius: var(--radius-sm); }

.row-label {
  font-size: var(--font-size-sm); color: var(--text-secondary);
  font-weight: var(--font-weight-medium);
}
.col-value { font-size: var(--font-size-sm); color: var(--text-primary); text-align: center; }
.col-value.mono { font-family: monospace; }
.col-header { font-size: var(--font-size-xs); color: var(--text-muted); }

.unit-label { color: var(--text-muted); font-size: var(--font-size-xs); font-style: italic; margin-left: 4px; }
.element-label { font-family: monospace; font-weight: var(--font-weight-bold); }
.sub-text { display: block; font-size: var(--font-size-xs); color: var(--text-muted); }
.processing-tag { text-transform: capitalize; }

/* Best/Worst Highlighting */
.best-value {
  color: var(--success); font-weight: var(--font-weight-bold);
  background: rgba(6, 214, 160, 0.1); border-radius: var(--radius-sm);
  padding: 2px 6px;
}
.worst-value {
  color: rgba(239, 71, 111, 0.7);
  background: rgba(239, 71, 111, 0.06); border-radius: var(--radius-sm);
  padding: 2px 6px;
}

/* Status Badges */
.status-mini {
  display: inline-block; padding: 2px 8px; border-radius: var(--radius-sm);
  font-size: var(--font-size-xs); font-weight: var(--font-weight-bold); text-transform: uppercase;
}
.status-mini.pass { background: rgba(6, 214, 160, 0.2); color: var(--success); }
.status-mini.reject { background: rgba(255, 193, 7, 0.2); color: #ffc107; }
.status-mini.fail { background: rgba(239, 71, 111, 0.2); color: var(--danger); }

.history-mode-badge {
  display: inline-block; padding: var(--space-xs) var(--space-sm);
  border-radius: var(--radius-sm); font-size: 0.75rem; font-weight: var(--font-weight-semibold);
}
.history-mode-badge.manual { background: rgba(99, 102, 241, 0.2); color: var(--primary); }
.history-mode-badge.auto { background: rgba(236, 72, 153, 0.2); color: var(--secondary); }

/* Transitions */
.fade-enter-active, .fade-leave-active { transition: opacity var(--transition-base); }
.fade-enter-from, .fade-leave-to { opacity: 0; }

/* Responsive */
@media (max-width: 768px) {
  .modal-overlay { padding: var(--space-md); }
  .comparison-modal { max-width: 100%; max-height: 90vh; }
  .modal-header { padding: var(--space-md); }
  .modal-body { padding: var(--space-md); }
  .section-label-row, .comparison-row {
    grid-template-columns: 120px repeat(var(--col-count, 2), 1fr);
    gap: var(--space-xs);
  }
  .row-label { font-size: var(--font-size-xs); }
}

@media (max-width: 480px) {
  .modal-overlay { padding: var(--space-sm); }
  .comparison-modal { border-radius: var(--radius-md); }
  .modal-header { padding: var(--space-sm) var(--space-md); }
  .modal-header h3 { font-size: var(--font-size-md); }
  .modal-body { padding: var(--space-sm); overflow-x: auto; }
  .section-label-row, .comparison-row {
    grid-template-columns: 80px repeat(var(--col-count, 2), 1fr);
    gap: 4px;
    font-size: var(--font-size-xs);
  }
  .col-value { font-size: var(--font-size-xs); }
  .modal-footer { padding: var(--space-sm) var(--space-md); }
}
</style>
