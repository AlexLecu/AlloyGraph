<script setup>
import { ref, computed, watch } from 'vue'
import { useToast } from '../composables/useToast'

const { showToast } = useToast()

const props = defineProps({
  history: { type: Array, required: true },
  maxItems: { type: Number, default: 20 },
  visible: { type: Boolean, default: false },
})

const emit = defineEmits(['load-item', 'clear', 'compare'])

// --- SELECTION STATE ---
const selectedIds = ref(new Set())
const maxSelected = 3

const selectedCount = computed(() => selectedIds.value.size)
const canCompare = computed(() => selectedCount.value >= 2)

const toggleSelection = (itemId, event) => {
  event.stopPropagation()
  const next = new Set(selectedIds.value)
  if (next.has(itemId)) {
    next.delete(itemId)
  } else {
    if (next.size >= maxSelected) {
      showToast(`Maximum ${maxSelected} items for comparison`, 'warning')
      return
    }
    next.add(itemId)
  }
  selectedIds.value = next
}

const isSelected = (itemId) => selectedIds.value.has(itemId)
const isSelectionDisabled = (itemId) => !selectedIds.value.has(itemId) && selectedIds.value.size >= maxSelected

const openComparison = () => {
  const items = props.history.filter(item => selectedIds.value.has(item.id))
  emit('compare', items)
}

const clearSelection = () => {
  selectedIds.value = new Set()
}

// Clear selection when history is cleared
watch(() => props.history.length, (len) => {
  if (len === 0) selectedIds.value = new Set()
})

// --- EXISTING ACTIONS ---
const clearHistory = () => {
  if (confirm('Are you sure you want to clear all design history?')) {
    emit('clear')
    showToast('History cleared', 'info')
  }
}

const exportHistory = () => {
  const data = selectedCount.value > 0
    ? props.history.filter(item => selectedIds.value.has(item.id))
    : props.history
  const dataStr = JSON.stringify(data, null, 2)
  const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr)
  const exportFileDefaultName = `alloy_history_${new Date().toISOString().split('T')[0]}.json`
  const linkElement = document.createElement('a')
  linkElement.setAttribute('href', dataUri)
  linkElement.setAttribute('download', exportFileDefaultName)
  linkElement.click()
  const msg = selectedCount.value > 0 ? `Exported ${selectedCount.value} items` : 'History exported'
  showToast(msg, 'success')
}
</script>

<template>
  <transition name="slide-down">
    <div v-if="visible" class="history-panel glass-card">
      <div class="history-panel-sticky">
        <div class="history-header">
          <h3>Design History ({{ history.length }}/{{ maxItems }})</h3>
          <div class="history-actions">
            <template v-if="selectedCount > 0">
              <span class="selection-count">{{ selectedCount }} selected</span>
              <button @click="openComparison" class="small-btn compare-btn" :disabled="!canCompare"
                      aria-label="Compare selected items">
                Compare
              </button>
              <button @click="exportHistory" class="small-btn" aria-label="Export selected items">
                Export
              </button>
              <button @click="clearSelection" class="small-btn" aria-label="Clear selection">
                Deselect
              </button>
            </template>
            <template v-else>
              <button @click="exportHistory" class="small-btn" :disabled="history.length === 0" aria-label="Export history as JSON">
                Export
              </button>
              <button @click="clearHistory" class="small-btn danger" :disabled="history.length === 0" aria-label="Clear all history">
                Clear
              </button>
            </template>
          </div>
        </div>
      </div>

      <div v-if="history.length === 0" class="empty-history">
        <p>No design history yet. Run a validation or design to start building your history.</p>
      </div>
      <div v-else class="history-list">
        <div v-for="item in history" :key="item.id"
             class="history-item" :class="{ 'item-selected': isSelected(item.id) }"
             @click="$emit('load-item', item)" role="button" tabindex="0"
             @keydown.enter="$emit('load-item', item)">
          <div class="history-item-header">
            <button
                   :class="['compare-checkbox', { checked: isSelected(item.id), disabled: isSelectionDisabled(item.id) }]"
                   :disabled="isSelectionDisabled(item.id)"
                   @click="toggleSelection(item.id, $event)"
                   @keydown.enter.stop="toggleSelection(item.id, $event)"
                   :aria-label="'Select for comparison: ' + new Date(item.timestamp).toLocaleString()"
                   :aria-pressed="isSelected(item.id)"
                   role="checkbox">
              <svg v-if="isSelected(item.id)" viewBox="0 0 16 16" fill="none" class="check-icon">
                <path d="M3.5 8.5L6.5 11.5L12.5 4.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
            </button>
            <span class="history-mode-badge" :class="item.mode">
              {{ item.mode === 'manual' ? 'Evaluate' : 'Design' }}
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
              {{ item.temperature }}&deg;C | {{ item.processing }}
            </div>
          </div>
        </div>
      </div>
    </div>
  </transition>
</template>

<style scoped>
.history-panel { padding: var(--space-xl); margin-bottom: var(--space-xl); max-height: 400px; overflow-y: auto; display: flex; flex-direction: column; }
.history-panel-sticky { position: sticky; top: 0; z-index: 1; background: var(--bg-card); padding-bottom: var(--space-sm); }
.history-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--space-lg); padding-bottom: var(--space-md); border-bottom: 1px solid var(--border-subtle); }
.history-header h3 { margin: 0; font-size: var(--font-size-lg); color: var(--text-primary); }
.history-actions { display: flex; gap: var(--space-sm); }
.small-btn { padding: var(--space-xs) var(--space-md); background: var(--bg-glass); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); color: var(--text-primary); cursor: pointer; transition: all var(--transition-base); font-size: var(--font-size-sm); font-family: var(--font-family); }
.small-btn:hover:not(:disabled) { background: var(--bg-elevated); transform: translateY(-1px); }
.small-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.small-btn.danger:hover:not(:disabled) { background: rgba(239, 71, 111, 0.2); border-color: var(--danger); color: var(--danger); }
.empty-history { text-align: center; padding: var(--space-xl); color: var(--text-muted); }
.history-list { display: flex; flex-direction: column; gap: var(--space-md); }
.history-item { background: var(--bg-glass); padding: var(--space-md); border-radius: var(--radius-md); border: 1px solid var(--border-subtle); cursor: pointer; transition: all var(--transition-base); }
.history-item:hover { background: var(--bg-elevated); border-color: var(--border-strong); transform: translateX(4px); }
.history-item.item-selected { background: rgba(0, 212, 255, 0.08); border-color: rgba(0, 212, 255, 0.3); }
.history-item-header { display: flex; align-items: center; gap: var(--space-sm); margin-bottom: var(--space-sm); }
.history-mode-badge { padding: var(--space-xs) var(--space-sm); border-radius: var(--radius-sm); font-size: 0.75rem; font-weight: var(--font-weight-semibold); }
.history-mode-badge.manual { background: rgba(99, 102, 241, 0.2); color: var(--primary); }
.history-mode-badge.auto { background: rgba(236, 72, 153, 0.2); color: var(--secondary); }
.history-timestamp { font-size: 0.7rem; color: var(--text-muted); margin-left: auto; }
.history-item-details { display: flex; flex-direction: column; gap: var(--space-sm); }
.history-comp-preview { display: flex; flex-wrap: wrap; gap: var(--space-xs); }
.mini-tag { background: var(--bg-input); padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-family: monospace; color: var(--text-secondary); }
.history-conditions { font-size: 0.75rem; color: var(--text-muted); font-style: italic; }

/* Comparison */
.selection-count {
  font-size: var(--font-size-sm); color: var(--primary);
  font-weight: var(--font-weight-semibold);
}
.compare-btn:not(:disabled) {
  background: rgba(0, 212, 255, 0.15); border-color: rgba(0, 212, 255, 0.3); color: var(--primary);
}
.compare-btn:not(:disabled):hover { background: rgba(0, 212, 255, 0.25); transform: translateY(-1px); }
.compare-checkbox {
  width: 20px; height: 20px; flex-shrink: 0; padding: 0;
  background: var(--bg-glass); border: 1.5px solid var(--border-strong);
  border-radius: 4px; cursor: pointer; display: flex; align-items: center; justify-content: center;
  transition: all var(--transition-base); color: transparent;
}
.compare-checkbox:hover:not(.disabled) { border-color: var(--primary); background: rgba(99, 102, 241, 0.1); }
.compare-checkbox.checked {
  background: var(--primary); border-color: var(--primary); color: white;
}
.compare-checkbox.disabled { opacity: 0.25; cursor: not-allowed; }
.check-icon { width: 12px; height: 12px; }

/* Transition */
.slide-down-enter-active, .slide-down-leave-active { transition: all 0.3s ease-out; }
.slide-down-enter-from { opacity: 0; transform: translateY(-20px); }
.slide-down-leave-to { opacity: 0; transform: translateY(-20px); }

@media (max-width: 768px) {
  .history-panel { max-height: 300px; }
  .history-header { flex-direction: column; align-items: flex-start; gap: var(--space-md); }
}
@media (max-width: 480px) {
  .history-comp-preview { flex-direction: column; }
  .mini-tag { width: 100%; }
}
</style>
