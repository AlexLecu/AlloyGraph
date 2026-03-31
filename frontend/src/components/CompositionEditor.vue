<script setup>
import { ref, computed, onMounted } from 'vue'
import { useToast } from '../composables/useToast'

const { showToast } = useToast()


const props = defineProps({
  modelValue: { type: Object, required: true },
  temperature: { type: Number, default: 20 },
  processing: { type: String, default: 'cast' },
  loading: { type: Boolean, default: false },
})

const emit = defineEmits(['update:modelValue', 'update:temperature', 'update:processing', 'submit'])

// --- COMPOSITION SUM VALIDATION ---
const compositionSum = computed(() => {
  return Object.values(props.modelValue).reduce((sum, v) => sum + (Number(v) || 0), 0)
})

const sumStatus = computed(() => {
  const s = compositionSum.value
  if (s < 50 || s > 150) return 'error'
  if (s < 95 || s > 105) return 'warning'
  return 'ok'
})

const elementCount = computed(() => Object.keys(props.modelValue).length)

const canSubmit = computed(() => {
  return elementCount.value >= 2 && sumStatus.value !== 'error' && !props.loading
})

// --- PRESETS ---
const BUILTIN_PRESETS = {
  "Waspaloy": { composition: {"Ni": 58.0, "Cr": 19.5, "Co": 13.5, "Mo": 4.3, "Al": 1.3, "Ti": 3.0, "C": 0.08, "B": 0.006, "Zr": 0.06}, processing: "wrought", builtin: true },
  "Inconel 718": { composition: { "Ni": 52.5, "Cr": 19.0, "Fe": 19.0, "Nb": 5.1, "Mo": 3.0, "Ti": 0.9, "Al": 0.5 }, processing: "wrought", builtin: true },
  "Udimet 720": { composition: { "Ni": 55.0, "Cr": 16.0, "Co": 14.7, "Ti": 5.0, "Al": 2.5, "Mo": 3.0, "W": 1.25 }, processing: "wrought", builtin: true },
  "IN738LC": { composition: {"Ni": 61.5, "Cr": 16.0, "Co": 8.5, "Mo": 1.75, "W": 2.6, "Al": 3.4, "Ti": 3.4, "Ta": 1.75, "Nb": 0.9, "C": 0.11, "B": 0.01, "Zr": 0.05}, processing: "cast", builtin: true },
  "Udimet 500": { composition: { "Ni": 54.0, "Cr": 18.0, "Co": 18.5, "Mo": 4.0, "Al": 2.9, "Ti": 2.9, "C": 0.08, "B": 0.006, "Zr": 0.05 }, processing: "wrought", builtin: true },
  "Haynes 282": { composition: { "Ni": 57.0, "Cr": 19.5, "Co": 10.0, "Mo": 8.5, "Ti": 2.1, "Al": 1.5, "Fe": 1.0, "Mn": 0.15, "Si": 0.1, "C": 0.06, "B": 0.005 }, processing: "wrought", builtin: true },
  "CMSX-4": { composition: {"Ni": 61.7, "Cr": 6.5, "Co": 9.0, "Mo": 0.6, "W": 6.0, "Al": 5.6, "Ti": 1.0, "Ta": 6.5, "Re": 3.0, "Hf": 0.1}, processing: "cast", builtin: true },
  "Rene 65": { composition: {"Ni": 51.6, "Cr": 16.0, "Co": 13.0, "Mo": 4.0, "W": 4.0, "Al": 2.1, "Ti": 3.7, "Nb": 0.7, "Fe": 1.0, "B": 0.016, "Zr": 0.05, "C": 0.01}, processing: "wrought", builtin: true }
}

const customPresets = ref({})
const selectedPreset = ref(null)

const loadCustomPresets = () => {
  try {
    const stored = localStorage.getItem('alloyCustomPresets')
    if (stored) customPresets.value = JSON.parse(stored)
  } catch (e) { console.error('Failed to load custom presets:', e) }
}

const saveCustomPresets = () => {
  try {
    localStorage.setItem('alloyCustomPresets', JSON.stringify(customPresets.value))
  } catch (e) { console.error('Failed to save custom presets:', e) }
}

const allPresets = computed(() => ({ ...BUILTIN_PRESETS, ...customPresets.value }))

const loadPreset = (name) => {
  const preset = allPresets.value[name]
  if (!preset) return
  emit('update:modelValue', { ...preset.composition })
  emit('update:processing', preset.processing)
  selectedPreset.value = name
}

// --- SAVE / DELETE PRESET ---
const showSavePreset = ref(false)
const newPresetName = ref('')

const openSavePreset = () => { newPresetName.value = ''; showSavePreset.value = true }

const saveAsPreset = () => {
  const name = newPresetName.value.trim()
  if (!name) return
  if (BUILTIN_PRESETS[name]) { alert('Cannot overwrite built-in presets. Choose a different name.'); return }
  customPresets.value[name] = { composition: { ...props.modelValue }, processing: props.processing, builtin: false }
  saveCustomPresets()
  selectedPreset.value = name
  showSavePreset.value = false
  showToast(`Preset "${name}" saved`, 'success')
}

const deletePreset = (name) => {
  if (BUILTIN_PRESETS[name]) { alert('Cannot delete built-in presets.'); return }
  if (confirm(`Delete preset "${name}"?`)) {
    delete customPresets.value[name]
    saveCustomPresets()
    if (selectedPreset.value === name) selectedPreset.value = null
    showToast(`Preset "${name}" deleted`, 'info')
  }
}

// --- JSON IMPORT ---
const showJsonImport = ref(false)
const jsonInput = ref('')
const jsonError = ref('')

const openJsonImport = () => { jsonInput.value = ''; jsonError.value = ''; showJsonImport.value = true }
const closeJsonImport = () => { showJsonImport.value = false; jsonInput.value = ''; jsonError.value = '' }

const importJsonComposition = () => {
  jsonError.value = ''
  if (!jsonInput.value.trim()) { jsonError.value = 'Please enter a JSON composition'; return }
  try {
    let parsed = JSON.parse(jsonInput.value.trim())
    if (parsed.composition && typeof parsed.composition === 'object') parsed = parsed.composition
    if (typeof parsed !== 'object' || Array.isArray(parsed)) {
      jsonError.value = 'Invalid format: Expected an object like {"Ni": 60, "Cr": 20}'; return
    }
    const cleaned = {}
    for (const [key, value] of Object.entries(parsed)) {
      const numVal = parseFloat(value)
      if (!isNaN(numVal) && numVal > 0) {
        const element = key.charAt(0).toUpperCase() + key.slice(1).toLowerCase()
        cleaned[element] = Math.round(numVal * 1000) / 1000
      }
    }
    if (Object.keys(cleaned).length === 0) {
      jsonError.value = 'No valid elements found. Use format: {"Ni": 60, "Cr": 20}'; return
    }
    emit('update:modelValue', cleaned)
    closeJsonImport()
    showToast('Composition imported', 'success')
  } catch (e) { jsonError.value = `Invalid JSON: ${e.message}` }
}

// --- ELEMENT ADD/REMOVE ---
const addElement = (el) => {
  if (el && !props.modelValue[el]) {
    emit('update:modelValue', { ...props.modelValue, [el]: 0.0 })
  }
}

const removeElement = (el) => {
  const copy = { ...props.modelValue }
  delete copy[el]
  emit('update:modelValue', copy)
}

const updateElement = (el, val) => {
  emit('update:modelValue', { ...props.modelValue, [el]: val })
}

const clearComposition = () => {
  emit('update:modelValue', {})
  selectedPreset.value = null
}

// --- FOCUS TRAP FOR MODALS ---
const trapFocus = (e, modalClass) => {
  const modal = document.querySelector(modalClass)
  if (!modal) return
  const focusable = modal.querySelectorAll('button, input, textarea, select, [tabindex]:not([tabindex="-1"])')
  if (focusable.length === 0) return
  const first = focusable[0]
  const last = focusable[focusable.length - 1]
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus() }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus() }
}

onMounted(() => { loadCustomPresets() })
</script>

<template>
  <div class="panel glass-card">
    <h3>Define Alloy Composition</h3>
    <p class="helper-text">Enter element percentages (should sum to ~100%). Use presets or add custom elements.</p>

    <!-- Presets Section -->
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
                    :aria-label="'Delete preset ' + name"
                    title="Delete preset">&times;</button>
          </div>
        </template>
        <span class="preset-divider">|</span>
        <button @click="openSavePreset" class="preset-btn action-btn" title="Save current composition as preset">
          Save Preset
        </button>
        <button @click="openJsonImport" class="preset-btn action-btn" title="Import composition from JSON">
          Import JSON
        </button>
        <button @click="clearComposition" class="preset-btn action-btn clear-btn" title="Clear all elements">
          Clear
        </button>
      </div>
    </div>

    <!-- Composition Grid -->
    <div class="section-label">Composition (wt%):</div>
    <div class="comp-grid">
      <div v-for="(val, el) in modelValue" :key="el" class="element-box">
        <label>{{ el }}</label>
        <input type="number" :value="val" @input="updateElement(el, Number($event.target.value))" step="0.1" />
        <span class="remove" @click="removeElement(el)" role="button" :aria-label="'Remove ' + el">&times;</span>
      </div>
      <div class="add-box">
        <input type="text" placeholder="+ Add Element" @keyup.enter="addElement($event.target.value); $event.target.value=''" class="add-input" aria-label="Add element" />
      </div>
    </div>

    <!-- Composition Sum Validation -->
    <div :class="['comp-sum-bar', sumStatus]" v-if="elementCount > 0">
      <span class="sum-label">Sum:</span>
      <span class="sum-value">{{ compositionSum.toFixed(1) }}%</span>
      <span v-if="sumStatus === 'warning'" class="sum-hint">(should be ~100%)</span>
      <span v-if="sumStatus === 'error'" class="sum-hint">(must be between 50-150%)</span>
    </div>

    <!-- Bottom Actions -->
    <div class="eval-controls">
      <div class="temp-inline">
        <label>Temperature:</label>
<input type="number" :value="temperature" @input="$emit('update:temperature', Number($event.target.value))" class="temp-simple" aria-label="Temperature in Celsius" />
        <span class="temp-unit">°C</span>

        <label style="margin-left: var(--space-lg);">Processing:</label>
        <select :value="processing" @change="$emit('update:processing', $event.target.value)" class="temp-simple" style="width: 100px" aria-label="Processing type">
          <option value="cast">Cast</option>
          <option value="wrought">Wrought</option>
        </select>
      </div>

      <button @click="$emit('submit')" :disabled="!canSubmit" class="primary-btn"
              :title="!canSubmit ? (elementCount < 2 ? 'Need at least 2 elements' : 'Fix composition sum') : ''">
        {{ loading ? 'Running Models...' : 'Predict Properties' }}
      </button>
    </div>
  </div>

  <!-- JSON Import Modal -->
  <transition name="modal-fade">
    <div v-if="showJsonImport" class="modal-overlay" @click.self="closeJsonImport" @keydown.tab="trapFocus($event, '.json-import-modal')" @keydown.escape="closeJsonImport">
      <div class="json-import-modal" role="dialog" aria-label="Import composition from JSON">
        <div class="modal-header">
          <h3>Import Composition from JSON</h3>
          <button class="modal-close" @click="closeJsonImport" aria-label="Close">&times;</button>
        </div>
        <div class="modal-body">
          <p class="modal-help">Paste a JSON object with element symbols and weight percentages.</p>
          <div class="json-examples">
            <span class="example-label">Examples:</span>
            <code>{"Ni": 60, "Cr": 20, "Al": 5}</code>
            <code>{"composition": {"Ni": 58, "Co": 13}}</code>
          </div>
          <textarea v-model="jsonInput" class="json-textarea"
            placeholder='{"Ni": 60, "Cr": 20, "Co": 10, "Al": 5, "Ti": 3}'
            rows="6" @keydown.ctrl.enter="importJsonComposition" @keydown.meta.enter="importJsonComposition"></textarea>
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
    <div v-if="showSavePreset" class="modal-overlay" @click.self="showSavePreset = false" @keydown.tab="trapFocus($event, '.save-preset-modal')" @keydown.escape="showSavePreset = false">
      <div class="save-preset-modal" role="dialog" aria-label="Save preset">
        <div class="modal-header">
          <h3>Save as Preset</h3>
          <button class="modal-close" @click="showSavePreset = false" aria-label="Close">&times;</button>
        </div>
        <div class="modal-body">
          <p class="modal-help">Save the current composition as a custom preset for quick access later.</p>
          <div class="preset-name-input">
            <label for="preset-name">Preset Name:</label>
            <input id="preset-name" type="text" v-model="newPresetName" placeholder="My Custom Alloy" @keydown.enter="saveAsPreset" autofocus />
          </div>
          <div class="preset-preview">
            <span class="preview-label">Composition:</span>
            <span class="preview-elements">
              {{ Object.entries(modelValue).map(([el, val]) => `${el}: ${val}%`).join(', ') }}
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
.panel { background: var(--bg-panel); padding: 1.5rem; border-radius: 8px; }
.section-label { font-size: var(--font-size-sm); font-weight: var(--font-weight-semibold); color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: var(--space-sm); display: block; }
.helper-text { color: var(--text-secondary); font-size: var(--font-size-sm); margin-bottom: var(--space-md); line-height: 1.5; }

/* Presets */
.presets-section { margin-bottom: var(--space-lg); padding: var(--space-md); background: var(--bg-glass); border-radius: var(--radius-md); border: 1px solid var(--border-subtle); }
.preset-buttons { display: flex; gap: var(--space-sm); flex-wrap: wrap; margin-top: var(--space-sm); }
.preset-btn { padding: var(--space-sm) var(--space-md); background: var(--bg-glass); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); color: var(--text-primary); cursor: pointer; transition: all var(--transition-base); font-size: var(--font-size-sm); font-weight: var(--font-weight-medium); }
.preset-btn:hover { background: var(--bg-elevated); border-color: var(--border-strong); transform: translateY(-1px); }
.preset-btn.preset-selected { background: rgba(0, 212, 255, 0.2); border-color: #00d4ff; color: #00d4ff; }
.preset-btn.action-btn { background: var(--bg-glass); border: 1px solid var(--border-subtle); }
.preset-btn.action-btn:hover { background: var(--bg-elevated); border-color: var(--border-strong); box-shadow: none; }
.preset-item { position: relative; display: inline-flex; align-items: center; }
.preset-delete-btn { position: absolute; top: -6px; right: -6px; width: 18px; height: 18px; padding: 0; background: #ff4757; border: none; border-radius: 50%; color: white; font-size: 12px; font-weight: bold; cursor: pointer; opacity: 0; transition: opacity 0.2s; display: flex; align-items: center; justify-content: center; }
.preset-item:hover .preset-delete-btn { opacity: 1; }
.preset-delete-btn:hover { background: #ff2f4a; transform: scale(1.1); }
.preset-btn.custom-preset { background: rgba(255, 215, 0, 0.1); border-color: rgba(255, 215, 0, 0.3); }
.preset-btn.custom-preset:hover { background: rgba(255, 215, 0, 0.2); border-color: rgba(255, 215, 0, 0.5); }
.preset-btn.custom-preset.preset-selected { background: rgba(255, 215, 0, 0.3); border-color: #ffd700; color: #ffd700; }
.preset-divider { color: var(--text-muted); margin: 0 var(--space-sm); opacity: 0.5; }
.clear-btn:hover { background: rgba(255, 80, 80, 0.15); border-color: rgba(255, 80, 80, 0.4); box-shadow: none; }

/* Composition Grid */
.comp-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 10px; margin-bottom: 0.75rem; }
.element-box { background: var(--bg-input); padding: 8px; border-radius: 6px; border: 1px solid var(--border-subtle); position: relative; }
.element-box label { display: block; font-size: 0.8rem; color: var(--text-muted); margin-bottom: 4px; }
.element-box input { width: 100%; background: transparent; border: none; color: var(--text-primary); font-size: 1.1rem; font-weight: bold; }
.remove { position: absolute; top: 2px; right: 6px; color: var(--text-muted); cursor: pointer; }
.remove:hover { color: var(--danger); }
.add-input { width: 100%; height: 100%; background: var(--bg-input); border: 1px dashed var(--text-muted); color: var(--text-primary); text-align: center; border-radius: 6px; }

/* Composition Sum Validation */
.comp-sum-bar { display: flex; align-items: center; gap: var(--space-sm); padding: var(--space-xs) var(--space-md); border-radius: var(--radius-md); font-size: var(--font-size-sm); margin-bottom: var(--space-md); }
.comp-sum-bar.ok { background: rgba(6, 214, 160, 0.1); border: 1px solid rgba(6, 214, 160, 0.2); }
.comp-sum-bar.ok .sum-value { color: var(--success); font-weight: 600; }
.comp-sum-bar.warning { background: rgba(255, 214, 10, 0.1); border: 1px solid rgba(255, 214, 10, 0.25); }
.comp-sum-bar.warning .sum-value { color: #ffd60a; font-weight: 600; }
.comp-sum-bar.error { background: rgba(239, 71, 111, 0.1); border: 1px solid rgba(239, 71, 111, 0.25); }
.comp-sum-bar.error .sum-value { color: var(--danger); font-weight: 600; }
.sum-label { color: var(--text-muted); }
.sum-hint { color: var(--text-muted); font-size: var(--font-size-xs); font-style: italic; }


/* Controls */
.eval-controls { display: flex; align-items: center; justify-content: space-between; gap: var(--space-lg); margin-top: var(--space-xl); padding-top: var(--space-lg); border-top: 1px solid var(--border-subtle); }
.temp-inline { display: flex; align-items: center; gap: var(--space-sm); font-size: var(--font-size-sm); color: var(--text-secondary); flex-wrap: wrap; }
.temp-simple { width: 80px; background: var(--bg-glass); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); color: var(--text-primary); padding: var(--space-xs) var(--space-sm); font-size: var(--font-size-sm); }
.temp-inline .temp-unit { color: var(--text-muted); font-size: var(--font-size-sm); }
.primary-btn { background: var(--primary); color: white; border: none; padding: 12px 24px; border-radius: 6px; font-size: 1rem; font-weight: bold; cursor: pointer; transition: transform 0.1s; }
.primary-btn:hover { background: var(--primary-dark); transform: scale(1.02); }
.primary-btn:disabled { background: var(--bg-elevated); color: var(--text-muted); cursor: not-allowed; }

/* Modals */
.modal-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0, 0, 0, 0.7); display: flex; align-items: center; justify-content: center; z-index: 1000; backdrop-filter: blur(4px); }
.json-import-modal { background: var(--bg-secondary); border-radius: var(--radius-lg); width: 90%; max-width: 550px; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5); border: 1px solid var(--border-subtle); }
.save-preset-modal { background: var(--bg-secondary); border-radius: var(--radius-lg); width: 90%; max-width: 400px; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5); border: 1px solid var(--border-subtle); }
.modal-header { display: flex; justify-content: space-between; align-items: center; padding: var(--space-lg); border-bottom: 1px solid var(--border-subtle); }
.modal-header h3 { margin: 0; font-size: 1.1rem; color: var(--text-primary); }
.modal-close { background: none; border: none; color: var(--text-muted); font-size: 1.5rem; cursor: pointer; padding: 0; line-height: 1; transition: color 0.2s; }
.modal-close:hover { color: var(--text-primary); }
.modal-body { padding: var(--space-lg); }
.modal-help { color: var(--text-secondary); margin-bottom: var(--space-md); font-size: 0.9rem; }
.json-examples { display: flex; flex-wrap: wrap; gap: var(--space-sm); align-items: center; margin-bottom: var(--space-md); padding: var(--space-sm); background: rgba(0, 0, 0, 0.2); border-radius: var(--radius-sm); }
.example-label { color: var(--text-muted); font-size: 0.8rem; }
.json-examples code { background: rgba(17, 153, 250, 0.15); padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; color: var(--primary); }
.json-textarea { width: 100%; background: var(--bg-primary); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); color: var(--text-primary); padding: var(--space-md); font-family: 'Monaco', 'Menlo', 'Consolas', monospace; font-size: 0.9rem; resize: vertical; min-height: 120px; }
.json-textarea:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 2px rgba(17, 153, 250, 0.2); }
.json-textarea::placeholder { color: var(--text-muted); opacity: 0.6; }
.json-error { color: #ff6b6b; font-size: 0.85rem; margin-top: var(--space-sm); padding: var(--space-sm); background: rgba(255, 80, 80, 0.1); border-radius: var(--radius-sm); }
.modal-footer { display: flex; justify-content: flex-end; gap: var(--space-md); padding: var(--space-lg); border-top: 1px solid var(--border-subtle); }
.cancel-btn { background: transparent; border: 1px solid var(--border-subtle); color: var(--text-secondary); padding: var(--space-sm) var(--space-lg); border-radius: var(--radius-md); cursor: pointer; transition: all 0.2s; }
.cancel-btn:hover { background: rgba(255, 255, 255, 0.05); color: var(--text-primary); }
.import-btn { background: var(--primary); border: none; color: white; padding: var(--space-sm) var(--space-lg); border-radius: var(--radius-md); cursor: pointer; font-weight: 500; transition: all 0.2s; }
.import-btn:hover { background: var(--primary-dark, #0d8ae0); transform: translateY(-1px); }
.import-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.preset-name-input { margin-bottom: var(--space-md); }
.preset-name-input label { display: block; margin-bottom: var(--space-xs); color: var(--text-secondary); font-size: var(--font-size-sm); }
.preset-name-input input { width: 100%; padding: var(--space-sm) var(--space-md); background: var(--bg-glass); border: 1px solid var(--border-strong); border-radius: var(--radius-md); color: var(--text-primary); font-size: var(--font-size-md); }
.preset-name-input input:focus { outline: none; border-color: var(--primary); }
.preset-preview { padding: var(--space-sm); background: var(--bg-glass); border-radius: var(--radius-sm); font-size: var(--font-size-sm); }
.preset-preview .preview-label { color: var(--text-muted); margin-right: var(--space-xs); }
.preset-preview .preview-elements { color: var(--text-secondary); }

/* Modal transitions */
.modal-fade-enter-active, .modal-fade-leave-active { transition: opacity 0.2s ease; }
.modal-fade-enter-active .json-import-modal, .modal-fade-leave-active .json-import-modal,
.modal-fade-enter-active .save-preset-modal, .modal-fade-leave-active .save-preset-modal { transition: transform 0.2s ease; }
.modal-fade-enter-from, .modal-fade-leave-to { opacity: 0; }

/* Responsive */
@media (max-width: 768px) {
  .panel { padding: 1rem; }
  .comp-grid { grid-template-columns: repeat(auto-fill, minmax(80px, 1fr)); }
  .eval-controls { flex-direction: column; align-items: stretch; }
  .temp-inline { flex-wrap: wrap; justify-content: center; }
  .preset-buttons { flex-direction: column; }
  .preset-btn { width: 100%; }
  .json-import-modal { max-width: 90vw; }
  .save-preset-modal { max-width: 90vw; }
}

@media (max-width: 480px) {
  .panel { padding: 0.75rem; }
  .comp-grid { grid-template-columns: repeat(auto-fill, minmax(70px, 1fr)); gap: 6px; }
  .element-box { padding: 6px; }
  .element-box input { font-size: 0.95rem; }
  .element-box label { font-size: 0.7rem; }
  .presets-section { padding: var(--space-sm); }
  .preset-btn { padding: var(--space-xs) var(--space-sm); font-size: var(--font-size-xs); }
  .primary-btn { padding: 10px 18px; font-size: 0.9rem; }
  .modal-body { padding: var(--space-md); }
  .modal-header { padding: var(--space-md); }
  .modal-footer { padding: var(--space-md); }
  .json-textarea { min-height: 80px; font-size: 0.8rem; }
}
</style>
