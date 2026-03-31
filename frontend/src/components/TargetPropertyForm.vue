<script setup>
const props = defineProps({
  modelValue: { type: Object, required: true },
  temperature: { type: Number, default: 20 },
  processing: { type: String, default: 'cast' },
  iterations: { type: Number, default: 3 },
  loading: { type: Boolean, default: false },
})

const emit = defineEmits(['update:modelValue', 'update:temperature', 'update:processing', 'update:iterations', 'submit'])

const updateField = (field, val) => {
  emit('update:modelValue', { ...props.modelValue, [field]: val })
}
</script>

<template>
  <div class="panel glass-card">
    <h3>Define Target Properties</h3>
    <p class="helper-text">Specify target values for desired properties. Set to 0 to skip any property.</p>

    <div class="target-grid">
      <div class="field">
        <label>Min Yield Strength <span class="default-hint">(MPa)</span></label>
        <input :value="modelValue.yield" @input="updateField('yield', Number($event.target.value))" type="number" placeholder="1200" />
      </div>
      <div class="field">
        <label>Min Tensile Strength <span class="default-hint">(MPa)</span></label>
        <input :value="modelValue.tensile" @input="updateField('tensile', Number($event.target.value))" type="number" placeholder="0" />
      </div>
      <div class="field">
        <label>Min Elongation <span class="default-hint">(%)</span></label>
        <input :value="modelValue.elongation" @input="updateField('elongation', Number($event.target.value))" type="number" step="0.1" placeholder="0" />
      </div>
      <div class="field">
        <label>Min Elastic Modulus <span class="default-hint">(GPa)</span></label>
        <input :value="modelValue.elastic_modulus" @input="updateField('elastic_modulus', Number($event.target.value))" type="number" step="1" placeholder="0" />
      </div>
      <div class="field">
        <label>Max Density <span class="default-hint">(g/cm3)</span></label>
        <input :value="modelValue.density" @input="updateField('density', Number($event.target.value))" type="number" step="0.1" placeholder="8.5" />
      </div>
      <div class="field">
        <label>Target Gamma Prime <span class="default-hint">(vol%)</span></label>
        <input :value="modelValue.gamma_prime" @input="updateField('gamma_prime', Number($event.target.value))" type="number" step="0.1" placeholder="0 = not specified" />
      </div>
    </div>

    <!-- Bottom Controls -->
    <div class="design-controls">
      <div class="control-row">
        <div class="inline-control">
          <label>Temperature:</label>
<input type="number" :value="temperature" @input="$emit('update:temperature', Number($event.target.value))" class="small-input" aria-label="Temperature in Celsius" />
          <span class="unit">°C</span>
        </div>

        <div class="inline-control">
          <label>Processing:</label>
          <select :value="processing" @change="$emit('update:processing', $event.target.value)" class="small-select" aria-label="Processing type">
            <option value="cast">Cast</option>
            <option value="wrought">Wrought</option>
          </select>
        </div>

        <div class="inline-control">
          <label>Max Iterations:</label>
          <input type="number" :value="iterations" @input="$emit('update:iterations', Number($event.target.value))" min="1" max="10" class="small-input" style="width: 60px" aria-label="Maximum iterations" />
        </div>
      </div>

      <button @click="$emit('submit')" :disabled="loading" class="primary-btn magic-btn">
        {{ loading ? 'Inventing Alloy...' : 'Auto-Design Alloy' }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.panel { background: var(--bg-panel); padding: 1.5rem; border-radius: 8px; }
.helper-text { color: var(--text-secondary); font-size: var(--font-size-sm); margin-bottom: var(--space-md); line-height: 1.5; }
.default-hint { color: var(--text-muted); font-size: var(--font-size-xs); font-weight: var(--font-weight-normal); font-style: italic; }

.target-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 10px; margin-bottom: 1.5rem; }
.field { background: var(--bg-input); padding: 8px; border-radius: 6px; border: 1px solid var(--border-subtle); position: relative; }
.field label { display: block; font-size: 0.8rem; color: var(--text-muted); margin-bottom: 4px; }
.field input { width: 100%; background: transparent; border: none; color: var(--text-primary); font-size: 1.1rem; font-weight: bold; }

.design-controls { margin-top: var(--space-xl); padding-top: var(--space-lg); border-top: 1px solid var(--border-subtle); display: flex; flex-direction: column; gap: var(--space-md); }
.control-row { display: flex; align-items: center; gap: var(--space-lg); flex-wrap: wrap; }
.inline-control { display: flex; align-items: center; gap: var(--space-sm); font-size: var(--font-size-sm); color: var(--text-secondary); }
.inline-control .unit { color: var(--text-muted); font-size: var(--font-size-sm); }
.small-input, .small-select { width: 100px; background: var(--bg-glass); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); color: var(--text-primary); padding: var(--space-xs) var(--space-sm); font-size: var(--font-size-sm); }


.primary-btn { background: var(--primary); color: white; border: none; padding: 12px 24px; border-radius: 6px; font-size: 1rem; font-weight: bold; cursor: pointer; transition: transform 0.1s; }
.primary-btn:hover { background: var(--primary-dark); transform: scale(1.02); }
.primary-btn:disabled { background: var(--bg-elevated); color: var(--text-muted); cursor: not-allowed; }
.magic-btn { background: linear-gradient(135deg, #6610f2, #d63384); }

@media (max-width: 768px) {
  .panel { padding: 1rem; }
  .target-grid { grid-template-columns: repeat(auto-fill, minmax(80px, 1fr)); }
  .control-row { flex-direction: column; align-items: stretch; }
  .inline-control { justify-content: space-between; }
}

@media (max-width: 480px) {
  .panel { padding: 0.75rem; }
  .target-grid { grid-template-columns: repeat(auto-fill, minmax(70px, 1fr)); gap: 6px; margin-bottom: 1rem; }
  .field { padding: 6px; }
  .field input { font-size: 0.95rem; }
  .field label { font-size: 0.7rem; }
  .primary-btn { padding: 10px 18px; font-size: 0.9rem; }
  .design-controls { margin-top: var(--space-md); padding-top: var(--space-md); }
}
</style>
