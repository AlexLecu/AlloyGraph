<script setup>
import { useToast } from '../composables/useToast'

const { toasts, removeToast } = useToast()
</script>

<template>
  <div class="toast-container" aria-live="polite">
    <TransitionGroup name="toast">
      <div
        v-for="toast in toasts"
        :key="toast.id"
        :class="['toast-item', 'toast-' + toast.type]"
        @click="removeToast(toast.id)"
        role="status"
      >
        <span class="toast-icon">
          {{ toast.type === 'success' ? '\u2713' : toast.type === 'error' ? '\u2717' : toast.type === 'warning' ? '!' : 'i' }}
        </span>
        <span class="toast-message">{{ toast.message }}</span>
      </div>
    </TransitionGroup>
  </div>
</template>

<style scoped>
.toast-container {
  position: fixed;
  bottom: 24px;
  right: 24px;
  z-index: 9999;
  display: flex;
  flex-direction: column-reverse;
  gap: 8px;
  pointer-events: none;
}

.toast-item {
  pointer-events: auto;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 20px;
  border-radius: var(--radius-md, 8px);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border-subtle);
  cursor: pointer;
  min-width: 220px;
  max-width: 380px;
  font-size: 0.9rem;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}

.toast-icon {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  font-weight: 700;
  flex-shrink: 0;
}

.toast-message {
  color: var(--text-primary, #fff);
  line-height: 1.4;
}

.toast-success {
  background: rgba(6, 214, 160, 0.15);
  border-color: rgba(6, 214, 160, 0.3);
}
.toast-success .toast-icon {
  background: rgba(6, 214, 160, 0.3);
  color: #06d6a0;
}

.toast-error {
  background: rgba(239, 71, 111, 0.15);
  border-color: rgba(239, 71, 111, 0.3);
}
.toast-error .toast-icon {
  background: rgba(239, 71, 111, 0.3);
  color: #ef476f;
}

.toast-warning {
  background: rgba(255, 214, 10, 0.15);
  border-color: rgba(255, 214, 10, 0.3);
}
.toast-warning .toast-icon {
  background: rgba(255, 214, 10, 0.3);
  color: #ffd60a;
}

.toast-info {
  background: rgba(0, 212, 255, 0.15);
  border-color: rgba(0, 212, 255, 0.3);
}
.toast-info .toast-icon {
  background: rgba(0, 212, 255, 0.3);
  color: #00d4ff;
}

.toast-enter-active {
  transition: all 0.3s ease-out;
}
.toast-leave-active {
  transition: all 0.2s ease-in;
}
.toast-enter-from {
  opacity: 0;
  transform: translateX(60px);
}
.toast-leave-to {
  opacity: 0;
  transform: translateX(60px);
}
</style>
