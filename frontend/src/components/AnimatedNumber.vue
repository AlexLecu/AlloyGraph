<script setup>
import { ref, watch, onMounted, onUnmounted } from 'vue'

const props = defineProps({
  value: { type: Number, required: true },
  decimals: { type: Number, default: 1 },
  duration: { type: Number, default: 800 }
})

const display = ref(0)
let animFrame = null

const animate = (from, to) => {
  if (animFrame) cancelAnimationFrame(animFrame)
  const start = performance.now()

  const step = (now) => {
    const progress = Math.min((now - start) / props.duration, 1)
    // ease-out cubic
    const eased = 1 - Math.pow(1 - progress, 3)
    display.value = from + (to - from) * eased
    if (progress < 1) {
      animFrame = requestAnimationFrame(step)
    }
  }
  animFrame = requestAnimationFrame(step)
}

onMounted(() => animate(0, props.value))

watch(() => props.value, (newVal, oldVal) => {
  animate(oldVal || 0, newVal)
})

onUnmounted(() => {
  if (animFrame) cancelAnimationFrame(animFrame)
})
</script>

<template>
  <span class="animated-number">{{ display.toFixed(decimals) }}</span>
</template>
