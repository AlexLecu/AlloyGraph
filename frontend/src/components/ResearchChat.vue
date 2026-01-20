<script setup>
import { ref, nextTick } from 'vue'

import { API_BASE_URL } from '../config'
import AlloyCard from './AlloyCard.vue'

// Generate unique session ID
const generateSessionId = () => {
  return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9)
}

const sessionId = ref(generateSessionId())
const messages = ref([
  { role: 'system', text: 'Ask me anything about alloys - compositions, properties, comparisons, or recommendations.' }
])
const input = ref('')
const loading = ref(false)
const messagesContainer = ref(null)
const inputField = ref(null)
const emit = defineEmits(['design'])

// Track which alloys have been shown (to show card only on first mention)
const shownAlloys = ref(new Set())

// UI State
const isExpanded = ref(false)
const toggleExpand = () => {
  isExpanded.value = !isExpanded.value
  scrollToBottom()
}

// Throttled scroll - only scrolls every 100ms max during streaming
let lastScrollTime = 0
const scrollToBottom = async (force = false) => {
  const now = Date.now()
  if (!force && now - lastScrollTime < 100) return
  lastScrollTime = now

  await nextTick()
  if (messagesContainer.value) {
    messagesContainer.value.scrollTo({
      top: messagesContainer.value.scrollHeight,
      behavior: 'smooth'
    })
  }
}

const focusInput = async () => {
  await nextTick()
  if (inputField.value) {
    inputField.value.focus()
  }
}

// Filter alloys to only show ones not yet shown in conversation
const getNewAlloys = (alloys) => {
  if (!alloys) return []
  const newAlloys = alloys.filter(a => !shownAlloys.value.has(a.name))
  newAlloys.forEach(a => shownAlloys.value.add(a.name))
  return newAlloys
}

// Simple markdown-like formatting
const formatText = (text) => {
  if (!text) return ''
  return text
    // Bold: **text** or __text__
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/__(.*?)__/g, '<strong>$1</strong>')
    // Lists: - item or * item
    .replace(/^[\-\*]\s+(.*)$/gm, '<li>$1</li>')
    // Line breaks
    .replace(/\n/g, '<br>')
}

const sendMessage = async () => {
  if (!input.value.trim()) return

  const prompt = input.value
  messages.value.push({ role: 'user', text: prompt })
  input.value = ''
  loading.value = true

  let assistantMsg = null
  let pendingAlloys = []  // Store alloys until streaming completes
  scrollToBottom(true)  // Force scroll after user message

  try {
    const history = messages.value
      .slice(0, -1)
      .map(m => ({
        role: m.role === 'system' ? 'assistant' : m.role,
        content: m.text
      }))

    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, sessionId: sessionId.value, history })
    })

    if (!response.ok) throw new Error(response.statusText)

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop()

      for (const line of lines) {
        if (!line.trim()) continue
        try {
          const chunk = JSON.parse(line)

          if (!assistantMsg && (chunk.type === 'data' || chunk.type === 'chunk')) {
            assistantMsg = {
              role: 'assistant',
              display: '',
              text: '',
              alloys: [],
              toolSuggestion: null
            }
            messages.value.push(assistantMsg)
          }

          if (chunk.type === 'data') {
            // Store alloys for later - don't render during streaming
            pendingAlloys = getNewAlloys(chunk.alloys)
          } else if (chunk.type === 'chunk' || chunk.type === 'text_chunk' || chunk.type === 'string_chunk') {
            assistantMsg.display += chunk.content
            assistantMsg.text += chunk.content
            scrollToBottom()
          } else if (chunk.type === 'tool_suggestion') {
            assistantMsg.toolSuggestion = { tool: chunk.tool, message: chunk.message }
          } else if (chunk.type === 'error') {
            assistantMsg.display += `\n[Error: ${chunk.content}]`
          }
        } catch (e) {
          // Skip malformed chunks
        }
      }
    }

    // Show alloy cards after streaming completes
    if (assistantMsg && pendingAlloys.length > 0) {
      assistantMsg.alloys = pendingAlloys
    }

    scrollToBottom(true)  // Force scroll at end
  } catch (error) {
    if (assistantMsg) assistantMsg.display += `\n[Error: ${error.message}]`
    scrollToBottom(true)
  } finally {
    loading.value = false
    focusInput()
  }
}

const clearChat = () => {
  sessionId.value = generateSessionId()
  messages.value = [
    { role: 'system', text: 'New conversation! Ask me anything about alloys.' }
  ]
  shownAlloys.value.clear()
  scrollToBottom()
  focusInput()
}

// Quick suggestion prompts
const suggestions = [
  'What is Inconel 718?',
  'Find an alloy with ~500 MPa yield strength',
  'Which alloy has the highest yield strength?'
]

const useSuggestion = (text) => {
  input.value = text
  sendMessage()
}
</script>

<template>
  <div :class="['chat-panel', { expanded: isExpanded }]">
    <!-- Header -->
    <header class="header">
      <div class="header-left">
        <div class="logo">🔬</div>
        <div class="header-text">
          <h2>Alloy Research</h2>
          <span class="subtitle">Knowledge Graph Assistant</span>
        </div>
      </div>
      <div class="header-actions">
        <button @click="clearChat" class="icon-btn" title="New conversation">
          <span>↻</span>
        </button>
        <button @click="toggleExpand" class="icon-btn" :title="isExpanded ? 'Collapse' : 'Expand'">
          <span>{{ isExpanded ? '⤡' : '⤢' }}</span>
        </button>
      </div>
    </header>

    <!-- Messages Area -->
    <div class="messages-area" ref="messagesContainer">
      <div
        v-for="(msg, i) in messages"
        :key="i"
        :class="['msg', msg.role]"
        v-show="msg.display || msg.text || (msg.alloys && msg.alloys.length)"
      >
        <!-- Avatar -->
        <div class="avatar">
          <span v-if="msg.role === 'user'">👤</span>
          <span v-else>🤖</span>
        </div>

        <!-- Content -->
        <div class="content">
          <div class="text" v-if="msg.display || msg.text" v-html="formatText(msg.display || msg.text)"></div>

          <!-- Alloy Cards -->
          <div v-if="msg.alloys && msg.alloys.length" class="cards">
            <AlloyCard
              v-for="alloy in msg.alloys"
              :key="alloy.name"
              :alloy="alloy"
              @design="$emit('design', $event)"
            />
          </div>
        </div>
      </div>

      <!-- Loading Indicator -->
      <div v-if="loading" class="msg assistant">
        <div class="avatar"><span>🤖</span></div>
        <div class="content">
          <div class="typing">
            <span></span><span></span><span></span>
          </div>
        </div>
      </div>

      <!-- Empty state with suggestions -->
      <div v-if="messages.length === 1 && !loading" class="suggestions">
        <p>Try asking:</p>
        <div class="chips">
          <button v-for="s in suggestions" :key="s" @click="useSuggestion(s)" class="chip">
            {{ s }}
          </button>
        </div>
      </div>
    </div>

    <!-- Input Area -->
    <div class="input-area">
      <input
        ref="inputField"
        v-model="input"
        @keyup.enter="sendMessage"
        :disabled="loading"
        placeholder="Ask about alloys..."
      />
      <button @click="sendMessage" :disabled="loading || !input.trim()" class="send-btn">
        <span v-if="loading">...</span>
        <span v-else>→</span>
      </button>
    </div>
  </div>
</template>

<style scoped>
.chat-panel {
  display: flex;
  flex-direction: column;
  height: 70vh;
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 16px;
  overflow: hidden;
  transition: all 0.3s ease;
}

.chat-panel.expanded {
  position: fixed;
  inset: 20px;
  height: auto;
  z-index: 1000;
  border-color: var(--primary);
  box-shadow: 0 25px 80px rgba(0, 0, 0, 0.5);
}

/* Header */
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 1.25rem;
  background: rgba(0, 0, 0, 0.2);
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.header-left {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.logo {
  font-size: 1.5rem;
  width: 2.5rem;
  height: 2.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(99, 102, 241, 0.15);
  border-radius: 10px;
}

.header-text h2 {
  margin: 0;
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.subtitle {
  font-size: 0.7rem;
  color: var(--text-muted);
  letter-spacing: 0.02em;
}

.header-actions {
  display: flex;
  gap: 0.5rem;
}

.icon-btn {
  width: 2rem;
  height: 2rem;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(255, 255, 255, 0.05);
  color: var(--text-muted);
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.icon-btn:hover {
  background: rgba(255, 255, 255, 0.1);
  color: var(--text-primary);
  border-color: rgba(255, 255, 255, 0.2);
}

/* Messages Area */
.messages-area {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.msg {
  display: flex;
  gap: 0.75rem;
  animation: slideIn 0.3s ease;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.msg.user {
  flex-direction: row-reverse;
}

.msg.user .content {
  align-items: flex-end;
}

.avatar {
  width: 2rem;
  height: 2rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1rem;
  flex-shrink: 0;
  background: rgba(255, 255, 255, 0.05);
}

.msg.user .avatar {
  background: rgba(99, 102, 241, 0.2);
}

.content {
  display: flex;
  flex-direction: column;
  max-width: 85%;
  gap: 0.5rem;
}

.text {
  padding: 0.75rem 1rem;
  border-radius: 12px;
  font-size: 0.9rem;
  line-height: 1.5;
  color: var(--text-primary);
}

.msg.assistant .text,
.msg.system .text {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 12px 12px 12px 4px;
}

.msg.user .text {
  background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
  color: white;
  border-radius: 12px 12px 4px 12px;
}

.text :deep(strong) {
  color: var(--primary-light);
  font-weight: 600;
}

.text :deep(li) {
  list-style: none;
  padding-left: 1rem;
  position: relative;
  margin: 0.25rem 0;
}

.text :deep(li)::before {
  content: '•';
  position: absolute;
  left: 0;
  color: var(--primary);
}

.cards {
  width: 100%;
}

/* Typing Indicator */
.typing {
  display: flex;
  gap: 4px;
  padding: 0.75rem 1rem;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 12px;
  width: fit-content;
}

.typing span {
  width: 8px;
  height: 8px;
  background: var(--primary);
  border-radius: 50%;
  animation: bounce 1.4s infinite;
}

.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
  0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
  30% { transform: translateY(-6px); opacity: 1; }
}

/* Suggestions */
.suggestions {
  text-align: center;
  padding: 2rem 1rem;
}

.suggestions p {
  font-size: 0.8rem;
  color: var(--text-muted);
  margin-bottom: 0.75rem;
}

.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  justify-content: center;
}

.chip {
  padding: 0.5rem 0.75rem;
  background: rgba(99, 102, 241, 0.1);
  border: 1px solid rgba(99, 102, 241, 0.2);
  border-radius: 20px;
  color: var(--primary-light);
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.2s;
}

.chip:hover {
  background: rgba(99, 102, 241, 0.2);
  border-color: var(--primary);
  transform: translateY(-1px);
}

/* Input Area */
.input-area {
  display: flex;
  gap: 0.75rem;
  padding: 1rem 1.25rem;
  background: rgba(0, 0, 0, 0.2);
  border-top: 1px solid rgba(255, 255, 255, 0.06);
}

.input-area input {
  flex: 1;
  padding: 0.75rem 1rem;
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(0, 0, 0, 0.3);
  color: var(--text-primary);
  font-size: 0.9rem;
  transition: all 0.2s;
}

.input-area input:focus {
  outline: none;
  border-color: var(--primary);
  background: rgba(0, 0, 0, 0.4);
}

.input-area input::placeholder {
  color: var(--text-muted);
}

.send-btn {
  width: 2.75rem;
  height: 2.75rem;
  border-radius: 12px;
  border: none;
  background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
  color: white;
  font-size: 1.25rem;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.send-btn:hover:not(:disabled) {
  transform: scale(1.05);
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
}

.send-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Scrollbar */
.messages-area::-webkit-scrollbar {
  width: 6px;
}

.messages-area::-webkit-scrollbar-track {
  background: transparent;
}

.messages-area::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
}

.messages-area::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.2);
}
</style>
