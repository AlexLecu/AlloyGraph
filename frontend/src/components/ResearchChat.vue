<script setup>
defineOptions({ name: 'ResearchChat' })

import { ref, nextTick, onMounted, onUnmounted } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'

import { API_BASE_URL } from '../config'
import AlloyCard from './AlloyCard.vue'

// ── Timestamp formatting ────────────────────────────────────────────────
const formatTimestamp = (ts) => {
  if (!ts) return ''
  const diff = Math.floor((Date.now() - ts) / 1000)
  if (diff < 10) return 'just now'
  if (diff < 60) return `${diff}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

// Refresh timestamps every 30s
let timestampInterval = null

let msgIdCounter = 0
const nextMsgId = () => ++msgIdCounter

// Generate unique session ID
const generateSessionId = () => {
  return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9)
}

const CHAT_KEY = 'alloygraph-chat-history'

const sessionId = ref(generateSessionId())
const messages = ref([
  { id: nextMsgId(), role: 'system', text: 'Ask me anything about alloys - compositions, properties, comparisons, or recommendations.', timestamp: Date.now() }
])

// ── Chat history persistence ────────────────────────────────────────────
const saveHistory = () => {
  try {
    const toSave = messages.value.map(m => ({
      role: m.role, text: m.text, display: m.display,
      timestamp: m.timestamp, error: m.error, retryPrompt: m.retryPrompt
    }))
    localStorage.setItem(CHAT_KEY, JSON.stringify({ sessionId: sessionId.value, messages: toSave }))
  } catch { /* quota exceeded */ }
}

const restoreHistory = () => {
  try {
    const stored = localStorage.getItem(CHAT_KEY)
    if (!stored) return false
    const { sessionId: sid, messages: msgs } = JSON.parse(stored)
    if (!msgs || msgs.length <= 1) return false
    sessionId.value = sid
    messages.value = msgs
    return true
  } catch { return false }
}
const input = ref('')
const loading = ref(false)
const messagesContainer = ref(null)
const inputField = ref(null)
const emit = defineEmits(['design'])

// Copy button state
const copiedIndex = ref(null)

// Scroll-to-bottom state
const showScrollBtn = ref(false)
const timestampTick = ref(0)

// Track which alloys have been shown (to show card only on first mention)
const shownAlloys = ref(new Set())

// Active request controller — allows cancellation
let abortController = null
let requestGeneration = 0  // guards finally block against stale cleanup

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

const autoResize = () => {
  const el = inputField.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 120) + 'px'
}

// Scroll-to-bottom detection
const onMessagesScroll = () => {
  const el = messagesContainer.value
  if (!el) return
  const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
  showScrollBtn.value = distFromBottom > 100
}

const scrollToBottomClick = () => {
  if (messagesContainer.value) {
    messagesContainer.value.scrollTo({
      top: messagesContainer.value.scrollHeight,
      behavior: 'smooth'
    })
  }
}

// Copy message text
const copyMessage = async (msg, index) => {
  try {
    await navigator.clipboard.writeText(msg.text || msg.display || '')
    copiedIndex.value = index
    setTimeout(() => { copiedIndex.value = null }, 1500)
  } catch {
    // Fallback — ignore
  }
}

// Lifecycle
onMounted(() => {
  timestampInterval = setInterval(() => { timestampTick.value++ }, 30000)
  if (restoreHistory()) {
    nextTick(() => scrollToBottom(true))
  }
})

onUnmounted(() => {
  if (timestampInterval) clearInterval(timestampInterval)
})

// Filter alloys to only show ones not yet shown in conversation
const getNewAlloys = (alloys) => {
  if (!alloys) return []
  const newAlloys = alloys.filter(a => !shownAlloys.value.has(a.name))
  newAlloys.forEach(a => shownAlloys.value.add(a.name))
  return newAlloys
}

// ── Markdown rendering (marked + DOMPurify) ────────────────────────────
marked.setOptions({
  breaks: true,
  gfm: true,
})

const formatText = (text) => {
  if (!text) return ''
  const raw = marked.parse(text)
  return DOMPurify.sanitize(raw)
}

// ── Send / Cancel ───────────────────────────────────────────────────────

const cancelRequest = () => {
  if (abortController) {
    abortController.abort()
    abortController = null
  }
}

const sendMessage = async () => {
  if (!input.value.trim()) return

  // Cancel any in-flight request
  cancelRequest()

  const prompt = input.value
  messages.value.push({ id: nextMsgId(), role: 'user', text: prompt, timestamp: Date.now() })
  input.value = ''
  nextTick(() => autoResize())
  loading.value = true

  let assistantMsg = null
  let pendingAlloys = []
  scrollToBottom(true)

  abortController = new AbortController()
  const thisGeneration = ++requestGeneration

  try {
    const history = messages.value
      .slice(0, -1)
      .filter(m => m.role !== 'system')
      .map(m => ({
        role: m.role,
        content: m.text
      }))

    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, sessionId: sessionId.value, history }),
      signal: abortController.signal
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

          if (!assistantMsg && (chunk.type === 'data' || chunk.type === 'chunk' || chunk.type === 'error')) {
            assistantMsg = {
              id: nextMsgId(),
              role: 'assistant',
              display: '',
              text: '',
              alloys: [],
              toolSuggestion: null,
              timestamp: Date.now()
            }
            messages.value.push(assistantMsg)
          }

          if (chunk.type === 'data') {
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

    scrollToBottom(true)
    saveHistory()
  } catch (error) {
    if (error.name === 'AbortError') {
      if (assistantMsg) {
        assistantMsg.display += '\n*(cancelled)*'
        assistantMsg.text = ''
      }
    } else {
      // If no assistant message exists yet (e.g., network error or HTTP 500
      // before any NDJSON chunk arrived), create one so the user sees the error
      if (!assistantMsg) {
        assistantMsg = { id: nextMsgId(), role: 'assistant', display: '', text: '', alloys: [], toolSuggestion: null, timestamp: Date.now() }
        messages.value.push(assistantMsg)
      }
      assistantMsg.display = ''
      assistantMsg.error = true
      assistantMsg.retryPrompt = prompt
      assistantMsg.text = error.message
    }
    saveHistory()
    scrollToBottom(true)
  } finally {
    // Only reset state if this is still the active request
    // (prevents a cancelled request's cleanup from stomping a newer request)
    if (thisGeneration === requestGeneration) {
      abortController = null
      loading.value = false
      focusInput()
    }
  }
}

const clearChat = () => {
  cancelRequest()
  sessionId.value = generateSessionId()
  messages.value = [
    { id: nextMsgId(), role: 'system', text: 'New conversation! Ask me anything about alloys.', timestamp: Date.now() }
  ]
  shownAlloys.value.clear()
  localStorage.removeItem(CHAT_KEY)
  scrollToBottom()
  focusInput()
}

// Categorized suggestion prompts
const suggestionGroups = [
  {
    label: 'Search',
    icon: '🔍',
    items: ['What is Inconel 718?', 'Find alloys similar to Haynes 282']
  },
  {
    label: 'Properties',
    icon: '⚡',
    items: ['Which alloy has the highest yield strength?', 'Compare Inconel 718 and Haynes 282']
  },
  {
    label: 'Learn',
    icon: '📚',
    items: ['How does γ\' strengthening work?', 'What makes CMSX-4 a good single crystal alloy?']
  }
]

const retryMessage = (msg) => {
  // Remove the failed message, then resend
  const idx = messages.value.indexOf(msg)
  if (idx > -1) messages.value.splice(idx, 1)
  input.value = msg.retryPrompt
  sendMessage()
}

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
    <div class="messages-area" ref="messagesContainer" @scroll="onMessagesScroll">
      <div
        v-for="(msg, i) in messages"
        :key="msg.id || i"
        :class="['msg', msg.role]"
        v-show="msg.display || msg.text || msg.error || (msg.alloys && msg.alloys.length)"
      >
        <!-- Avatar -->
        <div class="avatar">
          <span v-if="msg.role === 'user'">👤</span>
          <span v-else>🤖</span>
        </div>

        <!-- Content -->
        <div class="content">
          <!-- Error state with retry -->
          <div v-if="msg.error" class="error-bubble">
            <div class="error-icon">!</div>
            <div class="error-body">
              <span class="error-text">Failed to get response: {{ msg.text }}</span>
              <button class="retry-btn" @click="retryMessage(msg)">Retry</button>
            </div>
          </div>

          <div class="text" v-else-if="msg.display || msg.text" v-html="formatText(msg.display || msg.text)"></div>

          <!-- Copy button (assistant only) -->
          <button
            v-if="msg.role === 'assistant' && !msg.error && (msg.text || msg.display)"
            class="copy-btn"
            @click="copyMessage(msg, i)"
            :title="copiedIndex === i ? 'Copied!' : 'Copy message'"
          >
            <span v-if="copiedIndex === i">&#10003; Copied</span>
            <span v-else>&#128203; Copy</span>
          </button>

          <!-- Tool Suggestion (e.g. "Use Designer") -->
          <button
            v-if="msg.toolSuggestion && msg.toolSuggestion.tool === 'designer'"
            class="tool-suggestion-btn"
            @click="$emit('design', {})"
          >
            Open Alloy Designer
          </button>

          <!-- Alloy Cards -->
          <div v-if="msg.alloys && msg.alloys.length" class="cards">
            <AlloyCard
              v-for="alloy in msg.alloys"
              :key="alloy.name"
              :alloy="alloy"
              @design="$emit('design', $event)"
            />
          </div>

          <!-- Timestamp -->
          <span v-if="msg.timestamp" class="msg-timestamp" :key="timestampTick">
            {{ formatTimestamp(msg.timestamp) }}
          </span>
        </div>
      </div>

      <!-- Skeleton Loader (with cancel) -->
      <div v-if="loading" class="msg assistant">
        <div class="avatar"><span>🤖</span></div>
        <div class="content">
          <div class="skeleton-block">
            <div class="skeleton-line" style="width: 85%"></div>
            <div class="skeleton-line" style="width: 65%"></div>
            <div class="skeleton-line" style="width: 45%"></div>
          </div>
          <button class="cancel-btn" @click="cancelRequest" title="Cancel request">Stop</button>
        </div>
      </div>

      <!-- Empty state with categorized suggestions -->
      <div v-if="messages.length === 1 && !loading" class="empty-state">
        <div class="empty-state-title">How can I help you today?</div>
        <div class="suggestion-groups">
          <div v-for="group in suggestionGroups" :key="group.label" class="suggestion-group">
            <div class="group-label"><span class="group-icon">{{ group.icon }}</span> {{ group.label }}</div>
            <button v-for="s in group.items" :key="s" @click="useSuggestion(s)" class="chip">
              {{ s }}
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Scroll-to-bottom button -->
    <transition name="scroll-btn">
      <button v-if="showScrollBtn" class="scroll-to-bottom" @click="scrollToBottomClick" title="Scroll to bottom">
        ↓
      </button>
    </transition>

    <!-- Input Area -->
    <div class="input-area">
      <textarea
        ref="inputField"
        v-model="input"
        @keydown.enter.exact.prevent="sendMessage"
        @input="autoResize"
        :disabled="loading"
        placeholder="Ask about alloys..."
        rows="1"
      ></textarea>
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
  position: relative;
  background: var(--bg-card);
  backdrop-filter: blur(20px);
  border: 1px solid var(--border-subtle);
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
  background: var(--bg-glass);
  border-bottom: 1px solid var(--border-subtle);
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
  border: 1px solid var(--border-subtle);
  background: var(--bg-glass);
  color: var(--text-muted);
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.icon-btn:hover {
  background: var(--bg-elevated);
  color: var(--text-primary);
  border-color: var(--border-strong);
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
  background: var(--bg-glass);
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
  background: var(--bg-glass);
  border: 1px solid var(--border-subtle);
  border-radius: 12px 12px 12px 4px;
}

.msg.user .text {
  background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
  color: white;
  border-radius: 12px 12px 4px 12px;
}

.text :deep(p) {
  margin: 0.25rem 0;
}

.text :deep(p:first-child) {
  margin-top: 0;
}

.text :deep(p:last-child) {
  margin-bottom: 0;
}

.text :deep(strong) {
  color: var(--primary-light);
  font-weight: 600;
}

.text :deep(em) {
  font-style: italic;
  opacity: 0.85;
}

.text :deep(ul),
.text :deep(ol) {
  padding-left: 1.25rem;
  margin: 0.35rem 0;
}

.text :deep(li) {
  margin: 0.2rem 0;
}

.text :deep(ul > li) {
  list-style: disc;
}

.text :deep(ul > li)::marker {
  color: var(--primary);
}

.text :deep(ol > li)::marker {
  color: var(--primary);
  font-weight: 600;
}

.text :deep(h1),
.text :deep(h2),
.text :deep(h3),
.text :deep(h4) {
  margin: 0.5rem 0 0.25rem;
  font-weight: 600;
  color: var(--primary-light);
}

.text :deep(h1) { font-size: 1.1rem; }
.text :deep(h2) { font-size: 1.05rem; }
.text :deep(h3) { font-size: 1rem; }
.text :deep(h4) { font-size: 0.95rem; }

.text :deep(table) {
  border-collapse: collapse;
  width: 100%;
  margin: 0.5rem 0;
  font-size: 0.85rem;
}

.text :deep(th),
.text :deep(td) {
  border: 1px solid var(--border-subtle);
  padding: 0.35rem 0.6rem;
  text-align: left;
}

.text :deep(th) {
  background: var(--bg-glass);
  font-weight: 600;
  color: var(--primary-light);
}

.text :deep(blockquote) {
  border-left: 3px solid var(--primary);
  padding-left: 0.75rem;
  margin: 0.5rem 0;
  color: var(--text-secondary);
}

.text :deep(a) {
  color: var(--primary-light);
  text-decoration: underline;
}

.text :deep(pre) {
  background: var(--bg-input);
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  padding: 0.75rem;
  overflow-x: auto;
  margin: 0.5rem 0;
}

.text :deep(code) {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 0.85em;
}

.text :deep(:not(pre) > code) {
  background: var(--bg-glass);
  padding: 0.1em 0.35em;
  border-radius: 4px;
}

/* Error Bubble with Retry */
.error-bubble {
  display: flex;
  align-items: flex-start;
  gap: 0.6rem;
  padding: 0.75rem 1rem;
  background: rgba(239, 68, 68, 0.08);
  border: 1px solid rgba(239, 68, 68, 0.25);
  border-radius: 12px 12px 12px 4px;
}

.error-icon {
  width: 1.5rem;
  height: 1.5rem;
  border-radius: 50%;
  background: rgba(239, 68, 68, 0.2);
  color: #f87171;
  font-size: 0.85rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.error-body {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.error-text {
  font-size: 0.85rem;
  color: #fca5a5;
  line-height: 1.4;
}

.retry-btn {
  background: rgba(239, 68, 68, 0.15);
  border: 1px solid rgba(239, 68, 68, 0.3);
  color: #f87171;
  font-size: 0.8rem;
  font-weight: 500;
  padding: 0.3rem 0.85rem;
  border-radius: 6px;
  cursor: pointer;
  width: fit-content;
  transition: all 0.2s;
}

.retry-btn:hover {
  background: rgba(239, 68, 68, 0.25);
  border-color: rgba(239, 68, 68, 0.5);
  transform: translateY(-1px);
}

.tool-suggestion-btn {
  background: linear-gradient(135deg, var(--secondary) 0%, #8b5cf6 100%);
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  font-size: 0.85rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  width: fit-content;
}

.tool-suggestion-btn:hover {
  transform: translateY(-1px);
  filter: brightness(1.1);
  box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
}

.cards {
  width: 100%;
}

/* Skeleton Loader */
.skeleton-block {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 0.75rem 1rem;
  background: var(--bg-glass);
  border: 1px solid var(--border-subtle);
  border-radius: 12px 12px 12px 4px;
  min-width: 200px;
}

.skeleton-line {
  height: 0.75rem;
  border-radius: 4px;
  background: linear-gradient(90deg, var(--border-subtle) 25%, var(--bg-elevated) 50%, var(--border-subtle) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite ease-in-out;
}

@keyframes shimmer {
  0% { background-position: 200% center; }
  100% { background-position: -200% center; }
}

/* Copy Button */
.copy-btn {
  background: none;
  border: 1px solid var(--border-subtle);
  color: var(--text-muted);
  font-size: 0.7rem;
  padding: 0.15rem 0.5rem;
  border-radius: 6px;
  cursor: pointer;
  width: fit-content;
  transition: all 0.2s;
  opacity: 0;
}

.msg.assistant:hover .copy-btn {
  opacity: 1;
}

.copy-btn:hover {
  background: var(--bg-glass);
  border-color: var(--border-strong);
  color: var(--text-secondary);
}

/* Timestamp */
.msg-timestamp {
  font-size: 0.65rem;
  color: var(--text-muted);
  opacity: 0;
  transition: opacity 0.2s;
}

.msg:hover .msg-timestamp {
  opacity: 1;
}

/* Cancel button */
.cancel-btn {
  background: rgba(239, 68, 68, 0.15);
  border: 1px solid rgba(239, 68, 68, 0.3);
  color: #f87171;
  font-size: 0.75rem;
  padding: 0.25rem 0.75rem;
  border-radius: 6px;
  cursor: pointer;
  width: fit-content;
  transition: all 0.2s;
}

.cancel-btn:hover {
  background: rgba(239, 68, 68, 0.25);
  border-color: rgba(239, 68, 68, 0.5);
}

/* Empty State */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 1.5rem 1rem;
  gap: 1.25rem;
}

.empty-state-title {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text-secondary);
}

.suggestion-groups {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  width: 100%;
  max-width: 480px;
}

.suggestion-group {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.group-label {
  font-size: 0.7rem;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding-left: 0.25rem;
}

.group-icon {
  margin-right: 0.25rem;
}

.chip {
  padding: 0.5rem 0.75rem;
  background: var(--bg-glass);
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  color: var(--text-primary);
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.2s;
  text-align: left;
}

.chip:hover {
  background: var(--bg-elevated);
  border-color: var(--primary);
  color: var(--primary-light);
  transform: translateX(4px);
}

/* Scroll-to-bottom Button */
.scroll-to-bottom {
  position: absolute;
  bottom: 5.5rem;
  left: 50%;
  transform: translateX(-50%);
  width: 2.25rem;
  height: 2.25rem;
  border-radius: 50%;
  border: 1px solid var(--border-strong);
  background: var(--bg-card);
  color: var(--text-primary);
  font-size: 1rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: var(--shadow-md);
  transition: all 0.2s;
  z-index: 10;
}

.scroll-to-bottom:hover {
  background: var(--bg-elevated);
  border-color: var(--primary);
  color: var(--primary);
  box-shadow: var(--shadow-lg);
}

.scroll-btn-enter-active,
.scroll-btn-leave-active {
  transition: opacity 0.2s, transform 0.2s;
}

.scroll-btn-enter-from,
.scroll-btn-leave-to {
  opacity: 0;
  transform: translateX(-50%) translateY(8px);
}

/* Input Area */
.input-area {
  display: flex;
  gap: 0.75rem;
  padding: 1rem 1.25rem;
  background: var(--bg-glass);
  border-top: 1px solid var(--border-subtle);
}

.input-area textarea {
  flex: 1;
  padding: 0.75rem 1rem;
  border-radius: 12px;
  border: 1px solid var(--border-subtle);
  background: var(--bg-input);
  color: var(--text-primary);
  font-size: 0.9rem;
  font-family: var(--font-family);
  line-height: 1.5;
  resize: none;
  overflow-y: auto;
  min-height: 2.75rem;
  max-height: 120px;
  transition: border-color 0.2s;
}

.input-area textarea:focus {
  outline: none;
  border-color: var(--primary);
  background: var(--bg-input);
}

.input-area textarea::placeholder {
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
  background: var(--border-subtle);
  border-radius: 3px;
}

.messages-area::-webkit-scrollbar-thumb:hover {
  background: var(--border-strong);
}

/* === RESPONSIVE: TABLET (≤768px) === */
@media (max-width: 768px) {
  .chat-panel {
    height: calc(100vh - 120px);
    border-radius: 12px;
  }

  .header {
    padding: 0.75rem 1rem;
  }

  .logo {
    width: 2rem;
    height: 2rem;
    font-size: 1.2rem;
  }

  .messages-area {
    padding: 0.75rem;
  }

  .empty-state {
    padding: 1rem 0.75rem;
  }

  .input-area {
    padding: 0.75rem 1rem;
  }

  .scroll-to-bottom {
    bottom: 4.5rem;
  }
}

/* === RESPONSIVE: PHONE (≤480px) === */
@media (max-width: 480px) {
  .chat-panel {
    height: calc(100dvh - 80px);
    border-radius: 8px;
  }

  .header {
    padding: 0.5rem 0.75rem;
  }

  .header-text h2 {
    font-size: 0.9rem;
  }

  .subtitle {
    display: none;
  }

  .messages-area {
    padding: 0.5rem;
    gap: 0.75rem;
  }

  .content {
    max-width: 92%;
  }

  .text {
    padding: 0.6rem 0.75rem;
    font-size: 0.85rem;
  }

  .empty-state {
    padding: 0.75rem 0.5rem;
    gap: 0.75rem;
  }

  .empty-state-title {
    font-size: 0.95rem;
  }

  .chip {
    padding: 0.4rem 0.6rem;
    font-size: 0.75rem;
  }

  .input-area {
    padding: 0.5rem 0.75rem;
    gap: 0.5rem;
  }

  .input-area textarea {
    padding: 0.5rem 0.75rem;
    font-size: 0.85rem;
    min-height: 2.5rem;
  }

  .send-btn {
    width: 2.5rem;
    height: 2.5rem;
    font-size: 1.1rem;
  }

  .scroll-to-bottom {
    bottom: 4rem;
    width: 2rem;
    height: 2rem;
    font-size: 0.85rem;
  }

  .skeleton-block {
    min-width: 150px;
  }
}
</style>
