import { ref } from 'vue'

const STORAGE_KEY = 'alloygraph-tour-completed'
const tourCompleted = ref(localStorage.getItem(STORAGE_KEY) === 'true')

let driverInstance = null

function getTourSteps(switchToTab) {
  return [
    {
      element: '[data-tour="research-tab"]',
      popover: {
        title: 'Research Chat',
        description: 'Ask questions about nickel-based superalloys, search for similar alloys, or explore the Knowledge Graph.',
        side: 'bottom',
        align: 'center',
      },
    },
    {
      element: '[data-tour="design-tab"]',
      popover: {
        title: 'Evaluate & Design',
        description: 'Predict properties for known compositions or let the AI design a new alloy. Click Next to explore this tab.',
        side: 'bottom',
        align: 'center',
      },
    },
    {
      element: '[data-tour="evaluate-btn"]',
      popover: {
        title: 'Evaluate Mode',
        description: 'Enter a known alloy composition and get ML-predicted properties validated against physics constraints and the Knowledge Graph.',
        side: 'bottom',
        align: 'center',
      },
      onHighlightStarted: () => {
        if (switchToTab) switchToTab('design')
      },
    },
    {
      element: '[data-tour="design-btn"]',
      popover: {
        title: 'Design Mode',
        description: 'Set target properties (yield strength, density, etc.) and let the multi-agent AI invent a new alloy composition to meet them.',
        side: 'bottom',
        align: 'center',
      },
    },
    {
      element: '[data-tour="history-toggle"]',
      popover: {
        title: 'History & Compare',
        description: 'View past evaluations and designs. Select 2-3 runs and click Compare for a side-by-side breakdown of compositions and properties.',
        side: 'bottom',
        align: 'center',
      },
    },
    {
      element: '[data-tour="theme-toggle"]',
      popover: {
        title: 'Theme Toggle',
        description: 'Switch between dark and light mode. Your preference is saved automatically.',
        side: 'bottom',
        align: 'start',
      },
    },
    {
      element: '[data-tour="info-button"]',
      popover: {
        title: 'Help & Information',
        description: 'Opens a detailed guide with physics validation, confidence levels, and known limitations. You can also restart this tour from here.',
        side: 'left',
        align: 'center',
      },
    },
  ]
}

export function useTour() {
  const startTour = async (options = {}) => {
    const { driver } = await import('driver.js')
    await import('driver.js/dist/driver.css')

    if (driverInstance) {
      driverInstance.destroy()
      driverInstance = null
    }

    driverInstance = driver({
      showProgress: true,
      animate: true,
      allowClose: true,
      overlayColor: 'rgba(0, 0, 0, 0.7)',
      stagePadding: 8,
      stageRadius: 12,
      popoverClass: 'alloygraph-tour-popover',
      onDestroyStarted: () => {
        tourCompleted.value = true
        localStorage.setItem(STORAGE_KEY, 'true')
        const inst = driverInstance
        driverInstance = null
        if (inst) inst.destroy()
      },
      steps: getTourSteps(options.switchToTab),
    })

    driverInstance.drive()
  }

  return { tourCompleted, startTour }
}
