import { sweepGuideAt, sweepGuidePresentationAt, type GuideState, type SweepPhase } from '../sweep/guide'

const CAPTIONS: Record<SweepPhase, { title: string; description: string }> = {
  ready: { title: 'Before the sweep', description: 'Air, water and rock still cover the globe.' },
  atmosphere: { title: 'Atmosphere fading', description: 'Watch the blue rim at the edge of the globe.' },
  water: { title: 'Water receding', description: 'Ocean basins come into view.' },
  rock: { title: 'Rock dissolving', description: 'The soil nematode map comes into view.' },
  remains: { title: 'What remains', description: 'The veil maps soil nematodes on land. Marine populations lie outside the model.' },
}

/** Stable-height, scrubbable guide. Live announcements change by phase, not frame. */
export function createSweepGuide(): { element: HTMLElement; update(state: GuideState, realData: boolean, showIntro: boolean): void } {
  const element = document.createElement('section')
  element.className = 'sweep-guide'
  element.setAttribute('aria-label', 'What is being swept away')
  const intro = document.createElement('div')
  intro.className = 'sweep-intro'
  intro.hidden = true
  const premise = document.createElement('p')
  premise.className = 'sweep-intro-premise'
  const quote = document.createElement('blockquote')
  quote.className = 'sweep-intro-quote'
  const source = document.createElement('a')
  source.className = 'sweep-intro-source'
  source.target = '_blank'
  source.rel = 'noopener noreferrer'
  source.title = 'Read Cobb’s original passage, USDA Yearbook 1914, page 472'
  intro.append(premise, quote, source)
  const captions = document.createElement('div')
  captions.className = 'sweep-captions'
  captions.setAttribute('aria-hidden', 'true')
  const captionElements = Object.entries(CAPTIONS).map(([phase, copy]) => {
    const caption = document.createElement('div')
    caption.className = 'sweep-caption'
    caption.dataset.phase = phase
    const title = document.createElement('div')
    title.className = 'sweep-caption-title'
    title.textContent = copy.title
    const description = document.createElement('div')
    description.className = 'sweep-caption-description'
    description.textContent = copy.description
    caption.append(title, description)
    captions.append(caption)
    return { phase, caption, description }
  })
  const announcer = document.createElement('p')
  announcer.className = 'sweep-sr-only'
  announcer.setAttribute('role', 'status')
  announcer.setAttribute('aria-live', 'polite')
  announcer.setAttribute('aria-atomic', 'true')
  const markers = document.createElement('ol')
  markers.className = 'sweep-layers'
  markers.setAttribute('role', 'list')
  markers.setAttribute('aria-label', 'Layer status')
  const markerElements = ['Atmosphere', 'Water', 'Solid Earth'].map((name) => {
    const marker = document.createElement('li')
    const symbol = document.createElement('span')
    symbol.className = 'sweep-layer-symbol'
    symbol.setAttribute('aria-hidden', 'true')
    const statusText = document.createElement('span')
    statusText.className = 'sweep-sr-only'
    marker.append(symbol, name, statusText)
    markers.append(marker)
    return { marker, symbol, statusText }
  })
  element.append(intro, captions, announcer, markers)
  let previousKey = ''
  let previousAnnouncement = ''
  return {
    element,
    update(state, realData, showIntro) {
      const guide = sweepGuideAt(state)
      const presentation = sweepGuidePresentationAt(state, showIntro)
      const key = `${presentation.mode}:${guide.phase}:${guide.layers.map((layer) => layer.status).join(',')}:${realData}`
      if (key === previousKey) return
      previousKey = key
      element.dataset.phase = guide.phase
      const presentingIntro = presentation.mode === 'intro'
      intro.hidden = !presentingIntro
      captions.hidden = presentingIntro
      markers.hidden = presentingIntro
      element.setAttribute('aria-label', presentingIntro ? 'The thought experiment' : 'What is being swept away')
      if (presentingIntro) {
        premise.textContent = presentation.premise
        quote.textContent = `“${presentation.excerpt}”`
        source.textContent = presentation.attribution
        source.href = presentation.sourceHref
      }
      const remainsDescription = realData ? CAPTIONS.remains.description : 'The veil shows a synthetic test pattern.'
      for (const item of captionElements) {
        item.caption.dataset.active = String(item.phase === guide.phase)
        if (item.phase === 'remains') item.description.textContent = remainsDescription
      }
      guide.layers.forEach((layer, i) => {
        const { marker, symbol, statusText } = markerElements[i]!
        marker.dataset.status = layer.status
        const status = layer.status === 'leaving' ? 'being swept away' : layer.status
        statusText.textContent = `: ${status}`
        marker.title = `${layer.name}: ${status}`
        symbol.textContent = layer.status === 'gone' ? '✓' : layer.status === 'leaving' ? '◌' : '○'
      })
      const copy = CAPTIONS[guide.phase]
      // The visible excerpt is already in the reading order. Announce only
      // subsequent phase changes; repeating the introduction reads twice.
      const announcement = presentation.announce
        ? `${copy.title}. ${guide.phase === 'remains' ? remainsDescription : copy.description}`
        : ''
      if (announcement !== previousAnnouncement) {
        announcer.textContent = announcement
        previousAnnouncement = announcement
      }
    },
  }
}
