import { HUD_COPY, KEY_BINDINGS } from '../hud/copy'

const DIAGNOSTIC_KEYS = new Set(['d', 'r', 'v', 'p', 't', 'i'])

/** Shared policy for keyboard actions, their hints, and the visual-check handle. */
export function createViewerPolicy() {
  const development = import.meta.env.DEV === true
  const allowsKey = (key: string): boolean => development || !DIAGNOSTIC_KEYS.has(key.toLowerCase())
  return {
    allowsKey,
    keyBindings: KEY_BINDINGS.filter(({ key }) => allowsKey(key)),
    sourceHint: (source: 'real' | 'synthetic'): string => !development ? ''
      : source === 'real' ? HUD_COPY.dataSourceHintReal : HUD_COPY.dataSourceHintSynthetic,
    syntheticCaption: development ? HUD_COPY.cobbUnavailableSynthetic
      : 'The city comparison requires the soil model. The globe currently shows a synthetic pattern.',
    exposeDiagnostics(target: object, handle: object): void {
      if (development) Object.assign(target, { __nematode: handle })
    },
  }
}
