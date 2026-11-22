/**
 * Every sentence the HUD puts on screen, in one place, so it can be tested.
 *
 * The HUD used to build its copy inline at the point of use, which meant the
 * only way to check a claim was to read the running page — and the claims are
 * the load-bearing part of this piece. Three defects came out of that:
 *
 *  1. A stat line with NESTED parentheses that no lay reader could parse:
 *     `cobb inversion data (our own measurement: 243,294 native-resolution
 *     pixels · 20 cities · a = -0.0085 (95% CI -0.035, 0.019) · R² = -0.0003)`.
 *  2. A slider end labelled `data (measured: no effect)` — a FINDING written
 *     in the grammar of a disabled control.
 *  3. Eight key bindings on one run-on line: measured as the largest single
 *     ink block in the HUD, and the least important content in it.
 *
 * `test/hudCopy.test.ts` pins all three: parenthesis nesting depth, the
 * absence of on/off vocabulary in the slider labels, the presence of every
 * measured figure (including the unflattering ones), and the size of the
 * always-visible hint list.
 */

/**
 * Deepest parenthesis nesting in `s`. Unbalanced closers do not push the
 * depth negative, so `"a) b (c)"` is depth 1 and the balance check that
 * catches it is a separate assertion.
 */
export function maxParenDepth(s: string): number {
  let depth = 0
  let max = 0
  for (const ch of s) {
    if (ch === '(') { depth++; if (depth > max) max = depth }
    else if (ch === ')' && depth > 0) depth--
  }
  return max
}

/**
 * The finding, in three tiers rendered at three type weights.
 *
 * A lay reader stops after `findingResult` and has the whole point; a
 * statistician reads `findingStatistics` and can falsify it. No tier needs a
 * parenthesis, because the tiers do the work the parentheses were doing.
 */
export const HUD_COPY = {
  /** Small wordmark above the quotation. The old title also carried "— Regime 1 shell, interactive", which is a build note, not a title; it moved to the diagnostics panel. */
  wordmark: 'Nematode Earth',
  /** Attribution for the typed-in quotation. It was previously unattributed on screen — the viewer could not tell whose claim was about to be tested. */
  quotationAttribution: 'Nathan A. Cobb, 1914',

  /** Label for the count-in-view readout. */
  countLabel: 'estimated soil nematodes in view',

  findingPrediction: 'Cobb imagined that concentrations of certain nematodes would mark the sites of towns.',
  findingResult: 'Across 20 cities, our curve for built-up land fit the soil map no better than the average mapped value.',
  findingStatistics:
    '243,294 map pixels · R² = −0.0003. The linear coefficient was −0.0085. Its nominal 95% pixel-bootstrap interval, −0.035 to +0.019, includes zero.',

  /** Left end of the Cobb axis. Grammatically parallel to the right end, so neither reads as the "off" position. */
  cobbAxisMeasured: 'Soil model',
  /** Right end. Rendered italic and warm, so a prediction can never be mistaken for a measurement. */
  cobbAxisPredicted: 'Cobb illustration',

  /** Shown while the slider sits at the measured end — names the sameness as the result rather than leaving it looking like a dead control. */
  cobbCaptionMeasured:
    'Our comparison uses two maps. Field samples would be needed to test how urbanisation affects soil nematodes.',
  /**
   * Shown at the prediction end. Without this, a viewer drags to full strength,
   * sees almost nothing move, and concludes the CONTROL is broken — when what
   * they are actually seeing is how little of the planet is built up. "Up to
   * 4x" is exact: `field/cobb.ts`'s `cobbMultiplier(1) = 1 + 3 = 4`.
   */
  cobbCaptionPrediction:
    'The illustration increases modelled density with built-up cover, reaching 4× in a fully built-up cell. We chose that multiplier.',

  /** The slider genuinely does nothing on the synthetic field — it is built without an urban atlas, so the multiplier is not merely small, it is absent. Say so instead of reporting a mix. */
  cobbUnavailableSynthetic:
    'City comparisons use the soil model. Press d to switch back once it loads.',
  /** Same class of honesty problem, different cause: the GHSL atlas is a separate, later download. */
  cobbUnavailableUrbanLoading:
    'The comparison needs the built-up land map. It is unavailable until that map loads.',

  /**
   * Inline affordances next to the readouts they qualify, rather than buried
   * in the key list. Kept terse on purpose: moving these two out of the key
   * line and onto the data readout was a deliberate ink transfer, and at
   * full length they simply became the second-largest block instead.
   */
  dataSourceHintReal: 'press d for synthetic',
  dataSourceHintSynthetic: 'press d for the soil model',
  uncertaintyHint: 'u for model uncertainty',
} as const

/** A documented key binding. `group` drives both the panel's headings and the argument about which bindings may be hidden. */
export interface KeyBinding {
  readonly key: string
  readonly description: string
  /**
   * `evidence` — changes what the piece CLAIMS (which data, how uncertain).
   * `view`     — changes how it LOOKS.
   * `dev`      — instrumentation, useful only with a console open.
   */
  readonly group: 'evidence' | 'view' | 'dev'
}

/**
 * The full list, shown on `?`.
 *
 * The `evidence` group exists to make an argument enforceable: bindings that
 * change what the piece claims are surfaced NEXT TO the claim they change
 * (see `dataSourceHint*` / `uncertaintyHint` above), not only inside a panel
 * the viewer has to know to open. Hiding a view toggle costs a viewer some
 * fiddling; hiding `d` would cost them the knowledge that the piece has a
 * synthetic mode at all.
 */
export const KEY_BINDINGS: readonly KeyBinding[] = [
  { key: 'd', description: 'switch between the soil model and the synthetic pattern', group: 'evidence' },
  { key: 'u', description: 'model uncertainty overlay', group: 'evidence' },
  { key: ' ', description: 'play or pause the sweep', group: 'view' },
  { key: '[', description: 'dimmer exposure', group: 'view' },
  { key: ']', description: 'brighter exposure', group: 'view' },
  { key: 'r', description: 'shell / particles / blended', group: 'view' },
  { key: 's', description: 'show or hide sample locations', group: 'view' },
  { key: '?', description: 'show keyboard shortcuts', group: 'view' },
  { key: 'i', description: 'diagnostics readout', group: 'dev' },
  { key: 'v', description: 'deposit debug view', group: 'dev' },
  { key: 'p', description: 'log particle buffers', group: 'dev' },
  { key: 't', description: 'log deposit trace', group: 'dev' },
]

export const KEY_GROUP_TITLES: Record<KeyBinding['group'], string> = {
  evidence: 'Data',
  view: 'View',
  dev: 'Diagnostics',
}

/**
 * The only bindings that stay on screen unprompted. Two of them are mouse
 * gestures with no discoverable affordance of their own; the third is the way
 * into everything else.
 */
export const ALWAYS_VISIBLE_HINTS: readonly string[] = [
  'drag to orbit',
  // Named for BOTH input methods, in one string shown to both. "scroll to
  // descend" was a mouse instruction printed on every phone, telling a
  // visitor to do the one thing their device cannot do — and descent is the
  // entire piece, so the hint read as the piece being broken. Splitting it by
  // pointer type was rejected: a laptop with a touchscreen is genuinely both,
  // and the string is six characters longer this way.
  'pinch or scroll to descend',
  '? for keys',
]

/**
 * The one link out of the viewer. Separate from ALWAYS_VISIBLE_HINTS because
 * those are gestures and this is a destination -- and because it renders as an
 * anchor, which #hud's `pointer-events: none` would otherwise swallow.
 */
export const ABOUT_LINK_TEXT = 'Read the city comparison'

/**
 * Where that link goes, and where the no-WebGPU fallback's link goes too.
 *
 * One constant for both because they are the same destination reached by two
 * different visitors, and a path typed twice is a path that can be corrected
 * once. `vite.config.ts` must also list about.html as an entry point or this
 * href 404s on deploy while working perfectly under `pnpm dev` -- see the
 * note there.
 */
export const ABOUT_LINK_HREF = '/about.html'

/** Static fallback copy also printed in viewer.html for visitors without JavaScript. */
export const NO_WEBGPU_COPY = {
  heading: 'The soil model in one frame',
  requirement:
    'The interactive globe needs JavaScript and a device that supports WebGPU.',
  missing:
    'Air, water and rock fade over about 34 seconds. The veil that remains shows where the 2019 soil model ' +
    'predicts more nematodes. In the interactive view, you can compare it with published sample locations.',
  remedy:
    'To try the interactive view, enable JavaScript and check',
  remedyLinkText: 'WebGPU browser support',
  remedyLinkHref: 'https://caniuse.com/webgpu',
  posterCaption:
    'The soil model seen from 11,500 km. The rendering exaggerates the veil’s thickness and brightness so its shape is visible.',
  aboutLead:
    'The methods explain how the soil map was made and what the comparison with built-up land showed across 20 cities.',
  aboutLinkText: 'Read the science and methods',
  adapterFailure:
    'This browser could not start WebGPU on this device. The still image shows the final frame of the sweep.',
} as const

/** Every user-visible string above, named, for the copy-wide assertions. */
export function allCopyStrings(): Array<[string, string]> {
  return [
    ...Object.entries(HUD_COPY).map(([k, v]) => [k, v] as [string, string]),
    ...KEY_BINDINGS.map((b) => [`key ${b.key}`, b.description] as [string, string]),
    ...ALWAYS_VISIBLE_HINTS.map((h, i) => [`hint ${i}`, h] as [string, string]),
    ['about link', ABOUT_LINK_TEXT],
    ...Object.entries(NO_WEBGPU_COPY).map(([k, v]) => [`no-webgpu ${k}`, v] as [string, string]),
  ]
}
