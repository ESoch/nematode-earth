import type { SweepState } from './Timeline'

export type SweepPhase = 'ready' | 'atmosphere' | 'water' | 'rock' | 'remains'
export type LayerStatus = 'intact' | 'leaving' | 'gone'
export type GuideState = Pick<SweepState, 'frontAtmosphere' | 'frontWater' | 'frontRock' | 'bathymetry'>
export interface SweepGuide {
  phase: SweepPhase
  layers: { name: string; status: LayerStatus }[]
}

export type GuidePresentation = {
  mode: 'intro'
  announce: false
  premise: string
  excerpt: string
  attribution: string
  sourceHref: string
} | ({ mode: 'layers'; announce: true } & SweepGuide)

const layerStatus = (front: number): LayerStatus =>
  front >= 1 ? 'gone' : front > 0 ? 'leaving' : 'intact'

/**
 * Presentation follows the same uniforms as Matter, not nominal chapter times.
 * The ocean descent counts as motion, but only its dissolve front can mark
 * water gone. Several layers can be leaving at once. The first unfinished one
 * owns the caption; all three statuses remain visible. No progress is a mass %.
 */
export function sweepGuideAt(state: GuideState): SweepGuide {
  const atmosphere = layerStatus(state.frontAtmosphere)
  const water = state.frontWater >= 1 ? 'gone'
    : state.bathymetry > 0 ? 'leaving' : layerStatus(state.frontWater)
  const rock = layerStatus(state.frontRock)
  const phase: SweepPhase = atmosphere === 'leaving' ? 'atmosphere'
    : water === 'leaving' ? 'water'
      : rock === 'leaving' ? 'rock'
        : [atmosphere, water, rock].every((status) => status === 'gone') ? 'remains'
          : 'ready'
  return {
    phase,
    layers: [
      { name: 'Atmosphere', status: atmosphere },
      { name: 'Water', status: water },
      { name: 'Solid Earth', status: rock },
    ],
  }
}

/** The invitation owns the guide only before the visitor engages with Earth. */
export function sweepGuidePresentationAt(state: GuideState, showIntro: boolean): GuidePresentation {
  const guide = sweepGuideAt(state)
  if (showIntro && guide.phase === 'ready') return {
    mode: 'intro',
    announce: false,
    premise: 'In 1914, Nathan Cobb imagined what Earth would look like if only nematodes remained.',
    excerpt: 'our world would still be dimly recognizable',
    attribution: 'Nathan A. Cobb, 1914',
    sourceHref: 'https://archive.org/details/yoa1914/page/472/mode/2up',
  }
  return { mode: 'layers', announce: true, ...guide }
}
