import { OPENING_POSE, type ScaleCamera } from './ScaleCamera'
import { autoRotateDeltaLon } from './orbitInput'
import { Timeline, T_END } from '../sweep/Timeline'
import { playbackDelta } from '../sweep/pacing'

type Essential = 'density' | 'elevation'
type LoadState = 'loading' | 'ready' | 'error'
const ESSENTIALS = ['density', 'elevation'] as const
const MAX_LOAD_ATTEMPTS = 3 // Initial load plus two deliberate retries, per asset.
interface EssentialLoad {
  urls: readonly string[]
  loadAndBind: (url: string) => Promise<void>
  attempts: number
  running: Promise<void> | null
}
interface PlaybackKey {
  key: string
  repeat: boolean
  ctrlKey?: boolean
  metaKey?: boolean
  altKey?: boolean
}

/** Layout policy; the viewer moves the same buttons, keeping one set of actions. */
export function openingControlLayout(width: number, height: number, canBegin: boolean): {
  actions: 'dock' | 'menu'; showStatus: boolean; compactGuide: boolean
} {
  return {
    actions: width <= 700 || height <= 500 ? 'menu' : 'dock',
    showStatus: !canBegin,
    compactGuide: width <= 700 && height <= 700 && height >= width,
  }
}

/** CPU-only opening and transport policy shared by buttons, keys and gestures. */
export class OpeningExperience {
  private rotating = true
  private introActive = true
  private reducedMotion = false
  private essential: Record<Essential, LoadState> = { density: 'loading', elevation: 'loading' }
  private loads: Partial<Record<Essential, EssentialLoad>> = {}
  get autoRotate(): boolean { return this.rotating }
  get showIntro(): boolean { return this.introActive }
  constructor(private camera: ScaleCamera, private timeline: Timeline) {
    camera.setOrientation(OPENING_POSE.lonRad, OPENING_POSE.latRad)
    camera.setAltitude(OPENING_POSE.altM)
  }
  setEssential(asset: Essential, state: LoadState): void {
    // Once a usable atlas is bound, failed upgrades leave it usable.
    if (this.essential[asset] !== 'ready') this.essential[asset] = state
  }
  /** Register and start once. Retrying never rebuilds the camera or timeline. */
  loadEssential(asset: Essential, urls: readonly string[], loadAndBind: EssentialLoad['loadAndBind']): Promise<void> {
    const existing = this.loads[asset]
    if (existing) return existing.running ?? Promise.resolve()
    const load: EssentialLoad = { urls: [...urls], loadAndBind, attempts: 0, running: null }
    this.loads[asset] = load
    return this.runEssential(asset, load)
  }
  private runEssential(asset: Essential, load: EssentialLoad): Promise<void> {
    this.setEssential(asset, 'loading')
    load.attempts++
    load.running = Promise.resolve().then(async () => {
      for (const url of load.urls) {
        try {
          await load.loadAndBind(url)
          this.setEssential(asset, 'ready')
        } catch {
          // The viewer reports the error. Keep the last bound tier and try
          // the next planned tier before declaring the whole asset failed.
        }
      }
      this.setEssential(asset, 'error') // Ready is latched by setEssential.
    }).finally(() => { load.running = null })
    return load.running
  }
  private canRetryAsset(asset: Essential): boolean {
    const load = this.loads[asset]
    return this.essential[asset] === 'error' && !!load && !load.running && load.attempts < MAX_LOAD_ATTEMPTS
  }
  async retryDataLoading(): Promise<void> {
    await Promise.all(ESSENTIALS.map((asset) => {
      const load = this.loads[asset]
      if (!load) return
      if (load.running && load.attempts > 1) return load.running
      if (this.canRetryAsset(asset)) return this.runEssential(asset, load)
    }))
  }
  get readiness(): { canBegin: boolean; canRetry: boolean; retrying: boolean; retryExhausted: boolean; message: string } {
    const assets = ESSENTIALS
    const names = { density: 'soil density', elevation: 'elevation' }
    const failed = assets.filter((asset) => this.essential[asset] === 'error')
    const recovery = {
      canRetry: assets.some((asset) => this.canRetryAsset(asset)),
      retrying: assets.some((asset) => this.essential[asset] === 'loading'
        && (this.loads[asset]?.attempts ?? 0) > 1),
      retryExhausted: failed.some((asset) => (this.loads[asset]?.attempts ?? 0) >= MAX_LOAD_ATTEMPTS),
    }
    const retryMessage = recovery.canRetry ? ' Select Retry data loading to try again.' : ''
    const exhaustedMessage = recovery.retryExhausted ? ' The retry limit has been reached.' : ''
    if (failed.length) return {
      ...recovery, canBegin: false,
      message: `Failed to load ${failed.map((asset) => names[asset]).join(' and ')}. You can still explore the globe with placeholder data.${retryMessage}${exhaustedMessage}`,
    }
    const pending = assets.filter((asset) => this.essential[asset] !== 'ready')
    return pending.length ? {
      ...recovery, canBegin: false,
      message: `${recovery.retrying ? 'Retrying' : 'Loading'} ${pending.map((asset) => names[asset]).join(' and ')}. Explore while waiting.`,
    } : { ...recovery, canBegin: true, message: 'Soil density and elevation ready. Begin the sweep or explore.' }
  }
  /** Idle drift yields permanently to deliberate interaction or playback. */
  setAutoRotate(on: boolean): void { this.rotating = on && !this.reducedMotion }
  setReducedMotion(on: boolean): void {
    this.reducedMotion = on
    if (on) this.rotating = false
  }
  explore(): void {
    this.rotating = false
    this.introActive = false
    this.timeline.pause()
  }
  returnToOpening(): void {
    this.explore()
    this.camera.setOrientation(OPENING_POSE.lonRad, OPENING_POSE.latRad)
    this.camera.setAltitude(OPENING_POSE.altM)
  }
  togglePlayback(): void {
    if (this.timeline.playing) { this.timeline.pause(); return }
    if (!this.readiness.canBegin) return
    if (this.timeline.t >= T_END) { this.replay(); return }
    this.rotating = false
    this.introActive = false
    this.timeline.play()
  }
  /** Watch again / replay intentionally resets BOTH pose and time. */
  replay(): void {
    if (!this.readiness.canBegin) return
    this.returnToOpening()
    this.timeline.scrubTo(0)
    this.timeline.play()
  }
  scrubTo(t: number): void { this.explore(); this.timeline.scrubTo(t) }
  handleSpace(event: PlaybackKey, occupied = false): boolean {
    if (occupied || event.key !== ' ' || event.repeat || event.ctrlKey || event.metaKey || event.altKey) return false
    this.togglePlayback()
    return true
  }
  advance(dt: number): void {
    // A resumed background tab must not jump to a different continent.
    if (this.autoRotate) this.camera.orbitBy(autoRotateDeltaLon(Math.min(Math.max(dt, 0), 0.2)), 0)
    this.timeline.advance(playbackDelta(dt, this.timeline.t))
  }
}
