import { clampTimelineDt, LAST_VISIBLE_LAYER_T, SWEEP_T0, T_END } from './Timeline'

/**
 * Give the real air/water/rock removal the long middle of the film. After the
 * final rock fragment is gone there is no fourth matter layer to animate, so
 * the veil gets a short hold instead of a long, visually static checklist.
 * Authored time remains 0..20 and scrubbing remains direct.
 */
const SEGMENTS = [
  { end: SWEEP_T0, rate: 0.75 },             // Four-second invitation.
  { end: LAST_VISIBLE_LAYER_T, rate: 0.25 }, // Twenty-four seconds of visible erosion.
  { end: T_END, rate: 2 },                   // Five-and-a-half-second settle.
] as const

/** Elapsed wall seconds corresponding to any directly scrubbed authored time. */
export function playbackSeconds(authoredTime: number): number {
  let remaining = Math.min(T_END, Math.max(0, authoredTime))
  let previousEnd = 0
  let seconds = 0
  for (const { end, rate } of SEGMENTS) {
    const inSegment = Math.min(remaining, end - previousEnd)
    seconds += inSegment / rate
    remaining -= inSegment
    previousEnd = end
    if (remaining <= 0) break
  }
  return seconds
}

/** Exact piecewise integration, including frames that cross a phase boundary. */
export function playbackDelta(wallSeconds: number, authoredTime = 0): number {
  const start = Math.min(T_END, Math.max(0, authoredTime))
  let cursor = start
  let remaining = clampTimelineDt(Math.max(0, wallSeconds))
  for (const { end, rate } of SEGMENTS) {
    if (cursor >= end) continue
    const advance = Math.min(end - cursor, remaining * rate)
    cursor += advance
    remaining -= advance / rate
    if (remaining <= 1e-12) break
  }
  // Repeated frame deltas can land a few ulps short of the exact endpoint.
  if (T_END - cursor < 1e-9) cursor = T_END
  return cursor - start
}
