export interface AdapterReport {
  ok: boolean
  features: string[]
  hasTimestamp: boolean
  hasIndirectFirstInstance: boolean
  hasFloat32Filterable: boolean
  maxTextureDimension3D: number
  reason?: string
}

/**
 * Availability gate. Firefox 141+ is a partial implementation with no Linux and
 * no Intel macOS; Chrome is full only from 144. We accept that gap deliberately
 * and show a static poster instead (spec §10).
 */
export function isWebGPUAvailable(nav: { gpu?: unknown } = globalThis.navigator ?? {}): boolean {
  return nav.gpu != null
}

export async function probeAdapter(): Promise<AdapterReport> {
  const empty: AdapterReport = {
    ok: false, features: [], hasTimestamp: false, hasIndirectFirstInstance: false,
    hasFloat32Filterable: false, maxTextureDimension3D: 0,
  }
  if (!isWebGPUAvailable()) return { ...empty, reason: 'navigator.gpu absent' }

  // @webgpu/types declares Navigator.gpu, so no cast is needed here.
  // isWebGPUAvailable() above already confirmed it's present.
  const gpu = navigator.gpu!
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) return { ...empty, reason: 'requestAdapter returned null' }

  const features = [...adapter.features] as string[]
  return {
    ok: true,
    features,
    hasTimestamp: features.includes('timestamp-query'),
    hasIndirectFirstInstance: features.includes('indirect-first-instance'),
    hasFloat32Filterable: features.includes('float32-filterable'),
    maxTextureDimension3D: adapter.limits.maxTextureDimension3D,
  }
}
