import { describe, it, expect, vi } from 'vitest'
import { matterAtlasBindings, type MatterAtlases } from '../src/sweep/matterAtlases'
import { replaceMaterial } from '../src/core/materialSwap'

const PLACEHOLDER = 'placeholder'
const none: MatterAtlases<string> = { elevation: null, density: null, urban: null }

describe('matter binds each atlas the moment it lands, independently', () => {
  it('binds a loaded elevation atlas even when NOTHING else ever loads', () => {
    // The regression this exists for: `viewer.ts` used to gate the rebuild on
    // `elevationCubeTex && realCube && urbanCubeTex`, an all-or-nothing AND
    // over three INDEPENDENT network loads. A failed density cube or a failed
    // urban ladder meant the fully-downloaded 192 MiB elevation atlas was
    // fetched, decoded, validated — and then discarded, with nothing on
    // screen changing and only an unrelated console.error to show for it.
    const b = matterAtlasBindings({ ...none, elevation: 'elev' }, PLACEHOLDER)
    expect(b.elevation).toBe('elev')
    expect(b.density).toBe(PLACEHOLDER)
    expect(b.urban).toBe(PLACEHOLDER)
  })

  it('binds each slot from its own load and no other', () => {
    const slots = ['elevation', 'density', 'urban'] as const
    for (const slot of slots) {
      const b = matterAtlasBindings({ ...none, [slot]: 'loaded' }, PLACEHOLDER)
      for (const other of slots) {
        expect(b[other]).toBe(other === slot ? 'loaded' : PLACEHOLDER)
      }
    }
  })

  it('falls back to the placeholder only where an atlas is genuinely absent', () => {
    const b = matterAtlasBindings({ elevation: 'e', density: 'd', urban: 'u' }, PLACEHOLDER)
    expect(b).toEqual({ elevation: 'e', density: 'd', urban: 'u' })
  })
})

describe('replaceMaterial', () => {
  const fake = () => ({ dispose: vi.fn() })

  it('installs the replacement BEFORE disposing what it supersedes', () => {
    // Ordering is the whole point: a NodeMaterial disposed while still
    // assigned takes its compiled pipeline with it, and the next frame draws
    // with a released one.
    const order: string[] = []
    const previous = { dispose: () => order.push('dispose') }
    const next = { dispose: () => order.push('dispose-next') }
    replaceMaterial(previous, next, () => order.push('apply'))
    expect(order).toEqual(['apply', 'dispose'])
  })

  it('disposes the superseded material exactly once', () => {
    const previous = fake()
    replaceMaterial(previous, fake(), () => {})
    expect(previous.dispose).toHaveBeenCalledTimes(1)
  })

  it('does not dispose when there was nothing to supersede', () => {
    const next = fake()
    replaceMaterial(null, next, () => {})
    expect(next.dispose).not.toHaveBeenCalled()
  })

  it('never disposes a material that is still the live one', () => {
    const same = fake()
    replaceMaterial(same, same, () => {})
    expect(same.dispose).not.toHaveBeenCalled()
  })

  it('returns the installed material', () => {
    const next = fake()
    expect(replaceMaterial(null, next, () => {})).toBe(next)
  })
})
