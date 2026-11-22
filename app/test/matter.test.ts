import { describe, it, expect } from 'vitest'
import * as THREE from 'three/webgpu'
import { uniform } from 'three/tsl'
import { buildMatterMaterial, SUN_DIR, type MatterOpts } from '../src/sweep/Matter'
import { OPENING_POSE, ScaleCamera } from '../src/core/ScaleCamera'
import { OpeningExperience } from '../src/core/openingExperience'
import { LAST_VISIBLE_LAYER_T, Timeline } from '../src/sweep/Timeline'
import { createDissolveUniforms } from '../src/sweep/dissolveNodes'
import { bakeSyntheticCubeTexture } from '../src/field/bakeCube'
import matterSource from '../src/sweep/Matter.ts?raw'
import viewerSource from '../src/viewer.ts?raw'

// A throwaway cube texture: buildMatterMaterial only builds the TSL graph
// here, it never samples GPU-side, so any correctly-typed CubeTexture
// satisfies it - the synthetic bake's is the smallest one already on hand.
function fakeOpts(): MatterOpts {
  const fakeCube = bakeSyntheticCubeTexture({ size: 2 })
  return {
    atmosphere: createDissolveUniforms(),
    water: createDissolveUniforms(),
    rock: createDissolveUniforms(),
    bathymetry: uniform(0),
    exaggeration: uniform(1),
    elevationTexture: fakeCube,
    densityTexture: fakeCube,
    urbanTexture: fakeCube,
    texelAngularWidthRad: 0.01,
  }
}

describe('buildMatterMaterial', () => {
  it('builds without throwing and is opaque (occludes via normal depth test/write, never blends)', () => {
    const material = buildMatterMaterial(fakeOpts())
    expect(material.transparent).toBe(false)
  })
})

describe('the governing rule is structurally enforced, not just documented', () => {
  it('dissolves via Discard, never via a blended/faded alpha', () => {
    expect(matterSource).toMatch(/Discard\(/)
    // vec4(color, 1.0): a constant, non-dissolve-driven alpha - if this ever
    // reads a variable derived from `alpha`/`matterAlphaNode`, the mesh
    // would be FADING instead of being cleanly present-or-absent, which is
    // exactly what spec 6.1's governing rule forbids.
    expect(matterSource).toMatch(/vec4\(color,\s*1\.0\)/)
  })
  it('never imports regimes/Shell.ts - the veil is not this file\'s concern', () => {
    expect(matterSource).not.toMatch(/from\s+['"].*regimes\/Shell['"]/)
  })
})

// Ground at the terminator is lit by reddened sunlight; it does not EMIT
// amber. Both earlier attempts got this backwards and both produced the same
// artefact - a hard-edged warm stripe ruled straight across the disc,
// brightest exactly where the terrain under it was darkest - because an
// additive constant against near-black unlit ground is all signal and no
// shading. The fix is to tint the incoming light instead, so the effect is
// bounded by how much sun that fragment was receiving in the first place.
describe('terminator warmth reaches the terrain by tinting sunlight, never by addition', () => {
  it('multiplies the diffuse term by the sun transmittance', () => {
    expect(matterSource).toMatch(/ndotl\.mul\(DIFFUSE\)\.mul\(sunTransmittance\)/)
  })

  it('has no additive terrain terminator tint left to reintroduce the stripe', () => {
    expect(matterSource).not.toMatch(/TERRAIN_TERMINATOR_STRENGTH/)
    expect(matterSource).not.toMatch(/\.add\(terminatorTint\)/)
  })

  it('takes land fraction from the ELEVATION atlas, which is 8x sharper than the density one', () => {
    // The density cube's A channel is baked from a 512x256 equirect
    // extraction (~78 km/texel) and smears the coastline over ~275 km. The
    // elevation atlas's G channel is a real land mask at 9.77 km/texel,
    // registered to the same texels as the relief it shades.
    expect(matterSource).toMatch(/sampleLandMaskNode\(opts\.elevationTexture/)
    expect(matterSource).not.toMatch(/sampleLandFractionNode\(opts\.densityTexture/)
  })

  it('does not mix the atmosphere toward a constant colour, which would discard the view-angle term', () => {
    // `mix(scatterColor, warmColor, warmth)` returns warmColor outright at
    // the terminator - see terrain.ts::terminatorSunTransmittance's doc.
    expect(matterSource).not.toMatch(/mix\(\s*scatterColor/)
  })
})

// City lights only draw where `nightFactor > 0`, so the sun's LATITUDE
// decides which of the world's cities can ever be seen lit — a constraint no
// camera angle and no amount of tuning can work around.
//
// Over a full longitude rotation a point at latitude phi reaches a minimum
// `ndotSun` of `-cos(phi + phi_sun)`, so it touches the night side only when
// `phi + phi_sun < 90`. The previous sun sat at 54.2N, capping night at
// 35.8N: London, Paris, Berlin, Moscow, Beijing, Tokyo, New York and Chicago
// were in permanent daylight.
// The ocean-collapse beat: a multi-agent audit found `oceanProgress` was
// `clamp(max(1 - waterAlpha, opts.bathymetry), 0, 1)`, and `1 - waterAlpha`
// saturates to 1 (exactly, network-wide) at t=7.4s while `opts.bathymetry`
// only leaves 0 at t=9.0s (old WIN.bathymetry timing) - so `bathymetry` was
// mathematically dead, `oceanProgress` was ALWAYS driven by `1 - waterAlpha`,
// and since the SAME `waterAlpha` also drives the fragment discard, a
// fragment's odds of being visible AT ALL were exactly its own odds of NOT
// having dropped - the drop and the discard could never both be true at
// once, so the reveal was invisible by construction (expected visible signal
// <= 0.25 by alpha*(1-alpha)). None of this can be executed headless (no
// WebGPU in this test environment - see webgpu.test.ts), so - same as the
// rest of this describe block - these are structural regex checks on the
// TSL source, not numeric ones; the numeric formulas they wire together
// (effectiveExaggeration, hypsometricColor's oceanProgress gating) are
// pinned for real in terrain.test.ts, and the retimed WIN.bathymetry
// (bathymetry now finishes BEFORE water's discard front ever moves) in
// timeline.test.ts.
describe('the ocean-collapse beat: displacement decoupled from the discard', () => {
  it('drives oceanProgress from bathymetry alone, not from waterAlpha at all', () => {
    expect(matterSource).toMatch(/oceanProgress = clamp\(opts\.bathymetry, 0, 1\)/)
    expect(matterSource).not.toMatch(/max\(float\(1\)\.sub\(waterAlpha\)/)
  })

  it('gates the ocean colour ramp on the same oceanProgress, so the reveal is visible as colour', () => {
    expect(matterSource).toMatch(/hypsometricColorNode\(elevation, landFraction, latDeg, vegIndex, oceanProgress\)/)
  })

  it('uses the same drain progress for relief lighting and remaining water reflections', () => {
    expect(matterSource).toMatch(/terrainNormalNode\(\s*opts\.elevationTexture, dir, opts\.exaggeration, landFraction, oceanProgress,/)
    expect(matterSource).toContain('float(1).sub(clamp(landFraction, 0, 1)).mul(float(1).sub(oceanProgress))')
    expect(matterSource).toContain('mul(specIntensity).mul(sunTransmittance).mul(waterSurface)')
  })
})

describe('solid-Earth visibility during a scrub', () => {
  it('stops drawing once both opaque layers are gone and draws again after a rewind', () => {
    expect(viewerSource).toContain('matter.visible = sweepState.frontWater < 1 || sweepState.frontRock < 1')
    const timeline = new Timeline()
    const edge = LAST_VISIBLE_LAYER_T
    const times = [edge - 0.05, edge, edge + 1, edge - 0.05, 0]
    const visible = times.map((t) => {
      timeline.scrubTo(t)
      return timeline.state.frontWater < 1 || timeline.state.frontRock < 1
    })
    expect(visible).toEqual([true, false, false, true, true])
  })
})

describe('the sun is placed so the world\'s cities can actually be seen at night', () => {
  const subsolarLatDeg = (Math.asin(SUN_DIR.z) * 180) / Math.PI

  it('is a unit vector', () => {
    expect(SUN_DIR.length()).toBeCloseTo(1, 12)
  })

  it('keeps the sub-solar latitude at the ruled ~10N', () => {
    expect(subsolarLatDeg).toBeGreaterThan(5)
    expect(subsolarLatDeg).toBeLessThan(15)
  })

  it('lets every inhabited latitude reach the night side', () => {
    // Utqiagvik, Alaska at 71.3N is about as far north as year-round
    // settlement goes; Saint Petersburg at 59.9N is the northern limit of
    // anything GHSL would call a major city.
    for (const cityLatDeg of [71.3, 59.9, 51.5, 40.7, 35.7, 0, -34.6, -54.8]) {
      expect(cityLatDeg + subsolarLatDeg).toBeLessThan(90)
    }
  })

  it('is not so close to an equinox that the poles never light at all', () => {
    expect(Math.abs(subsolarLatDeg)).toBeGreaterThan(1)
    expect(Math.abs(subsolarLatDeg)).toBeLessThan(23.44) // never beyond a real solstice
  })
})

// The sun's LONGITUDE and the camera's opening longitude are one decision,
// not two. Held apart, they produced the defect this block exists to stop
// coming back: the sun sat at 155E while `OPENING_POSE` opened at 0, and a
// viewer who loaded the page and scrolled descended into the night side.
// Measured at the opening pose with the drift stopped and matter intact:
// frame mean 0.746/255 at 11,500 km, 0.0025 at 1,000 km and exactly 0.000
// at 381 km, against 36.9 / 170.0 / 170.1 at the sub-solar point.
//
// These assertions are on the PAIR. Moving either the sun's longitude or
// the opening longitude on its own fails them; moving both together in a
// way that keeps the piece lit passes.
describe('the opening frame descends onto lit ground', () => {
  const RAD = Math.PI / 180
  /** The actual controller advances the reveal without moving the view. */
  const cameraDirAt = (tSec: number): THREE.Vector3 => {
    const camera = new ScaleCamera()
    const opening = new OpeningExperience(camera, new Timeline())
    opening.setEssential('density', 'ready')
    opening.setEssential('elevation', 'ready')
    opening.togglePlayback()
    for (let t = 0; t < tSec; t += 0.1) opening.advance(0.1)
    return new THREE.Vector3(...camera.direction)
  }

  it('has the descent point in sunlight the instant the page loads', () => {
    // Sun at least 30 degrees above the local horizon (dot > sin(30°)).
    // The 10N/10W candidate scores 0.654; the original night-side bug -0.89.
    expect(cameraDirAt(0).dot(SUN_DIR)).toBeGreaterThan(0.5)
  })

  it('stays lit across the whole cinematic, not just at t=0', () => {
    for (let t = 0; t <= 40; t += 0.5) {
      expect(cameraDirAt(t).dot(SUN_DIR)).toBeGreaterThan(0.5)
    }
  })

  it('keeps sunlight constant while the guided reveal changes the layers', () => {
    expect(cameraDirAt(40).dot(SUN_DIR)).toBe(cameraDirAt(0).dot(SUN_DIR))
  })

  it('leaves substantial urban area on the night side for the city lights', () => {
    // Spec 6.3's rhyme needs cities visible at night on orbit. Each of these
    // must be on the unlit hemisphere at the opening instant. The night
    // hemisphere runs 130E through -50E, so this is the Americas plus the
    // western Pacific rim; mainland East Asia (Beijing 116E, Seoul 127E)
    // sits just inside daylight and rotates into night as the drift
    // continues — deliberately not asserted, because it is a terminator
    // straddle rather than a property the framing depends on.
    const cities: Array<[string, number, number]> = [
      ['Tokyo', 139.7, 35.7], ['Sydney', 151.2, -33.9], ['Auckland', 174.8, -36.8],
      ['New York', -74.0, 40.7], ['Los Angeles', -118.2, 34.1], ['Mexico City', -99.1, 19.4],
      ['Buenos Aires', -58.4, -34.6], ['Lima', -77.0, -12.0],
    ]
    for (const [name, lonDeg, latDeg] of cities) {
      const d = new THREE.Vector3(
        Math.cos(latDeg * RAD) * Math.cos(lonDeg * RAD),
        Math.cos(latDeg * RAD) * Math.sin(lonDeg * RAD),
        Math.sin(latDeg * RAD),
      )
      expect(d.dot(SUN_DIR), `${name} should be on the night side`).toBeLessThan(0)
    }
  })

  it('opens on a landmass rather than an ocean face', () => {
    // Longitude 0 at the equator faces Africa and Europe; the visible disc
    // spans roughly +-90 degrees of it. Asserted as a property of the pose
    // rather than of pixels: the opening longitude must sit within the
    // Africa/Europe/western-Asia span, not out over the Pacific.
    const lonDeg = ((OPENING_POSE.lonRad / RAD) % 360 + 540) % 360 - 180
    expect(lonDeg).toBeGreaterThan(-30)
    expect(lonDeg).toBeLessThan(60)
    expect(Math.abs(OPENING_POSE.latRad / RAD)).toBeLessThan(30)
  })
})
