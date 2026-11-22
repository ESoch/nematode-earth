import { describe, it, expect } from 'vitest'
import { DataUtils } from 'three/webgpu'
import {
  decodeElevation, landColor, oceanColor, hypsometricColor, terrainDisplacementM,
  effectiveExaggeration,
  centralDifferenceSlope, heightfieldNormal, tangentBasis, applyTangentSpaceBump,
  decodeBuiltUpFraction, nightFactor, cityLightIntensity,
  RAYLEIGH_WAVELENGTHS_NM, rayleighWeight, atmosphericPathLength, rayleighScatterIntensity,
  atmosphereScatterColor, terminatorWarmthWeight, atmosphereSunAvailability,
  schlickFresnel, specularLobe, oceanSpecularIntensity, iceElevationLoweringM,
  vegetationIndex, VEGETATION_ARID_LOG_TAU, VEGETATION_LUSH_LOG_TAU,
  highLatitudeCoolness, detailFadeByDistance, terminatorSunTransmittance, twilightAmbientScale,
  uncertaintyColor, uncertaintyCoverage, uncertaintyOverlayColor, formatUncertaintyCv,
  UNCERTAINTY_CV_LOW, UNCERTAINTY_CV_HIGH,
  type Color3,
} from '../src/field/terrain'

const dot3 = (a: readonly number[], b: readonly number[]): number => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!
const norm3 = (v: readonly number[]): number => Math.hypot(v[0]!, v[1]!, v[2]!)

describe('decodeElevation', () => {
  it('round-trips signed elevation through half-float within precision', () => {
    for (const m of [-10752, -4000, -1, 0, 400, 8157]) {
      const bits = DataUtils.toHalfFloat(m)
      expect(decodeElevation(bits)).toBeCloseTo(m, m === 0 ? 5 : -1)
    }
  })
})

describe('landColor / oceanColor', () => {
  it('clamps below the lowest stop and above the highest', () => {
    expect(landColor(-500)).toEqual(landColor(0))
    expect(landColor(9000)).toEqual(landColor(7000))
    expect(oceanColor(500)).toEqual(oceanColor(0))
    expect(oceanColor(-20000)).toEqual(oceanColor(-10500))
  })

  it('interpolates smoothly between stops (monotone red channel across the land ramp\'s early rise)', () => {
    const c0 = landColor(0)
    const c200 = landColor(200)
    const c400 = landColor(400)
    expect(c200[1]).toBeGreaterThan(c0[1])
    expect(c200[1]).toBeLessThan(c400[1])
  })

  it('is desaturated: no channel saturates at the extremes (spec: "restrained, not photoreal")', () => {
    for (const m of [-10500, -4000, 0, 400, 4500, 8157]) {
      for (const c of [...landColor(m), ...oceanColor(m)]) {
        expect(c).toBeGreaterThanOrEqual(0)
        expect(c).toBeLessThanOrEqual(1)
      }
    }
  })

  it('land is warmer/brighter than ocean at comparable elevations near datum', () => {
    const land = landColor(0)
    const ocean = oceanColor(0)
    // Land greener, ocean bluer-relative-to-green: land's green channel
    // exceeds its blue by more than ocean's does.
    expect(land[1] - land[2]).toBeGreaterThan(ocean[1] - ocean[2])
  })
})

describe('hypsometricColor', () => {
  it('is exactly oceanColor at landFraction 0 and exactly landColor at landFraction 1', () => {
    expect(hypsometricColor(-3000, 0)).toEqual(oceanColor(-3000))
    expect(hypsometricColor(100, 1)).toEqual(landColor(100))
  })

  it('blends continuously for partial land fraction (no hard coastline threshold)', () => {
    // Netherlands-like coastal texel: land fraction 0.54 (see geoProbe's
    // real-data probe). Must land strictly between the pure endpoints, not
    // snap to either one.
    const ocean = oceanColor(0)
    const land = landColor(0)
    const blended = hypsometricColor(0, 0.54)
    for (let i = 0; i < 3; i++) {
      const lo = Math.min(ocean[i]!, land[i]!)
      const hi = Math.max(ocean[i]!, land[i]!)
      expect(blended[i]).toBeGreaterThanOrEqual(lo)
      expect(blended[i]).toBeLessThanOrEqual(hi)
    }
  })

  it('never classifies a below-datum land point (Death Valley-like) as ocean colour', () => {
    // The whole point of using land fraction instead of thresholding
    // elevation at zero: a below-sea-level point with landFraction=1 must
    // still get the LAND ramp, not the ocean one.
    expect(hypsometricColor(-86, 1)).toEqual(landColor(-86))
  })
})

// The ocean-collapse beat's visible signal: as `oceanProgress` (bathymetry)
// rises 0 -> 1, ocean colour has to visibly run down OCEAN_STOPS from the
// shoreline stop toward the seafloor's true-depth stop - a purely
// geometric drop measured at well under one screen pixel at the piece's
// opening altitude (team lead's audit), so colour is the actual carrier of
// the beat, not displacement.
describe('hypsometricColor ocean-progress gating (the collapse beat)', () => {
  it('defaults to oceanProgress=1 (full depth), so every existing call site is unaffected', () => {
    expect(hypsometricColor(-4000, 0)).toEqual(oceanColor(-4000))
  })

  it('reads as the SHORELINE stop (elevation effectively 0) when oceanProgress is 0', () => {
    expect(hypsometricColor(-4000, 0, 0, 0.5, 0)).toEqual(oceanColor(0))
  })

  it('reaches the true-depth colour once oceanProgress is 1, matching the un-gated 4-arg call', () => {
    expect(hypsometricColor(-4000, 0, 0, 0.5, 1)).toEqual(hypsometricColor(-4000, 0))
  })

  it('darkens monotonically (runs down the ramp) as oceanProgress rises for a below-datum seafloor', () => {
    const luminance = (c: Color3) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    let previous = Infinity
    for (const p of [0, 0.25, 0.5, 0.75, 1]) {
      const lum = luminance(hypsometricColor(-6000, 0, 0, 0.5, p))
      expect(lum).toBeLessThanOrEqual(previous)
      previous = lum
    }
  })

  it('does not affect LAND colour at all - oceanProgress only gates the ocean branch', () => {
    const a = hypsometricColor(1000, 1, 20, 0.5, 0)
    const b = hypsometricColor(1000, 1, 20, 0.5, 1)
    expect(a).toEqual(b)
    expect(a).toEqual(landColor(1000, 0.5))
  })

  it('clamps out-of-range oceanProgress rather than extrapolating past the ramp ends', () => {
    expect(hypsometricColor(-4000, 0, 0, 0.5, -1)).toEqual(oceanColor(0))
    expect(hypsometricColor(-4000, 0, 0, 0.5, 2)).toEqual(oceanColor(-4000))
  })
})

describe('effectiveExaggeration', () => {
  it('is exactly 1 (unexaggerated) at landFraction 0 - marine relief never gets k (spec 7.5)', () => {
    expect(effectiveExaggeration(0, 3.6)).toBeCloseTo(1, 12)
  })

  it('is exactly the given exaggeration at landFraction 1', () => {
    expect(effectiveExaggeration(1, 3.6)).toBeCloseTo(3.6, 12)
  })

  it('blends continuously for partial land fraction', () => {
    expect(effectiveExaggeration(0.5, 3.6)).toBeCloseTo(1 + 0.5 * (3.6 - 1), 12)
  })

  it('is monotone increasing in land fraction', () => {
    const a = effectiveExaggeration(0, 3.0)
    const b = effectiveExaggeration(0.5, 3.0)
    const c = effectiveExaggeration(1, 3.0)
    expect(b).toBeGreaterThan(a)
    expect(c).toBeGreaterThan(b)
  })

  it('clamps out-of-range land fraction', () => {
    expect(effectiveExaggeration(-0.5, 3.6)).toBeCloseTo(1, 12)
    expect(effectiveExaggeration(1.5, 3.6)).toBeCloseTo(3.6, 12)
  })
})

describe('terrainDisplacementM', () => {
  it('land is always fully exaggerated regardless of bathymetry progress', () => {
    expect(terrainDisplacementM(1000, 1, 3.0, 0)).toBeCloseTo(3000, 6)
    expect(terrainDisplacementM(1000, 1, 3.0, 1)).toBeCloseTo(3000, 6)
  })

  it('ocean sits flat at datum when bathymetry progress is 0', () => {
    expect(terrainDisplacementM(-4000, 0, 3.0, 0)).toBeCloseTo(0, 6)
  })

  it('ocean drops to true (unexaggerated) depth as bathymetry progress reaches 1', () => {
    expect(terrainDisplacementM(-4000, 0, 3.0, 1)).toBeCloseTo(-4000, 6)
  })

  it('ocean displacement is monotone in bathymetry progress for a below-datum seafloor', () => {
    const at0 = terrainDisplacementM(-4000, 0, 3.0, 0)
    const atHalf = terrainDisplacementM(-4000, 0, 3.0, 0.5)
    const at1 = terrainDisplacementM(-4000, 0, 3.0, 1)
    expect(at0).toBeGreaterThan(atHalf)
    expect(atHalf).toBeGreaterThan(at1)
  })

  it('blends continuously across partial land fraction', () => {
    const d = terrainDisplacementM(1000, 0.5, 2.0, 0.5)
    // ocean term: 1000*0.5=500, land term: 1000*2.0=2000, blend at t=0.5 -> 1250
    expect(d).toBeCloseTo(1250, 6)
  })
})

describe('centralDifferenceSlope', () => {
  it('is zero for flat terrain', () => {
    expect(centralDifferenceSlope(100, 100, 1000)).toBe(0)
  })

  it('is positive when height increases in the positive direction', () => {
    expect(centralDifferenceSlope(0, 100, 1000)).toBeGreaterThan(0)
  })

  it('scales inversely with spacing', () => {
    const steep = centralDifferenceSlope(0, 100, 100)
    const gentle = centralDifferenceSlope(0, 100, 1000)
    expect(steep).toBeGreaterThan(gentle)
  })
})

describe('heightfieldNormal', () => {
  it('is exactly [0,0,1] for flat terrain', () => {
    const n = heightfieldNormal(0, 0)
    expect(n[0]).toBeCloseTo(0, 10)
    expect(n[1]).toBeCloseTo(0, 10)
    expect(n[2]).toBeCloseTo(1, 10)
  })

  it('is always unit length', () => {
    for (const [dhdx, dhdy] of [[0.5, -0.3], [2, 2], [-5, 0], [0, 0.01]] as const) {
      expect(norm3(heightfieldNormal(dhdx, dhdy))).toBeCloseTo(1, 10)
    }
  })

  it('tilts away from the ascending slope direction', () => {
    // A slope rising in +x should tilt the normal toward -x.
    const n = heightfieldNormal(1.0, 0)
    expect(n[0]).toBeLessThan(0)
    expect(n[2]).toBeGreaterThan(0)
  })
})

describe('tangentBasis', () => {
  it('is orthonormal and right-handed for a range of directions', () => {
    const dirs: Array<readonly [number, number, number]> = [
      [0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0], [0.6, 0.48, 0.64], [-0.3, 0.7, -0.65],
    ]
    for (const n of dirs) {
      const nn = ((): readonly [number, number, number] => {
        const len = norm3(n)
        return [n[0] / len, n[1] / len, n[2] / len]
      })()
      const { tangent, bitangent } = tangentBasis(nn)
      expect(norm3(tangent)).toBeCloseTo(1, 8)
      expect(norm3(bitangent)).toBeCloseTo(1, 8)
      expect(dot3(tangent, nn)).toBeCloseTo(0, 8)
      expect(dot3(bitangent, nn)).toBeCloseTo(0, 8)
      expect(dot3(tangent, bitangent)).toBeCloseTo(0, 8)
      // Right-handed: tangent x bitangent == n.
      const cross = [
        tangent[1] * bitangent[2] - tangent[2] * bitangent[1],
        tangent[2] * bitangent[0] - tangent[0] * bitangent[2],
        tangent[0] * bitangent[1] - tangent[1] * bitangent[0],
      ]
      expect(cross[0]).toBeCloseTo(nn[0], 8)
      expect(cross[1]).toBeCloseTo(nn[1], 8)
      expect(cross[2]).toBeCloseTo(nn[2], 8)
    }
  })
})

describe('applyTangentSpaceBump', () => {
  it('returns the base normal unchanged for a zero bump', () => {
    const base: readonly [number, number, number] = [0, 0, 1]
    const { tangent, bitangent } = tangentBasis(base)
    const result = applyTangentSpaceBump(base, tangent, bitangent, [0, 0, 1])
    expect(result[0]).toBeCloseTo(base[0], 8)
    expect(result[1]).toBeCloseTo(base[1], 8)
    expect(result[2]).toBeCloseTo(base[2], 8)
  })

  it('is always unit length regardless of bump direction', () => {
    const base: readonly [number, number, number] = [0.6, 0.48, 0.64]
    const { tangent, bitangent } = tangentBasis(base)
    for (const bump of [[0.3, -0.2, 1], [1, 1, 1], [-0.9, 0.1, 0.4]] as const) {
      const result = applyTangentSpaceBump(base, tangent, bitangent, bump)
      expect(norm3(result)).toBeCloseTo(1, 8)
    }
  })
})

describe('rayleighWeight', () => {
  it('scatters blue several times more strongly than red (1/wavelength^4 law)', () => {
    const red = rayleighWeight(RAYLEIGH_WAVELENGTHS_NM.r)
    const blue = rayleighWeight(RAYLEIGH_WAVELENGTHS_NM.b)
    expect(blue / red).toBeCloseTo(5.5, 0)
  })

  it('is exactly 1 at the reference (green) wavelength', () => {
    expect(rayleighWeight(RAYLEIGH_WAVELENGTHS_NM.g)).toBeCloseTo(1, 10)
  })

  it('is monotone decreasing in wavelength', () => {
    expect(rayleighWeight(400)).toBeGreaterThan(rayleighWeight(500))
    expect(rayleighWeight(500)).toBeGreaterThan(rayleighWeight(600))
  })
})

describe('atmosphericPathLength', () => {
  it('is 1 at normal incidence (looking straight down)', () => {
    expect(atmosphericPathLength(1)).toBeCloseTo(1, 6)
  })

  it('grows as the view angle grazes toward the limb', () => {
    expect(atmosphericPathLength(0.5)).toBeGreaterThan(atmosphericPathLength(1))
    expect(atmosphericPathLength(0.1)).toBeGreaterThan(atmosphericPathLength(0.5))
  })

  it('saturates at maxPathLength rather than diverging at the true limb', () => {
    expect(atmosphericPathLength(0, 10)).toBe(10)
    expect(atmosphericPathLength(0.0001, 10)).toBeLessThanOrEqual(10)
  })
})

describe('rayleighScatterIntensity', () => {
  it('is 0 with no path length', () => {
    expect(rayleighScatterIntensity(0, 1, 1)).toBe(0)
  })

  it('approaches (but never exceeds) 1 as path length grows', () => {
    expect(rayleighScatterIntensity(50, 1, 1)).toBeCloseTo(1, 6)
    expect(rayleighScatterIntensity(50, 1, 1)).toBeLessThanOrEqual(1)
  })

  it('is stronger for a higher wavelength weight (blue) than a lower one (red) at the same path length', () => {
    const red = rayleighScatterIntensity(3, 0.43, 0.6)
    const blue = rayleighScatterIntensity(3, 2.44, 0.6)
    expect(blue).toBeGreaterThan(red)
  })
})

describe('atmosphereScatterColor', () => {
  it('is blue-dominant (the limb tint), the direct consequence of the 1/wavelength^4 law', () => {
    const [r, , b] = atmosphereScatterColor(0.3, 0.6)
    expect(b).toBeGreaterThan(r)
  })

  it('brightens toward the limb (lower cosViewAngle) on every channel', () => {
    const nadir = atmosphereScatterColor(1, 0.6)
    const limb = atmosphereScatterColor(0.2, 0.6)
    for (let i = 0; i < 3; i++) expect(limb[i]!).toBeGreaterThan(nadir[i]!)
  })
})

describe('terminatorSunTransmittance', () => {
  // Production values (Matter.ts's ATMOSPHERE_DENSITY / TERMINATOR_SUN_PATH).
  const DENSITY = 0.14
  const PATH = 10

  it('is exactly unity away from the terminator, so the day-side limb is untouched', () => {
    for (const c of terminatorSunTransmittance(0, PATH, DENSITY)) expect(c).toBeCloseTo(1, 12)
  })

  it('removes light and never adds it — every channel stays within (0, 1]', () => {
    for (const warmth of [0, 0.1, 0.5, 0.9, 1]) {
      for (const c of terminatorSunTransmittance(warmth, PATH, DENSITY)) {
        expect(c).toBeGreaterThan(0)
        expect(c).toBeLessThanOrEqual(1)
      }
    }
  })

  it('strips blue fastest, then green, then red — which is WHY a terminator is orange', () => {
    const [r, g, b] = terminatorSunTransmittance(1, PATH, DENSITY)
    expect(r!).toBeGreaterThan(g!)
    expect(g!).toBeGreaterThan(b!)
  })

  it('is monotonically decreasing in warmth on every channel', () => {
    let previous = terminatorSunTransmittance(0, PATH, DENSITY)
    for (const warmth of [0.25, 0.5, 0.75, 1]) {
      const next = terminatorSunTransmittance(warmth, PATH, DENSITY)
      for (let i = 0; i < 3; i++) expect(next[i]!).toBeLessThan(previous[i]!)
      previous = next
    }
  })
})

// The bug this whole primitive exists to make impossible. The first version
// of the terminator warmth was `mix(scatterColor, TERMINATOR_GLOW_COLOR,
// warmth)`, which at the terminator (warmth == 1) returns the constant warm
// colour and DISCARDS scatterColor entirely — annihilating the view-angle
// dependence that is the entire physical basis of a limb glow. On screen that
// painted a hard-edged, uniform-brightness orange stripe straight across the
// disc, identical at the centre and at the limb, and roughly three times
// brighter than the terrain it crossed.
describe('the terminator glow keeps its view-angle dependence', () => {
  const DENSITY = 0.14
  const PATH = 10
  const STRENGTH = 0.16 // Matter.ts's ATMOSPHERE_STRENGTH
  const SIGMA = 0.06 // Matter.ts's TERMINATOR_GLOW_WIDTH
  /**
   * The material's full composition, not just the transmittance in
   * isolation — `sunAvailability` and `ATMOSPHERE_STRENGTH` both scale the
   * glow before it reaches the frame buffer, so a brightness assertion that
   * omits them is not comparing like with like against terrain radiance.
   */
  const glow = (cosViewAngle: number, ndotSun: number): Color3 => {
    const scatter = atmosphereScatterColor(cosViewAngle, DENSITY)
    const transmittance = terminatorSunTransmittance(terminatorWarmthWeight(ndotSun, SIGMA), PATH, DENSITY)
    const k = atmosphereSunAvailability(ndotSun, SIGMA) * STRENGTH
    return [
      scatter[0] * transmittance[0] * k,
      scatter[1] * transmittance[1] * k,
      scatter[2] * transmittance[2] * k,
    ]
  }
  const luminance = (c: Color3) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

  it('is far brighter at the limb than at the disc centre EVEN AT the terminator', () => {
    const centre = luminance(glow(1, 0))
    const limb = luminance(glow(0.1, 0))
    expect(limb).toBeGreaterThan(centre * 4)
  })

  it('is dimmer at the disc centre than the terrain it crosses, so no stripe rules across the face', () => {
    // The terrain it would be drawn over at the terminator: an ambient-lit
    // shoreline ocean texel, AMBIENT 0.10 on OCEAN_STOPS' 0 m stop. Under
    // the old mix-to-a-constant the glow was 0.050 here against terrain's
    // 0.018 — nearly 3x brighter, which is exactly what made it read as a
    // painted stripe rather than atmosphere.
    const ambientTerrain = luminance([0.13 * 0.1, 0.19 * 0.1, 0.23 * 0.1])
    expect(luminance(glow(1, 0))).toBeLessThan(ambientTerrain)
  })

  it('warms as it dims — the terminator limb is redder than the day-side limb', () => {
    const day = glow(0.1, 0.5) // well off the terminator: warmth ~ 0
    const terminator = glow(0.1, 0)
    expect(terminator[0]! / terminator[2]!).toBeGreaterThan(day[0]! / day[2]!)
    expect(terminator[0]!).toBeGreaterThan(terminator[2]!) // actually red-dominant, not merely less blue
  })
})

describe('terminatorWarmthWeight', () => {
  const deg = (d: number) => Math.sin((d * Math.PI) / 180)
  // Production sigma (Matter.ts's TERMINATOR_GLOW_WIDTH) — the band must be
  // "a few degrees" wide, not the ~40-50 degree wash an earlier, too-wide
  // (linear-tent) version produced.
  const SIGMA = 0.06

  it('peaks exactly at the terminator', () => {
    expect(terminatorWarmthWeight(0, SIGMA)).toBeCloseTo(1, 10)
  })

  it('is still strong within a couple degrees of the terminator', () => {
    expect(terminatorWarmthWeight(deg(2), SIGMA)).toBeGreaterThan(0.7)
  })

  it('is near zero by 10-15 degrees from the terminator (not a 40-50 degree wash)', () => {
    expect(terminatorWarmthWeight(deg(10), SIGMA)).toBeLessThan(0.02)
    expect(terminatorWarmthWeight(deg(15), SIGMA)).toBeLessThan(0.001)
  })

  it('is symmetric around the terminator', () => {
    expect(terminatorWarmthWeight(0.1, SIGMA)).toBeCloseTo(terminatorWarmthWeight(-0.1, SIGMA), 10)
  })

  it('falls off faster than the old linear tent shape would (steep Gaussian, not a ramp)', () => {
    // At 1 sigma out, a Gaussian (1/e ~ 0.368) is already well below what a
    // linear tent (clamp01(1 - x/width))^2 would give at the same point (0).
    // The real assertion that matters: by 3 sigma the Gaussian is
    // essentially gone, unlike a tent scaled to reach that far.
    expect(terminatorWarmthWeight(SIGMA, SIGMA)).toBeCloseTo(Math.exp(-1), 10)
    expect(terminatorWarmthWeight(3 * SIGMA, SIGMA)).toBeLessThan(0.0002)
  })
})

describe('atmosphereSunAvailability', () => {
  it('is full on the day side', () => {
    expect(atmosphereSunAvailability(1, 0.2, 0.15)).toBeCloseTo(1, 6)
  })

  it('eases down to the night floor on the deep night side', () => {
    expect(atmosphereSunAvailability(-1, 0.2, 0.15)).toBeCloseTo(0.15, 6)
  })

  it('is exactly halfway between floor and full at the terminator', () => {
    expect(atmosphereSunAvailability(0, 0.2, 0.15)).toBeCloseTo((1 + 0.15) / 2, 6)
  })
})

describe('schlickFresnel', () => {
  it('equals f0 at normal incidence', () => {
    expect(schlickFresnel(1, 0.02)).toBeCloseTo(0.02, 10)
  })

  it('rises to (near) 1 at a fully grazing angle', () => {
    expect(schlickFresnel(0, 0.02)).toBeCloseTo(1, 10)
  })

  it('is monotone increasing as the view angle grazes further', () => {
    expect(schlickFresnel(0.2, 0.02)).toBeGreaterThan(schlickFresnel(0.8, 0.02))
  })
})

describe('specularLobe', () => {
  it('peaks at ndotH = 1', () => {
    expect(specularLobe(1, 40)).toBeCloseTo(1, 10)
  })

  it('is 0 once the half-vector faces away from the surface', () => {
    expect(specularLobe(-0.1, 40)).toBe(0)
  })

  it('a higher shininess falls off faster away from the peak (narrower lobe)', () => {
    const broad = specularLobe(0.9, 10)
    const narrow = specularLobe(0.9, 100)
    expect(narrow).toBeLessThan(broad)
  })
})

describe('oceanSpecularIntensity', () => {
  it('combines the lobe and Fresnel multiplicatively, scaled by strength', () => {
    const lobe = specularLobe(0.95, 60)
    const fresnel = schlickFresnel(0.4, 0.02)
    expect(oceanSpecularIntensity(0.95, 0.4, 60, 0.02, 0.5)).toBeCloseTo(lobe * fresnel * 0.5, 10)
  })
})

describe('iceElevationLoweringM', () => {
  it('is 0 through the tropics and mid-latitudes', () => {
    expect(iceElevationLoweringM(0)).toBe(0)
    expect(iceElevationLoweringM(45)).toBe(0)
  })

  it('reaches (and stays at) the full lowering by the poles', () => {
    expect(iceElevationLoweringM(78)).toBeCloseTo(7000, 6)
    expect(iceElevationLoweringM(90)).toBeCloseTo(7000, 6)
  })

  it('is symmetric between hemispheres', () => {
    expect(iceElevationLoweringM(70)).toBeCloseTo(iceElevationLoweringM(-70), 10)
  })

  it('increases continuously (no hard threshold) through the sub-polar belt', () => {
    const a = iceElevationLoweringM(60)
    const b = iceElevationLoweringM(65)
    const c = iceElevationLoweringM(70)
    expect(b).toBeGreaterThan(a)
    expect(c).toBeGreaterThan(b)
  })
})

describe('hypsometricColor with latitude (polar ice)', () => {
  it('defaults to no latitude effect, so every existing 2-arg call site is unchanged', () => {
    expect(hypsometricColor(100, 1)).toEqual(landColor(100))
  })

  it('a low-elevation polar point reads pale like a high mountain at the equator (ice-whitened, then the high-latitude cool cast on top)', () => {
    const polar = hypsometricColor(200, 1, 85)
    for (const c of polar) expect(c).toBeGreaterThan(0.6)
  })

  it('the same low elevation is untouched by latitude at the equator', () => {
    expect(hypsometricColor(200, 1, 0)).toEqual(landColor(200))
  })

  it('blends continuously with latitude, not a hard threshold', () => {
    const c60 = hypsometricColor(200, 1, 60)
    const c70 = hypsometricColor(200, 1, 70)
    const c80 = hypsometricColor(200, 1, 80)
    // Monotonically whitening (every channel non-decreasing) as latitude
    // rises through the sub-polar belt.
    for (let i = 0; i < 3; i++) {
      expect(c70[i]!).toBeGreaterThanOrEqual(c60[i]!)
      expect(c80[i]!).toBeGreaterThanOrEqual(c70[i]!)
    }
  })
})

describe('vegetationIndex', () => {
  it('is 0 at and below the arid threshold', () => {
    expect(vegetationIndex(VEGETATION_ARID_LOG_TAU)).toBe(0)
    expect(vegetationIndex(VEGETATION_ARID_LOG_TAU - 5)).toBe(0)
  })

  it('is 1 at and above the lush threshold', () => {
    expect(vegetationIndex(VEGETATION_LUSH_LOG_TAU)).toBe(1)
    expect(vegetationIndex(VEGETATION_LUSH_LOG_TAU + 5)).toBe(1)
  })

  it('is a smooth (not hard-thresholded) gradient in between', () => {
    const mid = (VEGETATION_ARID_LOG_TAU + VEGETATION_LUSH_LOG_TAU) / 2
    const v = vegetationIndex(mid)
    expect(v).toBeGreaterThan(0)
    expect(v).toBeLessThan(1)
    expect(vegetationIndex(mid - 0.5)).toBeLessThan(v)
    expect(vegetationIndex(mid + 0.5)).toBeGreaterThan(v)
  })
})

describe('landColor with vegetation', () => {
  it('is exactly the arid ramp at vegetationIndex 0 and the lush ramp at 1', () => {
    const arid0 = landColor(0, 0)
    const lush0 = landColor(0, 1)
    expect(arid0).not.toEqual(lush0)
    // Arid reads warmer/redder than lush at the same low elevation.
    expect(arid0[0]).toBeGreaterThan(lush0[0])
  })

  it('converges to the SAME colour regardless of vegetation once elevation is high enough (rock/snow, not biome)', () => {
    const aridHigh = landColor(5000, 0)
    const lushHigh = landColor(5000, 1)
    expect(aridHigh).toEqual(lushHigh)
  })

  it('blends continuously between the two ramps', () => {
    const arid = landColor(0, 0)
    const lush = landColor(0, 1)
    const mid = landColor(0, 0.5)
    for (let i = 0; i < 3; i++) {
      const lo = Math.min(arid[i]!, lush[i]!)
      const hi = Math.max(arid[i]!, lush[i]!)
      expect(mid[i]!).toBeGreaterThanOrEqual(lo)
      expect(mid[i]!).toBeLessThanOrEqual(hi)
    }
  })
})

describe('highLatitudeCoolness', () => {
  it('is 0 through the tropics and temperate latitudes', () => {
    expect(highLatitudeCoolness(0)).toBe(0)
    expect(highLatitudeCoolness(35)).toBe(0)
  })

  it('reaches (and stays at) its max by the sub-polar belt', () => {
    expect(highLatitudeCoolness(62)).toBeCloseTo(0.3, 6)
    expect(highLatitudeCoolness(85)).toBeCloseTo(0.3, 6)
  })

  it('is well below the ice band\'s own effect - a smaller, independent term, not a hard threshold', () => {
    const a = highLatitudeCoolness(45)
    const b = highLatitudeCoolness(50)
    const c = highLatitudeCoolness(55)
    expect(b).toBeGreaterThan(a)
    expect(c).toBeGreaterThan(b)
    expect(c).toBeLessThanOrEqual(0.3)
  })

  it('is symmetric between hemispheres', () => {
    expect(highLatitudeCoolness(50)).toBeCloseTo(highLatitudeCoolness(-50), 10)
  })
})

describe('detailFadeByDistance', () => {
  it('is 1 (full detail) at and below the full-strength distance', () => {
    expect(detailFadeByDistance(0, 30_000, 300_000)).toBe(1)
    expect(detailFadeByDistance(30_000, 30_000, 300_000)).toBe(1)
  })

  it('is 0 (no detail) at and beyond the fade-out distance', () => {
    expect(detailFadeByDistance(300_000, 30_000, 300_000)).toBe(0)
    expect(detailFadeByDistance(1_000_000, 30_000, 300_000)).toBe(0)
  })

  it('is monotone decreasing with distance in between', () => {
    const near = detailFadeByDistance(50_000, 30_000, 300_000)
    const far = detailFadeByDistance(150_000, 30_000, 300_000)
    expect(near).toBeGreaterThan(far)
    expect(far).toBeGreaterThan(0)
  })
})

describe('decodeBuiltUpFraction', () => {
  it('round-trips a raw 0..1 fraction through half-float', () => {
    for (const f of [0, 0.001, 0.05, 0.5, 1]) {
      const bits = DataUtils.toHalfFloat(f)
      expect(decodeBuiltUpFraction(bits)).toBeCloseTo(f, 2)
    }
  })
})

describe('nightFactor', () => {
  it('is 0 on the day side and 1 on the night side', () => {
    expect(nightFactor(1)).toBe(0)
    expect(nightFactor(-1)).toBe(1)
  })

  it('is 0.5 exactly at the terminator (ndotSun = 0)', () => {
    expect(nightFactor(0)).toBeCloseTo(0.5, 10)
  })

  it('is monotone non-decreasing as ndotSun decreases (day -> night)', () => {
    const samples = [1, 0.5, 0.2, 0.05, 0, -0.05, -0.2, -0.5, -1].map((n) => nightFactor(n))
    for (let i = 1; i < samples.length; i++) {
      expect(samples[i]!).toBeGreaterThanOrEqual(samples[i - 1]!)
    }
  })

  it('a wider terminator softens the transition further from the equator line', () => {
    // Just inside a wide terminator band but outside a narrow one: narrow
    // reads full night (1), wide is still transitioning (< 1).
    const narrow = nightFactor(-0.05, 0.02)
    const wide = nightFactor(-0.05, 0.2)
    expect(narrow).toBe(1)
    expect(wide).toBeLessThan(1)
  })
})

describe('cityLightIntensity', () => {
  it('is 0 at zero built-up fraction and 1 at fully built-up', () => {
    expect(cityLightIntensity(0)).toBe(0)
    expect(cityLightIntensity(1)).toBeCloseTo(1, 10)
  })

  it('lifts faint fractions well above a linear response (gamma < 1)', () => {
    const faint = 0.01
    expect(cityLightIntensity(faint)).toBeGreaterThan(faint)
  })

  it('is monotone increasing in built-up fraction', () => {
    const a = cityLightIntensity(0.01)
    const b = cityLightIntensity(0.1)
    const c = cityLightIntensity(0.5)
    expect(a).toBeLessThan(b)
    expect(b).toBeLessThan(c)
  })

  it('clamps out-of-range input rather than producing NaN/negative output', () => {
    expect(cityLightIntensity(-0.5)).toBe(0)
    expect(cityLightIntensity(1.5)).toBeCloseTo(1, 10)
  })
})

// AMBIENT stands for sky and bounce light, and both of those ARE sunlight —
// they just took an extra scatter to get here. Applying the full value on
// the unlit hemisphere lit the night side to a readable dim daytime, so
// Australia's tan and south-east Asia's green competed with the very city
// lights they were supposed to be a backdrop for.
describe('twilightAmbientScale', () => {
  const deg = (d: number) => Math.sin((d * Math.PI) / 180)
  const FLOOR = 0.08

  it('is full strength across the whole day side', () => {
    for (const ndotSun of [1, 0.5, 0.2, 0.01]) {
      expect(twilightAmbientScale(ndotSun, FLOOR)).toBeCloseTo(1, 10)
    }
  })

  it('has not collapsed by civil twilight — dusk is a real, gradual thing', () => {
    expect(twilightAmbientScale(-deg(6), FLOOR)).toBeGreaterThan(0.4)
  })

  it('reaches the floor by the end of astronomical twilight, ~18 degrees down', () => {
    expect(twilightAmbientScale(-deg(18), FLOOR)).toBeCloseTo(FLOOR, 10)
    expect(twilightAmbientScale(-1, FLOOR)).toBeCloseTo(FLOOR, 10)
  })

  it('keeps a floor rather than going to zero — airglow and starlight are real', () => {
    expect(twilightAmbientScale(-1, FLOOR)).toBeGreaterThan(0)
  })

  it('is monotone non-increasing from day into night', () => {
    const samples = [1, 0.3, 0.05, 0, -0.05, -0.15, -0.31, -0.6, -1]
      .map((n) => twilightAmbientScale(n, FLOOR))
    for (let i = 1; i < samples.length; i++) {
      expect(samples[i]!).toBeLessThanOrEqual(samples[i - 1]!)
    }
  })
})

// --- Uncertainty overlay (`u`) ---------------------------------------------
//
// Channel B of `cube_plain_128` is the per-pixel bootstrap coefficient of
// variation from van den Hoogen et al. 2019 (100 stratified bootstrap
// iterations). Every constant these tests pin was MEASURED off the baked
// atlas rather than guessed — see terrain.ts's own section doc for the full
// distribution and how it was taken.
//
// The recurring failure mode in this project has been an instrument that
// reads the same whether the feature works or is entirely disconnected, so
// each test below is written to fail against a plausible broken version:
// a ramp that ignores its input, an ocean texel painted as "certain", a
// coastal texel trusted at face value, endpoints left at [0,1].
describe('uncertaintyColor: the ramp itself', () => {
  it('is anchored at the MEASURED land distribution, not at [0,1]', () => {
    // Land texels (A >= 0.99, n=19885) of cube_plain_128 span CV
    // 0.0916..0.5034, p05 0.1216, p50 0.1847, p95 0.2917. A [0,1] ramp
    // would put the entire planet inside its bottom third and read as
    // uniformly certain.
    expect(UNCERTAINTY_CV_LOW).toBeGreaterThan(0.0)
    expect(UNCERTAINTY_CV_LOW).toBeLessThan(0.12) // at or below the measured p05
    expect(UNCERTAINTY_CV_HIGH).toBeGreaterThan(0.29) // at or above the measured p95
    expect(UNCERTAINTY_CV_HIGH).toBeLessThan(0.55) // but not stretched to the lone max
  })

  it('actually depends on its input — a constant ramp would pass every other test here', () => {
    // The disconnected-instrument check: if uncertaintyColor ignored `cv`
    // entirely (returned a fixed colour), every "ocean differs from land"
    // assertion below would still pass. This one would not.
    const seen = new Set<string>()
    for (const cv of [0.09, 0.14, 0.19, 0.24, 0.29, 0.35]) seen.add(uncertaintyColor(cv).join(','))
    expect(seen.size).toBe(6)
  })

  it('rises monotonically in luminance across the whole ramp — brighter means less certain', () => {
    // A perceptually ORDERED ramp (the brief forbids rainbows: they invent
    // boundaries the data does not have). Monotone luma is what makes the
    // ordering readable without consulting the legend.
    const luma = (c: Color3): number => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    let prev = -Infinity
    for (let i = 0; i <= 24; i++) {
      const cv = UNCERTAINTY_CV_LOW + ((UNCERTAINTY_CV_HIGH - UNCERTAINTY_CV_LOW) * i) / 24
      const l = luma(uncertaintyColor(cv))
      expect(l).toBeGreaterThan(prev)
      prev = l
    }
  })

  it('clamps outside the measured range rather than extrapolating to nonsense', () => {
    expect(uncertaintyColor(-1)).toEqual(uncertaintyColor(UNCERTAINTY_CV_LOW))
    expect(uncertaintyColor(10)).toEqual(uncertaintyColor(UNCERTAINTY_CV_HIGH))
    for (const c of [...uncertaintyColor(-1), ...uncertaintyColor(10)]) {
      expect(c).toBeGreaterThanOrEqual(0)
      expect(c).toBeLessThanOrEqual(1)
    }
  })

  it('stays warm — it must not read as more veil, which is pale blue', () => {
    // regimes/Shell.ts tints the veil vec3(0.72, 0.86, 0.95). An overlay in
    // the same family would be invisible against it at any exposure.
    for (const cv of [0.12, 0.18, 0.29, 0.35]) {
      const [r, , b] = uncertaintyColor(cv)
      expect(r).toBeGreaterThan(b)
    }
  })
})

describe('uncertaintyCoverage: where the model actually has a domain', () => {
  it('is exactly zero over open ocean — there is no soil-nematode model there', () => {
    expect(uncertaintyCoverage(0)).toBe(0)
  })

  it('is full over solid land', () => {
    expect(uncertaintyCoverage(1)).toBeCloseTo(1, 10)
  })

  it('fades out partial-land texels, whose CV is measurably diluted', () => {
    // Measured off the atlas, CV by land-fraction band (p50):
    //   A<0.001  0.0000 | 0.001-0.1  0.0255 | 0.1-0.3  0.0793
    //   0.3-0.5  0.1238 | 0.5-0.7  0.1502   | 0.9-1.0  0.1821
    // A p50 of 0.0255 is far below the 0.0916 minimum seen on solid land,
    // so those texels carry ocean mixed into the average, not a real
    // low-uncertainty measurement. Painting them at face value would draw a
    // bright "well-measured" fringe around every coastline.
    expect(uncertaintyCoverage(0.05)).toBe(0)
    expect(uncertaintyCoverage(0.3)).toBeLessThan(uncertaintyCoverage(0.7))
  })

  it('is monotone non-decreasing in land fraction', () => {
    let prev = -Infinity
    for (let a = 0; a <= 1.0001; a += 0.05) {
      const c = uncertaintyCoverage(a)
      expect(c).toBeGreaterThanOrEqual(prev)
      prev = c
    }
  })
})

// THE hard requirement: ocean must not read as certain. Channel A is land
// fraction; where there is no land there is no model, so CV there is not
// low, it is ABSENT. Channel B is ~0 over ocean (measured: 67520 of 67520
// ocean texels at or near 0), which is exactly the value a naive ramp maps
// to its most-certain colour — asserting the ocean is the best-measured
// place on Earth, which is backwards.
describe('the ocean is excluded from the overlay, not painted as certain', () => {
  const BASE: Color3 = [0.72, 0.86, 0.95] // the veil's own tint

  it('does not map an ocean texel to the same output as a low-CV land texel', () => {
    // Ocean as it actually sits in the atlas: land fraction 0, B channel 0.
    const ocean = uncertaintyOverlayColor(BASE, 0, 0, 1)
    // The most-certain real land in the atlas (measured minimum, 0.0916).
    const certainLand = uncertaintyOverlayColor(BASE, 0.0916, 1, 1)
    expect(ocean).not.toEqual(certainLand)
    // Not merely unequal — visibly so, on a channel that survives the dark
    // palette. Two colours differing in the 3rd decimal would "pass" while
    // being indistinguishable on screen.
    const dist = Math.hypot(
      ocean[0] - certainLand[0], ocean[1] - certainLand[1], ocean[2] - certainLand[2],
    )
    expect(dist).toBeGreaterThan(0.25)
  })

  it('leaves ocean exactly as it was — unpainted, not recoloured to a neutral', () => {
    expect(uncertaintyOverlayColor(BASE, 0, 0, 1)).toEqual(BASE)
    // ...and it stays unpainted regardless of what the B channel happens to
    // hold there. 27 ocean texels carry nonzero B (up to 0.1034) as
    // resampling residue; none of them is a measurement.
    expect(uncertaintyOverlayColor(BASE, 0.1034, 0, 1)).toEqual(BASE)
    expect(uncertaintyOverlayColor(BASE, 0.5, 0, 1)).toEqual(BASE)
  })

  it('paints land, so "ocean is untouched" is not passing by doing nothing at all', () => {
    // The complement of the check above: if uncertaintyOverlayColor were a
    // no-op everywhere, both ocean assertions would pass. This fails then.
    expect(uncertaintyOverlayColor(BASE, 0.28, 1, 1)).not.toEqual(BASE)
  })

  it('is a no-op at zero strength, so the toggle can be a uniform rather than a rebuild', () => {
    expect(uncertaintyOverlayColor(BASE, 0.28, 1, 0)).toEqual(BASE)
  })
})

// The overlay's whole claim is that it shows where the model is guessing.
// If it did not reproduce the ordering that is actually in the data, it
// would be decoration. These are real probes of cube_plain_128, taken with
// the shipped directionToFaceUV mapping.
describe('the overlay reproduces the orderings actually present in the data', () => {
  const BASE: Color3 = [0.72, 0.86, 0.95]
  const luma = (c: Color3): number => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
  const at = (cv: number): number => luma(uncertaintyOverlayColor(BASE, cv, 1, 1))

  it('renders the arid interiors brighter than the densely-sampled temperate zone', () => {
    // Measured mean CV over land texels (A >= 0.9), by region:
    //   Gobi/Alxa 0.3323 | Borneo 0.3413 | Australia arid 0.2622
    //   Sahara 0.2592 | Arabian 0.2575 | Amazon 0.2271 | Congo 0.2025
    //   W. Europe 0.1297 | US cornbelt 0.1284
    const sahara = at(0.2592)
    const gobi = at(0.3323)
    const europe = at(0.1297)
    const cornbelt = at(0.1284)
    expect(sahara).toBeGreaterThan(europe)
    expect(sahara).toBeGreaterThan(cornbelt)
    expect(gobi).toBeGreaterThan(sahara)
    // and by a margin that is actually visible, not a rounding difference
    expect(sahara - europe).toBeGreaterThan(0.1)
  })

  it('separates the single most-certain and least-certain land texels measured', () => {
    expect(at(0.5034)).toBeGreaterThan(at(0.0736))
  })
})

describe('formatUncertaintyCv: the legend needs real units, or the ramp says nothing', () => {
  it('renders a coefficient of variation as the percentage spread it is', () => {
    expect(formatUncertaintyCv(0.09)).toBe('9%')
    expect(formatUncertaintyCv(0.35)).toBe('35%')
    expect(formatUncertaintyCv(0.1847)).toBe('18%')
  })
})
