import { describe, it, expect } from 'vitest'
import { vec3 } from 'three/tsl'
import {
  createDissolveUniforms, erosionPhiNode, layerAlphaNode, matterAlphaNode,
} from '../src/sweep/dissolveNodes'
import { LAYERS, DEFAULT_SEED } from '../src/sweep/Dissolve'
// Vite's `?raw` suffix (typed by vite/client, already in tsconfig's `types`)
// imports the file's own text - no `node:fs` needed, so no @types/node
// dependency this project doesn't otherwise have.
import dissolveNodesSource from '../src/sweep/dissolveNodes.ts?raw'

describe('dissolve uniforms', () => {
  it('defaults to the CPU seed and to noise enabled', () => {
    const u = createDissolveUniforms()
    expect(u.seed.value.toArray()).toEqual([...DEFAULT_SEED])
    expect(u.front.value).toBe(0)
    expect(u.noiseEnabled.value).toBe(1)
  })
  it('accepts an explicit seed', () => {
    const u = createDissolveUniforms([0, 0, 1])
    expect(u.seed.value.toArray()).toEqual([0, 0, 1])
  })
})

describe('node construction', () => {
  const u = createDissolveUniforms()
  const dir = vec3(0, 0, 1)
  it('builds a phi node for every layer without throwing', () => {
    for (const name of ['atmosphere', 'water', 'rock'] as const) {
      expect(() => erosionPhiNode(dir, LAYERS[name], u)).not.toThrow()
    }
  })
  it('builds the alpha node', () => {
    expect(() => layerAlphaNode(erosionPhiNode(dir, LAYERS.rock, u), u.front, LAYERS.rock.width)).not.toThrow()
  })
  it('builds the composed matter alpha for every layer', () => {
    for (const name of ['atmosphere', 'water', 'rock'] as const) {
      expect(() => matterAlphaNode(dir, name, u)).not.toThrow()
    }
  })
})

describe('the shader front algebra matches the CPU front algebra', () => {
  it('uses the same -2w .. 1+4w travel and +-w band', () => {
    expect(dissolveNodesSource).toMatch(/-2\s*\*\s*width/)
    expect(dissolveNodesSource).toMatch(/1\s*\+\s*4\s*\*\s*width/)
    expect(dissolveNodesSource).toMatch(/smoothstep\(/)
  })
  it('never samples a texture, so it needs no explicit LOD', () => {
    expect(dissolveNodesSource).not.toMatch(/\.sample\(/)
    expect(dissolveNodesSource).not.toMatch(/\.level\(/)
  })
})
