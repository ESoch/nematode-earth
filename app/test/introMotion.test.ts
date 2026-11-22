import { describe, expect, it, vi } from 'vitest'

// Vite processes CSS imports into an empty module in tests; read the actual
// stylesheet through Vitest without adding Node types to the browser tsconfig.
const { readFileSync } = await vi.importActual<{
  readFileSync(path: URL, encoding: 'utf8'): string
}>('node:fs')
const css = readFileSync(new URL('../src/hud/viewer.css', import.meta.url), 'utf8')

describe('opening quote motion', () => {
  it('settles once without hiding the premise, and is static for reduced motion', () => {
    expect(css).toMatch(/@keyframes cobb-quote-arrive\s*\{/)
    expect(css).toMatch(/\.sweep-intro-quote\s*\{[^}]*animation:\s*cobb-quote-arrive\s+\d+ms/s)
    expect(css).toMatch(/@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{[^}]*\.sweep-intro-quote\s*\{\s*animation:\s*none/s)
  })
})
