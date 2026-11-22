/**
 * Frame-time accumulator.
 *
 * Chrome quantizes GPU timestamp results to 100 us as a timing-attack
 * mitigation, so a 2 ms pass lands in only ~20 buckets. Any adaptive controller
 * must therefore average ACROSS frames rather than react per frame.
 * Spec section 10.
 */
export class FrameTimer {
  private buf: number[] = []
  constructor(private readonly cap = 240) {}

  push(ms: number): void {
    if (!Number.isFinite(ms)) return
    this.buf.push(ms)
    if (this.buf.length > this.cap) this.buf.shift()
  }

  get samples(): number { return this.buf.length }

  private quantile(q: number): number {
    if (this.buf.length === 0) return NaN
    const s = [...this.buf].sort((a, b) => a - b)
    const i = Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))
    return s[i]!
  }

  get p05(): number { return this.quantile(0.05) }
  get median(): number { return this.quantile(0.5) }
  get p95(): number { return this.quantile(0.95) }

  /**
   * Population standard deviation of the collected samples - a second,
   * distribution-shape-independent variance figure alongside p05/p95, so a
   * measurement whose spread is large is visible as such rather than
   * collapsing to a median that looks authoritative on its own.
   */
  get stddev(): number {
    if (this.buf.length === 0) return NaN
    const mean = this.buf.reduce((a, b) => a + b, 0) / this.buf.length
    const variance = this.buf.reduce((a, b) => a + (b - mean) ** 2, 0) / this.buf.length
    return Math.sqrt(variance)
  }
}
