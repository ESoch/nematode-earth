import { describe, it, expect, vi } from 'vitest'
import { runSequentially, type Runner, type RunAllProgress } from '../src/experiments/report'
import type { ExpResult } from '../src/experiments/types'

// The Session 1 harness's actual runners all need a live WebGPU adapter (no
// headless implementation in this environment - see webgpu.test.ts), so the
// real `runAll` is untestable end to end here. `runSequentially` is the
// sequencing/progress/error-handling logic factored OUT of that constraint:
// pure control flow over an injectable Runner[], exercised with fake runners
// below. This is what makes the top-level "which of the four experiments is
// running right now" visible at all - before this, main.ts's <pre> stayed
// on 'Running four experiments…' until every runner had resolved, which is
// EXACTLY what made a real hang indistinguishable from a slow-but-working
// run (the team lead's report: "no console errors, no partial output, no
// exception - the page just never finishes").

const ok = (id: string): ExpResult => ({ id, pass: true, detail: `${id} ok` })

describe('runSequentially', () => {
  it('runs runners in order and collects their results in order', async () => {
    const calls: string[] = []
    const runners: Runner[] = [
      { id: 'a', run: async () => { calls.push('a'); return ok('a') } },
      { id: 'b', run: async () => { calls.push('b'); return ok('b') } },
      { id: 'c', run: async () => { calls.push('c'); return ok('c') } },
    ]
    const results = await runSequentially(runners)
    expect(calls).toEqual(['a', 'b', 'c'])
    expect(results.map((r) => r.id)).toEqual(['a', 'b', 'c'])
  })

  it('runs sequentially, not concurrently - a slow runner delays the next one starting', async () => {
    // Mirrors report.ts's own doc comment on why runAll never uses
    // Promise.all: concurrent renderers would contend for the GPU. Proven
    // here by order of side effects, not by timing (timing assertions on
    // real clocks are flaky) - if runner b started before a's slow promise
    // resolved, 'b-start' would appear before 'a-done' in the log.
    const log: string[] = []
    let resolveA!: () => void
    const runners: Runner[] = [
      {
        id: 'a',
        run: () => new Promise<ExpResult>((resolve) => {
          resolveA = () => { log.push('a-done'); resolve(ok('a')) }
        }),
      },
      { id: 'b', run: async () => { log.push('b-start'); return ok('b') } },
    ]
    const promise = runSequentially(runners)
    await Promise.resolve() // let the loop reach and await runner a
    expect(log).toEqual([])
    resolveA()
    await promise
    expect(log).toEqual(['a-done', 'b-start'])
  })

  it("a runner that throws does not stop the sequence - matches runAll's per-runner try/catch", async () => {
    const runners: Runner[] = [
      { id: 'a', run: async () => ok('a') },
      { id: 'b', run: async () => { throw new Error('boom') } },
      { id: 'c', run: async () => ok('c') },
    ]
    const results = await runSequentially(runners)
    expect(results.map((r) => r.id)).toEqual(['a', 'b', 'c'])
    expect(results[1]!.pass).toBe(false)
    expect(results[1]!.detail).toMatch(/Threw: boom/)
    expect(results[2]!.pass).toBe(true) // c still ran after b's throw
  })

  it("a runner that throws a non-Error value still produces a result, not an unhandled rejection", () => {
    const runners: Runner[] = [{ id: 'a', run: async () => { throw 'not an Error object' } }]
    return expect(runSequentially(runners)).resolves.toMatchObject([
      { id: 'a', pass: false, detail: 'Threw: not an Error object' },
    ])
  })

  it('reports start before done for every runner, in order, when a progress callback is given', async () => {
    const events: RunAllProgress[] = []
    const runners: Runner[] = [
      { id: 'a', run: async () => ok('a') },
      { id: 'b', run: async () => { throw new Error('nope') } },
    ]
    await runSequentially(runners, (p) => events.push(p))
    expect(events).toEqual([
      { id: 'a', status: 'start' },
      { id: 'a', status: 'done' },
      { id: 'b', status: 'start' },
      { id: 'b', status: 'error' },
    ])
  })

  it('omitting the progress callback does not throw - it is optional, not required', async () => {
    const runners: Runner[] = [{ id: 'a', run: async () => ok('a') }]
    await expect(runSequentially(runners)).resolves.toBeDefined()
  })

  it('an empty runner list resolves to an empty result array without hanging', async () => {
    await expect(runSequentially([])).resolves.toEqual([])
  })

  it('a progress callback that throws does not corrupt the run - the result for that runner is still recorded', async () => {
    // A defensive property, not an expected use: a buggy onProgress (e.g. a
    // DOM update that throws because the element was removed) must not turn
    // an otherwise-working experiment into a silently missing result - the
    // whole point of this function is to make the harness MORE resilient to
    // exactly this class of "silently never finishes" failure, not to add a
    // new way to trigger it. So onProgress errors are swallowed, not
    // propagated: the real experiment result still comes back.
    const badProgress = vi.fn(() => { throw new Error('progress handler bug') })
    const runners: Runner[] = [{ id: 'a', run: async () => ok('a') }]
    const results = await runSequentially(runners, badProgress)
    expect(results).toEqual([ok('a')])
    expect(badProgress).toHaveBeenCalled()
  })

  // The bug that motivated all of this: exp-09 sat on an unresolved WebGPU
  // promise (GPU-process-level contention with a second tab's own live
  // renderer - confirmed by direct reproduction, not guessed at) and NOTHING
  // in the harness ever gave up on it, so main.ts's <pre> read "Running four
  // experiments…" forever with no way to tell hung from slow. A runner that
  // never settles is exactly the case a real GPU stall produces - these
  // tests use a promise that intentionally never resolves, standing in for
  // that stall, real timers (not vi.useFakeTimers - a stuck native WebGPU
  // promise wouldn't respect fake timers either, so the test should exercise
  // the same real setTimeout race the production code does).
  describe('timeoutMs: turning a hang into a loud, recoverable failure', () => {
    it('a runner that never resolves times out rather than hanging the whole sequence forever', async () => {
      const runners: Runner[] = [
        { id: 'stuck', run: () => new Promise<ExpResult>(() => { /* never settles */ }) },
      ]
      const results = await runSequentially(runners, undefined, 20)
      expect(results).toHaveLength(1)
      expect(results[0]!.pass).toBe(false)
      expect(results[0]!.detail).toMatch(/timed out after 20 ms/)
    })

    it('a runner that resolves before the timeout is unaffected - no false failure from a merely-close race', async () => {
      const runners: Runner[] = [
        { id: 'fast-enough', run: () => new Promise<ExpResult>((resolve) => setTimeout(() => resolve(ok('fast-enough')), 5)) },
      ]
      const results = await runSequentially(runners, undefined, 200)
      expect(results).toEqual([ok('fast-enough')])
    })

    it('a timed-out runner does not block the next one from starting and completing', async () => {
      const events: RunAllProgress[] = []
      const runners: Runner[] = [
        { id: 'stuck', run: () => new Promise<ExpResult>(() => {}) },
        { id: 'fine', run: async () => ok('fine') },
      ]
      const results = await runSequentially(runners, (p) => events.push(p), 15)
      expect(results.map((r) => r.pass)).toEqual([false, true])
      expect(events.map((e) => `${e.id}:${e.status}`)).toEqual(['stuck:start', 'stuck:error', 'fine:start', 'fine:done'])
    })

    it('omitting timeoutMs applies no timeout at all - a genuinely slow-but-working runner still completes', async () => {
      // Not "a very large timeout" - actually none. A caller that does not
      // opt in (report.test.ts's other tests, none of which pass this arg)
      // must see byte-for-byte the same untimed behaviour as before this
      // parameter existed.
      const runners: Runner[] = [
        { id: 'slow', run: () => new Promise<ExpResult>((resolve) => setTimeout(() => resolve(ok('slow')), 30)) },
      ]
      const results = await runSequentially(runners)
      expect(results).toEqual([ok('slow')])
    })
  })
})
