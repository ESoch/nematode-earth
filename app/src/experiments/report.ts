import { probeAdapter } from '../platform/webgpu'
import type { AdapterReport } from '../platform/webgpu'
import { runExp01 } from './exp01-march-cost'
import { runExp02 } from './exp02-wgsl'
import { runExp04 } from './exp04-storage-texture'
import { runExp09 } from './exp09-reverse-z'
import type { ExpResult } from './types'

export interface Runner {
  id: string
  run: () => Promise<ExpResult>
}

export interface RunAllProgress {
  id: string
  status: 'start' | 'done' | 'error'
}

/**
 * Races `promise` against a `timeoutMs` timer. Confirmed necessary, not
 * defensive-by-assumption: this is the exact fix for a REPRODUCED hang - an
 * always-on second tab (the interactive viewer, `setAnimationLoop` never
 * stopping) holding its own live WebGPU device left exp-09's very first
 * `renderer.renderAsync`/`readRenderTargetPixelsAsync` round-trip permanently
 * unresolved when both tabs contended for the GPU at once (closing the
 * second tab made exp-09 complete in 0.2s instead). The timer is always
 * cleared on either branch so it can't itself leak past this call.
 *
 * IMPORTANT LIMITATION: this stops the HARNESS from waiting, it does not
 * (and in plain JS cannot) cancel or free whatever the stuck runner is
 * still holding onto - `run()`'s own promise chain, and whatever GPU
 * resources it never got to `dispose()` in its `finally` block, keep
 * existing in the background after `withTimeout` gives up on them. A timed-
 * out runner therefore trades "the whole harness is unusable forever" for
 * "the harness reports what happened and the REMAINING runners still get a
 * chance" - real progress, not a full fix for the underlying contention.
 */
function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(
        `timed out after ${timeoutMs} ms (never resolved - see withTimeout's doc: this can happen when `
        + 'another tab holds its own live WebGPU device, e.g. the interactive viewer left open and still '
        + 'rendering; try again with that tab closed)',
      ))
    }, timeoutMs)
    promise.then(
      (value) => { clearTimeout(timer); resolve(value) },
      (err: unknown) => { clearTimeout(timer); reject(err) },
    )
  })
}

/**
 * Runs `runners` sequentially (never `Promise.all` — see `runAll`'s doc for
 * why), collecting one `ExpResult` per runner and reporting `onProgress`
 * before/after each. Factored out of `runAll` specifically so this
 * sequencing/error-handling logic is testable with fake runners: the real
 * ones all need a live WebGPU adapter, which this test environment does not
 * have (see `webgpu.test.ts`), so `runAll` itself cannot be exercised
 * end-to-end here — see `report.test.ts`.
 *
 * Before this existed, nothing observed the run in progress at all: `main.ts`
 * writes `'Running four experiments…'` once and does not touch the DOM again
 * until `runAll()`'s whole promise settles, so a genuinely hung runner and a
 * slow-but-working one were INDISTINGUISHABLE from outside — exactly the
 * failure mode a falsification harness must not have ("the harness should
 * either work or say why it cannot," not sit there forever with no signal).
 * `onProgress` is what `main.ts` now uses to update the page as each of the
 * four experiments starts and finishes, and it is deliberately isolated in
 * its own try/catch (a bug in a progress handler — e.g. a DOM update against
 * an element that no longer exists — must not derail the actual experiment
 * results, which are what this harness exists to produce).
 *
 * `timeoutMs`, when given, wraps EACH runner in `withTimeout` (see its doc
 * for the reproduced failure this answers) so one hung runner reports a
 * loud, specific failure and lets the rest of the sequence proceed, instead
 * of the whole page sitting on "Running four experiments…" forever. Omitted
 * entirely (as every call site before this parameter existed, and every
 * `report.test.ts` case that doesn't pass it), behaviour is untimed and
 * byte-for-byte unchanged.
 */
export async function runSequentially(
  runners: readonly Runner[],
  onProgress?: (p: RunAllProgress) => void,
  timeoutMs?: number,
): Promise<ExpResult[]> {
  const report = (p: RunAllProgress): void => {
    try {
      onProgress?.(p)
    } catch (err) {
      console.error('runSequentially: onProgress threw', err)
    }
  }

  const results: ExpResult[] = []
  for (const { id, run } of runners) {
    report({ id, status: 'start' })
    try {
      const result = timeoutMs !== undefined ? await withTimeout(run(), timeoutMs) : await run()
      results.push(result)
      report({ id, status: 'done' })
    } catch (err) {
      results.push({ id, pass: false, detail: `Threw: ${err instanceof Error ? err.message : String(err)}` })
      report({ id, status: 'error' })
    }
  }
  return results
}

// Generous, not tight: exp-01 (the slowest of the four - 240 frames, four
// full material compiles, forced-sync readback every frame) measured a
// contention-free ~16s end to end in the environment this was written and
// verified in; the other three finish in well under a second. 60s leaves
// wide headroom for slower hardware while still bounding a genuine hang
// (the reproduced failure this whole mechanism exists to catch - see
// withTimeout's doc) to "reports and moves on" rather than "forever".
const RUNNER_TIMEOUT_MS = 60_000

/**
 * Consolidates the four Session 1 experiment harnesses into one report.
 *
 * probeAdapter() is wrapped in its own try/catch, separate from the
 * per-experiment loop below: gpu.requestAdapter() can THROW (not just
 * resolve null) on some drivers, and probeAdapter has no internal guard
 * against that. If that throw were allowed to propagate out of runAll, it
 * would kill the entire report before a single experiment ran — the one
 * output this harness exists to produce. On throw we synthesize the same
 * "unavailable" AdapterReport shape probeAdapter itself uses, with `reason`
 * naming the error, so the rest of the report generation is unaffected.
 */
export async function runAll(onProgress?: (p: RunAllProgress) => void): Promise<{
  adapter: AdapterReport
  results: ExpResult[]
  allPass: boolean
}> {
  let adapter: AdapterReport
  try {
    adapter = await probeAdapter()
  } catch (err) {
    adapter = {
      ok: false,
      features: [],
      hasTimestamp: false,
      hasIndirectFirstInstance: false,
      hasFloat32Filterable: false,
      maxTextureDimension3D: 0,
      reason: `probeAdapter threw: ${err instanceof Error ? err.message : String(err)}`,
    }
  }

  // Stable ids paired with each runner: `run.name` is the minified/renamed
  // function identifier (e.g. `runExp09`), not the `exp-NN-...` id the
  // results-doc tables key on, so runSequentially's catch path (which keys
  // a thrown experiment's result to `id`) needs these, not `run.name`.
  const runners: Runner[] = [
    { id: 'exp-09-reverse-z', run: runExp09 },
    { id: 'exp-02-wgsl', run: runExp02 },
    { id: 'exp-04-storage-texture', run: runExp04 },
    { id: 'exp-01-march-cost', run: runExp01 },
  ]

  // Sequential, never Promise.all: each experiment builds its own
  // WebGPURenderer, and concurrent renderers would contend for the GPU and
  // corrupt Exp-1's timing measurement specifically. See runSequentially's
  // doc for why this is a separate, independently-testable function rather
  // than a loop inlined here. RUNNER_TIMEOUT_MS is what turns a repeat of
  // the reproduced exp-09 hang into a loud, specific failure instead of
  // another silent forever.
  const results = await runSequentially(runners, onProgress, RUNNER_TIMEOUT_MS)

  return { adapter, results, allPass: results.every((r) => r.pass) }
}

export function formatReport(r: Awaited<ReturnType<typeof runAll>>): string {
  const lines = [
    `# WebGPU experiment results`,
    ``,
    `Adapter: ${r.adapter.ok ? 'ok' : r.adapter.reason}`,
    `maxTextureDimension3D: ${r.adapter.maxTextureDimension3D}`,
    `timestamp-query: ${r.adapter.hasTimestamp}`,
    `indirect-first-instance: ${r.adapter.hasIndirectFirstInstance}`,
    `float32-filterable: ${r.adapter.hasFloat32Filterable}`,
    ``,
    `| Experiment | Result | Detail |`,
    `|---|---|---|`,
    ...r.results.map((e) => `| ${e.id} | ${e.pass ? 'PASS' : 'FAIL'} | ${e.detail.replace(/\|/g, '\\|')} |`),
    ``,
    `Passed ${r.results.filter((result) => result.pass).length} of ${r.results.length} graphics tests on this device.`,
    ``,
    // The pass/fail table above is not enough to fill in the results-doc
    // tables (p95Ms, submitOnlyMs, per-altitude failure lists, Exp-4's
    // sampled value, ...) - none of that survives through `detail` alone.
    // Dump each experiment's `data` verbatim so a pasted report is complete.
    `## Data`,
    ``,
    ...r.results.flatMap((e) => [
      `### ${e.id}`,
      '```json',
      JSON.stringify(e.data ?? {}, null, 2),
      '```',
      ``,
    ]),
  ]
  return lines.join('\n')
}
