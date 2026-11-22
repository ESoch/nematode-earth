import { isWebGPUAvailable } from './platform/webgpu'
import { runAll, formatReport } from './experiments/report'

const root = document.getElementById('app')!

if (!isWebGPUAvailable()) {
  // Static, locally-authored markup with no dynamic content: innerHTML is
  // fine here, there is nothing to inject.
  root.innerHTML = `<div id="gate">
    <h1>WebGPU required</h1>
    <p>These graphics tests need WebGPU. Check <a href="https://caniuse.com/webgpu">browser support</a>
    or <a href="/about.html">read the exhibit methods</a>.</p>
  </div>`
} else {
  // Everything from here on is computed at runtime (experiment details can
  // embed arbitrary strings, e.g. caught error messages), so it goes through
  // a <pre> via textContent rather than innerHTML. There's no injection path
  // today since the content is locally computed, not user- or network-
  // supplied, but textContent costs nothing and is strictly safer for a
  // plain-text report.
  const pre = document.createElement('pre')
  pre.id = 'gate'
  pre.textContent = 'Running four experiments…'
  root.replaceChildren(pre)

  // Progress, both on the page and in the console: before this, the <pre>
  // above was the ONLY thing painted until every experiment had resolved -
  // a genuinely hung runner and a slow-but-working one were indistinguishable
  // from outside (no partial output, no console line, nothing). Each line
  // below is appended, never replaced, so the visible history survives even
  // if a later runner is the one that hangs - see report.ts's
  // runSequentially doc for the full reasoning.
  const startedAt = performance.now()
  const elapsed = (): string => `${((performance.now() - startedAt) / 1000).toFixed(1)}s`
  const progressLines: string[] = []
  const logProgress = (line: string): void => {
    progressLines.push(line)
    pre.textContent = `Running four experiments…\n\n${progressLines.join('\n')}`
    console.log(line)
  }

  const r = await runAll((p) => {
    logProgress(p.status === 'start'
      ? `[${elapsed()}] ${p.id}: starting…`
      : `[${elapsed()}] ${p.id}: ${p.status}`)
  })
  const md = formatReport(r)
  pre.textContent = md
  console.log(md)
  Object.assign(globalThis, { __session1: r })
}
