import { randomUUID } from 'node:crypto'
import { resolve } from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import { pathToFileURL } from 'node:url'
import { siteRevision } from './site-metadata.mjs'

const endpoint = 'https://nematode.earth/release.json'

export async function verifyLiveRevision(expected, { request = fetch, attempts = 24, retryDelayMs = 10_000 } = {}) {
  if (!siteRevision(expected)) throw new Error('Pass the expected full Git commit hash')
  let lastObservation
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      const url = new URL(endpoint)
      url.searchParams.set('check', randomUUID())
      const response = await request(url, {
        headers: { Accept: 'application/json' },
        cache: 'no-store',
        redirect: 'manual',
        signal: AbortSignal.timeout(5_000),
      })
      if (response.status !== 200) {
        await response.body?.cancel()
        throw new Error(`HTTP ${response.status}`)
      }
      const release = await response.json()
      const revision = siteRevision(release?.revision)
      if (revision === expected) return revision
      lastObservation = `reported ${revision ?? 'no revision'}`
    } catch (error) {
      lastObservation = error.message
    }
    if (attempt < attempts) await delay(retryDelayMs)
  }
  throw new Error(`${endpoint} did not match ${expected} after ${attempts} attempts (${lastObservation})`)
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const revision = await verifyLiveRevision(process.argv[2])
  console.log(`Verified ${endpoint} serves ${revision}`)
}
