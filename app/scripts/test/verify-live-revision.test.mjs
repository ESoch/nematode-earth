import assert from 'node:assert/strict'
import { test } from 'node:test'
import { verifyLiveRevision } from '../verify-live-revision.mjs'

const expected = 'a'.repeat(40)
const previous = 'b'.repeat(40)

test('waits for a stale deployment to be replaced on the public domain', async () => {
  const requests = []
  const request = async (url, options) => {
    requests.push({ url, options })
    return Response.json({ revision: requests.length === 1 ? previous : expected })
  }
  assert.equal(await verifyLiveRevision(expected, { request, attempts: 2, retryDelayMs: 0 }), expected)
  assert.equal(requests.length, 2)
  for (const { url, options } of requests) {
    assert.equal(url.origin + url.pathname, 'https://nematode.earth/release.json')
    assert.equal(options.cache, 'no-store')
    assert.equal(options.redirect, 'manual')
    assert.ok(options.signal instanceof AbortSignal)
  }
  assert.notEqual(requests[0].url.search, requests[1].url.search)
})

test('fails within the retry limit and reports the last live revision', async () => {
  let calls = 0
  const request = async () => {
    calls++
    return Response.json({ revision: previous })
  }
  await assert.rejects(verifyLiveRevision(expected, { request, attempts: 2, retryDelayMs: 0 }), {
    message: new RegExp(`did not match ${expected} after 2 attempts .*reported ${previous}`),
  })
  assert.equal(calls, 2)
})

test('rejects redirects and error statuses even if their bodies report the expected hash', async () => {
  for (const status of [302, 503]) {
    const request = async () => Response.json({ revision: expected }, { status })
    await assert.rejects(verifyLiveRevision(expected, { request, attempts: 1 }), new RegExp(`HTTP ${status}`))
  }
})

test('recovers after a network failure or incomplete response', async () => {
  const responses = [new Error('network unavailable'), new Response('<html>'), Response.json({ revision: expected })]
  const request = async () => {
    const next = responses.shift()
    if (next instanceof Error) throw next
    return next
  }
  assert.equal(await verifyLiveRevision(expected, { request, attempts: 3, retryDelayMs: 0 }), expected)
  assert.equal(responses.length, 0)
})

test('requires a full expected revision before making a request', async () => {
  let calls = 0
  const request = () => { calls++ }
  for (const value of [undefined, 'main', expected.slice(0, 7)]) {
    await assert.rejects(verifyLiveRevision(value, { request, attempts: 1 }), /full Git commit hash/)
  }
  assert.equal(calls, 0)
})
