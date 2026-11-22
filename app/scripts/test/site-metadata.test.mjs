import { test } from 'node:test'
import assert from 'node:assert/strict'
import { siteOrigin, siteRevision } from '../site-metadata.mjs'

test('uses the chosen public origin, and accepts an explicit HTTPS origin override', () => {
  assert.equal(siteOrigin(), 'https://nematode.earth')
  assert.equal(siteOrigin('https://example.pages.dev/'), 'https://example.pages.dev')
})

test('publishes only a full source revision when one is supplied', () => {
  assert.equal(siteRevision(), null)
  for (const length of [40, 64]) assert.equal(siteRevision('a'.repeat(length)), 'a'.repeat(length))
  for (const value of ['main', '6c8748a', 'g'.repeat(40)]) {
    assert.throws(() => siteRevision(value), /full Git commit hash/)
  }
})

test('rejects local, insecure, credential-bearing and path-specific sharing origins', () => {
  for (const value of ['http://example.com', 'https://localhost', 'https://127.0.0.1', 'https://[::1]', 'https://user:pass@example.com', 'https://example.com/subpath', 'https://example.com/?a=b', 'https://example.com/#hash', 'not a URL']) {
    assert.throws(() => siteOrigin(value), /SITE_URL/)
  }
})
