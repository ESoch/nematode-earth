import { createHash } from 'node:crypto'
import { readdir, readFile } from 'node:fs/promises'
import { resolve } from 'node:path'

export const sha256 = bytes => createHash('sha256').update(bytes).digest('hex')

export async function walkFiles(root, prefix = '') {
  const files = []
  for (const entry of await readdir(resolve(root, prefix), { withFileTypes: true })) {
    const path = prefix ? `${prefix}/${entry.name}` : entry.name
    if (entry.isSymbolicLink()) throw new Error(`Release must not contain symlinks: ${path}`)
    if (entry.isDirectory()) files.push(...await walkFiles(root, path))
    else if (entry.isFile()) files.push(path)
  }
  return files.sort()
}

// Deliberately supports the path/wildcard subset used by this site's _headers,
// not all of Cloudflare's routing rules. Unknown syntax fails closed.
export async function readHeaderRules(root) {
  const rules = []
  for (const line of (await readFile(resolve(root, '_headers'), 'utf8')).split(/\r?\n/)) {
    if (!line.trim() || line.trimStart().startsWith('#')) continue
    if (!/^\s/.test(line)) {
      if (!/^\/[^\s:]*$/.test(line)) throw new Error(`Unsupported header path: ${line}`)
      const escaped = line.split('*').map(part => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('.*')
      rules.push({ pattern: new RegExp(`^${escaped}$`), headers: [] })
    } else {
      const match = line.trim().match(/^([^:]+):\s*(.+)$/)
      if (!match || !rules.length) throw new Error(`Invalid header rule: ${line}`)
      rules.at(-1).headers.push([match[1], match[2]])
    }
  }
  return rules
}

export function headersFor(rules, path) {
  const headers = new Headers()
  for (const rule of rules) {
    if (rule.pattern.test(path)) for (const [key, value] of rule.headers) headers.append(key, value)
  }
  return headers
}
