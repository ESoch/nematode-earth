// Intended address selected by the owner; this does not register or deploy it.
export function siteOrigin(value = 'https://nematode.earth') {
  let url
  try { url = new URL(value) } catch { throw new Error('SITE_URL must be a public HTTPS origin') }
  if (url.protocol !== 'https:' || url.username || url.password || url.pathname !== '/' || url.search || url.hash || url.port || ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname) || url.hostname.endsWith('.localhost')) {
    throw new Error('SITE_URL must be a public HTTPS origin without credentials, path, query or fragment')
  }
  return url.origin
}
