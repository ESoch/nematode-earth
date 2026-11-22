/**
 * Rebuild-then-dispose, in that order.
 *
 * Every progressive ladder in `viewer.ts` (elevation tiers, urban tiers, the
 * density cube) replaces a live `NodeMaterial` with a freshly built one, and
 * each replacement strands the superseded material and its compiled pipeline
 * unless it is released. The elevation path got this right; the urban path
 * did not, leaking two orphaned NodeMaterials per session as the ladder
 * climbed.
 *
 * The ORDER is the part worth encoding rather than repeating: dispose first
 * and the renderer is briefly holding a released pipeline for a material
 * that is still assigned. Install first, and the superseded one is provably
 * unreferenced by the time it is dropped.
 */
export interface Disposable {
  dispose(): void
}

export function replaceMaterial<M extends Disposable>(
  previous: M | null | undefined,
  next: M,
  apply: (next: M) => void,
): M {
  apply(next)
  if (previous && previous !== next) previous.dispose()
  return next
}
