/**
 * Which cube each of `sweep/Matter.ts`'s three texture slots should sample,
 * given whichever atlases have landed so far.
 *
 * Three separate networks loads — elevation (a 3-tier ladder, up to 192 MiB
 * decoded), the density cube, and the urban ladder — each with its own
 * failure mode and no ordering guarantee between them. `viewer.ts` used to
 * gate the material rebuild on all three at once
 * (`if (!(elevationCubeTex && realCube && urbanCubeTex)) return`), which
 * turned any single failure into the silent loss of every atlas that HAD
 * succeeded: bytes fetched, decoded, land-mask-validated, and then dropped,
 * with nothing on screen changing and only an unrelated `console.error` in
 * the log. Exactly the shape of "31.7 MiB fetched, nothing happens".
 *
 * Splitting it out as a function rather than three inline `??`s is so the
 * independence is a tested contract (`matterAtlases.test.ts`) instead of an
 * expression someone can re-couple without anything noticing.
 */

export interface MatterAtlases<T> {
  elevation: T | null
  density: T | null
  urban: T | null
}

export type MatterBindings<T> = { [K in keyof MatterAtlases<T>]-?: T }

/** Each slot takes its own atlas if it has one, and `placeholder` otherwise — never any other slot's state. */
export function matterAtlasBindings<T>(atlases: MatterAtlases<T>, placeholder: T): MatterBindings<T> {
  return {
    elevation: atlases.elevation ?? placeholder,
    density: atlases.density ?? placeholder,
    urban: atlases.urban ?? placeholder,
  }
}
