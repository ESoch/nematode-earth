import * as THREE from 'three/webgpu'

/**
 * Packs N equal-length per-face `Uint16Array` buffers (already GPU-ready —
 * e.g. half-float bit patterns, same layout as a `DataTexture`'s `.image.data`)
 * into one `THREE.DataArrayTexture`, layer `i` = `faces[i]`. Shared by
 * `bakeCube.ts` (single-channel R16Float synthetic bake) and `loadCube.ts`
 * (four-channel RGBA16Float real atlas) so both get the identical
 * concatenation logic instead of two hand-rolled copies that could drift.
 *
 * Exists so `field/cubeArrayNodes.ts` has something to sample from inside a
 * compute kernel: `THREE.CubeTextureNode` (the `cubeTexture()` TSL path
 * `regimes/Shell.ts` uses) cannot run in compute — see that file's doc for
 * the exact mechanism. A `DataArrayTexture`, sampled via the base
 * `TextureNode` + `.depth(layer)`, has no such dependency.
 *
 * Caller sets `format`/`type`/filtering/wrap afterward — this only
 * assembles the buffer and its width/height/depth, since those properties
 * differ between the two callers (R16Float vs RGBA16Float).
 */
export function packFacesIntoDataArrayTexture(
  faces: readonly Uint16Array[],
  faceSize: number,
): THREE.DataArrayTexture {
  const depth = faces.length
  if (depth === 0) {
    throw new Error('packFacesIntoDataArrayTexture: no faces given')
  }
  const firstFace = faces[0]
  if (!firstFace) throw new Error('packFacesIntoDataArrayTexture: no faces given')
  const texelsPerFace = firstFace.length
  const data = new Uint16Array(texelsPerFace * depth)

  faces.forEach((face, i) => {
    if (face.length !== texelsPerFace) {
      throw new Error(
        `packFacesIntoDataArrayTexture: face ${i} has ${face.length} elements, `
        + `expected ${texelsPerFace} (face 0's length)`,
      )
    }
    data.set(face, i * texelsPerFace)
  })

  return new THREE.DataArrayTexture(data, faceSize, faceSize, depth)
}
