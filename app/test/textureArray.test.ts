import { describe, it, expect } from 'vitest'
import { packFacesIntoDataArrayTexture } from '../src/field/textureArray'

describe('packFacesIntoDataArrayTexture', () => {
  it('concatenates faces in order, layer i = faces[i]', () => {
    const faces = [
      Uint16Array.from([1, 2, 3, 4]),
      Uint16Array.from([5, 6, 7, 8]),
      Uint16Array.from([9, 10, 11, 12]),
    ]
    const array = packFacesIntoDataArrayTexture(faces, 2)
    expect(array.image.width).toBe(2)
    expect(array.image.height).toBe(2)
    expect(array.image.depth).toBe(3)
    if (!array.image.data) throw new Error('array texture has no data')
    expect(Array.from(array.image.data)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
  })

  it('throws on an empty face list', () => {
    expect(() => packFacesIntoDataArrayTexture([], 2)).toThrow(/no faces/)
  })

  it('throws when a face length disagrees with face 0', () => {
    const faces = [Uint16Array.from([1, 2, 3, 4]), Uint16Array.from([5, 6, 7])]
    expect(() => packFacesIntoDataArrayTexture(faces, 2)).toThrow(/expected 4/)
  })
})
