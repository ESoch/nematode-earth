# Rendering checks

We compared the November 2026 renderer with commit `b4805ad` using the same soil atlas, camera pose, exposure and viewport. Browser measurements used Chrome 157 on an Apple M2 Pro with Metal. They describe this device and build.

## Coastlines and uncertainty

The disk files store log density so they can represent a wide range of values. The old renderer blended those logarithms before converting them to light. Between a supported soil cell and an empty neighbour, this pulled the result sharply toward the empty cell. The new renderer decodes density before filtering. It weights the uncertainty coefficient by supported area so an empty neighbour supplies neither a measurement nor a zero-uncertainty value. [Atlas metadata](../app/public/data/soil_plain_512_v2/cube_meta.json) · [Texture loader](../app/src/field/loadCube.ts)

Across 33,415 pairs of adjacent supported and empty cells on the six faces, we compared the midpoint with the arithmetic mean of the two optical depths. The median was about **0.54% of that mean** with the old filter and **100.0001%** with the new one. All 456,202 supported cells retained positive optical depth on upload. Summing the uploaded values at cell centres with the extraction’s area weights gave **4.614977 × 10²⁰**, about **0.009% above** the source integral. This checks numerical conversion. Light in a finished frame also depends on the viewing angle, the optical model and exposure. [September extraction and area convention](superpowers/results/2026-09-22-local-soil-extraction.md) · [Loader tests](../app/test/loadCube.test.ts)

To repeat the GPU check, start the development server and open `/soil-validation.html`. We checked 48 directions against the shipped data, including all twelve edges between cube faces. Separate controls checked cube orientation and fully supported, partially supported and empty cells. [GPU probe](../app/src/experiments/soilCandidateProbe.ts) · [Reference calculation](../app/src/experiments/soilCandidateReference.ts)

## Sample locations

The viewer draws a circle three CSS pixels across for each visible sample. A circle keeps its size as you zoom. Amber marks samples outside the Antarctic biome code and cyan marks that code. Native WebGPU point primitives cover a single device pixel, so the viewer draws the circles as instanced sprites. In a fixed view of Europe at one device pixel per CSS pixel, 28 isolated patches had a median visible width of two device pixels before the change and four afterward, including antialiased edges. We counted pixels whose strongest colour channel changed by more than 32 of 255 levels when toggling the overlay. The median shift in patch centre was zero. A patch can contain several samples. [Marker implementation](../app/src/regimes/SamplePoints.ts) · [three.js point sizing](https://threejs.org/docs/pages/PointsNodeMaterial.html)

## The sweep

The ocean starts at sea level. As it drains, the water reflection fades and the lighting begins to follow the exposed seafloor. In a 64 × 64 GPU control, two different seabeds produced identical colour under intact water. They differed after draining. Scrubbing back restored the original pixels. [Terrain shader](../app/src/sweep/Matter.ts) · [Relief tests](../app/test/terrain.test.ts)

After the last rock disappears, the far hemisphere fades into the view over two seconds of playback. Previously it appeared in a single frame. At 7° W, 10° N and 11,500 km altitude in a 1200 × 800 view, we compared authored times 12.99 and 13.00. The old transition changed a colour channel by more than 16 of 255 levels in 20,970 globe pixels. The revised transition changed five pixels by a single level. [Sweep timing](../app/src/sweep/Timeline.ts) · [Shell renderer](../app/src/regimes/Shell.ts)

At public viewing altitudes the camera’s particle-emission window lies entirely above the shell. Skipping that work took the opening and settled views from three compute dispatches to zero and from ten encoded render passes to two per frame. Both builds sustained the test display’s 120 Hz refresh rate at 1200 × 800. [Particle visibility test](../app/test/blend.test.ts)
