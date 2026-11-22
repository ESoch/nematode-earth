# WebGPU prototype measurements, 16 August 2026

These measurements cover a development prototype in Chrome 151 on macOS with a device pixel ratio of 2. The record identifies no GPU model. No second machine or integrated-GPU Windows laptop was tested. The cube texture in this experiment was baked from a synthetic field.

## Shell rendering time

The experiment drew a 1920 × 1080 target with up to 128 samples per ray, using a 50° vertical field of view and an altitude of about 8,700 km. The shell extended beyond the top and bottom of the frame. Each run covered 240 frames. The table gives GPU pass timings from the quietest recorded comparison, with the three configurations interleaved at the same camera pose.

| Density source | 5th percentile | Median | 95th percentile | Standard deviation |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 1.97 ms | 2.03 ms | 2.16 ms | 0.28 ms |
| Analytic noise evaluated in the shader | 11.67 ms | 12.19 ms | 12.98 ms | 0.70 ms |
| Precomputed cube texture | 2.69 ms | 2.75 ms | 2.88 ms | 0.069 ms |

The analytic shader took 4.4 times as long as the precomputed texture at the median. The texture run exceeded the proposed 2.5 ms threshold by 0.25 ms. These runs measured the prototype shell at full resolution. Half-resolution figures in the original investigation were extrapolations. The later exhibit adds terrain, urban and soil data and uses different framing. [Timing experiment](../../../app/src/experiments/exp01-march-cost.ts)

**Why the timing method matters.** Submission alone took about 0.10 ms. Timing through a one-pixel readback included GPU completion and queue delays. Six early uniform runs gave wall-clock medians of 2.50–2.60 ms. A later run under contention gave 10.90 ms with a 30.4 ms 95th percentile. GPU timestamps also varied under load. In a separate contended comparison the uniform and analytic passes took 6.16 and 27.59 ms at the median. Report the timing method and machine load alongside any rerun.

## Shader and depth checks

| Probe | Observation |
| --- | --- |
| Visible output | The uniform shell produced centre alpha 12/255, close to the prototype optical model’s 4.6% prediction. Alpha reached 33/255 toward the rim. |
| Generated WGSL | The initial shader contained no texture lookup. The subsequent cube-sampled shader used explicit texture levels inside the march loop and showed no implicit samples. The generated shader suppressed the derivative-uniformity diagnostic, so the test inspected its text. |
| Compute-to-render texture read | A compute pass wrote 0.75 to a storage texture. A render pass in the same frame read 0.749 through the tested three.js path. |
| Reverse depth | With near and far planes at 1 and 10⁹ m, the nearer surface won at all ten tested distances with a separation of 0.01% of distance. Conventional depth won five of ten. A fixed 1 cm gap was unresolved at 10⁶ and 10⁷ m even with reverse depth. |

The adapter reported a maximum 3D texture dimension of 2048 and exposed timestamp queries, indirect first instance and float32 filtering. An early readback accidentally followed the window-sized canvas after renderer initialization. Subsequent probes set the drawing dimensions explicitly and sampled a known on-target pixel. [Shader inspection](../../../app/src/experiments/exp02-wgsl.ts) · [Storage texture probe](../../../app/src/experiments/exp04-storage-texture.ts) · [Reverse-depth probe](../../../app/src/experiments/exp09-reverse-z.ts)

## Other prototype observations

- A 512-face synthetic cube took about 350–450 ms to bake on the CPU at startup in this harness. The public view downloads precomputed assets.
- The original camera prototype descended from 40,000 km to about 4.2 cm in 240 synthetic wheel ticks. At its 1 cm floor, the shader’s scale height was 200 m and its optical-depth cap was about 0.009. These were checks of camera and shader arithmetic. The public zoom stops at the soil data’s supported detail.
- In three.js 0.185.1 the tested TSL cube-texture compute path tried to read a scene environment during dispatch, which had no scene. The fragment shader’s cube sampling worked. A six-layer array texture offered another representation for compute experiments. [Cube array sampling](../../../app/src/field/cubeArrayNodes.ts)

For a new measurement, start the development server and open `/experiments.html`. The page runs the probes and exposes the report as `window.__session1`. Record the actual browser, GPU, dimensions and revision with the results. [Development setup](../../../CONTRIBUTING.md)
