# Particle projection and deposited light, 17 August 2026

This investigation covered the experimental particle renderer using synthetic density. The apparent loss of light had two distinct causes. One affected the shape drawn for each particle. The other placed particles behind the camera at the centre of the image. The public soil exhibit uses the shell renderer.

## Two rendering errors

**Only one of four target pixels received light.** The corners of a one-pixel square fell exactly on four pixel centres. The rasterizer covered only one. For uniformly distributed subpixel positions its bilinear weight averages one quarter, so the other three quarters of a particle’s intensity were lost on average. Enlarging the square to two pixels per side put all four centres inside it. The shader then calculated each pixel’s weight separately. [WebGPU rasterization rules](https://www.w3.org/TR/webgpu/#rasterization) · [Deposit shader](../../../app/src/regimes/SplatBlur.ts)

**Behind-camera points accumulated at screen centre.** The original guard divided projected coordinates by a large positive substitute for a nonpositive clip-space `w`. That sent coordinates near zero, at the centre of the viewport. The corrected shader gives all four corners of a behind-camera particle the same off-screen position, so its square has zero area. [Shader and regression tests](../../../app/test/splat.test.ts)

## Altitude comparison

The August camera looked straight down. Candidate particles were sampled within a box centred on the camera. At very low altitude much of the box in front of the camera lay underground, where candidates were rejected. Approximately 100,000 accepted particles remained in the near-ground trace.

| Recorded quantity | 23,200 m altitude | 0.012 m altitude |
| --- | ---: | ---: |
| Accepted particles behind camera, rounded | 0% | 100% |
| In front and on screen | 17,952 (18.0%) | 0 |
| Pixels with a nonzero deposit | 69,578 | 6 |
| Peak deposit coordinate | (611, 154) | (755, 402) |
| Nonzero pixels after resolve | 1,205,192 | 6 |

The near-ground camera geometry explains why so few candidates could appear. The faulty guard explains the six bright centre pixels. At 23.2 km only 18% of accepted candidates landed on screen, illustrating the cost of an unfitted sampling box in that pose. The 1.2 cm trace was far below the prototype’s intended range and provides no evidence about the appearance of individual nematodes.

## Diagnostic controls

The computed emission total divided by the reference window integral was 1.008. The blur weights summed to 1.0000000000000004. The rejection categories accounted for all 800,000 candidates. These measured checks helped localize the faults to projection and deposition.

For a repeatable projection probe, place a test particle away from screen centre and read the full render target. A 128 × 128 centre crop missed a valid splat at x = 1058. A particle 2,000 m in front of the camera returned `clip.w = +2000`, confirming the sign used in the later tests. Record camera pose, rejection counters and target readback from the same frame. [Particle renderer](../../../app/src/regimes/Particles.ts) · [Deposit and resolve](../../../app/src/regimes/SplatBlur.ts)
