import * as THREE from 'three/webgpu'
import type { Node } from 'three/webgpu'
import {
  Fn, If, Return, float, int, uint, vec3, vec4, uniform, instanceIndex,
  instancedArray, storage, atomicAdd, atomicStore, atomicLoad,
  exp, log, sqrt, cos, max, min, abs, normalize, length, smoothstep,
} from 'three/tsl'
import { CELL_BIAS } from '../math/lattice'
import { E_A } from '../math/optical'
import type { TslFieldPort } from '../field/probeField'
import { encodeIndexedIndirectArgs } from '../regime2/indirectDrawArgs'
import type { Vec3 } from '../math/raySpheroid'

/**
 * Regime 2 compute side: a world-anchored cell lattice, emitted by a compute
 * kernel, drawn via a 5-word indexed indirect draw.
 *
 * NOT TRANSCRIBED FROM THE REFERENCE PLAN. The reference's Task 7 compute
 * kernel calls `uWorldOffX(L)`/`uWorldOffY(L)`/`uWorldOffZ(L)` and
 * `uSigmaNormRatio()` - functions it uses but never defines anywhere in the
 * document - so it would not type-check as written. This file is written
 * fresh against the pure, tested modules from the prior session
 * (math/lod.ts, math/lattice.ts, math/crossSection.ts, regimes/Blend.ts)
 * and against `node_modules/three` source, cited inline at each load-bearing
 * claim, the same discipline `regimes/Shell.ts`'s `dumpShellWGSL` uses for
 * its one isolated internal-API cast.
 *
 * SIMPLIFICATIONS relative to the reference plan's full design, all
 * deliberate and documented at the point they matter (search "SIMPLIFIED"):
 * single-level lattice (no octave cross-fade between two adjacent levels);
 * a symmetric cell-index box around the camera rather than a frustum-fitted
 * one (`math/lattice.ts`'s `cellBoxFor` was not built - see the Session 4
 * report); the direct (non-incremental) reduced-altitude formula, justified
 * because this session's altitude band (Blend.ts's REGIME2_FULL_ALT..
 * REGIME2_TOP_ALT, ~3.3-33 km) stays far above the sub-metre range where
 * f32's ~1 m floor would matter; and a VERTICAL (nadir) transmittance
 * approximation rather than the full slant-path correction (math/slant.ts
 * was not built either). None of these compromise the two Nelson Max
 * conditions Blend.ts's doc names - every instance is still multiplied by
 * transmittance from the same sigma_t field, and the medium is still
 * emission-only - they trade geometric/precision fidelity, not the
 * partition's energy bookkeeping.
 *
 * VERIFIED AGAINST SOURCE (not assumed):
 * - `drawIndexedIndirect` really is a 5-word buffer read natively by
 *   `GPURenderPassEncoder.drawIndexedIndirect` - three.js's own dispatch
 *   for indexed geometry with `getIndirect() !== null` calls exactly that,
 *   passing only the buffer and offset (WebGPUBackend.js:1836-1858). The
 *   JS-side `drawParams.instanceCount` computed alongside it is used ONLY
 *   for `info.update()` stats in that branch, never passed to the GPU call.
 * - A plain (non-instanced) `THREE.BufferGeometry` gets `instanceCount = 1`
 *   in `RenderObject.getDrawParameters()` (the `isInstancedBufferGeometry`
 *   branch is skipped), so the `instanceCount === 0` early-return
 *   (RenderObject.js:616) never fires regardless of how many instances the
 *   INDIRECT buffer's own word actually specifies - confirming the
 *   plain-geometry choice is what lets a frame with zero live instances
 *   still reach the indirect draw call rather than being skipped in JS.
 * - Attribute usage-flag ordering: `Attributes.update()` only runs a given
 *   attribute's creation path ONCE (`data.version === undefined` gate,
 *   Attributes.js:73-97) and dispatches on the CALLER-supplied `type`
 *   (STORAGE vs INDIRECT) - so whichever code path touches the args buffer
 *   FIRST determines whether it ever gets the `GPUBufferUsage.INDIRECT`
 *   flag. `buildParticleSystem` never calls `storage()` on the args buffer
 *   at all (only the compute-side finalize step does, and only after
 *   `geometry.setIndirect` has already registered it) - see `finalizeKernel`.
 * - `instancedArray`/`storage`/`atomicAdd`/`atomicStore`/`.toAtomic()`/
 *   `.compute(count)`/`renderer.computeAsync()` signatures all confirmed
 *   against `@types/three`'s declarations, not guessed from the reference.
 *
 * NOT VERIFIED, NOT VERIFIABLE HERE: whether this actually compiles to
 * valid WGSL, whether the compute kernel actually places instances where
 * intended, whether the one-frame-latency ordering is race-free in
 * practice, and the specific "camera nodes are broken in compute shaders"
 * claim (three.js #27570) - taken as given per your own repeated statement
 * of it, not independently re-derived. All of this needs the browser run
 * you already said you'd do.
 *
 * PLACEMENT HERE, NOISE BUDGET IN math/lod.ts: `emitKernel` below places
 * particles on a world-space volumetric lattice confined to a small window
 * around the camera - NOT the whole-screen Poisson placement
 * `math/lod.ts`'s `fullPopulationPerPixel`/`instancesPerPixel` model.
 * Those two are different LAYERS (placement scheme vs. screen-space noise
 * argument), not competing implementations, but as of this file `math/
 * lod.ts` is called only for `sigmaPixAt`/`cellSizeForBudget` - its
 * per-pixel population functions are not exercised by anything that runs.
 * See the matching note at the top of `math/lod.ts`.
 */

const A = 6378137.0
const B = 6356752.314245

type FloatNode = Node<'float'>
type Vec3Node = Node<'vec3'>
type UintNode = Node<'uint'>

/** PCG avalanche on a uint node - mirrors math/lattice.ts's `pcgHashU32` term-for-term. */
const pcgTSL = Fn(([v]: [UintNode]): UintNode => {
  const state = v.mul(uint(747796405)).add(uint(2891336453))
  const word = state.shiftRight(state.shiftRight(uint(28)).add(uint(4))).bitXor(state).mul(uint(277803737))
  return word.shiftRight(uint(22)).bitXor(word)
})

/** Mirrors math/lattice.ts's `cellHash` term-for-term - see that file's doc for why. */
const cellHashTSL = Fn(([cx, cy, cz, salt]: [UintNode, UintNode, UintNode, UintNode]): UintNode => {
  const bx = int(cx).add(int(CELL_BIAS)).toUint()
  const by = int(cy).add(int(CELL_BIAS)).toUint()
  const bz = int(cz).add(int(CELL_BIAS)).toUint()
  const h0 = bx.mul(uint(0x27220a95))
  const h1 = pcgTSL(h0.bitXor(by.mul(uint(0x165667b1))))
  const h2 = pcgTSL(h1.bitXor(bz.mul(uint(0x9e3779b1))))
  return pcgTSL(h2.add(salt.mul(uint(0x85ebca6b))))
})

const hashUnitTSL = Fn(([h]: [UintNode]): FloatNode => h.toFloat().mul(1 / 2 ** 32))

export interface ParticleOpts {
  field: TslFieldPort
  /** Max instances the records buffer can hold - the "B" of spec 7.7's budget. Keep well under 4e6 for this foundation pass; see the report. */
  budget: number
  /** Max candidate cells enumerated per frame, before any survive taper/altitude/frustum rejection. */
  maxCandidates: number
}

export interface ParticleFrameState {
  /** Camera's TRUE planet-relative world position (matches Shell.ts's ShellOpts.cameraWorldPosition under floating origin). */
  camWorld: Vec3
  altM: number
  /** Half-width of the emission window in metres - instances exist only within this radius of the camera (Blend.ts's windowTaper). */
  windowOut: number
  windowIn: number
  Hs: number
  Hmax: number
  /** `sigma0(1, Hs, Hmax)` - same bracket Shell.ts's `sigma0PerTau` uses. */
  sigma0PerTau: number
  tauCap: number
  /** `handoffWeight(altM)` (Blend.ts). */
  w: number
  cellSize: number
}

export interface ParticleStats {
  cellSize: number
  candidates: number
}

/**
 * Everything the emit kernel needs but cannot get from camera nodes
 * (three.js #27570 - camera-node updates run through render-group
 * `onRenderUpdate`, which never fires for a compute pass). All explicit
 * uniforms, updated from `ParticleFrameState` each frame by `update()`.
 */
interface FrameUniforms {
  camWorld: THREE.UniformNode<'vec3', THREE.Vector3>
  windowIn: THREE.UniformNode<'float', number>
  windowOut: THREE.UniformNode<'float', number>
  Hs: THREE.UniformNode<'float', number>
  Hmax: THREE.UniformNode<'float', number>
  sigma0PerTau: THREE.UniformNode<'float', number>
  tauCap: THREE.UniformNode<'float', number>
  w: THREE.UniformNode<'float', number>
  cellSize: THREE.UniformNode<'float', number>
  minX: THREE.UniformNode<'int', number>
  minY: THREE.UniformNode<'int', number>
  minZ: THREE.UniformNode<'int', number>
  dim: THREE.UniformNode<'uint', number>
}

export interface ParticleSystem {
  /** Shared quad geometry, 4 vertices / 6 indices, `setIndirect`-wired to `argsAttr`. */
  geometry: THREE.BufferGeometry
  /** Per-instance records: xyz = render-space (camera-relative) position, w = radiance flux. Read by SplatBlur.ts via `records.element(instanceIndex)`. */
  records: THREE.StorageBufferNode<'vec4'>
  /**
   * The 5-word indexed-indirect-draw args buffer `finalizeKernel` writes and
   * `geometry`'s `setIndirect` reads (`indexCount, instanceCount, firstIndex,
   * baseVertex, firstInstance` — `math/indirectDrawArgs.ts`'s layout).
   * Exposed (not just internal to this closure) so a caller can read it
   * back with `renderer.getArrayBufferAsync(argsAttr)` for diagnostics —
   * both this and `counterAttr` below were created via
   * `createIndirectStorageAttribute`/`createStorageAttribute`, which
   * three.js's WebGPU backend gives `GPUBufferUsage.COPY_SRC` by default
   * (`WebGPUBackend.js`), so the readback needs no extra staging-buffer
   * plumbing here.
   */
  argsAttr: THREE.IndirectStorageBufferAttribute
  /**
   * The 6-word atomic per-stage histogram `emitKernel`/`finalizeKernel`
   * share — see `buildParticleSystem`'s `CTR_*` constants for the
   * authoritative index list: `emitted` (raw slot-claims, pre-`budget`
   * clamp), `overflow` (slot-claims dropped past `budget`), `ran` (every
   * invocation, unconditional — proves the dispatch size took effect),
   * `rejectedPastLiveCount`/`rejectedTaper`/`rejectedAltitude` (one per
   * early-return gate in `emitKernel`, in the order they run — verified
   * live to be a strict partition: `ran = rejectedPastLiveCount +
   * rejectedTaper + rejectedAltitude + emitted`, exactly, at every
   * altitude tested; team lead's earlier "does not close" was a misreading
   * of `rejectedPastLiveCount` as "total rejections" rather than what its
   * OLD name — `rejectedTotal` — actually meant: candidates past `.compute
   * (maxCandidates)`'s dispatch slack for this frame's live `dim^3`, see
   * `emitKernel`'s own `total` guard). `accepted` is not its own word — it
   * is `emitted - overflow` — and there is no "density threshold" word
   * because `emitKernel` has no such gate: every candidate past the three
   * rejection gates unconditionally claims a slot regardless of its
   * flux/tau magnitude.
   */
  counterAttr: THREE.StorageBufferAttribute
  /** One field sample from a fixed direction, `(1,0,0)` — see `sampleBuf`'s declaration for why (isolating "field returns 0/NaN" from "emission logic rejects everything"). */
  sampleAttr: THREE.StorageBufferAttribute
  /** `[dist, z]` from the box-centre candidate — see `probeBuf`'s declaration for why it, not instanceIndex 0, is the representative sample. */
  probeAttr: THREE.StorageBufferAttribute
  /** `emitted/accepted` — see `compFactorBuf`'s declaration for the budget-clamp energy-compensation this exists to carry into `SplatBlur.ts`'s deposit material. */
  compFactorAttr: THREE.StorageBufferAttribute
  resetKernel: THREE.ComputeNode
  emitKernel: THREE.ComputeNode
  finalizeKernel: THREE.ComputeNode
  /** Updates the per-frame uniforms and the candidate box from `state`. Does not itself call `renderer.computeAsync` - the caller sequences that (render-then-compute, one frame of latency by design). */
  update(state: ParticleFrameState): ParticleStats
  dispose(): void
}

const QUAD_INDEX_COUNT = 6

function buildQuadGeometry(argsAttr: THREE.IndirectStorageBufferAttribute): THREE.BufferGeometry {
  // 4 unique local vertices, corner index encoded directly in position.xy so
  // the draw material can read "which of the 4 nearest-pixel corners am I"
  // via positionGeometry with no extra attribute. z is unused (always 0).
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], 3,
  ))
  geometry.setIndex(new THREE.Uint16BufferAttribute([0, 1, 2, 0, 2, 3], 1))
  // A PLAIN (non-instanced) geometry gets instanceCount=1 from
  // RenderObject.getDrawParameters(), which is what keeps three.js from
  // skipping the draw entirely before the indirect buffer's own instance
  // count is ever read - see this file's doc comment.
  geometry.setIndirect(argsAttr)
  return geometry
}

export function buildParticleSystem(opts: ParticleOpts): ParticleSystem {
  const { field, budget, maxCandidates } = opts

  const records = instancedArray(budget, 'vec4')

  // Per-stage rejection histogram, not just emitted/overflow: a single
  // "emitted=0" counter cannot distinguish "the kernel body never reached
  // the accept" from "it reached it and rejected at some specific gate" -
  // this localises the failure without another round trip. Indices:
  //   0 emitted   - raw slot-claims (pre-budget-clamp), same as before
  //   1 overflow  - slot-claims that hit budget and were dropped
  //   2 ran       - EVERY invocation, incremented unconditionally at the
  //                 very top of emitKernel before any early-return - this
  //                 alone proves .compute(maxCandidates) actually dispatched
  //                 more than one invocation
  //   3 rejectedPastLiveCount - instanceIndex >= dim^3 (past the live
  //     candidate count for this frame - the dispatch slack .compute
  //     (maxCandidates) always carries; see emitKernel's "total" guard).
  //     WAS misleadingly named `rejectedTotal`, read by a reviewer as "sum
  //     of all rejections" rather than what it actually is - one specific
  //     gate among four. Renamed; the counting itself was always correct
  //     (`ran = rejectedPastLiveCount+rejectedTaper+rejectedAltitude+emitted`
  //     exactly, verified live).
  //   4 rejectedTaper    - outside the windowIn/windowOut candidate-box radius
  //   5 rejectedAltitude - z outside [0, Hmax]
  // There is no separate "density threshold" gate to instrument - every
  // candidate that survives 3/4/5 unconditionally claims a slot regardless
  // of its flux/tau magnitude (see emitKernel below) - so accepted =
  // emitted - overflow, not a 7th counter.
  const CTR_EMITTED = 0
  const CTR_OVERFLOW = 1
  const CTR_RAN = 2
  const CTR_REJECTED_PAST_LIVE_COUNT = 3
  const CTR_REJECTED_TAPER = 4
  const CTR_REJECTED_ALTITUDE = 5
  const COUNTER_WORDS = 6
  const counterAttr = new THREE.StorageBufferAttribute(new Uint32Array(COUNTER_WORDS), 1)
  const counter = storage(counterAttr, 'uint', COUNTER_WORDS).toAtomic()

  // Diagnostic only: one field sample from a fixed, arbitrary direction,
  // written unconditionally by instanceIndex 0 at the very top of
  // emitKernel (before any early-return) so it is populated every
  // dispatch regardless of whether any candidate is accepted - see
  // logParticleBuffers() in viewer.ts (the 'p' key). Exists to separate
  // "emission logic rejects every candidate" from "the field itself
  // returns 0/NaN in compute" when the counters alone read 0.
  const sampleAttr = new THREE.StorageBufferAttribute(new Float32Array(1), 1)
  const sampleBuf = storage(sampleAttr, 'float', 1)

  // Diagnostic only: [dist, z] from the CENTRE candidate of the live
  // dim^3 box (ix=iy=iz=dim/2), not instanceIndex 0 - instanceIndex 0 sits
  // at the box's MINIMUM CORNER (see cxRaw/cyRaw/czRaw below), which is
  // `half*cellSize*sqrt(3)` from the camera by construction and is
  // expected to fail the taper test even when everything is working. The
  // centre candidate sits (approximately) where the camera itself is, so
  // its `dist` should read close to 0 - the strongest possible signal for
  // "is the taper-window comparison using the coordinate/units it thinks
  // it is". Written unconditionally (before the taper/altitude early
  // returns can skip it) whenever instanceIndex hits the centre index, so
  // it is populated regardless of whether that candidate itself is
  // accepted.
  const probeAttr = new THREE.StorageBufferAttribute(new Float32Array(2), 1)
  const probeBuf = storage(probeAttr, 'float', 2)

  // Budget-clamp energy compensation. `emitted` (raw slot-claims) can
  // exceed `budget` - overflow candidates are dropped without writing a
  // record (see finalizeKernel below) - and until now nothing scaled the
  // survivors up to cover the dropped ones' share, so the RENDERED total
  // was `accepted/emitted` of the true total by construction: a real,
  // uncompensated energy loss confirmed live (accepted=200,000 against
  // emitted=385,944 at one measured altitude - a ~1.93x deficit from this
  // alone). `finalizeKernel` computes `compFactor = emitted/accepted`
  // (always >= 1, since accepted = min(emitted,budget) <= emitted) once it
  // knows the FINAL emitted count - `emitKernel` itself cannot do this
  // correctly, since each invocation writes its own record before the
  // dispatch-wide atomic count is final. `SplatBlur.ts`'s deposit material
  // reads this the same way it already reads `records` - a storage buffer
  // written by compute, read by the vertex stage - and multiplies it into
  // each instance's flux, so every surviving particle represents its own
  // share PLUS the dropped ones' share, restoring the true total energy in
  // expectation. Initialised to 1 (neutral) so frame 0's already-empty
  // draw (instanceCount=0) is unaffected either way.
  const compFactorAttr = new THREE.StorageBufferAttribute(new Float32Array([1]), 1)
  const compFactorBuf = storage(compFactorAttr, 'float', 1)
  // A uniform, NOT a literal vec3(1,0,0): WGSL constant-folds a fully
  // literal expression at PARSE time, and directionToFaceUvNode's
  // discarded-branch divisions (see its module doc) hit a literal 1.0/0.0
  // for an axis-aligned literal direction - a compile-time WGSL error, not
  // a runtime one. A uniform is never constant-folded, so the same
  // division at runtime is legal (IEEE754 infinity, discarded by
  // .select()). This is what actually unblocks compute-pipeline creation;
  // cubeArrayNodes.ts's safeDenom guard is the separate, independent fix
  // for the runtime case (a real axis-aligned direction, e.g. a pole).
  const probeDir = uniform(new THREE.Vector3(1, 0, 0))

  // Initial value matches finalizeIndexedIndirectArgs(QUAD_INDEX_COUNT, 0, budget):
  // instanceCount=0 so frame 0 draws nothing while the first compute is
  // still pending (render-then-compute one-frame latency, by design).
  const initialArgs = encodeIndexedIndirectArgs({
    indexCount: QUAD_INDEX_COUNT, instanceCount: 0, firstIndex: 0, baseVertex: 0, firstInstance: 0,
  })
  const argsAttr = new THREE.IndirectStorageBufferAttribute(initialArgs, 1)
  const args = storage(argsAttr, 'uint', 5)

  const geometry = buildQuadGeometry(argsAttr)

  const U: FrameUniforms = {
    camWorld: uniform(new THREE.Vector3()),
    windowIn: uniform(1000),
    windowOut: uniform(2000),
    Hs: uniform(2000),
    Hmax: uniform(12000),
    sigma0PerTau: uniform(1),
    tauCap: uniform(1),
    w: uniform(0),
    cellSize: uniform(8),
    minX: uniform(0, 'int'),
    minY: uniform(0, 'int'),
    minZ: uniform(0, 'int'),
    dim: uniform(1, 'uint'),
  }

  const resetKernel = Fn(() => {
    for (let i = 0; i < COUNTER_WORDS; i++) {
      atomicStore(counter.element(i), uint(0))
    }
  })().compute(1)

  // SIMPLIFIED: a symmetric `dim x dim x dim` cell-index box around the
  // camera's own cell, not a frustum-fitted one - math/lattice.ts's
  // `cellBoxFor` (frustum candidate enumeration) was not built this
  // session, see the report. `instanceIndex` walks a flat 0..dim^3-1 range,
  // decoded to (ix,iy,iz) below.
  const emitKernel = Fn(() => {
    // "ran": every invocation, before anything else can early-return -
    // proves .compute(maxCandidates) actually dispatched more than one
    // invocation, independent of whether any of them accept.
    atomicAdd(counter.element(CTR_RAN), uint(1))

    // Diagnostic write - see sampleBuf's declaration above. Unconditional
    // on instanceIndex 0, ahead of every early-return in this kernel
    // (including the total/dim guard just below), so a dim=0 bug or a
    // rejected candidate 0 cannot itself hide a broken field sample.
    If(instanceIndex.equal(uint(0)), () => {
      sampleBuf.element(0).assign(field.tauNadirAt(probeDir))
    })

    const total = U.dim.mul(U.dim).mul(U.dim)
    If(instanceIndex.greaterThanEqual(total), () => {
      atomicAdd(counter.element(CTR_REJECTED_PAST_LIVE_COUNT), uint(1))
      Return()
    })

    // The box-centre candidate's flat index - see probeBuf's declaration
    // above for why this, not instanceIndex 0, is the representative one.
    const half = U.dim.div(uint(2))
    const centreIdx = half.add(half.mul(U.dim)).add(half.mul(U.dim).mul(U.dim))
    const isCentre = instanceIndex.equal(centreIdx)

    const ix = instanceIndex.mod(U.dim)
    const iy = instanceIndex.div(U.dim).mod(U.dim)
    const iz = instanceIndex.div(U.dim.mul(U.dim))

    // RAW (possibly negative) signed cell offsets from the box's minimum
    // corner - cellHashTSL biases them internally, mirroring cellHash's
    // own contract (math/lattice.ts).
    const cxRaw = int(ix).add(U.minX)
    const cyRaw = int(iy).add(U.minY)
    const czRaw = int(iz).add(U.minZ)
    const cx = cxRaw.toUint()
    const cy = cyRaw.toUint()
    const cz = czRaw.toUint()

    const s = U.cellSize
    const jx = hashUnitTSL(cellHashTSL(cx, cy, cz, uint(0)))
    const jy = hashUnitTSL(cellHashTSL(cx, cy, cz, uint(1)))
    const jz = hashUnitTSL(cellHashTSL(cx, cy, cz, uint(2)))

    // Absolute (ECEF-ish) lattice position: an integer cell index times the
    // cell size, jittered within the cell - `cxRaw` etc. already encode
    // "cell index relative to the box minimum" (see above), and
    // `U.minX`/`minY`/`minZ` are chosen (in `update()`, on the CPU) to be
    // near the camera's own cell, but that only bounds how far `absPos` is
    // from `camWorld` - it does NOT make `absPos` itself small. `absPos`
    // is the true world position; `pr` below is what actually needs to
    // stay small.
    const absPos = vec3(
      float(cxRaw).add(jx).mul(s),
      float(cyRaw).add(jy).mul(s),
      float(czRaw).add(jz).mul(s),
    )

    // Render-space (camera-relative) position: absPos MINUS the camera's
    // own world position - was previously missing this subtraction
    // entirely (this file's earlier revision named the un-subtracted
    // `absPos` itself `pr` and called it "camera-relative" in a comment
    // that was never true of the code beneath it). Confirmed the effect
    // live: the box-centre candidate's `dist` read 6,388,820 - Earth's
    // radius plus this session's ~10.7 km altitude, agreeing to 17 m - so
    // `dist` was measuring distance from the PLANET'S centre, not the
    // camera, against a windowOut of a few km. That rejects every
    // candidate, every frame, regardless of altitude - exactly the
    // rejectedTaper=100%/rejectedAltitude=0% signature the histogram
    // showed (z's own degeneracy under the same bug is a separate,
    // not-fully-explained symptom - see the team message this fix
    // responds to).
    const pr = absPos.sub(U.camWorld)

    const dist = length(pr)
    If(isCentre, () => { probeBuf.element(0).assign(dist) })

    const taper = float(1.0).sub(smoothstep(U.windowIn, U.windowOut, dist))
    If(taper.lessThanEqual(0.0), () => {
      atomicAdd(counter.element(CTR_REJECTED_TAPER), uint(1))
      Return()
    })

    // SIMPLIFIED (direct, non-incremental) reduced altitude - see this
    // file's doc comment for why this is safe at regime 2's actual
    // altitude band (km-scale, not the sub-metre range the incremental
    // technique exists for). absPos IS already the true world position -
    // no camWorld add needed here (that add, applied to the OLD pr, is
    // what used to double-count camWorld).
    const pWorld = absPos
    const q = vec3(pWorld.x.div(A), pWorld.y.div(A), pWorld.z.div(B))
    const z = length(q).sub(1.0).mul(A)
    If(isCentre, () => { probeBuf.element(1).assign(z) })

    If(z.lessThan(0.0).or(z.greaterThan(U.Hmax)), () => {
      atomicAdd(counter.element(CTR_REJECTED_ALTITUDE), uint(1))
      Return()
    })

    const dir = normalize(pWorld)
    const tauSampled = field.tauNadirAt(dir) // uncapped "tauData"
    const tauRendered = min(tauSampled, U.tauCap)

    // Exact Blend.ts instanceShare: 1 - (1-w)*(tauRendered/tauSampled). The
    // shell's own coupling (regimes/Shell.ts's ShellPartitionOpts) uses the
    // complementary (1-w)*tauRendered directly on its sigma0; this is the
    // other half of the same identity, applied to the "true" (uncapped)
    // density this instance represents rather than to what the shell draws.
    const r = tauRendered.div(max(tauSampled, float(1e-9)))
    const instShare = float(1.0).sub(float(1.0).sub(U.w).mul(r))

    const sigmaT = instShare.mul(tauSampled).mul(U.sigma0PerTau).mul(exp(z.div(U.Hs).negate()))

    // Per-instance cross-section DRAWN FROM THE DISTRIBUTION (not the
    // mean), mirroring math/crossSection.ts's aggregateLogSigma2/
    // aggregateMean/drawAggregateArea exactly - REL_VAR=3.0804 is that
    // module's own constant, inlined here since TSL functions can't import
    // a plain-JS numeric constant across the CPU/GPU boundary.
    const omega = max(sigmaT.mul(s).mul(s).mul(s).div(E_A), float(1e-6))
    const s2 = log(float(1.0).add(float(3.0804).div(omega)))
    const u1 = max(hashUnitTSL(cellHashTSL(cx, cy, cz, uint(3))), float(1e-7))
    const u2 = hashUnitTSL(cellHashTSL(cx, cy, cz, uint(4)))
    const zN = sqrt(log(u1).mul(-2.0)).mul(cos(u2.mul(2 * Math.PI)))
    const area = exp(log(omega.mul(E_A)).sub(s2.mul(0.5)).add(sqrt(s2).mul(zN)))

    // VERTICAL (nadir) transmittance approximation - see this file's doc
    // comment for why the slant (grazing-angle) correction is not applied
    // here. Condition 1 of Blend.ts's Nelson Max doc ("every instance
    // multiplied by transmittance from the same sigma_t field") is still
    // met: this uses the SAME sigmaT this instance's own flux is built
    // from, just integrated vertically rather than along the true
    // line-of-sight.
    const camQ = vec3(U.camWorld.x.div(A), U.camWorld.y.div(A), U.camWorld.z.div(B))
    const camAlt = length(camQ).sub(1.0).mul(A)
    const tauPath = instShare.mul(tauSampled).mul(U.sigma0PerTau).mul(U.Hs)
      .mul(abs(exp(z.div(U.Hs).negate()).sub(exp(camAlt.div(U.Hs).negate()))))
    const T = exp(tauPath.negate())

    const flux = area.mul(T).mul(taper)

    const slot = atomicAdd(counter.element(CTR_EMITTED), uint(1))
    If(slot.greaterThanEqual(uint(budget)), () => {
      atomicAdd(counter.element(CTR_OVERFLOW), uint(1))
      Return()
    })
    records.element(slot).assign(vec4(pr, flux))
  })().compute(maxCandidates)
  // Dispatched at the fixed WORST-CASE count (maxCandidates), not the live
  // per-frame candidate count (`total = dim^3` above, which varies with
  // altitude/window and is recomputed in `update()`) - `.compute()`'s count
  // is baked in at construction, so it cannot itself track a live uniform.
  // This is safe and correct, not a workaround: `update()` clamps
  // `dim <= floor(cbrt(maxCandidates))`, so `total <= maxCandidates`
  // always, and every invocation beyond `total` hits the
  // `instanceIndex.greaterThanEqual(total)` guard at the top of this
  // kernel and returns immediately - the standard "dispatch the max,
  // early-return past the live count" GPU pattern. Was WRONGLY
  // `.compute(1)` (a single invocation, `instanceIndex` only ever 0) until
  // team lead's readback caught it: `emitted`/`overflow` counters both
  // reading 0 through `p` is exactly what "the kernel effectively never
  // runs" looks like.

  // firstInstance is written 0 UNCONDITIONALLY (indirectDrawArgs.ts's own
  // encodeIndexedIndirectArgs throws on a nonzero value on the CPU side;
  // this GPU-side write mirrors that same invariant directly, since the
  // finalize step has to run on the GPU to read the atomic counter without
  // a CPU readback stall every frame - see this file's doc comment on why
  // that JS function couldn't be called here literally). instanceCount is
  // clamped to `budget`, mirroring finalizeIndexedIndirectArgs exactly.
  const finalizeKernel = Fn(() => {
    const emitted = atomicLoad(counter.element(CTR_EMITTED))
    // min() has no uint/int overload (three/tsl's own MathNode.d.ts marks
    // this "TODO Allow int/uint"), so the clamp is a select rather than a
    // min() call - the same select-as-ternary pattern regimes/Shell.ts
    // already uses.
    const budgetU = uint(budget)
    const clamped = emitted.lessThan(budgetU).select(emitted, budgetU)
    args.element(0).assign(uint(QUAD_INDEX_COUNT))
    args.element(1).assign(clamped)
    args.element(2).assign(uint(0))
    args.element(3).assign(uint(0))
    args.element(4).assign(uint(0))

    // See compFactorBuf's declaration above. clampedForDiv floors the
    // divisor at 1 (select, not max() - same int/uint overload gap noted
    // elsewhere in this file) so a frame with zero accepted candidates
    // writes compFactor=0 (harmless: there are no records to scale) rather
    // than dividing by zero.
    const clampedForDiv = clamped.greaterThan(uint(0)).select(clamped, uint(1))
    compFactorBuf.element(0).assign(float(emitted).div(float(clampedForDiv)))
  })().compute(1)

  function update(state: ParticleFrameState): ParticleStats {
    U.camWorld.value.set(state.camWorld[0], state.camWorld[1], state.camWorld[2])
    U.windowIn.value = state.windowIn
    U.windowOut.value = state.windowOut
    U.Hs.value = state.Hs
    U.Hmax.value = state.Hmax
    U.sigma0PerTau.value = state.sigma0PerTau
    U.tauCap.value = state.tauCap
    U.w.value = state.w
    U.cellSize.value = state.cellSize

    // Candidate box: a cube of cells centred on the camera's own cell,
    // sized so windowOut fits inside it with one cell of margin, then
    // clamped so dim^3 never exceeds maxCandidates (coarsened by growing
    // the effective cell if it would - SIMPLIFIED from the reference's
    // octave-coarsening `cellBoxFor`: this clamps the CANDIDATE COUNT by
    // capping `dim` directly rather than re-deriving a coarser octave, so
    // a too-small `cellSize` for the requested window silently under-covers
    // the window instead of over-enumerating; call sites should choose
    // `cellSize` from `math/lod.ts`'s `cellSizeFor`/`octaveSplit` so this
    // clamp is not normally reached).
    const halfCells = Math.ceil(state.windowOut / state.cellSize) + 1
    const dimRaw = 2 * halfCells + 1
    const maxDim = Math.max(1, Math.floor(Math.cbrt(maxCandidates)))
    const dim = Math.max(1, Math.min(dimRaw, maxDim))

    const camCellX = Math.floor(state.camWorld[0] / state.cellSize)
    const camCellY = Math.floor(state.camWorld[1] / state.cellSize)
    const camCellZ = Math.floor(state.camWorld[2] / state.cellSize)
    const half = Math.floor(dim / 2)

    U.minX.value = camCellX - half
    U.minY.value = camCellY - half
    U.minZ.value = camCellZ - half
    U.dim.value = dim

    return { cellSize: state.cellSize, candidates: dim * dim * dim }
  }

  function dispose(): void {
    geometry.dispose()
  }

  return {
    geometry, records, argsAttr, counterAttr, sampleAttr, probeAttr, compFactorAttr,
    resetKernel, emitKernel, finalizeKernel, update, dispose,
  }
}
