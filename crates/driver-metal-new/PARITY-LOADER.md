# Parity ledger: `csrc/src/loader/` → `src/loader/`

Every entity in the C++ loader is listed here as ported, dropped (with the
reason), or missing (with the blocker). Same rules as `PARITY.md`.

## Heap planning — `src/loader/heap.rs`

`loader/heap_layout.hpp` (192 lines), pure offset arithmetic.

| C++ | Rust | |
|---|---|---|
| `align_up(n, a)` | `align_up` | ported |
| `HeapPlan` (7 regions + intermediates) | `HeapPlan` | ported |
| `plan_heap(g, weights, max_ctx, …)` | `plan_heap(g, tuning, weights, HeapParams)` | ported |
| defaulted trailing parameters ×5 | `HeapParams` with `Default` | ported |

What changed and why:

* The five defaulted positional parameters became `HeapParams`. Two of the
  five were both `int` and adjacent (`state_dtype_bytes`,
  `act_dtype_bytes`); a call that swaps them compiles and allocates fp32
  state at bf16 width. Named fields cannot swap.
* `plan_heap` takes `&Tuning` because the scratch slot's width reaches
  `Tuning::moe_tile_rows` through `scratch_slot_elems`. The C++ read a
  process-global tuning singleton from inside the arithmetic, which is why
  its heap plan could not be tested against two devices in one process.
* The scratch-slot derivation this header USED to carry is deleted, not
  ported — that copy had drifted (slot sized 8320 elements where the binder
  laid rows 16384 apart; every row past the halfway point wrote into the
  next colour). `src/batch/sizing.rs` is the one derivation and
  `plan_heap` calls it. The C++ fixed this the same way; the ledger entry
  exists so nobody re-introduces a second copy "for layering reasons".

## Load-plan compilation — `src/loader/plan.rs`

`loader/load_plan.hpp` (160 lines). The C++ reached the Rust loader through
the C ABI; this port calls `model` (author registry) and `model-loader`
(plan compiler) in-process, so the wire structs disappear.

| C++ | Rust | |
|---|---|---|
| `kMetalTileMapMask` | `METAL_TILE_MAP_MASK` | ported; one-sidedness vs the loader's model pinned by test |
| `kMetalPreferredAlignment` / `kMetalMaxTileBytes` | `METAL_PREFERRED_ALIGNMENT` / `METAL_MAX_TILE_BYTES` | ported |
| `metal_device_target()` | `metal_storage_target()` | ported; states the fields the C ABI defaulted (fusion_mask 0, BF16 encode scratch, no native MXFP4) |
| `plan_ties_embeddings` | `plan_ties_embeddings` | ported, with the two-wrong-configs story |
| `descriptor_for_testing` | `descriptor_for_testing` + `TestFacts` | ported; the round-trip through `ModelFacts::from_descriptor` is pinned by test |
| `compile_load_plan` | `compile_load_plan` | ported; returns the author's resolved `Mxfp4MoePolicy` like the C ABI did |
| `Checkpoint::open` + handle lifetime | — | dropped: `parse_checkpoint_metadata` is called in-process; there is no handle to keep alive |
| `plan.verify_model(request)` | file stat loop | dropped in part, with a reason: the verifier existed to hold the MARSHALLED plan to a re-authored contract — marshalling and author determinism both in scope. In-process there is no marshalling, and a same-process re-author is a restatement, not a second opinion. What still checks something real survives: every file the plan declares is stat'ed against the snapshot |
| exceptions | `LoadPlanError` (5 named variants) | ported |

## The expert slab — `src/loader/slab.rs` (+ `model-loader/src/group_slot.rs`)

`loader/expert_slab.hpp` (197) and its dependency
`pie_loader/group_slot_index.hpp` (163), which had no Rust counterpart.

| C++ | Rust | |
|---|---|---|
| `pie_loader::GroupSlotIndex` | `model_loader::group_slot::GroupSlotIndex` | ported — into the LOADER crate, per the header's own argument: two backends deciding residency by two eviction rules is two ways for the same checkpoint to thrash |
| `kAbsent` sentinel / `int32_t` slot | `Option<u32>` | ported |
| all-slots-pinned `runtime_error` | `AllSlotsPinned` (typed) | ported |
| `SlabTensor` (suffix, band, layer pointers) | `SlabTensor<'a>` (byte slices) | ported |
| `ExpertSlab` ctor's thrown strings | `SlabError` (9 named variants) | ported |
| null-pointer layer check | `ShortBank` length check | ported, stronger: a slice carries its length, so the real precondition (`experts * band_bytes` per bank, `slots * band_bytes` per slab) is checked instead of just non-null |
| `ensure_resident` / `end_batch` / stats | same; `ensure_resident` is `unsafe fn` | ported — the GPU-quiescence contract the C++ carried in prose is a `# Safety` section, and out-of-grid (layer, expert) is a typed error because the expert id is the ROUTER's readback: data fails the fire, it does not crash the process |
| `slot_data` pointer accessor | `slab(t)` + `slot_offset(t, slot)` | ported: binding needs the region and the offset separately |

The module keeps the two arguments that justify the design: residency has
to be a wired region whose contents change (`requestResidency` wires every
page — 18.4 GB for a streamed Qwen3-30B-A3B against 1.5 GB at rest, and an
Apple GPU aborts rather than faults on a non-resident touch), and a slot is
every tensor of one expert or nothing (one `expert_ids` buffer indexes every
routed projection, so per-tensor slot numbers cannot exist).

## Not yet started

| C++ | lines | blocker |
|---|---|---|
| `heap_bind.cpp` | 2044 | Metal-side: heap alloc + argument tables; needs `src/metal` runtime surface |
| `transcode.hpp` | 354 | tensor staging/transcode; portable, next candidate |
| `heap_bind_metal.hpp` | 209 | Metal-side companion of `heap_bind.cpp` |
