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

## Not yet started

| C++ | lines | blocker |
|---|---|---|
| `heap_bind.cpp` | 2044 | Metal-side: heap alloc + argument tables; needs `src/metal` runtime surface |
| `transcode.hpp` | 354 | tensor staging/transcode; portable, next candidate |
| `heap_bind_metal.hpp` | 209 | Metal-side companion of `heap_bind.cpp` |
| `expert_slab.hpp` | 197 | MoE expert paging slab; `expert_paging.hpp` (batch) waits on it |
