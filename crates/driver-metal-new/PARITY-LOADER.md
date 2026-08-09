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

## Not yet started

| C++ | lines | blocker |
|---|---|---|
| `heap_bind.cpp` | 2044 | Metal-side: heap alloc + argument tables; needs `src/metal` runtime surface |
| `transcode.hpp` | 354 | tensor staging/transcode; portable, next candidate |
| `heap_bind_metal.hpp` | 209 | Metal-side companion of `heap_bind.cpp` |
| `expert_slab.hpp` | 197 | MoE expert paging slab; `expert_paging.hpp` (batch) waits on it |
| `load_plan.hpp` | 160 | manifest → load plan; portable, next candidate |
