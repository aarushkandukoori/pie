# Parity ledger: `csrc/src/store/` → `src/store/`

Two small headers; same rules as `PARITY.md`.

## `linear_state_slots.hpp` (38) → `src/store/linear.rs`

| C++ | Rust | |
|---|---|---|
| `resize(count)` (clamps to ≥1) | `LinearStateSlots::new(count)` | ported; the clamp is dropped — it existed to give `at()`'s alias target a slot to land on, and there is no alias target any more |
| `reset_all` / `reset(slot)` | same, `reset` returns `Result` | ported |
| `copy(src, dst)` | `copy`, with the stale-half story in its docs | ported |
| `at(slot) -> int&` | `step` / `count` / `parity`, each `Result` | ported, defect fixed: the C++ returns SLOT 0's counter for any out-of-range slot, so a wild ABI slot id silently read and wrote slot 0's ping-pong parity. Refused as `WildSlot` instead — slot ids are data |
| `int` counter | `u64`, wrapping | ported; the wrap modulus is even so parity survives a wrap |

The meaning the C++ kept in a comment at the call site — the counter's
parity IS the conv-state ping-pong, per slot and not decoder-wide, and a
state copy must inherit the exact count because the buffers move verbatim
— is the module doc and two tests here.

## `kv_pool.hpp` (31) → `src/store/kv_move.rs` (+ deferred)

| C++ | Rust | |
|---|---|---|
| `KvMoveCell` | `KvMoveCell` (wire field order kept) | ported |
| `copy_kv_cells` validation + offsets | `plan_cell_moves` → `CellMovePlan` | ported: validate every cell BEFORE any offset exists; one plan serves K and V of every full-attention layer; `pages_touched` carries what the elastic ensure needs |
| `KvPagePool` (SlotHandles + counters) | — | missing: device state; lands with the Metal kv-pool binding, where the counters (`capacity`/`committed`) belong beside the buffers they describe |

The device half executes each `CellCopy` with `Region::copy`, whose
memmove semantics are load-bearing: a compaction slides overlapping rows.
