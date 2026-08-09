# Batch parity: `csrc/src/batch/` against `driver-metal-new`

The batch subsystem is ~11.6k lines: the forward executor (`forward.cpp`,
4.5k), the model-family glue (`simple_family.cpp`, 2k), and around them the
pure logic — scheduling, ticket composition, scratch coloring, decode
timing. Same rules as the other ledgers: every entry is **ported**,
**dropped** (with the reason the C++ needed it and the Rust does not), or
**missing** (with what blocks it). The portable half goes first.

## The batch shape — `src/batch/schedule.rs`

From `batch_schedule.hpp` (220 lines), the one file of `batch/` the C++
kept deliberately pure — and shipped without a checked build.

| C++ | Rust | |
|---|---|---|
| `BatchSchedule` | `BatchSchedule` | ported |
| `RequestSpan` | `RequestSpan` | ported |
| `build_batch_schedule` | `build_schedule` | ported |
| `find_request` | `find_request` | ported |
| `validate_paged_batch` | `validate_paged` | ported |
| `validate_paged_batch_capacity` | `validate_capacity` | ported |
| `BatchSchedule::m1` | `BatchSchedule::single` | ported |
| `kRsFlagReset` | `driver_abi::local::PIE_RS_FLAG_RESET` | dropped |
| `BatchStepInputs` | — | missing: the marshaling container belongs to the forward port |

The build/validate split survives — the geometry gate needs fire-time
arrays the build does not — but the build stops trusting its inputs:
`qo_hi - qo_lo` and `seqlen - new_tokens` were unchecked `u32` subtractions,
so a non-ascending `qo_indptr` or a span longer than its sequence produced
*wrapped* spans, and whether anyone noticed depended on whether that caller
also ran the validator. `build_schedule` refuses at construction, naming the
request; a `BatchSchedule` that exists has coherent spans.

`find_request` answered an out-of-range token with `R - 1` — a wrong request
shaped like a right one — and answers `None` here. `page_size <= 0` silently
became 32 inside the build; the default is now the caller's to take
(`DEFAULT_PAGE_SIZE`), not the build's to impose. The validator's
`bool` + static-string answers become `Rejected`, which names the request or
token at fault. `kRsFlagReset` was a hand copy of `PIE_RS_FLAG_RESET`,
"duplicated rather than included" for a test harness this crate does not
need; the masked read the C++ comment insists on (FOLD is not RESET) is kept
and tested.

Ten tests, portable, including the write-descriptor formula held exactly and
both wrap refusals.

## The wire mask — `src/batch/mask.rs`

From `wire_mask.hpp` (142 lines): whether a wire attention mask says
anything the kernel's own causal predicate does not already enforce.

| C++ | Rust | |
|---|---|---|
| `row_is_prefix` | `row_prefix` (private) | ported |
| `causal_prefix_lengths` | `causal_prefix_lengths` | ported |
| `first_kv_len_disagreement` | `kv_len_disagreement` | ported |

The semantics are kept exactly — a causal-prefix mask can be *dropped*, and
the answer is bit-identical because the kernel's predicate and the mask's
are then the same predicate; anything else refuses. What changes is the
plumbing: the C++ restates the CSR KV-length formula for the third time and
indexes `kv_page_indptr[r + 1]` / `kv_last_page_lens[r]` with no length
check anywhere — a mask table describing more requests than the CSR carries
is an out-of-bounds read. The Rust comparison takes the schedule's own
`RequestSpan`s (checked at construction, one owner for the formula), and a
mask describing more requests than the schedule is itself the first
disagreement. The `int` + out-param answer becomes `Disagreement`, carrying
the request and both numbers.

Eight portable tests, including multi-word rows, the sink and window
refusals, and the classic PAGE_T-16-vs-32 mismatch.

## The admission gate — `src/batch/admit.rs`

The first half of `compose.cpp`: the recurrent-state shapes this driver
refuses to run, extracted from `build_launch_view`'s refusal battery.

| C++ | Rust | |
|---|---|---|
| the RS refusals in `build_launch_view` | `admit_recurrent` | ported |
| `build_launch_view` (the slice wrapping) | — | dropped |
| `OwnedLaunchView::capture` | — | dropped |
| `OwnedLaunchView::view` | — | dropped |
| `kRsFlagFold` / `kRsFlagBufferWrite` / `kRsFlagFoldLenDevice` / `kRsFlagReset` | `driver_abi::local::PIE_RS_FLAG_*` | dropped |

The refusals are the file's real content and survive exactly — buffered
replay, device-resident fold lengths, mid-page buffer heads, a fold
boundary inside a fire's own tokens, a fold row with no tokens, mixed
persistence — each one a shape that fails *quietly* if admitted, corrupting
a recurrent state that cannot be recovered once folded. What changes is
their identity: the C++ throws `std::runtime_error` with prose as the only
discriminator, conflating "your launch is malformed" with "this driver
lacks the capability". `Refused` names each decision, keeps the C++'s prose
as `reason()`, carries the row at fault, and `is_malformed` answers the
question the exception type could not. A descending token CSR is refused as
malformed where the C++ wrapped the subtraction.

The wrapping half is dropped because its reason to exist is the C ABI: the
engine hands the C++ borrowed slices that die at return, so every launch was
re-wrapped (`build_launch_view`), deep-copied (`capture`) and re-wrapped
again (`view`). The Rust engine hands the driver an owned
`driver_abi::plan::LaunchPlan`; there is nothing to capture. The four
`kRsFlag*` hand copies fall to the ABI crate's own constants.

Nine portable tests, one per refusal plus the whole-row/no-fold admissions
and the malformed-rows fail-closed cases.

## The member description — `src/batch/member.rs`

The second half of `compose.cpp`: `build_member_forward_desc` and the type
it fills.

| C++ | Rust | |
|---|---|---|
| `MemberForwardDesc` | `ForwardDesc` | ported |
| `build_member_forward_desc` | `build_member_desc` | ported |
| `FireGeometry` (the consumed subset) | `ResolvedGeometry` | ported |
| `has_rs_slot` / `rs_slot_id` / `rs_reset` | `ForwardDesc::rs_slot` | dropped |
| `kv_last_page_len = 0` as "derive later" | `Option<u32>` | dropped |
| `StructuredMaskDescriptor` (here) | the `structured_mask` bit | dropped |

The RS dual-indexing fix — the shipped bug where the launch-wide form was
read from index 0, giving member 1 member 0's slot and barring two decodes
from ever sharing a forward — is kept with both forms matched explicitly
and a test that member 1 gets its own slot. The `bool` + out-param + error
string becomes `Result<ForwardDesc, BuildError>` with the C++'s own words
as `reason()`. The three-deep nested ternary deriving the final page fill
is `derive_key_len`, named and documented; the member-level RS triple is
derived from the per-request vectors instead of stored beside them; a zero
page size is refused where the C++ silently clamped it to one.

`ForwardDesc::extents`/`extents_from_readout` are
`m1_extents_from_forward_desc`/`m3_extents_from_forward_desc` — the two
entries `PARITY-M1.md` carried as missing — as methods on the type that
owns the fields.

Nine portable tests: wire slicing for both members, the pageless and
derived-fill paths, both RS forms and the dual-indexing regression, mask
shape and structured-mask refusals, malformed spans named, and the extents
round-trip.

## The activation colouring — `src/batch/color.rs`

From `scratch_color.hpp` (111 lines): the linear-scan colouring of
activation live ranges onto the ping-pong pool, extracted by the C++ "so a
second model family does not arrive with a second copy that drifts".

| C++ | Rust | |
|---|---|---|
| `Use` | `Use` | ported |
| `Coloring` | `Coloring` | ported |
| `color_live_ranges` | `color_live_ranges` | ported |
| `Coloring::hazard_free` | `ColoringError::HazardDetected` | dropped |

The algorithm survives byte for byte — first-use order, strictly-before
reuse, the concurrency-run extension, inclusive overlap. What does not: an
unused value took a fresh colour anyway (its `def = last = -1` never
matched `free_at < def`), inflating the count the scratch region is sized
by; it colours to `None` and costs nothing. An ordinal past the run table
silently skipped the extension — the one case the barrier-free-run rule
exists for — and is refused as malformed. And `hazard_free` was a
self-check behind a bool the caller must remember to read; the check now
runs unconditionally and a detected hazard is an `Err` that cannot be
ignored.

Seven portable tests: the ping-pong chain, same-dispatch write-after-read,
the run extension separating what solo ordinals may share, `no_recycle`,
unused values costing nothing, the refusals, and def-order robustness.

## Not yet started

The independent leaves of `batch/` are done; everything below hangs off
shared vocabulary. The porting order that follows from the includes:

1. **`decode_abi.hpp` (650) is the trunk** — Region/IoSlot enums, the ~30
   `bind::` argument-table layouts, `ArgmaxParams`, the `Kernel` kind enum
   and `ForwardGraphKey`. Pure ("NO Metal/ObjC — every lane includes it
   without a Metal dependency") and next. Its own best argument is the
   `KindCount` story at the bottom: the count used to be spelled
   `G4PleResidual + 1`, forty kinds short, so `psos[LmHeadUntied]` indexed
   past the array and the untied head ran the wrong pipeline — every logit
   zero, every token 0, and not one error anywhere. The enum's numeric
   values are ABI ("APPEND ONLY" ×5 in the comments), so the Rust port owes
   discriminant-pinning tests.
2. `decode_timing` (365) and the scratch schedule (`scratch.hpp/.cpp`,
   ~540) consume `Kernel`/`Dispatch`/`DecodeGeometry`; `Dispatch` and the
   geometry live under `model/<family>/` and land with the family port.
   (`DecodeGeometry` itself has since landed — see the geometry section.)
3. `expert_paging.hpp` (195): ported -- see "Expert paging" below.
4. `worker.hpp` (171) — ported (`worker.rs`); `simple_family` (2176),
   then `forward.cpp/.hpp` (5393) over everything. (`golden_tap` and
   `decode_psos`' multibatch half have since landed — see below.)

## The multibatch PSO plan — `src/batch/psos_mb.rs`

The portable half of `load_multibatch_psos` (`decode_psos.cpp`, the M>1
side of the 582): feature gating and the name grammar, emitted as
`MbRequest { slot, file, entry }` for the device half to batch-compile.

| C++ | Rust | |
|---|---|---|
| `MultiBatchPsoFeatures` (10 flags) | `MbFeatures` | ported, each flag's absence documented |
| `MultiBatchPsos` field table | `MbSlot` (39 variants, rung indices as payload) | ported: the C++ routes by pointer into a struct; the plan names the slot instead |
| `pie::kernels::entrypoint()` grammar | grammar inline + dev test | ported: names are products; existence is pinned against `kernels_metal::entrypoints()` on any host, as for the M=1 plan |
| `kQmmBMs` static_assert | `QMM_BMS` + rung-indexed slots | ported |
| batch-compile-per-file argument | module doc + `file` on each request | ported as a fact about the list |
| `compile_psos_from_files` + fill loop | — | device half; lands with the Metal PSO table |

Arguments preserved: `quant` has no default (half-supplied formats
compiled, bound, dispatched, and answered wrongly); the routed GEMM's
FP16 form is a NAME choice decided by tuning, identical in contract, and
llama — whose routed top-k moved under FP16 — never reaches this table;
the wide matvec follows the CHECKPOINT's format rather than the fp16
gate, because tying it to `fp16_strided` left an alt-quant kind with no
batched shape at all (pinned by test: g64/b8 still gets one).

## The golden taps — `src/batch/golden.rs`

`batch/golden_tap.hpp/.cpp` (316 together): the env-gated activation dump
the accuracy gate diffs against the MLX reference, tap by tap.

| C++ | Rust | |
|---|---|---|
| `tap_for` (the kernel→name/bind/width table) | `tap_for` | ported, with its three stories: GatedRms is `gdn_core` (the reference taps after the gate RMSNorm); routed SiluMul is `shared_act` at its own width (the `swiglu` tap was present-but-empty on every routed checkpoint); the mixture taps exist because the routed FFN was the one block no parity run could see |
| `Dispatch` parameter | `TapSite { kind, layer }` | ported narrower: tapping reads two fields, so it asks for two — the family `Dispatch` needn't exist yet |
| `write_npy` (silent on failed open) | `write_npy -> io::Result` | ported; the silence is deleted, not preserved — a "successful" dump that left nothing behind cost a bisect once, which is also why `dir_from_env` still mkdirs |
| `golden_tap_dir()` static | `dir_from_env()` | ported; the create failure is returned, not swallowed |
| `golden_taps_recycle` | `taps_recycle` | ported with its argument (the one defect class the no-recycle dump cannot see) |
| `dump_golden_taps` | `dump_taps` (unsafe, over `[R: Region]`) | ported; the last-writer-of-a-colour rule and its reason (in-place rewriters would publish the later tensor under the earlier name) pinned by test |
| `dump_golden_bf16` / `_sorted` / `_tokens` | `dump_bf16` / `dump_bf16_sorted` / `dump_tokens` | ported over slices; the sorted dump's perm-not-recomputed argument kept (within-expert order is atomics-decided) |

The C++ silently skipped a row whose read left the pool slot; the Rust
zero-fills it after a named bounds check — a dump is diagnostic, but a
short slot is now visible in the data instead of shortening it.

## The geometry — `src/batch/geometry.rs`

`DecodeGeometry` (`model/qwen3_5/geometry.hpp`, generic despite its path)
and `AffineFormat` (declared beside the quant kernels, shared with them).

| C++ | Rust | |
|---|---|---|
| `AffineFormat` / `kernel_suffix` | `AffineFormat` / `kernel_suffix` | ported |
| `DecodeGeometry` fields + defaults | `DecodeGeometry` / `Default` | ported |
| `is_full_attn` / `full_attn_layers` | same, interval<=1 semantics kept | ported |
| `gdn_conv_stride_bytes` / `gdn_recurrent_stride_bytes` | same | ported |
| `is_moe` / `has_shared_expert` / `ffn_width` | same | ported |
| `has_alt_quant` | `AffineFormat::is_set` + `has_alt_quant` | ported |

The stories stay on the fields that carry them: the affine width and
group are one fact (g64/b8 and g128/b4 pack identically; a pipeline built
for the wrong pair "compiled, bound, dispatched, and lied" — token 3504,
repeated), and `alt_quant` exists because mlx_lm spares the two routing
projections at 8 bits inside a 4-bit body — read as 4-bit they route to
almost the right experts (cosine 0.84) and weight them wrongly.

The deferral recorded here previously is closed: the scratch footprints
landed as `src/batch/sizing.rs` (below) and `plan_heap` as
`src/loader/heap.rs` (see `PARITY-LOADER.md`). `Tuning::moe_tile_rows`
already existed in `src/tuning.rs`; only the kernel bound needed porting.

## Scratch sizing — `src/batch/sizing.rs`

The sizing half of `batch/scratch.hpp`, plus the two helpers it reaches:
`pie::kernels::moe::sorted_rows` (`kernels-metal/include/pie/kernels/
moe.h`) and `moe_sorted_rows` (`model/qwen3_5/decode_step.hpp`, generic
despite its path).

| C++ | Rust | |
|---|---|---|
| `moe::sorted_rows(pairs, experts, tile)` | `sorted_rows` | ported |
| `moe_sorted_rows(g, n_tokens, batched)` | `moe_sorted_rows(g, tuning, n, RoutedProjection)` | ported |
| `shared_kernels::moe_sorted_rows(pairs, experts)` | — | dropped: it only glued the tile lookup to the bound; the two calls read better than a third name |
| `scratch_widest_elems(g)` | `scratch_widest_elems(g, tuning)` | ported |
| `scratch_slot_elems(g, max_tokens)` | `scratch_slot_elems(g, tuning, max_tokens)` | ported |
| `batched: bool` parameter | `RoutedProjection { Matmul, Matvec }` | ported: a bare `true` at a call site says nothing; the enum says which projection runs |
| `ScratchBind` / `ScratchDispatch` / the schedule | `ScratchBind` / `ScratchSchedule` / `schedule_scratch` (`color.rs`) | ported — see below |

The arguments preserved: the slot width was derived twice (here and in the
heap layout) and the copies drifted — 8320 elements against a 16384-element
row pitch, every row past the halfway point writing into the next colour —
so the second derivation is deleted, not synchronized. The mixture's stack
does not scale linearly with tokens (12800 rows where the linear bound says
5120, pinned by test), and `sorted_rows` takes its tile as a parameter
because a bound the kernel guarantees must not depend on a tuning decision.

The kernel bound has no Rust home on the kernels side yet: `kernels-metal`
is a signature table. When it grows launch-shape helpers, `sorted_rows`
should migrate there; until then the doc names its authority
(`moe_route.metal`'s sort).

`model/family_coloring.hpp` (72) folds into `color.rs` as
`schedule_scratch`: the C++ adapter was a template because every family
declared its own `Use` struct with identical fields, and widening them was
half its body. Rust families produce the shared `Use` directly, so what
remains is the fan-out from the per-value colouring to per-dispatch bind
lists — plus the argument for `color_of_value` travelling alongside
(sizing needs the widest VALUE sharing a colour; a routed expert stack is
`experts_per_token` times taller than the dense tensor beside it, and no
bind index shows that). Its `hazard_free` flag stays unrepresentable: a
hazard is an `Err` through the adapter too, pinned by test. A use whose
ordinal is past the DAG is refused (`OrdinalPastDag`) where the C++
indexed the table with it.

## Expert paging — `src/batch/paging.rs`

The portable half of `batch/expert_paging.hpp` (195), now that
`ExpertSlab` exists (`src/loader/slab.rs`).

| C++ | Rust | |
|---|---|---|
| `ExpertPaging::plan` | `plan_paging(cuts, dag_size, SlabShape, …)` | ported; five refusals become four named `PagingRefused` variants |
| ids-not-host-readable refusal | — | stays with the device half: a `SlotHandle`'s readability is a Metal fact |
| the in-place id rewrite inside `fire` | `renumber_routing` | ported; takes the slab's `ensure_resident` as a closure, so the rewrite is tested without a device |
| `fire`'s segment loop / `run_segments` | — | missing: drives a command queue; lands with the `src/metal/` paging glue. The pins-back-FIRST rule is stated in this module's docs because it is a budget fact, not a queue fact |
| `PIE_METAL_PAGING_TRACE` stderr dump | — | dropped: the crate denies `print_stderr`; a caller that wants the trace logs the buffer it owns |
| `[pie-metal] … experts paged through …` banner | `PagingPlan::worst_case_experts` + slab accessors | dropped as a print; the numbers it printed are readable off the plan and the slab |

Arguments preserved: the worst case is every expert ONE dispatch can read
(`min(n_experts, rows × experts_per_token)`) resident at once -- there is
no order in which a smaller cache could serve it, which is also why the
slab never needs to exceed one layer's bank. The strided-vs-packed story
is pinned by test: reading a strided prefill as packed renumbers row 0
`rows` times and leaves the rest holding true expert ids -- fluent wrong
text, not an error. `renumber_routing` refuses a short buffer before
touching a byte, because a partial rewrite is a state nobody asked for.

## The decode-step ABI — `src/batch/abi.rs`

The vocabulary half of `decode_abi.hpp`: regions, IO slots, the kernel-kind
enum, the argmax params and the graph key.

| C++ | Rust | |
|---|---|---|
| `Region` | `Region` | ported |
| `SCRATCH_POOL` | `SCRATCH_POOL` | ported |
| `IoSlot` / `kIoSlotCount` | `IoSlot` / `IO_SLOT_COUNT` | ported |
| `ArgmaxParams` | `ArgmaxParams` (+ size pin) | ported |
| `Kernel` (98 kinds) | `Kernel`, macro-derived | ported |
| `Kernel::KindCount` | `Kernel::COUNT` via `Kernel::ALL` | ported |
| `ForwardGraphKey` / `PAGE_BUCKET_GRAN` | `ForwardGraphKey::of` | ported |
| the ~30 `bind::` layouts | — | missing: each is one kernel's ABI and lands beside the encoder that binds it |

The argument is the count that was forty kinds short: `KindCount` was once
spelled `G4PleResidual + 1`, so `psos[LmHeadUntied]` indexed past every
kind-sized table — the untied head ran the wrong pipeline, every logit
zero, every token 0, not one error anywhere. The C++ fix made the count an
enum member; the Rust fix derives `ALL` and `COUNT` from the same token
list the variants come from, so there is no second spelling of the end to
fall behind, and a `[T; COUNT]` table indexed through a `Kernel` value
cannot be indexed past. The numeric values are ABI ("APPEND ONLY" five
times over), so eleven anchor discriminants pin every block boundary — an
insertion upstream of one fails loudly instead of renumbering silently.

| C++ | lines | |
|---|---|---|
| `compose.cpp` rest: `LaunchMember`, `LaunchJobData`, tickets | ~90 | missing — the job container, with the worker port |
| `scratch.hpp` / `scratch.cpp`: `build_scratch_schedule`, `bind_scratch`, the footprint helpers | ~540 | missing — coupled to `DecodeGeometry`/`Dispatch`, with the family port |
| — | — | — (`decode_timing` ported below) |

## The attribution — `src/batch/timing.rs`

From `decode_timing.cpp/.hpp` (365 lines), plus the kind names moved to
where the kinds live.

| C++ | Rust | |
|---|---|---|
| `attribute_step` | `attribute_step` | ported |
| `StepAttribution` / `DispatchAttribution` | same names | ported |
| `StepAttribution::valid` | `Result` + `BoundaryMismatch` | dropped |
| `kernel_name` | `Kernel::name`, macro-total | ported |
| `kernel_ablated` | `Ablation::parse` / `from_env` / `ablated` | ported |
| `print_attribution` | `StepAttribution::report` | ported |
| the `Dispatch` subset it reads | `DispatchInfo` | ported |

`kernel_name` was a hand-kept switch with `default: return "unknown"`, and
for a while 50 of the 99 kinds fell through it — the attribution report was
blind to gemma4's mixture and PLE, all of gpt-oss and both untied kinds at
once, and the ablation knob could not name any of them. The name is now an
argument of the `kernels!` macro: a variant without one does not compile,
and `from_name` is its exact inverse (tested unique and total). The one
legacy exception, `AttnGate = "gate"`, is pinned.

The ablation spec parser keeps its hard-won lesson — a typo'd token
"ablates NOTHING and this run reports the baseline while looking armed" —
but returns the unmatched tokens instead of printing them (`parse` is
env-free and testable; `from_env` is the one place the environment is
read). The substring-with-boundary-checks walk becomes an exact per-token
lookup. `valid = false` becomes an `Err` carrying both counts, and the
`FILE*` report becomes a returned `String`, because this crate denies
stdout/stderr by policy.

Five timing tests plus the abi name test; the monotonic guard (a clock
wrap attributes zero, not a negative share) is kept and tested.
| `expert_paging.hpp` | 195 | missing — `fire` needs `ExpertSlab` (loader) |
| `scratch.cpp` / `scratch.hpp` / `scratch_color.hpp` | 650 | missing |
| `batch_schedule.hpp` (done above) | — | — |
| — | — | — (`decode_psos`'s M=1 half ported below) |

## The PSO plan — `src/batch/psos.rs`

The pure half of `decode_psos.cpp`: which `(file, entrypoint)` pairs a
configuration compiles and which kinds each serves. The metal half is
`Compiler::compile_batch`, which already exists.

| C++ | Rust | |
|---|---|---|
| `PsoSpec` / the `want` gathering | `PsoRequest` / `plan_decode_psos` | ported |
| `DecodePsoFeatures` | `Features` | ported |
| the format-dependent `entrypoint()` names | `EntryNames`, table-checked | ported |
| `DecodeStepPsos` fan-out | `DecodePsoPlan::source_of` | ported |
| `load_decode_psos` (the compile loop) | `Compiler::compile_batch` + `source_of` | dropped |
| `load_multibatch_psos` / `MultiBatchPsos` | — | missing: the qmm tile grammar and tuning constants land with the family port |

The C++ `entrypoint()` refuses a name no shader instantiates, so an
uninstantiated format fails at load naming the formats that exist instead
of inside the Metal compiler. The plan holds the same line one step
earlier and on any host: a dev test validates every emittable name against
`kernels-metal`'s signature table (a new `default-features = false`
dev-dependency — the table, no Metal) and every file path against the
shipped tree. Each feature flag keeps its load-bearing absence documented
(`untied` once handed llama a wrong-format pipeline that answered wrongly;
`routed` would let an unrelated shader error fail a dense load), fan-out
order is later-claims-win (the GDN recurrent override), and `routing_only`
still clears the world down to the two second-format projections.

Five portable tests: the 25-kind base surface, the full feature set with
single-claim disjointness, the override ordering, `routing_only`, and the
signature-table validation.
| `golden_tap.cpp` | 238 | missing |
| — | — | — (`worker.hpp` ported below) |

## The executor worker — `src/batch/worker.rs`

| C++ | Rust | |
|---|---|---|
| `ExecutorWorker` | `Worker<S>` | ported |
| `run` / `post` / `drain` | `run` / `post` / `drain` | ported |
| `submitted` | `submitted` | ported |
| the same-thread inline re-entry | a refusal with instructions | dropped |
| `worker_thread_id` | — | dropped |

The C++ serializes every executor touch through one FIFO thread — the
thread-affinity Metal requires and the forward/control-op exclusion in one
mechanism — but the guarantee holds only *as long as everyone remembers to
go through the worker*: the context pointer stays reachable from any
thread. `Worker<S>` closes that by ownership: the state is constructed ON
the worker thread by a factory and never leaves it, jobs receive `&mut S`,
and `S` need not be `Send` — which is exactly what lets it hold the
runtime's `Rc`s and a `Stepper`. "Another thread touched the context" goes
from a discipline to unrepresentable, and a test proves an `Rc`-holding
state works.

`run` still resumes the job's panic on the caller (the C++ rethrows the
captured exception) and the worker survives; `post` contains panics so one
bad job cannot tear the thread down; drop drains before stopping. What did
not survive is inline same-thread re-entry: a Rust job already holds
`&mut S`, an inline nested job would alias it, so re-entry refuses with
instructions instead of deadlocking. `worker_thread_id` existed for that
inline check and for tests; the refusal owns the former and the tests ask
the worker directly.

Five portable tests: the `!Send`-state fence, FIFO + drain-as-barrier,
panic resume + survival, contained post panics, drop-as-barrier.
| `simple_family.cpp` / `.hpp` | 2176 | missing |
| `forward.cpp` / `forward.hpp` | 5393 | missing — the executor; last, over everything above |
