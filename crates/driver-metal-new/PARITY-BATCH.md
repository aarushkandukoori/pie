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
3. `expert_paging.hpp` (195): `plan`'s validation is pure modulo a
   three-field slab shape, but `fire` needs `ExpertSlab` (the loader port)
   and host-callback segments; port whole when the loader lands.
4. `decode_psos` (582), `golden_tap` (238), `worker.hpp` (171),
   `simple_family` (2176), then `forward.cpp/.hpp` (5393) over everything.

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
| `decode_timing.cpp` / `.hpp` | 365 | missing — consumes `Kernel`/`Dispatch` |
| `expert_paging.hpp` | 195 | missing — `fire` needs `ExpertSlab` (loader) |
| `scratch.cpp` / `scratch.hpp` / `scratch_color.hpp` | 650 | missing |
| `batch_schedule.hpp` (done above) | — | — |
| `decode_psos.cpp` / `.hpp` | 582 | missing |
| `golden_tap.cpp` | 238 | missing |
| `worker.hpp` | 171 | missing |
| `simple_family.cpp` / `.hpp` | 2176 | missing |
| `forward.cpp` / `forward.hpp` | 5393 | missing — the executor; last, over everything above |
