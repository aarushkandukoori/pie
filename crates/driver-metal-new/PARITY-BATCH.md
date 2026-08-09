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

## Not yet started

| C++ | lines | |
|---|---|---|
| `compose.cpp` rest: `build_member_forward_desc`, `LaunchMember`, tickets | ~380 | missing — needs `MemberForwardDesc` (`forward.hpp`) |
| `scratch.cpp` / `scratch.hpp` / `scratch_color.hpp` | 650 | missing |
| `batch_schedule.hpp` (done above) | — | — |
| `decode_abi.hpp` | 650 | missing |
| `decode_psos.cpp` / `.hpp` | 582 | missing |
| `decode_timing.cpp` / `.hpp` | 365 | missing |
| `expert_paging.hpp` | 195 | missing |
| `golden_tap.cpp` | 238 | missing |
| `worker.hpp` | 171 | missing |
| `simple_family.cpp` / `.hpp` | 2176 | missing |
| `forward.cpp` / `forward.hpp` | 5393 | missing — the executor; last, over everything above |
