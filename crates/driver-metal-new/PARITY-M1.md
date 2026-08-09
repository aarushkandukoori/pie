# Pipeline parity: `pipeline/m1_runtime.cpp` against `driver-metal-new`

The C++ is 3411 lines plus a 202-line header, and it is the only complete
statement of what the launch path does. Same rules as `PARITY.md`: every entry
is **ported**, **dropped** (with a reason that says why the C++ needed it and
the Rust does not), or **missing** (with what is blocking it). Nothing is
"ported" because a function with a similar name exists.

The port is in progress. The portable half — everything that is a function of
the plan and the fire's numbers rather than of the device — goes first, because
it is the half that can be tested without a GPU and it is where the C++'s
mistakes are.

## Cache identity — `src/pipeline/identity.rs`

| C++ | Rust | |
|---|---|---|
| `encode_m1_cache_identity` | `cache_identity` | ported |
| `encode_cache_identity` | — | dropped |
| `M1CacheIdentityVersions` | `Versions` | ported |
| `combined_signature` | `combined_signature` | ported |
| `fnv1a64` | `tensor_ir::fnv1a64` | dropped |
| `hex64` | — | dropped |
| `identity_bytes` | — | dropped |

`encode_cache_identity` is the two-argument wrapper that filled `Versions` from
`PTIR_COMPILER_VERSION` and friends. Those live in the compiler's headers and
this crate deliberately does not depend on the compiler, so the versions are a
parameter and the fill belongs to whoever assembles the driver. The emitter
version in particular must come from `ProgramRegistration::emitter_version`
rather than a driver-side copy: the C++'s copy said 23 while the host said 36.

`fnv1a64` is dropped because `tensor-ir` already owns it and the CUDA driver and
host program cache reach the same number through it. `hex64` and
`identity_bytes` are `format!("{:016x}")` and `u64::to_le_bytes`.

## Value shapes — `src/pipeline/extent.rs`

| C++ | Rust | |
|---|---|---|
| `M1RuntimeExtents` | `Extents` | ported |
| `symbolic_extent` | `Role` + `Extents::get` | ported |
| `describe_value` | `describe` | ported |
| `DeviceValueDesc` | `ValueDesc` | ported |
| `value_bytes` | `ValueDesc::device_bytes` | ported |
| `wire_value_bytes` | `ValueDesc::wire_bytes` | ported |
| `m1_extents_from_forward_desc` | — | missing |
| `m3_extents_from_forward_desc` | — | missing |
| `resolve_m1_shape_for_test` | — | dropped |
| `M1ResolvedShape` | — | dropped |

The two `*_from_forward_desc` constructors are field copies out of
`batch::MemberForwardDesc`, which has no Rust counterpart yet; they belong with
the `batch/` port and are missing rather than dropped.

`resolve_m1_shape_for_test` and its `M1ResolvedShape` exist because
`describe_value` is in an anonymous namespace and a test cannot reach it. It is
`pub` here and tested directly, so the hook has nothing to do.

Three C++ behaviours are refusals here: an unrecognised extent role (was
`return 1`), a rank past four (was silent truncation, which drops a factor from
the element count and under-sizes every allocation derived from it), and a
32-bit `len * 4` that reported zero bytes for a value of 2^30 f32 lanes.

## Scratch layout — `src/pipeline/scratch.rs`

| C++ | Rust | |
|---|---|---|
| `align_up` | `align_up` (private) | ported |
| `kMaxScratchBytes` | `MAX_BYTES` | ported |
| the value-offset loop in `execute` | `layout` | ported |
| the `dummy` subhandle at offset 0 | `DUMMY_BYTES` | ported |

The C++ accumulated the total with unchecked `+=` and tested the bound after,
so a wrapped total passed a check the real one would have failed. Every step is
checked here. The placeholder descriptor the C++ pushed onto an empty list is
not part of the layout — it is what keeps the *buffer allocation* non-empty, so
it belongs at the allocation site.

## Op parameters — `src/pipeline/params.rs`

| C++ | Rust | |
|---|---|---|
| `DeviceOpParams` | `OpParams` | ported |
| the record fill in `execute` | `OpParams::of` | ported |
| the record fill in the M2 path | `OpParams::of` | ported |
| `op.args.size() > 1 \|\| tag == PIVOT_THRESHOLD` | `binds_second_argument` | ported |

The C++ wrote the record twice — once in the M1 `execute` loop and once in the
M2 command builder — with the two copies agreeing by inspection. One function.
`sink_bytes` stays zero: it comes from the bound channel cell, not from the op.

## Readiness — `src/pipeline/readiness.rs`

| C++ | Rust | |
|---|---|---|
| `check_readiness_host` | `check` | ported |
| `M1PrepareOutcome` | `Readiness` | ported |
| `M1ChannelEffect` | `Effect` | ported |
| `batch::ChannelTicket` | `Ticket` | ported |
| `kNoTicket` | `NO_TICKET` | ported |

Every outcome in the C++ was a string with the channel index and the failure
kind encoded as arithmetic (`0x200 + channel` permanent, `0x300` early, `0x500`
an unorderable put). Nothing parsed them back, so the distinction was lost the
moment it was made. `Reason` names each case and `is_permanent` answers the
question the base addresses were encoding.

## M3 grouping — `src/pipeline/group.rs`

| C++ | Rust | |
|---|---|---|
| `m3_schedule_bucket` | `schedule_bucket` | ported |
| `m3_stage_key` | `GroupKey::of` | ported |
| `M1Runtime::m3_stage_group_key` | `GroupKey::of` | ported |
| `m3_used_channel_slots` | `used_channel_slots` | ported |
| `m3_channel_flags` | `channel_flags` | ported |
| `kM3Channel*` | `CHANNEL_*` | ported |
| `kMetalM1MaxChannels` | `MAX_CHANNELS` | ported |

The key was a `reinterpret_cast` of a `u64` into a `std::string` with a byte
pushed on; it is two numbers. "No key" was the empty string, which is itself a
usable map key, from three different causes; it is `None`.

`used_channel_slots` gains the bound the C++ applied to the declared channel
count and not to this one, even though this is the count that gets bound.

## Compile cache — `src/pipeline/cache.rs`

| C++ | Rust | |
|---|---|---|
| `programs` / `stage_cache` / `negative` | `Bounded` | ported |
| `kMaxProgramCacheEntries` etc. | `MAX_*_ENTRIES` | ported |
| `M1CompileFailureKind` | `Failure` | ported |
| `remember_negative` | `Bounded::insert` | ported |
| `M1CacheStats` | `Stats` | ported |
| `set_program_cache_capacity_for_test` | `Bounded::new` | dropped |
| `inject_stage_cache_entry_for_test` | `Bounded::insert` | dropped |

The two test hooks are dropped because the thing they inject into is a public
type with a public constructor here; the C++ needed them because the caches
were private members of an `Impl` behind a pimpl.

The behaviour change is the point of the slice: the C++'s positive caches never
evicted, and a full one returned a *retryable* failure — so the sixty-fifth
distinct program a process saw could never run, and the caller retried forever
against the one condition retrying cannot change. The negative cache evicted
`begin()`, which is neither the oldest nor the coldest entry. All three are LRU.

## Bind-time derivation — `src/pipeline/meta.rs`

| C++ | Rust | |
|---|---|---|
| `collect_singleton_metadata` | `op_metadata` | ported |
| `M1OpMeta` | `OpMeta` | ported |
| the inline walk in M2 validate | `op_metadata` | dropped |
| the inline walk in the M2 builder | `op_metadata` | dropped |
| the inline walk in the M3 builder | `op_metadata` | dropped |
| the `effects.resize` loop | `channel_effects` | ported |

The three dropped entries are the same running sum written out by hand at
three more call sites, each maintaining its own `result_base` local. They are
dropped in the sense that they do not become three Rust functions.

`M1OpMeta` carried a by-value copy of the whole `PlanOp` — a struct with two
vectors — so binding duplicated the op list next to the list it walked. The
Rust holds `node` and reads the op through it.

The walk gains two refusals. The base accumulated in an unchecked `uint32_t`,
and a wrapped base is not a large index that a later bounds check catches but a
small one that passes and aliases another op's results. And the header states
that the walker "assumes the plan is well-formed", justified by the host
validating first — true of the path the host emitted, not of a plan arriving
over the ABI, and the check is one comparison.

`channel_effects` gains a consistency check between a channel's declared
readiness and the ops that touch it. `PIE_READINESS_UNTOUCHED` on a channel
something takes means the gate the take needs was never computed, and the C++
would run the take against a ring it never checked was non-empty. A capacity of
zero is full and empty at once, so both gates are unsatisfiable; the C++
defaulted the field to 1 and then overwrote it with the plan's zero.

## Device status — `src/pipeline/status.rs`

| C++ | Rust | |
|---|---|---|
| `DeviceStatus` | `Status` | ported |
| the M1 status decode (1945–1968) | `Outcome::of` + `report` | ported |
| the M2 status decode (2385–2408) | `Outcome::of` + `report` | dropped |
| the M3 lane report (3169–3220) | `Outcome::of` + `report` | ported |
| the `site ==` chain | `Site` | ported |
| `static_assert(sizeof(DeviceStatus) == 16)` | `STATUS_BYTES` + a test | ported |
| — | `FAULT_CLASSES`, `describe_fault` | added |

Three copies of the decode, agreeing on `state == 4` and `state == 2` and on
nothing else. The M1 copy printed the fault in decimal — `160` for a code the
whole rest of the system writes as `0xA0` — and discarded `reserved0` and
`reserved1`, so the guard site the kernel deliberately recorded was thrown
away. The M3 copy printed hex and decoded the site. Same kernel, same fault,
two reports.

All three treated "not 4 and not 2" as an op fault, which swallows `state = 0`
(the kernel wrote nothing) and `state = 1` (the kernel started and stopped)
into "generated op fault 0". The M3 path had learned half of this — it guards
on `encoded`, because a group that never dispatched reads back as a lane-wide
zero fill and produced a GPU fault report for something the GPU was never asked
to do. The M1 path never learned it. `Diagnosis` separates all four.

`describe_fault` is new. `codegen/fault.rs` declares every code a kernel can
write, with the per-channel classes and the two that alias op tags, and its own
module doc says "Nothing decodes these: the drivers surface the number and a
human reads it". The table exists; the driver may as well read it. The mirror
is checked against `tensor_compiler::codegen::fault::CLASSES` in a test, with
the compiler as a dev-dependency only, so the copy cannot drift.

## Stage cache and its collision guard — `src/pipeline/stage_cache.rs`

| C++ | Rust | |
|---|---|---|
| `Impl::stage_cache` | `Stages` | ported |
| `pending_stages` | `Stages` pending half | ported |
| the guard against `stage_cache` | `Stages::lookup` | ported |
| the guard against `pending_stages` | `Stages::lookup` | dropped |
| `M1StageExecutable::stage_identity` | `Entry::identity` | ported |
| `identity_bytes` | — | dropped |
| `default_m1_cache_dir` | `metal::Archives::discover` | dropped |

Keying a stage on a hash and storing a second, independent identity beside it
to check after a hit is the right design and the C++ had it. What it did with a
detected collision is the defect: `reject_deterministic`, which is the
classification that says *this program* cannot compile and never will, and is
the classification the negative cache remembers. A collision is not a property
of the program being compiled — it is a property of which other program holds
the slot. The C++ blamed a program for a collision it did not cause and then
wrote the verdict down. A collision here evicts the incumbent and returns a
miss, and `Stages::collisions` counts it so the rate stays visible.

The guard was written out twice, identically, once for each map; the pending
map exists so a compile that fails partway leaves the cache untouched, which is
`commit`/`abandon` here rather than a second map with a second copy of the
guard. `stage_identity` was a `std::vector<std::uint8_t>` holding a `u64`'s
eight bytes, heap-allocated per entry to compare a number.

The capacity check was the program cache's mistake again — `size() + pending >=
max` returning a retryable failure — and is gone for the same reason.

`default_m1_cache_dir` is dropped because `metal::Archives::discover` already
does it, and does it better: the C++'s last resort was
`return ".pie-metal-ptir-cache"`, a *relative* path, so a process started
without `HOME` scattered a compile cache into whatever directory it happened to
be launched from. `Archives` has no cache at all in that case, which is the
honest answer.

## Emitted-kernel index — `src/pipeline/emitted.rs`

| C++ | Rust | |
|---|---|---|
| `HostEmittedKernels` | `Emitted` | ported |
| `HostEmittedKernels::find` | `Emitted::get` | ported |
| `HostEmittedKernels::Key` | the map's tuple key | ported |
| `HostEmittedKernels::KeyHash` | — | dropped |
| the `error`-before-`source` convention | `Slot` | ported |

`emplace` on an `unordered_map` keeps the entry already present and drops the
new one, silently. So a host that emitted two kernels for one `(kind, stage,
region)` got whichever came first in the vector — a choice between two kernels
made by array order, by a driver with no way to know which the host meant, and
if the two differ at all one of them is wrong. `Emitted::index` reports it.

The three states `EmittedKernel` packs into two strings become `Slot`'s
variants. The C++ has them right and says so in a comment on the container:
callers must read `error` before `source`, because an empty source with a
populated error is a *deliberate* refusal that the driver answers with its
slower path rather than a failure. That comment is not next to any of the call
sites that must obey it. The order is inside `get` here. `Slot::Malformed` is
the fourth state the C++ had no name for — both strings empty, which `find`
returned like any other entry and the caller compiled as `""`.

`KeyHash` is dropped rather than ported: it packed `stage << 24` over a
full-width `region`, so `(stage 1, region 0)` and `(stage 0, region
0x0100_0000)` hashed alike. The map compared full keys, so this cost lookups
rather than correctness — but it is a hand-written hash with a bug in it and
the standard one has neither.

## Already covered elsewhere in the crate

| C++ | Rust | |
|---|---|---|
| `inline_ptir_rng_preamble` | `shader::splice_with` | dropped |
| `kPtirRngInclude` | `shader::DIRECTIVE` | dropped |
| `default_m1_cache_dir` | `metal::Archives::discover` | dropped |
| `fnv1a64` | `tensor_ir::fnv1a64` | dropped |
| `align_up` | `scratch` (private) | dropped |
| `wire_value_bytes` | `pipeline::wire_cell_bytes` | dropped |

`inline_ptir_rng_preamble` is a `find`/`replace` loop over the literal text
`#include "ptir_rng.generated.metal"`, anywhere it appears, mutating the string
under the cursor it is scanning with. `shader::splice_with` was already written
against the same requirement and is stricter in the two ways that matter: it
honours a directive only at column zero, so the same characters inside a
comment or a string literal are left alone, and it builds the output forward
so the scan never revisits text a replacement introduced. It also handles
nested includes and bounds the depth, neither of which the C++ attempts.

## The buffer view — `src/metal/handle.rs`

| C++ | Rust | |
|---|---|---|
| `SlotHandle` | `Handle` | ported |
| `subhandle` | `Handle::slice` | ported |
| `external_handle` | `Handle::over` | ported |
| `SlotHandle::valid` | — | dropped |
| `SlotHandle::offset` | — | dropped |
| `SlotHandle::elastic` | — | dropped |

The first metal-side slice, and the type everything after it stores and binds.
Its tests are in `tests/device_handle.rs` and need a device, including one that
dispatches a kernel through a sliced address to prove the GPU lands where the
host pointer says.

`subhandle` checked nothing: a span past the base was minted rather than
refused, and a default (invalid) base is `nullptr + offset` — UB that in
practice fabricates a handle whose GPU address *is* the offset, which an
argument table binds like any other number. `slice` refuses the first with the
wrap-safe bound every `Region` uses; the second is unrepresentable, because an
invalid `Handle` is not a value of the type and "no handle yet" is
`Option<Handle>`. That is also why `valid()` is dropped.

`offset` was written at every construction and read nowhere on the launch
path; a diagnostic that wants it is one subtraction away. `elastic` is dropped
because it was per-copy state — a flag saying what type the buffer really was —
and the C++'s own `subhandle` demonstrates the failure mode: its designated
initializer names five of the six fields, so a sub-range of an elastic base
would come out ordinary and pass the `bytes <= size` capacity test with no
pages behind it. Elastic-ness here is the `Elastic` type, which a view cannot
mislay. `external_handle` additionally trusted `device_visible()` without
checking it, so a host-fallback ring would bind as GPU address zero; `over`
starts from a real `MTLBuffer` and refuses one the host cannot address.

The ownership flips from borrow to retain: the C++ view is "borrowed; lifetime
owned by RawMetalContext", a contract kept by hand at every copy. A `Handle`
retains its buffer, so the allocation cannot be freed while a view names it —
what retaining does not answer for is exclusivity over a recycled pool buffer,
which is why a handle still belongs beside the owner it was derived from.

## Not yet started

Everything below names a Metal type and will land under `src/metal/`. That is
why the split falls here rather than at a line number in the C++: everything
above tests on any machine, and everything below needs a device.

| C++ | lines | |
|---|---|---|
| `M1RegionExecutable` … `M3GroupCommand` | 388–546 | missing |
| `bind_m2_*` / `bind_m3_*` | 654–735 | missing |
| `PsoCompileTransaction` | ~700 | missing |
| `compile_program` | 736–1454 | missing |
| `prepare` / `execute` (M1 singleton) | 1455–1981 | missing |
| M2 fused placement | 1982–2411 | missing |
| M3 grouped lanes | 2412–3350 | missing |

## Where this stands

Twelve subjects ported, each one argued from a specific defect in the C++
rather than from a wish to have it in Rust. The portable half of
`m1_runtime.cpp` — everything that is a function of the plan and the fire's
numbers rather than of the device — is done, and it carries 122 tests that run
without a GPU. The C++ had none for any of it: every one of these functions
lived in an anonymous namespace behind a pimpl, reachable only through a
`*_for_test` hook or not at all.

The metal half has begun with the buffer view, whose seven tests need a
device. Everything still missing above builds on it.
