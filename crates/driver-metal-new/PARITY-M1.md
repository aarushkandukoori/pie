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

## Not yet started

| C++ | lines | |
|---|---|---|
| `M1RegionExecutable` … `M3GroupCommand` | 388–546 | missing |
| `PsoCompileTransaction` | ~700 | missing |
| `compile_program` | 736–1454 | missing |
| `prepare` / `execute` (M1 singleton) | 1455–1981 | missing |
| M2 fused placement | 1982–2411 | missing |
| M3 grouped lanes | 2412–3350 | missing |
| `inline_ptir_rng_preamble` | ~125 | missing |
| `HostEmittedKernels` | ~150–192 | missing |
| `subhandle` / `external_handle` | ~194–215 | missing |
| `DeviceStatus` + fault decoding | 81–87 | missing |
| `bind_m2_*` / `bind_m3_*` | 654–735 | missing |

Everything above this section is portable and tests without a GPU. Everything
below it names a Metal type and will land under `src/metal/`, which is why the
split falls where it does rather than at a line number in the C++.
