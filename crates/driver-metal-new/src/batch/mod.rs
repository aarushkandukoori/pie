//! The batch subsystem: scheduling, composition, and the forward's shape.
//!
//! `csrc/src/batch/` is ~11.6k lines and is mostly not about the GPU: it
//! derives batch shapes from the CSR view the engine marshals, composes
//! channel tickets, colors scratch, and only at the end encodes a forward.
//! The port follows the crate's rule — portable half first, into modules a
//! Linux `cargo test` reaches — and `PARITY-BATCH.md` is its ledger.
//!
//! [`schedule`] is the batch shape: request spans, the token→request
//! expansion, and the paged-geometry gate that runs before any pool cell can
//! be addressed. [`mask`] answers whether a wire attention mask says
//! anything the kernel's own causal predicate does not already enforce.

mod abi;
mod binds;
mod admit;
mod color;
mod dispatch;
mod geometry;
mod golden;
mod mask;
mod member;
mod paging;
mod psos;
mod psos_mb;
mod schedule;
mod sizing;
mod timing;
mod worker;

pub use abi::{
    ArgmaxParams, ForwardGraphKey, IO_SLOT_COUNT, IoSlot, Kernel, PAGE_BUCKET_GRAN, Region,
    SCRATCH_POOL,
};
pub use admit::{Refused, admit_recurrent};
pub use binds::{WeightBind, layer_prefix, weight_binds};
pub use color::{
    Coloring, ColoringError, ScheduleError, ScratchBind, ScratchSchedule, Use, color_live_ranges,
    schedule_scratch,
};
pub use dispatch::{
    DagOptions, Dispatch, Launch, barrier_after, build_decode_dag, concurrent_run_ends,
};
pub use geometry::{AffineFormat, DecodeGeometry};
pub use golden::{
    Tap, TapSite, dir_from_env, dump_bf16, dump_bf16_sorted, dump_taps, dump_tokens,
    sorted_dump_rows, tap_for, taps_recycle, write_npy,
};
pub use mask::{Disagreement, causal_prefix_lengths, kv_len_disagreement};
pub use member::{BuildError, ForwardDesc, ResolvedGeometry, build_member_desc};
pub use paging::{
    Cut, PagingPlan, PagingRefused, RenumberRefused, SlabShape, plan_paging, renumber_routing,
};
pub use psos::{DecodePsoPlan, EntryNames, Features as PsoFeatures, PsoRequest, plan_decode_psos};
pub use psos_mb::{
    MOE_TILE_WIDTHS, MbFeatures, MbRequest, MbSlot, QMM_BMS, QMM_SPLIT_BN, plan_multibatch_psos,
};
pub use schedule::{
    BatchSchedule, DEFAULT_PAGE_SIZE, Malformed, Rejected, RequestSpan, build_schedule,
    find_request, validate_capacity, validate_paged,
};
pub use sizing::{
    RoutedProjection, moe_sorted_rows, scratch_slot_elems, scratch_widest_elems, sorted_rows,
};
pub use timing::{
    Ablation, BoundaryMismatch, DispatchAttribution, DispatchInfo, StepAttribution, attribute_step,
};
pub use worker::Worker;
