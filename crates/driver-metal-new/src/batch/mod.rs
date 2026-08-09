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
mod admit;
mod color;
mod mask;
mod member;
mod psos;
mod schedule;
mod timing;
mod worker;

pub use abi::{
    ArgmaxParams, ForwardGraphKey, IO_SLOT_COUNT, IoSlot, Kernel, PAGE_BUCKET_GRAN, Region,
    SCRATCH_POOL,
};
pub use admit::{Refused, admit_recurrent};
pub use color::{Coloring, ColoringError, Use, color_live_ranges};
pub use mask::{Disagreement, causal_prefix_lengths, kv_len_disagreement};
pub use member::{BuildError, ForwardDesc, ResolvedGeometry, build_member_desc};
pub use psos::{DecodePsoPlan, EntryNames, Features as PsoFeatures, PsoRequest, plan_decode_psos};
pub use schedule::{
    BatchSchedule, DEFAULT_PAGE_SIZE, Malformed, Rejected, RequestSpan, build_schedule,
    find_request, validate_capacity, validate_paged,
};
pub use timing::{
    Ablation, BoundaryMismatch, DispatchAttribution, DispatchInfo, StepAttribution, attribute_step,
};
pub use worker::Worker;
