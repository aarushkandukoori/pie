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
mod binds;
mod color;
mod consts;
mod dataflow;
mod dispatch;
mod dispatch_gptoss;
mod dispatch_llama;
mod dispatch_mb;
mod geometry;
mod geometry_facts;
mod golden;
mod gptoss;
mod gptoss_consts;
mod gptoss_solve;
mod llama;
mod mask;
mod member;
mod paging;
mod psos;
mod psos_gptoss;
mod psos_llama;
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
pub use consts::{
    ExpertCombineParams, GatedRmsParams, GdnCoreParams, KN, MoeRouteParams, RmsParams,
    RouterParams, gdn_core_params, is_qmv, is_routed, qmv_kn,
};
pub use dataflow::{build_scratch_schedule, build_scratch_uses};
pub use dispatch::{
    DagOptions, Dispatch, Launch, barrier_after, build_decode_dag, concurrent_run_ends,
};
pub use dispatch_gptoss::{
    GptOssDagStats, build_gptoss_dag, build_gptoss_dag_mb, gptoss_dag_stats, gptoss_is_dense_proj,
    gptoss_mb_kind, gptoss_moe_qmm_bn, gptoss_moe_sorted_rows, gptoss_qmm_bn, gptoss_qmm_min_batch,
    gptoss_qmm_pool_rows, gptoss_qmm_rows, gptoss_scratch_elems_mb,
};
pub use dispatch_llama::{
    LlamaDagStats, build_llama_dag, build_llama_dag_mb, llama_dag_stats, llama_dense_qmm_bm,
    llama_fp16_format, llama_is_dense_proj, llama_moe_qmm_bn, llama_moe_sorted_rows, llama_qmm_bn,
    llama_qmm_min_batch, llama_qmm_pool_rows, llama_qmm_rows,
};
pub use dispatch_mb::{
    PREFILL_ORDINAL_BASE, PREFILL_ORDINAL_STRIDE, ROUTED_DECODE_BATCHED, SDPA_QUERY_TILE,
    build_decode_dag_mb, build_decode_prefill_dags, elementwise_mb, fp16_format, mb_geometry,
    mb_kind, qmm_bm, qmm_bm_slot, qmm_bn, qmm_bn_unsplit, qmm_mb_rows, qmm_t, qmv_mb, qmv_out_size,
    rms_mb, uses_alt_quant,
};
pub use geometry::{AffineFormat, DecodeGeometry};
pub use geometry_facts::{
    GeometryRefused, ROUTER_MAX_EXPERTS, ROUTER_MAX_TOP_K, geometry_from_facts,
};
pub use golden::{
    Tap, TapSite, dir_from_env, dump_bf16, dump_bf16_sorted, dump_taps, dump_tokens,
    sorted_dump_rows, tap_for, taps_recycle, write_npy,
};
pub use gptoss::{
    GptOssGeometry, gptoss_decode_geometry, gptoss_geometry_from_facts, gptoss_scratch_elems,
};
pub use gptoss_consts::{RowGatherParams, SwiGluParams, gptoss_qmv_kn, yarn_inv_freq, yarn_mscale};
pub use gptoss_solve::{StagedQuant, bits_from_extents, solve_quant_into, solve_staged_quant};
pub use llama::{LlamaGeometry, llama_decode_geometry, llama_geometry_from_facts, llama_qmv_kn};
pub use mask::{Disagreement, causal_prefix_lengths, kv_len_disagreement};
pub use member::{BuildError, ForwardDesc, ResolvedGeometry, build_member_desc};
pub use paging::{
    Cut, PagingPlan, PagingRefused, RenumberRefused, SlabShape, plan_paging, renumber_routing,
};
pub use psos::{DecodePsoPlan, EntryNames, Features as PsoFeatures, PsoRequest, plan_decode_psos};
pub use psos_gptoss::{
    GptOssPsoRequest, GptOssSlot, SDPA_MMA_HEAD_DIM, gptoss_kinds, gptoss_mb_kinds, gptoss_mb_plan,
    gptoss_step_plan, plan_gptoss_psos,
};
pub use psos_llama::{llama_entry_names, llama_step_plan, llama3_inv_freq};
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
