//! Pure addressing: gather, scatter, split, concat, transpose, embed.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::KernelSig;

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    kernel!(split_rows "layout::split_bf16_rows"),
    kernel!(split_qwen_gdn_ba "layout::split_qwen_gdn_ba_bf16"),
    // A copy that skips requests whose slot id is invalid: the launch happens
    // for every request every time and the slot decides whether it does
    // anything, so the dispatch is fixed and a CUDA graph replays.
    kernel!(copy_if_valid_slot "layout::copy_if_valid_slot", whole = true),
    kernel!(concat_rows "layout::concat_bf16_rows"),
    // Splits a packed gate/up bank by HALVES, where `deinterleave_rows`
    // splits by parity. Same shape, different layout, checkpoint decides.
    kernel!(split_gate_up "layout::split_gate_up_bf16"),
    // gpt-oss interleaves gate and up ROW BY ROW, so splitting them is a
    // parity deinterleave and not a slice. Weight-shaped, no token extent.
    kernel!(deinterleave_rows "layout::deinterleave_rows_bf16"),
    kernel!(deinterleave_vec "layout::deinterleave_vec_bf16"),
    // A vocab-sharded embedding: the rank holds `[local_vocab, hidden]` from
    // `vocab_offset` and writes zeros elsewhere, and the all-reduce after it
    // makes the row whole. The shard is a property of the WEIGHT, not of the
    // row range, so this splits like any gather.
    kernel!(embed_vocab_shard "layout::embed_bf16_vocab_shard"),
    // The PLE relay: [N, L, D] -> [L, N, D], so a layer reads a
    // contiguous slice. Addressing, not arithmetic.
    kernel!(transpose_nld_to_lnd "layout::transpose_bf16_nld_to_lnd"),
    kernel!(verify_stash_store "qwen35_verify_stash_store"),
    kernel!(verify_stash_load "qwen35_verify_stash_load"),
];
