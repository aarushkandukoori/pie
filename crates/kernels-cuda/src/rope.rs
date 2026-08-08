//! Rotary position encoding, and the norm+rope fusions that feed attention.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::KernelSig;

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    kernel!(rope_standard_table "rope::rope_standard_table"),
    kernel!(qk_rmsnorm_rope "rope::qk_rmsnorm_rope_bf16"),
    // A hooked pure-decode fire is graph-CAPTURED and its hook split rides a
    // DEVICE word (`win_d`), not a host row range. All four are `whole`, and
    // for a reason no other `whole` row here gives: the window is not a
    // number the lowering knows, so it cannot be a rectangle at all.
    kernel!(qk_rmsnorm_rope_devwin "rope::qk_rmsnorm_rope_bf16_devwin", whole = true),
    // YaRN and original-YaRN interpolate frequencies differently; which a
    // checkpoint wants is a load-time fact, so they are two rows.
    kernel!(rope_yarn "rope::rope_yarn_bf16"),
    // MROPE takes `[num_tokens, 3]` positions -- a (t, h, w) triple, because
    // a vision model's tokens sit in a grid. Not the plain qk_rmsnorm_rope
    // with a different theta.
    kernel!(qk_rmsnorm_mrope "rope::qk_rmsnorm_mrope_bf16"),
    // Ropes the LAST `rope_dim` channels rather than the first. A different
    // statement from `rope_partial_q_only`, not a flag on it: which end of
    // the channel axis carries position is a property of the checkpoint.
    kernel!(rope_partial_last "rope::rope_partial_last_bf16"),
    // Q-only rotation: a KV-shared layer's K was rotated at its source
    // layer. One operand is the statement.
    kernel!(rope_partial_q_only "rope::rope_partial_bf16"),
    // gemma-4 rounds where qwen3_5 does not, and bf16 rounding is which
    // numbers come out — so the symbol IS the statement.
    kernel!(qk_rmsnorm_rope_rounded "rope::qk_rmsnorm_rope_bf16_rounded"),
    // YaRN, as its paper spells it. A deployment's scaling is a load-time
    // config answer, so it picks a kernel here rather than an argument.
    kernel!(rope_yarn_original "rope::rope_yarn_original_bf16"),
    kernel!(rope_write_kv "rope::rope_write_kv_bf16", whole = true, sink = Some("kv.pages")),
];
