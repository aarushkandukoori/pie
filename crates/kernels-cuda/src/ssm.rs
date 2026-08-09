//! Linear attention and state-space recurrences: GDN, KDA, mamba, and the
//! causal conv that feeds them.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::KernelSig;

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    // The other mamba scan: nemotron_h takes FlashInfer's SSU on sm90+ and
    // its own batched kernel elsewhere.
    kernel!(flashinfer_mamba_ssu "ssm::flashinfer_mamba_ssu_bf16", whole = true),
    // Unbatched twins of the `_batched` forms below -- a legacy parity
    // entrypoint and a single-request fast path. Not `whole`, for the reason
    // the batched ones are not: their `B` is the batch, not a window into it.
    // The `_state_bf16` pairing is a precision BINDING a deployment states,
    // the same way the batched rows spell it.
    kernel!(gdn_step_single "ssm::recurrent_gated_delta_step"),
    kernel!(gdn_step_single_state_bf16 "ssm::recurrent_gated_delta_step_state_bf16"),
    kernel!(gdn_prefill_single "ssm::chunk_gated_delta_prefill"),
    kernel!(gdn_prefill_single_state_bf16 "ssm::chunk_gated_delta_prefill_state_bf16"),
    kernel!(causal_conv1d_prefill_single "ssm::causal_conv1d_prefill_bf16"),
    // The third linear-attention shape here, and not a variant of the other
    // two: mamba carries a `[head_dim, state_size]` slab per head and
    // advances it with a scalar `dA` from a per-token `dt` -- a selective
    // scan, not a delta rule. A different state SHAPE, which is why none of
    // the GDN or KDA rows stand in for it.
    kernel!(nemotron_mamba_split "ssm::nemotron_mamba_split_bf16"),
    kernel!(nemotron_prepare_mamba_params "ssm::nemotron_prepare_mamba_params"),
    kernel!(nemotron_prepare_mamba_dt_da "ssm::nemotron_prepare_mamba_dt_da"),
    // `whole` for both reasons this table collects: it addresses through
    // `slot_ids` and `qo_indptr`, and the scan carries state token to token,
    // so a row window would resume from the wrong slab.
    kernel!(nemotron_mamba_ssm "ssm::nemotron_mamba_ssm_batched_bf16", whole = true),
    // Advances a slot's conv window in place; a row window advances the
    // wrong slots.
    kernel!(causal_conv1d_update "ssm::causal_conv1d_update_bf16", whole = true),
    // kimi_k3's linear-attention half. The gated delta rule qwen3_5 runs,
    // with the decay per KEY CHANNEL rather than per head -- which is why
    // these exist beside the GDN kernels instead of reusing them with a
    // broadcast.
    kernel!(kda_gate_beta "ssm::kda_gate_beta_bf16"),
    // `slot_ids` is indexed `0..R` against the fire's request order, so a row
    // window would advance the wrong slots.
    kernel!(kda_recurrent_step "ssm::kda_recurrent_step_batched", whole = true),
    // `whole` twice over: it walks windows out of `qo_indptr`, and the
    // recurrence has a strict per-token state dependency -- a row window
    // would start the scan from the wrong state, which is a different answer
    // rather than a misaddressed one.
    kernel!(kda_prefill "ssm::kda_prefill_batched", whole = true),
    kernel!(kda_o_norm_gated "ssm::kda_o_norm_gated_bf16"),
    kernel!(gdn_conv_update "ssm::causal_conv1d_update_batched_bf16"),
    kernel!(gdn_conv_prefill "ssm::causal_conv1d_prefill_batched_bf16"),
    kernel!(gdn_step "ssm::recurrent_gated_delta_step_batched"),
    kernel!(gdn_step_gqa "ssm::recurrent_gated_delta_step_batched_gqa"),
    kernel!(gdn_step_state_bf16 "ssm::recurrent_gated_delta_step_batched_state_bf16"),
    kernel!(gdn_step_gqa_state_bf16 "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16"),
    kernel!(gdn_prefill_fla "ssm::chunk_gated_delta_prefill_batched"),
    kernel!(gdn_prefill_fla_state_bf16 "ssm::chunk_gated_delta_prefill_batched_state_bf16"),
    kernel!(gdn_prefill_cached "ssm::chunk_gated_delta_prefill_batched_cached"),
    kernel!(gdn_prefill_cached_state_bf16
        "ssm::chunk_gated_delta_prefill_batched_cached_state_bf16"),
    kernel!(gdn_prefill_warp_tiled_gqa "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa"),
    kernel!(gdn_prefill_warp_tiled_gqa_state_bf16
        "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16"),
    kernel!(repeat_interleave_heads "ssm::repeat_interleave_heads_fp32"),
    // KDA's arithmetic is fp32 throughout, so operands living in bf16 in the
    // workspace cross explicitly. Launches, so the trace records them.
    kernel!(l2norm_scale_to_f32 "ssm::l2norm_scale_bf16_to_fp32"),
    kernel!(bf16_to_f32 "ssm::bf16_to_fp32"),
    kernel!(f32_to_bf16 "ssm::fp32_to_bf16"),
    kernel!(zamba_rmsnorm_gated "ssm::zamba_rmsnorm_gated_bf16"),
    kernel!(build_nemotron_moe_ptrs_aligned "ssm::build_nemotron_moe_ptrs_aligned_bf16",
        whole = true),
    kernel!(build_nemotron_moe_ptrs_decode "ssm::build_nemotron_moe_ptrs_decode_batched_bf16",
        whole = true),
];
