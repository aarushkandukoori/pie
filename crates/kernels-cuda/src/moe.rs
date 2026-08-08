//! Mixture of experts: routing, the aligned permutation path, the routed
//! GEMMs and the weighted finalize.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::KernelSig;

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    kernel!(moe_grouped_gemm "launch_moe_grouped_gemm_bf16"),
    // `topk_idx` here is `[N, K]` in TOKEN order, not the route-major order
    // the aligned path sorts into, so a row window keeps each token's routing
    // intact and these are not `whole`.
    kernel!(wna16_gate_up_decode "launch_wna16_gate_up_decode_bf16"),
    kernel!(wna16_down_decode "launch_wna16_down_decode_bf16"),
    kernel!(apply_per_expert_scale "launch_apply_per_expert_scale_bf16"),
    // `topk_idx` is route-global, so a row window would pick the wrong
    // experts' biases.
    kernel!(add_moe_route_bias "launch_add_moe_route_bias_bf16", whole = true),
    kernel!(transpose_expert_scales "launch_transpose_expert_scales_u8"),
    kernel!(mxfp4_moe_gate_up_decode_grouped "launch_mxfp4_moe_gate_up_decode_grouped_bf16",
        whole = true),
    // Namespaced in the symbol because it lives in the vendored `marlin_moe`
    // tree, the same way the `ops::` entries do.
    kernel!(mxfp4_moe_gemm_w4a16 "marlin_moe::launch_mxfp4_moe_gemm_w4a16_bf16", whole = true),
    kernel!(topk_sqrtsoftplus "launch_topk_sqrtsoftplus_bf16"),
    // Expert INDICES from a table keyed by token id -- a route that is a pure
    // function of the token rather than of its activations. The WEIGHTS still
    // come from the router logits, so the logits GEMM above it does not go
    // away.
    kernel!(hash_route_lookup "launch_hash_route_lookup"),
    kernel!(topk_sigmoid_bias "launch_topk_sigmoid_bias_fp32"),
    // The UNPADDED counterpart of `moe_align`: exact per-expert counts the
    // host reads to build cuBLAS grouped shapes. `whole` for the same reason
    // -- the sort is over all routes.
    kernel!(moe_bucket_exact "launch_moe_bucket_exact", whole = true),
    kernel!(token_batched_weighted_sum_aligned "launch_token_batched_weighted_sum_aligned_bf16",
        whole = true),
    // glm5 and kimi_k3 route through a permutation rather than a loop: every
    // (token, expert) pair is a route, routes are bucketed by expert and
    // padded to fixed blocks so one batched GEMM covers all experts, and the
    // permutation is undone afterwards.
    //
    // Five of six are `whole`, for the same reason each time: the
    // permutation is computed over ALL routes in the fire, so a statement
    // addressed through `sorted_route_ids` cannot take a row window -- the
    // window would name different routes than the sort did.
    kernel!(moe_align "launch_moe_align_decode", whole = true),
    kernel!(gather_moe_aligned_inputs "launch_gather_moe_aligned_inputs_bf16", whole = true),
    kernel!(build_moe_ptrs_aligned "launch_build_moe_ptrs_aligned_bf16", whole = true),
    kernel!(reorder_moe_aligned_output "launch_reorder_moe_aligned_output_bf16", whole = true),
    // `out[dst_idx[i]] += src[i]·w[i]`, and `dst_idx` is route-global: a
    // window over output ROWS is not a window over routes.
    kernel!(scatter_add_weighted "launch_scatter_add_weighted_bf16", whole = true),
    // The exception, and it is the router: a token's top-k reads only its own
    // logits row, so this one splits like any elementwise statement.
    kernel!(topk_sigmoid "launch_topk_sigmoid_bf16"),
    // The router's top-k, then the decode GEMV leg's two routed
    // projections and its combine. The expert axis rides INSIDE the
    // value on this leg, so the whole branch stays a list of rectangles;
    // the grouped-GEMM and host-routed legs reach the same numbers by
    // shapes no `Dim` spells, and are named refusals, not entries.
    kernel!(topk_softmax "launch_topk_softmax_bf16"),
    // The whole routed block as one call — permute, both grouped GEMMs,
    // the activation and the weighted finalize. The leg decode actually
    // takes, and the only one that is a single rectangle.
    // Namespaced because it is not a `kernels::launch_*` at all: it is an
    // `ops::` entry point that installs tactics and runs a CUTLASS
    // pipeline. The symbol says so.
    kernel!(moe_fused_cutlass "ops::flashinfer_cutlass_moe_bf16"),
    kernel!(moe_gate_up_gemv "launch_moe_gate_up_decode_gemv_bf16"),
    kernel!(moe_down_gemv "launch_moe_down_decode_gemv_bf16"),
    // The combine folds the residual when the MoE output lands straight
    // on the stream (tp=1) — one launch where the semantic text has a
    // WeightedSum and a ResidualAdd.
    kernel!(moe_weighted_sum "launch_token_batched_weighted_sum_bf16"),
    kernel!(moe_weighted_sum_add "launch_token_batched_weighted_sum_add_bf16"),
    // The routed MXFP4 GEMVs. Like qwen3_5's GEMV leg the expert axis
    // rides INSIDE the value, so each is one rectangle over `N * k`
    // routes; unlike it, the weight slot names a per-expert POINTER
    // BANK, which is a binding question and not a shape one.
    kernel!(mxfp4_moe_gate_up "launch_mxfp4_moe_gate_up_decode_bf16"),
    kernel!(mxfp4_moe_down "launch_mxfp4_moe_down_decode_bf16"),
];
