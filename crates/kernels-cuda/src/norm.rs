//! Normalization and the residual stream — RMSNorm, the fused landings,
//! gemma-3n's AltUp and deepseek-v4's hyper-connections.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::KernelSig;

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    kernel!(sigmoid_scalar_gate_add "launch_sigmoid_scalar_gate_add_bf16"),
    kernel!(rmsnorm_gated_launch "launch_rmsnorm_gated_bf16"),
    kernel!(sigmoid_scalar_gate_strided_add "launch_sigmoid_scalar_gate_strided_add_bf16"),
    kernel!(rmsnorm_strided "launch_rmsnorm_strided_bf16"),
    kernel!(scale_rows "launch_scale_rows_bf16"),
    // gemma-4's end-of-layer shape: the scale sits BETWEEN the add and the
    // norm, which is why it is not `residual_add_rmsnorm` with a multiply
    // somewhere.
    kernel!(residual_add_scale_rmsnorm "launch_residual_add_scale_rmsnorm_bf16"),
    // gpt-oss ships its experts as MXFP4 -- 4-bit values with an E8M0
    // exponent byte per block of 32 -- and mixtral's shell runs them through
    // Marlin. Several of these operate on WEIGHTS rather than activations
    // (repacking a scale layout, splitting a fused bias) and have no token
    // extent at all; they are declared because they are launches the fire
    // performs.
    kernel!(add_bias_strided "launch_add_bias_bf16_strided"),
    // The fp16 copy is what the MXFP4 grouped GEMM consumes; producing it
    // here rather than casting afterwards is the binding.
    kernel!(rmsnorm_with_fp16 "launch_rmsnorm_bf16_with_fp16"),
    // The SECOND rank-K residual scheme here, and not AltUp's. gemma-3n
    // predicts each stream from a learned combination and corrects from one
    // ACTIVE stream; HC mixes with a per-token, sinkhorn-normalized matrix
    // and has no active stream -- every layer reads a weighted collapse of
    // all of them and writes back to all of them. Row-shaped throughout.
    kernel!(hc_rmsnorm_to_f32 "launch_hc_rmsnorm_to_f32"),
    // Where a rank-K residual BEGINS: replicate the embedding into K
    // streams. AltUp's equivalent is implicit in gemma-3n's workspace
    // layout; HC states it, which is the one a declaration can read.
    kernel!(hc_expand "launch_hc_expand_bf16"),
    kernel!(hc_pre "launch_hc_pre_postprocess_bf16"),
    kernel!(hc_post "launch_hc_post_bf16"),
    kernel!(hc_head "launch_hc_head_postprocess_bf16"),
    kernel!(per_head_rmsnorm "launch_per_head_rmsnorm_bf16"),
    kernel!(zamba_rmsnorm_gated "launch_zamba_rmsnorm_gated_bf16"),
    // KDA's arithmetic is fp32 throughout, so operands living in bf16 in the
    // workspace cross explicitly. Launches, so the trace records them.
    kernel!(l2norm_scale_to_f32 "launch_l2norm_scale_bf16_to_fp32"),
    // Residual add + the next block's pre-norm, fused. Numerically the
    // two-kernel sequence (the kernel matches `residual_add`'s bf16 rounding
    // before norming), which is what makes it a binding a declaration may
    // state rather than a different computation.
    kernel!(residual_add_rmsnorm "launch_residual_add_rmsnorm_bf16"),
    // A rank-K residual stream: K parallel streams predicted from each
    // other, one of them run through the real layer, the rest corrected
    // from the difference. See `dsl::cuda`'s AltUp block for the algebra.
    //
    // Not one of these carries a contract clause, and that is a claim
    // rather than an omission: every one is row-shaped -- token `t`'s
    // output reads only token `t`'s inputs -- so a peel may split it, it
    // obligates no host plan, and there is no seam capability for it to
    // refuse.
    kernel!(altup_predict "launch_altup_predict_bf16"),
    kernel!(altup_correct "launch_altup_correct_bf16"),
    kernel!(altup_unpack_predict_coefs "launch_altup_unpack_predict_coefs"),
    kernel!(altup_unpack_correct_coefs "launch_altup_unpack_correct_coefs"),
    kernel!(mean_streams "launch_mean_streams_bf16"),
    kernel!(compute_rms "launch_compute_rms_bf16"),
    kernel!(magnitude_rescale "launch_magnitude_rescale_bf16"),
    // Weightless per-head norm (the V-norm) — no gamma, so no variant.
    kernel!(rmsnorm_no_scale "launch_rmsnorm_no_scale_bf16"),
    // Four statements in one launch, and two: gemma-4 fuses the next
    // block's input norm into the previous block's landing, which is why
    // its layer body appears to be missing one.
    kernel!(norm_residual_scale_norm "launch_rmsnorm_residual_add_scale_rmsnorm_bf16"),
    kernel!(norm_residual_add "launch_rmsnorm_residual_add_bf16"),
    kernel!(scalar_mul "launch_scalar_mul_bf16"),
    kernel!(residual_add_cuda "launch_residual_add_bf16"),
];
