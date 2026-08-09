//! Normalization and the residual stream — RMSNorm, the fused landings,
//! gemma-3n's AltUp and deepseek-v4's hyper-connections.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::KernelSig;

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    kernel!(rmsnorm_gated_launch "norm::rmsnorm_gated_bf16"),
    kernel!(rmsnorm_strided "norm::rmsnorm_strided_bf16"),
    // gemma-4's end-of-layer shape: the scale sits BETWEEN the add and the
    // norm, which is why it is not `residual_add_rmsnorm` with a multiply
    // somewhere.
    kernel!(residual_add_scale_rmsnorm "norm::residual_add_scale_rmsnorm_bf16"),
    // gpt-oss ships its experts as MXFP4 -- 4-bit values with an E8M0
    // exponent byte per block of 32 -- and mixtral's shell runs them through
    // Marlin. Several of these operate on WEIGHTS rather than activations
    // (repacking a scale layout, splitting a fused bias) and have no token
    // extent at all; they are declared because they are launches the fire
    // performs.
    kernel!(add_bias_strided "norm::add_bias_bf16_strided"),
    // The fp16 copy is what the MXFP4 grouped GEMM consumes; producing it
    // here rather than casting afterwards is the binding.
    kernel!(rmsnorm_with_fp16 "norm::rmsnorm_bf16_with_fp16"),
    // The SECOND rank-K residual scheme here, and not AltUp's. gemma-3n
    // predicts each stream from a learned combination and corrects from one
    // ACTIVE stream; HC mixes with a per-token, sinkhorn-normalized matrix
    // and has no active stream -- every layer reads a weighted collapse of
    // all of them and writes back to all of them. Row-shaped throughout.
    kernel!(hc_rmsnorm_to_f32 "norm::hc_rmsnorm_to_f32"),
    // Where a rank-K residual BEGINS: replicate the embedding into K
    // streams. AltUp's equivalent is implicit in gemma-3n's workspace
    // layout; HC states it, which is the one a declaration can read.
    kernel!(hc_expand "norm::hc_expand_bf16"),
    kernel!(hc_pre "norm::hc_pre_postprocess_bf16"),
    kernel!(hc_post "norm::hc_post_bf16"),
    kernel!(hc_head "norm::hc_head_postprocess_bf16"),
    kernel!(per_head_rmsnorm "norm::per_head_rmsnorm_bf16"),
    // Residual add + the next block's pre-norm, fused. Numerically the
    // two-kernel sequence (the kernel matches `residual_add`'s bf16 rounding
    // before norming), which is what makes it a binding a declaration may
    // state rather than a different computation.
    kernel!(residual_add_rmsnorm "norm::residual_add_rmsnorm_bf16"),
    // A rank-K residual stream: K parallel streams predicted from each
    // other, one of them run through the real layer, the rest corrected
    // from the difference. See `dsl::cuda`'s AltUp block for the algebra.
    //
    // Not one of these carries a contract clause, and that is a claim
    // rather than an omission: every one is row-shaped -- token `t`'s
    // output reads only token `t`'s inputs -- so a peel may split it, it
    // obligates no host plan, and there is no seam capability for it to
    // refuse.
    kernel!(altup_predict "norm::altup_predict_bf16"),
    kernel!(altup_correct "norm::altup_correct_bf16"),
    kernel!(altup_unpack_predict_coefs "norm::altup_unpack_predict_coefs"),
    kernel!(altup_unpack_correct_coefs "norm::altup_unpack_correct_coefs"),
    kernel!(mean_streams "norm::mean_streams_bf16"),
    kernel!(compute_rms "norm::compute_rms_bf16"),
    kernel!(magnitude_rescale "norm::magnitude_rescale_bf16"),
    // Weightless per-head norm (the V-norm) — no gamma, so no variant.
    kernel!(rmsnorm_no_scale "norm::rmsnorm_no_scale_bf16", in_place = &[(0, 0)]),
    // Four statements in one launch, and two: gemma-4 fuses the next
    // block's input norm into the previous block's landing, which is why
    // its layer body appears to be missing one.
    // `(landed, mlp_in)` over `(x, y)`: the stream operand is the one it
    // lands on, and the landed stream is output 0.
    kernel!(norm_residual_scale_norm "norm::rmsnorm_residual_add_scale_rmsnorm_bf16",
        in_place = &[(0, 1)]),
    kernel!(norm_residual_add "norm::rmsnorm_residual_add_bf16", in_place = &[(0, 1)]),
    kernel!(scalar_mul "norm::scalar_mul_bf16", in_place = &[(0, 0)]),
    // Accumulates into its FIRST argument. Stating it is what lets a
    // text add into a window (`select`) and have the window keep the
    // result — see `KernelSig::in_place`.
    kernel!(residual_add_cuda "norm::residual_add_bf16", in_place = &[(0, 0)]),
    kernel!(tanh "norm::tanh_bf16"),
    kernel!(attn_sink_correction "norm::attn_sink_correction_bf16"),
];
