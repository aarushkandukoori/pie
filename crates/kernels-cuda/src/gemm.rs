//! Dense matmul: the x·Wᵀ family, batched, grouped, and the cuBLAS routes.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::KernelSig;

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    // The plain x·Wᵀ, which every family fires and which the table had
    // never carried -- invisible to the audit until its launcher regex
    // stopped requiring the return type to start the line (`inline void`).
    kernel!(gemm_xwt "gemm::act_x_wt_bf16"),
    // ── the WEIGHT REPRESENTATION axis ─────────────────────────────
    //
    // One row per way a weight can be stored, because the statement
    // NAMES which — `MatW::gemm_symbol`. The driver used to pick between
    // these by building a `WeightView` from a per-layer descriptor the
    // statement never mentioned, and `gemm::act_x_w` routed on it; a
    // kernel chosen by the driver is the shape every defect in this
    // arc's ledger had.
    //
    // Each takes the scales (and zero-points, where the checkpoint
    // carries them) as WEIGHTS — `MatW::scale_names` derives their names
    // off the weight's own, which is how the loader already finds them.
    // A dense statement names one tensor; a quantized one names two or
    // three, and says so.
    kernel!(gemm_xwt_tensor_scaled "gemm::act_x_wt_tensor_scaled"),
    kernel!(gemm_xwt_channel_scaled "gemm::act_x_wt_channel_scaled"),
    kernel!(gemm_xwt_grouped_scaled "gemm::act_x_wt_grouped_scaled"),
    // MXFP4 with E8M0 block scales — gpt-oss's expert banks. Its scales
    // are not a layout question, so it is its own row rather than a
    // `Scaled` variant.
    kernel!(gemm_xwt_mxfp4_marlin "gemm::act_x_wt_mxfp4_marlin"),
    // Its batched twin: one GEMM per pointer-array entry. `whole` for the
    // same reason `gemm_grouped` is -- the batch is addressed through
    // device pointer arrays built for the WHOLE fire, so a row window
    // would leave them pointing at rows the window does not own.
    kernel!(gemm_batched_xwt "gemm::batched_act_x_wt_bf16", whole = true),
    kernel!(gemm_cublas "gemm::act_x_wt_bf16_cublas"),
    kernel!(gemm_out_fp32 "gemm::act_x_wt_bf16_out_fp32"),
    // The group boundaries (`M_array`) are fire-global, so a row window would
    // cut a group in half.
    kernel!(gemm_grouped "gemm::grouped_act_x_wt_bf16", whole = true),
    kernel!(gemv3 "gemm::gemv3_bf16"),
    // The sink rescale, and the fp32 LSE it eats. The LSE has no row of
    // its own: it is a second OUTPUT of the decode dispatch, requested
    // by an argument, so the kernel that changes is none.
    // A projection with its bias in the EPILOGUE — one launch where a
    // matmul plus an AddBias is two, and a different accumulation order.
    kernel!(gemm_bias "gemm::act_x_wt_bias_bf16"),
];
