//! Dense matmul: the x·Wᵀ family, batched, grouped, and the cuBLAS routes.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::{KernelSig, Ret, operands};

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    // The plain x·Wᵀ, which every family fires and which the table had
    // never carried -- invisible to the audit until its launcher regex
    // stopped requiring the return type to start the line (`inline void`).
    // Like the MLA absorb pair, the handle is what these take instead of a
    // stream -- the stream is set on the handle.
    kernel!(gemm_xwt "gemm::act_x_wt_bf16",
        operands = operands![
            handle: CublasHandle, act: Buf, w: Buf, y: BufMut,
            m: I32, n: I32, k: I32, beta: F32,
        ]),
    // Its batched twin: one GEMM per pointer-array entry. `whole` for the
    // same reason `gemm_grouped` is -- the batch is addressed through
    // device pointer arrays built for the WHOLE fire, so a row window
    // would leave them pointing at rows the window does not own.
    kernel!(gemm_batched_xwt "gemm::batched_act_x_wt_bf16", whole = true,
        operands = operands![
            handle: CublasHandle, act_ptrs_dev: Bufs, w_ptrs_dev: Bufs,
            y_ptrs_dev: BufMuts,
            m: I32, n: I32, k: I32, batch_count: I32, beta: F32,
        ]),
    kernel!(gemm_cublas "gemm::act_x_wt_bf16_cublas",
        operands = operands![
            handle: CublasHandle, act: Buf, w: Buf, y: BufMut,
            m: I32, n: I32, k: I32, beta: F32,
        ]),
    kernel!(gemm_out_fp32 "gemm::act_x_wt_bf16_out_fp32",
        operands = operands![
            handle: CublasHandle, act: Buf, w: Buf, y: F32sMut,
            m: I32, n: I32, k: I32,
        ]),
    // The group boundaries (`M_array`) are fire-global, so a row window would
    // cut a group in half.
    kernel!(gemm_grouped "gemm::grouped_act_x_wt_bf16", whole = true,
        operands = operands![
            handle: CublasHandle, act_ptrs_host: Bufs, w_ptrs_host: Bufs,
            y_ptrs_host: BufMuts, m_array_host: I32s,
            group_count: I32, n: I32, k: I32, beta: F32,
        ]),
    // q/k/v in one launch: three weights, three row counts, one activation.
    // A TRIED launch -- returns false and touches nothing if any argument is
    // unsuitable, and the caller falls back to three GEMVs.
    kernel!(gemv3 "gemm::gemv3_bf16", ret = Ret::Bool,
        operands = operands![
            w0: Buf, w1: Buf, w2: Buf,
            b0: Buf | null, b1: Buf | null, b2: Buf | null,
            o0: BufMut, o1: BufMut, o2: BufMut, act: Buf,
            n0: I32, n1: I32, n2: I32, k: I32, stream: Stream,
        ]),
    // The sink rescale, and the fp32 LSE it eats. The LSE has no row of
    // its own: it is a second OUTPUT of the decode dispatch, requested
    // by an argument, so the kernel that changes is none.
    // A projection with its bias in the EPILOGUE — one launch where a
    // matmul plus an AddBias is two, and a different accumulation order.
    kernel!(gemm_bias "gemm::act_x_wt_bias_bf16",
        operands = operands![
            handle: CublasHandle, act: Buf, w: Buf, bias: Buf | null,
            y: BufMut, m: I32, n: I32, k: I32, stream: Stream, beta: F32,
        ]),
];
