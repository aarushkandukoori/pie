//! Feed-forward activations: the SwiGLU/GeGLU/SiTU spellings and their clamps.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::KernelSig;

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    // Two spellings of one arithmetic, and the BINDING picks: a packed
    // gate‖up bank feeds the chunked form, two narrow buffers the pair
    // form. A load-time fact, so the declaration states it.
    kernel!(chunked_swiglu "launch_chunked_swiglu_bf16"),
    kernel!(swiglu "launch_swiglu_bf16"),
    kernel!(chunked_swiglu_strided "launch_chunked_swiglu_strided_bf16"),
    kernel!(gpt_oss_glu_strided "launch_gpt_oss_glu_strided_bf16"),
    kernel!(swiglu_clamp "launch_swiglu_clamp_bf16"),
    kernel!(chunked_swiglu_clamp "launch_chunked_swiglu_clamp_bf16"),
    kernel!(relu2 "launch_relu2_bf16"),
    // SiTU is not a swiglu variant: the tanh saturates far enough out that a
    // bf16 intermediate loses the distinction the gate exists to make.
    kernel!(situ "launch_situ_bf16"),
    kernel!(chunked_situ "launch_chunked_situ_bf16"),
    kernel!(tanh "launch_tanh_bf16"),
    kernel!(gaussian_topk "launch_gaussian_topk_bf16"),
    // GeGLU-tanh is not a swiglu variant: `gelu_pytorch_tanh` on the
    // gate is a different function. The packed/pair split is the same
    // binding question.
    kernel!(geglu_tanh "launch_geglu_tanh_bf16"),
    kernel!(chunked_geglu_tanh "launch_chunked_geglu_tanh_bf16"),
    // SwiGLU with a clamp. `swiglu_limit` is a config constant, so this
    // is a different kernel and not a different argument.
    kernel!(gpt_oss_glu "launch_gpt_oss_glu_bf16"),
];
