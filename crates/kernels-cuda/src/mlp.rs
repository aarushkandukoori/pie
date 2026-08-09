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
    kernel!(chunked_swiglu "mlp::chunked_swiglu_bf16"),
    kernel!(swiglu "mlp::swiglu_bf16"),
    kernel!(chunked_swiglu_strided "mlp::chunked_swiglu_strided_bf16"),
    kernel!(gpt_oss_glu_strided "mlp::gpt_oss_glu_strided_bf16"),
    kernel!(swiglu_clamp "mlp::swiglu_clamp_bf16"),
    kernel!(chunked_swiglu_clamp "mlp::chunked_swiglu_clamp_bf16"),
    kernel!(relu2 "mlp::relu2_bf16"),
    // SiTU is not a swiglu variant: the tanh saturates far enough out that a
    // bf16 intermediate loses the distinction the gate exists to make.
    kernel!(situ "mlp::situ_bf16"),
    kernel!(chunked_situ "mlp::chunked_situ_bf16"),
    kernel!(gaussian_topk "mlp::gaussian_topk_bf16"),
    // GeGLU-tanh is not a swiglu variant: `gelu_pytorch_tanh` on the
    // gate is a different function. The packed/pair split is the same
    // binding question.
    // The PAIR form: `(gate, up, out)` with `out` over `gate`. gemma-4's
    // PLE gate is the same call with the relay slice as `up`.
    kernel!(geglu_tanh "mlp::geglu_tanh_bf16", in_place = &[(0, 0)]),
    kernel!(chunked_geglu_tanh "mlp::chunked_geglu_tanh_bf16"),
    // SwiGLU with a clamp. `swiglu_limit` is a config constant, so this
    // is a different kernel and not a different argument.
    kernel!(gpt_oss_glu "mlp::gpt_oss_glu_bf16"),
    kernel!(sigmoid_scalar_gate_add "mlp::sigmoid_scalar_gate_add_bf16"),
    kernel!(sigmoid_scalar_gate_strided_add "mlp::sigmoid_scalar_gate_strided_add_bf16"),
    kernel!(moe_shared_gate_dot "mlp::sigmoid_dot_scalar_gate_add_bf16"),
];
