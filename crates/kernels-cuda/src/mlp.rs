//! Feed-forward activations: the SwiGLU/GeGLU/SiTU spellings and their clamps.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::operands;
use kernels::KernelSig;

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    // Two spellings of one arithmetic, and the BINDING picks: a packed
    // gate‖up bank feeds the chunked form, two narrow buffers the pair
    // form. A load-time fact, so the declaration states it.
    kernel!(chunked_swiglu "mlp::chunked_swiglu_bf16",
        operands = operands![
            packed: Buf,
            y: BufMut,
            n: I32,
            i: I32,
            stream: Stream,
            gate_second: Bool,
        ]),
    kernel!(swiglu "mlp::swiglu_bf16",
        operands = operands![
            gate: Buf,
            up: Buf,
            y: BufMut,
            num_elements: I32,
            stream: Stream,
        ]),
    kernel!(chunked_swiglu_strided "mlp::chunked_swiglu_strided_bf16",
        operands = operands![
            packed: Buf,
            y: BufMut,
            n: I32,
            i: I32,
            row_stride: I32,
            stream: Stream,
        ]),
    kernel!(gpt_oss_glu_strided "mlp::gpt_oss_glu_strided_bf16",
        operands = operands![
            gate: Buf,
            up: Buf,
            y: BufMut,
            rows: I32,
            cols: I32,
            in_stride: I32,
            out_stride: I32,
            stream: Stream,
            limit: F32,
            alpha: F32,
        ]),
    kernel!(swiglu_clamp "mlp::swiglu_clamp_bf16",
        operands = operands![
            gate: Buf,
            up: Buf,
            y: BufMut,
            num_elements: I32,
            limit: F32,
            stream: Stream,
        ]),
    kernel!(chunked_swiglu_clamp "mlp::chunked_swiglu_clamp_bf16",
        operands = operands![
            packed: Buf,
            y: BufMut,
            n: I32,
            i: I32,
            limit: F32,
            stream: Stream,
        ]),
    kernel!(relu2 "mlp::relu2_bf16",
        operands = operands![
            x: Buf,
            y: BufMut,
            num_elements: I32,
            stream: Stream,
        ]),
    // SiTU is not a swiglu variant: the tanh saturates far enough out that a
    // bf16 intermediate loses the distinction the gate exists to make.
    kernel!(situ "mlp::situ_bf16",
        operands = operands![
            gate: Buf,
            up: Buf,
            y: BufMut,
            num_elements: I32,
            beta: F32,
            linear_beta: F32,
            stream: Stream,
        ]),
    kernel!(chunked_situ "mlp::chunked_situ_bf16",
        operands = operands![
            packed: Buf,
            y: BufMut,
            n: I32,
            i: I32,
            beta: F32,
            linear_beta: F32,
            gate_second: Bool,
            stream: Stream,
        ]),
    kernel!(gaussian_topk "mlp::gaussian_topk_bf16",
        operands = operands![
            x: BufMut,
            n: I32,
            dim: I32,
            std_multiplier: F32,
            stream: Stream,
        ]),
    // GeGLU-tanh is not a swiglu variant: `gelu_pytorch_tanh` on the
    // gate is a different function. The packed/pair split is the same
    // binding question.
    // The PAIR form: `(gate, up, out)` with `out` over `gate`. gemma-4's
    // PLE gate is the same call with the relay slice as `up`.
    kernel!(geglu_tanh "mlp::geglu_tanh_bf16", in_place = &[(0, 0)],
        operands = operands![
            gate: Buf,
            up: Buf,
            y: BufMut,
            num_elements: I32,
            stream: Stream,
        ]),
    kernel!(chunked_geglu_tanh "mlp::chunked_geglu_tanh_bf16",
        operands = operands![
            packed: Buf,
            y: BufMut,
            n: I32,
            i: I32,
            stream: Stream,
            gate_second: Bool,
        ]),
    // SwiGLU with a clamp. `swiglu_limit` is a config constant, so this
    // is a different kernel and not a different argument.
    // `gate = glu(gate, up)` -- the gate half is the destination, which
    // is why the driver passes its pointer twice.
    kernel!(gpt_oss_glu "mlp::gpt_oss_glu_bf16", in_place = &[(0, 0)],
        operands = operands![
            gate: Buf,
            up: Buf,
            y: BufMut,
            num_elements: I32,
            stream: Stream,
            limit: F32,
            alpha: F32,
            y_fp16: BufMut,
        ]),
    kernel!(sigmoid_scalar_gate_add "mlp::sigmoid_scalar_gate_add_bf16",
        operands = operands![
            out: BufMut,
            x: Buf,
            scalar_gate: Buf,
            n: I32,
            h: I32,
            stream: Stream,
        ]),
    kernel!(sigmoid_scalar_gate_strided_add "mlp::sigmoid_scalar_gate_strided_add_bf16",
        operands = operands![
            out: BufMut,
            x: Buf,
            scalar_gate: Buf,
            n: I32,
            h: I32,
            stride: I32,
            stream: Stream,
        ]),
    // The shared expert's landing: `out += sigmoid(x . gate) * y`, and
    // `out` IS the residual stream the statement takes as operand 1 --
    // the header calls it "in-place add destination" in as many words.
    kernel!(moe_shared_gate_dot "mlp::sigmoid_dot_scalar_gate_add_bf16",
        in_place = &[(0, 1)],
        operands = operands![
            x: Buf,
            gate_w: Buf,
            out: BufMut,
            y: Buf,
            n: I32,
            h: I32,
            stream: Stream,
        ]),
];
