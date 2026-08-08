//! Turning logits into tokens.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::KernelSig;

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    // Produces TOKEN IDS, not logits: a greedy-decode fast path that never
    // materializes the vocab-wide row, which is why it is its own statement
    // rather than `lm_head` followed by an argmax.
    kernel!(lm_head_gemv_argmax_int8 "sample::lm_head_gemv_argmax_int8"),
];
