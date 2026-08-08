//! Quantization, dequantization and dtype casts.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::KernelSig;

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    kernel!(dequant "launch_dequant_kv_cache_layer_to_bf16_active"),
    // 4-bit weights with a bf16 scale per group along K. Distinct from MXFP4
    // (E8M0 byte per 32) and from fp8 -- three quantizations, three
    // statements, because which one a checkpoint ships is a fact the
    // declaration reads.
    kernel!(dequant_wna16_int4b8 "launch_dequant_wna16_int4b8_to_bf16"),
    kernel!(cast_f32_to_bf16 "launch_cast_fp32_to_bf16"),
    kernel!(mxfp4_scales_to_marlin "launch_mxfp4_scales_to_marlin_e8m0"),
    // Three fp8 forms because the SCALE's shape differs -- per tensor, per
    // output channel, per group along K. A property of the checkpoint, so the
    // declaration states which; a driver that guessed would dequantize
    // correctly on one checkpoint and silently wrongly on another.
    kernel!(dequant_fp8_e4m3 "launch_dequant_fp8_e4m3_to_bf16"),
    kernel!(dequant_fp8_e4m3_per_channel "launch_dequant_fp8_e4m3_to_bf16_per_channel"),
    kernel!(dequant_fp8_e4m3_per_group "launch_dequant_fp8_e4m3_to_bf16_per_group"),
    kernel!(dequant_mxfp4 "launch_dequant_mxfp4_to_bf16"),
    kernel!(bf16_to_f32 "launch_bf16_to_fp32"),
    kernel!(f32_to_bf16 "launch_fp32_to_bf16"),
    kernel!(bf16_to_fp16 "launch_bf16_to_fp16"),
];
