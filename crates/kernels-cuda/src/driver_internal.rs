//! Launchers the DRIVER reaches for directly — no DSL statement, no place
//! in the planner's vocabulary, and deliberately not rows of [`crate::KERNELS`]:
//! `model`'s `kernels_table` holds that table and `dsl::cuda` to the same
//! set, and these have no statement a trace could record. The per-family
//! exhaustiveness tests classify them as `DriverInternal` for exactly this
//! reason.
//!
//! They are still LAUNCHES, and the Rust driver still has to make them —
//! which is what this second table is for. Same [`KernelSig`] rows, same
//! `abi::emit_c_shim` proof, same generated bindings; the only difference
//! is which invariant the table answers to. A row joins here when a live
//! seam or the executor needs a launcher the DSL surface correctly lacks.

use kernels::kernel;
use kernels::{KernelSig, operands};

#[rustfmt::skip]
pub static DRIVER_KERNELS: &[KernelSig] = &[
    // The envelope tier: seeded empty at materialize (`KvCacheDeviceOps`),
    // recomputed after eviction, merged after a write. The seed writes
    // +inf/-inf bf16 so the first real merge tightens from the identity.
    kernel!(envelope_seed "layout::launch_envelope_seed_empty_bf16",
        operands = operands![
            env_min: U16sMut, env_max: U16sMut,
            num_pages: I32, num_kv_heads: I32, head_dim: I32, stream: Stream,
        ]),
    kernel!(envelope_recompute "layout::launch_envelope_recompute_bf16",
        operands = operands![
            k_pages: U16s, page_live_lens: I32s,
            env_min: U16sMut, env_max: U16sMut,
            num_pages: I32, page_size: I32, num_kv_heads: I32, head_dim: I32,
            stream: Stream,
        ]),
    kernel!(envelope_merge_written "layout::launch_envelope_merge_written_bf16",
        operands = operands![
            k_curr: U16s, w_page: U32s, w_off: U32s, row_valid: U8s | null,
            env_min: U16sMut, env_max: U16sMut,
            num_tokens: I32, num_kv_heads: I32, head_dim: I32, stream: Stream,
        ]),
    // The QKV split the generated bodies call ~390 times — the loud case
    // the attn exhaustiveness test names.
    kernel!(split_qkv "attn::split_qkv_bf16",
        operands = operands![
            packed: Buf, q_out: BufMut, k_out: BufMut, v_out: BufMut,
            n_tokens: I32, q_dim: I32, kv_dim: I32, stream: Stream,
        ]),
    kernel!(split_qkv_devwin "attn::split_qkv_bf16_devwin",
        operands = operands![
            packed: Buf, q_out: BufMut, k_out: BufMut, v_out: BufMut,
            win_d: U32s, n_max: I32, q_dim: I32, kv_dim: I32, stream: Stream,
        ]),
    // The page-mask packers `FirePageMask` fires.
    kernel!(pack_dense_mask "attn::pack_dense_mask",
        operands = operands![
            kvm_dense: U8s, klen: U32s, qo_indptr: U32s, mask_indptr: I32s,
            packed: U8sMut, b: I32, p_page: I32, stream: Stream,
        ]),
    kernel!(pack_structured_mask "attn::pack_structured_mask",
        operands = operands![
            positions: U32s, klen: U32s, qo_indptr: U32s, mask_indptr: I32s,
            masks: StructuredMasks, packed: U8sMut, b: I32, stream: Stream,
        ]),
    // Beam-repair cell moves, per layer, disjoint spans by contract.
    kernel!(copy_kv_cells "attn::copy_kv_cells_bf16",
        operands = operands![
            layer: KvCacheLayerView, dst_page: U32s, dst_off: U32s,
            src_page: U32s, src_off: U32s, n: I32, stream: Stream,
        ]),
    // The three the LOWERING states without a DSL row, found by
    // `every_lowered_kernel_has_a_bridge_row` on its first run: the
    // emitter-chosen pair (a semantic op picks them, so no trace records
    // a Launch naming them) and the quantized dispatch entry, whose
    // `WeightView` BY VALUE is the operand the handoff predicted would
    // be gemm's friction.
    kernel!(rmsnorm "norm::rmsnorm_bf16",
        operands = operands![
            x: Buf, weight: Buf, y: BufMut,
            num_rows: I32, hidden: I32, eps: F32, stream: Stream,
        ]),
    kernel!(embed "layout::embed_bf16",
        operands = operands![
            token_ids: I32s, weight: Buf, y: BufMut,
            num_tokens: I32, hidden: I32, vocab: I32, stream: Stream,
        ]),
    kernel!(add_bias "norm::add_bias_bf16",
        operands = operands![
            out: BufMut, bias: Buf, num_rows: I32, dim: I32, stream: Stream,
        ]),
    kernel!(act_x_w "gemm::act_x_w",
        operands = operands![
            handle: CublasHandle, act: Buf, w: WeightView, y: BufMut,
            m: I32, n: I32, k: I32, beta: F32,
            act_dtype: DType, y_dtype: DType,
        ]),
];
