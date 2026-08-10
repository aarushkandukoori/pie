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
    // The qwen3_vl vision TOWER, bridged at tower granularity — one row
    // that is a whole subgraph, the flashinfer-dispatch precedent (see
    // the retirement wiki's VL judgment). The wrapper rebuilds the C++
    // weights struct from the flat tables; the walk and its host prep
    // (bilinear pos-embed interp, 2-D rope ids, spatial-merge reorder,
    // the f32→bf16 pixel cast) stay `qwen3_vl_tower.cu`'s. The pixel/
    // grid/anchor operands and the pointer tables are HOST pointers —
    // the step hands them over host-side, the C++ shape. `whole`: the
    // tower addresses rows through per-image anchor offsets, and a row
    // window would encode the wrong images.
    kernel!(qwen3vl_tower_scatter "vision::qwen3vl_scatter", whole = true,
        operands = operands![
            patch_w: Buf, patch_b: Buf | null, pos_embed: Buf,
            block_w: Bufs, depth: I32,
            merger_w: Bufs,
            deepstack_w: Bufs, deepstack_layers: I32s,
            hidden: I32, heads: I32, intermediate: I32, patch_size: I32,
            temporal_patch: I32, merge_size: I32, in_channels: I32,
            out_hidden: I32, num_pos_embed: I32, ln_eps: F32,
            rope_theta: F32,
            pixels_h: F32s, pixel_byte_indptr_h: U32s, grids_h: U32s,
            anchor_rows_h: U32s, num_images: I32,
            hidden_rows: BufMut, n_rows: I32,
            deepstack_scratch: BufMut | null, num_deep: I32,
            blas: CublasHandle, stream: Stream,
        ]),
    // gemma-4's STANDALONE towers — the encode-ABI pair (host pixels /
    // log-mel in, HOST bf16 embedding rows out, anchor-segmented CSR).
    // Layer tables are `Ty::Bufs` at stride 41 (vision) / 62 (audio);
    // the field orders live in `vision/gemma4_towers_c.hpp`. The output
    // operands are HOST buffers — `PieEncodeDesc`'s own shape.
    kernel!(gemma4_vision_encode "vision::gemma4_vision_encode", whole = true,
        operands = operands![
            patch_w: Buf, pos_table: Buf, embed_proj: Buf,
            layer_w: Bufs, depth: I32,
            hidden: I32, heads: I32, intermediate: I32,
            pos_table_size: I32, text_hidden: I32, pool_kernel: I32,
            eps: F32, theta: F32,
            pixels_h: F32s, pixel_byte_indptr_h: U32s,
            patch_positions_h: U32s, anchor_rows_h: U32s, num_images: I32,
            output_rows_h: U16sMut, output_bytes: Usize,
            output_row_indptr_h: U32sMut, stream: Stream,
        ]),
    kernel!(gemma4_audio_encode "vision::gemma4_audio_encode", whole = true,
        operands = operands![
            sscp0_conv: Buf, sscp0_norm: Buf, sscp1_conv: Buf,
            sscp1_norm: Buf, sscp_input_proj: Buf,
            output_proj_w: Buf, output_proj_b: Buf, embed_proj: Buf,
            layer_w: Bufs, depth: I32,
            hidden: I32, heads: I32, conv_kernel: I32, n_mel: I32,
            sscp_ch0: I32, sscp_ch1: I32, out_proj_dims: I32,
            text_hidden: I32, chunk_size: I32, context_left: I32,
            context_right: I32, logit_cap: F32, residual_weight: F32,
            eps: F32,
            features_h: F32s, feature_byte_indptr_h: U32s,
            anchor_rows_h: U32s, num_clips: I32,
            output_rows_h: U16sMut, output_bytes: Usize,
            output_row_indptr_h: U32sMut, stream: Stream,
        ]),
];
