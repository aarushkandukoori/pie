//! Linear attention and state-space recurrences: GDN, KDA, mamba, and the
//! causal conv that feeds them.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.
//!
//! Three spellings this family added to the operand vocabulary: `I64` (the
//! `long long` slot strides), `U16s`/`U16sMut` (the FlashInfer SSU speaks
//! `uint16_t*` for its bf16 pointers rather than `void*`), and the
//! pointer-table trio `Bufs`/`BufTableMut`/`BufMutTableMut` (the MoE
//! pointer builders read weight tables and write the A/B/C tables the
//! grouped GEMMs then consume).

use kernels::kernel;
use kernels::{KernelSig, Ret, operands};

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    // The other mamba scan: nemotron_h takes FlashInfer's SSU on sm90+ and
    // its own batched kernel elsewhere. A TRIED launch — the `bool` says
    // whether FlashInfer took it, and the caller falls back when it did not —
    // which is what `ret` states.
    kernel!(flashinfer_mamba_ssu "ssm::flashinfer_mamba_ssu_bf16", whole = true,
        ret = Ret::Bool,
        operands = operands![
            conv_out: U16s, dt: U16s, a: F32s, d: U16s, dt_bias: U16s,
            state_base: U16sMut, slot_ids: I32s, y: U16sMut,
            batch: I32, num_heads: I32, head_dim: I32, state_size: I32,
            num_groups: I32, conv_dim: I32, intermediate: I32,
            state_cache_size: I32, stream: Stream,
        ]),
    // Unbatched twins of the `_batched` forms below -- a legacy parity
    // entrypoint and a single-request fast path. Not `whole`, for the reason
    // the batched ones are not: their `B` is the batch, not a window into it.
    // The `_state_bf16` pairing is a precision BINDING a deployment states,
    // the same way the batched rows spell it.
    kernel!(gdn_step_single "ssm::recurrent_gated_delta_step",
        operands = operands![
            q_norm: F32s, k_norm: F32s, v: F32s, g_log: F32s, beta: F32s,
            state: F32sMut, out: F32sMut,
            b: I32, v_h: I32, k_d: I32, v_d: I32, stream: Stream,
        ]),
    kernel!(gdn_step_single_state_bf16 "ssm::recurrent_gated_delta_step_state_bf16",
        operands = operands![
            q_norm: F32s, k_norm: F32s, v: F32s, g_log: F32s, beta: F32s,
            state: BufMut, out: F32sMut,
            b: I32, v_h: I32, k_d: I32, v_d: I32, stream: Stream,
        ]),
    kernel!(gdn_prefill_single "ssm::chunk_gated_delta_prefill",
        operands = operands![
            q_norm: F32s, k_norm: F32s, v: F32s, g_log: F32s, beta: F32s,
            state: F32sMut, out: F32sMut,
            t: I32, v_h: I32, k_d: I32, v_d: I32, chunk_size: I32,
            stream: Stream,
        ]),
    kernel!(gdn_prefill_single_state_bf16 "ssm::chunk_gated_delta_prefill_state_bf16",
        operands = operands![
            q_norm: F32s, k_norm: F32s, v: F32s, g_log: F32s, beta: F32s,
            state: BufMut, out: F32sMut,
            t: I32, v_h: I32, k_d: I32, v_d: I32, chunk_size: I32,
            stream: Stream,
        ]),
    kernel!(causal_conv1d_prefill_single "ssm::causal_conv1d_prefill_bf16",
        operands = operands![
            x: Buf, weight: Buf, bias: Buf | null, y: BufMut,
            state_out: BufMut | null,
            n: I32, c: I32, k: I32, stream: Stream,
        ]),
    // The third linear-attention shape here, and not a variant of the other
    // two: mamba carries a `[head_dim, state_size]` slab per head and
    // advances it with a scalar `dA` from a per-token `dt` -- a selective
    // scan, not a delta rule. A different state SHAPE, which is why none of
    // the GDN or KDA rows stand in for it.
    kernel!(nemotron_mamba_split "ssm::nemotron_mamba_split_bf16",
        operands = operands![
            projected: Buf, gate: BufMut, conv_in: BufMut, dt: BufMut,
            n: I32, projection_dim: I32, intermediate: I32, conv_dim: I32,
            num_heads: I32, stream: Stream,
        ]),
    kernel!(nemotron_prepare_mamba_params "ssm::nemotron_prepare_mamba_params",
        operands = operands![
            a_log: Buf, d: Buf, dt_bias: Buf,
            a: F32sMut, d_f32: F32sMut, dt_bias_f32: F32sMut,
            num_heads: I32, stream: Stream,
        ]),
    kernel!(nemotron_prepare_mamba_dt_da "ssm::nemotron_prepare_mamba_dt_da",
        operands = operands![
            dt: Buf, a: F32s, dt_bias: F32s, dt_out: F32sMut, da_out: F32sMut,
            n: I32, num_heads: I32, time_step_min: F32, stream: Stream,
        ]),
    // `whole` for both reasons this table collects: it addresses through
    // `slot_ids` and `qo_indptr`, and the scan carries state token to token,
    // so a row window would resume from the wrong slab.
    kernel!(nemotron_mamba_ssm "ssm::nemotron_mamba_ssm_batched_bf16", whole = true,
        operands = operands![
            conv_out: Buf, dt: Buf, a: F32s, d: F32s, dt_bias: F32s,
            dt_precomputed: F32s | null, da_precomputed: F32s | null,
            ssm_state_base: BufMut, slot_ids: I32s, qo_indptr: U32s,
            y: BufMut, r: I32, num_heads: I32, head_dim: I32, state_size: I32,
            n_groups: I32, conv_dim: I32, intermediate: I32,
            time_step_min: F32, sequence_prefill: Bool, stream: Stream,
        ]),
    // Advances a slot's conv window in place; a row window advances the
    // wrong slots.
    kernel!(causal_conv1d_update "ssm::causal_conv1d_update_bf16", whole = true,
        operands = operands![
            x: Buf, weight: Buf, bias: Buf | null, state: BufMut, y: BufMut,
            c: I32, k: I32, stream: Stream,
        ]),
    // kimi_k3's linear-attention half. The gated delta rule qwen3_5 runs,
    // with the decay per KEY CHANNEL rather than per head -- which is why
    // these exist beside the GDN kernels instead of reusing them with a
    // broadcast.
    kernel!(kda_gate_beta "ssm::kda_gate_beta_bf16",
        operands = operands![
            raw_g: Buf, raw_beta: Buf, a_log: F32s, dt_bias: F32s,
            gate_out: F32sMut, beta_out: F32sMut,
            t: I32, h: I32, d: I32, lower_bound: F32, stream: Stream,
        ]),
    // `slot_ids` is indexed `0..R` against the fire's request order, so a row
    // window would advance the wrong slots.
    kernel!(kda_recurrent_step "ssm::kda_recurrent_step_batched", whole = true,
        operands = operands![
            q_norm: F32s, k_norm: F32s, v: F32s, gate: F32s, beta: F32s,
            state_base: F32sMut, slot_ids: I32s, slot_stride_elems: I64,
            out: F32sMut, r: I32, h: I32, d: I32, stream: Stream,
        ]),
    // `whole` twice over: it walks windows out of `qo_indptr`, and the
    // recurrence has a strict per-token state dependency -- a row window
    // would start the scan from the wrong state, which is a different answer
    // rather than a misaddressed one.
    kernel!(kda_prefill "ssm::kda_prefill_batched", whole = true,
        operands = operands![
            q_norm: F32s, k_norm: F32s, v: F32s, gate: F32s, beta: F32s,
            state_base: F32sMut, slot_ids: I32s, qo_indptr: U32s,
            slot_stride_elems: I64, out: F32sMut,
            r: I32, h: I32, d: I32, stream: Stream,
        ]),
    kernel!(kda_o_norm_gated "ssm::kda_o_norm_gated_bf16",
        operands = operands![
            o: F32s, g: Buf, weight: F32s, out: BufMut,
            t: I32, h: I32, d: I32, eps: F32, stream: Stream,
        ]),
    kernel!(gdn_conv_update "ssm::causal_conv1d_update_batched_bf16",
        operands = operands![
            x: Buf, weight: Buf, bias: Buf | null, state_base: BufMut,
            slot_ids: I32s, slot_stride_elems: I64, y: BufMut,
            r: I32, c: I32, k: I32, stream: Stream,
        ]),
    // The defaulted tail (`write_state = true`, null `commit_len` and
    // `write_state_mask`) is stated, not omitted: a default is a value the
    // caller passes by saying nothing, and the row records what the launch
    // receives.
    kernel!(gdn_conv_prefill "ssm::causal_conv1d_prefill_batched_bf16",
        operands = operands![
            x: Buf, weight: Buf, bias: Buf | null, y: BufMut,
            state_out_base: BufMut, slot_ids: I32s, qo_indptr: U32s,
            slot_stride_elems: I64, r: I32, c: I32, k: I32, stream: Stream,
            write_state: Bool, commit_len: I32s | null,
            write_state_mask: U8s | null,
        ]),
    // The fused post-conv prep: q/k split + L2 norm, v widened to fp32,
    // g/beta gating from a/b with A_log + dt_bias. `qkv_post` is the
    // conv's `[N, 2*K_h*K_d + V_h*V_d]` bf16 output in [q|k|v] order.
    kernel!(qwen_gdn_post_conv_prep "ssm::qwen_gdn_post_conv_prep_bf16",
        operands = operands![
            qkv_post: Buf, a: Buf, b: Buf, a_log: Buf, dt_bias: Buf,
            q_norm_kh: F32sMut, k_norm_kh: F32sMut, v_fp32: F32sMut,
            g_log_out: F32sMut, beta_out: F32sMut,
            n: I32, k_h: I32, v_h: I32, k_d: I32, v_d: I32, conv_dim: I32,
            stream: Stream,
        ]),
    kernel!(gdn_step "ssm::recurrent_gated_delta_step_batched",
        operands = operands![
            q_norm: F32s, k_norm: F32s, v: F32s, g_log: F32s, beta: F32s,
            state_base: F32sMut, slot_ids: I32s, slot_stride_elems: I64,
            out: F32sMut, r: I32, v_h: I32, k_d: I32, v_d: I32,
            stream: Stream,
        ]),
    kernel!(gdn_step_gqa "ssm::recurrent_gated_delta_step_batched_gqa",
        operands = operands![
            q_norm_kh: F32s, k_norm_kh: F32s, v: F32s, g_log: F32s,
            beta: F32s, state_base: F32sMut, slot_ids: I32s,
            slot_stride_elems: I64, out: F32sMut,
            r: I32, k_h: I32, v_h: I32, k_d: I32, v_d: I32, stream: Stream,
        ]),
    kernel!(gdn_step_state_bf16 "ssm::recurrent_gated_delta_step_batched_state_bf16",
        operands = operands![
            q_norm: F32s, k_norm: F32s, v: F32s, g_log: F32s, beta: F32s,
            state_base: BufMut, slot_ids: I32s, slot_stride_elems: I64,
            out: F32sMut, r: I32, v_h: I32, k_d: I32, v_d: I32,
            stream: Stream,
        ]),
    kernel!(gdn_step_gqa_state_bf16 "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16",
        operands = operands![
            q_norm_kh: F32s, k_norm_kh: F32s, v: F32s, g_log: F32s,
            beta: F32s, state_base: BufMut, slot_ids: I32s,
            slot_stride_elems: I64, out: F32sMut,
            r: I32, k_h: I32, v_h: I32, k_d: I32, v_d: I32, stream: Stream,
        ]),
    kernel!(gdn_prefill_fla "ssm::chunk_gated_delta_prefill_batched",
        operands = operands![
            q_norm: F32s, k_norm: F32s, v: F32s, g_log: F32s, beta: F32s,
            state_base: F32sMut, slot_ids: I32s, qo_indptr: U32s,
            slot_stride_elems: I64, out: F32sMut,
            r: I32, k_h: I32, v_h: I32, k_d: I32, v_d: I32, stream: Stream,
            write_state: Bool, commit_len: I32s | null,
            write_state_mask: U8s | null,
        ]),
    kernel!(gdn_prefill_fla_state_bf16 "ssm::chunk_gated_delta_prefill_batched_state_bf16",
        operands = operands![
            q_norm: F32s, k_norm: F32s, v: F32s, g_log: F32s, beta: F32s,
            state_base: BufMut, slot_ids: I32s, qo_indptr: U32s,
            slot_stride_elems: I64, out: F32sMut,
            r: I32, k_h: I32, v_h: I32, k_d: I32, v_d: I32, stream: Stream,
            write_state: Bool, commit_len: I32s | null,
            write_state_mask: U8s | null,
        ]),
    // The `_cached` pair drops `K_h` (no GQA form) and `commit_len` (the
    // repair forward advances state; the verify never boundary-writes) --
    // its arity differs from the `_fla` pair by TWO operands, not one.
    kernel!(gdn_prefill_cached "ssm::chunk_gated_delta_prefill_batched_cached",
        operands = operands![
            q_norm: F32s, k_norm: F32s, v: F32s, g_log: F32s, beta: F32s,
            state_base: F32sMut, slot_ids: I32s, qo_indptr: U32s,
            slot_stride_elems: I64, out: F32sMut,
            r: I32, v_h: I32, k_d: I32, v_d: I32, stream: Stream,
            write_state: Bool, write_state_mask: U8s | null,
        ]),
    kernel!(gdn_prefill_cached_state_bf16
        "ssm::chunk_gated_delta_prefill_batched_cached_state_bf16",
        operands = operands![
            q_norm: F32s, k_norm: F32s, v: F32s, g_log: F32s, beta: F32s,
            state_base: BufMut, slot_ids: I32s, qo_indptr: U32s,
            slot_stride_elems: I64, out: F32sMut,
            r: I32, v_h: I32, k_d: I32, v_d: I32, stream: Stream,
            write_state: Bool, write_state_mask: U8s | null,
        ]),
    kernel!(gdn_prefill_warp_tiled_gqa "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa",
        operands = operands![
            q_norm_kh: F32s, k_norm_kh: F32s, v: F32s, g_log: F32s,
            beta: F32s, state_base: F32sMut, slot_ids: I32s, qo_indptr: U32s,
            slot_stride_elems: I64, out: F32sMut,
            r: I32, k_h: I32, v_h: I32, k_d: I32, v_d: I32, stream: Stream,
            write_state: Bool, write_state_mask: U8s | null,
        ]),
    kernel!(gdn_prefill_warp_tiled_gqa_state_bf16
        "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16",
        operands = operands![
            q_norm_kh: F32s, k_norm_kh: F32s, v: F32s, g_log: F32s,
            beta: F32s, state_base: BufMut, slot_ids: I32s, qo_indptr: U32s,
            slot_stride_elems: I64, out: F32sMut,
            r: I32, k_h: I32, v_h: I32, k_d: I32, v_d: I32, stream: Stream,
            write_state: Bool, write_state_mask: U8s | null,
        ]),
    kernel!(repeat_interleave_heads "ssm::repeat_interleave_heads_fp32",
        operands = operands![
            input: F32s, out: F32sMut,
            n: I32, k_h: I32, v_h: I32, d: I32, stream: Stream,
        ]),
    // KDA's arithmetic is fp32 throughout, so operands living in bf16 in the
    // workspace cross explicitly. Launches, so the trace records them.
    kernel!(l2norm_scale_to_f32 "ssm::l2norm_scale_bf16_to_fp32",
        operands = operands![
            x: Buf, y: F32sMut, n: I32, hidden: I32, scale: F32, eps: F32,
            stream: Stream,
        ]),
    kernel!(bf16_to_f32 "ssm::bf16_to_fp32",
        operands = operands![
            x: Buf, y: F32sMut, n: Usize, stream: Stream,
        ]),
    kernel!(f32_to_bf16 "ssm::fp32_to_bf16",
        operands = operands![
            x: F32s, y: BufMut, n: Usize, stream: Stream,
        ]),
    kernel!(zamba_rmsnorm_gated "ssm::zamba_rmsnorm_gated_bf16",
        operands = operands![
            x: Buf, gate: Buf, weight: Buf, y: BufMut,
            n: I32, hidden: I32, gate_stride: I32, group_size: I32, eps: F32,
            stream: Stream,
        ]),
    kernel!(build_nemotron_moe_ptrs_aligned "ssm::build_nemotron_moe_ptrs_aligned_bf16",
        whole = true,
        operands = operands![
            expert_ids: I32s, up_weight_ptrs: Bufs, down_weight_ptrs: Bufs,
            aligned_in: Buf, aligned_up: BufMut, aligned_act: BufMut,
            aligned_out: BufMut,
            a_up_ptrs: BufTableMut, b_up_ptrs: BufTableMut,
            c_up_ptrs: BufMutTableMut,
            a_down_ptrs: BufTableMut, b_down_ptrs: BufTableMut,
            c_down_ptrs: BufMutTableMut,
            max_blocks: I32, block_size: I32, hidden: I32, intermediate: I32,
            stream: Stream,
        ]),
    kernel!(build_nemotron_moe_ptrs_decode "ssm::build_nemotron_moe_ptrs_decode_batched_bf16",
        whole = true,
        operands = operands![
            topk_idx: I32s, topk_w: F32s, up_weight_ptrs: Bufs,
            down_weight_ptrs: Bufs, norm_x: Buf,
            expert_up: BufMut, expert_act: BufMut, expert_out: BufMut,
            a_up_ptrs: BufTableMut, b_up_ptrs: BufTableMut,
            c_up_ptrs: BufMutTableMut,
            a_down_ptrs: BufTableMut, b_down_ptrs: BufTableMut,
            c_down_ptrs: BufMutTableMut, weights_out: F32sMut,
            n: I32, top_k: I32, hidden: I32, intermediate: I32,
            stream: Stream,
        ]),
];
