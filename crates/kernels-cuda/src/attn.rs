//! Attention: the paged dispatches, the KV writes, MLA, DSA and the sinks.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::{Cap, KernelSig, Prepare};

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    kernel!(flashinfer_decode "attn::dispatch_attention_flashinfer_decode",
        needs = Prepare::DecodePlan, sink = Some("kv.pages"),
        depth_prefix_plan = true),
    kernel!(flashinfer_decode_capture "attn::dispatch_attention_flashinfer_decode_capture",
        needs = Prepare::DecodePlan, sink = Some("kv.pages")),
    kernel!(flashinfer_prefill "attn::dispatch_attention_flashinfer_prefill_bf16",
        needs = Prepare::PrefillPlan, sink = Some("kv.pages")),
    // The plan-free prefill wrapper: it builds an R-shaped plan on the
    // way in, so it owes its caller nothing and cannot be handed a row
    // window — `whole`, and `FireWide` for the same reason XQA is.
    kernel!(flashinfer_prefill_planless "attn::attention_flashinfer_prefill",
        whole = true, needs = Prepare::FireWide, sink = Some("kv.pages")),
    // Head dims flashinfer's prefill template rejects (gemma-4's 512)
    // take a naive paged kernel instead. No plan at all; fire-shaped.
    kernel!(attention_naive_paged "attn::attention_naive_paged",
        whole = true, sink = Some("kv.pages")),
    kernel!(flashinfer_prefill_capture "attn::dispatch_attention_flashinfer_prefill_capture_bf16",
        needs = Prepare::PrefillPlan, sink = Some("kv.pages")),
    kernel!(flashinfer_custom "attn::dispatch_attention_flashinfer_prefill_custom",
        needs = Prepare::CustomPlan, sink = Some("kv.pages")),
    // XQA: its prepare is fire-wide (R-shaped), so the kernel cannot be
    // given a row window — `whole`. And no capture variant of it
    // exists, so it cannot publish scores — `lacks Scores`. Both are
    // hand-written rules today: the first is the model body's
    // `window_one && c.xqa_decode` test, the second a C++ throw.
    kernel!(xqa_decode "attn::attention_xqa_decode_bf16_prepared",
        whole = true, needs = Prepare::FireWide, lacks = &[Cap::Scores]),
    kernel!(qkv_decode_fused "attn::qkv_decode_qk_norm_rope_write_kv_bf16"),
    kernel!(write_kv_explicit "attn::write_kv_explicit_bf16"),
    kernel!(write_kv_to_pages "attn::write_kv_to_pages"),
    kernel!(qkv_decode_fused_devwin "attn::qkv_decode_qk_norm_rope_write_kv_bf16_devwin",
        whole = true, sink = Some("kv.pages")),
    kernel!(write_kv_to_pages_devwin "attn::write_kv_to_pages_bf16_devwin",
        whole = true, sink = Some("kv.pages")),
    kernel!(write_kv_explicit_devwin "attn::write_kv_explicit_bf16_devwin",
        whole = true, sink = Some("kv.pages")),
    // The pair is what `head_dim_padded` COSTS; stating it turns
    // `if (c.head_dim_padded)` in the model body into a fact the trace
    // carries. Row-shaped -- each token's heads pad independently.
    kernel!(pad_head_dim "attn::pad_head_dim_bf16"),
    kernel!(strip_head_dim "attn::strip_head_dim_bf16"),
    // The KV-split's other half: it merges `num_index_sets` partials whose
    // boundaries are the split's, not a row range's.
    kernel!(merge_attention_states "attn::merge_attention_states_bf16", whole = true),
    // Rewrites `[R+1]` indptr arrays, so a row window would compact the wrong
    // requests' page lists.
    kernel!(compact_page_csr "attn::compact_page_csr", whole = true),
    kernel!(attn_score_fold_heads "attn::attn_score_fold_heads", whole = true),
    // MLA's absorb pair -- cuBLAS ops rather than raw launches, which is why
    // a launcher is "anything that issues DEVICE work" and not "anything
    // taking a cudaStream_t". `scripts/kernel-vocabulary-audit.py` learned
    // that the hard way.
    kernel!(mla_absorb_q_to_latent "mla_absorb_q_to_latent_bf16"),
    kernel!(mla_absorb_latent_to_v "mla_absorb_latent_to_v_bf16"),
    // MTP drafts several tokens per step and repairs on rejection, which
    // needs an attention that sees a HISTORY buffer beside the pages (the
    // drafted tokens are not committed -- committing them before acceptance
    // is the thing MTP must not do) and a per-slot pending-hidden shuffle.
    // All four address through `slot_ids` or `qo_indptr`.
    kernel!(attention_mtp_paged_history "attn::attention_mtp_paged_history_bf16",
        whole = true, lacks = &[Cap::Scores]),
    kernel!(flashinfer_prefill_sm90 "attn::dispatch_attention_flashinfer_prefill_sm90_bf16",
        needs = Prepare::PrefillPlan, sink = Some("kv.pages")),
    // Both walk `src_indptr[R+1]`. The window view is how sliding-window
    // attention is expressed without a second cache -- the window is a VIEW
    // over the same pages.
    kernel!(build_window_page_view "attn::build_window_page_view", whole = true),
    kernel!(build_full_split_view "attn::build_full_split_view", whole = true),
    kernel!(flashinfer_decode_bf16 "attn::dispatch_attention_flashinfer_decode_bf16",
        needs = Prepare::DecodePlan, sink = Some("kv.pages")),
    // A SECOND KV cache beside the fine-grained one, holding one entry per
    // `ratio` tokens. Every query attends both and the outputs are merged by
    // their log-sum-exps -- exact, not an approximation: the same algebra
    // flashinfer's own KV-split merge uses.
    kernel!(dsv4_boundary_meta_decode "attn::dsv4_boundary_meta_decode"),
    // Both address through `kv_page_indptr` and the boundary arrays.
    kernel!(dsv4_compress_gather_paged "attn::dsv4_compress_gather_paged_bf16", whole = true),
    kernel!(dsv4_store_comp_entries "attn::dsv4_store_comp_entries_bf16", whole = true),
    // `qo_indptr` + `kv_page_indptr`, like every other paged attention here.
    // No capture variant, so it cannot publish scores; it does publish an LSE,
    // which is what the combine below consumes.
    kernel!(attention_compressed_paged "attn::attention_compressed_paged_bf16",
        whole = true, lacks = &[Cap::Scores]),
    kernel!(combine_attn_outputs "attn::combine_attn_outputs_bf16"),
    // FlashInfer publishes its LSE in log2 and the combine works in ln. A
    // unit conversion, stated so a reader never has to guess which base an
    // LSE is in.
    kernel!(lse_log2_to_ln "attn::lse_log2_to_ln"),
    kernel!(write_kv_to_pages_bf16 "attn::write_kv_to_pages_bf16"),
    kernel!(attention_naive_paged_bf16 "attn::attention_naive_paged_bf16", whole = true),
    kernel!(attn_res_blend "attn::attn_res_blend_bf16"),
    // The unfused counterpart of `mla_prepare`. `tokens` is their only
    // extent, so unlike the fused prepare they are NOT `whole` -- which is
    // the reason a deployment might bind them instead.
    kernel!(kimi_split_kv_a_norm "attn::kimi_split_kv_a_norm_bf16"),
    kernel!(kimi_split_q_b "attn::kimi_split_q_b_bf16"),
    // glm5 attends SPARSELY: a small side network scores every (query, key)
    // pair and only the top-k keys per query are attended.
    kernel!(dsa_index_q_rope "attn::dsa_index_q_rope_bf16"),
    kernel!(dsa_index_knorm_rope "attn::dsa_index_knorm_rope_bf16"),
    // `whole`, and here the reason is the ALGEBRA rather than the addressing:
    // query `i` scores keys `0..=i`, so a row window starting anywhere but
    // zero cannot see the keys it must rank against.
    kernel!(dsa_index_topk_mask "attn::dsa_index_topk_mask", whole = true),
    // deepseek_v4, glm5 and kimi_k3 attend through a compressed KV: a
    // `kv_lora_rank`-wide latent row plus a small rope-carrying companion,
    // with the heads reconstructed on the way in. A different attention
    // algebra, not a different head count.
    //
    // The two paged statements are `whole` because they address through
    // `qo_indptr` / `kv_page_indptr` / `kv_last_page_lens`, which are
    // R-shaped: a row window would leave that arithmetic pointing at the
    // wrong request. The dispatch is not -- like the flashinfer dispatches,
    // it reads a plan built over the whole fire and still covers a row range.
    kernel!(mla_prepare "attn::mla_prepare_bf16", whole = true),
    kernel!(write_mla_to_pages "attn::write_mla_to_pages", whole = true),
    // No capture variant of this dispatch exists, so it cannot publish the
    // score matrix an `attn.out` observer asks for. It does publish an LSE,
    // which is a different thing and not what the capability names.
    kernel!(attention_mla "attn::dispatch_attention_mla_bf16",
        needs = Prepare::MlaPlan, lacks = &[Cap::Scores]),
    // The custom-mask prefill in its PLAN-FREE form: it takes the indptrs and
    // the mask directly and builds its R-shaped plan on the way in, so it
    // owes no prepare and cannot take a row window -- `whole`, and `FireWide`
    // for the same reason XQA is. gemma-3n binds this rather than the planned
    // `flashinfer_custom` above.
    kernel!(flashinfer_custom_planless "attn::attention_flashinfer_prefill_custom",
        whole = true, needs = Prepare::FireWide, sink = Some("kv.pages")),
    kernel!(logit_softcap "attn::logit_softcap_bf16"),
    // Six statements in one launch; the only value that survives is q.
    kernel!(qkv_packed_post "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16",
        sink = Some("kv.pages")),
    kernel!(attention_sink_rescale "attn::attention_sink_rescale_bf16"),
    kernel!(mtp_shift_hidden "attn::mtp_shift_hidden_bf16", whole = true),
    kernel!(mtp_update_pending_hidden "attn::mtp_update_pending_hidden_bf16", whole = true),
    kernel!(dequant "attn::dequant_kv_cache_layer_to_bf16_active"),
];
