//! The decode-step ABI: regions, IO slots, kernel kinds, and the graph key.
//!
//! `decode_abi.hpp` is the backend-agnostic contract three lanes share — the
//! heap allocator, the Metal-4 wrappers and the encoder — and declares
//! itself pure: "NO Metal/ObjC types in this header". This module is its
//! vocabulary half: the heap [`Region`]s, the [`IoSlot`] table, the
//! [`Kernel`] kind enum every attribution/PSO/weight table is indexed by,
//! [`ArgmaxParams`], and the [`ForwardGraphKey`] the command-buffer cache
//! buckets on. The ~30 `bind::` argument-table layouts are *not* here: each
//! is the ABI of one kernel and lands beside the encoder that binds it,
//! where its slot documentation has something to be checked against.
//!
//! ## The count that was forty kinds short
//!
//! Every table indexed by `Kernel` is sized from the kind count, and the
//! C++ once spelled that count `G4PleResidual + 1` — forty-four kinds short
//! of the real end. `psos[LmHeadUntied]` then wrote and read past the
//! array, the untied head's dispatch got the multi-batch table's GDN
//! pipeline, and the logits buffer was left exactly as it found it: every
//! logit zero, every token 0, and not one error anywhere. The C++ fix made
//! `KindCount` an enum member so it tracks the end by construction. The
//! Rust fix is stronger: the [`kernels!`](macro) macro emits the variant
//! list and [`Kernel::ALL`]/[`Kernel::COUNT`] from the *same* token list,
//! so a kind appended to the enum is counted because it is the enum — there
//! is no second spelling of the end to fall behind. And a `[T; Kernel::COUNT]`
//! table indexed through [`Kernel::index`] cannot be indexed past, because
//! the index is the discriminant of a value that exists.
//!
//! ## The values are ABI
//!
//! The C++ says "APPEND ONLY" five separate times: the numeric values of
//! `Kernel`, `IoSlot` and `Region` are part of the M=1 argument-table ABI
//! and the serialized surfaces. The anchor tests at the bottom pin the
//! block boundaries, so an insertion anywhere upstream of one moves a
//! pinned value and fails loudly instead of renumbering forty kinds
//! silently.

/// One fixed region of the decode heap.
///
/// One `MTLHeap`, fixed offsets, residency requested once; per token only
/// the IO slot contents change (invariant I2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Region {
    /// Load-once read-only weights: matvec banks, norms, the tied head.
    Weights = 0,
    /// The M=1 contiguous K/V ring for the full-attention layers.
    Kv = 1,
    /// GDN resident conv + recurrent state, updated in place (I4).
    State = 2,
    /// The activation ping-pong pool ([`SCRATCH_POOL`] buffers).
    Scratch = 3,
    /// Per-token CPU/GPU-touched scalars and the logits.
    Io = 4,
    /// Multi-batch CSR IO buffers (the [`IoSlot`] tail). Zero-sized at M=1.
    MbIo = 5,
    /// The separate NHD paged K/V pool the paged kernels read. The M=1 ring
    /// above is untouched.
    KvPagePool = 6,
}

/// The activation ping-pong pool's size cap.
///
/// The cap, not the allocation: the executor commits `colors_used` slots —
/// six for a dense stack, eight routed, nine routed with a shared expert.
/// Deliberately the current peak and not a round number above it, so the
/// next value that does not fit says so instead of binding to nothing.
pub const SCRATCH_POOL: usize = 9;

/// One slot of the IO region: GPU-read buffers, never encode-time bytes.
///
/// Invariant I1: keeping the scalars in buffers is what makes the encoded
/// command buffer byte-identical every token, so encode(N+1) overlaps
/// GPU(N). M=1 writes index `[0]` of each scalar slot; the CSR tail is
/// bound only for M>1 fires. Values are append-only ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum IoSlot {
    /// `u32[max_tokens]` — the fired token ids.
    TokenId = 0,
    /// `u32[max_tokens]` — absolute positions (rope and the KV append read).
    Position = 1,
    /// `u32[max_tokens]` — per-token KV extent (the decode SDPA reads).
    SeqLen = 2,
    /// `bf16[vocab]` out at M=1; `bf16[max_tokens, vocab]` for paged fires.
    Logits = 3,
    /// `u32[max_tokens]` — the optional device-argmax substrate (I3).
    NextToken = 4,
    /// `u32[R+1]` — per-request token spans.
    QoIndptr = 5,
    /// `u32[R+1]` — per-request page-list base.
    KvPageIndptr = 6,
    /// `u32[total_pages]` — flat physical page ids.
    KvPageIndices = 7,
    /// `u32[R]` — fill count of each request's last page.
    KvLastPageLens = 8,
    /// `u32[R]` — recurrent-state slot per request.
    RsSlotIds = 9,
    /// `u8[R]` — per-slot fresh/continue flags.
    RsSlotFlags = 10,
    /// `u32[N]` — per-token owning request.
    ReqOfToken = 11,
    /// `u32[N]` — `rs_slot_ids[req_of_token[t]]`; the slotted GDN kernel
    /// indexes state by token row, and keeping the expansion distinct from
    /// [`RsSlotIds`](Self::RsSlotIds) keeps mixed fires unambiguous.
    SlotOfToken = 12,
    /// `u32[N]` — explicit physical destination page per appended token.
    /// Separate from the read CSR: a fork may write a new page while
    /// retaining a shared prefix.
    WPage = 13,
    /// `u32[N]` — in-page destination offset per appended token.
    WOff = 14,
    /// `u8[N, stride]` — the dense attention allow-mask.
    AttnMask = 15,
    /// `u32[1]` — the dense mask's row stride.
    AttnMaskStride = 16,
    /// `u8[N]` — whether each row consumes the mask.
    AttnMaskEnabled = 17,
    /// `u32[S]` — which body rows the fire samples, in readout order. The
    /// tail runs over these and no others: the LM head is the step's most
    /// expensive dispatch and a prefill reads one row per request.
    SampleRows = 18,
}

/// How many [`IoSlot`]s there are.
pub const IO_SLOT_COUNT: usize = IoSlot::SampleRows as usize + 1;

/// The argmax kernel's constant block, replicated exactly.
///
/// The EOS ids ride inline (at most eight); `n_eos = 0` means the EOS flag
/// never fires. Shared storage, so the executor rewrites vocab and stop ids
/// per generation without a rebind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct ArgmaxParams {
    /// The logits row width.
    pub vocab: u32,
    /// Valid entries in [`eos_ids`](Self::eos_ids).
    pub n_eos: u32,
    /// Stop-token ids the device compares the winner against.
    pub eos_ids: [u32; 8],
}

/// The size the Metal side agrees on.
const _: () = assert!(size_of::<ArgmaxParams>() == 40);

/// Declares [`Kernel`] and derives its list and count from one token list.
///
/// This is the structural fix for the count that was forty kinds short:
/// `ALL` and `COUNT` come from the same tokens the variants do, so there is
/// no second spelling of the enum's end to fall behind.
macro_rules! kernels {
    ($(#[$enum_meta:meta])* $vis:vis enum $name:ident {
        $($(#[$meta:meta])* $variant:ident),+ $(,)?
    }) => {
        $(#[$enum_meta])*
        $vis enum $name {
            $($(#[$meta])* $variant),+
        }
        impl $name {
            /// Every kind, in ABI order.
            $vis const ALL: [$name; [$($name::$variant),+].len()] =
                [$($name::$variant),+];
            /// How many kinds there are — the size of every table indexed
            /// by [`Kernel::index`].
            $vis const COUNT: usize = Self::ALL.len();
            /// This kind's table index: its ABI discriminant.
            #[must_use]
            $vis const fn index(self) -> usize {
                self as usize
            }
        }
    };
}

kernels! {
    /// One dispatch kind of the decode DAG.
    ///
    /// A kind is a *weight name* as much as a kernel: `weights_for_kind`
    /// switches on it and nothing else, which is why families that reuse a
    /// kernel under a different tensor name get their own kinds, and why
    /// the numeric values are append-only ABI. Kinds sharing a `.metal`
    /// differ only by dispatch dims and golden tag.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    #[repr(u8)]
    pub enum Kernel {
        /// Embedding gather off the tied 4-bit head bundle.
        EmbedGather,
        /// Pre-attention RMS norm.
        Rms,
        /// GDN in-projection, 4-bit qkv.
        QmvIn,
        /// GDN in-projection, 4-bit z gate.
        QmvInZ,
        /// GDN `a` projection, dense bf16.
        GdnInA,
        /// GDN `b` projection, dense bf16.
        GdnInB,
        /// The hoisted GDN q/k prologue (one dispatch per head).
        GdnPrep,
        /// The fused GDN core: conv+silu, norms, gating, recurrent step.
        GdnCore,
        /// The gated RMS norm closing the GDN block.
        GatedRms,
        /// GDN out-projection.
        QmvOut,
        /// Residual add.
        Residual,
        /// Attention q projection.
        QmvQ,
        /// Deinterleave of the 2x-wide gated-q projection.
        QSplit,
        /// Attention k projection.
        QmvK,
        /// Attention v projection.
        QmvV,
        /// Per-head q norm.
        QNorm,
        /// Per-head k norm.
        KNorm,
        /// Rope on q.
        Rope,
        /// Rope on k.
        RopeK,
        /// The M=1 contiguous-ring KV append.
        KvAppend,
        /// The M=1 single-pass decode attention.
        Sdpa,
        /// `attn *= sigmoid(gate)` before the o projection.
        AttnGate,
        /// Attention o projection.
        QmvO,
        /// Pre-FFN RMS norm.
        FfnRms,
        /// FFN gate projection.
        QmvGate,
        /// FFN up projection.
        QmvUp,
        /// SwiGLU.
        SiluMul,
        /// FFN down projection.
        QmvDown,
        /// The layer's closing residual add.
        LayerOut,
        /// The final RMS norm.
        FinalRms,
        /// The tied LM head matvec.
        QmvLmHead,
        /// The optional device argmax (I3 substrate).
        Argmax,
        /// The paged KV scatter (M>1).
        KvAppendPaged,
        /// The paged-attention read (M>1).
        SdpaPaged,
        /// The slot-indexed GDN core (S>1).
        GdnCoreSlotted,
        /// The slot-indexed GDN prologue (S>1).
        GdnPrepSlotted,
        /// gemma4 `post_attention_layernorm`.
        G4AttnPostNorm,
        /// gemma4 `pre_feedforward_layernorm`.
        G4FfnPreNorm,
        /// gemma4 `post_feedforward_layernorm`.
        G4FfnPostNorm,
        /// gemma4's weightless RMS on v before the KV write.
        G4VNorm,
        /// gemma4 `gelu_tanh(gate) * up`.
        G4Geglu,
        /// gemma4's learned per-layer gain.
        G4LayerScalar,
        /// gemma4 `cap * tanh(logits / cap)`.
        G4Softcap,
        /// gemma4's sampled-row compaction before the tail.
        G4RowGather,
        /// gemma4 sliding-window decode attention.
        G4SdpaSliding,
        /// gemma4 `embed_tokens_per_layer` gather.
        G4PleTokenGather,
        /// gemma4 `per_layer_model_projection` matvec.
        G4PleProjGemv,
        /// gemma4 `per_layer_projection_norm`.
        G4PleProjNorm,
        /// gemma4 `(proj + token) * 1/sqrt(2)`.
        G4PleCombine,
        /// gemma4 `per_layer_input_gate` matvec.
        G4PleGateGemv,
        /// gemma4 `gelu_tanh(gate) * ple`.
        G4PleGeglu,
        /// gemma4 `per_layer_projection` matvec.
        G4PleProjLayerGemv,
        /// gemma4 `post_per_layer_input_norm`.
        G4PleNorm,
        /// gemma4 `hidden += ple`. The variant the wrong count once ended
        /// at, forty-four kinds early.
        G4PleResidual,
        /// gemma4's fused post-attention norm + residual.
        G4AttnPostResidual,
        /// gemma4's fused post-FFN norm + residual.
        G4FfnPostResidual,
        /// gemma4's fused PLE norm + residual, scaled.
        G4PleResidualScaled,
        /// An untied quantized embedding (`model.embed_tokens`).
        EmbedUntied,
        /// An untied quantized LM head. The kind whose dispatch once ran
        /// the wrong pipeline off the short table.
        LmHeadUntied,
        /// gpt-oss biased q projection.
        GoQmvQ,
        /// gpt-oss biased k projection.
        GoQmvK,
        /// gpt-oss biased v projection.
        GoQmvV,
        /// gpt-oss biased o projection.
        GoQmvO,
        /// gpt-oss decode attention with the learned per-head sink.
        GoSdpaSink,
        /// gpt-oss router (8-bit affine, biased).
        GoRouter,
        /// gpt-oss routed expert gate projection.
        GoExpertGate,
        /// gpt-oss routed expert up projection.
        GoExpertUp,
        /// gpt-oss routed expert down projection.
        GoExpertDown,
        /// Top-k + softmax over the router's logits.
        GoRouterTopK,
        /// gpt-oss's clamped SwiGLU variant.
        GoSwiGlu,
        /// The weighted sum of the k experts' outputs.
        GoExpertCombine,
        /// Qwen-MoE router (`mlp.gate`, no bias).
        LlRouter,
        /// Qwen-MoE stacked expert gate projections.
        LlExpertGate,
        /// Qwen-MoE stacked expert up projections.
        LlExpertUp,
        /// Qwen-MoE stacked expert down projections.
        LlExpertDown,
        /// The batched mixture's expert-major sort.
        LlMoeSort,
        /// The sorted-row gather.
        LlMoeGather,
        /// The sorted-results combine, through the sort's inverse.
        LlMoeCombine,
        /// Shared expert gate projection (`mlp.shared_expert.gate_proj`).
        LlSharedGate,
        /// Shared expert up projection.
        LlSharedUp,
        /// Shared expert down projection.
        LlSharedDown,
        /// `mlp.shared_expert_gate` — hidden to one logit a token.
        LlSharedGateProj,
        /// `routed + sigmoid(gate) * shared`.
        LlSharedCombine,
        /// The mixture's SwiGLU over the sorted stack — split from
        /// [`SiluMul`](Kernel::SiluMul) because a routed layer runs both at
        /// different extents.
        LlExpertSiluMul,
        /// gemma4 MoE router (`router.proj` + per-expert scale).
        G4Router,
        /// gemma4 router norm (`router.scale`, folded at load).
        G4RouterNorm,
        /// gemma4 top-k + softmax + gain.
        G4RouterTopK,
        /// gemma4 `pre_feedforward_layernorm_2`.
        G4MoeNorm,
        /// gemma4 `post_feedforward_layernorm_1`.
        G4DenseBranchNorm,
        /// gemma4 `post_feedforward_layernorm_2`.
        G4MoeBranchNorm,
        /// gemma4 stacked expert gate projections.
        G4ExpertGate,
        /// gemma4 stacked expert up projections.
        G4ExpertUp,
        /// gemma4 stacked expert down projections.
        G4ExpertDown,
        /// GeGLU over the sorted stack — gemma's activation.
        G4ExpertGeglu,
        /// gemma4's expert-major sort.
        G4MoeSort,
        /// gemma4's sorted-row gather.
        G4MoeGather,
        /// gemma4's expert combine.
        G4ExpertCombine,
        /// The dense and mixture branches meeting.
        G4BranchAdd,
    }
}

/// The bucketed command-buffer key.
///
/// Grid dims change with the batch shape, so "byte-identical command
/// buffer" relaxes to "byte-identical within a shape bucket": encoded
/// buffers are cached by this key and reused on a hit. M=1 single-stream is
/// one stable bucket — `{1, 1, bucket 0, pure}` every token — which is what
/// keeps encode(N+1) overlapping GPU(N).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ForwardGraphKey {
    /// Requests in the fire.
    pub requests: u32,
    /// Total tokens across the batch.
    pub tokens: u32,
    /// `max_pages_in_batch`, coarsened by [`PAGE_BUCKET_GRAN`] so the cache
    /// does not thrash on every +1 page.
    pub page_bucket: u32,
    /// Every request contributes exactly one token.
    pub is_pure_decode: bool,
}

/// The page-count bucketing granularity for [`ForwardGraphKey`].
pub const PAGE_BUCKET_GRAN: u32 = 8;

impl ForwardGraphKey {
    /// The key for a fire of `requests`/`tokens` whose largest request
    /// holds `max_pages` pages.
    #[must_use]
    pub const fn of(requests: u32, tokens: u32, max_pages: u32, is_pure_decode: bool) -> Self {
        Self {
            requests,
            tokens,
            page_bucket: max_pages.div_ceil(PAGE_BUCKET_GRAN),
            is_pure_decode,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The count is derived from the variant list itself, so this holds by
    /// construction — the assertion documents the invariant the C++ lost.
    #[test]
    fn the_count_is_the_end_of_the_enum_by_construction() {
        assert_eq!(Kernel::COUNT, Kernel::ALL.len());
        assert_eq!(
            Kernel::ALL.last().map(|k| k.index()),
            Some(Kernel::COUNT - 1),
            "the last kind's discriminant is one below the count"
        );
        // Discriminants are dense and ordered: index i holds the kind whose
        // discriminant is i, which is what makes `[T; COUNT]` tables safe.
        for (position, kind) in Kernel::ALL.iter().enumerate() {
            assert_eq!(kind.index(), position);
        }
    }

    /// The numeric values are ABI, "APPEND ONLY" five times over in the
    /// C++. These anchors pin every block boundary: an insertion upstream
    /// of one moves it and fails here, instead of renumbering forty kinds
    /// silently.
    #[test]
    fn the_abi_anchor_values_hold() {
        assert_eq!(Kernel::EmbedGather.index(), 0);
        assert_eq!(Kernel::Residual.index(), 10);
        assert_eq!(Kernel::QmvO.index(), 22);
        assert_eq!(Kernel::Argmax.index(), 31);
        assert_eq!(Kernel::KvAppendPaged.index(), 32);
        assert_eq!(Kernel::G4AttnPostNorm.index(), 36);
        assert_eq!(
            Kernel::G4PleResidual.index(),
            53,
            "the variant the wrong count once ended at"
        );
        assert_eq!(Kernel::LmHeadUntied.index(), 58);
        assert_eq!(Kernel::LlRouter.index(), 71);
        assert_eq!(Kernel::G4Router.index(), 84);
        assert_eq!(Kernel::G4BranchAdd.index(), 97);
        assert_eq!(Kernel::COUNT, 98);
        // The bug the count fix answers: the short spelling reached 54 of
        // 98, and psos[LmHeadUntied] at 58 indexed past it.
        assert!(Kernel::LmHeadUntied.index() > Kernel::G4PleResidual.index() + 1);

        assert_eq!(IoSlot::TokenId as usize, 0);
        assert_eq!(IoSlot::SampleRows as usize, 18);
        assert_eq!(IO_SLOT_COUNT, 19);
        assert_eq!(Region::KvPagePool as usize, 6);
    }

    #[test]
    fn a_kind_indexed_table_covers_every_kind_by_construction() {
        let mut table = [0u32; Kernel::COUNT];
        for kind in Kernel::ALL {
            table[kind.index()] += 1;
        }
        assert!(table.iter().all(|&hits| hits == 1));
    }

    #[test]
    fn the_graph_key_buckets_pages_and_keeps_m1_stable() {
        // M=1 decode is one stable bucket however long the sequence grows
        // within a granule.
        let a = ForwardGraphKey::of(1, 1, 3, true);
        let b = ForwardGraphKey::of(1, 1, 8, true);
        assert_eq!(a, b, "3 and 8 pages share ceil(n/8) = 1");
        let c = ForwardGraphKey::of(1, 1, 9, true);
        assert_ne!(a, c, "9 pages crosses into bucket 2");
        assert_eq!(ForwardGraphKey::of(1, 1, 0, true).page_bucket, 0);
        // Shape changes re-key.
        assert_ne!(a, ForwardGraphKey::of(2, 2, 3, true));
        assert_ne!(a, ForwardGraphKey::of(1, 1, 3, false));
    }
}
