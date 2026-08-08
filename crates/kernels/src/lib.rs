//! ② KERNEL SIGNATURES — the vocabulary. The rows live with the kernels
//! (`.wiki/tart/dsl.md` ②).
//!
//! `dsl::cuda` has ten wrappers over five attention kernels because
//! `_region` / `_planned` / `_capture` / `_dequant` encode the DISPATCH
//! CONTEXT in the wrapper name. The context is a property of the call site;
//! what belongs to the kernel is its symbol and its contract. A [`KernelSig`]
//! is that contract, once per symbol.
//!
//! Four declarations, each replacing something that is a hand-written runtime
//! rule today:
//!
//! | declaration | replaces |
//! |---|---|
//! | `whole`   | `if c.head_dim_padded \|\| (window_one && c.xqa_decode)` in the model body |
//! | `lacks`   | "a score-wanting program under XQA fails loudly PTIR-side" (a C++ throw) |
//! | `needs`   | the prepare a stated kernel obligates, named nowhere |
//! | `sink`    | `emit_cuda::emit_masked_pages_bracket`'s hardcoded page substitution |
//!
//! `whole` is CHECKED at trace time — which is load time, since a declaration
//! is traced when the model loads. The other three are declared but not yet
//! consumed: `needs`/`sink` are the emitter's knowledge until the launch ABI
//! flattens (migration step 6), and `lacks` needs the deployment's
//! servable-seam set, which is the support-matrix work. Declaring them first
//! is the point — the table is where they land, and it exists.
//!
//! ## Why this is its own crate
//!
//! The rows are in [`kernels-cuda`](../kernels_cuda/index.html) and
//! [`kernels-metal`](../kernels_metal/index.html), one crate per backend,
//! each beside the `.cu`/`.metal` it describes — so a new kernel is one
//! source file and one table row in the same directory and the same diff
//! hunk. Both tables have to be written in the same words, and neither
//! backend owns those words, so they are here.
//!
//! Bare-named for the same reason [`driver`](../driver/index.html) is: it is
//! the shared floor under a `-`-prefixed pair, holding what both members
//! speak rather than anything either one does. Nothing depends on it but the
//! two tables and the compiler that reads them, and it depends on nothing at
//! all — a row must be writable next to its kernel without dragging a
//! dependency graph along.

/// A capability a seam may ask of the kernel covering its rows. Named after
/// the seam vocabulary (`.wiki/tart/dsl.md` ①), because that is what a
/// `lacks` line refuses to serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cap {
    /// The attention scores, published for an `attn.out` observer.
    Scores,
    /// The page-mask sink an `attn.q` tap writes.
    PageMaskSink,
}

/// The host-side plan a kernel's contract obligates: stated so a reader of
/// the model text can see which prepare a launch drags in, rather than
/// reading the driver to find out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prepare {
    /// No host plan.
    None,
    /// The FlashInfer decode plan (per fire, per layer group).
    DecodePlan,
    /// The FlashInfer ragged prefill plan.
    PrefillPlan,
    /// The custom-mask plan (`attn_page_mask`'s consumer).
    CustomPlan,
    /// XQA's fire-wide prepare — R-shaped, so it cannot be built per row
    /// window. This is why `xqa_decode` is also `whole`.
    FireWide,
    /// MLA's plan (`kernels::attn::plan_attention_mla_bf16`), which is its own kind
    /// rather than a FlashInfer plan under another name: it is built from
    /// `kv_lora_rank` and `qk_rope_head_dim` — a latent KV geometry no other
    /// prepare here has a field for — and it is cached in an `MlaPlanCache`
    /// the dispatch borrows, not in the shared attention workspace.
    MlaPlan,
}

/// One point of one instantiation axis, and the text it contributes to a name.
///
/// See [`KernelSig::axes`] for why a row has these at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Axis {
    /// What varies. Prose, for a reader of the table; the matcher never reads
    /// it.
    pub what: &'static str,
    /// The suffixes this axis can contribute, in the order a name spells them.
    /// Exactly one is present in any entrypoint the axis reaches.
    ///
    /// A point MAY be `""`, for an axis whose default specialisation adds no
    /// text — `sdpa_paged_decode<…, 0, false, 32>` is spelled
    /// `sdpa_paged_decode_bfloat16_d_128` and the two others are `…_p32` and
    /// `…_p32_sg8`, off ONE template. Two rules follow and both are checked by
    /// [`KernelSig::covers`]'s ordering rather than asserted:
    ///
    /// * the empty point goes LAST, because matching is first-wins and an
    ///   empty suffix matches everything;
    /// * a longer point goes before a shorter one it ends with (`_p32_sg8`
    ///   before `_p32`), for the same reason.
    pub points: &'static [&'static str],
}

/// One kernel's contract.
pub struct KernelSig {
    /// The dsl-side name (what a model text spells).
    pub name: &'static str,
    /// The C++ launcher symbol the trace records.
    pub symbol: &'static str,
    /// The kernel REFUSES a row split: it may not be stated inside a peel's
    /// regions, because its addressing (a fire-wide prepare, a padded staging
    /// buffer) is not row-offsettable. `model-compiler`'s `OpKind::Peel` is
    /// the op this refuses, and its `check_plan` is what enforces the refusal.
    pub whole: bool,
    /// The host plan its contract obligates.
    pub needs: Prepare,
    /// Capabilities this kernel cannot serve — a seam asking for one of these
    /// over rows this kernel covers is unservable.
    pub lacks: &'static [Cap],
    /// Where a sink-writing seam's output lands, if this kernel accepts one
    /// (`sink pages -> kv.pages`).
    pub sink: Option<&'static str>,
    /// On a union tail layer this dispatch pairs the DEPTH PREFIX plan (and
    /// its dedicated workspace) instead of the fire's own plan.
    ///
    /// This was the `PrefixPlanSwap` half of the retired per-op `DepthRole` —
    /// a word the IR carried on one launch per layer of every depth-declaring
    /// trace, restating a fact about the KERNEL. Migration step 5 moved it
    /// here.
    pub depth_prefix_plan: bool,
    /// The axes `symbol` is instantiated over, if it names a FAMILY of
    /// entrypoints rather than one.
    ///
    /// Empty is the CUDA case and the default: a launcher there is an authored
    /// C++ function, so one row is one symbol and [`sig_in`] matches it whole.
    ///
    /// Metal's are generated. `quantized_qmm_t.metal` holds one template body
    /// and a macro that stamps it over `(group, bits) × (bm, bn)`, so `54` of
    /// its entrypoints are one kernel evaluated at 54 points. Enumerating them
    /// as 54 rows would state the macro's job a second time, by hand, and
    /// `.wiki/kernel-refactor.md` §5's own rule — *would the two share one C++
    /// definition?* — says they are not distinct kernels. So the row is the
    /// base and the axes are declared beside it.
    ///
    /// This is not a Metal-only idea. CUDA writes the same product into
    /// FILENAMES (`attn/flashinfer_hd{64,128,256,512}.cu`,
    /// `attn/xqa_gqa{2,4,8}.cu`) and cannot state it, because each of those is
    /// separately authored. When that changes, the axis is already spelled
    /// here.
    pub axes: &'static [Axis],
}

impl KernelSig {
    /// Does `symbol` name this row at one point of its axes?
    ///
    /// Order matters and is the whole implementation: the axes are declared in
    /// the order a name spells them, so this peels suffixes from the END, one
    /// axis at a time, and what must remain is the base. That refuses
    /// `qmm_t_bfloat16_gs_64_b_4` (a `bm`/`bn` short of a real entrypoint) and
    /// refuses a permuted spelling, both of which a "contains all the points"
    /// test would wave through.
    /// Does `symbol` name this row — as the kernel itself, or at one point of
    /// its axes?
    ///
    /// Both are legitimate and they come from different places. **A model text
    /// states the KERNEL**, because the axis point is a deployment fact: which
    /// affine format a checkpoint is, how wide its heads are. `dsl::metal`
    /// records `affine_qmv_fast` and the driver resolves
    /// `affine_qmv_fast_bfloat16_gs_64_b_4` at load, from `AffineFormat`. **The
    /// driver and the audit name the POINT**, because that is what a pipeline
    /// is built from.
    ///
    /// So the base resolves, and so does every point. What does not resolve is
    /// anything between them: [`Self::covers_point`] peels the axes from the
    /// END in declaration order, so a half-spelled name is refused rather than
    /// rounded to the nearest row.
    pub fn covers(&self, symbol: &str) -> bool {
        self.symbol == symbol || self.covers_point(symbol)
    }

    /// `symbol` is this row at one point of its axes — not the bare base.
    ///
    /// Order is the whole implementation: the axes are declared in the order a
    /// name spells them, so this peels suffixes from the end, one axis at a
    /// time, and what must remain is the base. That refuses
    /// `qmm_t_bfloat16_gs_64_b_4` (a tile short of a real entrypoint) and
    /// refuses a permuted spelling, both of which a "contains all the points"
    /// test would wave through.
    pub fn covers_point(&self, symbol: &str) -> bool {
        if self.axes.is_empty() {
            return false;
        }
        let mut rest = symbol;
        for axis in self.axes.iter().rev() {
            match axis
                .points
                .iter()
                .find(|point| rest.len() > point.len() && rest.ends_with(**point))
            {
                Some(point) => rest = &rest[..rest.len() - point.len()],
                None => return false,
            }
        }
        rest == self.symbol
    }

    /// Every entrypoint this row names: the product of its axes, appended in
    /// declaration order. One element (the symbol itself) when there are none.
    ///
    /// This is the other half of [`KernelSig::covers`], and the reason both
    /// exist: `covers` answers "is this name mine", `entrypoints` answers
    /// "what are all of mine", and `scripts/metal-kernel-audit.py` compares
    /// the second against the shader tree. A row that generates a name no
    /// shader instantiates, or misses one that exists, fails there — which is
    /// the invariant `.wiki/kernel-metal-refactor.md` §6 (1) states.
    pub fn entrypoints(&self) -> Vec<String> {
        let mut out = vec![self.symbol.to_string()];
        for axis in self.axes {
            out = out
                .iter()
                .flat_map(|stem| {
                    axis.points.iter().map(move |point| format!("{stem}{point}"))
                })
                .collect();
        }
        out
    }
}

/// Declare one kernel. The syntax is `.wiki/tart/dsl.md` ②'s, minus the
/// operand shapes: those stay with the emitter until the launch ABI flattens,
/// and stating them twice would be the duplication this redesign exists to
/// remove.
///
/// Exported so the two backend tables can declare rows in the same words. It
/// names [`KernelSig`], [`Prepare`] and [`Cap`] through `$crate`, so a table
/// crate needs no `use` beyond the macro itself.
#[macro_export]
macro_rules! kernel {
    ($name:ident $symbol:literal $(, $key:ident = $value:expr)* $(,)?) => {
        $crate::KernelSig {
            name: stringify!($name),
            symbol: $symbol,
            $($key: $value,)*
            ..$crate::KernelSig {
                name: "",
                symbol: "",
                whole: false,
                needs: $crate::Prepare::None,
                lacks: &[],
                sink: None,
                depth_prefix_plan: false,
                axes: &[],
            }
        }
    };
}

/// The contract for one symbol, in `table`.
///
/// A linear scan: the tables are ~100 and ~90 rows, and the call sites are
/// load-time (a declaration is traced when the model loads), not per-fire.
///
/// Exact matches on the symbol are tried first and across the WHOLE table,
/// before any row is allowed to claim `symbol` as a point of its axes. Without
/// that two-pass order a row could swallow a sibling whose base happens to end
/// in one of its points, and which row won would depend on declaration order.
///
/// CUDA's rows carry no axes, so for them the second pass never fires and this
/// is the same linear scan it always was.
pub fn sig_in(table: &'static [KernelSig], symbol: &str) -> Option<&'static KernelSig> {
    table
        .iter()
        .find(|k| k.symbol == symbol)
        .or_else(|| table.iter().find(|k| k.covers_point(symbol)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const AFFINE: Axis = Axis {
        what: "affine group and width",
        points: &["_gs_32_b_4", "_gs_64_b_4", "_gs_128_b_4",
                  "_gs_32_b_8", "_gs_64_b_8", "_gs_128_b_8"],
    };
    const TILE: Axis = Axis {
        what: "routed GEMM tile",
        points: &["_bm_16_bn_16", "_bm_32_bn_32", "_bm_64_bn_64"],
    };
    const DTYPE: Axis = Axis { what: "activation dtype", points: &["_bfloat16"] };

    static TABLE: &[KernelSig] = &[
        kernel!(qmv "affine_qmv_fast", axes = &[DTYPE, AFFINE]),
        kernel!(qmm_t "affine_qmm_t", axes = &[DTYPE, AFFINE, TILE]),
        // A base that is ALSO a legal entrypoint, next to its dtyped form.
        kernel!(route_sort "moe_route_sort"),
        kernel!(router "router_topk", axes = &[DTYPE]),
    ];

    fn named(symbol: &str) -> Option<&'static str> {
        sig_in(TABLE, symbol).map(|k| k.name)
    }

    #[test]
    fn a_row_covers_every_point_of_its_axes() {
        assert_eq!(named("affine_qmv_fast_bfloat16_gs_64_b_4"), Some("qmv"));
        assert_eq!(named("affine_qmv_fast_bfloat16_gs_128_b_8"), Some("qmv"));
        assert_eq!(
            named("affine_qmm_t_bfloat16_gs_32_b_4_bm_64_bn_64"),
            Some("qmm_t")
        );
    }

    /// The axes are peeled from the END in declaration order, so a name that
    /// stops short of a full point set is NOT covered. This is the case a
    /// "contains all the points" test would wave through, and it is exactly
    /// the shape of the bug the table exists to catch: `decode_psos.cpp`
    /// building `"affine_qmm_t" + q` and forgetting the tile.
    #[test]
    fn a_partial_or_permuted_spelling_is_refused() {
        assert_eq!(named("affine_qmm_t_bfloat16_gs_64_b_4"), None); // no tile
        assert_eq!(named("affine_qmm_t_bm_16_bn_16"), None); // no dtype/affine
        // Right points, wrong order.
        assert_eq!(named("affine_qmm_t_bfloat16_bm_16_bn_16_gs_64_b_4"), None);
        // A point that is not on the axis.
        assert_eq!(named("affine_qmv_fast_bfloat16_gs_16_b_4"), None);
    }

    /// A row whose base is itself an entrypoint keeps it, and does not get
    /// eaten by a sibling that could peel to the same text.
    #[test]
    fn a_row_resolves_by_its_base_and_by_every_point() {
        assert_eq!(named("moe_route_sort"), Some("route_sort"));
        assert_eq!(named("router_topk_bfloat16"), Some("router"));
        // The BASE resolves too, and this is not a convenience: a model text
        // states the kernel, not the instantiation, because the affine format
        // is a checkpoint fact the lowering does not have. `dsl::metal` records
        // `affine_qmv_fast`; the driver resolves the point at load.
        assert_eq!(named("router_topk"), Some("router"));
        assert_eq!(named("affine_qmm_t"), Some("qmm_t"));
    }

    /// CUDA's rows carry no axes, and this is the assertion that the addition
    /// changed nothing for them: an axisless row matches its symbol and
    /// nothing else, prefix or suffix.
    #[test]
    fn an_axisless_row_is_unchanged_by_the_axis_machinery() {
        assert_eq!(named("moe_route_sort_bfloat16"), None);
        assert_eq!(named("moe_route_sor"), None);
        assert_eq!(named("xmoe_route_sort"), None);
    }

    /// The `sdpa_paged_decode` case, and the reason `points` may hold `""`.
    ///
    /// Three macros in `sdpa_paged.metal` stamp ONE template —
    /// `sdpa_paged_decode<itype, d, v, sink, PAGES, FIXED, SG>` — at
    /// `<…, 0, false, 32>`, `<…, 32, true, 32>` and `<…, 32, true, 8>`. Same
    /// body, three points, and the first contributes no text.
    #[test]
    fn an_axis_may_have_a_point_that_adds_no_text() {
        const DIM: Axis = Axis { what: "head dim", points: &["_d_64", "_d_128"] };
        // Longest first, empty last: both orderings are load-bearing.
        const PAGE: Axis = Axis {
            what: "page table width and simdgroup count",
            points: &["_p32_sg8", "_p32", ""],
        };
        static T: &[KernelSig] =
            &[kernel!(sdpa_paged "sdpa_paged_decode", axes = &[DTYPE, DIM, PAGE])];

        for name in [
            "sdpa_paged_decode_bfloat16_d_128",
            "sdpa_paged_decode_bfloat16_d_128_p32",
            "sdpa_paged_decode_bfloat16_d_64_p32_sg8",
        ] {
            assert!(sig_in(T, name).is_some(), "{name}");
        }
        // Still not a licence to match anything.
        assert!(sig_in(T, "sdpa_paged_decode_bfloat16_d_256").is_none());
        assert!(sig_in(T, "sdpa_paged_decode_bfloat16").is_none());
        assert_eq!(T[0].entrypoints().len(), 1 * 2 * 3);
    }

    /// `covers` and `entrypoints` are two directions on one relation, and the
    /// audit script trusts both. Round-trip them.
    #[test]
    fn everything_a_row_generates_is_something_it_covers() {
        for row in TABLE {
            for name in row.entrypoints() {
                assert_eq!(
                    sig_in(TABLE, &name).map(|k| k.name),
                    Some(row.name),
                    "{name}"
                );
            }
        }
    }
}
