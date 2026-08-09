//! GPT-OSS's PSO plan: which entrypoints this family compiles, keyed by
//! the three facts the staged tensors decided.
//!
//! The geometry decides and every choice is refused-not-defaulted, because
//! either wrong answer RUNS: `router_bits` selects the router's matvec (8
//! for the width mlx_lm's predicate usually leaves, 4 for a uniform
//! checkpoint — either kernel over the other's packing is fluent wrong
//! text); `mxfp4_experts` selects which routed matvec reads the bank; and
//! `head_dim` names the attention instantiation — this was once a literal
//! 64 while the geometry read the config, so a variant shipping any other
//! width would have run a d=64 pipeline over its heads, striding past the
//! end of each and writing zeros. Spelled from the geometry, an
//! uninstantiated width fails to build BY NAME at load.

use super::geometry::AffineFormat;
use super::gptoss::GptOssGeometry;

/// The head width the matrix-unit attention is instantiated at.
pub const SDPA_MMA_HEAD_DIM: u32 = 64;

/// Which table slot a compiled gpt-oss pipeline lands in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(missing_docs)] // the slot names ARE the C++ field names
pub enum GptOssSlot {
    QmvTail,
    QmvTailBias,
    QmvRoutedBias,
    QmvRouter,
    RouterTopK,
    MoeSort,
    MoeGather,
    MoeCombine,
    SwiGlu,
    SdpaSink,
    SdpaSinkPaged,
    SdpaSinkPagedTiled,
    SdpaSinkPagedMma,
    RopeFreqs,
    RopeFreqsMb,
    RowGather,
}

/// One entrypoint the plan wants compiled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GptOssPsoRequest {
    /// Where the pipeline goes.
    pub slot: GptOssSlot,
    /// The shader file, relative to the kernels directory.
    pub file: &'static str,
    /// The entrypoint name.
    pub entry: String,
}

fn suffix(bits: u32, group: u32) -> String {
    AffineFormat { bits, group }.kernel_suffix()
}

/// Lay out the compile list for `g`.
///
/// K = 2880 is a whole number of quantization groups but not of any
/// reduction block, so every projection here runs the TAIL-handling
/// matvec. The mixture movers and the top-k are the shared kernels — the
/// shape is a property of `router_topk`, not of gpt-oss.
#[must_use]
pub fn plan_gptoss_psos(g: &GptOssGeometry) -> Vec<GptOssPsoRequest> {
    let mut out = Vec::new();
    let mut want = |slot: GptOssSlot, file: &'static str, entry: String| {
        out.push(GptOssPsoRequest { slot, file, entry });
    };
    let qmv = "quant/qmv.metal";
    want(
        GptOssSlot::QmvTail,
        qmv,
        format!("affine_qmv_tail{}", suffix(g.proj_bits, 64)),
    );
    want(
        GptOssSlot::QmvTailBias,
        qmv,
        format!("affine_qmv_tail_bias{}", suffix(g.proj_bits, 64)),
    );
    // The bank's matvec: the checkpoint's own MXFP4 (block exponents,
    // group 32, no zero point) or the loader's affine U4.
    want(
        GptOssSlot::QmvRoutedBias,
        qmv,
        if g.mxfp4_experts {
            format!("mxfp4_qmv_routed_bias{}", suffix(4, 32))
        } else {
            format!("affine_qmv_routed_bias{}", suffix(4, 64))
        },
    );
    want(
        GptOssSlot::QmvRouter,
        qmv,
        format!("affine_qmv_tail_bias{}", suffix(g.router_bits, 64)),
    );
    want(
        GptOssSlot::RouterTopK,
        "moe/route.metal",
        "router_topk_bfloat16".to_string(),
    );
    want(
        GptOssSlot::MoeSort,
        "moe/route.metal",
        "route_sort".to_string(),
    );
    want(
        GptOssSlot::MoeGather,
        "moe/route.metal",
        "route_gather".to_string(),
    );
    want(
        GptOssSlot::MoeCombine,
        "moe/route.metal",
        "combine_sorted".to_string(),
    );
    want(
        GptOssSlot::SwiGlu,
        "mlp/gated.metal",
        "gptoss_swiglu_bfloat16".to_string(),
    );
    let d = format!("_d_{}", g.head_dim);
    want(
        GptOssSlot::SdpaSink,
        "attn/sdpa_sliding.metal",
        format!("sdpa_vector_decode_sink_bfloat16{d}"),
    );
    want(
        GptOssSlot::SdpaSinkPaged,
        "attn/sdpa_paged.metal",
        format!("sdpa_paged_decode_sink_bfloat16{d}"),
    );
    want(
        GptOssSlot::SdpaSinkPagedTiled,
        "attn/sdpa_paged.metal",
        format!("sdpa_paged_tiled_sink_bfloat16{d}"),
    );
    if g.head_dim == SDPA_MMA_HEAD_DIM {
        want(
            GptOssSlot::SdpaSinkPagedMma,
            "attn/sdpa_paged_mma.metal",
            format!("sdpa_paged_mma_sink_bfloat16{d}"),
        );
    }
    want(
        GptOssSlot::RopeFreqs,
        "rope/neox.metal",
        "neox_freqs_decode_bfloat16".to_string(),
    );
    want(
        GptOssSlot::RopeFreqsMb,
        "rope/neox.metal",
        "neox_freqs_mb_bfloat16".to_string(),
    );
    want(
        GptOssSlot::RowGather,
        "layout/row_gather.metal",
        "row_gather_bfloat16".to_string(),
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_the_solver_can_pick_is_a_shipped_entrypoint() {
        let table: std::collections::HashSet<String> =
            kernels_metal::entrypoints().into_iter().collect();
        let variants = [
            GptOssGeometry::default(), // affine experts, router 8, proj 4
            GptOssGeometry {
                mxfp4_experts: true,
                router_bits: 4,
                proj_bits: 8,
                ..GptOssGeometry::default()
            },
        ];
        for g in variants {
            for request in plan_gptoss_psos(&g) {
                assert!(
                    table.contains(&request.entry),
                    "{} is not in the signature table (slot {:?}, mxfp4 {})",
                    request.entry,
                    request.slot,
                    g.mxfp4_experts
                );
                let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .expect("crates/")
                    .join("kernels-metal/kernels")
                    .join(request.file);
                assert!(path.exists(), "{} does not exist", path.display());
            }
        }
    }

    #[test]
    fn an_unusual_head_width_skips_the_matrix_unit_rather_than_lying() {
        let wide = GptOssGeometry {
            head_dim: 128,
            ..GptOssGeometry::default()
        };
        let plan = plan_gptoss_psos(&wide);
        assert!(
            plan.iter().all(|r| r.slot != GptOssSlot::SdpaSinkPagedMma),
            "the MMA pipeline is instantiated at d=64; other widths fall back"
        );
        // And the named instantiation carries the width, so an
        // uninstantiated one fails BY NAME at load instead of striding
        // past every head.
        let sink = plan
            .iter()
            .find(|r| r.slot == GptOssSlot::SdpaSink)
            .unwrap();
        assert!(sink.entry.ends_with("_d_128"));
    }
}
