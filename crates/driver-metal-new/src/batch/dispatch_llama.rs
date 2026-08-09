//! The llama families' per-token dispatch DAG, in the shared [`Kernel`]
//! vocabulary.
//!
//! A pure function, exactly like the qwen and gpt-oss builders. Almost
//! every kind here IS an existing kind — this family's whole point is
//! that the driver already had the pieces — but the DAG cannot be the
//! shared builder under a flag: that walk is qwen3.5's, whose q
//! projection is the 2×-wide `[query | gate]` that `QSplit` then halves
//! and `AttnGate` consumes. llama's q is plain, its gate nonexistent,
//! and its QK-norm OPTIONAL where qwen's is structural. A builder that
//! tried to be both behind options would carry every difference as a
//! branch the other family must dodge; two builders each state one
//! shape, over one set of launch helpers that cannot drift.
//!
//! What is deliberately absent at M=1: the C++ `RowGather` — the same
//! argument as gpt-oss's builder — and any fused residual, which the
//! C++ llama path never had.

use crate::tuning::Tuning;

use super::abi::Kernel;
use super::dispatch::{
    Dispatch, Launch, embed, kv_append, qmv, residual, rms, rope, route_rows, route_sort,
    routed_qmv, router_topk, sdpa, silu_mul,
};
use super::llama::LlamaGeometry;
use super::sizing::sorted_rows;

/// Emit the ordered per-token DAG for `g`.
///
/// Within a stage the order clusters independent dispatches (q/k/v
/// together, then the norms, then the ropes) — hazard-neutral, and what
/// lets a concurrency group form.
#[must_use]
pub fn build_llama_dag(g: &LlamaGeometry, tuning: &Tuning, with_argmax: bool) -> Vec<Dispatch> {
    let mut dag: Vec<Dispatch> = Vec::with_capacity(g.n_layers as usize * 21 + 4);
    let emit = |dag: &mut Vec<Dispatch>, kind: Kernel, layer: Option<u32>, launch: Launch| {
        let ordinal = u32::try_from(dag.len()).expect("a DAG is hundreds of dispatches");
        dag.push(Dispatch {
            kind,
            ordinal,
            layer,
            launch,
            fuse_residual: false,
            qmm_bn: 0,
            qmm_split: 1,
            qmm_bm: 16,
        });
    };
    // The sort runs at M=1 too, where the tile the tuning answers for
    // one row's pairs is 1 — a grouping with no padding. The same
    // deliberate choice the shared builder records: one routed dataflow
    // shape, not a decode shape and a prefill shape kept agreeing.
    let tile = tuning.moe_tile_rows(g.experts_per_token, g.n_experts);
    let sorted = u32::try_from(sorted_rows(g.experts_per_token, g.n_experts, tile))
        .expect("an M=1 sort is small");

    // Tied checkpoints read `shared_embedding`, untied ones their own
    // `embed_tokens` — a kind is a weight name, so they are two kinds.
    emit(
        &mut dag,
        if g.tied_embeddings {
            Kernel::EmbedGather
        } else {
            Kernel::EmbedUntied
        },
        None,
        embed(g.hidden),
    );

    for layer in 0..g.n_layers {
        let at = Some(layer);
        emit(&mut dag, Kernel::Rms, at, rms(g.hidden, 1));
        emit(&mut dag, Kernel::QmvQ, at, qmv(g.q_width()));
        emit(&mut dag, Kernel::QmvK, at, qmv(g.kv_width()));
        emit(&mut dag, Kernel::QmvV, at, qmv(g.kv_width()));
        // Qwen3 only: per-head RMS over head_dim, before the rotation.
        // Not emitting the pair on a checkpoint that has one would be a
        // wrong model that still produces fluent text — but that is the
        // geometry's refusal to make, not this builder's: `qk_norm` came
        // from whether the checkpoint ships `self_attn.q_norm`.
        if g.qk_norm {
            emit(&mut dag, Kernel::QNorm, at, rms(g.head_dim, g.n_q_heads));
            emit(&mut dag, Kernel::KNorm, at, rms(g.head_dim, g.n_kv_heads));
        }
        emit(
            &mut dag,
            Kernel::Rope,
            at,
            rope(g.rotary_dims(), g.n_q_heads),
        );
        emit(
            &mut dag,
            Kernel::RopeK,
            at,
            rope(g.rotary_dims(), g.n_kv_heads),
        );
        emit(
            &mut dag,
            Kernel::KvAppend,
            at,
            kv_append(g.head_dim, g.n_kv_heads),
        );
        emit(&mut dag, Kernel::Sdpa, at, sdpa(g.n_q_heads));
        emit(&mut dag, Kernel::QmvO, at, qmv(g.hidden));
        emit(&mut dag, Kernel::Residual, at, residual(g.hidden));

        emit(&mut dag, Kernel::FfnRms, at, rms(g.hidden, 1));
        if g.is_moe() {
            // The same nine dispatches the shared builder emits for
            // qwen3.6's mixture, minus the shared expert this family
            // does not have. `mlp.gate` is `[n_experts, hidden]` — the
            // same shape as a narrow attention projection, so the router
            // is an ordinary matvec rather than a new kernel.
            emit(&mut dag, Kernel::LlRouter, at, qmv(g.n_experts));
            emit(&mut dag, Kernel::GoRouterTopK, at, router_topk(g.n_experts));
            emit(&mut dag, Kernel::LlMoeSort, at, route_sort(g.n_experts));
            emit(
                &mut dag,
                Kernel::LlMoeGather,
                at,
                route_rows(g.hidden, sorted),
            );
            emit(
                &mut dag,
                Kernel::LlExpertGate,
                at,
                routed_qmv(g.moe_intermediate, 1, sorted),
            );
            emit(
                &mut dag,
                Kernel::LlExpertUp,
                at,
                routed_qmv(g.moe_intermediate, 1, sorted),
            );
            emit(
                &mut dag,
                Kernel::LlExpertSiluMul,
                at,
                route_rows(g.moe_intermediate, sorted),
            );
            emit(
                &mut dag,
                Kernel::LlExpertDown,
                at,
                routed_qmv(g.hidden, 1, sorted),
            );
            emit(&mut dag, Kernel::LlMoeCombine, at, route_rows(g.hidden, 1));
        } else {
            emit(&mut dag, Kernel::QmvGate, at, qmv(g.intermediate));
            emit(&mut dag, Kernel::QmvUp, at, qmv(g.intermediate));
            emit(&mut dag, Kernel::SiluMul, at, silu_mul(g.intermediate));
            emit(&mut dag, Kernel::QmvDown, at, qmv(g.hidden));
        }
        emit(&mut dag, Kernel::LayerOut, at, residual(g.hidden));
    }

    emit(&mut dag, Kernel::FinalRms, None, rms(g.hidden, 1));
    emit(
        &mut dag,
        if g.tied_embeddings {
            Kernel::QmvLmHead
        } else {
            Kernel::LmHeadUntied
        },
        None,
        qmv(g.vocab),
    );
    if with_argmax {
        emit(
            &mut dag,
            Kernel::Argmax,
            None,
            Launch {
                grid: [1024, 1, 1],
                tg: [1024, 1, 1],
            },
        );
    }
    dag
}

/// What the step costs, countable with no GPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LlamaDagStats {
    /// Every dispatch.
    pub total: usize,
    /// Decoder layers.
    pub layers: u32,
    /// The matvecs — the step's bandwidth, and what any performance
    /// claim about this family is really about.
    pub gemv: usize,
    /// Of those, the ones whose weights the router picks.
    pub routed: usize,
}

/// Count the DAG's costs.
#[must_use]
pub fn llama_dag_stats(dag: &[Dispatch], g: &LlamaGeometry) -> LlamaDagStats {
    let mut s = LlamaDagStats {
        total: dag.len(),
        layers: g.n_layers,
        ..LlamaDagStats::default()
    };
    for d in dag {
        match d.kind {
            Kernel::QmvQ
            | Kernel::QmvK
            | Kernel::QmvV
            | Kernel::QmvO
            | Kernel::QmvGate
            | Kernel::QmvUp
            | Kernel::QmvDown
            | Kernel::LlRouter
            | Kernel::QmvLmHead
            | Kernel::LmHeadUntied => s.gemv += 1,
            Kernel::LlExpertGate | Kernel::LlExpertUp | Kernel::LlExpertDown => {
                s.gemv += 1;
                s.routed += 1;
            }
            _ => {}
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::build_scratch_schedule;

    #[test]
    fn the_8b_step_is_sixteen_dispatches_a_layer_plus_the_ends() {
        let g = LlamaGeometry::default();
        let dag = build_llama_dag(&g, &Tuning::default(), true);
        // 1 embed + 32 × 16 + final norm + head + argmax.
        assert_eq!(dag.len(), 1 + 32 * 16 + 3);
        assert!(dag.iter().enumerate().all(|(i, d)| d.ordinal as usize == i));
        // llama has no q|gate fusion and no attention gate — the qwen
        // kinds must not leak in.
        assert!(dag.iter().all(|d| d.kind != Kernel::QSplit
            && d.kind != Kernel::AttnGate
            && d.kind != Kernel::QNorm));
        // Untied: the 8B ships both matrices.
        assert!(dag.iter().any(|d| d.kind == Kernel::LmHeadUntied));
        assert!(dag.iter().any(|d| d.kind == Kernel::EmbedUntied));
        let q = dag.iter().find(|d| d.kind == Kernel::QmvQ).unwrap();
        assert_eq!(
            q.launch.tg,
            [32, 2, 1],
            "the plain q projection is a matvec at its own width"
        );
        let stats = llama_dag_stats(&dag, &g);
        assert_eq!(stats.gemv, 32 * 7 + 1, "seven a layer plus the head");
        assert_eq!(stats.routed, 0);
        build_scratch_schedule(&dag, false).expect("the dense DAG colours");
    }

    #[test]
    fn qwen3_gets_the_norm_pair_between_v_and_the_rotation() {
        let g = LlamaGeometry {
            qk_norm: true,
            tied_embeddings: true,
            ..LlamaGeometry::default()
        };
        let dag = build_llama_dag(&g, &Tuning::default(), false);
        assert_eq!(dag.len(), 1 + 32 * 18 + 2);
        let v = dag.iter().position(|d| d.kind == Kernel::QmvV).unwrap();
        assert_eq!(dag[v + 1].kind, Kernel::QNorm);
        assert_eq!(dag[v + 2].kind, Kernel::KNorm);
        assert_eq!(dag[v + 3].kind, Kernel::Rope);
        // Tied: one table serves both ends.
        assert!(dag.iter().any(|d| d.kind == Kernel::EmbedGather));
        assert!(dag.iter().any(|d| d.kind == Kernel::QmvLmHead));
        build_scratch_schedule(&dag, false).expect("the qk-norm DAG colours");
    }

    #[test]
    fn the_mixture_swaps_four_dense_dispatches_for_the_nine_routed_ones() {
        let g = LlamaGeometry {
            qk_norm: true,
            n_experts: 128,
            experts_per_token: 8,
            moe_intermediate: 768,
            ..LlamaGeometry::default()
        };
        let dag = build_llama_dag(&g, &Tuning::default(), true);
        assert_eq!(dag.len(), 1 + 32 * (18 - 4 + 9) + 3);
        let stats = llama_dag_stats(&dag, &g);
        assert_eq!(stats.routed, 32 * 3);
        assert_eq!(stats.gemv, 32 * 8 + 1, "q k v o, router, three routed");
        // Eight sorted rows at decode: top-8, tile 1 — a grouping with
        // no padding.
        let gate = dag.iter().find(|d| d.kind == Kernel::LlExpertGate).unwrap();
        assert_eq!(gate.launch.grid[0], 32 * 8);
        // No shared expert in this family.
        assert!(dag.iter().all(|d| d.kind != Kernel::LlSharedGate));
        build_scratch_schedule(&dag, false).expect("the routed DAG colours");
    }
}
