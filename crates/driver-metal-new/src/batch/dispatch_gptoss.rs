//! GPT-OSS's per-token dispatch DAG, in the shared [`Kernel`] vocabulary.
//!
//! A pure function, exactly like the qwen builder: it emits the ordered
//! dispatch list for one token and touches no Metal, so the step's shape —
//! dispatch count, the sliding/full split, what an MoE layer costs — is
//! checked with no GPU and no checkpoint.
//!
//! The C++ kept this family's kinds in their own namespace until the
//! family could be bound; the shared enum has since absorbed them (the
//! `Go*` kinds carry the family's weight names in `weight_binds`), so this
//! port builds directly in the shared vocabulary and reuses the whole M=1
//! machinery. The mixture's movers are the SHARED sort and gather — the
//! same kernels every routed family runs — and the residuals are the
//! shared adds.
//!
//! What is deliberately absent at M=1: the C++ `RowGather`, which compacts
//! the rows a fire will SAMPLE so the tail runs on that prefix. At one
//! token the compaction is the identity, and emitting an identity mover
//! would cost a dispatch and a scratch colour for nothing; it returns with
//! the prefill, where it is most of the fire's savings.

use super::abi::Kernel;
use super::dispatch::{
    Dispatch, Launch, kv_append, qmv, residual, rms, rope, route_rows, route_sort, router_topk,
    sdpa,
};
use super::gptoss::GptOssGeometry;
use super::sizing::sorted_rows;

fn elementwise(width: u32) -> Launch {
    Launch {
        grid: [width, 1, 1],
        tg: [256, 1, 1],
    }
}

/// Emit the ordered per-token DAG for `g`.
///
/// Within a stage the order clusters independent dispatches (q/k/v
/// together, then the ropes) — hazard-neutral, and what lets a concurrency
/// group form.
#[must_use]
pub fn build_gptoss_dag(g: &GptOssGeometry, with_argmax: bool) -> Vec<Dispatch> {
    let mut dag: Vec<Dispatch> = Vec::with_capacity(g.n_layers as usize * 21 + 4);
    let mut emit = |dag: &mut Vec<Dispatch>, kind: Kernel, layer: Option<u32>, launch: Launch| {
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
    // At decode the sort is a pure grouping: tile 1, one row per
    // (token, slot) pair.
    let sorted = u32::try_from(sorted_rows(g.experts_per_token, g.n_experts, 1))
        .expect("a decode sort is small");

    emit(
        &mut dag,
        Kernel::EmbedUntied,
        None,
        Launch {
            grid: [g.hidden, 1, 1],
            tg: [256, 1, 1],
        },
    );

    for layer in 0..g.n_layers {
        let at = Some(layer);
        emit(&mut dag, Kernel::Rms, at, rms(g.hidden, 1));
        emit(&mut dag, Kernel::GoQmvQ, at, qmv(g.q_dim()));
        emit(&mut dag, Kernel::GoQmvK, at, qmv(g.kv_dim()));
        emit(&mut dag, Kernel::GoQmvV, at, qmv(g.kv_dim()));
        // Full rotary: every head dim rotates (no partial factor here).
        emit(&mut dag, Kernel::Rope, at, rope(g.head_dim, g.n_q_heads));
        emit(&mut dag, Kernel::RopeK, at, rope(g.head_dim, g.n_kv_heads));
        emit(
            &mut dag,
            Kernel::KvAppend,
            at,
            kv_append(g.head_dim, g.n_kv_heads),
        );
        // The sink attention: same launch as the plain decode SDPA — the
        // sink is one more denominator term, not one more thread.
        emit(&mut dag, Kernel::GoSdpaSink, at, sdpa(g.n_q_heads));
        emit(&mut dag, Kernel::GoQmvO, at, qmv(g.hidden));
        emit(&mut dag, Kernel::Residual, at, residual(g.hidden));

        emit(&mut dag, Kernel::FfnRms, at, rms(g.hidden, 1));
        emit(&mut dag, Kernel::GoRouter, at, qmv(g.n_experts));
        emit(&mut dag, Kernel::GoRouterTopK, at, router_topk(g.n_experts));
        emit(&mut dag, Kernel::LlMoeSort, at, route_sort(g.n_experts));
        emit(
            &mut dag,
            Kernel::LlMoeGather,
            at,
            route_rows(g.hidden, sorted),
        );
        emit(&mut dag, Kernel::GoExpertGate, at, {
            let mut launch = qmv(g.intermediate);
            launch.grid[0] *= sorted;
            launch
        });
        emit(&mut dag, Kernel::GoExpertUp, at, {
            let mut launch = qmv(g.intermediate);
            launch.grid[0] *= sorted;
            launch
        });
        // The clamped SwiGLU over the sorted stack: gate*sigmoid(alpha*gate)
        // * (up + 1), both operands clamped — the +1 and the clamp are why
        // this cannot reuse silu_mul.
        emit(
            &mut dag,
            Kernel::GoSwiGlu,
            at,
            elementwise(g.intermediate * sorted),
        );
        emit(&mut dag, Kernel::GoExpertDown, at, {
            let mut launch = qmv(g.hidden);
            launch.grid[0] *= sorted;
            launch
        });
        emit(&mut dag, Kernel::GoExpertCombine, at, {
            let w = g.hidden.max(1);
            Launch {
                grid: [w, 1, 1],
                tg: [w.min(256), 1, 1],
            }
        });
        emit(&mut dag, Kernel::LayerOut, at, residual(g.hidden));
    }

    emit(&mut dag, Kernel::FinalRms, None, rms(g.hidden, 1));
    emit(&mut dag, Kernel::LmHeadUntied, None, qmv(g.vocab));
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
pub struct GptOssDagStats {
    /// Every dispatch.
    pub total: usize,
    /// Layers attending the full context.
    pub full_attn_layers: u32,
    /// Layers attending the sliding window.
    pub sliding_attn_layers: u32,
    /// The matvecs — the step's bandwidth.
    pub gemv: usize,
    /// Of those, the ones whose weights the router picks.
    pub routed: usize,
}

/// Count the DAG's costs.
#[must_use]
pub fn gptoss_dag_stats(dag: &[Dispatch], g: &GptOssGeometry) -> GptOssDagStats {
    let mut s = GptOssDagStats {
        total: dag.len(),
        ..GptOssDagStats::default()
    };
    for layer in 0..g.n_layers {
        if g.is_full_attn(layer) {
            s.full_attn_layers += 1;
        } else {
            s.sliding_attn_layers += 1;
        }
    }
    for d in dag {
        match d.kind {
            Kernel::GoQmvQ
            | Kernel::GoQmvK
            | Kernel::GoQmvV
            | Kernel::GoQmvO
            | Kernel::GoRouter
            | Kernel::LmHeadUntied => s.gemv += 1,
            Kernel::GoExpertGate | Kernel::GoExpertUp | Kernel::GoExpertDown => {
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

    #[test]
    fn the_20b_step_is_twenty_one_dispatches_a_layer_plus_the_ends() {
        let g = GptOssGeometry::default();
        let dag = build_gptoss_dag(&g, true);
        // 1 embed + 24 x 21 + final norm + head + argmax.
        assert_eq!(dag.len(), 1 + 24 * 21 + 3);
        assert!(dag.iter().enumerate().all(|(i, d)| d.ordinal as usize == i));
        let stats = gptoss_dag_stats(&dag, &g);
        assert_eq!(stats.full_attn_layers, 12);
        assert_eq!(stats.sliding_attn_layers, 12);
        assert_eq!(
            stats.gemv,
            24 * 8 + 1,
            "seven per layer plus the router, plus the head"
        );
        assert_eq!(stats.routed, 24 * 3);
    }

    #[test]
    fn the_routed_projections_launch_over_the_sorted_stack() {
        let g = GptOssGeometry::default();
        let dag = build_gptoss_dag(&g, false);
        let gate = dag
            .iter()
            .find(|d| d.kind == Kernel::GoExpertGate)
            .expect("every layer routes");
        // Four sorted rows at decode (top-4, tile 1).
        assert_eq!(gate.launch.grid[0], 32 * 4);
        let plain = dag.iter().find(|d| d.kind == Kernel::GoQmvQ).unwrap();
        assert_eq!(plain.launch.grid[0], 32, "the dense matvec stays one row");
    }
}
