//! ② KERNEL SIGNATURES — the compiler's end of the tables.
//!
//! The rows moved out. A kernel's contract belongs beside the kernel, so the
//! CUDA table is [`kernels_cuda::KERNELS`] and the Metal one is
//! [`kernels_metal::KERNELS`], each in the crate that also holds the `.cu` /
//! `.metal` it describes — one source file and one table row, same directory,
//! same diff hunk. The words a row is written in are the `kernels` crate's,
//! which is also where the reasons for `whole` / `needs` / `lacks` / `sink`
//! are written down.
//!
//! What stays here is what is about the COMPILER rather than about a kernel:
//! [`Backend`], which reads a backend out of a traced family name, and
//! [`check_plan`], which walks a [`ForwardPlan`]. Both are re-exported
//! through this module so call sites keep saying `kernels::sig(..)`.
//!
//! The tables are consumed with `default-features = false`, so reading a
//! symbol's contract does not build a single `.cu`.

use crate::trace::{ForwardPlan, OpKind};

// The vocabulary and the two tables, re-exported so this module reads as one
// surface. `Cap` and `Prepare` are named by model texts and by the tests
// below; `KernelSig` is what `trace` and `emit_cuda` hold a reference to.
pub use kernels::{Cap, KernelSig, Prepare};
pub use kernels_cuda::KERNELS;
pub use kernels_metal::KERNELS as KERNELS_METAL;

/// Which backend's kernels a lowered trace states.
///
/// The table is per-BACKEND because a kernel signature is backend-owned
/// (`.wiki/tart/dsl.md` ②). A model text is written for one backend and
/// states that backend's symbols; the family name says which —
/// `llama_like.cuda.decode` is CUDA's, `llama_like.metal.decode` is Metal's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cuda,
    Metal,
}

impl Backend {
    /// The backend a traced family name names, or `None` for a SEMANTIC
    /// trace — which states no kernels at all, so no table applies.
    pub fn of_family(family: &str) -> Option<Backend> {
        let mut parts = family.split('.').skip(1);
        match parts.next() {
            Some("cuda") => Some(Backend::Cuda),
            Some("metal") => Some(Backend::Metal),
            _ => None,
        }
    }

    pub fn table(self) -> &'static [KernelSig] {
        match self {
            Backend::Cuda => KERNELS,
            Backend::Metal => KERNELS_METAL,
        }
    }
}

/// The contract for one recorded symbol, in `backend`'s table.
pub fn sig_in(backend: Backend, symbol: &str) -> Option<&'static KernelSig> {
    kernels::sig_in(backend.table(), symbol)
}

/// The contract for one CUDA symbol.
pub fn sig(symbol: &str) -> Option<&'static KernelSig> {
    sig_in(Backend::Cuda, symbol)
}

/// LOAD-TIME check of a traced form against the kernel table.
///
/// Two rules, both of which are runtime failures today:
///
/// 1. a `whole` kernel may not be stated inside a [`OpKind::Peel`]'s
///    regions — the peel gives each region a row window, and a
///    fire-wide-prepared kernel has no way to honour one;
/// 2. every launched symbol must be declared, so the table cannot rot
///    while the model texts move on.
///
/// Returns the failures rather than panicking, so a caller can name the
/// family it was loading.
pub fn check_plan(plan: &ForwardPlan) -> Vec<String> {
    let mut problems = Vec::new();
    let backend = Backend::of_family(&plan.family);
    // Ops inside a Peel's two regions, as a countdown over the flat op
    // list (regions are consecutive: prefix then tail, right after the
    // op — `OpKind::Peel`'s doc).
    let mut peeled = 0usize;
    for op in &plan.ops {
        let inside_peel = peeled > 0;
        peeled = peeled.saturating_sub(1);
        match &op.kind {
            OpKind::Peel {
                prefix_ops,
                tail_ops,
                ..
            } => {
                peeled = peeled.max(*prefix_ops as usize + *tail_ops as usize);
            }
            OpKind::Launch { kernel, .. } => match backend.and_then(|b| sig_in(b, kernel)) {
                None => problems.push(format!(
                    "{}: launches `{kernel}`, which no {} kernel! signature declares",
                    plan.family,
                    match backend {
                        Some(b) => format!("{b:?}").to_lowercase(),
                        // A semantic trace states no kernels; one that
                        // does has a family name that does not say
                        // whose they are.
                        None => "backend's".to_string(),
                    }
                )),
                Some(k) if k.whole && inside_peel => problems.push(format!(
                    "{}: `{kernel}` is declared `whole` (needs {:?}) but is stated \
                     inside a Peel region, which gives it a row window it cannot honour",
                    plan.family, k.needs
                )),
                Some(_) => {}
            },
            _ => {}
        }
    }
    problems
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facts::{LlamaLikeCudaFacts, LlamaLikeFacts, Qwen35CudaFacts, Qwen35HybridFacts};
    use crate::family;
    use crate::trace::FireClass;

    use crate::trace::{Op, OpKind};

    fn launch(symbol: &str) -> Op {
        Op {
            kind: OpKind::Launch {
                kernel: symbol.to_string(),
                weights: vec![],
                state: None,
            },
            inputs: vec![],
            outputs: vec![],
            layer: Some(0),
        }
    }

    fn plan_of(ops: Vec<Op>) -> ForwardPlan {
        ForwardPlan {
            // A family name that says whose kernels these are — the
            // check resolves the table from it.
            family: "llama_like.cuda.decode".to_string(),
            values: vec![],
            ops,
            depth_window: false,
            seams: vec![],
        }
    }

    /// The rules FIRE — without this, a green `live_traces_satisfy_the_table`
    /// proves nothing but that the walk found no launches.
    #[test]
    fn the_check_is_not_vacuous() {
        // An undeclared symbol.
        let problems = check_plan(&plan_of(vec![launch("launch_something_new")]));
        assert_eq!(problems.len(), 1, "{problems:#?}");
        assert!(problems[0].contains("launch_something_new"));

        // A `whole` kernel given a row window by a Peel.
        let peel = Op {
            kind: OpKind::Peel {
                prefix_ops: 1,
                tail_ops: 1,
                window: crate::trace::PeelWindow::HookFreePrefix,
            },
            inputs: vec![],
            outputs: vec![],
            layer: Some(0),
        };
        let xqa = "launch_attention_xqa_decode_bf16_prepared";
        let problems = check_plan(&plan_of(vec![
            peel,
            launch(xqa),
            launch("dispatch_attention_flashinfer_decode"),
        ]));
        assert_eq!(problems.len(), 1, "{problems:#?}");
        assert!(problems[0].contains("whole"), "{}", problems[0]);

        // The same kernel OUTSIDE a peel is fine — the fire-level
        // statement the model body takes today.
        assert!(check_plan(&plan_of(vec![launch(xqa)])).is_empty());
    }

    /// A family name says whose kernels a text states.
    #[test]
    fn the_backend_is_read_off_the_family() {
        assert_eq!(Backend::of_family("llama_like.cuda.decode"), Some(Backend::Cuda));
        assert_eq!(
            Backend::of_family("qwen3_5_hybrid.cuda.commit_advance"),
            Some(Backend::Cuda)
        );
        assert_eq!(Backend::of_family("llama_like.metal.decode"), Some(Backend::Metal));
        // Semantic traces state no kernels, so no table applies.
        assert_eq!(Backend::of_family("llama_like"), None);
        assert_eq!(Backend::of_family("qwen3_5_moe_mlp_block"), None);
    }

    /// Metal's table is empty, and that REFUSES rather than permits: a
    /// `llama_like.metal.*` text cannot state a kernel it has not
    /// declared. This is the discipline that will fill the table when
    /// the first Metal text is written.
    #[test]
    fn an_empty_backend_table_refuses() {
        let mut p = plan_of(vec![launch("metal_gemm_bf16")]);
        p.family = "llama_like.metal.decode".to_string();
        // (the same symbol under CUDA's table is refused too — this is
        // about WHICH table, not about permissiveness)
        let problems = check_plan(&p);
        assert_eq!(problems.len(), 1, "{problems:#?}");
        assert!(problems[0].contains("metal"), "{}", problems[0]);
    }

    /// The table is exactly the set of symbols `dsl::cuda` can record.
    ///
    /// This is the argument that [`check_plan`]'s coverage rule — which
    /// runs at LOAD and fails the trace — can never fire spuriously on a
    /// live deployment: reachability is a property of the dsl surface,
    /// not of which fact combinations a test happens to exercise. And it
    /// is the guard that makes the table's other three declarations get
    /// written: a new `cuda::` wrapper fails this test until its
    /// contract exists.
    #[test]
    fn the_table_is_exactly_the_dsl_surface() {
        let dsl = include_str!("dsl.rs");
        let mut stated: Vec<&str> = dsl
            .split('"')
            .skip(1)
            .step_by(2)
            .filter(|s| {
                ["launch_", "dispatch_", "ops::", "pie_lora", "qwen35_verify"]
                    .iter()
                    .any(|p| s.starts_with(p))
            })
            .collect();
        stated.sort_unstable();
        stated.dedup();
        let mut declared: Vec<&str> = KERNELS.iter().map(|k| k.symbol).collect();
        declared.sort_unstable();
        assert_eq!(
            stated, declared,
            "the kernel! table and dsl::cuda's stated symbols have drifted"
        );
    }

    /// The retired `DepthRole`'s two facts, DERIVED, on a live
    /// depth-declaring trace: membership is the layer tag, and exactly
    /// one launch per layer swaps to the prefix plan.
    ///
    /// The wire word `ffi::arena` writes is computed from these two, so
    /// this pins the C ABI's `depth_role` byte without the IR carrying
    /// it. (The one-off proof that the derivation reproduced the stored
    /// word was 11,399 ops across all 23 goldens, zero mismatches.)
    #[test]
    fn the_depth_axis_derives_from_the_layer_tag() {
        let facts = LlamaLikeFacts::qwen3_0_6b();
        let plan = family::llama_like_cuda(
            &facts,
            &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
            FireClass::Decode,
        );
        assert!(plan.depth_window, "this deployment declares the axis");

        let windowed = plan.ops.iter().filter(|op| plan.depth_windowed(op)).count();
        let layered = plan.ops.iter().filter(|op| op.layer.is_some()).count();
        assert_eq!(windowed, layered, "every layer-tagged op is on the axis");
        assert!(
            plan.ops
                .iter()
                .all(|op| op.layer.is_some() || !plan.depth_windowed(op)),
            "the prologue/epilogue are outside it"
        );

        // Three planned-decode dispatches per layer take the swap: the
        // mask arm's unmasked-prefix rows, and the plain body's
        // score-capturing and plain arms.
        let swaps = plan.ops.iter().filter(|op| plan.depth_prefix_plan(op)).count();
        assert_eq!(swaps, 3 * facts.layers as usize);
        assert!(
            plan.ops.iter().filter(|op| plan.depth_prefix_plan(op)).all(|op| matches!(
                &op.kind,
                OpKind::Launch { kernel, .. }
                    if kernel == "dispatch_attention_flashinfer_decode"
            )),
            "only the planned decode dispatch swaps"
        );

        // PREFILL declares the axis too (the cutover's last decline
        // class was a truncated prefill), and its layer-tagged ops are
        // on it — but NOTHING there takes the prefix-plan swap, because
        // that is a property of the planned DECODE dispatch and a
        // prefill fire does not run one. Which is the whole difference
        // between the two halves of the axis: stopping after layer `k`
        // costs a prefill nothing, and narrowing rows under it would
        // cost it a plan it has no way to build.
        let prefill = family::llama_like_cuda(
            &facts,
            &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
            FireClass::Prefill,
        );
        assert!(prefill.depth_window);
        assert!(prefill
            .ops
            .iter()
            .any(|op| prefill.depth_windowed(op)));
        assert_eq!(
            prefill.ops.iter().filter(|op| prefill.depth_prefix_plan(op)).count(),
            0,
            "a prefill fire runs no planned decode dispatch, so nothing swaps"
        );

        // A PADDED-HEAD deployment declares the axis too. It cannot serve
        // the narrowing half — its staging offsets are physical width
        // while a row window's are logical — but stopping after layer `k`
        // addresses nothing, and `k` is a runtime input the trace does
        // not have. So the trace states the axis and the DRIVER refuses
        // the shapes that narrow (`PaddedHeadNarrowing`), which is the
        // same division of labour the Prefill class settled.
        let padded = family::llama_like_cuda(
            &facts,
            &LlamaLikeCudaFacts {
                head_dim_padded: true,
                ..LlamaLikeCudaFacts::qwen3_0_6b_l40s()
            },
            FireClass::Prefill,
        );
        assert!(padded.depth_window);

        // The XQA decode deployment is the one that still withholds it:
        // its prepare is fire-wide and R-shaped, so even the free half
        // has nothing to stand on.
        let xqa = family::llama_like_cuda(
            &facts,
            &LlamaLikeCudaFacts {
                xqa_decode: true,
                ..LlamaLikeCudaFacts::qwen3_0_6b_l40s()
            },
            FireClass::Decode,
        );
        assert!(!xqa.depth_window);
        assert!(xqa.ops.iter().all(|op| !xqa.depth_windowed(op)));
    }

    /// No symbol is declared twice, and no dsl-side name is either.
    #[test]
    fn table_is_unambiguous() {
        for (i, k) in KERNELS.iter().enumerate() {
            for other in &KERNELS[i + 1..] {
                assert_ne!(k.symbol, other.symbol, "symbol declared twice");
                assert_ne!(k.name, other.name, "name declared twice");
            }
        }
    }

    /// Every kernel every live deployment states is declared, and no
    /// live trace puts a `whole` kernel under a row split. This is the
    /// check running against real traces — the table is not decorative.
    #[test]
    fn live_traces_satisfy_the_table() {
        let mut plans = Vec::new();
        for class in [FireClass::Decode, FireClass::Prefill] {
            plans.push(family::llama_like_cuda(
                &LlamaLikeFacts::qwen3_0_6b(),
                &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
                class,
            ));
            plans.push(family::llama_like_cuda(
                &LlamaLikeFacts::mistral_7b_v03(),
                &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
                class,
            ));
        }
        for class in [
            FireClass::Decode,
            FireClass::Prefill,
            FireClass::StateOnly,
            FireClass::CommitAdvance,
            FireClass::FrozenVerify,
        ] {
            plans.push(family::qwen3_5_hybrid_cuda(
                &Qwen35HybridFacts::qwen3_5_0_8b(),
                &Qwen35CudaFacts::qwen3_5_0_8b_synthetic(),
                class,
            ));
            // Qwen3.6-27B: the same text at a different geometry, and
            // the first one whose GDN half is GQA.
            plans.push(family::qwen3_5_hybrid_cuda(
                &Qwen35HybridFacts::qwen3_6_27b(),
                &Qwen35CudaFacts::qwen3_5_0_8b_synthetic(),
                class,
            ));
        }
        // gemma-4: every symbol its decode reading states has a
        // contract here.
        plans.push(family::gemma4_cuda(
            &crate::facts::Gemma4Facts::gemma_4_e4b(),
            &crate::facts::Gemma4CudaFacts::gemma_4_e4b_synthetic(),
            FireClass::Decode,
        ));
        for plan in &plans {
            let problems = check_plan(plan);
            assert!(problems.is_empty(), "{problems:#?}");
        }
    }
}
