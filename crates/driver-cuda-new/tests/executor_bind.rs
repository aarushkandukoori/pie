//! The binder over a REAL lowering (retirement plan phase C, first brick).
//!
//! Not a synthetic launch list: `qwen3_0_6b`'s traced decode and prefill
//! forms — the parity-anchored deployment, the same texts the committed
//! `.inc`s are emitted from — lowered over plain rows, then EVERY launch
//! bound through `executor::bind`. What that proves, GPU-free:
//!
//! * every arena offset the lowering assigns is inside the arena it
//!   sized (`arena_bytes` and the offsets agree with each other);
//! * every weight and named value the trace states reaches the resolver
//!   (the map is the only per-family piece left, as designed);
//! * every kernel symbol the lowering emits has a STATED row in the
//!   bridge's tables — DSL or driver-internal — so a generated
//!   `pie_k_*` entry exists for the dispatch half to call. This is the
//!   claim that phase C's dispatch can be written at all.

#![cfg(feature = "_cuda")]

use std::collections::BTreeSet;
use std::ffi::c_void;

use driver_cuda_new::model::executor::{BindRefusal, Frame, Resolver, bind};
use model::families::llama_like::forward::facts::{LlamaLikeCudaFacts, LlamaLikeFacts};
use model::families::llama_like::forward::llama_like_cuda;
use model::qwen_3_5::forward::facts::{Qwen35CudaFacts, Qwen35HybridFacts};
use model::qwen_3_5::forward::qwen3_5_hybrid_cuda;
use model_compiler::lower::{Fire, Lowered, Row, lower};
use model_compiler::trace::{FireClass, ValueId};

/// Answers every name with a distinct sentinel and records what was asked.
#[derive(Default)]
struct Sentinels {
    weights: BTreeSet<String>,
    named: BTreeSet<ValueId>,
}

impl Resolver for Sentinels {
    fn weight(&mut self, name: &str) -> Option<*const c_void> {
        self.weights.insert(name.to_string());
        Some(0x1000 as *const c_void)
    }
    fn named(&mut self, value: ValueId) -> Option<*mut c_void> {
        self.named.insert(value);
        Some(0x2000 as *mut c_void)
    }
}

fn plan_of(class: FireClass) -> model_compiler::trace::ForwardPlan {
    llama_like_cuda(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
        class,
    )
}

fn lowered(class: FireClass, rows: usize) -> Lowered {
    let plan = llama_like_cuda(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
        class,
    );
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; rows];
    lower(&plan, &rows, Fire { captures_across_splits: false }).expect("the live form lowers")
}

/// The qwen3_5 hybrid (E-gate family #1): `Qwen3.5-0.8B-Base`'s facts
/// with the LIVE L40S cuda set — the `emissions.rs` values, not the
/// synthetic fixture (warp-tiled and cached prefill env-gated off,
/// prefill_decode on, dense MLP so the MoE fields are the no-fused-leg
/// zeros).
fn qwen35_live_cuda() -> Qwen35CudaFacts {
    Qwen35CudaFacts {
        state_bf16: true,
        warp_tiled: false,
        warp_tiled_max: 64,
        cached_max: 0,
        verify_stash: true,
        prefill_decode: true,
        moe_cutlass_max_rows: 0,
        moe_residual_fold: false,
        moe_shared_gate_dot: false,
        moe_streamed_experts: false,
        moe_force_general: false,
        gate_up_fused: true,
    }
}

fn qwen35_lowered(class: FireClass, rows: usize) -> Lowered {
    let plan =
        qwen3_5_hybrid_cuda(&Qwen35HybridFacts::qwen3_5_0_8b(), &qwen35_live_cuda(), class);
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; rows];
    lower(&plan, &rows, Fire { captures_across_splits: false }).expect("the hybrid lowers")
}

/// gemma-2 (E-gate family #2): the 9b facts, DECODE class — the only
/// class the family states today.
fn gemma2_lowered(rows: usize) -> Lowered {
    let plan = model::gemma_2::forward::gemma2_cuda(
        &model::gemma_2::forward::facts::Gemma2Facts::gemma_2_9b(),
        FireClass::Decode,
    );
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; rows];
    lower(&plan, &rows, Fire { captures_across_splits: false }).expect("gemma2 lowers")
}

#[test]
fn every_launch_of_the_gemma2_deployment_binds() {
    let l = gemma2_lowered(4);
    assert!(!l.launches.is_empty());
    let frame = Frame { arena: 0x10000 as *mut c_void, arena_bytes: l.arena_bytes };
    let mut resolver = Sentinels::default();
    for launch in &l.launches {
        let bound = bind(&l, launch, frame, &mut resolver)
            .unwrap_or_else(|r| panic!("gemma2: launch refused: {r:?}"));
        assert_eq!(bound.args.len(), (launch.args.end - launch.args.start) as usize);
    }
    // gemma2's ARG-level weights are all `scale.*` constants (which bind
    // without the resolver); the tensor weights ride the op join.
    let plan = model::gemma_2::forward::gemma2_cuda(
        &model::gemma_2::forward::facts::Gemma2Facts::gemma_2_9b(),
        FireClass::Decode,
    );
    let dp = driver_cuda_new::model::executor::DispatchPlan::new(&plan, &l);
    assert!(
        (0..l.launches.len()).any(|i| {
            dp.spec(i).weight.as_deref().is_some_and(|w| !w.starts_with("scale."))
        }),
        "a forward that names no tensor weights did not lower the model"
    );
}

#[test]
fn every_lowered_gemma2_kernel_has_a_bridge_row() {
    let bridged = bridged_symbols();
    let mut unreachable = BTreeSet::new();
    for symbol in &gemma2_lowered(4).kernels {
        if !bridged.contains(symbol.as_str()) {
            unreachable.insert(symbol.clone());
        }
    }
    assert!(
        unreachable.is_empty(),
        "gemma2 kernels with no stated bridge row: {unreachable:?}"
    );
}

#[test]
#[ignore = "enumeration aid, not a claim"]
fn print_the_gemma2_vocabulary() {
    let l = gemma2_lowered(4);
    eprintln!("=== gemma2 decode: {} launches, arena {}", l.launches.len(), l.arena_bytes);
    for (i, k) in l.kernels.iter().enumerate() {
        let n = l.launches.iter().filter(|x| x.kernel as usize == i).count();
        eprintln!("  {k}  x{n}");
    }
    for launch in l.launches.iter().take(30) {
        let args = &l.args[launch.args.start as usize..launch.args.end as usize];
        eprintln!("  L {} rows={:?} args={args:?}", l.kernels[launch.kernel as usize], launch.rows);
    }
}

/// gemma-4 (the gemma anchor WITH a cached checkpoint — E2B; gemma-2's
/// 2b-it is gated upstream): both stated classes, the synthetic cuda set.
fn gemma4_lowered(class: FireClass, rows: usize) -> Lowered {
    let plan = model::gemma_4::forward::gemma4_cuda(
        &model::gemma_4::forward::facts::Gemma4Facts::gemma_4_e2b(),
        &model::gemma_4::forward::facts::Gemma4CudaFacts::gemma_4_e4b_synthetic(),
        class,
    );
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; rows];
    lower(&plan, &rows, Fire { captures_across_splits: false }).expect("gemma4 lowers")
}

#[test]
fn every_launch_of_the_gemma4_deployment_binds() {
    for (class, rows) in [(FireClass::Decode, 4), (FireClass::Prefill, 7)] {
        let l = gemma4_lowered(class, rows);
        assert!(!l.launches.is_empty());
        let frame = Frame { arena: 0x10000 as *mut c_void, arena_bytes: l.arena_bytes };
        let mut resolver = Sentinels::default();
        for launch in &l.launches {
            let bound = bind(&l, launch, frame, &mut resolver)
                .unwrap_or_else(|r| panic!("gemma4 {class:?}: launch refused: {r:?}"));
            assert_eq!(bound.args.len(), (launch.args.end - launch.args.start) as usize);
        }
    }
}

#[test]
fn every_lowered_gemma4_kernel_has_a_bridge_row() {
    let bridged = bridged_symbols();
    let mut unreachable = BTreeSet::new();
    for (class, rows) in [(FireClass::Decode, 4), (FireClass::Prefill, 7)] {
        for symbol in &gemma4_lowered(class, rows).kernels {
            if !bridged.contains(symbol.as_str()) {
                unreachable.insert(symbol.clone());
            }
        }
    }
    assert!(
        unreachable.is_empty(),
        "gemma4 kernels with no stated bridge row: {unreachable:?}"
    );
}

#[test]
#[ignore = "enumeration aid, not a claim"]
fn print_the_gemma4_vocabulary() {
    for (class, rows) in [(FireClass::Decode, 4), (FireClass::Prefill, 7)] {
        let l = gemma4_lowered(class, rows);
        eprintln!("=== gemma4 {class:?}: {} launches, arena {}", l.launches.len(), l.arena_bytes);
        for (i, k) in l.kernels.iter().enumerate() {
            let n = l.launches.iter().filter(|x| x.kernel as usize == i).count();
            eprintln!("  {k}  x{n}");
        }
        for launch in &l.launches {
            let k = &l.kernels[launch.kernel as usize];
            if k.contains("packed") || k.contains("residual_add") || k.contains("rounded")
                || k.contains("naive") || k.contains("attention_flashinfer_prefill")
                || k.contains("transpose") || k.contains("no_scale") || k.contains("geglu")
            {
                let args = &l.args[launch.args.start as usize..launch.args.end as usize];
                eprintln!("  L {k} rows={:?} layers={:?} args={args:?}", launch.rows, launch.layers);
            }
        }
    }
}

/// Every symbol the bridge can dispatch: the DSL tables plus the
/// driver-internal one.
fn bridged_symbols() -> BTreeSet<&'static str> {
    kernels_cuda::KERNELS
        .iter()
        .chain(kernels_cuda::driver_internal::DRIVER_KERNELS)
        .filter(|k| !k.operands.is_empty())
        .map(|k| k.symbol)
        .collect()
}

#[test]
fn every_launch_of_the_anchor_deployment_binds() {
    for (class, rows) in [(FireClass::Decode, 4), (FireClass::Prefill, 7)] {
        let l = lowered(class, rows);
        assert!(!l.launches.is_empty(), "{class:?} lowered to nothing");

        // The arena the frame would allocate — the binder only addresses
        // it, so a dangling sentinel base is fine off-device.
        let frame = Frame { arena: 0x10000 as *mut c_void, arena_bytes: l.arena_bytes };
        let mut resolver = Sentinels::default();

        for launch in &l.launches {
            let bound = bind(&l, launch, frame, &mut resolver)
                .unwrap_or_else(|r| panic!("{class:?}: launch refused: {r:?}"));
            assert!(!bound.kernel.is_empty());
            assert_eq!(
                bound.args.len(),
                (launch.args.end - launch.args.start) as usize,
                "every stated operand binds"
            );
        }
        assert!(
            !resolver.weights.is_empty(),
            "{class:?}: a forward that names no weights did not lower the model"
        );
    }
}

/// The hybrid's bind claim, GPU-free: every launch of the qwen3_5
/// deployment's decode and prefill texts binds — arena offsets inside
/// the sized arena, every weight and ctx value reaching the resolver.
/// The fire classes the shell fires today; the service classes
/// (StateOnly, CommitAdvance) join when spec-decode does.
#[test]
fn every_launch_of_the_hybrid_deployment_binds() {
    for (class, rows) in [(FireClass::Decode, 4), (FireClass::Prefill, 7)] {
        let l = qwen35_lowered(class, rows);
        assert!(!l.launches.is_empty(), "{class:?} lowered to nothing");
        let frame = Frame { arena: 0x10000 as *mut c_void, arena_bytes: l.arena_bytes };
        let mut resolver = Sentinels::default();
        for launch in &l.launches {
            let bound = bind(&l, launch, frame, &mut resolver)
                .unwrap_or_else(|r| panic!("hybrid {class:?}: launch refused: {r:?}"));
            assert_eq!(
                bound.args.len(),
                (launch.args.end - launch.args.start) as usize,
                "every stated operand binds"
            );
        }
        assert!(!resolver.weights.is_empty());
    }
}

/// The hybrid's dispatchability claim — same as the anchor's, separate
/// test so a missing row names the family that needs it.
#[test]
fn every_lowered_hybrid_kernel_has_a_bridge_row() {
    let bridged = bridged_symbols();
    let mut unreachable = BTreeSet::new();
    for (class, rows) in [(FireClass::Decode, 4), (FireClass::Prefill, 7)] {
        for symbol in &qwen35_lowered(class, rows).kernels {
            if !bridged.contains(symbol.as_str()) {
                unreachable.insert(symbol.clone());
            }
        }
    }
    assert!(
        unreachable.is_empty(),
        "hybrid kernels with no stated bridge row: {unreachable:?}"
    );
}

/// The dispatchability claim: nothing lowers to a kernel the bridge
/// cannot reach. A symbol failing here is not a test problem — it is a
/// row that needs writing (DSL family or driver-internal) BEFORE the
/// dispatch half meets it.
#[test]
fn every_lowered_kernel_has_a_bridge_row() {
    let bridged = bridged_symbols();
    let mut unreachable = BTreeSet::new();
    for (class, rows) in [(FireClass::Decode, 4), (FireClass::Prefill, 7)] {
        for symbol in &lowered(class, rows).kernels {
            if !bridged.contains(symbol.as_str()) {
                unreachable.insert(symbol.clone());
            }
        }
    }
    assert!(
        unreachable.is_empty(),
        "lowered kernels with no stated bridge row: {unreachable:?}"
    );
}

/// The refusals refuse: an arena smaller than the lowering sized is
/// caught at the offending offset, and an unknown weight is named.
#[test]
fn the_binder_diagnoses_drift_rather_than_addressing_through_it() {
    let l = lowered(FireClass::Decode, 4);

    let starved = Frame { arena: 0x10000 as *mut c_void, arena_bytes: 1 };
    let mut resolver = Sentinels::default();
    let refusal = l
        .launches
        .iter()
        .find_map(|launch| bind(&l, launch, starved, &mut resolver).err());
    assert!(
        matches!(refusal, Some(BindRefusal::ArenaOutOfBounds { arena_bytes: 1, .. })),
        "a one-byte arena must refuse: {refusal:?}"
    );

    struct NoWeights;
    impl Resolver for NoWeights {
        fn weight(&mut self, _: &str) -> Option<*const c_void> {
            None
        }
        fn named(&mut self, _: ValueId) -> Option<*mut c_void> {
            Some(0x2000 as *mut c_void)
        }
    }
    let frame = Frame { arena: 0x10000 as *mut c_void, arena_bytes: l.arena_bytes };
    let refusal = l
        .launches
        .iter()
        .find_map(|launch| bind(&l, launch, frame, &mut NoWeights).err());
    assert!(
        matches!(refusal, Some(BindRefusal::UnknownWeight(_))),
        "a weightless store must be diagnosed by NAME: {refusal:?}"
    );
}

#[test]
#[ignore = "enumeration aid, not a claim"]
fn print_all_deployment_vocabularies() {
    // Each deployment's OWN cuda facts — the emissions fixtures' values.
    let deployments: Vec<(&str, LlamaLikeFacts, LlamaLikeCudaFacts)> = vec![
        ("olmo2_1b", LlamaLikeFacts::olmo2_1b(), LlamaLikeCudaFacts {
            xqa_decode: false, decode_fused_post: true, rope_table: true,
            force_prefill_path: false, head_dim_padded: false, gate_up_fused: true,
        }),
        ("qwen2_5_1_5b", LlamaLikeFacts::qwen2_5_1_5b(), LlamaLikeCudaFacts {
            xqa_decode: false, decode_fused_post: false, rope_table: true,
            force_prefill_path: true, head_dim_padded: false, gate_up_fused: true,
        }),
        ("mistral_7b_v03", LlamaLikeFacts::mistral_7b_v03(), LlamaLikeCudaFacts {
            xqa_decode: false, decode_fused_post: true, rope_table: true,
            force_prefill_path: false, head_dim_padded: false, gate_up_fused: true,
        }),
        ("phi3_mini", LlamaLikeFacts::phi3_mini(), LlamaLikeCudaFacts {
            xqa_decode: false, decode_fused_post: false, rope_table: true,
            force_prefill_path: false, head_dim_padded: true, gate_up_fused: true,
        }),
    ];
    let bridged = bridged_symbols();
    for (name, facts, cuda) in &deployments {
        for class in [FireClass::Decode, FireClass::Prefill] {
            let plan = llama_like_cuda(facts, cuda, class);
            let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; 4];
            let l = lower(&plan, &rows, Fire { captures_across_splits: false })
                .expect("lowers");
            let missing: Vec<&String> = l
                .kernels
                .iter()
                .filter(|k| !bridged.contains(k.as_str()))
                .collect();
            let frame = Frame { arena: 0x10000 as *mut c_void, arena_bytes: l.arena_bytes };
            let mut r = Sentinels::default();
            for launch in &l.launches {
                let _ = bind(&l, launch, frame, &mut r);
            }
            let dp = driver_cuda_new::model::executor::DispatchPlan::new(&plan, &l);
            for i in 0..l.launches.len() {
                if let Some(w) = &dp.spec(i).weight {
                    r.weights.insert(w.clone());
                }
            }
            let mut names: Vec<_> = r
                .weights
                .iter()
                .filter(|n| !n.contains("layer.") || n.contains("layer.0."))
                .collect();
            names.sort();
            eprintln!(
                "{name} {class:?}: kernels={:?}\n  MISSING_ROWS={missing:?}\n  weights0={names:?}",
                l.kernels
            );
            for launch in &l.launches {
                let k = &l.kernels[launch.kernel as usize];
                if k == "rope::rope_bf16" || k == "norm::residual_add_bf16" || k == "norm::add_bias_bf16" {
                    let args = &l.args[launch.args.start as usize..launch.args.end as usize];
                    eprintln!("  L {k} rows={:?} args={args:?}", launch.rows);
                }
            }
        }
    }
}

#[test]
#[ignore = "enumeration aid, not a claim"]
fn print_the_hybrid_vocabulary() {
    for (class, rows) in [(FireClass::Decode, 4), (FireClass::Prefill, 7)] {
        let l = qwen35_lowered(class, rows);
        eprintln!("=== hybrid {class:?}: {} launches, arena {} bytes", l.launches.len(), l.arena_bytes);
        for (i, k) in l.kernels.iter().enumerate() {
            let n = l.launches.iter().filter(|x| x.kernel as usize == i).count();
            eprintln!("  {k}  x{n}");
        }
        for launch in l.launches.iter().take(40) {
            let args = &l.args[launch.args.start as usize..launch.args.end as usize];
            eprintln!(
                "  L kernel={} rows={:?} args={args:?}",
                l.kernels[launch.kernel as usize], launch.rows
            );
        }
    }
}

#[test]
#[ignore = "enumeration aid, not a claim"]
fn print_the_anchor_vocabulary() {
    for (class, rows) in [(FireClass::Decode, 4), (FireClass::Prefill, 7)] {
        let l = lowered(class, rows);
        eprintln!("=== {class:?}: {} launches, arena {} bytes", l.launches.len(), l.arena_bytes);
        for (i, k) in l.kernels.iter().enumerate() {
            let n = l.launches.iter().filter(|x| x.kernel as usize == i).count();
            eprintln!("  {k}  x{n}");
        }
        {
            let frame = Frame { arena: 0x10000 as *mut c_void, arena_bytes: l.arena_bytes };
            let mut r = Sentinels::default();
            for (i, launch) in l.launches.iter().enumerate() {
                let _ = bind(&l, launch, frame, &mut r);
                let _ = i;
            }
            let dp = driver_cuda_new::model::executor::DispatchPlan::new(&plan_of(class), &l);
            for i in 0..l.launches.len() {
                if let Some(w) = &dp.spec(i).weight {
                    r.weights.insert(w.clone());
                }
            }
            let mut names: Vec<_> = r.weights.iter().collect();
            names.sort();
            let head: Vec<_> = names.iter().filter(|n| !n.contains("layer.") || n.contains("layer.0.") || n.contains("layer.27.")).collect();
            eprintln!("  weights: {head:?}");
        }
        for launch in l.launches.iter().take(14) {
            let args = &l.args[launch.args.start as usize..launch.args.end as usize];
            eprintln!(
                "  L kernel={} rows={:?} args={args:?}",
                l.kernels[launch.kernel as usize], launch.rows
            );
        }
    }
}
