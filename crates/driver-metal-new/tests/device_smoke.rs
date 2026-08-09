//! The first fired token: the whole chain against a real checkpoint.
//!
//! `config.json` → descriptor → facts → geometry → load plan → staged
//! storage → scratch schedule → PSO table → bound step → one dispatch walk
//! on the GPU. Nothing here checks the ANSWER yet — that is the accuracy
//! gate's job (golden taps, token-exact decode) — this pins that the
//! assembly holds together: every weight the DAG asks for was staged,
//! every constant bound, every pipeline compiled, and the command buffer
//! retires.
//!
//! Gated on `PIE_METAL_SMOKE_CHECKPOINT` naming a qwen3.5/3.6-family MLX
//! snapshot directory, because a checkpoint is a machine's, not the
//! repo's. Without it the test states it skipped and why.

#![cfg(target_vendor = "apple")]

use std::path::PathBuf;

use driver_metal_new::batch::{
    AffineFormat, DagOptions, EntryNames, PsoFeatures, build_decode_dag, build_scratch_schedule,
    geometry_from_facts, plan_decode_psos, scratch_slot_elems,
};
use driver_metal_new::facts::ModelFacts;
use driver_metal_new::loader::{compile_load_plan, metal_storage_target};
use driver_metal_new::metal::Compiler;
use driver_metal_new::metal::{Context, DecodeStep, Stepper, load_step_psos, stage_decode_storage};
use driver_metal_new::tuning::Tuning;

fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join("kernels-metal/kernels")
}

#[test]
fn the_assembly_fires_one_token_end_to_end() {
    let Some(snapshot) = std::env::var_os("PIE_METAL_SMOKE_CHECKPOINT") else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to a qwen3.5-family MLX snapshot");
        return;
    };
    let snapshot = PathBuf::from(snapshot);
    let config = std::fs::read_to_string(snapshot.join("config.json"))
        .expect("the snapshot has a config.json");
    let root: serde_json::Value = serde_json::from_str(&config).expect("config.json parses");
    let descriptor = model::config::descriptor(&root, snapshot.to_str().expect("utf8 path"))
        .expect("the config converts to a descriptor");
    let descriptor_json = descriptor.to_string();

    // Facts and geometry, refused rather than defaulted.
    let facts = ModelFacts::from_descriptor(&descriptor_json)
        .expect("the driver's facts read the descriptor");
    let mut geometry = geometry_from_facts(&facts).expect("the config describes this family");
    geometry.quant = AffineFormat {
        bits: u32::try_from(facts.quant_bits).unwrap_or(4),
        group: u32::try_from(facts.quant_group_size).unwrap_or(64),
    };
    eprintln!(
        "geometry: {} layers, hidden {}, vocab {}, moe={}",
        geometry.n_layers,
        geometry.hidden,
        geometry.vocab,
        geometry.is_moe()
    );

    // The load plan, authored in-process.
    let target = metal_storage_target();
    let (plan, _moe) = compile_load_plan(&snapshot, &target, &descriptor_json)
        .expect("the plan compiles and its files exist");

    // Device side: stage, schedule, compile, bind.
    let context = Context::new().expect("a Metal device answers");
    let tuning = Tuning::default();
    let max_ctx = 4096u32;
    let scratch_bytes = scratch_slot_elems(&geometry, &tuning, 1) * 2;
    let storage = stage_decode_storage(
        &context,
        &plan,
        &snapshot,
        &geometry,
        max_ctx,
        scratch_bytes,
    )
    .expect("every region allocates and every tensor stages");
    eprintln!("staged {} weights", storage.weights.len());

    let options = DagOptions::default();
    let dag = build_decode_dag(&geometry, &tuning, options);
    let schedule = build_scratch_schedule(&dag, false).expect("the DAG schedules hazard-free");

    let features = PsoFeatures {
        gdn: true,
        gated_attention: true,
        sdpa_d256: geometry.head_dim == 256,
        routed: geometry.is_moe(),
        untied: !geometry.tied_embeddings,
        ..PsoFeatures::default()
    };
    let pso_plan = plan_decode_psos(&EntryNames::bf16_g64_b4(), features);
    let compiler = Compiler::new(&context).expect("the shader compiler starts");
    let psos = load_step_psos(&compiler, &context, &kernels_dir(), &pso_plan)
        .expect("every planned entrypoint compiles");

    let step = DecodeStep::prepare(
        &context, &storage, &geometry, &tuning, options, &schedule, psos, max_ctx,
    )
    .expect("the step binds whole");

    // Fire <bos> at position 0: TokenId/Position/SeqLen are already zeroed,
    // which IS that fire.
    let mut stepper = Stepper::new(&context).expect("a stepper");
    let timing = step.fire(&mut stepper).expect("the command buffer retires");
    eprintln!("fired: encode {:?}, gpu {:?}", timing.encode, timing.gpu);
}
