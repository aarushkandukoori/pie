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
    AffineFormat, DagOptions, EntryNames, IoSlot, PsoFeatures, build_decode_dag,
    build_scratch_schedule, geometry_from_facts, plan_decode_psos, scratch_slot_elems,
};
use driver_metal_new::facts::ModelFacts;
use driver_metal_new::loader::{compile_load_plan, metal_storage_target};
use driver_metal_new::metal::Compiler;
use driver_metal_new::metal::{Context, DecodeStep, Stepper, load_step_psos, stage_decode_storage};
use driver_metal_new::region::Region as _;
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

    let options = DagOptions {
        with_argmax: true,
        ..DagOptions::default()
    };
    let dag = build_decode_dag(&geometry, &tuning, options);
    let schedule = build_scratch_schedule(&dag, false).expect("the DAG schedules hazard-free");

    let features = PsoFeatures {
        argmax: true,
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

    // Fire the checkpoint's own <bos> at position 0. SeqLen is position+1.
    // A multimodal wrapper keeps its text facts one level down.
    let bos: u32 = [&root, root.get("text_config").unwrap_or(&root)]
        .iter()
        .find_map(|level| level.get("bos_token_id"))
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .expect("the config states its bos");
    let io = |slot: IoSlot| storage.io[slot as usize].as_ref().expect("io slot");
    // SAFETY: nothing is encoded yet; the buffers are host-owned.
    unsafe {
        io(IoSlot::TokenId).write(0, &bos.to_le_bytes()).unwrap();
        io(IoSlot::Position).write(0, &0u32.to_le_bytes()).unwrap();
        io(IoSlot::SeqLen).write(0, &1u32.to_le_bytes()).unwrap();
    }

    let mut stepper = Stepper::new(&context).expect("a stepper");
    let timing = step.fire(&mut stepper).expect("the command buffer retires");
    eprintln!("fired: encode {:?}, gpu {:?}", timing.encode, timing.gpu);

    // The first answer check: the logits must be finite, non-degenerate
    // numbers and the argmax a real token. "Token 0 forever" and "all
    // zeros" are this family's two historical silent failures; both are
    // visible from here without a reference.
    let logits = io(IoSlot::Logits);
    let vocab = geometry.vocab as usize;
    // SAFETY: the step retired; the GPU is done with the pool.
    let bytes =
        unsafe { std::slice::from_raw_parts(logits.contents().cast::<u8>().as_ptr(), vocab * 2) };
    let mut finite = 0usize;
    let mut nonzero = 0usize;
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, pair) in bytes.chunks_exact(2).enumerate() {
        let value = f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
        if value.is_finite() {
            finite += 1;
            if value != 0.0 {
                nonzero += 1;
            }
            if value > best.1 {
                best = (i, value);
            }
        }
    }
    let next = {
        // SAFETY: as above.
        let raw = unsafe {
            std::slice::from_raw_parts(io(IoSlot::NextToken).contents().cast::<u8>().as_ptr(), 4)
        };
        u32::from_le_bytes(raw.try_into().unwrap())
    };
    eprintln!(
        "logits: {finite}/{vocab} finite, {nonzero} nonzero; host argmax {} ({:.3}); device argmax {next}",
        best.0, best.1
    );
    assert_eq!(
        finite, vocab,
        "a NaN in the logits is a wrong kernel upstream"
    );
    assert!(
        nonzero > vocab / 2,
        "logits mostly zero: the head never ran or wrote elsewhere"
    );
    assert_eq!(
        next as usize, best.0,
        "the device argmax must agree with the host's read of the same logits"
    );
}
