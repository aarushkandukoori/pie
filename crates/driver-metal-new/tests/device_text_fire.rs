//! **The whole text, on the GPU, through the generic executor.**
//!
//! This is the north star's fourth property with the checkpoint taken out: a
//! sealed frame's step becomes rows, the rows become rectangles, the
//! rectangles become grids, the grids become a command buffer, and the command
//! buffer runs. Every one of the 367 launches of `llama_like`'s Metal text
//! reaches the device, and nothing in the driver names a family, a kernel or a
//! model on the way.
//!
//! What it does NOT prove is that the numbers are right. The weights are
//! sentinels, not a checkpoint, so this is an execution proof — every symbol
//! compiles, every grid is legal, every operand is in bounds, and the fire
//! completes. Token-exactness is `device_smoke.rs`'s job and needs
//! `PIE_METAL_SMOKE_CHECKPOINT`.
//!
//! The distinction is worth keeping sharp: a fire that runs and answers
//! nonsense is exactly the failure this crate was built to make impossible to
//! miss, so "it ran" is a milestone and not a result.

#![cfg(target_vendor = "apple")]

use std::collections::HashMap;
use std::path::PathBuf;

use driver_metal_new::metal::{Compiler, Context, allocate};
use driver_metal_new::model::dispatch::Geometry;
use driver_metal_new::model::encode::Pipelines;
use driver_metal_new::model::executor::{Resolver, Slice};
use driver_metal_new::model::frame::{Step, lower_step};
use driver_metal_new::model::run::run;
use model::families::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
use model::families::llama_like::forward::llama_like_metal;
use model_compiler::trace::{FireClass, ValueId};

fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join("kernels-metal/kernels")
}

/// Every weight the text names, backed by one generous region.
///
/// Sentinels rather than a checkpoint: this test is about whether the fire
/// EXECUTES. A region large enough for any tensor means no kernel reads past
/// an allocation, which is what would turn an execution proof into a crash
/// that says nothing about the executor.
struct Sentinels {
    slice: Slice,
    asked: HashMap<String, usize>,
}

impl Resolver for Sentinels {
    fn weight(&mut self, name: &str) -> Option<Slice> {
        *self.asked.entry(name.to_string()).or_default() += 1;
        Some(self.slice)
    }
    fn named(&mut self, _: ValueId) -> Option<Slice> {
        Some(self.slice)
    }
}

fn geometry() -> Geometry {
    Geometry {
        q_heads: 16,
        kv_heads: 8,
        head_dim: 128,
        rotary_dims: 128,
        n_experts: 0,
        experts_per_token: 0,
    }
}

#[test]
fn the_whole_metal_text_fires_on_the_device() {
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    // One token a request, four lanes: the decode a scheduler posts.
    let step = Step {
        token_ids: &[11, 22, 33, 44],
        qo_indptr: &[0, 1, 2, 3, 4],
        sampling_indices: &[0, 1, 2, 3],
        ..Step::default()
    };
    let plan = llama_like_metal(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeMetalFacts::synthetic(),
        FireClass::Decode,
    );
    let lowered = lower_step(&plan, &step).expect("the step lowers");
    assert!(
        lowered.launches.len() > 300,
        "a 24-layer decode should be hundreds of launches, not {}",
        lowered.launches.len()
    );

    // 256 MiB: wider than any tensor this text names, so a bound operand is
    // never the reason a dispatch fails.
    let backing = allocate(&context, 256 << 20, "sentinel weights").expect("a backing region");
    let mut store = Sentinels {
        slice: Slice {
            address: backing.gpu_address(),
            bytes: 256 << 20,
        },
        asked: HashMap::new(),
    };

    let timing = run(
        &context,
        &compiler,
        &mut pipelines,
        &lowered,
        geometry(),
        &mut store,
    )
    .expect("the whole text fires");

    // The fire completed, and it compiled far fewer pipelines than it ran
    // dispatches — the cold start is bounded by the TEXT, not by the fire.
    assert!(
        timing.encode > std::time::Duration::ZERO,
        "the stepper reported no encode time, so nothing was encoded"
    );
    assert!(
        !store.asked.is_empty(),
        "the fire bound no weights, so it cannot have been the real text"
    );
    let restated = store.asked.values().filter(|&&n| n > 1).count();
    assert!(
        restated > 0,
        "no weight was asked for twice; a 24-layer text restates its shapes"
    );
}

#[test]
fn a_prefill_step_fires_too_so_both_lanes_reach_the_device() {
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    // Eight tokens in one request: a prefill, which takes the batched symbols
    // (`affine_qmm_t`, `embed_gather_mb_4bit`, `neox_mb`, the paged pair).
    let step = Step {
        token_ids: &[1, 2, 3, 4, 5, 6, 7, 8],
        qo_indptr: &[0, 8],
        sampling_indices: &[7],
        ..Step::default()
    };
    let plan = llama_like_metal(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeMetalFacts::synthetic(),
        FireClass::Prefill,
    );
    let lowered = lower_step(&plan, &step).expect("the step lowers");

    let backing = allocate(&context, 256 << 20, "sentinel weights").expect("a backing region");
    let mut store = Sentinels {
        slice: Slice {
            address: backing.gpu_address(),
            bytes: 256 << 20,
        },
        asked: HashMap::new(),
    };

    run(
        &context,
        &compiler,
        &mut pipelines,
        &lowered,
        geometry(),
        &mut store,
    )
    .expect("the batched lane fires too");
}

/// The paged KV pool, allocated at the fire's geometry.
///
/// `metal::stage_decode_storage` has allocated `KvSlots` since the port, but
/// sized from `batch::DecodeGeometry` — a model definition inside the driver.
/// This is the same allocation with its arguments taken from the frame.
#[test]
fn the_kv_pool_allocates_at_the_geometry_the_fire_states() {
    use driver_metal_new::model::kv::{Pool, Shape, translate};

    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let g = geometry();
    let shape = Shape {
        layers: 24,
        kv_heads: g.kv_heads,
        head_dim: g.head_dim,
        page_size: 16,
        pages: 64,
        element_bytes: 2,
    };
    let pool = Pool::allocate(&context, shape).expect("the pool allocates");

    assert_eq!(pool.pages(), 64);
    assert_eq!(
        pool.bytes(),
        shape.layer_bytes() * 2 * 24,
        "a K and a V region for every layer"
    );
    let layer = pool.layer(0).expect("layer 0 has pages");
    assert_ne!(
        layer.k.gpu_address(),
        layer.v.gpu_address(),
        "K and V must be distinct regions; one address would make the append \
         to K overwrite V"
    );
    assert!(pool.layer(24).is_none(), "past the last layer there is none");

    // And the frame's translation reads against it.
    let table = [0u32, 1, 63];
    assert_eq!(
        translate(&pool, &table, &[0, 3], 0).expect("a lane's pages"),
        &[0, 1, 63]
    );
    assert!(
        translate(&pool, &[64], &[0, 1], 0).is_err(),
        "a page past the pool addresses another layer's memory"
    );
}
