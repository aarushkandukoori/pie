//! The load path through `model-loader`'s plan, on a real checkpoint.
//!
//! What this holds: a compiled plan executes into DEVICE memory, the tensors
//! it names are where the plan says they are, and the fused projections the
//! shell used to build by hand come out of the plan instead.

#![cfg(all(feature = "cuda-13", feature = "abi"))]

use std::path::PathBuf;

use driver_cuda_new::loader::plan::{compile_load_plan, cuda_storage_target};
use driver_cuda_new::loader::stage::stage_plan_weights;

/// A cached HF snapshot, or `None` to skip.
fn snapshot(repo: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(format!("models--{repo}"))
        .join("snapshots");
    std::fs::read_dir(snaps).ok()?.find_map(|e| {
        let p = e.ok()?.path();
        p.join("model.safetensors").is_file().then_some(p)
    })
}

fn descriptor() -> Option<String> {
    let p = PathBuf::from(
        "/tmp/claude-0/-root--patissier-work-tart-alpha/\
         7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad/qwen3_descriptor.json",
    );
    std::fs::read_to_string(p).ok()
}

#[test]
fn a_checkpoint_stages_into_device_memory_through_its_plan() {
    let (Some(snap), Some(desc)) = (snapshot("Qwen--Qwen3-0.6B"), descriptor()) else {
        eprintln!("skipped: no cached Qwen3-0.6B or descriptor");
        return;
    };
    let meta = model_loader::checkpoint::read::parse_checkpoint_metadata(&snap)
        .expect("the checkpoint parses");
    let target = cuda_storage_target();
    let (plan, _moe) =
        compile_load_plan(&snap, &meta, &target, &desc).expect("the plan compiles");

    // THE JOINS ARE IN THE PLAN. `Projections::Fused` is what the CUDA
    // GEMMs want, and the shell used to satisfy it by reading q/k/v back
    // off the device and re-uploading their concatenation.
    let fused = plan
        .tensors
        .iter()
        .filter(|t| t.name.contains("qkv_proj.fused"))
        .count();
    assert!(
        fused > 0,
        "the plan carries no fused qkv; the driver would have to build them"
    );

    let alloc = driver_cuda_new::cuda::Allocator::new();
    let staged = stage_plan_weights(&plan, &snap, &alloc).expect("the plan executes");

    assert!(
        staged.spans.len() >= plan.tensors.len(),
        "every tensor the plan names is staged: {} spans for {} tensors",
        staged.spans.len(),
        plan.tensors.len()
    );
    for (name, span) in &staged.spans {
        assert!(!span.ptr.is_null(), "{name} has no address");
        assert!(span.bytes > 0, "{name} is empty");
    }
    // The arena is one allocation, so the whole model is contiguous and the
    // spans are offsets into it — the property that makes this cheaper than
    // a per-tensor `cudaMalloc`.
    assert_eq!(
        staged.owned.len(),
        1,
        "a resident plan should leave nothing outside the arena"
    );
}
