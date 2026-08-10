//! Regenerate the committed static-C++ forms of the lowered class traces
//! (north-star-dsl.md rung 3), one TU per deployment:
//!
//! ```text
//! cargo run -p pie-forward --bin emit-cuda
//! ```
//!
//! The deployment list — which files, from which fact sets — lives in
//! `model::emissions`, shared with `tests/generated_cuda.rs` so the check
//! and the writer cannot disagree. The facts are each deployment's
//! LIVE-anchored set; the driver runs a generated pair only when its own
//! derived facts digest matches the constant embedded in that file (drift →
//! interpreter, loudly — the mechanism that corrects any guessed fact on
//! first live run).

fn main() {
    for e in model::emissions::committed_cuda_emissions() {
        let path = format!("{}/../{}", env!("CARGO_MANIFEST_DIR"), e.rel_path());
        std::fs::create_dir_all(std::path::Path::new(&path).parent().unwrap()).unwrap();
        std::fs::write(&path, &e.text).unwrap();
        println!("wrote {path}");
    }
}
