//! How far ONE driver would already reach.
//!
//! The flat list is family-independent now: a rectangle carries its
//! kernel by index and its operands as slots, so a driver walking it
//! needs nothing per-family except a name-to-tensor map. What it still
//! needs is an ARM per launcher symbol — the call itself.
//!
//! Four executors exist and between them resolve a set of symbols. Seven
//! families were declared without one. This measures the overlap, which
//! is the size of the remaining work and the only honest way to state it:
//! not "seven executors to write" but "N symbols that no arm covers".
//!
//! It is a measurement, not a gate — it prints, and only fails if the
//! registries stop being readable.

use model_compiler::trace::{FireClass, ForwardPlan, OpKind};
use std::collections::BTreeSet;

fn arms() -> BTreeSet<String> {
    let root = format!(
        "{}/../driver-cuda/csrc/src/model",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut out = BTreeSet::new();
    for fam in ["llama_like", "qwen3_5", "gemma4", "mixtral"] {
        let path = format!("{root}/{fam}/declared_forward.cpp");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
        for (i, _) in text.match_indices("== \"") {
            let before = &text[..i];
            if !(before.ends_with("k ") || before.ends_with("kernel ")) {
                continue;
            }
            if let Some(end) = text[i + 4..].find('"') {
                out.insert(text[i + 4..i + 4 + end].to_string());
            }
        }
    }
    assert!(!out.is_empty(), "the registries stopped being literal compares");
    out
}

fn stated(plan: &ForwardPlan) -> BTreeSet<String> {
    plan.ops
        .iter()
        .filter_map(|o| match &o.kind {
            OpKind::Launch { kernel, .. } => Some(kernel.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn how_many_symbols_the_undriven_families_still_owe() {
    use model::*;
    let d = FireClass::Decode;
    let plans: Vec<(&str, ForwardPlan)> = vec![
        ("glm5", glm5::forward::glm5_cuda(&glm5::forward::facts::Glm5Facts::glm5_106b_a12b(), d)),
        ("kimi_k2", kimi_k2::forward::kimi_cuda(
            &kimi_k2::forward::facts::KimiFacts::kimi_k2(),
            &kimi_k2::forward::facts::KimiCudaFacts::kimi_k2_synthetic(),
            d,
        )),
        ("kimi_k3", kimi_k3::forward::kimi_k3_cuda(
            &kimi_k3::forward::facts::KimiK3Facts::kimi_k3_synthetic(), d)),
        ("deepseek_v4", deepseek_v4::forward::dsv4_cuda(
            &deepseek_v4::forward::facts::Dsv4Facts::dsv4_synthetic(), d)),
        ("nemotron_h", nemotron_h::forward::nemotron_h_cuda(
            &nemotron_h::forward::facts::NemotronHFacts::nemotron_h_synthetic(), d)),
        ("gemma3n", gemma3n::forward::gemma3n_cuda(
            &gemma3n::forward::facts::Gemma3nFacts::gemma3n_synthetic(), d)),
        ("gemma_2", gemma_2::forward::gemma2_cuda(
            &gemma_2::forward::facts::Gemma2Facts::gemma_2_9b(), d)),
    ];

    let have = arms();
    let mut owed_all: BTreeSet<String> = BTreeSet::new();
    println!("arms across the four existing executors: {}", have.len());
    for (name, plan) in &plans {
        let s = stated(plan);
        let owed: Vec<&String> = s.iter().filter(|k| !have.contains(*k)).collect();
        println!(
            "{name:12} states {:2}  covered {:2}  owes {:2}",
            s.len(),
            s.len() - owed.len(),
            owed.len()
        );
        owed_all.extend(owed.into_iter().cloned());
    }
    println!("\nDISTINCT symbols no arm covers: {}", owed_all.len());
    for k in &owed_all {
        println!("  {k}");
    }
}
