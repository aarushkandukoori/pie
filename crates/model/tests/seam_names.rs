//! The SEAM: what a forward pass names, against what `wire()` can answer.
//!
//! Two vocabularies invent names for the same tensors and nothing compares
//! them. The DSL invents TRACE names (`layer.3.qkv`) as it records a
//! forward pass; a load contract invents PUBLISHED names
//! (`model.layers.3.self_attn.qkv_proj.fused.weight`) as it authors the
//! staging; and [`model::weight_names::wire`] is the one bridge between
//! them. A trace name `wire()` cannot answer reaches the driver's
//! resolver, which returns `None` — and `Resolver::weight`'s own doc says
//! what that is:
//!
//! > `None` — which is DRIFT, not absence: a trace that names a weight the
//! > store lacks was traced against a different binding.
//!
//! **The seam fails silently by construction.** `Wiring::alias` records a
//! row only `if self.has(&published)`, so a name the contract never
//! published is dropped with no diagnostic, and the refusal surfaces one
//! name at a time at FIRE time as `BindRefusal::UnknownWeight`. The design
//! knows it is drift and detects it one request too late. This moves the
//! whole class to CI.
//!
//! ## What this asks, precisely
//!
//! Not "does this checkpoint wire" — that needs a checkpoint. It asks the
//! stronger and cheaper question: **can `wire()` EVER emit this name, for
//! any checkpoint at all?** A name outside its reachable set is
//! unanswerable by construction, and no fixture can rescue it.
//!
//! So the published side is a `|_| true` predicate under each of the three
//! naming schemes `wire()` recognises, which yields the maximal set. That
//! makes a listed stem a genuine hole rather than a fixture artefact, and
//! it makes this test's failure mode the safe one: it can miss a
//! deployment-specific gap, and it cannot invent one.
//!
//! Layer indices are normalised to `layer.*`, because a trace at four
//! layers and a `wire()` at eight would otherwise disagree about
//! everything for no reason. The question is about SPELLINGS.

#![cfg(all(feature = "forward", feature = "config"))]

use std::collections::{BTreeMap, BTreeSet};

use model::config::HfConfig;
use model_compiler::lower::{Arg, Fire, Row, lower};
use model_compiler::trace::{FireClass, ForwardPlan};

/// The three naming schemes `wire()` recognises, each by the one tensor
/// only it ships.
///
/// They have to be given separately rather than as one all-true
/// predicate, because the schemes are mutually exclusive BY DESIGN:
/// `qwen3_5` returns early when the gemma-4 per-layer embedding table is
/// present, since the two share a prefix. An all-true predicate would
/// therefore suppress qwen3.5's aliases entirely and understate the
/// reachable set — which would report holes that are not there.
fn schemes() -> [(&'static str, fn(&str) -> bool); 3] {
    [
        ("llama-like", |n: &str| !n.starts_with("model.language_model.")),
        ("gemma-4", |n: &str| n.starts_with("model.language_model.")),
        ("qwen3.5", |n: &str| {
            n.starts_with("model.language_model.")
                && n != "model.language_model.embed_tokens_per_layer.weight"
        }),
    ]
}

/// Every trace name `wire()` can emit under any scheme, layer-normalised.
fn answerable() -> BTreeSet<String> {
    let hf = HfConfig { num_hidden_layers: 4, ..HfConfig::default() };
    let mut out = BTreeSet::new();
    for (_, published) in schemes() {
        let w = model::weight_names::wire(&hf, &published);
        for (trace, _) in &w.aliases {
            out.insert(normalise(trace));
        }
        for (trace, _) in &w.joins {
            out.insert(normalise(trace));
        }
    }
    out
}

/// `layer.3.qkv` -> `layer.*.qkv`.
fn normalise(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for part in name.split('.') {
        if !out.is_empty() {
            out.push('.');
        }
        if part.parse::<u64>().is_ok() {
            out.push('*');
        } else {
            out.push_str(part);
        }
    }
    out
}

/// The `Arg::Weight` stems a family's decode plan names, layer-normalised
/// and narrowed to the ones that are actually TENSORS.
///
/// Two kinds of name come out of `Arg::Weight` and are not weights, and
/// counting either would report a hole where there is none.
///
/// **`scale.…` is a HOST SCALAR.** `dsl::cuda::scalar_mul` given no value
/// names a `scale.*` that the driver looks up in `ctx.scales`, a table it
/// built from a config — the arm strips the prefix and never touches the
/// weight store. `wire()` has a third channel for these
/// (`Wiring::scalars`, which the driver reads into
/// `gemma_layer_scalars`), and it maps PUBLISHED names rather than trace
/// names, so it cannot be checked the same way. That channel is
/// unchecked; this test says so rather than pretending otherwise.
///
/// **The empty stem is a WEIGHTLESS statement.**
/// `norm::per_head_rmsnorm_bf16` is the V-norm without a gamma, and the
/// trace records an `Arg::Weight("")` for the slot it does not fill.
/// Nothing resolves it because nothing needs to.
fn stems(plan: &ForwardPlan) -> BTreeSet<String> {
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; 4];
    let l = lower(plan, &rows, Fire { captures_across_splits: false })
        .expect("the corpus lowers");
    l.args
        .iter()
        .filter_map(|a| match a {
            Arg::Weight(name) if !name.is_empty() && !name.starts_with("scale.") => {
                Some(normalise(name))
            }
            Arg::Weight(_) | Arg::Named { .. } | Arg::Arena { .. } => None,
        })
        .collect()
}

/// Every declared forward family, at its own facts fixture, decode class.
///
/// The same eleven `golden_plans.rs` holds — deliberately, because a
/// family that has a golden and no row here is a family whose seam nobody
/// is checking, and the two lists diverging is itself the bug.
fn corpus() -> Vec<(&'static str, ForwardPlan)> {
    use model::families::llama_like::forward::facts::{LlamaLikeCudaFacts, LlamaLikeFacts};
    use model::gemma_4::forward::facts::{Gemma4CudaFacts, Gemma4Facts};
    use model::gpt_oss::forward::facts::{GptOssCudaFacts, GptOssFacts};
    use model::qwen_3_5::forward::facts::{Qwen35CudaFacts, Qwen35HybridFacts};

    vec![
        (
            "llama_like",
            model::families::llama_like::forward::llama_like_cuda(
                &LlamaLikeFacts::qwen3_0_6b(),
                &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
                FireClass::Decode,
            ),
        ),
        (
            "qwen3_5",
            model::qwen_3_5::forward::qwen3_5_hybrid_cuda(
                &Qwen35HybridFacts::qwen3_5_0_8b(),
                &Qwen35CudaFacts::qwen3_5_0_8b_synthetic(),
                FireClass::Decode,
            ),
        ),
        (
            "gemma_4",
            model::gemma_4::forward::gemma4_cuda(
                &Gemma4Facts::gemma_4_e4b(),
                &Gemma4CudaFacts::gemma_4_e4b_synthetic(),
                FireClass::Decode,
            ),
        ),
        (
            "gpt_oss",
            model::gpt_oss::forward::gpt_oss_cuda(
                &GptOssFacts::gpt_oss_20b(),
                &GptOssCudaFacts::gpt_oss_20b_synthetic(),
                FireClass::Decode,
            ),
        ),
        (
            "gemma_2",
            model::gemma_2::forward::gemma2_cuda(
                &model::gemma_2::forward::facts::Gemma2Facts::gemma_2_9b(),
                FireClass::Decode,
            ),
        ),
        (
            "gemma3n",
            model::gemma3n::forward::gemma3n_cuda(
                &model::gemma3n::forward::facts::Gemma3nFacts::gemma3n_synthetic(),
                FireClass::Decode,
            ),
        ),
        (
            "deepseek_v4",
            model::deepseek_v4::forward::dsv4_cuda(
                &model::deepseek_v4::forward::facts::Dsv4Facts::dsv4_synthetic(),
                FireClass::Decode,
            ),
        ),
        (
            "glm5",
            model::glm5::forward::glm5_cuda(
                &model::glm5::forward::facts::Glm5Facts::glm5_106b_a12b(),
                FireClass::Decode,
            ),
        ),
        (
            "kimi_k2",
            model::kimi_k2::forward::kimi_cuda(
                &model::kimi_k2::forward::facts::KimiFacts::kimi_k2(),
                &model::kimi_k2::forward::facts::KimiCudaFacts::kimi_k2_synthetic(),
                FireClass::Decode,
            ),
        ),
        (
            "kimi_k3",
            model::kimi_k3::forward::kimi_k3_cuda(
                &model::kimi_k3::forward::facts::KimiK3Facts::kimi_k3_synthetic(),
                FireClass::Decode,
            ),
        ),
        (
            "nemotron_h",
            model::nemotron_h::forward::nemotron_h_cuda(
                &model::nemotron_h::forward::facts::NemotronHFacts::nemotron_h_synthetic(),
                FireClass::Decode,
            ),
        ),
    ]
}

/// The stems `wire()` cannot answer today, per family — the seam's debt.
///
/// A CLOSED list, and that is the whole point: it shrinks as builders land
/// and a stem that JOINS it is a family that has started naming something
/// nothing can resolve, which is a fire away from `UnknownWeight`. Sorted
/// within a family so the diff when one leaves is one line.
///
/// `gpt_oss` is the row that bites TODAY and the reason this list is a
/// test rather than a note: it is the only family here with both a
/// `FACTS_ROWS` entry in the CUDA shell and a Prefill arm, so a gpt-oss
/// checkpoint LOADS, reports itself healthy, and dies at its first fire.
/// The other eight are not yet reachable for other reasons, so their debt
/// is owed and not yet due.
#[rustfmt::skip]
const NOT_YET_WIRED: &[(&str, &[&str])] = &[
    // The three families `wire()` has builders for, and the only three
    // that can serve a checkpoint end to end today.
    ("llama_like", &[]),
    ("qwen3_5", &[]),
    ("gemma_4", &[]),
    // ONE STEM, AND IT IS A SPELLING. `llama_like`'s gemma branch wires
    // `post_feedforward_layernorm` to the trace name `mlp_norm`, and
    // gemma-2's forward asks for `post_mlp_norm`. The tensor is staged
    // and named; the two halves of the seam simply chose different
    // words for it, which is the cheapest possible instance of exactly
    // what this test exists to catch.
    ("gemma_2", &["layer.*.post_mlp_norm"]),
    // THE ROW THAT BITES. gpt_oss is the only family below with both a
    // `FACTS_ROWS` entry in the CUDA shell and a Prefill arm, so a
    // gpt-oss checkpoint LOADS, reports itself healthy, and dies at its
    // first fire on `UnknownWeight("layer.0.router")`. The other six owe
    // the same debt and are not yet reachable, so theirs is not yet due.
    ("gpt_oss", &[
        "layer.*.attn_sinks",
        "layer.*.expert_down_bank",
        "layer.*.expert_gate_up_bank",
        "layer.*.router",
        "layer.*.router_bias",
    ]),
    ("gemma3n", &[
        "layer.*.altup_correct_norm",
        "layer.*.altup_norm",
        "layer.*.laurel_post_norm",
        "layer.*.post_mlp_norm",
    ]),
    // MLA and the latent cache: three families, one shape. `kv_b_proj`
    // and `q_a_norm` are the latent projection's two halves and all
    // three name them.
    ("deepseek_v4", &[
        "layer.*.attn_sink",
        "layer.*.expert.{e}.down",
        "layer.*.expert.{e}.gate_up",
        "layer.*.kv_norm",
        "layer.*.router_bias",
    ]),
    ("glm5", &[
        "layer.*.expert.{e}.down",
        "layer.*.expert.{e}.gate_up",
        "layer.*.idx_weights_proj",
        "layer.*.idx_wk",
        "layer.*.idx_wq_b",
        "layer.*.kv_b_proj",
        "layer.*.q_a_norm",
    ]),
    ("kimi_k2", &[
        "layer.*.experts.down_packed",
        "layer.*.experts.down_scale",
        "layer.*.experts.gate_packed",
        "layer.*.experts.gate_scale",
        "layer.*.experts.up_packed",
        "layer.*.experts.up_scale",
        "layer.*.kv_b_proj",
        "layer.*.q_a_norm",
    ]),
    ("kimi_k3", &[
        "layer.*.attn_res_norm",
        "layer.*.attn_res_proj",
        "layer.*.expert.{e}.down",
        "layer.*.expert.{e}.gate_up",
        "layer.*.kda_a_log",
        "layer.*.kda_dt_bias",
        "layer.*.kda_k_conv",
        "layer.*.kda_o_norm",
        "layer.*.kda_q_conv",
        "layer.*.kda_v_conv",
        "layer.*.kv_a_norm",
        "layer.*.kv_b_proj",
        "layer.*.q_a_norm",
    ]),
    ("nemotron_h", &[
        "layer.*.expert.{e}.down",
        "layer.*.expert.{e}.up",
        "layer.*.mamba_a_log",
        "layer.*.mamba_conv",
        "layer.*.mamba_d",
        "layer.*.mamba_dt_bias",
        "layer.*.mamba_norm",
        "layer.*.norm",
        "layer.*.router",
        "layer.*.router_bias",
    ]),
];

/// Every weight a family's decode plan names is one `wire()` can emit, or
/// is written down.
///
/// The failure message is the whole value: a name that JOINED means a
/// forward pass started asking for something no checkpoint can answer, and
/// a name that LEFT means a builder landed and the line should go.
#[test]
fn every_traced_weight_is_a_name_wire_can_emit() {
    let can = answerable();
    assert!(
        can.contains("layer.*.qkv") && can.contains("embed"),
        "the answerable set lost its anchors, so `wire()`'s shape changed \
         rather than a family's: {can:?}"
    );

    let expected: BTreeMap<&str, BTreeSet<&str>> = NOT_YET_WIRED
        .iter()
        .map(|(f, names)| (*f, names.iter().copied().collect()))
        .collect();

    let mut actual: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
    for (family, plan) in corpus() {
        let missing: BTreeSet<String> =
            stems(&plan).into_iter().filter(|s| !can.contains(s)).collect();
        actual.insert(family, missing);
    }

    assert_eq!(
        actual.keys().copied().collect::<BTreeSet<_>>(),
        expected.keys().copied().collect::<BTreeSet<_>>(),
        "the family list moved: NOT_YET_WIRED and the corpus must name the \
         same families, or a family's seam is unchecked"
    );

    let mut report = String::new();
    for (family, missing) in &actual {
        let want: BTreeSet<String> =
            expected[family].iter().map(|s| (*s).to_string()).collect();
        for joined in missing.difference(&want) {
            report.push_str(&format!(
                "  {family}: `{joined}` is named by the forward pass and \
                 `wire()` can never emit it — the first fire that reaches \
                 this weight fails with UnknownWeight.\n"
            ));
        }
        for left in want.difference(missing) {
            report.push_str(&format!(
                "  {family}: `{left}` is wired now — delete its line from \
                 NOT_YET_WIRED.\n"
            ));
        }
    }
    assert!(report.is_empty(), "the seam moved:\n{report}");
}

/// The fact that could only ever be false.
///
/// `abi_shell.rs` derives kimi's fused latent projection as
/// `aliases.contains_key("layer.0.q_kv_a_fused")`. The contract publishes
/// that join and the forward consumes it — but `wire()` has no kimi
/// builder, so the alias is never created, the fact is permanently
/// `false`, and the fusion is paid for at load and never read. Were it
/// ever true, the launch would fail with `UnknownWeight`.
///
/// Its own test because it is the SHARPEST case: not a name a fire has not
/// reached yet, but a name whose absence is silently load-bearing
/// somewhere else. When a kimi builder lands this flips, and the driver's
/// derivation starts telling the truth for the first time.
#[test]
fn kimis_fused_latent_projection_is_still_unreachable() {
    let can = answerable();
    assert!(
        !can.contains("layer.*.q_kv_a_fused"),
        "a kimi builder landed: `wire()` can now emit `q_kv_a_fused`, so \
         `abi_shell`'s `aliases.contains_key(\"layer.0.q_kv_a_fused\")` is \
         no longer permanently false. Check that the forward, the contract \
         and the driver agree before deleting this test."
    );
}
