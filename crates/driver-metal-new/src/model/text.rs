//! Which text the loaded checkpoint is.
//!
//! # Selecting is not choosing
//!
//! "Nothing in the driver may choose a kernel" is the crate's governing rule,
//! and a reader could mistake this module for a breach of it. It is not, and
//! the distinction is worth stating precisely because it will be reached for
//! again when the other three families land.
//!
//! A *choice* is the driver deciding what to run. What happens here is a
//! **lookup**: the checkpoint states its architecture, and this answers with
//! the text written for it. Nothing about which kernels fire is decided here —
//! the text names every symbol, the lowering flattens it, and the executor
//! walks the result. Remove this module and the same kernels would run; you
//! would simply have no way to say which model you loaded.
//!
//! The test is the one `metal.md` gives for the whole crate: *does removing it
//! change which kernels fire?* It does not.
//!
//! # Why the driver and not the engine
//!
//! It could sit in the seam instead. It sits here because running the model it
//! loaded is the driver's job, and because `driver-cuda-new`'s shell does the
//! same in `pie_cuda_launch` for the same reason. A seam that selected texts
//! would have to know every family, and it would learn a new one every time a
//! driver did.

use model_compiler::trace::{FireClass, ForwardPlan};

/// Why no text could be selected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unfamiliar {
    /// No text has been written for this architecture.
    ///
    /// Three of four families are in this state today (`gemma4`, `gpt-oss`,
    /// `qwen`), and it is the honest report: their driver code still exists
    /// but the text that would replace it does not.
    NoText {
        /// What the checkpoint called itself.
        arch: String,
        /// The architectures a text does exist for.
        known: Vec<&'static str>,
    },
}

/// The architectures `llama_like`'s text serves.
///
/// One text covers many architectures — that is what makes it a *family* — so
/// this is a list rather than a name. Every entry is a deployment the family's
/// facts can describe.
const LLAMA_LIKE: &[&str] = &[
    "llama", "llama3", "llama4", "mistral", "phi3", "olmo2", "qwen2", "qwen3",
];

/// Every architecture some text serves.
#[must_use]
pub fn known() -> Vec<&'static str> {
    LLAMA_LIKE.to_vec()
}

/// The text for `arch`, traced for `class`.
///
/// `facts` and `metal` are the deployment's, and they are the caller's to
/// supply because they come from the checkpoint's descriptor — not from
/// anything this module could derive.
///
/// # Errors
///
/// [`Unfamiliar::NoText`], naming the architecture and what is known. A driver
/// that guessed a text here would run a different model's program against this
/// checkpoint's weights, which is fluent nonsense rather than a failure.
pub fn plan_for(
    arch: &str,
    class: FireClass,
    facts: &model::families::llama_like::forward::facts::LlamaLikeFacts,
    metal: &model::families::llama_like::forward::facts::LlamaLikeMetalFacts,
) -> Result<ForwardPlan, Unfamiliar> {
    if LLAMA_LIKE.contains(&arch) {
        return Ok(model::families::llama_like::forward::llama_like_metal(
            facts, metal, class,
        ));
    }
    Err(Unfamiliar::NoText {
        arch: arch.to_string(),
        known: known(),
    })
}

/// Whether any text serves `arch`.
#[must_use]
pub fn serves(arch: &str) -> bool {
    LLAMA_LIKE.contains(&arch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_text_serves_many_architectures_which_is_what_a_family_is() {
        assert!(serves("qwen3"));
        assert!(serves("llama3"));
        assert!(serves("mistral"));
    }

    #[test]
    fn an_architecture_with_no_text_says_so_and_says_what_is_known() {
        // Three of four families are here today. A driver that guessed would
        // run another model's program against these weights, which is fluent
        // nonsense rather than a failure.
        assert!(!serves("gemma4"));
        let facts = model::families::llama_like::forward::facts::LlamaLikeFacts::qwen3_0_6b();
        let metal = model::families::llama_like::forward::facts::LlamaLikeMetalFacts::synthetic();
        match plan_for("gemma4", FireClass::Decode, &facts, &metal) {
            Err(Unfamiliar::NoText { arch, known }) => {
                assert_eq!(arch, "gemma4");
                assert!(known.contains(&"qwen3"), "and what IS served: {known:?}");
            }
            Ok(_) => panic!("gemma4 has no metal text yet"),
        }
    }

    #[test]
    fn a_served_architecture_traces_a_plan_of_that_family() {
        let facts = model::families::llama_like::forward::facts::LlamaLikeFacts::qwen3_0_6b();
        let metal = model::families::llama_like::forward::facts::LlamaLikeMetalFacts::synthetic();
        let plan = plan_for("qwen3", FireClass::Decode, &facts, &metal).expect("qwen3 is served");
        assert!(
            plan.family.starts_with("llama_like"),
            "the plan states its family: {}",
            plan.family
        );
    }
}
