//! How a rectangle becomes a launch.
//!
//! A lowered `Launch` gives rows and layers — an **iteration space**. A
//! dispatch needs a thread grid and a threadgroup. Something has to turn one
//! into the other, and *where that something lives* decides whether the
//! executor is a loop or a switch.
//!
//! # The rule is named, and the rule stays a function
//!
//! The obvious move is to put the geometry on the row as numbers, or as a
//! little expression grammar the row can spell in `const`. Both were tried on
//! paper and both are worse than what is here.
//!
//! Numbers cannot work: a kernel's geometry is a function of the fire — rows,
//! widths, head counts — so a row would have to state a formula, not a value.
//!
//! A grammar can express every rule in the driver today; they are all
//! `source → max → min → divide-rounding-up → multiply`. But writing
//! `Term { floor: 1, cap: 1024, div_ceil: 32, mul: 32 }` **loses the sentence
//! that says why**, and in this codebase those sentences are load-bearing:
//! `dispatch::qmv`'s doc records that its round-up is the difference between
//! computing every output and silently dropping the last few. A grammar buys
//! `const` and pays in explanation.
//!
//! So: the row names a [`Rule`], and the rule remains the documented function
//! it already is. The consequences are the point —
//!
//! * **The driver's dispatch is a loop.** `sig.launch.eval(dims)` for every
//!   launch, with no per-family branch and no per-kernel arm.
//! * **The match is arm-per-RULE, not arm-per-kernel.** Sixteen arms, shared by
//!   every family, every text and every backend that reuses the vocabulary. A
//!   new kernel that launches like an existing one costs zero arms — it names
//!   the rule.
//! * **Every doc comment survives**, beside the code it explains, which is
//!   where this project keeps its arguments.
//!
//! # Where this belongs
//!
//! On [`KernelSig`], as `launch = Rule::Qmv` beside `whole`, `needs` and
//! `lacks` — a launch shape is a contract fact exactly as those are. It is
//! here rather than in `kernels` because the vocabulary had to be shown to
//! cover the existing rules before the tables adopt it, and the test below is
//! that proof: **every rule the driver hand-writes today is reproduced through
//! this enum, exactly.**
//!
//! [`KernelSig`]: https://docs.rs/kernels

use crate::batch::{self as dispatch, Launch};

/// The fire-time quantities a launch rule may read.
///
/// Named rather than positional because a rule takes two or three of them and
/// two adjacent `u32`s that can be swapped is the defect `PARITY-LOADER.md`
/// records in `plan_heap`. Every field is a fact the lowering or the geometry
/// already states; nothing here is derived by the driver.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dims {
    /// Rows the rectangle covers.
    pub rows: u32,
    /// Elements per row of the operand that sizes the launch — a projection's
    /// output width, a norm's row width, an MLP's intermediate.
    pub width: u32,
    /// Query heads.
    pub q_heads: u32,
    /// Key/value heads.
    pub kv_heads: u32,
    /// Elements per head.
    pub head_dim: u32,
    /// Channels a partial rope rotates.
    pub rotary_dims: u32,
    /// Experts the router scores.
    pub n_experts: u32,
    /// Experts each token routes to.
    pub experts_per_token: u32,
}

/// The launch rule a kernel declares.
///
/// One variant per *shape of launch*, not per kernel: `Rms` serves every
/// row-wise norm, `Elementwise` every 256-wide pointwise pass. The names are
/// the ones `batch::dispatch` already gave them, and each variant delegates to
/// that function so the explanation and the arithmetic stay together.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Rule {
    /// The row has not said. Nothing may be dispatched from it — the same
    /// meaning `Source::Unbound` has for operands.
    #[default]
    Unstated,
    /// Affine GEMV: four outputs per simdgroup, two simdgroups per
    /// threadgroup, rounded up.
    Qmv,
    /// Row-wise norm: one threadgroup per row, four elements per thread,
    /// capped at the widest threadgroup Metal allows.
    Rms,
    /// Rope: half the rotary channels per head.
    Rope,
    /// Pointwise over one row, 256-wide — residual adds, embeddings, silu-mul.
    Elementwise,
    /// One threadgroup per head, `head_dim` wide — the q/k/v split and the KV
    /// append.
    PerHead,
    /// Single-pass decode attention: one 1024-thread threadgroup per query
    /// head.
    SdpaVector,
    /// Pointwise over every head's channels, 256-wide.
    PerHeadElementwise,
    /// Gated norm over the GDN value heads.
    GatedRms,
    /// One threadgroup as wide as the expert count, rounded to a simd multiple.
    RouterLane,
    /// One threadgroup per row, as wide as the row, capped at 256.
    RouteRows,
    /// Routed GEMV: `Qmv` per row, per expert slot.
    RoutedQmv,
}

/// Why a rule could not produce a launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ungeometric {
    /// The row states no rule, so nothing can be dispatched from it.
    Unstated,
}

impl Rule {
    /// The launch this rule produces for `dims`.
    ///
    /// # Errors
    ///
    /// [`Ungeometric::Unstated`] when the row has not named a rule. That is
    /// drift, not a runtime condition: a symbol reached dispatch whose
    /// contract does not say how to launch it.
    pub fn eval(self, dims: Dims) -> Result<Launch, Ungeometric> {
        Ok(match self {
            Rule::Unstated => return Err(Ungeometric::Unstated),
            Rule::Qmv => dispatch::qmv(dims.width),
            Rule::Rms => dispatch::rms(dims.width, dims.rows),
            Rule::Rope => dispatch::rope(dims.rotary_dims, dims.q_heads),
            Rule::Elementwise => dispatch::residual(dims.width),
            Rule::PerHead => dispatch::kv_append(dims.head_dim, dims.kv_heads),
            Rule::SdpaVector => dispatch::sdpa(dims.q_heads),
            Rule::PerHeadElementwise => dispatch::attn_gate(dims.q_heads, dims.head_dim),
            Rule::GatedRms => dispatch::gated_rms(dims.kv_heads, dims.head_dim),
            Rule::RouterLane => dispatch::router_topk(dims.n_experts),
            Rule::RouteRows => dispatch::route_rows(dims.width, dims.rows),
            Rule::RoutedQmv => dispatch::routed_qmv(dims.width, dims.experts_per_token, dims.rows),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims() -> Dims {
        Dims {
            rows: 7,
            width: 4096,
            q_heads: 16,
            kv_heads: 4,
            head_dim: 128,
            rotary_dims: 64,
            n_experts: 128,
            experts_per_token: 8,
        }
    }

    /// The proof the vocabulary is right: every rule the driver hand-writes
    /// today is reachable through the enum and produces the same launch.
    ///
    /// If this holds, moving the rule onto the row is mechanical — the row
    /// names a variant and nothing else changes. If a new kernel ever needs a
    /// launch no variant produces, that is a new variant with its own
    /// documented function, not a special case in the executor.
    #[test]
    fn every_rule_reproduces_the_function_the_driver_already_uses() {
        let d = dims();
        for (rule, expected) in [
            (Rule::Qmv, dispatch::qmv(d.width)),
            (Rule::Rms, dispatch::rms(d.width, d.rows)),
            (Rule::Rope, dispatch::rope(d.rotary_dims, d.q_heads)),
            (Rule::Elementwise, dispatch::residual(d.width)),
            (Rule::PerHead, dispatch::kv_append(d.head_dim, d.kv_heads)),
            (Rule::SdpaVector, dispatch::sdpa(d.q_heads)),
            (
                Rule::PerHeadElementwise,
                dispatch::attn_gate(d.q_heads, d.head_dim),
            ),
            (Rule::GatedRms, dispatch::gated_rms(d.kv_heads, d.head_dim)),
            (Rule::RouterLane, dispatch::router_topk(d.n_experts)),
            (Rule::RouteRows, dispatch::route_rows(d.width, d.rows)),
            (
                Rule::RoutedQmv,
                dispatch::routed_qmv(d.width, d.experts_per_token, d.rows),
            ),
        ] {
            assert_eq!(
                rule.eval(d).expect("a stated rule evaluates"),
                expected,
                "{rule:?} does not reproduce its function"
            );
        }
    }

    #[test]
    fn the_shapes_that_share_a_function_are_one_variant_not_three() {
        // `residual`, `embed` and `silu_mul` are the same 256-wide pointwise
        // launch; the C++ and the driver spell it three times. A kernel that
        // launches like an existing one should cost zero arms.
        let d = dims();
        let ew = Rule::Elementwise.eval(d).expect("stated");
        assert_eq!(ew, dispatch::residual(d.width));
        assert_eq!(ew, dispatch::embed(d.width));
        assert_eq!(ew, dispatch::silu_mul(d.width));

        // The same for the per-head pair: the q/k/v split and the KV append
        // launch identically, over whichever head count they address.
        assert_eq!(
            Rule::PerHead.eval(d).expect("stated"),
            dispatch::q_split(d.head_dim, d.kv_heads),
            "one rule, read with the head count the operand names"
        );
    }

    #[test]
    fn an_unstated_rule_refuses_rather_than_launching_something_plausible() {
        // The default. A symbol whose contract does not say how to launch it
        // has reached dispatch by drift, and a guessed grid is a kernel that
        // runs over the wrong extent — which the hardware does not report.
        assert_eq!(
            Rule::default().eval(dims()),
            Err(Ungeometric::Unstated),
            "unstated must not fall back to anything"
        );
    }

    #[test]
    fn a_rule_reads_only_the_dims_it_names() {
        // Changing a dimension a rule does not use must not move its launch.
        // This is what makes `Dims` safe to grow: a new field cannot silently
        // change an existing rule's geometry.
        let d = dims();
        let wider = Dims {
            n_experts: 999,
            experts_per_token: 3,
            ..d
        };
        assert_eq!(Rule::Rms.eval(d), Rule::Rms.eval(wider));
        assert_eq!(Rule::SdpaVector.eval(d), Rule::SdpaVector.eval(wider));
        assert_ne!(
            Rule::RouterLane.eval(d),
            Rule::RouterLane.eval(wider),
            "and one that DOES name it must move"
        );
    }
}
