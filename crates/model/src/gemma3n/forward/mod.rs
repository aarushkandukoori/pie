//! gemma3n's forward.
//!
//! FACTS ONLY. The traced body is blocked on one missing DSL primitive,
//! and the gap is worth stating precisely because it is not gemma3n's
//! problem — it is the first time any family has needed it.
//!
//! # What AltUp needs that the DSL cannot say
//!
//! `altup_predict` produces `[k, Tokens, hidden]` — all K streams. The
//! layer body then runs on ONE of them, the ACTIVE stream, and
//! `altup_correct` folds the result back into all K. In `gemma3n.cpp`
//! that selection is a pointer offset: no kernel launches, nothing is
//! copied, the body is simply handed `predictions + active * N * H`.
//!
//! The DSL has no way to name it. Every `Val` is the output of an op, and
//! there is no view/slice primitive — so the text either states a launch
//! that does not happen, or hands the body a rank-3 value every
//! downstream statement then refuses (`sigmoid_gate_mul operands must
//! share a shape`, which is exactly how this was found).
//!
//! A view is a BUFFER question, and amendment A gives buffer assignment
//! to `lower::Buffers`. So the primitive belongs there: a `Val` that
//! re-windows another without recording an op. That is a real change to
//! what a `Val` can be, and inventing it to make one family's text
//! compile is how a declaration starts describing the DSL instead of the
//! model — the same reason kimi_k3's MLA output gate is refused rather
//! than approximated.
//!
//! deepseek_v4's hyper-connections do NOT need this and that is the
//! useful contrast: HC mixes all K streams every layer and never selects
//! one, so its statements are whole-value throughout. Two rank-K residual
//! schemes, and only one of them asks the IR a new question.
//!
//! The facts below are complete and tested; the body follows the
//! primitive.

pub mod facts;
