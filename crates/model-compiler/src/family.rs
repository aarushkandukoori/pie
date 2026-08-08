//! Family declarations.
//!
//! Each function here is a forward pass written as ordinary Rust over a
//! [`TraceBuilder`]; running it *is* the trace. Branches on facts execute
//! now and vanish — a deployment that binds no fused QKV traces three
//! matmuls and no split, and the traced forms differ the way two compiled
//! programs differ, not the way two runtime paths do.

//! ## One module per family
//!
//! This was one 3,754-line file. A family is the unit a reader arrives
//! looking for and the unit the migration to `crates/model` moves, so it is
//! the unit the file system shows. Nothing moved between families in the
//! split -- each module holds exactly the functions and the tests that were
//! already grouped under its section comment.

pub mod gemma4;
pub mod gpt_oss;
pub mod llama_like;
pub mod qwen3_5;

// Flat re-export: every call site says `family::llama_like(..)`, and a
// family's name is already in its function names, so `family::llama_like::llama_like`
// would say it twice.
pub use gemma4::*;
pub use gpt_oss::*;
pub use llama_like::*;
pub use qwen3_5::*;
