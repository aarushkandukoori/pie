//! The model executor: running a lowered fire.
//!
//! `model_compiler::lower` states what to run — a flat list of launches, each
//! naming a kernel symbol and carrying its operands. Nothing here chooses a
//! kernel; see `DIRECTION.md` and `model-compiler/DSL-DESIGN.md`.
//!
//! * [`executor`] — binding a launch's operands. Host logic, no device.
//! * [`geometry`] — turning a rectangle into a thread grid: the rule a row
//!   names, so the executor is a loop rather than a switch.

pub mod executor;
pub mod geometry;

pub use executor::{BindRefusal, BoundArg, BoundLaunch, Frame, Resolver, Slice, bind, resolve_arg};
pub use geometry::{Dims, Rule, Ungeometric};
