//! The checkpoint loader's portable half: heap planning.
//!
//! Everything here is offset arithmetic over
//! [`DecodeGeometry`](crate::batch::DecodeGeometry), compiled and tested on
//! any host. The Metal side — allocating the heap, binding the argument
//! tables, staging tensors — layers on top and stays under
//! `src/metal/`. The ledger is `PARITY-LOADER.md`.

mod heap;

pub use heap::{HeapParams, HeapPlan, align_up, plan_heap};
