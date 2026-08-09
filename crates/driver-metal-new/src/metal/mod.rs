//! The Apple half: every type here names a Metal or IOKit symbol.
//!
//! Gated on `cfg(target_vendor = "apple")` as a whole, which is what lets the
//! rest of the crate compile and test on a Linux host. The boundary is drawn
//! at "does this need a GPU to be correct", not at "is this about the GPU":
//! the tuning table is about the GPU and lives outside, because its inputs
//! are two integers.
//!
//! # `unsafe`
//!
//! Every objc2 message send is `unsafe`, so this half cannot carry the
//! workspace's `unsafe_code = "forbid"`. What it carries instead is the rule
//! that an `unsafe` block states the invariant it is relying on -- Metal's
//! own API contract does not stop being a contract because it is written in
//! Objective-C.
//!
//! # What is not here yet
//!
//! The device query is first because it is self-contained: it depends on no
//! other Metal object and it feeds [`crate::tuning`], which is already
//! complete and tested. The MTL4 context (queue, allocators, argument table,
//! residency set), the placement heap and the pipeline compiler follow, and
//! they follow on a machine that can compile them -- this module cannot be
//! type-checked on the Linux hosts the rest of the workspace builds on, so
//! writing it ahead of a build is writing it blind.

mod device;

pub use device::DeviceInfo;
