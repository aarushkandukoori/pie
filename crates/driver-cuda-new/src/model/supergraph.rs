//! The union cache: one instantiated graph per (R, N) bucket.
//!
//! # What is in the key, and what is deliberately not
//!
//! A captured graph bakes in every address and every launch geometry it
//! recorded. So whatever a capture may not vary over has to be in the
//! key, and whatever it CAN vary over must not be — putting a variant bit
//! in the key is exactly how a union stops being a union and becomes N
//! separate captures with extra steps.
//!
//! In the key:
//!
//! * **R**, the request count, and **N**, the token count. The launch
//!   geometry is a function of them.
//! * the fire class. `Decode` and `Prefill` are different traces, not
//!   variants of one.
//! * which model is loaded. Two deployments share nothing.
//!
//! NOT in the key, and this is the whole point:
//!
//! * hook attachment, mask kind, correction arm, LoRA presence — every
//!   `GuardPred` axis. These are FOLDED: the union lowering emits all
//!   arms, the arms become conditional nodes, and a device predicate word
//!   selects between them per launch.
//!
//! The measure of success is that a bucket's exec count stays at one as
//! structurally-distinct requests arrive, rather than growing with the
//! number of distinct programs.

use std::collections::HashMap;

use crate::cuda::{GraphExec, StreamRef};
use crate::error::Result;

/// What a capture may NOT vary over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BucketKey {
    /// Requests in the fire.
    pub requests: u32,
    /// Token rows in the fire.
    pub tokens: u32,
    /// The fire class, as `FireClass as u8` — a plain integer because the
    /// key is a hash key and the trace type carries no `Hash`.
    pub fire: u8,
    /// Which loaded model this graph addresses. Two deployments' captures
    /// share no buffer, so they may not share a key.
    pub model: u64,
}

impl BucketKey {
    /// The key for a fire.
    #[must_use]
    pub const fn new(
        requests: u32,
        tokens: u32,
        fire: model_compiler::trace::FireClass,
        model: u64,
    ) -> Self {
        Self { requests, tokens, fire: fire as u8, model }
    }
}

/// The instantiated graphs, by bucket.
///
/// Deliberately not an LRU yet: a bucket set is small (the R×N shapes a
/// deployment actually fires) and evicting a graph while a launch is in
/// flight is a use-after-free rather than a miss, so eviction is a
/// decision that wants the replay path to exist first.
#[derive(Debug, Default)]
pub struct SupergraphCache {
    execs: HashMap<BucketKey, GraphExec>,
    hits: u64,
    misses: u64,
}

impl SupergraphCache {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The exec for `key`, if one is captured.
    pub fn get(&mut self, key: BucketKey) -> Option<&GraphExec> {
        if self.execs.contains_key(&key) {
            self.hits += 1;
        } else {
            self.misses += 1;
        }
        self.execs.get(&key)
    }

    /// Install a freshly instantiated exec.
    pub fn insert(&mut self, key: BucketKey, exec: GraphExec) {
        self.execs.insert(key, exec);
    }

    /// Replay `key`'s graph onto `stream`, if it is captured.
    ///
    /// Returns `Ok(false)` for a miss, which is the caller's cue to
    /// capture — not an error, because a cold bucket is the normal first
    /// fire of every shape.
    ///
    /// # Errors
    ///
    /// If the launch refuses.
    pub fn replay(&mut self, key: BucketKey, stream: StreamRef<'_>) -> Result<bool> {
        let Some(exec) = self.get(key) else { return Ok(false) };
        exec.launch(stream)?;
        Ok(true)
    }

    /// How many execs are live — the number this design exists to keep
    /// small.
    #[must_use]
    pub fn len(&self) -> usize {
        self.execs.len()
    }

    /// Is the cache empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.execs.is_empty()
    }

    /// Hits and misses since construction, for the metric that says
    /// whether the union is actually folding anything.
    #[must_use]
    pub const fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_compiler::trace::FireClass;

    #[test]
    fn variant_bits_are_not_in_the_key() {
        // There is no field for them to occupy. This test is a shape
        // assertion, not a behaviour one: it fails to COMPILE if someone
        // adds a mask or lora bit to the key, which is the review this
        // design most needs.
        let a = BucketKey::new(4, 4, FireClass::Decode, 7);
        let b = BucketKey::new(4, 4, FireClass::Decode, 7);
        assert_eq!(a, b);
    }

    #[test]
    fn the_shape_axes_are() {
        let base = BucketKey::new(4, 4, FireClass::Decode, 7);
        assert_ne!(base, BucketKey::new(5, 4, FireClass::Decode, 7));
        assert_ne!(base, BucketKey::new(4, 8, FireClass::Decode, 7));
        assert_ne!(base, BucketKey::new(4, 4, FireClass::Prefill, 7));
        assert_ne!(base, BucketKey::new(4, 4, FireClass::Decode, 8));
    }

    #[test]
    fn a_miss_is_not_an_error() {
        let mut c = SupergraphCache::new();
        assert!(c.is_empty());
        assert!(c.get(BucketKey::new(1, 1, FireClass::Decode, 0)).is_none());
        assert_eq!(c.stats(), (0, 1));
    }
}
