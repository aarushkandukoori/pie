//! Per-device tuned constants: the table, the family selection, and the env
//! overrides that measured them.
//!
//! Portable on purpose. The only Apple-specific part of tuning is asking the
//! device what it is ([`crate::metal::DeviceInfo`]); everything downstream of
//! that answer is arithmetic over two integers, so it lives here where it can
//! be tested on any host. The C++ shell splits the same way, for the same
//! reason -- `device_tuning_apple.mm` asks and `device_tuning.cpp` decides.
//!
//! # Provenance
//!
//! Every constant below is a measurement, and the comment on it is the run.
//! Changing one without a run is how a tuning table becomes a table of
//! numbers nobody can defend, so the comments are the load-bearing part.
//!
//! # A note on the C++ this replaces
//!
//! `csrc/src/device_tuning.hpp` declares `qmm_min_batch_moe` twice in one
//! struct (8, then 12) and `csrc/src/device_tuning.cpp` has two `case 8:`
//! arms in one `switch`. Both are hard compile errors, so the C++ tuning
//! layer does not currently build; see [`Tuning::for_device`] for how the
//! conflict is resolved here.

use std::env;

/// What the device is, as far as tuning cares.
///
/// Both fields are 0 when unknown, and 0 selects the defaults -- the
/// constants this driver shipped before the tuning layer existed. That is why
/// the type is `Default`-able and why nothing here branches on "did the query
/// succeed": a device that would not answer gets the M1 numbers, which is the
/// same thing every device got before there was a table.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Device {
    /// `MTLGPUFamilyApple<N>`, resolved newest-first because the families are
    /// cumulative. 0 when no Metal device answered.
    pub apple_family: u32,
    /// IOKit's `gpu-core-count`, the only place the count is published --
    /// `MTLDevice` does not expose it. 0 when absent.
    pub gpu_core_count: u32,
}

/// The tuned constants, defaulted to the M1 Max measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tuning {
    /// The batch at which the ported steel GEMM overtakes the batched GEMV,
    /// for a checkpoint whose GEMM reaches the FP16 matrix path.
    ///
    /// M1 Max: 8. Every dense checkpoint measured prefers 8 over the 12 this
    /// used to read, by between 7% and 68% aggregate tok/s; the 12 came from
    /// a sweep taken while the batched GEMM still emulated a bfloat matrix
    /// unit, which it no longer does.
    pub qmm_min_batch: u32,

    /// The same crossover for a ROUTED (mixture) checkpoint.
    ///
    /// Split from [`Self::qmm_min_batch`] because the two measure differently
    /// once the expert GEMM stops emulating a matrix unit: the dense half
    /// moved to 8 and the routed half did not follow on every family.
    pub qmm_min_batch_moe: u32,

    /// The same crossover for a checkpoint whose quantization does NOT reach
    /// the FP16 matrix path (group-128, say), and so runs the emulated GEMM.
    pub qmm_min_batch_emulated: u32,

    /// The threadgroup count at which the unsplit GEMM's BN=32 tile overtakes
    /// the wide one. Set by how soon the wide tile's smaller grid fills the
    /// machine, so it moves with CORE COUNT rather than with family.
    pub qmm_bn_crossover_tg: u32,

    /// Mixture tiling: rows per mid tile.
    pub moe_tile_mid_per: u32,
    /// Mixture tiling: rows per wide tile. The default is effectively
    /// "never split", which is what `1 << 24` says.
    pub moe_tile_wide_per: u32,

    /// Whether the 4-bit dense projections stage to FP16 and use the matrix
    /// instruction the hardware has, rather than emulating it.
    pub fp16_qmm: bool,

    /// Rows per request below which SDPA does not tile.
    pub sdpa_tile_min_rows_per_request: u32,
    /// Whether SDPA uses the matrix path.
    pub sdpa_mma: bool,

    /// Minimum rows per expert before a mixture batches that expert.
    pub moe_batch_min_per_expert: u32,

    /// Gated-delta-net scan geometry: lanes.
    pub gdn_scan_lanes: u32,
    /// Gated-delta-net scan geometry: rows.
    pub gdn_scan_rows: u32,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            qmm_min_batch: 8,
            qmm_min_batch_moe: 8,
            qmm_min_batch_emulated: 12,
            qmm_bn_crossover_tg: 160,
            moe_tile_mid_per: 32,
            moe_tile_wide_per: 1 << 24,
            fp16_qmm: true,
            sdpa_tile_min_rows_per_request: 32,
            sdpa_mma: true,
            moe_batch_min_per_expert: 1,
            gdn_scan_lanes: 32,
            gdn_scan_rows: 4,
        }
    }
}

impl Tuning {
    /// The table entry for `device`, BEFORE environment overrides.
    ///
    /// Kept separate from [`Self::resolve`] so the table can be tested
    /// without the process environment in the way -- the overrides read
    /// globals, and a test that sets them is a test that cannot run beside
    /// another one.
    ///
    /// # The Apple8 conflict
    ///
    /// The C++ has two `case 8:` arms and they disagree. The first sets
    /// `qmm_min_batch = 8` and leaves the routed crossover inherited; the
    /// second leaves the dense one to the default and sets
    /// `qmm_min_batch_moe = 12`. Since the default for the dense crossover IS
    /// 8, the first arm is a no-op restatement of the default and the second
    /// is the only one that says anything, so the second is taken here: on
    /// Apple8, dense 8 (inherited) and routed 12 (named).
    ///
    /// That reading matches the second arm's own comment -- "the DENSE
    /// crossover here is eight, which is now the default and is not
    /// restated" -- and it makes the M2 entry mean what it meant before the
    /// default moved under it. It is a reading of intent rather than a
    /// measurement, and it should be confirmed against the M2 Max sweep the
    /// comments cite before this crate carries traffic.
    #[must_use]
    pub fn for_device(device: Device) -> Self {
        let mut t = Self::default();
        match device.apple_family {
            // M3/M4 generation, measured on an M4 Pro (20 cores) with
            // gemma-4-E4B at concurrency 8: 138.90 tok/s at 12 against 144.04
            // at 8, +3.7%.
            9 => {
                // With 20 cores rather than 32 the wide tile's smaller grid
                // fills the machine sooner, so the tile crossover moves down.
                t.qmm_bn_crossover_tg = 96;
                // 8 is what this device measured and shipped with while there
                // was one crossover, and a mixture ran it too. The routed half
                // is unmeasured on this family; leaving it at 12 would be a
                // change dressed as a default.
                t.qmm_min_batch_moe = 8;
            }
            // M2 generation, measured on an M2 Max (38 cores). The dense
            // crossover is 8, which is the default and is not restated. At the
            // same batches the GEMV still won on every mixture measured --
            // Qwen3-30B by 8%, gemma-4-26B by 12%, gpt-oss-20B by nothing
            // either way -- so the routed crossover stays at 12.
            8 => t.qmm_min_batch_moe = 12,
            _ => {}
        }
        t
    }

    /// [`Self::for_device`] with the environment overrides applied.
    ///
    /// Every tuned constant gets one, and for the same reason the first one
    /// did: measuring a crossover means running the same binary twice with
    /// different answers, and a rebuild between the arms is a different
    /// binary.
    #[must_use]
    pub fn resolve(device: Device) -> Self {
        let mut t = Self::for_device(device);

        t.qmm_min_batch = env_u32("PIE_METAL_QMM_MIN_BATCH", t.qmm_min_batch);
        // The dense override carries the routed and emulated ones unless they
        // are named separately. A sweep that moved only the dense number on a
        // mixture would measure a model that never changed path and read the
        // resulting flat curve as the crossover not mattering.
        let dense_override = env_u32("PIE_METAL_QMM_MIN_BATCH", t.qmm_min_batch_moe);
        t.qmm_min_batch_moe = env_u32("PIE_METAL_QMM_MIN_BATCH_MOE", dense_override);
        let dense_override = env_u32("PIE_METAL_QMM_MIN_BATCH", t.qmm_min_batch_emulated);
        t.qmm_min_batch_emulated = env_u32("PIE_METAL_QMM_MIN_BATCH_EMULATED", dense_override);

        t.qmm_bn_crossover_tg = env_u32("PIE_METAL_QMM_BN_CROSSOVER_TG", t.qmm_bn_crossover_tg);
        t.moe_tile_mid_per = env_u32("PIE_METAL_MOE_TILE_MID_PER", t.moe_tile_mid_per);
        t.moe_tile_wide_per = env_u32("PIE_METAL_MOE_TILE_WIDE_PER", t.moe_tile_wide_per);
        t.fp16_qmm = env_bool("PIE_METAL_FP16_QMM", t.fp16_qmm);
        t.sdpa_tile_min_rows_per_request = env_u32(
            "PIE_METAL_SDPA_TILE_MIN_ROWS",
            t.sdpa_tile_min_rows_per_request,
        );
        t.sdpa_mma = env_bool("PIE_METAL_SDPA_MMA", t.sdpa_mma);
        t.moe_batch_min_per_expert = env_u32(
            "PIE_METAL_MOE_BATCH_MIN_PER_EXPERT",
            t.moe_batch_min_per_expert,
        );
        t.gdn_scan_lanes = env_u32("PIE_METAL_GDN_SCAN_LANES", t.gdn_scan_lanes);
        t.gdn_scan_rows = env_u32("PIE_METAL_GDN_SCAN_ROWS", t.gdn_scan_rows);

        t
    }

    /// The GEMM/GEMV crossover for a given checkpoint shape.
    #[must_use]
    pub fn qmm_min_batch_for(&self, is_moe: bool, fp16_gemm: bool) -> u32 {
        if !fp16_gemm {
            return self.qmm_min_batch_emulated;
        }
        if is_moe {
            self.qmm_min_batch_moe
        } else {
            self.qmm_min_batch
        }
    }

    /// Whether a `bits`/`group` quantization reaches the FP16 matrix path.
    #[must_use]
    pub fn fp16_gemm_format(&self, bits: u32, group: u32) -> bool {
        self.fp16_qmm && bits == 4 && group == 64
    }
}

/// A positive integer from the environment, or `fallback`.
///
/// Zero and negative values are REJECTED rather than accepted, matching the
/// C++: every constant this parses is a count or a batch size, and 0 would
/// disable the thing being measured rather than tune it.
fn env_u32(name: &str, fallback: u32) -> u32 {
    let Ok(raw) = env::var(name) else {
        return fallback;
    };
    match raw.parse::<u32>() {
        Ok(v) if v > 0 => v,
        _ => fallback,
    }
}

/// A boolean from the environment, or `fallback`.
///
/// Unlike [`env_u32`], `0` is a VALUE here and not a rejected one -- it is
/// how a sweep turns a path off. Anything else non-empty is true.
fn env_bool(name: &str, fallback: bool) -> bool {
    let Ok(raw) = env::var(name) else {
        return fallback;
    };
    if raw.is_empty() {
        return fallback;
    }
    raw != "0"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_device_gets_the_m1_defaults() {
        assert_eq!(Tuning::for_device(Device::default()), Tuning::default());
    }

    #[test]
    fn apple9_lowers_the_tile_crossover_and_keeps_the_dense_routed_pair() {
        let t = Tuning::for_device(Device {
            apple_family: 9,
            gpu_core_count: 20,
        });
        assert_eq!(t.qmm_bn_crossover_tg, 96);
        assert_eq!(t.qmm_min_batch, 8);
        assert_eq!(t.qmm_min_batch_moe, 8);
    }

    /// The arm the duplicated C++ `case 8:` could not express. See
    /// [`Tuning::for_device`].
    #[test]
    fn apple8_names_the_routed_crossover_and_inherits_the_dense_one() {
        let t = Tuning::for_device(Device {
            apple_family: 8,
            gpu_core_count: 38,
        });
        assert_eq!(t.qmm_min_batch, 8, "dense is the default, not restated");
        assert_eq!(t.qmm_min_batch_moe, 12, "routed is named on this family");
    }

    #[test]
    fn a_future_family_falls_back_rather_than_guessing() {
        let t = Tuning::for_device(Device {
            apple_family: 10,
            gpu_core_count: 64,
        });
        assert_eq!(t, Tuning::default());
    }

    #[test]
    fn the_emulated_crossover_is_selected_by_format_not_by_routing() {
        let t = Tuning::default();
        assert_eq!(t.qmm_min_batch_for(false, false), t.qmm_min_batch_emulated);
        assert_eq!(t.qmm_min_batch_for(true, false), t.qmm_min_batch_emulated);
        assert_eq!(t.qmm_min_batch_for(false, true), t.qmm_min_batch);
        assert_eq!(t.qmm_min_batch_for(true, true), t.qmm_min_batch_moe);
    }

    #[test]
    fn only_group64_4bit_reaches_the_fp16_path() {
        let t = Tuning::default();
        assert!(t.fp16_gemm_format(4, 64));
        assert!(!t.fp16_gemm_format(4, 128));
        assert!(!t.fp16_gemm_format(8, 64));

        let off = Tuning {
            fp16_qmm: false,
            ..Tuning::default()
        };
        assert!(!off.fp16_gemm_format(4, 64));
    }
}
