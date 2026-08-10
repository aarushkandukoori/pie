//! The committed generated `.inc`s match what the emitter produces —
//! the cbindgen-header rule applied to rung 3's artifacts: a drift between
//! the declaration (or the emitter) and the committed static C++ cannot
//! happen quietly. Regenerate with `cargo run -p pie-forward --bin
//! emit-cuda` and review the diff; then re-run the three-way parity gate.
//!
//! The deployment list comes from `model::emissions` — the SAME list the
//! bin writes. This file used to hold a hand-mirrored copy of every fact
//! set ("must mirror `bin/emit-cuda.rs` exactly"), which is the
//! `workspace_bytes` shape: two statements of one list, and the test goes
//! on proving the committed files match an emission nothing writes anymore.

fn generated_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// Every emission's committed file is byte-identical to a fresh emission.
#[test]
fn committed_incs_are_regeneration_clean() {
    for e in model::emissions::committed_cuda_emissions() {
        let path = generated_root().join(e.rel_path());
        let committed = std::fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("committed {}: {err}", e.rel_path()));
        assert_eq!(
            committed,
            e.text,
            "generated {}.inc drifted from the emitter; regenerate with \
             `cargo run -p pie-forward --bin emit-cuda`, review the diff, and \
             re-run the three-way parity gate",
            e.name
        );
    }
}

/// The other direction: every committed `generated/*.inc` IS an emission.
///
/// The byte comparison above proves each listed file is fresh; it cannot see
/// a file the list no longer names — a deployment removed from the emitter
/// leaves its `.inc` behind, still compiled into the driver by whatever
/// `#include` names it, silently pinned at its last regeneration. This walk
/// closes that direction: the set of files on disk equals the set the
/// emitter writes, exactly.
#[test]
fn every_committed_inc_is_an_emission() {
    let emissions = model::emissions::committed_cuda_emissions();
    let listed: std::collections::BTreeSet<String> =
        emissions.iter().map(|e| e.rel_path()).collect();

    let mut on_disk = std::collections::BTreeSet::new();
    let model_dir = generated_root().join("driver-cuda/csrc/src/model");
    for family in std::fs::read_dir(&model_dir).expect("driver-cuda model dir") {
        let family = family.expect("dir entry").path();
        let generated = family.join("generated");
        if !generated.is_dir() {
            continue;
        }
        for f in std::fs::read_dir(&generated).expect("generated dir") {
            let f = f.expect("dir entry").path();
            if f.extension().is_some_and(|e| e == "inc") {
                let rel = f
                    .strip_prefix(generated_root())
                    .expect("under crates/")
                    .to_string_lossy()
                    .into_owned();
                on_disk.insert(rel);
            }
        }
    }

    assert_eq!(
        on_disk, listed,
        "the committed generated/*.inc set and the emitter's list disagree; \
         a file only on disk is pinned at its last regeneration, a file only \
         in the list has never been written — run \
         `cargo run -p pie-forward --bin emit-cuda`"
    );
}
