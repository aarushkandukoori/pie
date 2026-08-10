//! **A real checkpoint's weights, through the generic executor, and what came
//! out.**
//!
//! `device_text_fire.rs` proves the fire executes against sentinels;
//! `device_checkpoint_names.rs` proves every name binds against a checkpoint.
//! Neither looks at a NUMBER, and the gap between them is where a driver hides
//! its worst defects: a fire that runs to completion over correctly-addressed
//! weights and computes nonsense is indistinguishable from a working one
//! unless somebody reads the output.
//!
//! So this reads the output. Not against a reference — that is the accuracy
//! gate's job and it wants one — but against the three failure modes that
//! account for most of the distance:
//!
//!   * **all zeros.** A projection told its extents are zero no-ops; a weight
//!     bound to an unwritten arena slot contributes nothing. Both leave the
//!     residual stream exactly as the embedding left it, or empty.
//!   * **non-finite.** A norm handed a zero epsilon divides by the root of the
//!     mean square alone; a NaN anywhere spreads to everything downstream
//!     within one layer.
//!   * **degenerate.** Every row identical means the per-token axis is not
//!     reaching the kernels — a launch whose grid collapsed, or a gather
//!     reading token 0 for every lane.
//!
//! None of those three is subtle and all three are invisible without a read.
//! Passing here is not correctness; it is the floor beneath which correctness
//! cannot be discussed.
//!
//! Gated on `PIE_METAL_SMOKE_CHECKPOINT`, the same variable the other
//! checkpoint tests take. Run against
//! `mlx-community/Llama-3.2-1B-Instruct-4bit`.

#![cfg(target_vendor = "apple")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use driver_metal_new::metal::{Compiler, Context, allocate};
use driver_metal_new::model::dispatch::Geometry;
use driver_metal_new::model::encode::Pipelines;
use driver_metal_new::model::executor::{FireTable, Resolver, Slice};
use driver_metal_new::model::frame::{Step, lower_step};
use driver_metal_new::model::kv::{Pool, Shape};
use driver_metal_new::model::load::load;
use driver_metal_new::model::resolve::{Names, Store};
use driver_metal_new::region::Region as _;
use model::families::llama_like::forward::llama_like_metal;
use model_compiler::trace::FireClass;

fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join("kernels-metal/kernels")
}

fn snapshot() -> Option<PathBuf> {
    std::env::var_os("PIE_METAL_SMOKE_CHECKPOINT").map(PathBuf::from)
}

/// The `pie.model/1` descriptor for a snapshot. See
/// `device_checkpoint_names.rs` — a TEST may normalize a checkpoint.
fn descriptor_for(snapshot: &Path) -> String {
    let raw = std::fs::read_to_string(snapshot.join("config.json"))
        .expect("the snapshot has a config.json");
    let root: serde_json::Value = serde_json::from_str(&raw).expect("config.json parses");
    model::config::descriptor(&root, snapshot.to_str().expect("utf8 path"))
        .expect("the config normalizes to a descriptor")
        .to_string()
}

/// What a run of the whole arena found.
#[derive(Debug, Default)]
struct Census {
    finite_nonzero: usize,
    zero: usize,
    nan: usize,
    inf: usize,
    /// The widest magnitude seen, which says whether anything saturated.
    max_abs: f32,
}

/// Count what is in `bytes`, read at `element` bytes per value.
///
/// The element width is NOT a constant over an arena, and assuming it was is
/// the first thing this gate got wrong about itself. 89% of a llama-1B
/// decode's arena is the readout, which is `DType::F32`; the rest is the
/// residual stream, which is bf16. Reading the f32 half as bf16 reports the
/// LOW sixteen bits of every logit as a number, which came out as 5.8e11 and
/// looked exactly like saturation.
///
/// `Arg::Arena` states `bytes` per element for precisely this reason -- its
/// own doc says a driver that windows a rectangle needs the stride and that
/// every hand windowing in the CUDA executor multiplied by two -- so the
/// census asks the lowering rather than guessing.
fn census(bytes: &[u8], element: usize) -> Census {
    let mut c = Census::default();
    for chunk in bytes.chunks_exact(element) {
        let v = if element == 4 {
            f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
        } else {
            // A bf16 is the TOP half of an f32, so widening is a shift.
            f32::from_bits(u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16)
        };
        if v.is_nan() {
            c.nan += 1;
        } else if v.is_infinite() {
            c.inf += 1;
        } else if v == 0.0 {
            c.zero += 1;
        } else {
            c.finite_nonzero += 1;
            c.max_abs = c.max_abs.max(v.abs());
        }
    }
    c
}

/// The checkpoint's weights, the fire's tables, and the pool's geometry.
struct Live<'a> {
    store: Store<'a>,
    tables: Slice,
    /// Where each table starts in `tables`, and how long it is, in u32s.
    spans: Vec<(usize, usize)>,
    shape: Shape,
    pages: &'a dyn Fn(u16, bool) -> Option<Slice>,
}

impl Resolver for Live<'_> {
    fn weight(&mut self, name: &str) -> Option<Slice> {
        self.store.weight(name)
    }
    fn named(&mut self, value: model_compiler::trace::ValueId) -> Option<Slice> {
        self.store.named(value)
    }
    fn kv(&mut self, layer: u16, values: bool) -> Option<Slice> {
        (self.pages)(layer, values)
    }
    fn fire(&mut self, which: FireTable) -> Option<Slice> {
        // The REAL tables, staged the way the seam stages them. A zeroed
        // region for all of them was the first draft, and it decodes token 0
        // at position 0 on every lane -- a legitimate input, and a degenerate
        // one that tells you nothing about whether the per-token axis works.
        let i = match which {
            FireTable::TokenIds => 0,
            FireTable::Positions => 1,
            FireTable::RequestOfToken => 2,
            FireTable::KvPageIndices => 3,
            FireTable::KvPageIndptr => 4,
            FireTable::KvWritePage => 5,
            FireTable::KvWriteOffset => 6,
            // No custom mask on this path, and no pool number is an address.
            _ => return None,
        };
        let (at, len) = self.spans[i];
        (len > 0).then(|| Slice {
            address: self.tables.address + (at * 4) as u64,
            bytes: (len * 4) as u64,
        })
    }
    fn pool(&mut self, which: FireTable) -> Option<u32> {
        Some(match which {
            FireTable::KvHeadStride => self.shape.head_dim,
            FireTable::KvSeqStride => self.shape.kv_heads * self.shape.head_dim,
            FireTable::KvPageSize => self.shape.page_size,
            _ => return None,
        })
    }
}

#[test]
fn a_real_checkpoints_weights_produce_finite_varied_activations() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let descriptor = descriptor_for(&snapshot);
    let loaded = load(&context, &snapshot, &descriptor).expect("the checkpoint loads");
    let model_facts = driver_metal_new::facts::ModelFacts::from_descriptor(&descriptor)
        .expect("the descriptor states the model's facts");
    let dg =
        driver_metal_new::batch::geometry_from_facts(&model_facts).expect("a decodable geometry");
    let (facts, metal) =
        driver_metal_new::model::text::facts_from(&dg, |t| loaded.tensors.contains_key(t));

    // Four lanes, one token each: the decode a scheduler posts.
    let step = Step {
        token_ids: &[128_000, 9906, 1917, 128_001],
        qo_indptr: &[0, 1, 2, 3, 4],
        sampling_indices: &[0, 1, 2, 3],
        ..Step::default()
    };
    let plan = llama_like_metal(&facts, &metal, FireClass::Decode);
    let lowered = lower_step(&plan, &step).expect("the step lowers");

    let shape = Shape {
        layers: facts.layers,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        page_size: 16,
        pages: 64,
        element_bytes: 2,
    };
    let pool = Pool::allocate(&context, shape).expect("a pool");
    let pages = |layer: u16, values: bool| {
        pool.layer(u32::from(layer)).map(|l| Slice {
            address: if values { l.v.gpu_address() } else { l.k.gpu_address() },
            bytes: shape.layer_bytes(),
        })
    };

    // The fire's own tables, staged into one region exactly as the engine seam
    // stages them. FOUR DIFFERENT tokens at four different positions, which is
    // what makes the per-token check below able to fail.
    let tokens: Vec<u32> = step.token_ids.to_vec();
    let positions: Vec<u32> = vec![0, 1, 2, 3];
    let req_of_token: Vec<u32> = vec![0, 1, 2, 3];
    // One page per request, and the CSR that says so.
    let kv_page_indices: Vec<u32> = vec![0, 1, 2, 3];
    let kv_page_indptr: Vec<u32> = vec![0, 1, 2, 3, 4];
    let w_page: Vec<u32> = kv_page_indices.clone();
    let w_off: Vec<u32> = positions.iter().map(|p| p % shape.page_size).collect();

    let mut blob: Vec<u32> = Vec::new();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for table in [
        &tokens,
        &positions,
        &req_of_token,
        &kv_page_indices,
        &kv_page_indptr,
        &w_page,
        &w_off,
    ] {
        spans.push((blob.len(), table.len()));
        blob.extend_from_slice(table);
    }
    let staged = allocate(&context, (blob.len() * 4) as u64, "fire tables").expect("a table region");
    // SAFETY: freshly allocated, nothing encoded against it yet.
    unsafe {
        let raw = core::slice::from_raw_parts(blob.as_ptr().cast::<u8>(), blob.len() * 4);
        staged.write(0, raw).expect("the tables stage");
    }

    let named = HashMap::new();
    let mut live = Live {
        store: Store::new(Names::mlx(), &loaded.tensors, &named),
        tables: Slice {
            address: staged.gpu_address(),
            bytes: (blob.len() * 4) as u64,
        },
        spans,
        shape,
        pages: &pages,
    };

    let geometry = Geometry {
        q_heads: facts.q_heads,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        rotary_dims: facts.head_dim,
        n_experts: facts.n_experts,
        experts_per_token: facts.experts_per_token,
    };
    let (timing, arena) = driver_metal_new::model::run::run_keeping_arena(
        &context,
        &compiler,
        &mut pipelines,
        &lowered,
        geometry,
        &mut live,
    )
    .expect("the fire runs against real weights");

    assert!(
        timing.encode > std::time::Duration::ZERO,
        "nothing was encoded"
    );
    assert!(
        live.store.missed().is_empty(),
        "the fire asked for {} name(s) the checkpoint does not answer, so the \
         census below would be about sentinels: {:?}",
        live.store.missed().len(),
        live.store.missed()
    );

    let mut read = vec![0u8; arena.len() as usize];
    // SAFETY: the command buffer retired before `run_keeping_arena` returned,
    // so nothing is writing the arena.
    unsafe {
        let raw = core::slice::from_raw_parts(
            arena.contents().as_ptr().cast_const().cast::<u8>(),
            arena.len() as usize,
        );
        read.copy_from_slice(raw);
    }

    // Every arena region the lowering states, censused at ITS element width.
    // Regions rather than the whole buffer: an arena is mixed-dtype and one
    // census over all of it is meaningful only for the dtype that happens to
    // dominate.
    let mut regions: Vec<(usize, usize, usize)> = lowered
        .args
        .iter()
        .filter_map(|a| match a {
            model_compiler::lower::Arg::Arena { at, width, bytes } => {
                Some((*at, *width as usize * *bytes as usize, *bytes as usize))
            }
            _ => None,
        })
        .collect();
    regions.sort_unstable();
    regions.dedup();

    let mut c = Census::default();
    let mut widest_by_element: Vec<(usize, f32)> = Vec::new();
    for (at, len, element) in &regions {
        let end = (at + len).min(read.len());
        if *at >= end {
            continue;
        }
        let r = census(&read[*at..end], *element);
        c.finite_nonzero += r.finite_nonzero;
        c.zero += r.zero;
        c.nan += r.nan;
        c.inf += r.inf;
        widest_by_element.push((*element, r.max_abs));
        c.max_abs = c.max_abs.max(r.max_abs);
        eprintln!(
            "  @{at:>8} {len:>8} B x{element}: {:>7} nz, {:>7} zero, max |v| = {}",
            r.finite_nonzero, r.zero, r.max_abs
        );
    }
    let widest = |e: usize| {
        widest_by_element
            .iter()
            .filter(|(el, _)| *el == e)
            .map(|(_, v)| *v)
            .fold(0.0f32, f32::max)
    };
    eprintln!(
        "arena {} B in {} region(s): {} finite non-zero, {} zero, {} NaN, {} inf; \
         widest |v| = {} (bf16 {}, f32 {})",
        read.len(),
        regions.len(),
        c.finite_nonzero,
        c.zero,
        c.nan,
        c.inf,
        c.max_abs,
        widest(2),
        widest(4),
    );

    // ── the three failure modes ──
    assert_eq!(
        c.nan, 0,
        "the fire produced {} NaN(s). A NaN anywhere spreads to everything \
         downstream within one layer, so this is not a rounding question.",
        c.nan
    );
    assert_eq!(
        c.inf, 0,
        "the fire produced {} infinity(ies), which is what a norm handed a \
         zero epsilon does to a near-zero row.",
        c.inf
    );
    assert!(
        c.finite_nonzero > c.zero * 4,
        "the arena is {} zero to {} non-zero. A projection told its extents \
         are zero no-ops and leaves exactly this, so a mostly-empty arena is \
         a fire that ran and did not compute.",
        c.zero,
        c.finite_nonzero
    );

    // MAGNITUDES, and the bounds are loose on purpose: what is being caught is
    // saturation, not inaccuracy. A llama-1B decode measures its widest
    // activation under one and its widest logit around 0.08 -- both small,
    // both finite -- and the bounds sit orders of magnitude out so a real
    // drift trips them and a different checkpoint does not.
    assert!(
        c.max_abs > 1e-4 && c.max_abs < 1e3,
        "the widest value anywhere is {}, which is saturation or silence \
         rather than a forward pass.",
        c.max_abs
    );

    // The READOUT, by name rather than by dtype: it is the widest region the
    // text states, because a vocabulary is wider than anything else in a
    // decode.
    let readout = regions
        .iter()
        .max_by_key(|(_, len, _)| *len)
        .copied()
        .expect("the text states a readout");
    let (at, len, element) = readout;
    let end = (at + len).min(read.len());
    let r = census(&read[at..end], element);
    assert!(
        r.finite_nonzero > r.zero,
        "the readout is {} zero to {} non-zero. Exactly half zero is a dtype \
         disagreement -- the kernel writing bf16 into a slot sized for f32 -- \
         and mostly zero is a readout that did not run.",
        r.zero,
        r.finite_nonzero
    );
    assert!(
        r.max_abs > 1e-4,
        "every logit is under 1e-4, so the readout accumulated nothing."
    );

    // The per-token axis: four lanes decoded four DIFFERENT tokens, so four
    // identical readout rows means the axis never reached the kernels -- a
    // grid that collapsed, or a gather reading token 0 for every lane. It is
    // the one failure of the three that survives every magnitude check,
    // because one correct row repeated four times looks correct everywhere
    // else.
    let row = &read[at..end];
    if row.len() >= 4 {
        let lane = row.len() / 4;
        let lanes: Vec<&[u8]> = row.chunks_exact(lane).take(4).collect();
        assert!(
            lanes.windows(2).any(|w| w[0] != w[1]),
            "every lane's readout is byte-identical, so the per-token axis \
             never reached the kernels: four different tokens produced one \
             answer."
        );
    }
}
