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
//! It found four defects in its first afternoon, and the fourth is the one
//! that argues for the file:
//!
//!   1. **No barrier between dispatches.** Metal does not order two dispatches
//!      in one compute encoder and the executor's loop emitted none. Three
//!      runs of one fire gave widest activations of 11.7, 23.1 and 4.5e12 --
//!      TWO OF THE THREE looked entirely plausible.
//!   2. **The readout's dtype.** The text said `F32`, `affine_qmv_fast` writes
//!      bfloat, and the logits came back exactly half zero.
//!   3. **Unzeroed arena and KV pool.** A fresh Metal buffer is usually zero
//!      and nothing promises it, so an attention read past what a fire wrote
//!      attended to whatever the allocator last held.
//!   4. **The single-row gather.** `embed_gather_4bit` reads `id[0]` and
//!      writes one row whatever grid it is handed, and the text picked it by
//!      CLASS -- but a decode of four requests is four rows. One readout lane
//!      of four held anything, and NOTHING ELSE WAS WRONG: every launch stated
//!      four rows, every grid covered them, and every other kernel read the
//!      row where the grid put it.
//!
//! Three measurements track what is left, each pinned so it can only improve:
//! declared outputs nothing fills (**0**, was 5), readout lanes that hold
//! anything (**4** of 4, was 1), and the arena's non-zero share (**99%**, was
//! 26%).
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

/// How many dispatches the fire plans, and how many have an empty grid.
fn plan_count(
    lowered: &model_compiler::lower::Lowered,
    facts: &model::families::llama_like::forward::facts::LlamaLikeFacts,
    live: &mut Live<'_>,
) -> String {
    let dispatches = driver_metal_new::model::dispatch::plan(
        lowered,
        driver_metal_new::model::run::table(),
        driver_metal_new::model::executor::Frame {
            arena: Slice {
                address: 0x1_0000_0000,
                bytes: 1 << 30,
            },
        },
        Geometry {
            q_heads: facts.q_heads,
            kv_heads: facts.kv_heads,
            head_dim: facts.head_dim,
            rotary_dims: facts.head_dim,
            n_experts: facts.n_experts,
            experts_per_token: facts.experts_per_token,
        },
        live,
    )
    .expect("the fire plans");
    let empty = dispatches
        .iter()
        .filter(|d| d.grid.contains(&0) || d.threadgroup.contains(&0))
        .count();
    format!("{} ({empty} with an empty grid)", dispatches.len())
}

/// The fire's own tables, staged into one region exactly as the engine seam
/// stages them.
///
/// FOUR DIFFERENT tokens at four different positions, which is what makes the
/// per-token checks able to fail at all. A zeroed region for every table was
/// the first draft and it decodes token 0 at position 0 on every lane -- a
/// legitimate input, and a degenerate one that says nothing about whether the
/// per-token axis works.
fn stage_tables(
    context: &Context,
    step: &Step<'_>,
    page_size: u32,
) -> (driver_metal_new::metal::Handle, Vec<(usize, usize)>) {
    let tokens: Vec<u32> = step.token_ids.to_vec();
    let positions: Vec<u32> = (0..tokens.len() as u32).collect();
    let req_of_token: Vec<u32> = (0..tokens.len() as u32).collect();
    // One page per request, and the CSR that says so.
    let kv_page_indices: Vec<u32> = (0..tokens.len() as u32).collect();
    let kv_page_indptr: Vec<u32> = (0..=tokens.len() as u32).collect();
    let w_page: Vec<u32> = kv_page_indices.clone();
    let w_off: Vec<u32> = positions.iter().map(|p| p % page_size.max(1)).collect();

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
    let staged =
        allocate(context, (blob.len() * 4) as u64, "fire tables").expect("a table region");
    // SAFETY: freshly allocated, nothing encoded against it yet.
    unsafe {
        let raw = core::slice::from_raw_parts(blob.as_ptr().cast::<u8>(), blob.len() * 4);
        staged.write(0, raw).expect("the tables stage");
    }
    (staged, spans)
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

    let (staged, spans) = stage_tables(&context, &step, shape.page_size);

    let named = HashMap::new();
    let mut live = Live {
        store: Store::new(Names::mlx(), &loaded.tensors, &named),
        tables: Slice {
            address: staged.gpu_address(),
            bytes: staged.len(),
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
    // Widen each region to where the NEXT one starts. `width * bytes` is one
    // ROW and a decode's value is `rows` of them, so censusing the stated
    // width looks at the first token's slice and calls it the region -- which
    // cannot tell "nothing wrote this" from "the write landed at the wrong
    // offset inside it".
    let starts: Vec<usize> = regions.iter().map(|(at, _, _)| *at).collect();
    for (i, (at, len, _)) in regions.iter_mut().enumerate() {
        let next = starts
            .iter()
            .skip(i + 1)
            .find(|s| **s > *at)
            .copied()
            .unwrap_or(read.len());
        *len = (*len).max(next - *at).min(read.len() - *at);
    }

    // Which statement each arena offset belongs to, and whether it is that
    // statement's OUTPUT. A region nothing wrote is diagnosable only if the
    // report says which launch was supposed to write it.
    let mut writers: HashMap<usize, String> = HashMap::new();
    for launch in &lowered.launches {
        let symbol = &lowered.kernels[launch.kernel as usize];
        let args = &lowered.args[launch.args.start as usize..launch.args.end as usize];
        // The trace states inputs, then OUTPUTS, then weights, and the row
        // says how many of the widthed operands are results — the same split
        // `dispatch::reorder` makes. A region that is only ever an INPUT is
        // one nothing was ever supposed to write.
        let results = kernels::sig_in(kernels_metal::KERNELS, symbol)
            .map(|sig| {
                sig.operands
                    .iter()
                    .filter_map(|o| match o.source {
                        kernels::Source::Out(i) => Some(usize::from(i) + 1),
                        _ => None,
                    })
                    .max()
                    .unwrap_or(1)
            })
            .unwrap_or(1);
        let widthed: Vec<&model_compiler::lower::Arg> = args
            .iter()
            .filter(|a| !matches!(a, model_compiler::lower::Arg::Weight(_)))
            .collect();
        let split = widthed.len().saturating_sub(results);
        for arg in widthed.iter().skip(split) {
            if let model_compiler::lower::Arg::Arena { at, .. } = arg {
                writers
                    .entry(*at)
                    .or_insert_with(|| format!("written by {symbol}"));
            }
        }
    }

    {
        let mut hist: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
        for l in &lowered.launches {
            *hist.entry(l.rows.end - l.rows.start).or_default() += 1;
        }
        eprintln!("launch rows histogram: {hist:?}");
    }
    eprintln!(
        "{} launch(es) -> {} dispatch(es)",
        lowered.launches.len(),
        plan_count(&lowered, &facts, &mut live)
    );

    let mut c = Census::default();
    let mut unwritten: Vec<String> = Vec::new();
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
        if r.finite_nonzero == 0 && writers.contains_key(at) {
            unwritten.push(format!(
                "  @{at} ({} elements x{element}): {}",
                len / element,
                writers[at]
            ));
        }
        widest_by_element.push((*element, r.max_abs));
        c.max_abs = c.max_abs.max(r.max_abs);
        eprintln!(
            "  @{at:>8} {len:>8} B x{element}: {:>7} nz, {:>7} zero, max |v| = {}{}",
            r.finite_nonzero,
            r.zero,
            r.max_abs,
            if r.finite_nonzero == 0 {
                format!(
                    "   <- NOTHING WROTE THIS ({})",
                    writers
                        .get(at)
                        .map_or("NO LAUNCH WRITES IT — read-only", String::as_str)
                )
            } else {
                String::new()
            }
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
    // Measured 648205 non-zero to 8179 zero: 99% of the arena holds a value.
    // It was 171268 to 485116 while the gather was the single-row one, which
    // is the same defect the lane count below names.
    assert!(
        c.finite_nonzero > c.zero * 10,
        "the arena is {} zero to {} non-zero. A projection told its extents \
         are zero no-ops and leaves exactly this, so a near-empty arena is a \
         fire that ran and did not compute.",
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
    // Row ZERO of it, not all four: three of the four are empty and the lane
    // count below is what tracks that. What this asks is whether the readout
    // that DID run produced a distribution.
    //
    // Exactly half zero would mean something else entirely -- a kernel writing
    // bf16 into a slot sized for f32 -- and that is a defect this gate found
    // and closed on its first run.
    let lane_bytes = (end - at) / 4;
    let r = census(&read[at..at + lane_bytes], element);
    assert!(
        r.finite_nonzero > r.zero,
        "the readout's first lane is {} zero to {} non-zero. Half zero is a \
         dtype disagreement; mostly zero is a readout that did not run.",
        r.zero,
        r.finite_nonzero
    );
    assert!(
        r.max_abs > 1e-4,
        "every logit is under 1e-4, so the readout accumulated nothing."
    );

    // ── the regions a launch declares and does not fill ──
    //
    // ZERO, down from FIVE. All five were the same defect the lane count below
    // names: the text picked the single-row `embed_gather_4bit`, so every lane
    // but the first was zero from statement zero onward, and the branches only
    // those lanes fed never held anything.
    //
    // The NUMBER is what made it findable. It said "five regions", the writer
    // attribution said which statements, and a prefix bisection
    // (`the_second_lane_stops_somewhere_and_this_says_where`) put the stop at
    // statement 0 -- three steps, each narrowing, none of them a guess.
    eprintln!("{} declared output(s) nothing filled", unwritten.len());
    assert!(
        unwritten.is_empty(),
        "{} statement(s) declare an output nothing filled. A statement whose \
         output stays zero is a branch of the forward pass that computes \
         nothing.\n{}",
        unwritten.len(),
        unwritten.join("\n")
    );

    // ── THE PER-TOKEN AXIS ──
    //
    // Four lanes decoded four different tokens, so the readout should hold
    // four different rows. It holds ONE: 128256 of 513024 values non-zero,
    // which is exactly one row of a 128256-wide vocabulary, and rows one
    // through three are zero all the way through.
    //
    // Measured 2026-08-10, and it is the largest remaining gap between this
    // executor and a model that answers. Nothing about it is a grid: every
    // launch states `rows 0..4`, `qmv_mb` puts the row on `grid.x` and
    // `qmv_fast_impl` reads it there (`y += tid.x * out_vec_size`), and the
    // dispatches come out `[128, 512, 1]` over `[32, 2, 1]` -- four
    // threadgroups on x, one per row. All 227 launches plan and none has an
    // empty grid.
    //
    // So the arithmetic is right and the rows still do not appear, which
    // means the next thing to look at is what the FIRST statement writes:
    // every later row being zero is what a gather that emitted one row looks
    // like four launches downstream. Reading between dispatches is the
    // instrument that settles it and this file does not have one yet.
    //
    // Pinned at one, and the number to want is four.
    let lanes = {
        let row = &read[at..end];
        let stride = row.len() / 4;
        (0..4)
            .filter(|i| {
                row[i * stride..(i + 1) * stride]
                    .chunks_exact(element)
                    .any(|c| c.iter().any(|&b| b != 0))
            })
            .count()
    };
    eprintln!("{lanes} of 4 readout lane(s) hold anything");
    assert_eq!(
        lanes, 4,
        "the per-token axis lost a lane: {lanes} of four readout rows hold \
         anything. A fire that answers one token for four is the failure this \
         gate exists to catch, because every magnitude check passes through it."
    );
}

/// **Where the second lane stops.**
///
/// The instrument the test above says it lacks: run the first `n` dispatches
/// of the fire and read the arena, for every `n`, and report the first prefix
/// after which no arena region holds anything in its second row.
///
/// A bisection rather than a guess. "Every later row is zero" is true of a
/// gather that emitted one row and of a projection that did, and four
/// launches downstream they look identical -- so the only thing that
/// distinguishes them is running fewer launches.
#[test]
fn the_second_lane_stops_somewhere_and_this_says_where() {
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
    let (facts, _metal) =
        driver_metal_new::model::text::facts_from(&dg, |t| loaded.tensors.contains_key(t));
    let (_, metal) = driver_metal_new::model::text::facts_from(&dg, |t| loaded.tensors.contains_key(t));

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
    let (staged, spans) = stage_tables(&context, &step, shape.page_size);

    let named = HashMap::new();
    let mut live = Live {
        store: Store::new(Names::mlx(), &loaded.tensors, &named),
        tables: Slice {
            address: staged.gpu_address(),
            bytes: staged.len(),
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

    // Every launch's OUTPUT rectangle, so a prefix can be judged by what its
    // last statement was supposed to write rather than by the whole arena.
    let outs: Vec<(usize, usize, usize, String)> = lowered
        .launches
        .iter()
        .map(|l| {
            let symbol = lowered.kernels[l.kernel as usize].clone();
            let args = &lowered.args[l.args.start as usize..l.args.end as usize];
            let last = args
                .iter()
                .rev()
                .find_map(|a| match a {
                    model_compiler::lower::Arg::Arena { at, width, bytes } => {
                        Some((*at, *width as usize, *bytes as usize))
                    }
                    _ => None,
                })
                .unwrap_or((0, 0, 0));
            (last.0, last.1, last.2, symbol)
        })
        .collect();

    // The prefixes worth running: the first twelve statements are one layer's
    // worth, which is where a per-row defect either appears or does not.
    let mut first_bad: Option<(usize, String)> = None;
    for n in 1..=12.min(lowered.launches.len()) {
        let arena = allocate(
            &context,
            (lowered.arena_bytes as u64).max(1),
            "bisect arena",
        )
        .expect("an arena");
        // SAFETY: freshly allocated.
        unsafe { arena.zero(0, arena.len()).expect("it zeroes") };
        let dispatches = driver_metal_new::model::dispatch::plan(
            &lowered,
            driver_metal_new::model::run::table(),
            driver_metal_new::model::executor::Frame {
                arena: Slice {
                    address: arena.gpu_address(),
                    bytes: arena.len(),
                },
            },
            geometry,
            &mut live,
        )
        .expect("the fire plans");
        let prefix = &dispatches[..n];
        let prepared = driver_metal_new::model::run::prepare(&context, &lowered, prefix)
            .expect("the prefix prepares");
        pipelines
            .ensure(&context, &compiler, prefix)
            .expect("the pipelines compile");
        let mut stepper = driver_metal_new::metal::Stepper::new(&context).expect("a stepper");
        stepper
            .run(|encoder| {
                driver_metal_new::model::encode::encode(
                    encoder,
                    &prepared.table,
                    &pipelines,
                    &prepared.params,
                    prefix,
                )
            })
            .expect("the prefix runs");

        let mut read = vec![0u8; arena.len() as usize];
        // SAFETY: the command buffer retired.
        unsafe {
            let raw = core::slice::from_raw_parts(
                arena.contents().as_ptr().cast_const().cast::<u8>(),
                arena.len() as usize,
            );
            read.copy_from_slice(raw);
        }

        // The nth statement's own output, row 0 against row 1.
        let (at, width, element, symbol) = &outs[n - 1];
        let row = width * element;
        let live_row = |i: usize| {
            let (a, b) = (at + i * row, (at + (i + 1) * row).min(read.len()));
            a < b && read[a..b].iter().any(|&x| x != 0)
        };
        let (r0, r1) = (live_row(0), live_row(1));
        eprintln!(
            "  [{:>2}] {symbol:<44} @{at} row0 {} row1 {}",
            n - 1,
            if r0 { "yes" } else { "NO " },
            if r1 { "yes" } else { "NO " },
        );
        if r0 && !r1 && first_bad.is_none() {
            first_bad = Some((n - 1, symbol.clone()));
        }
    }

    match &first_bad {
        Some((i, symbol)) => eprintln!(
            "\nThe second lane stops at statement {i}, `{symbol}`: it wrote row 0 \
             and not row 1."
        ),
        None => eprintln!("\nEvery statement in the first layer wrote both rows."),
    }
}
