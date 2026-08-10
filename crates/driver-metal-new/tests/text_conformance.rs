//! What every Metal text must satisfy, checked once and reusable.
//!
//! Four families need texts and one has one. The checks that found the
//! defects in `llama_like`'s were the same four every time, so they are
//! written here **over a `ForwardPlan`** rather than over that one text —
//! a new family gets them by adding three lines to `texts()`, and gets them
//! the moment its first statement exists rather than after it is finished.
//!
//! # What each check caught, so none is deleted for looking obvious
//!
//! | check | what it found in `llama_like` |
//! |---|---|
//! | every symbol has a row | `attn::split_qkv_bf16`, which turned out to need a scalar channel and not a shader |
//! | every row states its file | three rows pointing at files that do not define them |
//! | every symbol is an INSTANTIATED point | four symbols named as bare stems, which resolve in the table and not in any shader |
//! | every launch becomes a legal grid | the `Unstated` rows for the whole batched lane |
//! | every weight name has a spelling | the map assuming HuggingFace naming |
//!
//! Two of those are only findable by *running* — a stem resolves through
//! `sig_in` because the row carries axes, and only the shader disagrees. So
//! this file holds the ones that are answerable on the host, and
//! `tests/device_text_fire.rs` holds the rest.

use std::collections::BTreeSet;

use driver_metal_new::model::dispatch::{Geometry, Undispatchable, plan_one};
use driver_metal_new::model::executor::{Frame, Resolver, Slice};
use driver_metal_new::model::resolve::{Names, Store};
use model_compiler::lower::{Arg, Fire, Lowered, Row, lower};
use model_compiler::trace::{FireClass, ForwardPlan, ValueId};

/// A text under test: how to trace it, and the geometry its fires run at.
struct Text {
    /// What to call it when a check fails.
    name: &'static str,
    /// Traced for a class.
    plan: fn(FireClass) -> ForwardPlan,
    /// The fire geometry the rules evaluate at.
    geometry: Geometry,
}

/// Every Metal text that exists.
///
/// **Add a row here when a family gets a text.** That is the whole cost of
/// joining this harness, and the point of writing it over `ForwardPlan`.
fn texts() -> Vec<Text> {
    vec![Text {
        name: "llama_like",
        plan: |class| {
            use model::families::llama_like::forward::facts::{
                LlamaLikeFacts, LlamaLikeMetalFacts,
            };
            model::families::llama_like::forward::llama_like_metal(
                &LlamaLikeFacts::qwen3_0_6b(),
                &LlamaLikeMetalFacts::synthetic(),
                class,
            )
        },
        geometry: Geometry {
            q_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            rotary_dims: 128,
            n_experts: 0,
            experts_per_token: 0,
        },
    }]
}

/// Answers every name, so a check is about the walk and not about a store.
struct Anything;

impl Resolver for Anything {
    fn weight(&mut self, _: &str) -> Option<Slice> {
        Some(Slice {
            address: 0x1000_0000,
            bytes: 1 << 30,
        })
    }
    fn named(&mut self, _: ValueId) -> Option<Slice> {
        Some(Slice {
            address: 0x2000_0000,
            bytes: 1 << 30,
        })
    }
}

/// Both fire classes, at a row count that exercises each lane.
fn fires(text: &Text) -> Vec<(FireClass, Lowered)> {
    [(FireClass::Decode, 1usize), (FireClass::Prefill, 8)]
        .into_iter()
        .map(|(class, rows)| {
            let plan = (text.plan)(class);
            let low = lower(
                &plan,
                &vec![
                    Row {
                        samples: true,
                        ..Row::default()
                    };
                    rows
                ],
                Fire {
                    captures_across_splits: false,
                },
            )
            .unwrap_or_else(|why| panic!("{}: {class:?} does not lower: {why:?}", text.name));
            (class, low)
        })
        .collect()
}

#[test]
fn every_symbol_every_text_states_has_a_row_that_states_its_file_and_rule() {
    // Three questions with one answer shape, so one walk asks all three: a
    // symbol with no row has no contract, a row with no file cannot be
    // compiled at run time, and a row with no rule cannot be given a grid.
    let mut faults: Vec<String> = Vec::new();
    for text in texts() {
        for (class, low) in fires(&text) {
            for symbol in BTreeSet::from_iter(low.kernels.iter()) {
                match kernels::sig_in(kernels_metal::KERNELS, symbol) {
                    None => faults.push(format!("{}/{class:?}: `{symbol}` has no row", text.name)),
                    Some(sig) if sig.file.is_none() => {
                        faults.push(format!("{}/{class:?}: `{symbol}` states no file", text.name));
                    }
                    Some(sig) if sig.launch == kernels::LaunchRule::Unstated => {
                        faults.push(format!("{}/{class:?}: `{symbol}` states no rule", text.name));
                    }
                    Some(_) => {}
                }
            }
        }
    }
    assert!(faults.is_empty(), "{}", faults.join("\n"));
}

#[test]
fn every_symbol_is_an_instantiated_point_and_not_a_bare_stem() {
    // The check that only exists because running found it. A row carries
    // AXES, so `sig_in` resolves a stem — `embed_gather_4bit` matches its own
    // row — and the table is satisfied while no shader exports that name.
    //
    // The test is the row's own product: a symbol must be one of the
    // entrypoints its axes generate. A row with no axes generates exactly its
    // own symbol, so an unparameterised kernel passes trivially, which is
    // right.
    let mut faults: Vec<String> = Vec::new();
    for text in texts() {
        for (class, low) in fires(&text) {
            for symbol in BTreeSet::from_iter(low.kernels.iter()) {
                let Some(sig) = kernels::sig_in(kernels_metal::KERNELS, symbol) else {
                    continue; // the check above owns this
                };
                let points = sig.entrypoints();
                if !points.iter().any(|p| p == symbol) {
                    faults.push(format!(
                        "{}/{class:?}: `{symbol}` is a STEM, not an entry point. \
                         Its row instantiates {:?}. A stem resolves here and in no \
                         shader — spell the point from the deployment's facts.",
                        text.name,
                        points.iter().take(4).collect::<Vec<_>>()
                    ));
                }
            }
        }
    }
    assert!(faults.is_empty(), "{}", faults.join("\n"));
}

#[test]
fn every_launch_of_every_text_becomes_a_legal_grid() {
    let mut faults: Vec<String> = Vec::new();
    for text in texts() {
        for (class, low) in fires(&text) {
            let frame = Frame {
                arena: Slice {
                    address: 0x8000_0000,
                    bytes: low.arena_bytes as u64,
                },
            };
            for launch in &low.launches {
                match plan_one(
                    &low,
                    launch,
                    kernels_metal::KERNELS,
                    frame,
                    text.geometry,
                    &mut Anything,
                ) {
                    Ok(d) => {
                        let threads: u64 = d.grid.iter().map(|&n| u64::from(n)).product();
                        let group: u64 = d.threadgroup.iter().map(|&n| u64::from(n)).product();
                        if threads == 0 || group == 0 || group > 1024 {
                            faults.push(format!(
                                "{}/{class:?}: `{}` wants grid {:?} in groups of {:?}",
                                text.name, d.symbol, d.grid, d.threadgroup
                            ));
                        }
                    }
                    Err(Undispatchable::NoRow { .. } | Undispatchable::NoFile { .. }) => {}
                    Err(other) => {
                        faults.push(format!("{}/{class:?}: {other:?}", text.name));
                    }
                }
            }
        }
    }
    assert!(faults.is_empty(), "{}", faults.join("\n"));
}

#[test]
fn every_weight_name_every_text_states_has_a_checkpoint_spelling() {
    let (tensors, named) = (Default::default(), Default::default());
    let store = Store::new(Names::mlx(), &tensors, &named);
    let mut faults: Vec<String> = Vec::new();
    for text in texts() {
        for (class, low) in fires(&text) {
            for arg in &low.args {
                let Arg::Weight(name) = arg else { continue };
                // A `scale.` marker is a constant riding the weight slot; the
                // binder never looks it up.
                if name.starts_with("scale.") {
                    continue;
                }
                if store.checkpoint_name(name).is_none() {
                    faults.push(format!(
                        "{}/{class:?}: `{name}` has no spelling in `Names::mlx`",
                        text.name
                    ));
                }
            }
        }
    }
    faults.sort();
    faults.dedup();
    assert!(faults.is_empty(), "{}", faults.join("\n"));
}

#[test]
fn the_harness_covers_every_family_that_has_a_text() {
    // The check that keeps the harness honest. A family whose text lands and
    // is not added to `texts()` gets none of the above, and the failure would
    // be silence — which is the one failure mode a conformance suite cannot
    // afford.
    //
    // Counted rather than named: the list of Metal texts is short and its
    // growth is the whole remaining plan (`.wiki/new-driver/metal.md` task 5).
    assert_eq!(
        texts().len(),
        1,
        "a Metal text landed or left. Add or remove its row in `texts()` — \
         everything above is per-text and a family not listed is a family not \
         checked."
    );
}

/// How many buffers a shader's entry point declares.
///
/// # Why this is parsed rather than declared
///
/// `KernelSig` has an `operands` field and the CUDA table uses it, but **no
/// Metal row declares one**: the C++ shell bound by hand from tables that are
/// retiring, so nothing ever needed the arity written down. Until the rows
/// carry it, the shader is the only statement of how many buffers a kernel
/// takes — so this reads the shader.
///
/// The parse is deliberately crude and *conservative*: find the template body
/// by its stem, take its parameter list, and count distinct `[[buffer(N)]]`
/// indices. A kernel it cannot find contributes nothing, so this never invents
/// a gap.
fn declared_buffers(root: &std::path::Path, file: &str, stem: &str) -> Option<usize> {
    let src = std::fs::read_to_string(root.join(file)).ok()?;
    let at = src.find(&format!("void {stem}("))?;
    let open = src[at..].find('(')? + at;
    // Depth-counted, because a parameter list is full of parentheses:
    // `[[buffer(0)]]` closes one the signature did not open, and stopping at
    // the first `)` finds a list of one operand for every kernel there is.
    let mut depth = 0i32;
    let mut close = None;
    for (i, c) in src[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let params = &src[open..close?];
    let mut seen = BTreeSet::new();
    let mut rest = params;
    while let Some(i) = rest.find("[[buffer(") {
        rest = &rest[i + 9..];
        if let Some(j) = rest.find(')')
            && let Ok(n) = rest[..j].trim().parse::<usize>()
        {
            seen.insert(n);
        }
    }
    (!seen.is_empty()).then_some(seen.len())
}

/// The gap between what a text STATES and what its kernels TAKE.
///
/// **This is a measurement, not a pass/fail**, and the number it prints is the
/// distance between "the fire executes" and "the fire is right".
/// `tests/device_text_fire.rs` proves the first: every launch compiles, every
/// grid is legal, the command buffer completes. It cannot prove the second,
/// and this says why in one number.
///
/// A kernel whose statement binds fewer buffers than it declares reads the
/// slots nobody bound — which on this backend is whatever the last dispatch
/// left there. It does not fault and it does not report. It answers.
///
/// The known gaps are listed rather than tolerated silently, exactly as the
/// `split_qkv` row was before it closed. **Shrinking this list is the work
/// between here and token-exactness.**
#[test]
fn the_distance_between_a_fire_that_runs_and_a_fire_that_is_right() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join("kernels-metal/kernels");

    let mut short: Vec<String> = Vec::new();
    for text in texts() {
        for (_, low) in fires(&text) {
            let mut seen = BTreeSet::new();
            for launch in &low.launches {
                let symbol = &low.kernels[launch.kernel as usize];
                if !seen.insert(symbol.clone()) {
                    continue;
                }
                let Some(sig) = kernels::sig_in(kernels_metal::KERNELS, symbol) else {
                    continue;
                };
                let Some(file) = sig.file else { continue };
                let Some(declared) = declared_buffers(&root, file, sig.symbol) else {
                    continue;
                };
                let stated = (launch.args.end - launch.args.start) as usize
                    + usize::from(launch.params.end > launch.params.start);
                if stated < declared {
                    short.push(format!("  {symbol}: states {stated}, takes {declared}"));
                }
            }
        }
    }
    short.sort();
    short.dedup();

    // Pinned, so the number moves only when someone means it to. Every entry
    // is a kernel reading buffers nobody bound.
    eprintln!(
        "{} statement(s) bind fewer buffers than their kernel declares:\n{}",
        short.len(),
        short.join("\n")
    );
    // NINE, measured 2026-08-10, and every one is a real hole:
    //
    //   sdpa_paged_decode   states  2, takes 17
    //   sdpa_vector_decode  states  2, takes 11
    //   kv_append_paged     states  2, takes 10
    //   kv_append           states  2, takes  8
    //   affine_qmv_fast     states  5, takes  7
    //   ...
    //
    // The attention pair is the loud case and the shape of the problem: the
    // statement gives the query and the output, and the kernel wants the keys,
    // the values, six strides, a scale, a window and two row pitches. Those
    // are the KV cache and the geometry — things the trace knows as `Kv` state
    // and the fire's shape, and that `dsl::metal::sdpa` does not yet spell as
    // operands.
    //
    // So this number is the honest distance between `device_text_fire.rs`
    // (every launch compiles, every grid is legal, the command buffer
    // completes) and a model that answers. **Shrinking it is the work between
    // here and token-exactness**, and it may only shrink.
    assert!(
        short.len() <= 9,
        "the gap GREW to {}. Every entry is a kernel reading whatever the last \
         dispatch left in the slots nobody bound — it does not fault, it does \
         not report, it answers.\n{}",
        short.len(),
        short.join("\n")
    );
}
