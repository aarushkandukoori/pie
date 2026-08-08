//! THE HOST ASSIGNS BUFFERS. This is the test that makes that safe.
//!
//! Two allocators existed. `Buffers::assign` here, over the traced plan,
//! and `declared::ValueArena` in the CUDA driver, over the same plan at
//! fire time. Keeping both means they must agree byte-for-byte forever,
//! and they already do not: the driver's copy predates `Select`, the
//! `kernel!` in-place table and `Dim::MoeAlignedRoutes`, so on a text
//! using any of the three it would size or place a value differently —
//! silently, because an allocator that hands back a plausible pointer
//! reports nothing.
//!
//! So the host wins and the driver stops allocating: a rectangle's
//! operand already crosses as `Arg::Arena { at, width }`, which is an
//! address, and `Lowered::arena_bytes` is the block it must fit. That
//! makes this file the load-bearing one — a driver that only adds `at`
//! to a base pointer cannot notice an assignment that overlaps, so the
//! overlap has to be impossible HERE.
//!
//! The check is a write trace. Walk the ops in order; stamp each
//! output's byte range with its value; before an op reads an input,
//! demand that the input's range still carries its own stamp. A value
//! placed over a buffer somebody still reads shows up as a stamp that
//! changed underneath its reader.
//!
//! ALIASING is the wrinkle, and it is intended rather than accidental in
//! exactly two places: a `Select` output IS a window of its operand, and
//! an in-place launcher's output IS the operand it accumulates into.
//! Both are unions, so the trace stamps by the union's ROOT and the
//! intended sharing passes while an accidental one still fails.

use model_compiler::lower::{value_bytes, Buffers, Row};
use model_compiler::trace::{FireClass, ForwardPlan, OpKind, ValueId};

/// A decode-shaped fire: every row samples, so the epilogue's row space
/// is the full row count.
fn plain(n: usize) -> Vec<Row> {
    vec![
        Row {
            samples: true,
            ..Row::default()
        };
        n
    ]
}

/// Union-find over the value ids two ops are allowed to share bytes.
struct Alias(Vec<ValueId>);

impl Alias {
    fn new(n: usize) -> Alias {
        Alias((0..n as ValueId).collect())
    }

    fn root(&mut self, v: ValueId) -> ValueId {
        let mut v = v;
        while self.0[v as usize] != v {
            let up = self.0[v as usize];
            self.0[v as usize] = self.0[up as usize];
            v = self.0[v as usize];
        }
        v
    }

    fn join(&mut self, a: ValueId, b: ValueId) {
        let (ra, rb) = (self.root(a), self.root(b));
        if ra != rb {
            self.0[rb as usize] = ra;
        }
    }
}

/// The two constructs whose whole meaning is that the output shares the
/// input's bytes. Read off the plan, not listed by hand — a third one
/// added to `Buffers::assign` without being added here fails loudly,
/// which is the right way round.
fn aliases(plan: &ForwardPlan) -> Alias {
    let mut alias = Alias::new(plan.values.len());
    for op in &plan.ops {
        match &op.kind {
            OpKind::Select { .. } => {
                if let (Some(&src), Some(&out)) = (op.inputs.first(), op.outputs.first()) {
                    alias.join(src, out);
                }
            }
            OpKind::Launch { kernel, .. } => {
                let Some(idx) = model_compiler::kernels::in_place_operand(plan, kernel) else {
                    continue;
                };
                if let (Some(&src), Some(&out)) =
                    (op.inputs.get(idx as usize), op.outputs.first())
                {
                    alias.join(src, out);
                }
            }
            _ => {}
        }
    }
    alias
}

fn first_clobber(plan: &ForwardPlan, rows: &[Row]) -> Option<String> {
    walk(plan, rows, &Buffers::assign(plan, rows))
}

/// Walks one assignment and returns the first place a reader's bytes had
/// been taken from under it.
///
/// Takes the assignment rather than computing it, so the negative
/// control can hand it a deliberately broken one.
fn walk(plan: &ForwardPlan, rows: &[Row], buffers: &Buffers) -> Option<String> {
    let n_tokens = rows.len();
    let n_requests = rows
        .iter()
        .filter(|r| !r.multi_token)
        .count()
        .max(rows.iter().filter(|r| r.samples).count())
        .max(1);

    let mut alias = aliases(plan);

    // One stamp per arena byte: which value's ROOT owns it right now.
    const FREE: ValueId = ValueId::MAX;
    let mut owner = vec![FREE; buffers.bytes];

    let extent = |v: ValueId| -> Option<(usize, usize)> {
        let at = *buffers.offset.get(v as usize)?;
        if at == Buffers::NAMED {
            return None; // the backend binds it; not the arena's bytes
        }
        Some((at, value_bytes(plan, v, n_tokens, n_requests)))
    };

    for (i, op) in plan.ops.iter().enumerate() {
        for &v in &op.inputs {
            let Some((at, len)) = extent(v) else { continue };
            let want = alias.root(v);
            for b in at..(at + len).min(owner.len()) {
                if owner[b] != want {
                    return Some(format!(
                        "op {i} ({:?}) reads value {v} at [{at}, {}), but byte \
                         {b} now belongs to value {} — the arena placed it \
                         over a buffer this op still reads",
                        op.kind,
                        at + len,
                        owner[b]
                    ));
                }
            }
        }
        for &v in &op.outputs {
            let Some((at, len)) = extent(v) else { continue };
            let root = alias.root(v);
            for b in at..(at + len).min(owner.len()) {
                owner[b] = root;
            }
        }
    }
    None
}

fn families() -> Vec<(&'static str, FireClass, ForwardPlan)> {
    use model::*;
    let mut out: Vec<(&'static str, FireClass, ForwardPlan)> = Vec::new();
    for f in [FireClass::Decode, FireClass::Prefill] {
        out.push((
            "llama_like",
            f,
            families::llama_like::forward::llama_like_cuda(
                &families::llama_like::forward::facts::LlamaLikeFacts::qwen3_0_6b(),
                &families::llama_like::forward::facts::LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
                f,
            ),
        ));
    }
    // The DRIVEN families, both classes. These matter most and were the
    // last to be swept: a declared-only family whose assignment overlaps
    // has nothing to corrupt yet, while these three are executing.
    for f in [FireClass::Decode, FireClass::Prefill] {
        out.push((
            "gemma_4",
            f,
            gemma_4::forward::gemma4_cuda(
                &gemma_4::forward::facts::Gemma4Facts::gemma_4_e4b(),
                &gemma_4::forward::facts::Gemma4CudaFacts::gemma_4_e4b_synthetic(),
                f,
            ),
        ));
        out.push((
            "gpt_oss",
            f,
            gpt_oss::forward::gpt_oss_cuda(
                &gpt_oss::forward::facts::GptOssFacts::gpt_oss_20b(),
                &gpt_oss::forward::facts::GptOssCudaFacts::gpt_oss_20b_synthetic(),
                f,
            ),
        ));
        out.push((
            "qwen3_5",
            f,
            qwen_3_5::forward::qwen3_5_hybrid_cuda(
                &qwen_3_5::forward::facts::Qwen35HybridFacts::qwen3_5_0_8b(),
                &qwen_3_5::forward::facts::Qwen35CudaFacts::qwen3_5_0_8b_synthetic(),
                f,
            ),
        ));
    }

    let d = FireClass::Decode;
    out.push((
        "glm5",
        d,
        glm5::forward::glm5_cuda(&glm5::forward::facts::Glm5Facts::glm5_106b_a12b(), d),
    ));
    out.push((
        "kimi_k2",
        d,
        kimi_k2::forward::kimi_cuda(
            &kimi_k2::forward::facts::KimiFacts::kimi_k2(),
            &kimi_k2::forward::facts::KimiCudaFacts::kimi_k2_synthetic(),
            d,
        ),
    ));
    out.push((
        "kimi_k3",
        d,
        kimi_k3::forward::kimi_k3_cuda(&kimi_k3::forward::facts::KimiK3Facts::kimi_k3_synthetic(), d),
    ));
    out.push((
        "deepseek_v4",
        d,
        deepseek_v4::forward::dsv4_cuda(&deepseek_v4::forward::facts::Dsv4Facts::dsv4_synthetic(), d),
    ));
    out.push((
        "nemotron_h",
        d,
        nemotron_h::forward::nemotron_h_cuda(
            &nemotron_h::forward::facts::NemotronHFacts::nemotron_h_synthetic(),
            d,
        ),
    ));
    out.push((
        "gemma3n",
        d,
        gemma3n::forward::gemma3n_cuda(&gemma3n::forward::facts::Gemma3nFacts::gemma3n_synthetic(), d),
    ));
    out.push((
        "gemma_2",
        d,
        gemma_2::forward::gemma2_cuda(&gemma_2::forward::facts::Gemma2Facts::gemma_2_9b(), d),
    ));
    out
}

/// The invariant, over every declared family: nothing the arena hands
/// out lands on bytes a later op still reads.
///
/// Row counts are chosen to move the two extents independently — 1 is
/// the decode fire, 8 is the batched one, and a fire whose rows are
/// sampled separates `Dim::Requests` from `Dim::Tokens`, which is where
/// an epilogue value would be under-sized.
#[test]
fn no_value_lands_on_bytes_a_later_op_still_reads() {
    for (name, class, plan) in families() {
        for n in [1usize, 8] {
            let mut rows = plain(n);
            rows[0].samples = true;
            if let Some(why) = first_clobber(&plan, &rows) {
                panic!("{name} ({class:?}), {n} rows: {why}");
            }
        }
    }
}

/// A value the arena placed must FIT the arena it reports, and the
/// report is what sizes the driver's block.
///
/// Separate from the clobber walk because it fails differently: an
/// out-of-range offset is a driver segfault, not a wrong number, and it
/// would be invisible above (the walk clamps to the block it was given).
#[test]
fn every_placed_value_fits_the_arena_it_reports() {
    for (name, class, plan) in families() {
        let rows = plain(8);
        let buffers = Buffers::assign(&plan, &rows);
        let n_requests = 1usize.max(rows.len());
        // The block the driver's workspace must hold for this family,
        // printed because nothing else states it: `ws.declared_values`
        // is sized by a formula today, and this is the number it has to
        // cover once the host is the one assigning.
        // Placed vs NAMED is the shape of the remaining driver work, not
        // just a statistic. A value the host PLACES is one whose arm has
        // to stop naming a workspace field; a value it leaves NAMED stays
        // exactly where it is, because a seam exposes it and machinery
        // outside the walk reaches it by name. The four executors between
        // them name about twelve buffer roles (`ws.y`, `ws.norm_x`,
        // `ws.q`, …), so the migration is counted in roles, and this says
        // how many of those roles the host is even asking about.
        let named = buffers
            .offset
            .iter()
            .filter(|&&at| at == Buffers::NAMED)
            .count();
        println!(
            "{name:12} {class:?}  arena {:>9} bytes  {} values ({named} named, {} placed)",
            buffers.bytes,
            plan.values.len(),
            plan.values.len() - named
        );
        for v in 0..plan.values.len() {
            let at = buffers.offset[v];
            if at == Buffers::NAMED {
                continue;
            }
            let len = value_bytes(&plan, v as ValueId, rows.len(), n_requests);
            assert!(
                at + len <= buffers.bytes,
                "{name} ({class:?}): value {v} at [{at}, {}) past the \
                 reported arena of {} bytes",
                at + len,
                buffers.bytes
            );
        }
    }
}

/// The sanity check on the check: a deliberately broken assignment must
/// be caught. Without this, a `first_clobber` that silently walked zero
/// ops would read as every family being sound.
#[test]
fn the_walk_catches_an_overlap_it_is_given() {
    let plan = families()
        .into_iter()
        .find(|(n, _, _)| *n == "gemma_2")
        .map(|(_, _, p)| p)
        .expect("gemma_2 declares a decode text");
    let rows = plain(8);
    assert!(
        first_clobber(&plan, &rows).is_none(),
        "the family must be sound before the negative control means anything"
    );

    // Collapse the arena to ONE buffer — every value at offset 0, which
    // is the crudest possible wrong assignment. The same walk must now
    // report a clobber, and if it does not, its silence on the real
    // assignments above means nothing.
    let broken = {
        let mut b = Buffers::assign(&plan, &rows);
        for at in b.offset.iter_mut() {
            if *at != Buffers::NAMED {
                *at = 0;
            }
        }
        b
    };
    assert!(
        walk(&plan, &rows, &broken).is_some(),
        "an arena that puts every value at offset 0 must clobber"
    );
}
