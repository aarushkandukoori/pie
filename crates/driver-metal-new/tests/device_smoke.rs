//! The first fired token: the whole chain against a real checkpoint.
//!
//! `config.json` → descriptor → facts → geometry → load plan → staged
//! storage → scratch schedule → PSO table → bound step → one dispatch walk
//! on the GPU. Nothing here checks the ANSWER yet — that is the accuracy
//! gate's job (golden taps, token-exact decode) — this pins that the
//! assembly holds together: every weight the DAG asks for was staged,
//! every constant bound, every pipeline compiled, and the command buffer
//! retires.
//!
//! Gated on `PIE_METAL_SMOKE_CHECKPOINT` naming a qwen3.5/3.6-family MLX
//! snapshot directory, because a checkpoint is a machine's, not the
//! repo's. Without it the test states it skipped and why.

#![cfg(target_vendor = "apple")]

use std::path::PathBuf;

use driver_metal_new::batch::{
    AffineFormat, DagOptions, EntryNames, IoSlot, PsoFeatures, build_decode_dag,
    build_scratch_schedule, geometry_from_facts, plan_decode_psos, scratch_slot_elems,
};
use driver_metal_new::facts::ModelFacts;
use driver_metal_new::loader::{compile_load_plan, metal_storage_target};
use driver_metal_new::metal::Compiler;
use driver_metal_new::metal::{Context, DecodeStep, Stepper, load_step_psos, stage_decode_storage};
use driver_metal_new::region::Region as _;
use driver_metal_new::tuning::Tuning;

fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join("kernels-metal/kernels")
}

#[test]
fn the_assembly_fires_one_token_end_to_end() {
    let Some(snapshot) = std::env::var_os("PIE_METAL_SMOKE_CHECKPOINT") else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to a qwen3.5-family MLX snapshot");
        return;
    };
    let snapshot = PathBuf::from(snapshot);
    let config = std::fs::read_to_string(snapshot.join("config.json"))
        .expect("the snapshot has a config.json");
    let root: serde_json::Value = serde_json::from_str(&config).expect("config.json parses");
    let descriptor = model::config::descriptor(&root, snapshot.to_str().expect("utf8 path"))
        .expect("the config converts to a descriptor");
    let descriptor_json = descriptor.to_string();

    // Facts and geometry, refused rather than defaulted.
    let facts = ModelFacts::from_descriptor(&descriptor_json)
        .expect("the driver's facts read the descriptor");
    let mut geometry = geometry_from_facts(&facts).expect("the config describes this family");
    geometry.quant = AffineFormat {
        bits: u32::try_from(facts.quant_bits).unwrap_or(4),
        group: u32::try_from(facts.quant_group_size).unwrap_or(64),
    };
    eprintln!(
        "geometry: {} layers, hidden {}, vocab {}, moe={}",
        geometry.n_layers,
        geometry.hidden,
        geometry.vocab,
        geometry.is_moe()
    );

    // The load plan, authored in-process.
    let target = metal_storage_target();
    let (plan, _moe) = compile_load_plan(&snapshot, &target, &descriptor_json)
        .expect("the plan compiles and its files exist");

    // Device side: stage, schedule, compile, bind.
    let context = Context::new().expect("a Metal device answers");
    let tuning = Tuning::default();
    let max_ctx = 4096u32;
    let scratch_bytes = scratch_slot_elems(&geometry, &tuning, 1) * 2;
    let storage = stage_decode_storage(
        &context,
        &plan,
        &snapshot,
        &geometry,
        max_ctx,
        scratch_bytes,
    )
    .expect("every region allocates and every tensor stages");
    eprintln!("staged {} weights", storage.weights.len());

    // Staging integrity: the arena-offset map is new code, and a wrong
    // offset is a fluent model with the wrong weights — the exact symptom
    // nothing downstream can diagnose. Re-run the plan on the host and
    // hold a sample of staged slices to the executor's own bytes.
    {
        let host = model_loader::executor::host::execute_plan(&plan, &snapshot)
            .expect("the host executor agrees to run the plan twice");
        let mut checked = 0usize;
        for (name, bytes) in host.tensors.iter().take(4096) {
            let Some(slice) = storage.weights.get(name) else {
                continue;
            };
            if slice.len() != bytes.len() as u64 {
                panic!(
                    "{name}: staged {} bytes, executor produced {}",
                    slice.len(),
                    bytes.len()
                );
            }
            // SAFETY: nothing is encoded yet.
            let staged = unsafe {
                std::slice::from_raw_parts(slice.contents().cast::<u8>().as_ptr(), bytes.len())
            };
            assert_eq!(
                staged,
                &bytes[..],
                "{name}: the staged bytes drifted from the plan's"
            );
            checked += 1;
        }
        eprintln!("staging verified for {checked} tensors");
        assert!(checked > 0, "the probe compared nothing");
    }

    let options = DagOptions {
        with_argmax: true,
        ..DagOptions::default()
    };
    let dag = build_decode_dag(&geometry, &tuning, options);
    let schedule = build_scratch_schedule(&dag, false).expect("the DAG schedules hazard-free");

    let features = PsoFeatures {
        argmax: true,
        gdn: true,
        gated_attention: true,
        sdpa_d256: geometry.head_dim == 256,
        routed: geometry.is_moe(),
        untied: !geometry.tied_embeddings,
        ..PsoFeatures::default()
    };
    let pso_plan = plan_decode_psos(&EntryNames::bf16_g64_b4(), features);
    let compiler = Compiler::new(&context).expect("the shader compiler starts");
    let psos = load_step_psos(&compiler, &context, &kernels_dir(), &pso_plan)
        .expect("every planned entrypoint compiles");

    let step = DecodeStep::prepare(
        &context, &storage, &geometry, &tuning, options, &schedule, psos, max_ctx,
    )
    .expect("the step binds whole");

    // Fire the checkpoint's own <bos> at position 0. SeqLen is position+1.
    // A multimodal wrapper keeps its text facts one level down.
    let bos: u32 = [&root, root.get("text_config").unwrap_or(&root)]
        .iter()
        .find_map(|level| level.get("bos_token_id"))
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .expect("the config states its bos");
    let io = |slot: IoSlot| storage.io[slot as usize].as_ref().expect("io slot");
    // SAFETY: nothing is encoded yet; the buffers are host-owned.
    unsafe {
        io(IoSlot::TokenId).write(0, &bos.to_le_bytes()).unwrap();
        io(IoSlot::Position).write(0, &0u32.to_le_bytes()).unwrap();
        io(IoSlot::SeqLen).write(0, &1u32.to_le_bytes()).unwrap();
    }

    let mut stepper = Stepper::new(&context).expect("a stepper");
    let timing = step.fire(&mut stepper).expect("the command buffer retires");
    eprintln!("fired: encode {:?}, gpu {:?}", timing.encode, timing.gpu);

    // The first answer check: the logits must be finite, non-degenerate
    // numbers and the argmax a real token. "Token 0 forever" and "all
    // zeros" are this family's two historical silent failures; both are
    // visible from here without a reference.
    let logits = io(IoSlot::Logits);
    let vocab = geometry.vocab as usize;
    // SAFETY: the step retired; the GPU is done with the pool.
    let bytes =
        unsafe { std::slice::from_raw_parts(logits.contents().cast::<u8>().as_ptr(), vocab * 2) };
    let mut finite = 0usize;
    let mut nonzero = 0usize;
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, pair) in bytes.chunks_exact(2).enumerate() {
        let value = f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
        if value.is_finite() {
            finite += 1;
            if value != 0.0 {
                nonzero += 1;
            }
            if value > best.1 {
                best = (i, value);
            }
        }
    }
    let next = {
        // SAFETY: as above.
        let raw = unsafe {
            std::slice::from_raw_parts(io(IoSlot::NextToken).contents().cast::<u8>().as_ptr(), 4)
        };
        u32::from_le_bytes(raw.try_into().unwrap())
    };
    eprintln!(
        "logits: {finite}/{vocab} finite, {nonzero} nonzero; host argmax {} ({:.3}); device argmax {next}",
        best.0, best.1
    );
    assert_eq!(
        finite, vocab,
        "a NaN in the logits is a wrong kernel upstream"
    );
    assert!(
        nonzero > vocab / 2,
        "logits mostly zero: the head never ran or wrote elsewhere"
    );
    assert_eq!(
        next as usize, best.0,
        "the device argmax must agree with the host's read of the same logits"
    );

    // ── Multi-step decode: feed the argmax back and keep the GDN's
    // ping-pong honest. Step i reads what i-1 wrote, so the conv binds
    // swap by the slot's own parity — the counter is the ported
    // LinearStateSlots, so this exercises the same bookkeeping the
    // executor will use. ──
    let mut slots = driver_metal_new::store::LinearStateSlots::new(1);
    slots.step(0).unwrap(); // the <bos> fire above was step 0
    let mut step = step;
    let mut token = next;
    let mut sequence = vec![bos, token];
    for position in 1..12u32 {
        // SAFETY: the previous step retired; the buffers are host-owned
        // between steps.
        unsafe {
            io(IoSlot::TokenId).write(0, &token.to_le_bytes()).unwrap();
            io(IoSlot::Position)
                .write(0, &position.to_le_bytes())
                .unwrap();
            io(IoSlot::SeqLen)
                .write(0, &(position + 1).to_le_bytes())
                .unwrap();
        }
        step.set_gdn_parity(&context, &storage, slots.parity(0).unwrap())
            .expect("the parity rebind holds");
        step.fire(&mut stepper).expect("the step retires");
        slots.step(0).unwrap();
        // SAFETY: retired, as above.
        let raw = unsafe {
            std::slice::from_raw_parts(io(IoSlot::NextToken).contents().cast::<u8>().as_ptr(), 4)
        };
        token = u32::from_le_bytes(raw.try_into().unwrap());
        sequence.push(token);
    }
    eprintln!("greedy sequence: {sequence:?}");
    let distinct: std::collections::HashSet<_> = sequence.iter().collect();
    assert!(
        distinct.len() > 2,
        "a decode stuck on one token is this family's classic silent failure: {sequence:?}"
    );
}

// ── The bisect: dump every tap of one <bos> step and hold the head of the
// chain to host-computed values. The first tap that disagrees names the
// broken kernel; everything before it is exonerated. ──

fn read_npy(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|_| panic!("{} missing", path.display()));
    let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    bytes[10 + len..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Dequantize one row of an affine g64/b4 tensor from its staged triplet.
fn dequant_row(w: &[u8], scales: &[u8], biases: &[u8], row: usize, k: usize) -> Vec<f32> {
    let groups = k / 64;
    let mut out = Vec::with_capacity(k);
    for g in 0..groups {
        let scale = bf16(u16::from_le_bytes([
            scales[(row * groups + g) * 2],
            scales[(row * groups + g) * 2 + 1],
        ]));
        let bias = bf16(u16::from_le_bytes([
            biases[(row * groups + g) * 2],
            biases[(row * groups + g) * 2 + 1],
        ]));
        for i in 0..64 {
            let at = row * k / 2 + (g * 64 + i) / 2;
            let code = if i % 2 == 0 { w[at] & 0xf } else { w[at] >> 4 };
            out.push(f32::from(code) * scale + bias);
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-20)
}

#[test]
fn the_first_step_taps_agree_with_the_host() {
    let Some(snapshot) = std::env::var_os("PIE_METAL_SMOKE_CHECKPOINT") else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT");
        return;
    };
    let snapshot = std::path::PathBuf::from(snapshot);
    let config = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let root: serde_json::Value = serde_json::from_str(&config).unwrap();
    let descriptor = model::config::descriptor(&root, snapshot.to_str().unwrap()).unwrap();
    let descriptor_json = descriptor.to_string();
    let facts = ModelFacts::from_descriptor(&descriptor_json).unwrap();
    let mut geometry = geometry_from_facts(&facts).unwrap();
    geometry.quant = AffineFormat { bits: 4, group: 64 };
    let target = metal_storage_target();
    let (plan, _) = compile_load_plan(&snapshot, &target, &descriptor_json).unwrap();
    let context = Context::new().unwrap();
    let tuning = Tuning::default();
    let max_ctx = 4096u32;
    let slot_bytes = scratch_slot_elems(&geometry, &tuning, 1) * 2;
    let mut storage =
        stage_decode_storage(&context, &plan, &snapshot, &geometry, max_ctx, slot_bytes).unwrap();

    let options = DagOptions::default();
    let dag = build_decode_dag(&geometry, &tuning, options);
    // No recycling: every value keeps its own buffer so the dump reads what
    // each kernel wrote, not what overwrote it.
    let schedule = build_scratch_schedule(&dag, true).unwrap();
    storage.scratch = driver_metal_new::metal::scratch_pool(
        &context,
        schedule.coloring.colors_used as usize,
        slot_bytes,
    )
    .expect("the no-recycle pool allocates");
    eprintln!("no-recycle pool: {} buffers", schedule.coloring.colors_used);

    let features = PsoFeatures {
        gdn: true,
        gated_attention: true,
        sdpa_d256: geometry.head_dim == 256,
        routed: geometry.is_moe(),
        untied: !geometry.tied_embeddings,
        ..PsoFeatures::default()
    };
    let pso_plan = plan_decode_psos(&EntryNames::bf16_g64_b4(), features);
    let compiler = Compiler::new(&context).unwrap();
    let psos = load_step_psos(&compiler, &context, &kernels_dir(), &pso_plan).unwrap();
    let step = DecodeStep::prepare(
        &context, &storage, &geometry, &tuning, options, &schedule, psos, max_ctx,
    )
    .unwrap();

    let bos: u32 = [&root, root.get("text_config").unwrap_or(&root)]
        .iter()
        .find_map(|level| level.get("bos_token_id"))
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .unwrap();
    let io = |slot: IoSlot| storage.io[slot as usize].as_ref().unwrap();
    unsafe {
        io(IoSlot::TokenId).write(0, &bos.to_le_bytes()).unwrap();
        io(IoSlot::SeqLen).write(0, &1u32.to_le_bytes()).unwrap();
    }
    let mut stepper = Stepper::new(&context).unwrap();
    step.fire(&mut stepper).unwrap();

    let dir = std::env::temp_dir().join("pie-golden-bisect");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let sites: Vec<_> = dag
        .iter()
        .map(|d| driver_metal_new::batch::TapSite {
            kind: d.kind,
            layer: d.layer,
        })
        .collect();
    unsafe {
        driver_metal_new::batch::dump_taps(
            &dir,
            &sites,
            &schedule,
            &storage.scratch,
            &geometry,
            1,
            0,
        )
    }
    .unwrap();

    // Host reference, tap by tap. embed: the dequantized bos row.
    let staged = |name: &str| {
        let h = storage
            .weights
            .get(name)
            .unwrap_or_else(|| panic!("{name} staged"));
        unsafe { std::slice::from_raw_parts(h.contents().cast::<u8>().as_ptr(), h.len() as usize) }
    };
    let hidden = geometry.hidden as usize;
    let embed = dequant_row(
        staged("embed_tokens.weight"),
        staged("embed_tokens.scales"),
        staged("embed_tokens.biases"),
        bos as usize,
        hidden,
    );
    let tap = read_npy(&dir.join("embed.npy"));
    let c = cosine(&embed, &tap);
    eprintln!("embed cosine {c:.6}");
    assert!(
        c > 0.999,
        "embed diverges: the gather or its binds are wrong"
    );

    // 0.attn_norm: RMS over the embed row with layer 0's weight.
    let w = staged("layers.0.input_layernorm.weight");
    let rms = {
        let mean: f32 = embed.iter().map(|x| x * x).sum::<f32>() / hidden as f32;
        let inv = 1.0 / (mean + geometry.eps).sqrt();
        embed
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let wi = bf16(u16::from_le_bytes([w[i * 2], w[i * 2 + 1]]));
                x * inv * wi
            })
            .collect::<Vec<_>>()
    };
    let tap = read_npy(&dir.join("0.attn_norm.npy"));
    let c = cosine(&rms, &tap);
    eprintln!("0.attn_norm cosine {c:.6}");
    assert!(
        c > 0.999,
        "attn_norm diverges: RmsParams or its binds are wrong"
    );

    // 0.gdn_in_qkv: the first quantized matvec, host-recomputed whole.
    let conv_dim = geometry.gdn_conv_dim as usize;
    let wq = staged("layers.0.linear_attn.in_proj_qkv.weight");
    let sq = staged("layers.0.linear_attn.in_proj_qkv.scales");
    let bq = staged("layers.0.linear_attn.in_proj_qkv.biases");
    let mut qkv = Vec::with_capacity(conv_dim);
    for n in 0..conv_dim {
        let row = dequant_row(wq, sq, bq, n, hidden);
        qkv.push(row.iter().zip(&rms).map(|(w, x)| w * x).sum::<f32>());
    }
    let tap = read_npy(&dir.join("0.gdn_in_qkv.npy"));
    let c = cosine(&qkv, &tap);
    eprintln!("0.gdn_in_qkv cosine {c:.6}");
    assert!(
        c > 0.99,
        "the quantized matvec diverges: Qmv K/N or the triplet binds"
    );
    eprintln!("bisect: embed, attn_norm and the first matvec agree with the host");
}
