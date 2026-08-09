//! The launch bridge, smoked end to end: one generated binding, one real
//! kernel, one round trip (retirement plan phase A step 3).
//!
//! The cheapest row in the table does the proving:
//! `quant::cast_fp32_to_bf16` reads fp32, writes bf16, and its answer is
//! bit-checkable on the host (bf16 IS the top half of the fp32 for values
//! that need no rounding). If this passes, the whole chain held — the
//! bindings module compiled against the generated declarations, the shim
//! compiled against the real headers, the archive linked, and a launcher
//! this crate did not write ran device code on this crate's stream.
//!
//! Skipped without a device, like every GPU test here.

use driver_cuda_new::cuda::{Allocator, OwnedStream};
use driver_cuda_new::launch::ffi;

mod common;
use common::{device_or_skip, gpu_guard};

#[test]
fn a_generated_binding_reaches_a_real_kernel() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("bridge smoke") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let alloc = Allocator::new();

    // Powers of two need no rounding, so each bf16 is exactly the fp32's
    // top sixteen bits and the expectation can be computed with a shift.
    let src: Vec<f32> = (0..64).map(|i| (i as f32) * 0.25).collect();
    let src_bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut d_src = alloc.alloc(src_bytes.len()).expect("src alloc");
    d_src.copy_from_host(&src_bytes, stream.as_ref()).expect("h2d");
    let d_dst = alloc.alloc(src.len() * 2).expect("dst alloc");

    unsafe {
        ffi::pie_k_quant_cast_fp32_to_bf16(
            d_src.as_ptr(),
            d_dst.as_ptr(),
            src.len(),
            stream.as_ref().as_raw().cast(),
        );
    }

    let mut back = vec![0u8; src.len() * 2];
    d_dst.copy_to_host(&mut back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");

    for (i, v) in src.iter().enumerate() {
        let expect = (v.to_bits() >> 16) as u16;
        let got = u16::from_le_bytes([back[i * 2], back[i * 2 + 1]]);
        assert_eq!(
            got, expect,
            "element {i}: bf16 0x{got:04x} != expected 0x{expect:04x} (fp32 {v})"
        );
    }
}

/// The second table's chain, smoked the same way: a DRIVER-INTERNAL
/// launcher (no DSL row — the envelope seed) reached through its generated
/// binding via `LiveKvCacheOps`, the first bridge-gated seam impl. Empty
/// envelopes are +inf/-inf bf16 by construction, so the whole tier is
/// bit-checkable on the host.
#[test]
fn a_driver_internal_binding_seeds_the_envelope_tier() {
    use driver_cuda_new::store::kv_cache_live::{KvCacheDeviceOps, LiveKvCacheOps};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("driver-internal envelope seed") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let alloc = Allocator::new();

    // 2 pages x 2 kv heads x 4 dims = 16 bf16 elements per plane.
    let elems = 2 * 2 * 4;
    let mut d_min = alloc.alloc(elems * 2).expect("env_min");
    let mut d_max = alloc.alloc(elems * 2).expect("env_max");
    d_min.memset(0, stream.as_ref()).expect("zero min");
    d_max.memset(0, stream.as_ref()).expect("zero max");

    let mut ops = LiveKvCacheOps::new(stream.as_ref().as_raw().cast());
    ops.envelope_seed(d_min.as_ptr().cast(), d_max.as_ptr().cast(), 2, 2, 4);
    ops.stream_synchronize();

    let mut min_back = vec![0u8; elems * 2];
    let mut max_back = vec![0u8; elems * 2];
    d_min.copy_to_host(&mut min_back, stream.as_ref()).expect("d2h min");
    d_max.copy_to_host(&mut max_back, stream.as_ref()).expect("d2h max");
    stream.as_ref().synchronize().expect("sync");

    for i in 0..elems {
        let min = u16::from_le_bytes([min_back[i * 2], min_back[i * 2 + 1]]);
        let max = u16::from_le_bytes([max_back[i * 2], max_back[i * 2 + 1]]);
        assert_eq!(min, 0x7F80, "element {i}: empty env_min is +inf bf16");
        assert_eq!(max, 0xFF80, "element {i}: empty env_max is -inf bf16");
    }
}

/// The live `ScoreOps`: memset, the CSR upload, and the fold launch. With
/// ONE query head the fold's per-position average is over a single value,
/// so `folded == raw` over the request's span regardless of the kernel's
/// internal layout — an identity that checks the whole chain without
/// re-deriving the indexing the score oracle already pinned.
#[test]
fn the_live_score_ops_upload_memset_and_fold() {
    use driver_cuda_new::model::attn_score::{LiveScoreOps, ScoreOps};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("live score ops") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let alloc = Allocator::new();
    let mut ops = LiveScoreOps::new(stream.as_ref().as_raw().cast());

    // memset: 64 bytes of 0xA5.
    let scratch = alloc.alloc(64).expect("scratch");
    ops.memset_async(scratch.as_ptr().cast(), 0xa5, 64);
    let mut back = vec![0u8; 64];
    scratch.copy_to_host(&mut back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");
    assert!(back.iter().all(|&b| b == 0xa5));

    // CSR upload: one request spanning 4 positions.
    let indptr: Vec<i32> = vec![0, 4];
    let d_indptr = alloc.alloc(indptr.len() * 4).expect("indptr");
    ops.upload_csr(d_indptr.as_ptr().cast(), &indptr);

    // Fold: 1 request, 1 q head, kv_len 4 in one page of 16.
    let raw: Vec<f32> = vec![0.25, 0.5, 0.125, 1.0];
    let raw_bytes: Vec<u8> = raw.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut d_raw = alloc.alloc(raw_bytes.len()).expect("raw");
    d_raw.copy_from_host(&raw_bytes, stream.as_ref()).expect("h2d raw");
    let page_indptr: Vec<u8> = [0u32, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut d_pages = alloc.alloc(page_indptr.len()).expect("pages");
    d_pages.copy_from_host(&page_indptr, stream.as_ref()).expect("h2d pages");
    let last_lens: Vec<u8> = [4u32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut d_lens = alloc.alloc(last_lens.len()).expect("lens");
    d_lens.copy_from_host(&last_lens, stream.as_ref()).expect("h2d lens");
    let d_folded = alloc.alloc(raw_bytes.len()).expect("folded");

    ops.fold_heads(
        d_raw.as_ptr().cast(),
        d_indptr.as_ptr().cast(),
        d_pages.as_ptr().cast(),
        d_lens.as_ptr().cast(),
        16,
        1,
        1,
        d_folded.as_ptr().cast(),
    );

    let mut folded_back = vec![0u8; raw_bytes.len()];
    d_folded.copy_to_host(&mut folded_back, stream.as_ref()).expect("d2h folded");
    stream.as_ref().synchronize().expect("sync");
    for (i, v) in raw.iter().enumerate() {
        let got = f32::from_le_bytes(folded_back[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!(got, *v, "position {i}: average over one head is identity");
    }
}

/// The live `LoraOps`: the cast through its DSL binding, and the pointer
/// slab landing device-resident with its values intact.
#[test]
fn the_live_lora_ops_cast_and_upload_the_slab() {
    use driver_cuda_new::model::lora::{LiveLoraOps, LoraOps};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("live lora ops") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let alloc = Allocator::new();
    let mut ops = LiveLoraOps::new(stream.as_ref().as_raw().cast());

    // Cast: same bit-check as the bridge smoke, through the seam.
    let src: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5).collect();
    let src_bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut d_src = alloc.alloc(src_bytes.len()).expect("src");
    d_src.copy_from_host(&src_bytes, stream.as_ref()).expect("h2d");
    let d_dst = alloc.alloc(src.len() * 2).expect("dst");
    ops.cast_fp32_to_bf16(d_src.as_ptr(), d_dst.as_ptr(), src.len());
    let mut back = vec![0u8; src.len() * 2];
    d_dst.copy_to_host(&mut back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");
    for (i, v) in src.iter().enumerate() {
        let expect = (v.to_bits() >> 16) as u16;
        let got = u16::from_le_bytes([back[i * 2], back[i * 2 + 1]]);
        assert_eq!(got, expect, "element {i}");
    }

    // Slab: four sentinel pointers, round-tripped.
    let slots: Vec<*const std::ffi::c_void> =
        [0x1000usize, 0x2000, 0x3000, 0x4000].iter().map(|&a| a as _).collect();
    let d_slab = alloc.alloc(slots.len() * 8).expect("slab");
    ops.upload_slab(d_slab.as_ptr(), &slots);
    let mut slab_back = vec![0u8; slots.len() * 8];
    d_slab.copy_to_host(&mut slab_back, stream.as_ref()).expect("d2h slab");
    stream.as_ref().synchronize().expect("sync");
    for (i, &p) in slots.iter().enumerate() {
        let got = u64::from_le_bytes(slab_back[i * 8..i * 8 + 8].try_into().unwrap());
        assert_eq!(got, p as u64, "slot {i} survived the upload");
    }
}

/// The executor's two halves over the REAL anchor lowering (retirement
/// plan phase C): bind + dispatch walk `qwen3_0_6b`'s decode launches on
/// the device until the first kernel without an arm, and the numbers are
/// checked against host math. What this pins beyond the plumbing is the
/// OPERAND ORDER inside each arm — a swapped input is wrong values, not a
/// type error, and only host arithmetic notices.
#[test]
fn the_executor_prefix_runs_the_anchor_decode_on_device() {
    use std::collections::BTreeMap;

    use driver_cuda_new::cuda::cublas::{CublasHandle, LiveCublas};
    use driver_cuda_new::model::executor::{
        DispatchCtx, DispatchPlan, DispatchRefusal, Frame, Resolver, bind, dispatch,
    };
    use model::families::llama_like::forward::facts::{LlamaLikeCudaFacts, LlamaLikeFacts};
    use model::families::llama_like::forward::llama_like_cuda;
    use model_compiler::lower::{Fire, Row, lower};
    use model_compiler::trace::{FireClass, ValueId};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("executor prefix") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = Allocator::new();

    // The real traced decode form, over four rows.
    let plan = llama_like_cuda(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
        FireClass::Decode,
    );
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; 4];
    let l = lower(&plan, &rows, Fire { captures_across_splits: false }).expect("lowers");

    let arena = alloc.alloc(l.arena_bytes).expect("arena");
    let frame = Frame { arena: arena.as_ptr(), arena_bytes: l.arena_bytes };

    // Sixty-four fake vocabulary rows: token t's embedding alternates
    // +a(t), -a(t) with a(t) = 0.5 + 0.25 t, so rmsnorm collapses every
    // row to alternating ±1 whatever the token — two exact expectations
    // from one pattern.
    const HIDDEN: usize = 1024;
    const VOCAB: usize = 64;
    let tokens: [i32; 4] = [1, 2, 3, 5];
    let amp = |t: i32| 0.5 + 0.25 * t as f32;
    let bf16 = |v: f32| (v.to_bits() >> 16) as u16;
    let mut embed_host = vec![0u8; VOCAB * HIDDEN * 2];
    for t in 0..VOCAB {
        for c in 0..HIDDEN {
            let v = if c % 2 == 0 { amp(t as i32) } else { -amp(t as i32) };
            let b = bf16(v).to_le_bytes();
            embed_host[(t * HIDDEN + c) * 2] = b[0];
            embed_host[(t * HIDDEN + c) * 2 + 1] = b[1];
        }
    }
    let ones_host: Vec<u8> = std::iter::repeat_n(bf16(1.0).to_le_bytes(), HIDDEN)
        .flatten()
        .collect();
    let ids_host: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();

    struct Live {
        embed: driver_cuda_new::cuda::DeviceBuffer,
        ones: driver_cuda_new::cuda::DeviceBuffer,
        zeros: driver_cuda_new::cuda::DeviceBuffer,
        ids: driver_cuda_new::cuda::DeviceBuffer,
        named: BTreeMap<ValueId, *mut std::ffi::c_void>,
    }
    impl Resolver for Live {
        fn weight(&mut self, name: &str) -> Option<*const std::ffi::c_void> {
            Some(if name.contains("embed") {
                self.embed.as_ptr()
            } else if name.contains("norm") {
                self.ones.as_ptr()
            } else {
                self.zeros.as_ptr()
            })
        }
        fn named(&mut self, value: ValueId) -> Option<*mut std::ffi::c_void> {
            // Every pinned input in the prefix is a per-row i32 array
            // (token ids, positions); one buffer serves each id.
            Some(*self.named.entry(value).or_insert(self.ids.as_ptr()))
        }
    }

    let mut embed_dev = alloc.alloc(embed_host.len()).expect("embed w");
    embed_dev.copy_from_host(&embed_host, stream.as_ref()).expect("h2d embed");
    let mut ones_dev = alloc.alloc(ones_host.len()).expect("ones");
    ones_dev.copy_from_host(&ones_host, stream.as_ref()).expect("h2d ones");
    let mut zeros_dev = alloc.alloc(16 << 20).expect("zeros");
    zeros_dev.memset(0, stream.as_ref()).expect("zero");
    let mut ids_dev = alloc.alloc(ids_host.len()).expect("ids");
    ids_dev.copy_from_host(&ids_host, stream.as_ref()).expect("h2d ids");
    let mut resolver = Live {
        embed: embed_dev,
        ones: ones_dev,
        zeros: zeros_dev,
        ids: ids_dev,
        named: BTreeMap::new(),
    };

    let mut cublas_ops = LiveCublas;
    let mut cublas = CublasHandle::create(&mut cublas_ops, raw_stream).expect("cublas");
    let ctx = DispatchCtx {
        stream: raw_stream,
        cublas: cublas.handle().expect("created").cast(),
        rms_eps: 1e-6,
        rope_theta: 1e6,
        head_dim: 128,
        vocab: VOCAB as i32,
        gate_second: false,
        rope_interleaved: false,
        token_ids: resolver.ids.as_ptr(),
        positions: resolver.ids.as_ptr(),
        final_logit_softcap: 0.0,
        ple_dim: 0,
        scales: std::collections::BTreeMap::new(),
    };
    let dplan = DispatchPlan::new(&plan, &l);

    // Walk until the first kernel without an arm, remembering where the
    // embed and the first rmsnorm wrote.
    let mut embed_out: Option<usize> = None;
    let mut norm_out: Option<usize> = None;
    let mut dispatched = 0usize;
    let mut stopped_at = String::new();
    for (i, launch) in l.launches.iter().enumerate() {
        let bound = bind(&l, launch, frame, &mut resolver).expect("binds");
        let offset_of = |p: *mut std::ffi::c_void| p as usize - frame.arena as usize;
        match dispatch(&bound, dplan.spec(i), frame, &mut resolver, &ctx, None, None) {
            Ok(()) => {
                if bound.kernel == "layout::embed_bf16" {
                    embed_out.get_or_insert(offset_of(bound.args[0].ptr));
                } else if bound.kernel == "norm::rmsnorm_bf16" {
                    norm_out.get_or_insert(offset_of(bound.args[1].ptr));
                }
                dispatched += 1;
            }
            // The smoke runs WITHOUT attention context on purpose — the
            // fused qkv arm refusing on that is the intended boundary.
            Err(DispatchRefusal::NoArm(k) | DispatchRefusal::NoAttnCtx(k)) => {
                stopped_at = k;
                break;
            }
            Err(e) => panic!("arm drift: {e:?}"),
        }
    }
    stream.as_ref().synchronize().expect("sync");
    assert!(dispatched >= 4, "only {dispatched} launches ran before the stop");
    assert_eq!(
        stopped_at, "attn::qkv_decode_qk_norm_rope_write_kv_bf16",
        "the walk should stop at the fused attention step"
    );

    let mut arena_back = vec![0u8; l.arena_bytes];
    arena.copy_to_host(&mut arena_back, stream.as_ref()).expect("d2h arena");
    stream.as_ref().synchronize().expect("sync");
    let bf16_at = |off: usize, i: usize| {
        u16::from_le_bytes([arena_back[off + i * 2], arena_back[off + i * 2 + 1]])
    };

    // The embed rows are the pattern rows for tokens [1, 2, 3, 5].
    let e = embed_out.expect("embed ran");
    for (r, t) in tokens.iter().enumerate() {
        for c in [0usize, 1, 511, 1023] {
            let want = bf16(if c % 2 == 0 { amp(*t) } else { -amp(*t) });
            let got = bf16_at(e, r * HIDDEN + c);
            assert_eq!(got, want, "embed row {r} (token {t}) col {c}");
        }
    }

    // RMSNorm of an alternating ±a row is alternating ±1 (times the ones
    // weight), whatever a was — bf16-exactly, since 1.0 is representable
    // and the kernel normalizes in fp32.
    let n = norm_out.expect("rmsnorm ran");
    for r in 0..tokens.len() {
        for c in [0usize, 1, 512, 1023] {
            let want = bf16(if c % 2 == 0 { 1.0 } else { -1.0 });
            let got = bf16_at(n, r * HIDDEN + c);
            assert_eq!(got, want, "rmsnorm row {r} col {c}");
        }
    }

    cublas.release(&mut cublas_ops);
}

/// FlashInfer's decode planner, driven from Rust end to end: a real
/// `DecodePlanCache` through the hand-written extras, staged into a LIVE
/// workspace slot under the begin/end fence — the C++ prepare flow,
/// re-spoken. This is the riskiest unlit piece of the attention step
/// (host planning code deep inside the archive), which is why it gets its
/// own smoke before any attention arm consumes the plan.
#[test]
fn the_decode_planner_plans_a_real_geometry() {
    use driver_cuda_new::model::attention_workspace::{AttentionWorkspace, LiveStagingOps};
    use driver_cuda_new::model::executor::DecodePlan;

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("decode planner") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let raw = stream.as_ref().as_raw().cast::<std::ffi::c_void>();

    let mut ops = LiveStagingOps;
    let mut ws = AttentionWorkspace::allocate(&mut ops, 32 << 20, 16 << 20, 2)
        .expect("workspace");

    // Four decode requests, one 16-token page each — qwen3-0.6b geometry.
    let indptr: [u32; 5] = [0, 1, 2, 3, 4];
    let mut plan = DecodePlan::new();
    ws.begin_plan_update(&mut ops).expect("begin");
    plan.plan_decode(&indptr, 16, 8, 128, 16, ws.view(), raw, false, -1);
    ws.end_plan_update(&mut ops, raw);
    stream.as_ref().synchronize().expect("the staged upload retires");

    // Plan again with different geometry — the cache is reusable per
    // fire, which is how the driver holds it.
    let indptr2: [u32; 3] = [0, 2, 4];
    ws.begin_plan_update(&mut ops).expect("begin 2");
    plan.plan_decode(&indptr2, 16, 8, 128, 16, ws.view(), raw, false, -1);
    ws.end_plan_update(&mut ops, raw);
    stream.as_ref().synchronize().expect("second upload retires");

    ws.release(&mut ops);
}

/// The FULL decode: all 257 launches of `qwen3_0_6b`'s real lowering,
/// walked on the device with a live `AttnCtx` — KV pools, page CSRs,
/// write descriptors, a planned FlashInfer cache. All-zero weights make
/// the whole forward analytically checkable: every projection returns
/// zero, attention over zero V returns zero, the beta-1 residual folds
/// add zero — so the residual stream must equal the embed rows
/// BIT-EXACTLY after 28 layers, and the logits must be all-zero. A single
/// swapped operand anywhere in the walk breaks one of the two.
#[test]
fn the_full_zero_weight_decode_walks_every_launch() {
    use std::collections::BTreeMap;

    use driver_cuda_new::cuda::cublas::{CublasHandle, LiveCublas};
    use driver_cuda_new::dtype::DType;
    use driver_cuda_new::launch::{KvCacheLayerView, KvCacheScheme};
    use driver_cuda_new::model::attention_workspace::{AttentionWorkspace, LiveStagingOps};
    use driver_cuda_new::model::executor::{
        AttnCtx, DecodePlan, DispatchCtx, DispatchPlan, Frame, Resolver, run,
    };
    use model::families::llama_like::forward::facts::{LlamaLikeCudaFacts, LlamaLikeFacts};
    use model::families::llama_like::forward::llama_like_cuda;
    use model_compiler::lower::{Arg, Fire, Row, lower};
    use model_compiler::trace::{FireClass, ValueId};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("full zero-weight decode") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = Allocator::new();

    const HIDDEN: usize = 1024;
    const LAYERS: usize = 28;
    const KV_HEADS: i32 = 8;
    const Q_HEADS: i32 = 16;
    const HEAD_DIM: i32 = 128;
    const PAGE: i32 = 16;
    const ROWS: usize = 4;
    // The real vocabulary: the checkpoint is TIED, so the lm_head resolves
    // to "embed" and reads all [vocab, hidden] of it — a 64-row fake table
    // was the first version's illegal address. Pattern rows for the first
    // 64 tokens, zeros beyond: the logits become analytically checkable.
    const VOCAB: usize = 151_936;
    const PATTERNED: usize = 64;

    let plan = llama_like_cuda(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
        FireClass::Decode,
    );
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; ROWS];
    let l = lower(&plan, &rows, Fire { captures_across_splits: false }).expect("lowers");
    let dplan = DispatchPlan::new(&plan, &l);

    let arena = alloc.alloc(l.arena_bytes).expect("arena");
    let frame = Frame { arena: arena.as_ptr(), arena_bytes: l.arena_bytes };

    // ── Weights: embed pattern, norm ones, everything else zero. ──
    let bf16 = |v: f32| (v.to_bits() >> 16) as u16;
    let amp = |t: i32| 0.5 + 0.25 * t as f32;
    let tokens: [i32; ROWS] = [1, 2, 3, 5];
    let mut embed_host = vec![0u8; VOCAB * HIDDEN * 2];
    for t in 0..PATTERNED {
        for c in 0..HIDDEN {
            let v = if c % 2 == 0 { amp(t as i32) } else { -amp(t as i32) };
            let b = bf16(v).to_le_bytes();
            embed_host[(t * HIDDEN + c) * 2] = b[0];
            embed_host[(t * HIDDEN + c) * 2 + 1] = b[1];
        }
    }
    let mut embed_dev = alloc.alloc(embed_host.len()).expect("embed");
    embed_dev.copy_from_host(&embed_host, stream.as_ref()).expect("h2d");
    let ones_host: Vec<u8> =
        std::iter::repeat_n(bf16(1.0).to_le_bytes(), HIDDEN).flatten().collect();
    let mut ones_dev = alloc.alloc(ones_host.len()).expect("ones");
    ones_dev.copy_from_host(&ones_host, stream.as_ref()).expect("h2d");
    let mut zeros_dev = alloc.alloc(8 * 3072 * HIDDEN * 2).expect("zeros");
    zeros_dev.memset(0, stream.as_ref()).expect("zero");

    // ── Named pins, preallocated from the lowering's own widths. ──
    let mut named_widths: BTreeMap<ValueId, u32> = BTreeMap::new();
    for a in &l.args {
        if let Arg::Named { value, width } = a {
            named_widths.insert(*value, *width);
        }
    }
    for i in 0..l.launches.len() {
        for a in &dplan.spec(i).outs {
            if let Arg::Named { value, width } = a {
                named_widths.insert(*value, *width);
            }
        }
    }
    let mut named_bufs: BTreeMap<ValueId, driver_cuda_new::cuda::DeviceBuffer> = named_widths
        .iter()
        .map(|(&v, &w)| {
            let mut b = alloc.alloc(ROWS * w as usize * 2).expect("pin");
            b.memset(0, stream.as_ref()).expect("zero pin");
            (v, b)
        })
        .collect();

    struct Live<'a> {
        embed: *const std::ffi::c_void,
        ones: *const std::ffi::c_void,
        zeros: *const std::ffi::c_void,
        named: &'a mut BTreeMap<ValueId, driver_cuda_new::cuda::DeviceBuffer>,
    }
    impl Resolver for Live<'_> {
        fn weight(&mut self, name: &str) -> Option<*const std::ffi::c_void> {
            Some(if name.contains("embed") || name.contains("lm_head") {
                if name.contains("lm_head") { self.zeros } else { self.embed }
            } else if name.contains("norm") {
                self.ones
            } else {
                self.zeros
            })
        }
        fn named(&mut self, value: ValueId) -> Option<*mut std::ffi::c_void> {
            self.named.get(&value).map(|b| b.as_ptr())
        }
    }

    // ── The fire's KV side: pools, views, CSRs, write descriptors. ──
    let plane = (4 * PAGE * KV_HEADS * HEAD_DIM) as usize * 2;
    let pools: Vec<(driver_cuda_new::cuda::DeviceBuffer, driver_cuda_new::cuda::DeviceBuffer)> =
        (0..LAYERS)
            .map(|_| {
                let mut k = alloc.alloc(plane).expect("k pool");
                let mut v = alloc.alloc(plane).expect("v pool");
                k.memset(0, stream.as_ref()).expect("zk");
                v.memset(0, stream.as_ref()).expect("zv");
                (k, v)
            })
            .collect();
    let layers: Vec<KvCacheLayerView> = pools
        .iter()
        .enumerate()
        .map(|(i, (k, v))| KvCacheLayerView {
            layer: i as i32,
            source_layer: i as i32,
            num_pages: 4,
            page_size: PAGE,
            num_kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            scheme: KvCacheScheme::Native,
            storage_dtype: DType::Bf16,
            block_size: 0,
            k_pages: k.as_ptr(),
            v_pages: v.as_ptr(),
            k_scales: core::ptr::null_mut(),
            v_scales: core::ptr::null_mut(),
            // The NATIVE alias the C++ `layer_view` maintains: the dispatch
            // reads the bf16 MIRROR planes, and for a native cache those
            // are the storage pages themselves.
            k_bf16_pages: k.as_ptr(),
            v_bf16_pages: v.as_ptr(),
            k_env_min: core::ptr::null_mut(),
            k_env_max: core::ptr::null_mut(),
            hnd_layout: false,
            native_bf16: true,
        })
        .collect();

    let up = |data: &[u8]| {
        let mut b = alloc.alloc(data.len()).expect("csr");
        b.copy_from_host(data, stream.as_ref()).expect("h2d csr");
        b
    };
    let u32s = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let csr_indices = up(&u32s(&[0, 1, 2, 3]));
    let csr_indptr = up(&u32s(&[0, 1, 2, 3, 4]));
    let csr_lens = up(&u32s(&[1, 1, 1, 1]));
    let w_page = up(&u32s(&[0, 1, 2, 3]));
    let w_off = up(&u32s(&[0, 0, 0, 0]));
    let row_valid = up(&[1u8, 1, 1, 1]);
    let ids = up(&tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>());
    let positions = up(&[0i32, 0, 0, 0].iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>());
    let lse = alloc.alloc(ROWS * Q_HEADS as usize * 4).expect("lse");

    // ── Workspace + the planned decode cache. ──
    let mut sops = LiveStagingOps;
    let mut ws = AttentionWorkspace::allocate(&mut sops, 32 << 20, 16 << 20, 2).expect("ws");
    let mut dplan_cache = DecodePlan::new();
    ws.begin_plan_update(&mut sops).expect("begin");
    dplan_cache.plan_decode(&[0, 1, 2, 3, 4], Q_HEADS, KV_HEADS, HEAD_DIM, PAGE, ws.view(), raw_stream, false, -1);
    ws.end_plan_update(&mut sops, raw_stream);

    // The guard-owned attention values: q is the dispatch's Named arg
    // (the observed-query pin), o is what the following o_proj reads.
    let fi = l
        .launches
        .iter()
        .position(|x| l.kernels[x.kernel as usize] == "attn::dispatch_attention_flashinfer_decode")
        .expect("a decode fire dispatches attention");
    let q_pin_value = match &l.args[l.launches[fi].args.start as usize] {
        Arg::Named { value, .. } => *value,
        other => panic!("the dispatch's q is a pin, got {other:?}"),
    };
    let o_off = match &l.args[l.launches[fi + 1].args.start as usize] {
        Arg::Arena { at, .. } => *at,
        other => panic!("o_proj reads the attention slot, got {other:?}"),
    };

    let attn = AttnCtx {
        decode_plan: dplan_cache.as_ptr(),
        prefill_plan: core::ptr::null_mut(),
        workspace: ws.view(),
        layers,
        q_out: named_bufs[&q_pin_value].as_ptr(),
        o_out: unsafe { arena.as_ptr().cast::<u8>().add(o_off) }.cast(),
        kv_page_indices_d: csr_indices.as_ptr().cast(),
        kv_page_indptr_d: csr_indptr.as_ptr().cast(),
        kv_last_page_lens_d: csr_lens.as_ptr().cast(),
        qo_indptr_d: core::ptr::null(),
        qo_indptr_h: core::ptr::null(),
        kv_page_indptr_h: core::ptr::null(),
        num_requests: ROWS as i32,
        num_pages_in_batch: 4,
        first_token: 0,
        w_page_d: w_page.as_ptr().cast(),
        w_off_d: w_off.as_ptr().cast(),
        row_valid_d: row_valid.as_ptr().cast(),
        lse_out_d: lse.as_ptr().cast(),
        window_left: -1,
        window_left_by_layer: Vec::new(),
        logits_soft_cap: 0.0,
        sm_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
    };

    let mut cublas_ops = LiveCublas;
    let mut cublas = CublasHandle::create(&mut cublas_ops, raw_stream).expect("cublas");
    let ctx = DispatchCtx {
        stream: raw_stream,
        cublas: cublas.handle().expect("created").cast(),
        rms_eps: 1e-6,
        rope_theta: 1e6,
        head_dim: HEAD_DIM,
        vocab: VOCAB as i32,
        gate_second: false,
        rope_interleaved: false,
        token_ids: ids.as_ptr(),
        positions: positions.as_ptr(),
        final_logit_softcap: 0.0,
        ple_dim: 0,
        scales: std::collections::BTreeMap::new(),
    };

    // ── The walk: every launch, no refusals allowed. ──
    let mut resolver = Live {
        embed: embed_dev.as_ptr(),
        ones: ones_dev.as_ptr(),
        zeros: zeros_dev.as_ptr(),
        named: &mut named_bufs,
    };
    let mut embed_out = None;
    let mut logits_value: Option<ValueId> = None;
    for (i, launch) in l.launches.iter().enumerate() {
        if l.kernels[launch.kernel as usize] == "layout::embed_bf16"
            && let Arg::Arena { at, .. } = &l.args[launch.args.start as usize]
        {
            embed_out.get_or_insert(*at);
        }
        if let Some(Arg::Named { value, .. }) = dplan.spec(i).outs.first()
            && i == l.launches.len() - 1
        {
            logits_value = Some(*value);
        }
    }
    let ran = run(&l, &dplan, frame, &mut resolver, &ctx, Some(&attn), None)
        .unwrap_or_else(|e| panic!("the walk refused: {e:?}"));
    assert_eq!(ran, l.launches.len(), "every launch ran");
    stream.as_ref().synchronize().expect("the whole decode retires");

    // ── Invariant 1: the residual equals the embed rows, bit-exactly. ──
    let mut arena_back = vec![0u8; l.arena_bytes];
    arena.copy_to_host(&mut arena_back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");
    let e = embed_out.expect("embed ran");
    for (r, t) in tokens.iter().enumerate() {
        for c in [0usize, 1, 700, 1023] {
            let want = bf16(if c % 2 == 0 { amp(*t) } else { -amp(*t) });
            let off = e + (r * HIDDEN + c) * 2;
            let got = u16::from_le_bytes([arena_back[off], arena_back[off + 1]]);
            assert_eq!(got, want, "residual row {r} col {c} drifted from the embed");
        }
    }

    // ── Invariant 2: the logits are the tied lm_head's exact algebra. ──
    //
    // The residual is the embed row (invariant 1), the final norm turns it
    // into alternating ±1, and the tied lm_head dots that against every
    // pattern row: logit[r][t] = Σ_c (±1)(±amp(t)) = HIDDEN · amp(t) for
    // t < PATTERNED (the signs align by construction), 0 beyond. The same
    // value for EVERY row r — and bf16-representable at every checked t.
    let lv = logits_value.expect("the last launch writes a named pin (the logits)");
    let logits = &named_bufs[&lv];
    let mut back = vec![0u8; logits.len()];
    logits.copy_to_host(&mut back, stream.as_ref()).expect("d2h logits");
    stream.as_ref().synchronize().expect("sync");
    let logit = |r: usize, t: usize| {
        let off = (r * VOCAB + t) * 2;
        u16::from_le_bytes([back[off], back[off + 1]])
    };
    for r in 0..ROWS {
        for t in [1usize, 2, 3, 5, 63] {
            let want = bf16(HIDDEN as f32 * amp(t as i32));
            assert_eq!(logit(r, t), want, "logit row {r} token {t}");
        }
        for t in [64usize, VOCAB - 1] {
            assert_eq!(logit(r, t), 0, "logit row {r} token {t} beyond the pattern");
        }
    }

    ws.release(&mut sops);
    cublas.release(&mut cublas_ops);
}

/// The FULL prefill: the decode walk's twin over `qwen3_0_6b`'s real
/// prefill lowering — two requests (3 + 4 tokens), the five prefill arms
/// (split, the staged in-place rope, the KV write, the dequant staging,
/// the planned FlashInfer prefill), same zero-weight algebra: residual ==
/// embed rows bit-exactly, logits == the tied lm_head's dot with the
/// pattern table. Causal attention over zero V is zero, so every layer's
/// residual fold adds nothing.
#[test]
fn the_full_zero_weight_prefill_walks_every_launch() {
    use std::collections::BTreeMap;

    use driver_cuda_new::cuda::cublas::{CublasHandle, LiveCublas};
    use driver_cuda_new::dtype::DType;
    use driver_cuda_new::launch::{KvCacheLayerView, KvCacheScheme};
    use driver_cuda_new::model::attention_workspace::{AttentionWorkspace, LiveStagingOps};
    use driver_cuda_new::model::executor::{
        AttnCtx, DispatchCtx, DispatchPlan, Frame, PrefillPlan, Resolver, run,
    };
    use model::families::llama_like::forward::facts::{LlamaLikeCudaFacts, LlamaLikeFacts};
    use model::families::llama_like::forward::llama_like_cuda;
    use model_compiler::lower::{Arg, Fire, Row, lower};
    use model_compiler::trace::{FireClass, ValueId};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("full zero-weight prefill") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = Allocator::new();

    const HIDDEN: usize = 1024;
    const LAYERS: usize = 28;
    const KV_HEADS: i32 = 8;
    const Q_HEADS: i32 = 16;
    const HEAD_DIM: i32 = 128;
    const PAGE: i32 = 16;
    const TOKENS: usize = 7;
    const VOCAB: usize = 151_936;
    const PATTERNED: usize = 64;

    let plan = llama_like_cuda(
        &LlamaLikeFacts::qwen3_0_6b(),
        &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
        FireClass::Prefill,
    );
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; TOKENS];
    let l = lower(&plan, &rows, Fire { captures_across_splits: false }).expect("lowers");
    let dplan = DispatchPlan::new(&plan, &l);

    let arena = alloc.alloc(l.arena_bytes).expect("arena");
    let frame = Frame { arena: arena.as_ptr(), arena_bytes: l.arena_bytes };

    let bf16 = |v: f32| (v.to_bits() >> 16) as u16;
    let amp = |t: i32| 0.5 + 0.25 * t as f32;
    let tokens: [i32; TOKENS] = [1, 2, 3, 5, 7, 11, 13];
    let mut embed_host = vec![0u8; VOCAB * HIDDEN * 2];
    for t in 0..PATTERNED {
        for c in 0..HIDDEN {
            let v = if c % 2 == 0 { amp(t as i32) } else { -amp(t as i32) };
            let b = bf16(v).to_le_bytes();
            embed_host[(t * HIDDEN + c) * 2] = b[0];
            embed_host[(t * HIDDEN + c) * 2 + 1] = b[1];
        }
    }
    let mut embed_dev = alloc.alloc(embed_host.len()).expect("embed");
    embed_dev.copy_from_host(&embed_host, stream.as_ref()).expect("h2d");
    let ones_host: Vec<u8> =
        std::iter::repeat_n(bf16(1.0).to_le_bytes(), HIDDEN).flatten().collect();
    let mut ones_dev = alloc.alloc(ones_host.len()).expect("ones");
    ones_dev.copy_from_host(&ones_host, stream.as_ref()).expect("h2d");
    let mut zeros_dev = alloc.alloc(8 * 3072 * HIDDEN * 2).expect("zeros");
    zeros_dev.memset(0, stream.as_ref()).expect("zero");

    let mut named_widths: BTreeMap<ValueId, u32> = BTreeMap::new();
    for a in &l.args {
        if let Arg::Named { value, width } = a {
            named_widths.insert(*value, *width);
        }
    }
    for i in 0..l.launches.len() {
        for a in &dplan.spec(i).outs {
            if let Arg::Named { value, width } = a {
                named_widths.insert(*value, *width);
            }
        }
    }
    let named_bufs: BTreeMap<ValueId, driver_cuda_new::cuda::DeviceBuffer> = named_widths
        .iter()
        .map(|(&v, &w)| {
            let mut b = alloc.alloc(TOKENS * w as usize * 2).expect("pin");
            b.memset(0, stream.as_ref()).expect("zero pin");
            (v, b)
        })
        .collect();

    struct Live<'a> {
        embed: *const std::ffi::c_void,
        ones: *const std::ffi::c_void,
        zeros: *const std::ffi::c_void,
        named: &'a BTreeMap<ValueId, driver_cuda_new::cuda::DeviceBuffer>,
    }
    impl Resolver for Live<'_> {
        fn weight(&mut self, name: &str) -> Option<*const std::ffi::c_void> {
            Some(if name.contains("embed") {
                self.embed
            } else if name.contains("norm") {
                self.ones
            } else {
                self.zeros
            })
        }
        fn named(&mut self, value: ValueId) -> Option<*mut std::ffi::c_void> {
            self.named.get(&value).map(|b| b.as_ptr())
        }
    }

    // Two requests: tokens [0,3) and [3,7), one page each.
    let plane = (2 * PAGE * KV_HEADS * HEAD_DIM) as usize * 2;
    let pools: Vec<(driver_cuda_new::cuda::DeviceBuffer, driver_cuda_new::cuda::DeviceBuffer)> =
        (0..LAYERS)
            .map(|_| {
                let mut k = alloc.alloc(plane).expect("k pool");
                let mut v = alloc.alloc(plane).expect("v pool");
                k.memset(0, stream.as_ref()).expect("zk");
                v.memset(0, stream.as_ref()).expect("zv");
                (k, v)
            })
            .collect();
    let layers: Vec<KvCacheLayerView> = pools
        .iter()
        .enumerate()
        .map(|(i, (k, v))| KvCacheLayerView {
            layer: i as i32,
            source_layer: i as i32,
            num_pages: 2,
            page_size: PAGE,
            num_kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            scheme: KvCacheScheme::Native,
            storage_dtype: DType::Bf16,
            block_size: 0,
            k_pages: k.as_ptr(),
            v_pages: v.as_ptr(),
            k_scales: core::ptr::null_mut(),
            v_scales: core::ptr::null_mut(),
            k_bf16_pages: k.as_ptr(),
            v_bf16_pages: v.as_ptr(),
            k_env_min: core::ptr::null_mut(),
            k_env_max: core::ptr::null_mut(),
            hnd_layout: false,
            native_bf16: true,
        })
        .collect();

    let up = |data: &[u8]| {
        let mut b = alloc.alloc(data.len()).expect("upload");
        b.copy_from_host(data, stream.as_ref()).expect("h2d");
        b
    };
    let u32s = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let qo_indptr_h: [u32; 3] = [0, 3, 7];
    let page_indptr_h: [u32; 3] = [0, 1, 2];
    let last_lens_h: [u32; 2] = [3, 4];
    let csr_indices = up(&u32s(&[0, 1]));
    let csr_indptr = up(&u32s(&page_indptr_h));
    let csr_lens = up(&u32s(&last_lens_h));
    let qo_indptr = up(&u32s(&qo_indptr_h));
    let row_valid = up(&[1u8; TOKENS]);
    let ids = up(&tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>());
    let positions =
        up(&[0i32, 1, 2, 0, 1, 2, 3].iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>());
    let lse = alloc.alloc(TOKENS * Q_HEADS as usize * 4).expect("lse");

    let mut sops = LiveStagingOps;
    let mut ws = AttentionWorkspace::allocate(&mut sops, 32 << 20, 16 << 20, 2).expect("ws");
    let mut pplan = PrefillPlan::new();
    ws.begin_plan_update(&mut sops).expect("begin");
    pplan.plan_prefill(
        &qo_indptr_h, &page_indptr_h, &last_lens_h,
        Q_HEADS, KV_HEADS, HEAD_DIM, PAGE, ws.view(), raw_stream, false, -1,
    );
    ws.end_plan_update(&mut sops, raw_stream);

    let fi = l
        .launches
        .iter()
        .position(|x| {
            l.kernels[x.kernel as usize] == "attn::dispatch_attention_flashinfer_prefill_bf16"
        })
        .expect("a prefill fire dispatches attention");
    let o_off = match &l.args[l.launches[fi + 1].args.start as usize] {
        Arg::Arena { at, .. } => *at,
        other => panic!("o_proj reads the attention slot, got {other:?}"),
    };

    let attn = AttnCtx {
        decode_plan: core::ptr::null_mut(),
        prefill_plan: pplan.as_ptr(),
        workspace: ws.view(),
        layers,
        q_out: core::ptr::null_mut(),
        o_out: unsafe { arena.as_ptr().cast::<u8>().add(o_off) }.cast(),
        kv_page_indices_d: csr_indices.as_ptr().cast(),
        kv_page_indptr_d: csr_indptr.as_ptr().cast(),
        kv_last_page_lens_d: csr_lens.as_ptr().cast(),
        qo_indptr_d: qo_indptr.as_ptr().cast(),
        qo_indptr_h: core::ptr::null(),
        kv_page_indptr_h: core::ptr::null(),
        num_requests: 2,
        num_pages_in_batch: 2,
        first_token: 0,
        w_page_d: core::ptr::null(),
        w_off_d: core::ptr::null(),
        row_valid_d: row_valid.as_ptr().cast(),
        lse_out_d: lse.as_ptr().cast(),
        window_left: -1,
        window_left_by_layer: Vec::new(),
        logits_soft_cap: 0.0,
        sm_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
    };

    let mut cublas_ops = LiveCublas;
    let mut cublas = CublasHandle::create(&mut cublas_ops, raw_stream).expect("cublas");
    let ctx = DispatchCtx {
        stream: raw_stream,
        cublas: cublas.handle().expect("created").cast(),
        rms_eps: 1e-6,
        rope_theta: 1e6,
        head_dim: HEAD_DIM,
        vocab: VOCAB as i32,
        gate_second: false,
        rope_interleaved: false,
        token_ids: ids.as_ptr(),
        positions: positions.as_ptr(),
        final_logit_softcap: 0.0,
        ple_dim: 0,
        scales: std::collections::BTreeMap::new(),
    };

    let mut embed_out = None;
    let mut logits_value: Option<ValueId> = None;
    for (i, launch) in l.launches.iter().enumerate() {
        if l.kernels[launch.kernel as usize] == "layout::embed_bf16"
            && let Arg::Arena { at, .. } = &l.args[launch.args.start as usize]
        {
            embed_out.get_or_insert(*at);
        }
        if let Some(Arg::Named { value, .. }) = dplan.spec(i).outs.first()
            && i == l.launches.len() - 1
        {
            logits_value = Some(*value);
        }
    }

    let mut resolver = Live {
        embed: embed_dev.as_ptr(),
        ones: ones_dev.as_ptr(),
        zeros: zeros_dev.as_ptr(),
        named: &named_bufs,
    };
    let ran = run(&l, &dplan, frame, &mut resolver, &ctx, Some(&attn), None)
        .unwrap_or_else(|e| panic!("the walk refused: {e:?}"));
    assert_eq!(ran, l.launches.len(), "every launch ran");
    stream.as_ref().synchronize().expect("the whole prefill retires");

    let mut arena_back = vec![0u8; l.arena_bytes];
    arena.copy_to_host(&mut arena_back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");
    let e = embed_out.expect("embed ran");
    for (r, t) in tokens.iter().enumerate() {
        for c in [0usize, 1, 700, 1023] {
            let want = bf16(if c % 2 == 0 { amp(*t) } else { -amp(*t) });
            let off = e + (r * HIDDEN + c) * 2;
            let got = u16::from_le_bytes([arena_back[off], arena_back[off + 1]]);
            assert_eq!(got, want, "residual row {r} col {c} drifted from the embed");
        }
    }

    let lv = logits_value.expect("the last launch writes the logits pin");
    let logits = &named_bufs[&lv];
    let mut back = vec![0u8; logits.len()];
    logits.copy_to_host(&mut back, stream.as_ref()).expect("d2h logits");
    stream.as_ref().synchronize().expect("sync");
    let logit = |r: usize, t: usize| {
        let off = (r * VOCAB + t) * 2;
        u16::from_le_bytes([back[off], back[off + 1]])
    };
    for r in 0..TOKENS {
        for t in [1usize, 2, 3, 5, 63] {
            let want = bf16(HIDDEN as f32 * amp(t as i32));
            assert_eq!(logit(r, t), want, "logit row {r} token {t}");
        }
        for t in [64usize, VOCAB - 1] {
            assert_eq!(logit(r, t), 0, "logit row {r} token {t} beyond the pattern");
        }
    }

    ws.release(&mut sops);
    cublas.release(&mut cublas_ops);
}

/// The qwen3_5 HYBRID's decode, walked end to end on device (E-gate
/// family #1's GPU smoke): 24 layers — 18 GDN (conv → prep → bf16-state
/// recurrence → gated norm) and 6 full-attention (2×-wide gated q, partial
/// rope, flashinfer) — against driver-owned conv/recurrent state slabs and
/// a per-layer seam-value pool, with synthetic weights chosen so the
/// residual stream is analytically checkable:
///
/// * every in-projection reads only EVEN channels (the embed pattern
///   alternates sign per channel, so an even-only weight sums to a real
///   value instead of cancelling to zero — finite activations, no NaN);
/// * every landing projection (`o_proj`, `down`) is zero, so the residual
///   equals the embed rows BIT-EXACTLY after 24 layers (invariant 1);
/// * the tied lm_head then dots the Gemma-folded final norm (±2) against
///   the pattern rows: `logit[r][t] = 2 · HIDDEN · amp(t)` (invariant 2).
#[test]
#[allow(clippy::too_many_lines)]
fn the_hybrid_zero_weight_decode_walks_every_launch() {
    use std::collections::BTreeMap;

    use driver_cuda_new::cuda::cublas::{CublasHandle, LiveCublas};
    use driver_cuda_new::dtype::DType;
    use driver_cuda_new::launch::{KvCacheLayerView, KvCacheScheme};
    use driver_cuda_new::model::attention_workspace::{AttentionWorkspace, LiveStagingOps};
    use driver_cuda_new::model::executor::{
        AttnCtx, DecodePlan, DispatchCtx, DispatchPlan, Frame, GdnCtx, Resolver, run,
    };
    use model::qwen_3_5::forward::facts::{Qwen35CudaFacts, Qwen35HybridFacts};
    use model::qwen_3_5::forward::qwen3_5_hybrid_cuda;
    use model_compiler::lower::{Arg, Fire, Row, lower};
    use model_compiler::trace::{FireClass, ValueId};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("hybrid zero-weight decode") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = Allocator::new();

    const HIDDEN: usize = 1024;
    const LAYERS: usize = 24;
    const KV_HEADS: i32 = 2;
    const Q_HEADS: i32 = 8;
    const HEAD_DIM: i32 = 256;
    const PAGE: i32 = 16;
    const ROWS: usize = 4;
    const VOCAB: usize = 248_320;
    const PATTERNED: usize = 64;
    // GDN geometry (the 0.8B facts' own numbers).
    const K_H: i32 = 16;
    const V_H: i32 = 16;
    const K_D: i32 = 128;
    const V_D: i32 = 128;
    const CONV_DIM: i32 = 6144;
    const CONV_K: i32 = 4;
    const SLOTS: usize = 4;

    let hybrid = Qwen35HybridFacts::qwen3_5_0_8b();
    // The LIVE L40S cuda set (`emissions.rs`), not the synthetic fixture.
    let cuda = Qwen35CudaFacts {
        state_bf16: true,
        warp_tiled: false,
        warp_tiled_max: 64,
        cached_max: 0,
        verify_stash: true,
        prefill_decode: true,
        moe_cutlass_max_rows: 0,
        moe_residual_fold: false,
        moe_shared_gate_dot: false,
        moe_streamed_experts: false,
        moe_force_general: false,
        gate_up_fused: true,
    };
    let plan = qwen3_5_hybrid_cuda(&hybrid, &cuda, FireClass::Decode);
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; ROWS];
    let l = lower(&plan, &rows, Fire { captures_across_splits: false }).expect("lowers");
    let dplan = DispatchPlan::new(&plan, &l);

    let arena = alloc.alloc(l.arena_bytes).expect("arena");
    let frame = Frame { arena: arena.as_ptr(), arena_bytes: l.arena_bytes };

    // ── Weights. Embed: the pattern rows. In-projections: even-only
    // small. Landings: zero. Norms: ones (bf16 or fp32 as consumed). ──
    let bf16 = |v: f32| (v.to_bits() >> 16) as u16;
    let amp = |t: i32| 0.5 + 0.25 * t as f32;
    let tokens: [i32; ROWS] = [1, 2, 3, 5];
    let mut embed_host = vec![0u8; VOCAB * HIDDEN * 2];
    for t in 0..PATTERNED {
        for c in 0..HIDDEN {
            let v = if c % 2 == 0 { amp(t as i32) } else { -amp(t as i32) };
            let b = bf16(v).to_le_bytes();
            embed_host[(t * HIDDEN + c) * 2] = b[0];
            embed_host[(t * HIDDEN + c) * 2 + 1] = b[1];
        }
    }
    let mut embed_dev = alloc.alloc(embed_host.len()).expect("embed");
    embed_dev.copy_from_host(&embed_host, stream.as_ref()).expect("h2d");
    // Even-channel-only in-projection bank, big enough for the widest
    // ([CONV_DIM, HIDDEN]) and reused by every in-proj and the conv (whose
    // [CONV_DIM, 1, K] flat layout just reads a prefix).
    let inproj_elems = CONV_DIM as usize * HIDDEN;
    let mut inproj_host = vec![0u8; inproj_elems * 2];
    let small = bf16(1.0 / 1024.0).to_le_bytes();
    for j in 0..inproj_elems {
        if j % 2 == 0 {
            inproj_host[j * 2] = small[0];
            inproj_host[j * 2 + 1] = small[1];
        }
    }
    let mut inproj_dev = alloc.alloc(inproj_host.len()).expect("inproj");
    inproj_dev.copy_from_host(&inproj_host, stream.as_ref()).expect("h2d");
    let ones_host: Vec<u8> =
        std::iter::repeat_n(bf16(1.0).to_le_bytes(), HIDDEN).flatten().collect();
    let mut ones_dev = alloc.alloc(ones_host.len()).expect("ones");
    ones_dev.copy_from_host(&ones_host, stream.as_ref()).expect("h2d");
    let ones_f32: Vec<u8> =
        std::iter::repeat_n(1.0f32.to_le_bytes(), V_D as usize).flatten().collect();
    let mut ones_f32_dev = alloc.alloc(ones_f32.len()).expect("ones f32");
    ones_f32_dev.copy_from_host(&ones_f32, stream.as_ref()).expect("h2d");
    let mut zeros_f32_dev = alloc.alloc(V_H as usize * 4).expect("zeros f32");
    zeros_f32_dev.memset(0, stream.as_ref()).expect("zero");
    let mut zeros_dev = alloc.alloc(2 * 3584 * HIDDEN * 2).expect("zeros");
    zeros_dev.memset(0, stream.as_ref()).expect("zero");

    // ── The seam-value pool: every Named value the lowering states,
    // allocated at fp32 width (the widest dtype any pin carries). ──
    let mut named_widths: BTreeMap<ValueId, u32> = BTreeMap::new();
    for a in &l.args {
        if let Arg::Named { value, width } = a {
            named_widths.insert(*value, *width);
        }
    }
    for i in 0..l.launches.len() {
        for a in &dplan.spec(i).outs {
            if let Arg::Named { value, width } = a {
                named_widths.insert(*value, *width);
            }
        }
    }
    let mut named_bufs: BTreeMap<ValueId, driver_cuda_new::cuda::DeviceBuffer> = named_widths
        .iter()
        .map(|(&v, &w)| {
            let mut b = alloc.alloc(ROWS * w as usize * 4).expect("pin");
            b.memset(0, stream.as_ref()).expect("zero pin");
            (v, b)
        })
        .collect();

    struct Live<'a> {
        embed: *const std::ffi::c_void,
        inproj: *const std::ffi::c_void,
        ones: *const std::ffi::c_void,
        ones_f32: *const std::ffi::c_void,
        zeros_f32: *const std::ffi::c_void,
        zeros: *const std::ffi::c_void,
        named: &'a mut BTreeMap<ValueId, driver_cuda_new::cuda::DeviceBuffer>,
    }
    impl Resolver for Live<'_> {
        fn weight(&mut self, name: &str) -> Option<*const std::ffi::c_void> {
            if name.ends_with("conv_bias") {
                return None; // the 0.8B conv has no bias — the null path
            }
            Some(if name == "embed" {
                self.embed
            } else if name.contains("a_log") {
                self.zeros_f32
            } else if name.contains("gate_norm") {
                self.ones_f32
            } else if name.contains("dt_bias") {
                self.zeros
            } else if name.contains("norm") {
                self.ones
            } else if name.contains("in_proj")
                || name.contains("conv")
                || name.contains("q_proj")
                || name.contains("k_proj")
                || name.contains("v_proj")
            {
                self.inproj
            } else {
                // o_proj, gate_up, down — the landings stay zero so the
                // residual is the embed rows, exactly.
                self.zeros
            })
        }
        fn named(&mut self, value: ValueId) -> Option<*mut std::ffi::c_void> {
            self.named.get(&value).map(|b| b.as_ptr())
        }
    }

    // ── KV pools for the SIX full-attention layers; placeholder views
    // (never dereferenced) at the GDN indices. ──
    let plane = (4 * PAGE * KV_HEADS * HEAD_DIM) as usize * 2;
    let pools: Vec<Option<(driver_cuda_new::cuda::DeviceBuffer, driver_cuda_new::cuda::DeviceBuffer)>> =
        (0..LAYERS)
            .map(|i| {
                if !hybrid.is_full_attn(u32::try_from(i).expect("layer")) {
                    return None;
                }
                let mut k = alloc.alloc(plane).expect("k pool");
                let mut v = alloc.alloc(plane).expect("v pool");
                k.memset(0, stream.as_ref()).expect("zk");
                v.memset(0, stream.as_ref()).expect("zv");
                Some((k, v))
            })
            .collect();
    let layers: Vec<KvCacheLayerView> = pools
        .iter()
        .enumerate()
        .map(|(i, kv)| {
            let (k, v) = kv.as_ref().map_or(
                (core::ptr::null_mut(), core::ptr::null_mut()),
                |(k, v)| (k.as_ptr(), v.as_ptr()),
            );
            KvCacheLayerView {
                layer: i32::try_from(i).expect("layer"),
                source_layer: i32::try_from(i).expect("layer"),
                num_pages: 4,
                page_size: PAGE,
                num_kv_heads: KV_HEADS,
                head_dim: HEAD_DIM,
                scheme: KvCacheScheme::Native,
                storage_dtype: DType::Bf16,
                block_size: 0,
                k_pages: k,
                v_pages: v,
                k_scales: core::ptr::null_mut(),
                v_scales: core::ptr::null_mut(),
                k_bf16_pages: k,
                v_bf16_pages: v,
                k_env_min: core::ptr::null_mut(),
                k_env_max: core::ptr::null_mut(),
                hnd_layout: false,
                native_bf16: true,
            }
        })
        .collect();

    // ── GDN state: conv + recurrent slabs for the EIGHTEEN linear
    // layers, slot-indirected. ──
    let conv_stride = (CONV_K * CONV_DIM) as usize;
    let state_stride = (V_H * K_D * V_D) as usize;
    let gdn_slabs: Vec<Option<(driver_cuda_new::cuda::DeviceBuffer, driver_cuda_new::cuda::DeviceBuffer)>> =
        (0..LAYERS)
            .map(|i| {
                if hybrid.is_full_attn(u32::try_from(i).expect("layer")) {
                    return None;
                }
                let mut c = alloc.alloc(SLOTS * conv_stride * 2).expect("conv slab");
                let mut s = alloc.alloc(SLOTS * state_stride * 2).expect("state slab");
                c.memset(0, stream.as_ref()).expect("zc");
                s.memset(0, stream.as_ref()).expect("zs");
                Some((c, s))
            })
            .collect();
    let up = |data: &[u8]| {
        let mut b = alloc.alloc(data.len()).expect("upload");
        b.copy_from_host(data, stream.as_ref()).expect("h2d");
        b
    };
    let slot_ids =
        up(&[0i32, 1, 2, 3].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    let gdn = GdnCtx {
        k_h: K_H,
        v_h: V_H,
        k_d: K_D,
        v_d: V_D,
        conv_dim: CONV_DIM,
        conv_k: CONV_K,
        conv_state: gdn_slabs
            .iter()
            .map(|s| s.as_ref().map_or(0, |(c, _)| c.as_ptr() as u64))
            .collect(),
        conv_stride_elems: i64::try_from(conv_stride).expect("stride"),
        recurrent_state: gdn_slabs
            .iter()
            .map(|s| s.as_ref().map_or(0, |(_, r)| r.as_ptr() as u64))
            .collect(),
        state_stride_elems: i64::try_from(state_stride).expect("stride"),
        slot_ids_d: slot_ids.as_ptr().cast(),
        write_state: true,
    };

    // ── CSRs, write descriptors, plan, workspace. ──
    let u32s = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let csr_indices = up(&u32s(&[0, 1, 2, 3]));
    let csr_indptr = up(&u32s(&[0, 1, 2, 3, 4]));
    let csr_lens = up(&u32s(&[1, 1, 1, 1]));
    let w_page = up(&u32s(&[0, 1, 2, 3]));
    let w_off = up(&u32s(&[0, 0, 0, 0]));
    let row_valid = up(&[1u8, 1, 1, 1]);
    let ids = up(&tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>());
    let positions =
        up(&[0i32, 0, 0, 0].iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>());
    let lse = alloc.alloc(ROWS * Q_HEADS as usize * 4).expect("lse");

    let mut sops = LiveStagingOps;
    let mut ws = AttentionWorkspace::allocate(&mut sops, 32 << 20, 16 << 20, 2).expect("ws");
    let mut dplan_cache = DecodePlan::new();
    ws.begin_plan_update(&mut sops).expect("begin");
    dplan_cache.plan_decode(
        &[0, 1, 2, 3, 4],
        Q_HEADS,
        KV_HEADS,
        HEAD_DIM,
        PAGE,
        ws.view(),
        raw_stream,
        false,
        -1,
    );
    ws.end_plan_update(&mut sops, raw_stream);

    let fi = l
        .launches
        .iter()
        .position(|x| {
            l.kernels[x.kernel as usize] == "attn::dispatch_attention_flashinfer_decode"
        })
        .expect("the hybrid decode dispatches attention");
    let q_pin_value = match &l.args[l.launches[fi].args.start as usize] {
        Arg::Named { value, .. } => *value,
        other => panic!("the dispatch's q is a pin, got {other:?}"),
    };
    // The attention output is guard-owned (the dispatch launch records no
    // SSA outputs); the launch AFTER the dispatch reads it first — the
    // sigmoid output gate's `x`.
    let o_out: *mut std::ffi::c_void =
        match &l.args[l.launches[fi + 1].args.start as usize] {
            Arg::Arena { at, .. } => unsafe { arena.as_ptr().cast::<u8>().add(*at) }.cast(),
            Arg::Named { value, .. } => named_bufs[value].as_ptr(),
            other => panic!("the gate reads the attention slot, got {other:?}"),
        };

    let attn = AttnCtx {
        decode_plan: dplan_cache.as_ptr(),
        prefill_plan: core::ptr::null_mut(),
        workspace: ws.view(),
        layers,
        q_out: named_bufs[&q_pin_value].as_ptr(),
        o_out,
        kv_page_indices_d: csr_indices.as_ptr().cast(),
        kv_page_indptr_d: csr_indptr.as_ptr().cast(),
        kv_last_page_lens_d: csr_lens.as_ptr().cast(),
        // The hybrid writes KV through the EXPLICIT kernel, which walks
        // the qo CSR even on decode — one row per request, trivially.
        qo_indptr_d: csr_indptr.as_ptr().cast(),
        qo_indptr_h: core::ptr::null(),
        kv_page_indptr_h: core::ptr::null(),
        num_requests: i32::try_from(ROWS).expect("rows"),
        num_pages_in_batch: 4,
        first_token: 0,
        w_page_d: w_page.as_ptr().cast(),
        w_off_d: w_off.as_ptr().cast(),
        row_valid_d: row_valid.as_ptr().cast(),
        lse_out_d: lse.as_ptr().cast(),
        window_left: -1,
        window_left_by_layer: Vec::new(),
        logits_soft_cap: 0.0,
        sm_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
    };

    let mut cublas_ops = LiveCublas;
    let mut cublas = CublasHandle::create(&mut cublas_ops, raw_stream).expect("cublas");
    let ctx = DispatchCtx {
        stream: raw_stream,
        cublas: cublas.handle().expect("created").cast(),
        rms_eps: 1e-6,
        rope_theta: 1e6,
        head_dim: HEAD_DIM,
        vocab: i32::try_from(VOCAB).expect("vocab"),
        gate_second: false,
        rope_interleaved: false,
        token_ids: ids.as_ptr(),
        positions: positions.as_ptr(),
        final_logit_softcap: 0.0,
        ple_dim: 0,
        scales: std::collections::BTreeMap::new(),
    };

    // ── The walk. ──
    let mut resolver = Live {
        embed: embed_dev.as_ptr(),
        inproj: inproj_dev.as_ptr(),
        ones: ones_dev.as_ptr(),
        ones_f32: ones_f32_dev.as_ptr(),
        zeros_f32: zeros_f32_dev.as_ptr(),
        zeros: zeros_dev.as_ptr(),
        named: &mut named_bufs,
    };
    let mut embed_out = None;
    let mut logits_value: Option<ValueId> = None;
    for (i, launch) in l.launches.iter().enumerate() {
        if l.kernels[launch.kernel as usize] == "layout::embed_bf16"
            && let Arg::Arena { at, .. } = &l.args[launch.args.start as usize]
        {
            embed_out.get_or_insert(*at);
        }
        if let Some(Arg::Named { value, .. }) = dplan.spec(i).outs.first()
            && i == l.launches.len() - 1
        {
            logits_value = Some(*value);
        }
    }
    let per_launch_sync = std::env::var("HYBRID_SMOKE_SYNC").is_ok();
    let ran = if per_launch_sync {
        use driver_cuda_new::model::executor::{bind, dispatch};
        for (i, launch) in l.launches.iter().enumerate() {
            let kernel = l.kernels[launch.kernel as usize].clone();
            let bound = bind(&l, launch, frame, &mut resolver)
                .unwrap_or_else(|e| panic!("launch {i} {kernel}: bind {e:?}"));
            dispatch(&bound, dplan.spec(i), frame, &mut resolver, &ctx, Some(&attn), Some(&gdn))
                .unwrap_or_else(|e| panic!("launch {i} {kernel}: dispatch {e:?}"));
            stream
                .as_ref()
                .synchronize()
                .unwrap_or_else(|e| panic!("launch {i} {kernel} left the stream poisoned: {e:?}"));
        }
        l.launches.len()
    } else {
        run(&l, &dplan, frame, &mut resolver, &ctx, Some(&attn), Some(&gdn))
            .unwrap_or_else(|e| panic!("the hybrid walk refused: {e:?}"))
    };
    assert_eq!(ran, l.launches.len(), "every launch ran");
    stream.as_ref().synchronize().expect("the whole hybrid decode retires");

    // ── Invariant 1: the residual equals the embed rows, bit-exactly —
    // 18 GDN and 6 attention landings all through zero projections. ──
    let mut arena_back = vec![0u8; l.arena_bytes];
    arena.copy_to_host(&mut arena_back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");
    let e = embed_out.expect("embed ran");
    for (r, t) in tokens.iter().enumerate() {
        for c in [0usize, 1, 700, 1023] {
            let want = bf16(if c % 2 == 0 { amp(*t) } else { -amp(*t) });
            let off = e + (r * HIDDEN + c) * 2;
            let got = u16::from_le_bytes([arena_back[off], arena_back[off + 1]]);
            assert_eq!(got, want, "residual row {r} col {c} drifted from the embed");
        }
    }

    // ── Invariant 2: the logits are the tied lm_head's exact algebra —
    // the GEMMA final norm folds (1 + 1), so ±2 against ±amp(t):
    // logit[r][t] = 2 · HIDDEN · amp(t). ──
    let lv = logits_value.expect("the last launch writes the logits pin");
    let logits = &named_bufs[&lv];
    let mut back = vec![0u8; logits.len()];
    logits.copy_to_host(&mut back, stream.as_ref()).expect("d2h logits");
    stream.as_ref().synchronize().expect("sync");
    let logit = |r: usize, t: usize| {
        let off = (r * VOCAB + t) * 2;
        u16::from_le_bytes([back[off], back[off + 1]])
    };
    for r in 0..ROWS {
        for t in [1usize, 2, 3, 5, 63] {
            let want = bf16(2.0 * HIDDEN as f32 * amp(i32::try_from(t).expect("t")));
            assert_eq!(logit(r, t), want, "logit row {r} token {t}");
        }
        for t in [64usize, VOCAB - 1] {
            assert_eq!(logit(r, t), 0, "logit row {r} token {t} beyond the pattern");
        }
    }

    ws.release(&mut sops);
    cublas.release(&mut cublas_ops);
}

/// The hybrid's PREFILL, walked the same way: two requests over seven
/// tokens, the conv prefill walk + chunked FLA recurrence advancing the
/// GDN state slabs, flashinfer prefill on the full-attention layers.
/// Same synthetic weights, same two invariants as the decode smoke.
#[test]
#[allow(clippy::too_many_lines)]
fn the_hybrid_zero_weight_prefill_walks_every_launch() {
    use std::collections::BTreeMap;

    use driver_cuda_new::cuda::cublas::{CublasHandle, LiveCublas};
    use driver_cuda_new::dtype::DType;
    use driver_cuda_new::launch::{KvCacheLayerView, KvCacheScheme};
    use driver_cuda_new::model::attention_workspace::{AttentionWorkspace, LiveStagingOps};
    use driver_cuda_new::model::executor::{
        AttnCtx, DispatchCtx, DispatchPlan, Frame, GdnCtx, PrefillPlan, Resolver, run,
    };
    use model::qwen_3_5::forward::facts::{Qwen35CudaFacts, Qwen35HybridFacts};
    use model::qwen_3_5::forward::qwen3_5_hybrid_cuda;
    use model_compiler::lower::{Arg, Fire, Row, lower};
    use model_compiler::trace::{FireClass, ValueId};

    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("hybrid zero-weight prefill") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let raw_stream = stream.as_ref().as_raw().cast::<std::ffi::c_void>();
    let alloc = Allocator::new();

    const HIDDEN: usize = 1024;
    const LAYERS: usize = 24;
    const KV_HEADS: i32 = 2;
    const Q_HEADS: i32 = 8;
    const HEAD_DIM: i32 = 256;
    const PAGE: i32 = 16;
    const TOKENS: usize = 7;
    const REQUESTS: usize = 2;
    const VOCAB: usize = 248_320;
    const PATTERNED: usize = 64;
    const K_H: i32 = 16;
    const V_H: i32 = 16;
    const K_D: i32 = 128;
    const V_D: i32 = 128;
    const CONV_DIM: i32 = 6144;
    const CONV_K: i32 = 4;
    const SLOTS: usize = 4;

    let hybrid = Qwen35HybridFacts::qwen3_5_0_8b();
    let cuda = Qwen35CudaFacts {
        state_bf16: true,
        warp_tiled: false,
        warp_tiled_max: 64,
        cached_max: 0,
        verify_stash: true,
        prefill_decode: true,
        moe_cutlass_max_rows: 0,
        moe_residual_fold: false,
        moe_shared_gate_dot: false,
        moe_streamed_experts: false,
        moe_force_general: false,
        gate_up_fused: true,
    };
    let plan = qwen3_5_hybrid_cuda(&hybrid, &cuda, FireClass::Prefill);
    let rows: Vec<Row> = vec![Row { samples: true, ..Row::default() }; TOKENS];
    let l = lower(&plan, &rows, Fire { captures_across_splits: false }).expect("lowers");
    let dplan = DispatchPlan::new(&plan, &l);

    let arena = alloc.alloc(l.arena_bytes).expect("arena");
    let frame = Frame { arena: arena.as_ptr(), arena_bytes: l.arena_bytes };

    let bf16 = |v: f32| (v.to_bits() >> 16) as u16;
    let amp = |t: i32| 0.5 + 0.25 * t as f32;
    let tokens: [i32; TOKENS] = [1, 2, 3, 5, 7, 11, 13];
    let mut embed_host = vec![0u8; VOCAB * HIDDEN * 2];
    for t in 0..PATTERNED {
        for c in 0..HIDDEN {
            let v = if c % 2 == 0 { amp(t as i32) } else { -amp(t as i32) };
            let b = bf16(v).to_le_bytes();
            embed_host[(t * HIDDEN + c) * 2] = b[0];
            embed_host[(t * HIDDEN + c) * 2 + 1] = b[1];
        }
    }
    let mut embed_dev = alloc.alloc(embed_host.len()).expect("embed");
    embed_dev.copy_from_host(&embed_host, stream.as_ref()).expect("h2d");
    let inproj_elems = CONV_DIM as usize * HIDDEN;
    let mut inproj_host = vec![0u8; inproj_elems * 2];
    let small = bf16(1.0 / 1024.0).to_le_bytes();
    for j in 0..inproj_elems {
        if j % 2 == 0 {
            inproj_host[j * 2] = small[0];
            inproj_host[j * 2 + 1] = small[1];
        }
    }
    let mut inproj_dev = alloc.alloc(inproj_host.len()).expect("inproj");
    inproj_dev.copy_from_host(&inproj_host, stream.as_ref()).expect("h2d");
    let ones_host: Vec<u8> =
        std::iter::repeat_n(bf16(1.0).to_le_bytes(), HIDDEN).flatten().collect();
    let mut ones_dev = alloc.alloc(ones_host.len()).expect("ones");
    ones_dev.copy_from_host(&ones_host, stream.as_ref()).expect("h2d");
    let ones_f32: Vec<u8> =
        std::iter::repeat_n(1.0f32.to_le_bytes(), V_D as usize).flatten().collect();
    let mut ones_f32_dev = alloc.alloc(ones_f32.len()).expect("ones f32");
    ones_f32_dev.copy_from_host(&ones_f32, stream.as_ref()).expect("h2d");
    let mut zeros_f32_dev = alloc.alloc(V_H as usize * 4).expect("zeros f32");
    zeros_f32_dev.memset(0, stream.as_ref()).expect("zero");
    let mut zeros_dev = alloc.alloc(2 * 3584 * HIDDEN * 2).expect("zeros");
    zeros_dev.memset(0, stream.as_ref()).expect("zero");

    let mut named_widths: BTreeMap<ValueId, u32> = BTreeMap::new();
    for a in &l.args {
        if let Arg::Named { value, width } = a {
            named_widths.insert(*value, *width);
        }
    }
    for i in 0..l.launches.len() {
        for a in &dplan.spec(i).outs {
            if let Arg::Named { value, width } = a {
                named_widths.insert(*value, *width);
            }
        }
    }
    let mut named_bufs: BTreeMap<ValueId, driver_cuda_new::cuda::DeviceBuffer> = named_widths
        .iter()
        .map(|(&v, &w)| {
            let mut b = alloc.alloc(TOKENS * w as usize * 4).expect("pin");
            b.memset(0, stream.as_ref()).expect("zero pin");
            (v, b)
        })
        .collect();

    struct Live<'a> {
        embed: *const std::ffi::c_void,
        inproj: *const std::ffi::c_void,
        ones: *const std::ffi::c_void,
        ones_f32: *const std::ffi::c_void,
        zeros_f32: *const std::ffi::c_void,
        zeros: *const std::ffi::c_void,
        named: &'a mut BTreeMap<ValueId, driver_cuda_new::cuda::DeviceBuffer>,
    }
    impl Resolver for Live<'_> {
        fn weight(&mut self, name: &str) -> Option<*const std::ffi::c_void> {
            if name.ends_with("conv_bias") {
                return None;
            }
            Some(if name == "embed" {
                self.embed
            } else if name.contains("a_log") {
                self.zeros_f32
            } else if name.contains("gate_norm") {
                self.ones_f32
            } else if name.contains("dt_bias") {
                self.zeros
            } else if name.contains("norm") {
                self.ones
            } else if name.contains("in_proj")
                || name.contains("conv")
                || name.contains("q_proj")
                || name.contains("k_proj")
                || name.contains("v_proj")
            {
                self.inproj
            } else {
                self.zeros
            })
        }
        fn named(&mut self, value: ValueId) -> Option<*mut std::ffi::c_void> {
            self.named.get(&value).map(|b| b.as_ptr())
        }
    }

    let plane = (2 * PAGE * KV_HEADS * HEAD_DIM) as usize * 2;
    let pools: Vec<Option<(driver_cuda_new::cuda::DeviceBuffer, driver_cuda_new::cuda::DeviceBuffer)>> =
        (0..LAYERS)
            .map(|i| {
                if !hybrid.is_full_attn(u32::try_from(i).expect("layer")) {
                    return None;
                }
                let mut k = alloc.alloc(plane).expect("k pool");
                let mut v = alloc.alloc(plane).expect("v pool");
                k.memset(0, stream.as_ref()).expect("zk");
                v.memset(0, stream.as_ref()).expect("zv");
                Some((k, v))
            })
            .collect();
    let layers: Vec<KvCacheLayerView> = pools
        .iter()
        .enumerate()
        .map(|(i, kv)| {
            let (k, v) = kv.as_ref().map_or(
                (core::ptr::null_mut(), core::ptr::null_mut()),
                |(k, v)| (k.as_ptr(), v.as_ptr()),
            );
            KvCacheLayerView {
                layer: i32::try_from(i).expect("layer"),
                source_layer: i32::try_from(i).expect("layer"),
                num_pages: 2,
                page_size: PAGE,
                num_kv_heads: KV_HEADS,
                head_dim: HEAD_DIM,
                scheme: KvCacheScheme::Native,
                storage_dtype: DType::Bf16,
                block_size: 0,
                k_pages: k,
                v_pages: v,
                k_scales: core::ptr::null_mut(),
                v_scales: core::ptr::null_mut(),
                k_bf16_pages: k,
                v_bf16_pages: v,
                k_env_min: core::ptr::null_mut(),
                k_env_max: core::ptr::null_mut(),
                hnd_layout: false,
                native_bf16: true,
            }
        })
        .collect();

    let conv_stride = (CONV_K * CONV_DIM) as usize;
    let state_stride = (V_H * K_D * V_D) as usize;
    let gdn_slabs: Vec<Option<(driver_cuda_new::cuda::DeviceBuffer, driver_cuda_new::cuda::DeviceBuffer)>> =
        (0..LAYERS)
            .map(|i| {
                if hybrid.is_full_attn(u32::try_from(i).expect("layer")) {
                    return None;
                }
                let mut c = alloc.alloc(SLOTS * conv_stride * 2).expect("conv slab");
                let mut s = alloc.alloc(SLOTS * state_stride * 2).expect("state slab");
                c.memset(0, stream.as_ref()).expect("zc");
                s.memset(0, stream.as_ref()).expect("zs");
                Some((c, s))
            })
            .collect();
    let up = |data: &[u8]| {
        let mut b = alloc.alloc(data.len()).expect("upload");
        b.copy_from_host(data, stream.as_ref()).expect("h2d");
        b
    };
    let slot_ids = up(&[0i32, 1].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    let gdn = GdnCtx {
        k_h: K_H,
        v_h: V_H,
        k_d: K_D,
        v_d: V_D,
        conv_dim: CONV_DIM,
        conv_k: CONV_K,
        conv_state: gdn_slabs
            .iter()
            .map(|s| s.as_ref().map_or(0, |(c, _)| c.as_ptr() as u64))
            .collect(),
        conv_stride_elems: i64::try_from(conv_stride).expect("stride"),
        recurrent_state: gdn_slabs
            .iter()
            .map(|s| s.as_ref().map_or(0, |(_, r)| r.as_ptr() as u64))
            .collect(),
        state_stride_elems: i64::try_from(state_stride).expect("stride"),
        slot_ids_d: slot_ids.as_ptr().cast(),
        write_state: true,
    };

    let u32s = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let qo_indptr_h: [u32; 3] = [0, 3, 7];
    let page_indptr_h: [u32; 3] = [0, 1, 2];
    let last_lens_h: [u32; 2] = [3, 4];
    let csr_indices = up(&u32s(&[0, 1]));
    let csr_indptr = up(&u32s(&page_indptr_h));
    let csr_lens = up(&u32s(&last_lens_h));
    let qo_indptr = up(&u32s(&qo_indptr_h));
    let row_valid = up(&[1u8; TOKENS]);
    let ids = up(&tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>());
    let positions =
        up(&[0i32, 1, 2, 0, 1, 2, 3].iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>());
    let lse = alloc.alloc(TOKENS * Q_HEADS as usize * 4).expect("lse");

    let mut sops = LiveStagingOps;
    let mut ws = AttentionWorkspace::allocate(&mut sops, 32 << 20, 16 << 20, 2).expect("ws");
    let mut pplan = PrefillPlan::new();
    ws.begin_plan_update(&mut sops).expect("begin");
    pplan.plan_prefill(
        &qo_indptr_h, &page_indptr_h, &last_lens_h,
        Q_HEADS, KV_HEADS, HEAD_DIM, PAGE, ws.view(), raw_stream, false, -1,
    );
    ws.end_plan_update(&mut sops, raw_stream);

    let fi = l
        .launches
        .iter()
        .position(|x| {
            l.kernels[x.kernel as usize] == "attn::dispatch_attention_flashinfer_prefill_bf16"
        })
        .expect("the hybrid prefill dispatches attention");
    let q_pin_value = match &l.args[l.launches[fi].args.start as usize] {
        Arg::Named { value, .. } => *value,
        other => panic!("the dispatch's q is a pin, got {other:?}"),
    };
    let o_out: *mut std::ffi::c_void =
        match &l.args[l.launches[fi + 1].args.start as usize] {
            Arg::Arena { at, .. } => unsafe { arena.as_ptr().cast::<u8>().add(*at) }.cast(),
            Arg::Named { value, .. } => named_bufs[value].as_ptr(),
            other => panic!("the gate reads the attention slot, got {other:?}"),
        };

    let attn = AttnCtx {
        decode_plan: core::ptr::null_mut(),
        prefill_plan: pplan.as_ptr(),
        workspace: ws.view(),
        layers,
        q_out: named_bufs[&q_pin_value].as_ptr(),
        o_out,
        kv_page_indices_d: csr_indices.as_ptr().cast(),
        kv_page_indptr_d: csr_indptr.as_ptr().cast(),
        kv_last_page_lens_d: csr_lens.as_ptr().cast(),
        qo_indptr_d: qo_indptr.as_ptr().cast(),
        qo_indptr_h: core::ptr::null(),
        kv_page_indptr_h: core::ptr::null(),
        num_requests: i32::try_from(REQUESTS).expect("requests"),
        num_pages_in_batch: 2,
        first_token: 0,
        w_page_d: core::ptr::null(),
        w_off_d: core::ptr::null(),
        row_valid_d: row_valid.as_ptr().cast(),
        lse_out_d: lse.as_ptr().cast(),
        window_left: -1,
        window_left_by_layer: Vec::new(),
        logits_soft_cap: 0.0,
        sm_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
    };

    let mut cublas_ops = LiveCublas;
    let mut cublas = CublasHandle::create(&mut cublas_ops, raw_stream).expect("cublas");
    let ctx = DispatchCtx {
        stream: raw_stream,
        cublas: cublas.handle().expect("created").cast(),
        rms_eps: 1e-6,
        rope_theta: 1e6,
        head_dim: HEAD_DIM,
        vocab: i32::try_from(VOCAB).expect("vocab"),
        gate_second: false,
        rope_interleaved: false,
        token_ids: ids.as_ptr(),
        positions: positions.as_ptr(),
        final_logit_softcap: 0.0,
        ple_dim: 0,
        scales: std::collections::BTreeMap::new(),
    };

    let mut resolver = Live {
        embed: embed_dev.as_ptr(),
        inproj: inproj_dev.as_ptr(),
        ones: ones_dev.as_ptr(),
        ones_f32: ones_f32_dev.as_ptr(),
        zeros_f32: zeros_f32_dev.as_ptr(),
        zeros: zeros_dev.as_ptr(),
        named: &mut named_bufs,
    };
    let mut embed_out = None;
    let mut logits_value: Option<ValueId> = None;
    for (i, launch) in l.launches.iter().enumerate() {
        if l.kernels[launch.kernel as usize] == "layout::embed_bf16"
            && let Arg::Arena { at, .. } = &l.args[launch.args.start as usize]
        {
            embed_out.get_or_insert(*at);
        }
        if let Some(Arg::Named { value, .. }) = dplan.spec(i).outs.first()
            && i == l.launches.len() - 1
        {
            logits_value = Some(*value);
        }
    }
    let ran = run(&l, &dplan, frame, &mut resolver, &ctx, Some(&attn), Some(&gdn))
        .unwrap_or_else(|e| panic!("the hybrid prefill walk refused: {e:?}"));
    assert_eq!(ran, l.launches.len(), "every launch ran");
    stream.as_ref().synchronize().expect("the whole hybrid prefill retires");

    let mut arena_back = vec![0u8; l.arena_bytes];
    arena.copy_to_host(&mut arena_back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");
    let e = embed_out.expect("embed ran");
    for (r, t) in tokens.iter().enumerate() {
        for c in [0usize, 1, 700, 1023] {
            let want = bf16(if c % 2 == 0 { amp(*t) } else { -amp(*t) });
            let off = e + (r * HIDDEN + c) * 2;
            let got = u16::from_le_bytes([arena_back[off], arena_back[off + 1]]);
            assert_eq!(got, want, "residual row {r} col {c} drifted from the embed");
        }
    }

    let lv = logits_value.expect("the last launch writes the logits pin");
    let logits = &named_bufs[&lv];
    let mut back = vec![0u8; logits.len()];
    logits.copy_to_host(&mut back, stream.as_ref()).expect("d2h logits");
    stream.as_ref().synchronize().expect("sync");
    let logit = |r: usize, t: usize| {
        let off = (r * VOCAB + t) * 2;
        u16::from_le_bytes([back[off], back[off + 1]])
    };
    for r in 0..TOKENS {
        for t in [1usize, 2, 3, 5, 63] {
            let want = bf16(2.0 * HIDDEN as f32 * amp(i32::try_from(t).expect("t")));
            assert_eq!(logit(r, t), want, "logit row {r} token {t}");
        }
        for t in [64usize, VOCAB - 1] {
            assert_eq!(logit(r, t), 0, "logit row {r} token {t} beyond the pattern");
        }
    }

    ws.release(&mut sops);
    cublas.release(&mut cublas_ops);
}
