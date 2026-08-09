//! The linker proves the ABI: `driver_abi::local` DECLARES the thirteen
//! `pie_cuda_*` symbols (the engine's consumer side), this crate's `abi`
//! feature DEFINES them, and this test resolving the declaration against
//! the definition is the same proof shape as the launch bridge — a
//! drifted signature is a link error, not a runtime surprise.

#![cfg(all(feature = "_cuda", feature = "abi"))]

use driver_abi::local::{
    PIE_DRIVER_ABI_VERSION, PIE_STATUS_INVALID_ARGUMENT, PIE_STATUS_OK, PieDriverCaps,
    PieDriverCreateDesc,
};

mod common;
#[allow(unused_imports)] // abi tests take only the guard
use common::gpu_guard;

#[test]
fn the_shell_answers_the_engines_own_declarations() {
    // Force the defining objects into this binary: an rlib's members are
    // pulled on REFERENCE, and the `extern` declarations alone reference
    // nothing Rust-side. With the definitions present, the declarations
    // below resolve to them — which is the link-level proof.
    let _providers: [*const (); 13] = [
        driver_cuda_new::abi_shell::pie_cuda_create as *const (),
        driver_cuda_new::abi_shell::pie_cuda_load_model as *const (),
        driver_cuda_new::abi_shell::pie_cuda_register_program as *const (),
        driver_cuda_new::abi_shell::pie_cuda_register_channel as *const (),
        driver_cuda_new::abi_shell::pie_cuda_bind_instance as *const (),
        driver_cuda_new::abi_shell::pie_cuda_launch as *const (),
        driver_cuda_new::abi_shell::pie_cuda_encode as *const (),
        driver_cuda_new::abi_shell::pie_cuda_copy_kv as *const (),
        driver_cuda_new::abi_shell::pie_cuda_copy_state as *const (),
        driver_cuda_new::abi_shell::pie_cuda_resize_pool as *const (),
        driver_cuda_new::abi_shell::pie_cuda_close_instance as *const (),
        driver_cuda_new::abi_shell::pie_cuda_close_channel as *const (),
        driver_cuda_new::abi_shell::pie_cuda_destroy as *const (),
    ];
    // A wrong version is refused with null, before any state exists.
    let bad = PieDriverCreateDesc { abi_version: 1, ..Default::default() };
    let d = unsafe { driver_abi::local::pie_cuda_create(&bad, std::ptr::null_mut()) };
    assert!(d.is_null(), "a mismatched ABI version must refuse");

    // The real version creates, hands back live caps, and destroys.
    let desc =
        PieDriverCreateDesc { abi_version: PIE_DRIVER_ABI_VERSION, ..Default::default() };
    let mut caps = PieDriverCaps { json_bytes: std::ptr::null(), json_len: 0 };
    let d = unsafe { driver_abi::local::pie_cuda_create(&desc, &mut caps) };
    assert!(!d.is_null(), "create with the pinned ABI version");
    assert!(caps.json_len > 0, "caps came back");
    let json = unsafe { std::slice::from_raw_parts(caps.json_bytes, caps.json_len) };
    assert!(std::str::from_utf8(json).expect("utf8").contains("driver-cuda-new"));

    // The stated refusals refuse with the stated code, and the closes
    // close.
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_load_model(d, std::ptr::null(), std::ptr::null_mut()) },
        PIE_STATUS_INVALID_ARGUMENT,
        "a null load desc is an argument error, not a refusal"
    );
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_close_instance(d, 7) },
        PIE_STATUS_OK
    );
    let load = driver_abi::local::PieModelLoadDesc::default();
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_load_model(d, &load, std::ptr::null_mut()) },
        PIE_STATUS_INVALID_ARGUMENT,
        "an empty snapshot_dir is an argument error"
    );
    unsafe { driver_abi::local::pie_cuda_destroy(d) };
}

/// `load_model` over a REAL snapshot: the boot TOML carries the
/// descriptor path (the C++ shell's own channel), the loader parses the
/// HF safetensors layout, ~1.2 GB lands on the device, and the caps JSON
/// answers with the parsed facts. GPU + checkpoint required; skips
/// without either.
#[test]
fn load_model_loads_a_real_snapshot_through_the_abi() {
    let _gpu = gpu_guard();
    use driver_abi::local::{PieBytes, PieModelLoadDesc};

    let home = std::env::var("HOME").expect("HOME");
    let snaps =
        std::path::PathBuf::from(&home).join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            p.join("model.safetensors").is_file().then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached Qwen3-0.6B");
        return;
    };
    let descriptor = std::path::PathBuf::from(std::env::var("PIE_TEST_SCRATCH").unwrap_or_else(
        |_| "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad".into(),
    ))
    .join("qwen3_descriptor.json");
    if !descriptor.is_file() {
        eprintln!("skipped: no generated descriptor at {descriptor:?}");
        return;
    }

    let boot = format!("[model]\ndescriptor = \"{}\"\n", descriptor.display());
    let desc = PieDriverCreateDesc {
        abi_version: PIE_DRIVER_ABI_VERSION,
        config_bytes: PieBytes { ptr: boot.as_ptr(), len: boot.len() },
        ..Default::default()
    };
    let d = unsafe { driver_abi::local::pie_cuda_create(&desc, std::ptr::null_mut()) };
    assert!(!d.is_null());

    let snap_str = snap.to_string_lossy().into_owned();
    let load = PieModelLoadDesc {
        snapshot_dir: PieBytes { ptr: snap_str.as_ptr(), len: snap_str.len() },
        ..Default::default()
    };
    let mut caps = PieDriverCaps { json_bytes: std::ptr::null(), json_len: 0 };
    let status = unsafe { driver_abi::local::pie_cuda_load_model(d, &load, &mut caps) };
    assert_eq!(status, PIE_STATUS_OK, "the real snapshot loads");
    let json = unsafe { std::slice::from_raw_parts(caps.json_bytes, caps.json_len) };
    let json = std::str::from_utf8(json).expect("utf8");
    assert!(json.contains("\"model_type\":\"qwen3\""), "caps carry the parsed facts: {json}");
    assert!(json.contains("\"layers\":28"), "{json}");

    unsafe { driver_abi::local::pie_cuda_destroy(d) };
}

/// The id lifecycle: registering the same program hash twice answers one
/// id; binding requires a registered program; a requested instance id is
/// honored; closing is idempotent and actually closes (a rebind of the
/// same id succeeds after close, refuses before).
#[test]
fn the_registries_run_the_id_lifecycle() {
    use driver_abi::local::{PieInstanceBinding, PieInstanceDesc, PieProgramDesc};

    let desc =
        PieDriverCreateDesc { abi_version: PIE_DRIVER_ABI_VERSION, ..Default::default() };
    let d = unsafe { driver_abi::local::pie_cuda_create(&desc, std::ptr::null_mut()) };
    assert!(!d.is_null());

    let prog = PieProgramDesc { program_hash: 0xC3C3, ..Default::default() };
    let mut id1 = 0u64;
    let mut id2 = 0u64;
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_register_program(d, &prog, &mut id1) },
        PIE_STATUS_OK
    );
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_register_program(d, &prog, &mut id2) },
        PIE_STATUS_OK
    );
    assert_eq!(id1, id2, "the hash is the dedup key");

    let unbound = PieInstanceDesc { program_id: 999, ..Default::default() };
    let mut binding = PieInstanceBinding::default();
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_bind_instance(d, &unbound, &mut binding) },
        PIE_STATUS_INVALID_ARGUMENT,
        "an unregistered program refuses the bind"
    );

    let inst = PieInstanceDesc {
        program_id: id1,
        requested_instance_id: 42,
        geometry_class: 7,
        ..Default::default()
    };
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_bind_instance(d, &inst, &mut binding) },
        PIE_STATUS_OK
    );
    assert_eq!(binding.instance_id, 42, "the requested id is honored");
    assert_eq!(binding.geometry_class, 7, "the geometry echoes");

    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_bind_instance(d, &inst, &mut binding) },
        PIE_STATUS_INVALID_ARGUMENT,
        "an id in use refuses"
    );
    assert_eq!(unsafe { driver_abi::local::pie_cuda_close_instance(d, 42) }, PIE_STATUS_OK);
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_bind_instance(d, &inst, &mut binding) },
        PIE_STATUS_OK,
        "closed means reusable"
    );

    unsafe { driver_abi::local::pie_cuda_destroy(d) };
}

/// The whole ABI, end to end: create → load the real checkpoint →
/// register → bind → LAUNCH one decode frame — a single token over one
/// KV page — and the shell runs the actual forward on the device,
/// publishes the terminal cell, and notifies the runtime. This is the
/// engine's own call sequence, driven through the engine's own
/// declarations.
#[test]
fn a_real_decode_frame_launches_through_the_abi() {
    let _gpu = gpu_guard();
    use std::sync::atomic::{AtomicU64, Ordering};

    use driver_abi::local::{
        PIE_TERMINAL_OUTCOME_PENDING, PIE_TERMINAL_OUTCOME_SUCCESS, PieBytes, PieCompletion,
        PieFrameDesc, PieInstanceBinding, PieInstanceDesc, PieModelLoadDesc, PieProgramDesc,
        PieRuntimeCallbacks, PieStepDesc, PieTerminalCell, PieTerminalCellPtrSlice,
        PieU32Slice, PieU64Slice,
    };

    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            p.join("model.safetensors").is_file().then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached Qwen3-0.6B");
        return;
    };
    let descriptor = std::path::PathBuf::from(
        "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad",
    )
    .join("qwen3_descriptor.json");
    if !descriptor.is_file() {
        eprintln!("skipped: no generated descriptor");
        return;
    }

    static NOTIFIED: AtomicU64 = AtomicU64::new(0);
    unsafe extern "C" fn notify(_ctx: *mut std::ffi::c_void, wait_id: u64, _epoch: u64) {
        NOTIFIED.store(wait_id, Ordering::SeqCst);
    }

    let boot = format!("[model]\ndescriptor = \"{}\"\n", descriptor.display());
    let desc = PieDriverCreateDesc {
        abi_version: PIE_DRIVER_ABI_VERSION,
        config_bytes: PieBytes { ptr: boot.as_ptr(), len: boot.len() },
        runtime: PieRuntimeCallbacks {
            abi_version: PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            ctx: std::ptr::null_mut(),
            notify: Some(notify),
        },
        ..Default::default()
    };
    let d = unsafe { driver_abi::local::pie_cuda_create(&desc, std::ptr::null_mut()) };
    assert!(!d.is_null());

    let snap_str = snap.to_string_lossy().into_owned();
    let load = PieModelLoadDesc {
        snapshot_dir: PieBytes { ptr: snap_str.as_ptr(), len: snap_str.len() },
        ..Default::default()
    };
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_load_model(d, &load, std::ptr::null_mut()) },
        PIE_STATUS_OK
    );

    let prog = PieProgramDesc { program_hash: 0xF12E, ..Default::default() };
    let mut program_id = 0u64;
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_register_program(d, &prog, &mut program_id) },
        PIE_STATUS_OK
    );
    let inst = PieInstanceDesc { program_id, ..Default::default() };
    let mut binding = PieInstanceBinding::default();
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_bind_instance(d, &inst, &mut binding) },
        PIE_STATUS_OK
    );

    // One decode step: instance's first token (id 7) at position 0, one
    // KV page, appended at offset 0.
    let mut cell = PieTerminalCell { outcome: PIE_TERMINAL_OUTCOME_PENDING, reserved0: 0 };
    let cell_ptr: *mut PieTerminalCell = &mut cell;
    let roster_rows: [u32; 1] = [0];
    let sub_batch_indptr: [u32; 2] = [0, 1];
    let sub_batch_class: [u32; 1] = [driver_abi::local::PIE_GEOMETRY_CLASS_HOST];
    let token_ids: [u32; 1] = [7];
    let position_ids: [u32; 1] = [0];
    let kv_page_indices: [u32; 1] = [0];
    let kv_page_indptr: [u32; 2] = [0, 1];
    let kv_last_page_lens: [u32; 1] = [1];
    let qo_indptr: [u32; 2] = [0, 1];
    let u32s = |v: &[u32]| PieU32Slice { ptr: v.as_ptr(), len: v.len() };
    let step = PieStepDesc {
        roster_rows: u32s(&roster_rows),
        sub_batch_indptr: u32s(&sub_batch_indptr),
        sub_batch_class: u32s(&sub_batch_class),
        terminal_cells: PieTerminalCellPtrSlice { ptr: &cell_ptr, len: 1 },
        token_ids: u32s(&token_ids),
        position_ids: u32s(&position_ids),
        kv_page_indices: u32s(&kv_page_indices),
        kv_page_indptr: u32s(&kv_page_indptr),
        kv_last_page_lens: u32s(&kv_last_page_lens),
        qo_indptr: u32s(&qo_indptr),
        ..Default::default()
    };
    let instance_ids: [u64; 1] = [binding.instance_id];
    let frame = PieFrameDesc {
        abi_version: PIE_DRIVER_ABI_VERSION,
        instance_ids: PieU64Slice { ptr: instance_ids.as_ptr(), len: 1 },
        required_kv_pages: 1,
        steps: driver_abi::local::PieStepDescSlice { ptr: &step, len: 1 },
        ..Default::default()
    };
    let completion = PieCompletion {
        wait_id: 0xBEEF,
        target_epoch: 1,
        terminal_cell: std::ptr::null_mut(),
    };
    let status = unsafe { driver_abi::local::pie_cuda_launch(d, &frame, completion) };
    assert_eq!(status, PIE_STATUS_OK, "the frame launches");
    assert_eq!(cell.outcome, PIE_TERMINAL_OUTCOME_SUCCESS, "the terminal cell published");
    assert_eq!(NOTIFIED.load(Ordering::SeqCst), 0xBEEF, "the runtime was notified");

    unsafe { driver_abi::local::pie_cuda_destroy(d) };
}

/// The channel endpoint contract: a registered channel answers with a
/// pinned mirror of `(capacity + 1)` wire cells and four zeroed control
/// words at indices 0..4; bool bit-packs; duplicates and oversized rings
/// refuse; closing frees and is idempotent.
#[test]
fn channels_bind_the_ring_contract() {
    let _gpu = gpu_guard();
    use driver_abi::local::{
        PIE_CHANNEL_DTYPE_BOOL, PieChannelDesc, PieChannelEndpointBinding, PieU32Slice,
    };

    let desc =
        PieDriverCreateDesc { abi_version: PIE_DRIVER_ABI_VERSION, ..Default::default() };
    let d = unsafe { driver_abi::local::pie_cuda_create(&desc, std::ptr::null_mut()) };
    assert!(!d.is_null());

    let shape: [u32; 2] = [4, 8]; // 32 elements
    let ch = PieChannelDesc {
        channel_id: 5,
        shape: PieU32Slice { ptr: shape.as_ptr(), len: 2 },
        capacity: 7,
        ..Default::default()
    };
    let mut b = PieChannelEndpointBinding::default();
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_register_channel(d, &ch, &mut b) },
        PIE_STATUS_OK
    );
    assert_eq!(b.cell_bytes, 32 * 4, "f32 wire cells are four bytes per element");
    assert_eq!(b.mirror_bytes, u64::from(b.cell_bytes) * 8, "capacity + 1 cells");
    assert_eq!(
        (b.head_word_index, b.tail_word_index, b.poison_word_index, b.closed_word_index),
        (0, 1, 2, 3)
    );
    let words =
        unsafe { std::slice::from_raw_parts(b.word_base as *const u64, 4) };
    assert_eq!(words, &[0, 0, 0, 0], "control words start zeroed");

    // Duplicate id refuses; a bool channel bit-packs.
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_register_channel(d, &ch, &mut b) },
        PIE_STATUS_INVALID_ARGUMENT
    );
    let boolch = PieChannelDesc {
        channel_id: 6,
        shape: PieU32Slice { ptr: shape.as_ptr(), len: 2 },
        dtype: PIE_CHANNEL_DTYPE_BOOL,
        capacity: 1,
        ..Default::default()
    };
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_register_channel(d, &boolch, &mut b) },
        PIE_STATUS_OK
    );
    assert_eq!(b.cell_bytes, 4, "32 bools bit-pack to four bytes");

    // An oversized ring refuses; closes are real and idempotent.
    let big = PieChannelDesc { channel_id: 9, capacity: 64, ..ch };
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_register_channel(d, &big, &mut b) },
        PIE_STATUS_INVALID_ARGUMENT,
        "capacity + 1 must stay within the ring maximum"
    );
    assert_eq!(unsafe { driver_abi::local::pie_cuda_close_channel(d, 5) }, PIE_STATUS_OK);
    assert_eq!(unsafe { driver_abi::local::pie_cuda_close_channel(d, 5) }, PIE_STATUS_OK);
    let again = PieChannelDesc { channel_id: 5, ..ch };
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_register_channel(d, &again, &mut b) },
        PIE_STATUS_OK,
        "a closed id re-registers"
    );

    unsafe { driver_abi::local::pie_cuda_destroy(d) };
}

/// The delivery: the engine's whole loop, with the output coming BACK.
/// A `[vocab]` f32 reader channel attaches to the instance, the reference
/// prompt prefills through `pie_cuda_launch`, and the ring's first cell
/// holds the last row's logits — checked against the SAME transformers
/// reference the executor A/B pinned. The tail word advanced exactly
/// once; head stays the engine's.
#[test]
fn logits_come_back_through_the_ring() {
    let _gpu = gpu_guard();
    use driver_abi::local::{
        PIE_CHANNEL_HOST_ROLE_READER, PieBytes, PieChannelDesc, PieChannelEndpointBinding,
        PieCompletion, PieFrameDesc, PieInstanceBinding, PieInstanceDesc, PieModelLoadDesc,
        PieProgramDesc, PieStepDesc, PieU32Slice, PieU64Slice,
    };

    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            p.join("model.safetensors").is_file().then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached Qwen3-0.6B");
        return;
    };
    let scratch = std::path::PathBuf::from(
        "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad",
    );
    let descriptor = scratch.join("qwen3_descriptor.json");
    if !descriptor.is_file() {
        eprintln!("skipped: no generated descriptor");
        return;
    }
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "oracle/real_decode/reference.json"
    ))
    .expect("reference");

    let boot = format!("[model]\ndescriptor = \"{}\"\n", descriptor.display());
    let desc = PieDriverCreateDesc {
        abi_version: PIE_DRIVER_ABI_VERSION,
        config_bytes: PieBytes { ptr: boot.as_ptr(), len: boot.len() },
        ..Default::default()
    };
    let d = unsafe { driver_abi::local::pie_cuda_create(&desc, std::ptr::null_mut()) };
    assert!(!d.is_null());
    let snap_str = snap.to_string_lossy().into_owned();
    let load = PieModelLoadDesc {
        snapshot_dir: PieBytes { ptr: snap_str.as_ptr(), len: snap_str.len() },
        ..Default::default()
    };
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_load_model(d, &load, std::ptr::null_mut()) },
        PIE_STATUS_OK
    );

    const VOCAB: usize = 151_936;
    let shape: [u32; 1] = [VOCAB as u32];
    let ch = PieChannelDesc {
        channel_id: 77,
        shape: PieU32Slice { ptr: shape.as_ptr(), len: 1 },
        host_role: PIE_CHANNEL_HOST_ROLE_READER,
        capacity: 3,
        ..Default::default()
    };
    let mut chb = PieChannelEndpointBinding::default();
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_register_channel(d, &ch, &mut chb) },
        PIE_STATUS_OK
    );

    let prog = PieProgramDesc { program_hash: 0xF13E, ..Default::default() };
    let mut program_id = 0u64;
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_register_program(d, &prog, &mut program_id) },
        PIE_STATUS_OK
    );
    let channel_ids: [u64; 1] = [77];
    let inst = PieInstanceDesc {
        program_id,
        channel_ids: PieU64Slice { ptr: channel_ids.as_ptr(), len: 1 },
        ..Default::default()
    };
    let mut binding = PieInstanceBinding::default();
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_bind_instance(d, &inst, &mut binding) },
        PIE_STATUS_OK
    );

    // The reference prompt as one prefill request over one page.
    let prompt: Vec<u32> = reference["prompt_ids"]
        .as_array().expect("ids").iter()
        .map(|v| v.as_u64().expect("id") as u32).collect();
    let tokens = prompt.len();
    let positions: Vec<u32> = (0..tokens as u32).collect();
    let roster_rows: Vec<u32> = vec![0; tokens];
    let sub_batch_indptr: [u32; 2] = [0, tokens as u32];
    let sub_batch_class: [u32; 1] = [driver_abi::local::PIE_GEOMETRY_CLASS_HOST];
    let kv_page_indices: [u32; 1] = [0];
    let kv_page_indptr: [u32; 2] = [0, 1];
    let kv_last_page_lens: [u32; 1] = [tokens as u32];
    let qo_indptr: [u32; 2] = [0, tokens as u32];
    let u32s = |v: &[u32]| PieU32Slice { ptr: v.as_ptr(), len: v.len() };
    let step = PieStepDesc {
        roster_rows: u32s(&roster_rows),
        sub_batch_indptr: u32s(&sub_batch_indptr),
        sub_batch_class: u32s(&sub_batch_class),
        token_ids: u32s(&prompt),
        position_ids: u32s(&positions),
        kv_page_indices: u32s(&kv_page_indices),
        kv_page_indptr: u32s(&kv_page_indptr),
        kv_last_page_lens: u32s(&kv_last_page_lens),
        qo_indptr: u32s(&qo_indptr),
        ..Default::default()
    };
    let instance_ids: [u64; 1] = [binding.instance_id];
    let frame = PieFrameDesc {
        abi_version: PIE_DRIVER_ABI_VERSION,
        instance_ids: PieU64Slice { ptr: instance_ids.as_ptr(), len: 1 },
        required_kv_pages: 1,
        steps: driver_abi::local::PieStepDescSlice { ptr: &step, len: 1 },
        ..Default::default()
    };
    let completion =
        PieCompletion { wait_id: 1, target_epoch: 1, terminal_cell: std::ptr::null_mut() };
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_launch(d, &frame, completion) },
        PIE_STATUS_OK
    );

    // The ring advanced once, and cell 0 holds the last row's logits.
    let words = unsafe { std::slice::from_raw_parts(chb.word_base as *const u64, 4) };
    assert_eq!(words[1], 1, "the tail advanced exactly once");
    assert_eq!(words[0], 0, "the head is the engine's to move");
    let cell = unsafe {
        std::slice::from_raw_parts(chb.mirror_base as *const f32, VOCAB)
    };
    let hf_argmax = reference["argmax"].as_u64().expect("argmax") as usize;
    let (mut best_t, mut best_v) = (0usize, f32::NEG_INFINITY);
    for (t, &v) in cell.iter().enumerate() {
        if v > best_v {
            (best_t, best_v) = (t, v);
        }
    }
    assert_eq!(best_t, hf_argmax, "the ring carried the right logits (top {best_v})");
    let hf_top1 = reference["top5_logits"].as_array().expect("top5")[0]
        .as_f64().expect("v") as f32;
    assert!((best_v - hf_top1).abs() < 0.25, "top-1 {best_v} vs HF {hf_top1}");

    unsafe { driver_abi::local::pie_cuda_destroy(d) };
}

/// Multi-step decode continuity + resize migration + copy_kv, in one
/// story: prefill the reference prompt (step 1) and decode its argmax
/// token (step 2) IN THE SAME FRAME — the decode's logits ride the ring
/// as cell 1. Then resize the pool larger and copy page 0 to page 2, and
/// a decode against the COPIED page must produce the same logits cell —
/// the migration and the page copy both preserved the KV bytes.
#[test]
fn multi_step_resize_and_copy_preserve_the_kv() {
    let _gpu = gpu_guard();
    use driver_abi::local::{
        PIE_CHANNEL_HOST_ROLE_READER, PieBytes, PieChannelDesc, PieChannelEndpointBinding,
        PieCompletion, PieFrameDesc, PieInstanceBinding, PieInstanceDesc, PieKvCopyDesc,
        PieModelLoadDesc, PiePoolResizeDesc, PieProgramDesc, PieStepDesc, PieU32Slice,
        PieU64Slice,
    };

    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            p.join("model.safetensors").is_file().then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached Qwen3-0.6B");
        return;
    };
    let descriptor = std::path::PathBuf::from(
        "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad",
    )
    .join("qwen3_descriptor.json");
    if !descriptor.is_file() {
        eprintln!("skipped: no generated descriptor");
        return;
    }
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "oracle/real_decode/reference.json"
    ))
    .expect("reference");

    let boot = format!("[model]\ndescriptor = \"{}\"\n", descriptor.display());
    let desc = PieDriverCreateDesc {
        abi_version: PIE_DRIVER_ABI_VERSION,
        config_bytes: PieBytes { ptr: boot.as_ptr(), len: boot.len() },
        ..Default::default()
    };
    let d = unsafe { driver_abi::local::pie_cuda_create(&desc, std::ptr::null_mut()) };
    assert!(!d.is_null());
    let snap_str = snap.to_string_lossy().into_owned();
    let load = PieModelLoadDesc {
        snapshot_dir: PieBytes { ptr: snap_str.as_ptr(), len: snap_str.len() },
        ..Default::default()
    };
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_load_model(d, &load, std::ptr::null_mut()) },
        PIE_STATUS_OK
    );
    const VOCAB: usize = 151_936;
    let shape: [u32; 1] = [VOCAB as u32];
    let ch = PieChannelDesc {
        channel_id: 9,
        shape: PieU32Slice { ptr: shape.as_ptr(), len: 1 },
        host_role: PIE_CHANNEL_HOST_ROLE_READER,
        capacity: 7,
        ..Default::default()
    };
    let mut chb = PieChannelEndpointBinding::default();
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_register_channel(d, &ch, &mut chb) },
        PIE_STATUS_OK
    );
    let prog = PieProgramDesc { program_hash: 0xF14E, ..Default::default() };
    let mut program_id = 0u64;
    unsafe { driver_abi::local::pie_cuda_register_program(d, &prog, &mut program_id) };
    let channel_ids: [u64; 1] = [9];
    let inst = PieInstanceDesc {
        program_id,
        channel_ids: PieU64Slice { ptr: channel_ids.as_ptr(), len: 1 },
        ..Default::default()
    };
    let mut binding = PieInstanceBinding::default();
    unsafe { driver_abi::local::pie_cuda_bind_instance(d, &inst, &mut binding) };

    let prompt: Vec<u32> = reference["prompt_ids"]
        .as_array().expect("ids").iter()
        .map(|v| v.as_u64().expect("id") as u32).collect();
    let n = prompt.len();
    let hf_argmax = reference["argmax"].as_u64().expect("argmax") as u32;

    let u32s = |v: &[u32]| PieU32Slice { ptr: v.as_ptr(), len: v.len() };
    // Step 1: prefill the prompt. Step 2: decode the argmax token at
    // position n against the same page.
    let positions1: Vec<u32> = (0..n as u32).collect();
    let roster1: Vec<u32> = vec![0; n];
    let sbi1: [u32; 2] = [0, n as u32];
    let cls: [u32; 1] = [driver_abi::local::PIE_GEOMETRY_CLASS_HOST];
    let pages: [u32; 1] = [0];
    let indptr: [u32; 2] = [0, 1];
    let lens1: [u32; 1] = [n as u32];
    let qo1: [u32; 2] = [0, n as u32];
    let step1 = PieStepDesc {
        roster_rows: u32s(&roster1),
        sub_batch_indptr: u32s(&sbi1),
        sub_batch_class: u32s(&cls),
        token_ids: u32s(&prompt),
        position_ids: u32s(&positions1),
        kv_page_indices: u32s(&pages),
        kv_page_indptr: u32s(&indptr),
        kv_last_page_lens: u32s(&lens1),
        qo_indptr: u32s(&qo1),
        ..Default::default()
    };
    let tok2: [u32; 1] = [hf_argmax];
    let pos2: [u32; 1] = [n as u32];
    let roster2: [u32; 1] = [0];
    let sbi2: [u32; 2] = [0, 1];
    let lens2: [u32; 1] = [n as u32 + 1];
    let qo2: [u32; 2] = [0, 1];
    let step2 = PieStepDesc {
        roster_rows: u32s(&roster2),
        sub_batch_indptr: u32s(&sbi2),
        sub_batch_class: u32s(&cls),
        token_ids: u32s(&tok2),
        position_ids: u32s(&pos2),
        kv_page_indices: u32s(&pages),
        kv_page_indptr: u32s(&indptr),
        kv_last_page_lens: u32s(&lens2),
        qo_indptr: u32s(&qo2),
        ..Default::default()
    };
    let steps = [step1, step2];
    let instance_ids: [u64; 1] = [binding.instance_id];
    let frame = PieFrameDesc {
        abi_version: PIE_DRIVER_ABI_VERSION,
        instance_ids: PieU64Slice { ptr: instance_ids.as_ptr(), len: 1 },
        required_kv_pages: 1,
        steps: driver_abi::local::PieStepDescSlice { ptr: steps.as_ptr(), len: 2 },
        ..Default::default()
    };
    let completion =
        PieCompletion { wait_id: 2, target_epoch: 1, terminal_cell: std::ptr::null_mut() };
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_launch(d, &frame, completion) },
        PIE_STATUS_OK,
        "the two-step frame launches"
    );
    let words = unsafe { std::slice::from_raw_parts(chb.word_base as *const u64, 4) };
    assert_eq!(words[1], 2, "both steps delivered");
    let cell1 = unsafe {
        std::slice::from_raw_parts(
            (chb.mirror_base as *const f32).add(VOCAB),
            VOCAB,
        )
    };
    let decode_logits: Vec<f32> = cell1.to_vec();
    let best1 = decode_logits
        .iter().enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1)).map(|(t, _)| t).unwrap();

    // Resize larger (migrates page 0), copy page 0 → page 2, then decode
    // AGAINST PAGE 2: same context bytes, so the same logits cell.
    let resize = PiePoolResizeDesc { target_pages: 4, ..Default::default() };
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_resize_pool(d, &resize, completion) },
        PIE_STATUS_OK
    );
    let src: [u32; 1] = [0];
    let dst: [u32; 1] = [2];
    let copy = PieKvCopyDesc {
        src_domain: driver_abi::local::PIE_MEMORY_DOMAIN_CUDA_DEVICE,
        dst_domain: driver_abi::local::PIE_MEMORY_DOMAIN_CUDA_DEVICE,
        src_page_ids: u32s(&src),
        dst_page_ids: u32s(&dst),
        ..Default::default()
    };
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_copy_kv(d, &copy, completion) },
        PIE_STATUS_OK
    );
    let pages2: [u32; 1] = [2];
    let step3 = PieStepDesc { kv_page_indices: u32s(&pages2), ..step2 };
    let steps3 = [step3];
    let frame3 = PieFrameDesc {
        steps: driver_abi::local::PieStepDescSlice { ptr: steps3.as_ptr(), len: 1 },
        required_kv_pages: 4,
        ..frame
    };
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_launch(d, &frame3, completion) },
        PIE_STATUS_OK
    );
    assert_eq!(words[1], 3, "the third fire delivered");
    let cell2 = unsafe {
        std::slice::from_raw_parts(
            (chb.mirror_base as *const f32).add(2 * VOCAB),
            VOCAB,
        )
    };
    let best2 = cell2
        .iter().enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1)).map(|(t, _)| t).unwrap();
    assert_eq!(best2, best1, "the migrated + copied page carries the same context");
    for t in [best1, 0, 1000] {
        assert!(
            (cell2[t] - decode_logits[t]).abs() < 1e-3,
            "token {t}: {} vs {}",
            cell2[t],
            decode_logits[t]
        );
    }

    unsafe { driver_abi::local::pie_cuda_destroy(d) };
}

/// The mini-soak, and it is real GENERATION: prefill the reference
/// prompt, then fifty greedy decode steps, each feeding the previous
/// argmax back through its own `pie_cuda_launch` — the inference loop an
/// engine runs, driven twice. Gates: every step delivers on the ring,
/// the two runs produce IDENTICAL token sequences (determinism), the
/// first decoded token matches the HF reference argmax, and device free
/// memory is stable across the chain (per-fire allocations all retire).
#[test]
fn a_fifty_step_greedy_chain_is_deterministic_and_leak_free() {
    let _gpu = gpu_guard();
    use driver_abi::local::{
        PIE_CHANNEL_HOST_ROLE_READER, PieBytes, PieChannelDesc, PieChannelEndpointBinding,
        PieCompletion, PieFrameDesc, PieInstanceBinding, PieInstanceDesc, PieModelLoadDesc,
        PieProgramDesc, PieStepDesc, PieU32Slice, PieU64Slice,
    };

    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            p.join("model.safetensors").is_file().then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached Qwen3-0.6B");
        return;
    };
    let descriptor = std::path::PathBuf::from(
        "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad",
    )
    .join("qwen3_descriptor.json");
    if !descriptor.is_file() {
        eprintln!("skipped: no generated descriptor");
        return;
    }
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "oracle/real_decode/reference.json"
    ))
    .expect("reference");
    let prompt: Vec<u32> = reference["prompt_ids"]
        .as_array().expect("ids").iter()
        .map(|v| v.as_u64().expect("id") as u32).collect();
    let hf_argmax = reference["argmax"].as_u64().expect("argmax") as u32;

    const VOCAB: usize = 151_936;
    const STEPS: usize = 50;
    const PAGE: u32 = 16;

    let chain = |run_tag: u64| -> Vec<u32> {
        let boot = format!("[model]\ndescriptor = \"{}\"\n", descriptor.display());
        let desc = PieDriverCreateDesc {
            abi_version: PIE_DRIVER_ABI_VERSION,
            config_bytes: PieBytes { ptr: boot.as_ptr(), len: boot.len() },
            ..Default::default()
        };
        let d = unsafe { driver_abi::local::pie_cuda_create(&desc, std::ptr::null_mut()) };
        assert!(!d.is_null());
        let snap_str = snap.to_string_lossy().into_owned();
        let load = PieModelLoadDesc {
            snapshot_dir: PieBytes { ptr: snap_str.as_ptr(), len: snap_str.len() },
            ..Default::default()
        };
        assert_eq!(
            unsafe { driver_abi::local::pie_cuda_load_model(d, &load, std::ptr::null_mut()) },
            PIE_STATUS_OK
        );
        let shape: [u32; 1] = [VOCAB as u32];
        let ch = PieChannelDesc {
            channel_id: 1,
            shape: PieU32Slice { ptr: shape.as_ptr(), len: 1 },
            host_role: PIE_CHANNEL_HOST_ROLE_READER,
            capacity: 3,
            ..Default::default()
        };
        let mut chb = PieChannelEndpointBinding::default();
        assert_eq!(
            unsafe { driver_abi::local::pie_cuda_register_channel(d, &ch, &mut chb) },
            PIE_STATUS_OK
        );
        let prog = PieProgramDesc { program_hash: run_tag, ..Default::default() };
        let mut program_id = 0u64;
        unsafe { driver_abi::local::pie_cuda_register_program(d, &prog, &mut program_id) };
        let channel_ids: [u64; 1] = [1];
        let inst = PieInstanceDesc {
            program_id,
            channel_ids: PieU64Slice { ptr: channel_ids.as_ptr(), len: 1 },
            ..Default::default()
        };
        let mut binding = PieInstanceBinding::default();
        unsafe { driver_abi::local::pie_cuda_bind_instance(d, &inst, &mut binding) };
        let instance_ids: [u64; 1] = [binding.instance_id];
        let completion =
            PieCompletion { wait_id: 1, target_epoch: 1, terminal_cell: std::ptr::null_mut() };

        let u32s = |v: &[u32]| PieU32Slice { ptr: v.as_ptr(), len: v.len() };
        let total_pages = ((prompt.len() + STEPS) as u32).div_ceil(PAGE);
        let all_pages: Vec<u32> = (0..total_pages).collect();
        let read_cell = |i: u64| -> usize {
            let cell = unsafe {
                std::slice::from_raw_parts(
                    (chb.mirror_base as *const f32).add((i % 4) as usize * VOCAB),
                    VOCAB,
                )
            };
            cell.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(t, _)| t).unwrap()
        };
        let fire = |kv_len: u32, tokens: &[u32], positions: &[u32], qo_end: u32| {
            let pages_used = kv_len.div_ceil(PAGE).max(1);
            let indices = &all_pages[..pages_used as usize];
            let indptr: [u32; 2] = [0, pages_used];
            let lens: [u32; 1] = [kv_len - (pages_used - 1) * PAGE];
            let qo: [u32; 2] = [0, qo_end];
            let roster: Vec<u32> = vec![0; tokens.len()];
            let sbi: [u32; 2] = [0, tokens.len() as u32];
            let cls: [u32; 1] = [driver_abi::local::PIE_GEOMETRY_CLASS_HOST];
            let step = PieStepDesc {
                roster_rows: u32s(&roster),
                sub_batch_indptr: u32s(&sbi),
                sub_batch_class: u32s(&cls),
                token_ids: u32s(tokens),
                position_ids: u32s(positions),
                kv_page_indices: u32s(indices),
                kv_page_indptr: u32s(&indptr),
                kv_last_page_lens: u32s(&lens),
                qo_indptr: u32s(&qo),
                ..Default::default()
            };
            let steps_arr = [step];
            let frame = PieFrameDesc {
                abi_version: PIE_DRIVER_ABI_VERSION,
                instance_ids: PieU64Slice { ptr: instance_ids.as_ptr(), len: 1 },
                required_kv_pages: total_pages,
                steps: driver_abi::local::PieStepDescSlice { ptr: steps_arr.as_ptr(), len: 1 },
                ..Default::default()
            };
            assert_eq!(
                unsafe { driver_abi::local::pie_cuda_launch(d, &frame, completion) },
                PIE_STATUS_OK
            );
        };

        // Prefill, then the greedy chain. The engine's half of the ring:
        // advance the head as each cell is consumed.
        let positions: Vec<u32> = (0..prompt.len() as u32).collect();
        fire(prompt.len() as u32, &prompt, &positions, prompt.len() as u32);
        let words = chb.word_base as *mut u64;
        let mut consumed = 0u64;
        let mut next = read_cell(consumed) as u32;
        consumed += 1;
        unsafe { words.read_volatile() }; // head untouched by us conceptually
        unsafe { words.write_volatile(consumed) };
        let mut generated = vec![next];
        for s in 0..STEPS - 1 {
            let pos = prompt.len() as u32 + s as u32;
            let toks: [u32; 1] = [next];
            let poss: [u32; 1] = [pos];
            fire(pos + 1, &toks, &poss, 1);
            next = read_cell(consumed) as u32;
            consumed += 1;
            unsafe { words.write_volatile(consumed) };
            generated.push(next);
        }
        unsafe { driver_abi::local::pie_cuda_destroy(d) };
        generated
    };

    let free_before = {
        let (free, _) = common::device_or_skip("soak").map(|d| d.memory_info().unwrap()).unwrap();
        free
    };
    let run1 = chain(0xA);
    let mid = {
        let (free, _) = common::device_or_skip("soak").map(|d| d.memory_info().unwrap()).unwrap();
        free
    };
    let run2 = chain(0xB);
    let free_after = {
        let (free, _) = common::device_or_skip("soak").map(|d| d.memory_info().unwrap()).unwrap();
        free
    };

    assert_eq!(run1[0], hf_argmax, "the chain starts where the reference points");
    assert_eq!(run1, run2, "greedy generation is deterministic across drivers");
    assert!(
        run1.iter().skip(1).any(|&t| t != run1[0]),
        "fifty steps that repeat one token would be a broken chain: {run1:?}"
    );
    // Each chain creates and destroys a full driver (weights included), so
    // free memory must come back to within a small slack of the start.
    let slack: usize = 256 * 1024 * 1024;
    assert!(
        free_after + slack > free_before && mid + slack > free_before.saturating_sub(2_000_000_000usize),
        "device memory drifted: before {free_before} mid {mid} after {free_after}"
    );
    eprintln!("[soak] generated: {:?}", &run1[..10.min(run1.len())]);
}

/// THE SOAK, at the C++ gate's round count: 711 fires in ONE driver
/// lifetime — fourteen generation chains (prefill + fifty decodes each),
/// pages rewound per chain, device free memory sampled at every chain
/// boundary and required FLAT after warmup. The C++ soak's own gates
/// (many rounds, RSS steady), spoken through the new ABI.
#[test]
#[ignore = "the scaled soak: ~1 minute of GPU; run explicitly"]
fn the_711_fire_soak_holds_steady() {
    let _gpu = gpu_guard();
    use driver_abi::local::{
        PIE_CHANNEL_HOST_ROLE_READER, PieBytes, PieChannelDesc, PieChannelEndpointBinding,
        PieCompletion, PieFrameDesc, PieInstanceBinding, PieInstanceDesc, PieModelLoadDesc,
        PieProgramDesc, PieStepDesc, PieU32Slice, PieU64Slice,
    };

    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            p.join("model.safetensors").is_file().then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached Qwen3-0.6B");
        return;
    };
    let descriptor = std::path::PathBuf::from(
        "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad",
    )
    .join("qwen3_descriptor.json");
    if !descriptor.is_file() {
        eprintln!("skipped: no generated descriptor");
        return;
    }
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "oracle/real_decode/reference.json"
    ))
    .expect("reference");
    let prompt: Vec<u32> = reference["prompt_ids"]
        .as_array().expect("ids").iter()
        .map(|v| v.as_u64().expect("id") as u32).collect();

    const VOCAB: usize = 151_936;
    const CHAINS: usize = 14;
    const DECODES: usize = 50;
    const PAGE: u32 = 16;

    let boot = format!("[model]\ndescriptor = \"{}\"\n", descriptor.display());
    let desc = PieDriverCreateDesc {
        abi_version: PIE_DRIVER_ABI_VERSION,
        config_bytes: PieBytes { ptr: boot.as_ptr(), len: boot.len() },
        ..Default::default()
    };
    let d = unsafe { driver_abi::local::pie_cuda_create(&desc, std::ptr::null_mut()) };
    assert!(!d.is_null());
    let snap_str = snap.to_string_lossy().into_owned();
    let load = PieModelLoadDesc {
        snapshot_dir: PieBytes { ptr: snap_str.as_ptr(), len: snap_str.len() },
        ..Default::default()
    };
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_load_model(d, &load, std::ptr::null_mut()) },
        PIE_STATUS_OK
    );
    let shape: [u32; 1] = [VOCAB as u32];
    let ch = PieChannelDesc {
        channel_id: 1,
        shape: PieU32Slice { ptr: shape.as_ptr(), len: 1 },
        host_role: PIE_CHANNEL_HOST_ROLE_READER,
        capacity: 3,
        ..Default::default()
    };
    let mut chb = PieChannelEndpointBinding::default();
    unsafe { driver_abi::local::pie_cuda_register_channel(d, &ch, &mut chb) };
    let prog = PieProgramDesc { program_hash: 0x50AC, ..Default::default() };
    let mut program_id = 0u64;
    unsafe { driver_abi::local::pie_cuda_register_program(d, &prog, &mut program_id) };
    let channel_ids: [u64; 1] = [1];
    let inst = PieInstanceDesc {
        program_id,
        channel_ids: PieU64Slice { ptr: channel_ids.as_ptr(), len: 1 },
        ..Default::default()
    };
    let mut binding = PieInstanceBinding::default();
    unsafe { driver_abi::local::pie_cuda_bind_instance(d, &inst, &mut binding) };
    let instance_ids: [u64; 1] = [binding.instance_id];
    let completion =
        PieCompletion { wait_id: 1, target_epoch: 1, terminal_cell: std::ptr::null_mut() };
    let words = chb.word_base as *mut u64;

    let u32s = |v: &[u32]| PieU32Slice { ptr: v.as_ptr(), len: v.len() };
    let total_pages = ((prompt.len() + DECODES) as u32).div_ceil(PAGE);
    let all_pages: Vec<u32> = (0..total_pages).collect();
    let mut fires = 0usize;
    let mut consumed = 0u64;
    let mut baseline = None;
    let mut first_chain_head = Vec::new();
    for chain in 0..CHAINS {
        let fire = |kv_len: u32, tokens: &[u32], positions: &[u32]| {
            let pages_used = kv_len.div_ceil(PAGE).max(1);
            let indices = &all_pages[..pages_used as usize];
            let indptr: [u32; 2] = [0, pages_used];
            let lens: [u32; 1] = [kv_len - (pages_used - 1) * PAGE];
            let qo: [u32; 2] = [0, tokens.len() as u32];
            let roster: Vec<u32> = vec![0; tokens.len()];
            let sbi: [u32; 2] = [0, tokens.len() as u32];
            let cls: [u32; 1] = [driver_abi::local::PIE_GEOMETRY_CLASS_HOST];
            let step = PieStepDesc {
                roster_rows: u32s(&roster),
                sub_batch_indptr: u32s(&sbi),
                sub_batch_class: u32s(&cls),
                token_ids: u32s(tokens),
                position_ids: u32s(positions),
                kv_page_indices: u32s(indices),
                kv_page_indptr: u32s(&indptr),
                kv_last_page_lens: u32s(&lens),
                qo_indptr: u32s(&qo),
                ..Default::default()
            };
            let steps_arr = [step];
            let frame = PieFrameDesc {
                abi_version: PIE_DRIVER_ABI_VERSION,
                instance_ids: PieU64Slice { ptr: instance_ids.as_ptr(), len: 1 },
                required_kv_pages: total_pages,
                steps: driver_abi::local::PieStepDescSlice {
                    ptr: steps_arr.as_ptr(),
                    len: 1,
                },
                ..Default::default()
            };
            assert_eq!(
                unsafe { driver_abi::local::pie_cuda_launch(d, &frame, completion) },
                PIE_STATUS_OK
            );
        };
        let read_argmax = |i: u64| -> u32 {
            let cell = unsafe {
                std::slice::from_raw_parts(
                    (chb.mirror_base as *const f32).add((i % 4) as usize * VOCAB),
                    VOCAB,
                )
            };
            cell.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(t, _)| t as u32)
                .unwrap()
        };

        let positions: Vec<u32> = (0..prompt.len() as u32).collect();
        fire(prompt.len() as u32, &prompt, &positions);
        fires += 1;
        let mut next = read_argmax(consumed);
        consumed += 1;
        unsafe { words.write_volatile(consumed) };
        let mut head_tokens = vec![next];
        for s in 0..DECODES {
            let pos = prompt.len() as u32 + s as u32;
            let toks: [u32; 1] = [next];
            let poss: [u32; 1] = [pos];
            fire(pos + 1, &toks, &poss);
            fires += 1;
            next = read_argmax(consumed);
            consumed += 1;
            unsafe { words.write_volatile(consumed) };
            if head_tokens.len() < 8 {
                head_tokens.push(next);
            }
        }
        if chain == 0 {
            first_chain_head = head_tokens;
        } else {
            assert_eq!(
                head_tokens, first_chain_head,
                "chain {chain}: rewound pages must reproduce chain 0"
            );
        }
        let (free, _) = common::device_or_skip("soak")
            .map(|dev| dev.memory_info().unwrap())
            .unwrap();
        match baseline {
            None => baseline = Some(free),
            Some(b) => assert!(
                free + (64 << 20) > b,
                "chain {chain}: device memory drifting ({free} vs baseline {b})"
            ),
        }
    }
    assert_eq!(fires, CHAINS * (DECODES + 1), "the full round count ran");
    eprintln!("[soak] {fires} fires, memory flat at {:?}", baseline);
    unsafe { driver_abi::local::pie_cuda_destroy(d) };
}

/// The qwen3_5 HYBRID through the 13-symbol ABI, end to end (E-gate
/// family #1's shell gate): `load_model` parses the VL-shaped config and
/// admits the fp32 GDN parameters, `launch` prefills the reference
/// prompt through the hybrid plan — 18 GDN layers against driver-owned,
/// ENGINE-slotted state slabs (`rs_slot_ids` + RESET) — and the ring
/// carries logits meeting the family's calibrated bar (argmax within
/// HF's top-5; `real_hybrid.rs` documents why not equality). Then
/// `copy_state` clones slot 0 → slots 1 and 2, and the SAME decode
/// fired against two copies must agree — same argmax, logits within
/// 0.25. (Not bit-identity: two sequential fires jitter at ~0.1 even on
/// identical state — per-fire allocations shift addresses and with them
/// GEMM reduction orders — measured copy-vs-copy, same pattern as
/// live-vs-copy. A wrong stride or a missed plane blows 0.25 wide open;
/// the jitter does not.)
#[test]
fn the_hybrid_loads_fires_and_copies_state_through_the_abi() {
    let _gpu = gpu_guard();
    use driver_abi::local::{
        PIE_CHANNEL_HOST_ROLE_READER, PIE_RS_FLAG_RESET, PieBytes, PieChannelDesc,
        PieChannelEndpointBinding, PieCompletion, PieFrameDesc, PieInstanceBinding,
        PieInstanceDesc, PieModelLoadDesc, PieProgramDesc, PieStateCopyDesc,
        PieStateCopyRange, PieStepDesc, PieU32Slice, PieU64Slice,
    };

    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            (p.join("model.safetensors").is_file()
                || p.join("model.safetensors.index.json").is_file())
            .then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached Qwen3.5-0.8B-Base");
        return;
    };
    let scratch = std::path::PathBuf::from(std::env::var("PIE_TEST_SCRATCH").unwrap_or_else(
        |_| "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad".into(),
    ));
    let descriptor = scratch.join("qwen3_5_descriptor.json");
    if !descriptor.is_file() {
        eprintln!("skipped: no generated qwen3_5 descriptor");
        return;
    }
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("oracle/real_decode/qwen3_5_0_8b.json"))
            .expect("reference");

    let boot = format!("[model]\ndescriptor = \"{}\"\n", descriptor.display());
    let desc = driver_abi::local::PieDriverCreateDesc {
        abi_version: PIE_DRIVER_ABI_VERSION,
        config_bytes: PieBytes { ptr: boot.as_ptr(), len: boot.len() },
        ..Default::default()
    };
    let d = unsafe { driver_abi::local::pie_cuda_create(&desc, std::ptr::null_mut()) };
    assert!(!d.is_null());
    let snap_str = snap.to_string_lossy().into_owned();
    let load = PieModelLoadDesc {
        snapshot_dir: PieBytes { ptr: snap_str.as_ptr(), len: snap_str.len() },
        ..Default::default()
    };
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_load_model(d, &load, std::ptr::null_mut()) },
        PIE_STATUS_OK,
        "the hybrid checkpoint loads (fp32 GDN parameters included)"
    );

    const VOCAB: usize = 248_320;
    let shape: [u32; 1] = [VOCAB as u32];
    let ch = PieChannelDesc {
        channel_id: 88,
        shape: PieU32Slice { ptr: shape.as_ptr(), len: 1 },
        host_role: PIE_CHANNEL_HOST_ROLE_READER,
        capacity: 3,
        ..Default::default()
    };
    let mut chb = PieChannelEndpointBinding::default();
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_register_channel(d, &ch, &mut chb) },
        PIE_STATUS_OK
    );
    let prog = PieProgramDesc { program_hash: 0x35B, ..Default::default() };
    let mut program_id = 0u64;
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_register_program(d, &prog, &mut program_id) },
        PIE_STATUS_OK
    );
    let channel_ids: [u64; 1] = [88];
    let inst = PieInstanceDesc {
        program_id,
        channel_ids: PieU64Slice { ptr: channel_ids.as_ptr(), len: 1 },
        ..Default::default()
    };
    let mut binding = PieInstanceBinding::default();
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_bind_instance(d, &inst, &mut binding) },
        PIE_STATUS_OK
    );

    let u32s = |v: &[u32]| PieU32Slice { ptr: v.as_ptr(), len: v.len() };
    let instance_ids: [u64; 1] = [binding.instance_id];
    let fire = |step: &PieStepDesc, wait: u64| {
        let frame = PieFrameDesc {
            abi_version: PIE_DRIVER_ABI_VERSION,
            instance_ids: PieU64Slice { ptr: instance_ids.as_ptr(), len: 1 },
            required_kv_pages: 1,
            steps: driver_abi::local::PieStepDescSlice { ptr: step, len: 1 },
            ..Default::default()
        };
        let completion =
            PieCompletion { wait_id: wait, target_epoch: 1, terminal_cell: std::ptr::null_mut() };
        unsafe { driver_abi::local::pie_cuda_launch(d, &frame, completion) }
    };

    // ── Prefill on slot 0 (RESET — a fresh sequence). ──
    let prompt: Vec<u32> = reference["prompt_ids"]
        .as_array().expect("ids").iter()
        .map(|v| v.as_u64().expect("id") as u32).collect();
    let tokens = prompt.len();
    let positions: Vec<u32> = (0..tokens as u32).collect();
    let roster_rows: Vec<u32> = vec![0; tokens];
    let sub_batch_indptr: [u32; 2] = [0, tokens as u32];
    let sub_batch_class: [u32; 1] = [driver_abi::local::PIE_GEOMETRY_CLASS_HOST];
    let kv_page_indices: [u32; 1] = [0];
    let kv_page_indptr: [u32; 2] = [0, 1];
    let kv_last_page_lens: [u32; 1] = [tokens as u32];
    let qo_indptr: [u32; 2] = [0, tokens as u32];
    let rs_slots: [u32; 1] = [0];
    let rs_flags: [u8; 1] = [PIE_RS_FLAG_RESET];
    let step = PieStepDesc {
        roster_rows: u32s(&roster_rows),
        sub_batch_indptr: u32s(&sub_batch_indptr),
        sub_batch_class: u32s(&sub_batch_class),
        token_ids: u32s(&prompt),
        position_ids: u32s(&positions),
        kv_page_indices: u32s(&kv_page_indices),
        kv_page_indptr: u32s(&kv_page_indptr),
        kv_last_page_lens: u32s(&kv_last_page_lens),
        qo_indptr: u32s(&qo_indptr),
        rs_slot_ids: u32s(&rs_slots),
        rs_slot_flags: driver_abi::local::PieU8Slice {
            ptr: rs_flags.as_ptr(),
            len: 1,
        },
        ..Default::default()
    };
    assert_eq!(fire(&step, 1), PIE_STATUS_OK, "the hybrid prefill fires");

    let words = unsafe { std::slice::from_raw_parts(chb.word_base as *const u64, 4) };
    assert_eq!(words[1], 1, "the tail advanced once");
    let cell0 = unsafe { std::slice::from_raw_parts(chb.mirror_base as *const f32, VOCAB) };
    let argmax_of = |cell: &[f32]| {
        let (mut bt, mut bv) = (0usize, f32::NEG_INFINITY);
        for (t, &v) in cell.iter().enumerate() {
            if v > bv {
                (bt, bv) = (t, v);
            }
        }
        (bt, bv)
    };
    let (best_t, best_v) = argmax_of(cell0);
    let ids5: Vec<usize> = reference["top5_ids"]
        .as_array().expect("top5").iter()
        .map(|v| v.as_u64().expect("id") as usize).collect();
    assert!(
        ids5.contains(&best_t),
        "prefill argmax {best_t} ({best_v}) outside HF's top-5 {ids5:?}"
    );
    let next_token = best_t as u32;

    // ── The state fork: slot 0 → slots 1 AND 2 (two identical copies,
    // so the comparison below is copy vs copy — no live-slot asymmetry).
    let ranges = [
        PieStateCopyRange { src_slot_id: 0, dst_slot_id: 1, ..Default::default() },
        PieStateCopyRange { src_slot_id: 0, dst_slot_id: 2, ..Default::default() },
    ];
    let copy = PieStateCopyDesc {
        slot_ranges: driver_abi::local::PieStateCopyRangeSlice {
            ptr: ranges.as_ptr(),
            len: 2,
        },
        ..Default::default()
    };
    let completion =
        PieCompletion { wait_id: 2, target_epoch: 1, terminal_cell: std::ptr::null_mut() };
    assert_eq!(
        unsafe { driver_abi::local::pie_cuda_copy_state(d, &copy, completion) },
        PIE_STATUS_OK,
        "the state fork copies"
    );

    // ── The same decode against slot 0 and against slot 1. ──
    let decode_on = |slot: u32, wait: u64| {
        let dec_ids: [u32; 1] = [next_token];
        let dec_pos: [u32; 1] = [tokens as u32];
        let dec_roster: [u32; 1] = [0];
        let dec_sbi: [u32; 2] = [0, 1];
        let dec_lens: [u32; 1] = [tokens as u32 + 1];
        let dec_qo: [u32; 2] = [0, 1];
        let dec_slots: [u32; 1] = [slot];
        let step = PieStepDesc {
            roster_rows: u32s(&dec_roster),
            sub_batch_indptr: u32s(&dec_sbi),
            sub_batch_class: u32s(&sub_batch_class),
            token_ids: u32s(&dec_ids),
            position_ids: u32s(&dec_pos),
            kv_page_indices: u32s(&kv_page_indices),
            kv_page_indptr: u32s(&kv_page_indptr),
            kv_last_page_lens: u32s(&dec_lens),
            qo_indptr: u32s(&dec_qo),
            rs_slot_ids: u32s(&dec_slots),
            ..Default::default()
        };
        assert_eq!(fire(&step, wait), PIE_STATUS_OK, "the decode fires (slot {slot})");
    };
    decode_on(1, 3);
    decode_on(2, 4);
    assert_eq!(words[1], 3, "three cells published");
    let ring = 4usize; // capacity 3 + 1
    let _ = ring;
    let cell1 = unsafe {
        std::slice::from_raw_parts((chb.mirror_base as *const f32).add(VOCAB), VOCAB)
    };
    let cell2 = unsafe {
        std::slice::from_raw_parts((chb.mirror_base as *const f32).add(2 * VOCAB), VOCAB)
    };
    let (t1, v1) = argmax_of(cell1);
    let (t2, v2) = argmax_of(cell2);
    assert_eq!(t1, t2, "decode from the COPIED slot flips the argmax ({v1} vs {v2})");
    let (mut max_d, mut at, mut n_diff) = (0f32, 0usize, 0usize);
    for t in 0..VOCAB {
        let d = (cell1[t] - cell2[t]).abs();
        if d > 0.0 {
            n_diff += 1;
        }
        if d > max_d {
            (max_d, at) = (d, t);
        }
    }
    eprintln!(
        "copy-vs-copy decode: {n_diff} differing logits, max |d| = {max_d} at {at} \
         ({} vs {})",
        cell1[at], cell2[at]
    );
    assert!(
        max_d < 0.25,
        "the copied slots' decodes drifted past inter-fire jitter: |d|={max_d} at {at}"
    );

    unsafe { driver_abi::local::pie_cuda_destroy(d) };
}
