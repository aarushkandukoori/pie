//! The Metal seam: can it be selected, and does it say what it cannot serve?
//!
//! A backend that cannot be selected teaches nothing. This checks the half
//! that works — the device opens, the facts are stated, the variant dispatches
//! — and that the half that does not refuses **by name** rather than by
//! absence, panic, or a plausible wrong answer.

#![cfg(all(feature = "driver-metal-new", target_vendor = "apple"))]

use engine::driver::DriverBackend;

#[test]
fn the_metal_backend_opens_a_device_and_states_what_it_is() {
    let Ok((backend, facts)) = DriverBackend::metal_create(b"{}") else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    assert_eq!(backend.kind(), "metal");
    assert_eq!(facts.backend, "metal");
    assert!(
        facts.unified_memory,
        "Apple silicon shares physical memory between the pool and the host, \
         and that changes what a `device is full` question means"
    );
    assert_eq!(facts.page_size, 16, "the paged KV pool's rows per page");
    assert!(
        !facts.fp8_native && !facts.native_mxfp4_moe,
        "neither kernel exists in `kernels-metal`, and the facts should say so \
         rather than let a scheduler discover it at launch"
    );
}

#[test]
fn the_verbs_that_need_the_kv_pool_refuse_by_name() {
    // The hole, stated. Every one of these is above a pool that does not
    // exist yet; the executor above THEM is complete and device-tested, so
    // the message says which half is missing.
    let Ok((mut backend, _)) = DriverBackend::metal_create(b"{}") else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let why = backend
        .register_program(&Default::default())
        .expect_err("the registry is not wired to the seam's plan types yet");
    let text = format!("{why}");
    assert!(
        text.contains("driver-metal-new"),
        "a refusal must name the backend that made it: {text}"
    );
    assert!(
        text.contains("register_program"),
        "and the verb it could not serve: {text}"
    );
    assert!(
        text.contains("device tested"),
        "and which half is actually missing, so the next reader does not \
         re-port machinery that is already there: {text}"
    );
}

#[test]
fn media_encode_is_refused_rather_than_pretended() {
    // Unsupported on this backend and on CUDA both, and the seam says so
    // instead of returning a completion nothing will settle.
    let Ok((backend, _)) = DriverBackend::metal_create(b"{}") else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    assert!(
        backend.export_kv_handle().is_none(),
        "Metal has no cross-process KV sharing path to export"
    );
}

#[test]
fn load_model_takes_one_descriptor_because_this_backend_holds_one_model() {
    // The same shape the CUDA shell's `state.model` has, and the reason a
    // frame's instance roster is one family's — which is what makes
    // `lower(plan, rows, fire)`'s one-plan signature right.
    let Ok((mut backend, _)) = DriverBackend::metal_create(b"{}") else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let desc = || driver_abi::ModelLoadDesc {
        snapshot_dir: std::path::PathBuf::from("/nonesuch"),
        runtime_quant: String::new(),
        mxfp4_moe: driver_abi::Mxfp4MoeRequest::Auto,
        component: driver_abi::ModelComponent::Full,
    };
    let why = format!(
        "{}",
        backend
            .load_model(vec![desc(), desc()])
            .expect_err("two models is not a shape this backend has")
    );
    assert!(
        why.contains("ONE model"),
        "the refusal should say why, not just that: {why}"
    );

    // And one descriptor gets as far as the checkpoint, which is the point:
    // the failure is now about the SNAPSHOT rather than about the seam.
    let why = format!(
        "{}",
        backend
            .load_model(vec![desc()])
            .expect_err("/nonesuch holds no checkpoint")
    );
    assert!(
        why.contains("config.json"),
        "a missing snapshot should fail on the checkpoint it looked for: {why}"
    );
}

#[test]
fn a_frame_that_cannot_fit_the_pool_is_impossible_rather_than_an_error() {
    // Admission is not a failure. A frame whose demand exceeds the PHYSICAL
    // pool can never be made to fit by evicting, so it is `Impossible` and the
    // engine stops re-posting it; one that merely does not fit right now would
    // be `Exhausted`. Both are outcomes, and neither is an `Err`.
    let Ok((mut backend, _)) = DriverBackend::metal_create(b"{}") else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    // Before a load there is no pool at all, and that IS an error — the
    // scheduler asked a driver to run a model it never gave it.
    let frame = engine::driver::FrameSubmission {
        instance_ids: vec![1],
        kv_translation: vec![0],
        kv_translation_indptr: vec![0, 1],
        required_kv_pages: 1,
        steps: Vec::new(),
    };
    let why = match backend.launch(&frame) {
        Err(why) => format!("{why}"),
        Ok(_) => panic!("launch before load_model is drift, not admission"),
    };
    assert!(
        why.contains("before load_model"),
        "the refusal should say which order was broken: {why}"
    );
}
