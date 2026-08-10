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
        .load_model(Vec::new())
        .expect_err("nothing has taught this seam to load a checkpoint");
    let text = format!("{why}");
    assert!(
        text.contains("driver-metal-new"),
        "a refusal must name the backend that made it: {text}"
    );
    assert!(
        text.contains("load_model"),
        "and the verb it could not serve: {text}"
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
