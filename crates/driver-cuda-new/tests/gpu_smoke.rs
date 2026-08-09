//! Smoke test: does this crate actually talk to a GPU?
//!
//! Skipped when no device is present: the crate is deliberately buildable and
//! testable on machines with no CUDA at all -- that is what the `cuda-12` /
//! `cuda-13` feature pair plus `fallback-dynamic-loading` buys, and a suite
//! that failed without hardware would throw it away.
//!
//! Run with `--nocapture` to see what was found.

use driver_cuda_new::cuda::{Allocator, Event, OwnedStream};

mod common;
use common::{device_or_skip, gpu_guard};

#[test]
fn a_device_can_be_bound_and_described() {
    let Some(dev) = device_or_skip("device query") else { return };
    let (major, minor) = dev.compute_capability().expect("compute capability");
    let sms = dev.sm_count().expect("sm count");
    let (free, total) = dev.memory_info().expect("memory info");
    let vmm = dev.supports_vmm().expect("vmm support");
    eprintln!(
        "device {}: sm_{major}{minor}, {sms} SMs, {} MiB free / {} MiB total, vmm={vmm}",
        dev.ordinal(),
        free / (1 << 20),
        total / (1 << 20)
    );
    assert!(major >= 5, "this crate targets Maxwell and later");
    assert!(sms > 0);
    assert!(total > 0 && free <= total);
}

#[test]
fn a_round_trip_through_device_memory_returns_what_went_in() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("memcpy round trip") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let alloc = Allocator::new();

    let src: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
    let mut buf = alloc.alloc(src.len()).expect("alloc");
    buf.copy_from_host(&src, stream.as_ref()).expect("h2d");

    let mut back = vec![0u8; src.len()];
    buf.copy_to_host(&mut back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");

    assert_eq!(back, src);
}

#[test]
fn memset_reaches_the_device() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("memset") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let alloc = Allocator::new();

    let mut buf = alloc.alloc(1024).expect("alloc");
    buf.memset(0xab, stream.as_ref()).expect("memset");
    let mut back = vec![0u8; 1024];
    buf.copy_to_host(&mut back, stream.as_ref()).expect("d2h");
    stream.as_ref().synchronize().expect("sync");

    assert!(back.iter().all(|&b| b == 0xab), "memset did not take");
}

#[test]
fn an_event_orders_work_across_two_streams() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("cross-stream event") else { return };
    let a = OwnedStream::new(0).expect("stream a");
    let b = OwnedStream::new(0).expect("stream b");
    let alloc = Allocator::new();

    let payload = vec![7u8; 1 << 20];
    let mut buf = alloc.alloc(payload.len()).expect("alloc");

    buf.copy_from_host(&payload, a.as_ref()).expect("h2d on a");
    let done = Event::new().expect("event");
    a.as_ref().record(&done).expect("record");
    b.as_ref().wait_event(&done).expect("wait");

    let mut back = vec![0u8; payload.len()];
    buf.copy_to_host(&mut back, b.as_ref()).expect("d2h on b");
    b.as_ref().synchronize().expect("sync b");

    assert_eq!(back, payload, "stream b read before stream a's write landed");
}

#[test]
fn a_timing_event_pair_measures_something_nonnegative() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("event timing") else { return };
    let stream = OwnedStream::new(0).expect("stream");
    let alloc = Allocator::new();
    let start = Event::with_timing().expect("start");
    let end = Event::with_timing().expect("end");

    let mut buf = alloc.alloc(1 << 22).expect("alloc");
    stream.as_ref().record(&start).expect("record start");
    buf.memset(1, stream.as_ref()).expect("memset");
    stream.as_ref().record(&end).expect("record end");
    end.synchronize().expect("sync");

    let ms = start.elapsed_ms(&end).expect("elapsed");
    assert!((0.0..10_000.0).contains(&ms), "implausible elapsed time {ms}ms");
}
