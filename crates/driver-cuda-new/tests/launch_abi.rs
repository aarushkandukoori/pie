//! The launch ABI pilot: `rope`'s twelve launchers, proven by the C++ compiler.
//!
//! Every other area in this crate is proven by a differential oracle — build
//! the C++, sweep both over a grid, require byte-identical output, pin the
//! C++'s hash. That protocol does not reach the launcher boundary, and the
//! reason is worth stating: a differential oracle proves that two
//! implementations of a *described* contract agree. The launchers have no
//! described contract. `KernelSig` carried a kernel's name, its plan, its
//! capabilities and its sink, and not one word about how to call it.
//!
//! So the ABI pilot does not port a launcher. It writes the contract down —
//! `KernelSig::operands` — and then proves the writing.
//!
//! ## The proof
//!
//! `kernels_cuda::abi::emit_c_shim` generates one `extern "C"` function per
//! row whose body CALLS the launcher, with the real `rope.hpp` in scope. The
//! generated translation unit is then compiled. A row that misstates an
//! operand's type, constness, width, position, or the arity of the whole list
//! does not compile, because C++ overload resolution is deciding — not a
//! string comparison, and not a golden that could drift.
//!
//! This is STRICTER than a golden, and it is worth being explicit about why:
//! a golden proves the two sides agreed on the grid that was swept, and a
//! mutation suite estimates how much of the contract the grid reached. Here
//! there is no grid and nothing to estimate. The compiler checks the entire
//! signature or refuses the file.
//!
//! Consequently this file pins no hash and has no `mutate.sh`. What replaces
//! them is [`a_wrong_row_does_not_compile`], which corrupts a row and
//! requires the compile to FAIL — the same question a mutation suite asks
//! ("would the proof notice?"), answered exactly instead of statistically.

#![cfg(feature = "_cuda")]

use std::sync::atomic::{AtomicU64, Ordering};

use std::path::{Path, PathBuf};
use std::process::Command;

use driver_cuda_new::launch::{
    AttentionWorkspaceView, HopperPrefillPlan, KvCacheLayerView, MlaCacheLayerView,
    YarnOriginalParams,
};
use kernels::{KernelSig, Operand, Ty};
use kernels_cuda::abi::Record;

/// Where `kernels-cuda`'s sources are, relative to this crate.
fn csrc() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels-cuda/csrc/src")
}

/// The header the pilot's rows are declared in.
const ROPE_HPP: &str = "rope/rope.hpp";

/// Compile a generated shim against the real headers.
///
/// `-fsyntax-only`: nothing is linked, so this needs neither the built
/// archive nor nvcc. And the only CUDA name `rope.hpp` uses is
/// `cudaStream_t`, which `tests/oracle/launch_abi/stub/` supplies — so it needs
/// no CUDA toolkit either, which is what the CI job that runs this crate
/// promises. The stub directory is the ONLY include path added besides
/// `csrc/src`, so the answer does not depend on which CUDA is installed.
///
/// The scratch directory is per CALL, not per process. Test binaries run
/// their cases on threads of one process, so a pid-named directory is shared
/// state: two cases race on `shim.cpp`, and the one that reads a neighbour's
/// text is answered about the wrong shim. That failure is silent in the only
/// direction that matters — a corrupted row compiles, because what actually
/// got compiled was the good one — so it reads as "the proof is not
/// watching" rather than as a harness bug.
fn compile(shim: &str) -> Result<(), String> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "pie-launch-abi-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let src = dir.join("shim.cpp");
    std::fs::write(&src, shim).expect("write shim");

    let stub = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle/launch_abi/stub");
    let out = Command::new("g++")
        .arg("-std=c++20")
        .arg("-fsyntax-only")
        .arg(format!("-I{}", stub.display()))
        .arg(format!("-I{}", csrc().display()))
        .arg(&src)
        .output()
        .expect("g++ must be available");
    let _ = std::fs::remove_dir_all(&dir);
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).into_owned())
    }
}

fn rope_shim(table: &'static [KernelSig]) -> String {
    kernels_cuda::abi::emit_c_shim(&[table], &[ROPE_HPP]).expect("no entry-point collisions")
}

/// The headers `attn`'s rows are declared in.
///
/// Every `attn/*.hpp`, plus `gemm/gemm.hpp` — two of `attn`'s rows are the MLA
/// absorb pair, which live in `gemm.hpp` because they are `cublas` calls. A
/// family is not proven while two of its rows sit outside the shim, so the
/// list follows the ROWS rather than the directory.
fn attn_headers() -> Vec<String> {
    let mut hs: Vec<String> = std::fs::read_dir(csrc().join("attn"))
        .expect("attn/")
        .filter_map(|e| {
            let n = e.ok()?.file_name().into_string().ok()?;
            n.ends_with(".hpp").then(|| format!("attn/{n}"))
        })
        .collect();
    hs.sort();
    hs.push("gemm/gemm.hpp".into());
    hs
}

fn attn_shim(table: &'static [KernelSig]) -> String {
    let hs = attn_headers();
    let refs: Vec<&str> = hs.iter().map(String::as_str).collect();
    kernels_cuda::abi::emit_c_shim(&[table], &refs).expect("no entry-point collisions")
}

/// The headers `norm`'s rows are declared in — every `norm/*.hpp`.
fn norm_headers() -> Vec<String> {
    let mut hs: Vec<String> = std::fs::read_dir(csrc().join("norm"))
        .expect("norm/")
        .filter_map(|e| {
            let n = e.ok()?.file_name().into_string().ok()?;
            n.ends_with(".hpp").then(|| format!("norm/{n}"))
        })
        .collect();
    hs.sort();
    hs
}

/// `norm`'s rows, proven the same way `rope`'s and `attn`'s are.
///
/// Twenty-eight launchers across seven headers, and the family every
/// other one leans on: a wrong row here is a wrong argument in an arm
/// that four executors reach.
#[test]
fn every_norm_row_states_its_launcher_exactly() {
    let table = kernels_cuda::norm::KERNELS;
    let stated = table.iter().filter(|k| !k.operands.is_empty()).count();
    assert_eq!(
        stated,
        table.len(),
        "{} of {} norm rows are unstated, so the shim silently skips them",
        table.len() - stated,
        table.len()
    );
    let hs = norm_headers();
    let refs: Vec<&str> = hs.iter().map(String::as_str).collect();
    let shim = kernels_cuda::abi::emit_c_shim(&[table], &refs)
        .expect("no entry-point collisions");
    if let Err(err) = compile(&shim) {
        panic!(
            "the generated shim does not compile, so a row misstates its \
             launcher:\n{err}"
        );
    }
}

/// `mlp`'s rows. Sixteen activations across two headers, and the family
/// whose default arguments make a hand-written binding easiest to get
/// wrong -- `gpt_oss_glu_bf16` alone carries three.
#[test]
fn every_mlp_row_states_its_launcher_exactly() {
    let table = kernels_cuda::mlp::KERNELS;
    let stated = table.iter().filter(|k| !k.operands.is_empty()).count();
    assert_eq!(
        stated,
        table.len(),
        "{} of {} mlp rows are unstated, so the shim silently skips them",
        table.len() - stated,
        table.len()
    );
    let shim =
        kernels_cuda::abi::emit_c_shim(&[table], &["mlp/swiglu.hpp", "mlp/gaussian_topk.hpp"])
            .expect("no entry-point collisions");
    if let Err(err) = compile(&shim) {
        panic!(
            "the generated shim does not compile, so a row misstates its \
             launcher:\n{err}"
        );
    }
}

/// `quant`'s and `moe`'s rows, and the STATED half of `layout`'s and
/// `gemm`'s.
///
/// `gemm`'s scaled entry points are here because 1b made them
/// spellable: the storage a weight is in used to reach the dispatcher
/// inside a `WeightView`, a descriptor the driver BUILT from a
/// per-layer struct no statement mentioned. Assembling it inside the
/// launcher and taking its fields flat is what let a row describe the
/// call at all -- a struct is not something the operand vocabulary can
/// state, and giving it a kind would have stated nothing.
///
/// Not a whole-family assertion like `norm`'s and `mlp`'s, because two
/// of these families carry rows the shim cannot reach yet and saying so
/// is better than a count that quietly excludes them:
///
///   * `dist::` and `comm::` name METHODS on `NcclComm`, not free
///     launchers, so `::pie_cuda_driver::kernels::dist::all_reduce_bf16`
///     does not exist to forward to. They need free wrappers first.
///   * `gemm`'s remaining rows take a `WeightView` or pointer arrays,
///     which the operand vocabulary has no kind for yet.
///
/// What IS stated compiles, which is the claim this test can make.
#[test]
fn the_stated_quant_layout_gemm_and_moe_rows_describe_their_launchers() {
    let tables: [&'static [KernelSig]; 4] = [
        kernels_cuda::quant::KERNELS,
        kernels_cuda::layout::KERNELS,
        kernels_cuda::gemm::KERNELS,
        kernels_cuda::moe::KERNELS,
    ];
    let headers = [
        "quant/dequant_fp4.hpp",
        "quant/dequant_fp8.hpp",
        "quant/dequant_wna16.hpp",
        "quant/dtype_cast.hpp",
        "quant/mxfp4_marlin.hpp",
        "layout/embed.hpp",
        "layout/gather_rows.hpp",
        "layout/slot_ops.hpp",
        "layout/split_gate_up.hpp",
        "layout/deinterleave.hpp",
        "gemm/gemm.hpp",
        "gemm/gemv.hpp",
        "comm/custom_all_reduce.hpp",
        "moe/dsv4_routing.hpp",
        "moe/moe_dispatch.hpp",
        "moe/moe_grouped_gemm.hpp",
        "moe/flashinfer_moe.hpp",
        "sample/argmax.hpp",
        "../third_party/marlin_moe/marlin_moe_wrapper.hpp",
        "moe/topk_sigmoid.hpp",
        "moe/topk_softmax.hpp",
    ];
    let shim = kernels_cuda::abi::emit_c_shim(&tables, &headers)
        .expect("no entry-point collisions");
    if let Err(err) = compile(&shim) {
        panic!(
            "the generated shim does not compile, so a row misstates its \
             launcher:\n{err}"
        );
    }
}

/// `ssm`'s rows — the largest single family, and the one whose ten
/// recurrence spellings differ only by which state dtype and whether the
/// heads are grouped.
///
/// Ten near-identical argument lists is exactly where a hand-written
/// binding goes wrong quietly: `state_base` is `float*` in six of them
/// and `void*` in four, and the two are the same pointer at a call site.
#[test]
fn every_ssm_row_states_its_launcher_exactly() {
    let table = kernels_cuda::ssm::KERNELS;
    // THREE rows stay unstated, and naming them is the point of pinning
    // the count rather than asserting it away:
    //
    //
    // The two `build_nemotron_moe_ptrs_*` builders came IN with the
    // pointer-array kinds -- they take `const void* const*` for the
    // weights they read and `void**` for the arrays they fill, and only
    // `BufArray` vs `BufArrayOutMut` makes the difference a compile
    // error instead of a builder writing an array it was handed to
    // read.
    let stated = table.iter().filter(|k| !k.operands.is_empty()).count();
    assert_eq!(
        stated,
        table.len(),
        "{} of {} ssm rows are unstated, so the shim silently skips them",
        table.len() - stated,
        table.len()
    );
    let shim = kernels_cuda::abi::emit_c_shim(
        &[table],
        &[
            "ssm/causal_conv1d.hpp",
            "ssm/flashinfer_mamba.hpp",
            "ssm/gated_delta_net.hpp",
            "ssm/kda.hpp",
            "ssm/nemotron_h.hpp",
        ],
    )
    .expect("no entry-point collisions");
    if let Err(err) = compile(&shim) {
        panic!(
            "the generated shim does not compile, so a row misstates its \
             launcher:\n{err}"
        );
    }
}

/// The generated DISPATCH, which is what the shim was proved for.
///
/// A row that states its operand types AND where each argument comes
/// from is a row the call can be derived from — so the arm nobody
/// writes is the arm nobody writes wrong. This prints what comes out
/// for the rows that have said both.
#[test]
fn the_dispatch_generates_from_rows_that_state_their_sources() {
    let text = kernels_cuda::abi::emit_dispatch(
        &[
            kernels_cuda::rope::KERNELS,
            kernels_cuda::norm::KERNELS,
            kernels_cuda::mlp::KERNELS,
            kernels_cuda::layout::KERNELS,
            kernels_cuda::quant::KERNELS,
            kernels_cuda::moe::KERNELS,
            kernels_cuda::attn::KERNELS,
        ],
        "c",
    );
    let cases = text.matches("if (sym == ").count();
    eprintln!("{text}");
    eprintln!("generated {cases} dispatch case(s)");
    // A generator that produced nothing would pass every other
    // assertion here, so the floor is the assertion.
    assert!(
        cases >= 1,
        "no row states its sources, so nothing was generated"
    );
}

/// The committed dispatch is what the generator emits today.
///
/// `.inc` discipline, one layer down: the file is C++ a reader opens,
/// so it is committed and the diff is the review — and a table edit
/// that changes what it emits has to be regenerated rather than
/// silently diverge from the arms it replaced.
#[test]
fn the_committed_dispatch_is_regeneration_clean() {
    let path = format!(
        "{}/../driver-cuda/csrc/src/model/declared/generated_dispatch.inc",
        env!("CARGO_MANIFEST_DIR")
    );
    let committed = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let fresh = kernels_cuda::abi::emit_dispatch(
        &[
            kernels_cuda::attn::KERNELS,
            kernels_cuda::gemm::KERNELS,
            kernels_cuda::layout::KERNELS,
            kernels_cuda::mlp::KERNELS,
            kernels_cuda::moe::KERNELS,
            kernels_cuda::norm::KERNELS,
            kernels_cuda::quant::KERNELS,
            kernels_cuda::rope::KERNELS,
            kernels_cuda::sample::KERNELS,
            kernels_cuda::ssm::KERNELS,
        ],
        "c",
    );
    assert_eq!(
        committed, fresh,
        "the committed dispatch is stale; regenerate with \
         `cargo run -p kernels-cuda --bin emit-dispatch`"
    );
}

/// The pilot itself: every stated `rope` row describes its launcher exactly.
#[test]
fn every_rope_row_states_its_launcher_exactly() {
    let shim = rope_shim(kernels_cuda::rope::KERNELS);
    if let Err(err) = compile(&shim) {
        panic!(
            "the generated shim does not compile, so a row misstates its \
             launcher:\n{err}\n--- shim ---\n{shim}"
        );
    }
}

/// The same proof at family scale: all fifty `attn` rows, ~700 operands.
///
/// `rope` was twelve rows of scalars and buffers. `attn` is what the ABI
/// actually has to survive: views passed BY VALUE, plan caches passed as
/// `const&` to a type the header never defines, a `cublasHandle_t` where a
/// stream would be, and both halves of every const/mut pointer pair. If the
/// vocabulary in `kernels::Ty` were short of any of that, this would not
/// compile — which is the point of running it as one shim rather than fifty.
#[test]
fn every_attn_row_states_its_launcher_exactly() {
    let table = kernels_cuda::attn::KERNELS;
    let stated = table.iter().filter(|k| !k.operands.is_empty()).count();
    assert_eq!(
        stated,
        table.len(),
        "{} of {} attn rows are unstated, so the shim silently skips them",
        table.len() - stated,
        table.len()
    );
    let shim = attn_shim(table);
    if let Err(err) = compile(&shim) {
        panic!(
            "the generated shim does not compile, so a row misstates its \
             launcher:\n{err}"
        );
    }
}

/// Every launcher `rope.hpp` declares has a row.
///
/// The other half of the crate's invariant, and the half a generated shim
/// cannot reach: emitting from the table proves each ROW is real, never that
/// the table is complete. This is what the pilot found — twelve declarations
/// against ten rows, with `rope_bf16` and `rope_partial_bf16_position_delta`
/// present in the header, called by the driver, and named nowhere in the
/// table the compiler plans against.
#[test]
fn every_launcher_the_header_declares_has_a_row() {
    let text = std::fs::read_to_string(csrc().join(ROPE_HPP)).expect("rope.hpp");
    let declared: Vec<String> = text
        .lines()
        .filter_map(|l| l.strip_prefix("void "))
        .filter_map(|l| l.split_once('('))
        .map(|(name, _)| name.trim().to_string())
        .collect();
    assert!(
        declared.len() >= 12,
        "the scan found {} declarations, so its shape assumption broke",
        declared.len()
    );

    let missing: Vec<&String> = declared
        .iter()
        .filter(|d| {
            !kernels_cuda::rope::KERNELS
                .iter()
                .any(|k| k.symbol == format!("rope::{d}"))
        })
        .collect();
    assert!(missing.is_empty(), "declared but not in the table: {missing:?}");
}

/// Why a launcher `attn`'s headers declare is allowed to have no row.
#[derive(Clone, Copy, PartialEq)]
enum NoRow {
    /// A prepare. The table carries these as `needs = Prepare::*` on the
    /// dispatch that obligates them, not as rows of their own — see the
    /// `kernels` crate's own table of what each declaration replaces.
    Prepare,
    /// One-time device work at startup. Real work, but not part of any
    /// forward, so no declaration ever names it.
    Warmup,
    /// The driver calls it; `dsl::cuda` does not name it. The table's subject
    /// is the planner's vocabulary — `model`'s `kernels_table` asserts the
    /// two are EQUAL — so a launcher the driver reaches for on its own is
    /// correctly absent. `split_qkv_bf16` is the loud case: 390 call sites.
    DriverInternal,
    /// Only sibling `.cu` files in this crate call it. Checked below rather
    /// than believed.
    KernelsInternal,
}

/// Every launcher `attn`'s headers declare is a row, or is one of four
/// documented kinds of not-a-row.
///
/// The rope pilot could assert the flat thing — every declaration has a row —
/// because for `rope` it is true. For `attn` it is false BY DESIGN, and a
/// test that asserted it anyway would have to be deleted rather than
/// answered. What is actually load-bearing is that no launcher joins these
/// headers without someone deciding which kind it is: 77 declarations against
/// 50 rows is not a gap to close, it is 27 decisions, and this is where they
/// are written down.
///
/// `KernelsInternal` is not taken on trust — the claim is "the driver never
/// calls this", the driver's sources are next door, and so it is checked.
#[test]
fn every_attn_launcher_is_a_row_or_a_stated_exception() {
    #[rustfmt::skip]
    let exceptions: &[(&str, NoRow)] = &[
        ("plan_attention_flashinfer_decode_bf16",       NoRow::Prepare),
        ("plan_attention_flashinfer_prefill_bf16",      NoRow::Prepare),
        ("plan_attention_flashinfer_prefill_sm90_bf16", NoRow::Prepare),
        ("plan_attention_mla_bf16",                     NoRow::Prepare),
        ("prepare_attention_xqa_decode_bf16",           NoRow::Prepare),
        ("set_decode_plan_int_base",                    NoRow::Prepare),
        ("xqa_decode_bf16_warmup_current_device",       NoRow::Warmup),
        ("xqa_decode_bf16_gqa5_warmup_current_device",  NoRow::Warmup),
        ("split_qkv_bf16",                              NoRow::DriverInternal),
        ("split_qkv_bf16_devwin",                       NoRow::DriverInternal),
        ("pack_dense_mask",                             NoRow::DriverInternal),
        ("pack_structured_mask",                        NoRow::DriverInternal),
        ("copy_kv_cells_bf16",                          NoRow::DriverInternal),
        ("attention_flashinfer_prefill_bf16",           NoRow::KernelsInternal),
        ("attention_flashinfer_prefill_custom_bf16",    NoRow::KernelsInternal),
        ("dispatch_attention_flashinfer_decode_capture_bf16", NoRow::KernelsInternal),
        ("dispatch_attention_flashinfer_prefill_custom_bf16", NoRow::KernelsInternal),
        ("attention_mtp_history_bf16",                  NoRow::KernelsInternal),
        ("attention_naive_bf16",                        NoRow::KernelsInternal),
        ("attention_naive_paged_custom",                NoRow::KernelsInternal),
        ("attention_naive_paged_decode",                NoRow::KernelsInternal),
        ("attention_xqa_decode_bf16",                   NoRow::KernelsInternal),
        ("add_ape_f32",                                 NoRow::KernelsInternal),
        ("attention_compressed_bf16",                   NoRow::KernelsInternal),
        ("average_pool_bf16",                           NoRow::KernelsInternal),
        ("dsv4_compress_gather_bf16",                   NoRow::KernelsInternal),
        ("gated_softmax_pool_bf16",                     NoRow::KernelsInternal),
        ("write_kv_to_pages_at_positions_bf16",         NoRow::KernelsInternal),
        ("write_mla_to_pages_bf16",                     NoRow::KernelsInternal),
    ];

    let declared = declared_launchers();
    assert!(
        declared.len() >= 77,
        "the scan found {} declarations, so its shape assumption broke",
        declared.len()
    );

    let has_row = |n: &str| {
        kernels_cuda::attn::KERNELS
            .iter()
            .any(|k| k.symbol == format!("attn::{n}") || k.symbol.ends_with(&format!("::{n}")))
    };
    let undecided: Vec<&String> = declared
        .iter()
        .filter(|d| !has_row(d) && !exceptions.iter().any(|(n, _)| n == d))
        .collect();
    assert!(
        undecided.is_empty(),
        "declared in attn/, no row, and no stated reason: {undecided:?}"
    );

    let stale: Vec<&str> = exceptions
        .iter()
        .map(|(n, _)| *n)
        .filter(|n| !declared.iter().any(|d| d == n))
        .collect();
    assert!(stale.is_empty(), "exception for a launcher no header declares: {stale:?}");

    // The `KernelsInternal` claim, checked against the driver's sources.
    let driver = Path::new(env!("CARGO_MANIFEST_DIR")).join("../driver-cuda/csrc/src");
    let mut driver_text = String::new();
    collect_sources(&driver, &mut driver_text);
    assert!(
        driver_text.len() > 1_000_000,
        "only {} bytes of driver source found, so the check is vacuous",
        driver_text.len()
    );
    let wrong: Vec<&str> = exceptions
        .iter()
        .filter(|(_, why)| *why == NoRow::KernelsInternal)
        .map(|(n, _)| *n)
        .filter(|n| mentions_word(&driver_text, n))
        .collect();
    assert!(
        wrong.is_empty(),
        "called `KernelsInternal` but the driver calls it, so it is really \
         DriverInternal or a missing row: {wrong:?}"
    );
}

/// Every `void` launcher declared across `attn/*.hpp`, by name.
fn declared_launchers() -> Vec<String> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(csrc().join("attn")).expect("attn/") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("hpp") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("header");
        for line in text.lines() {
            let Some(rest) = line.strip_prefix("void ") else { continue };
            let Some((name, _)) = rest.split_once('(') else { continue };
            if name.chars().all(|c| c.is_alphanumeric() || c == '_') && !name.is_empty() {
                out.push(name.to_string());
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

fn collect_sources(dir: &Path, out: &mut String) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_sources(&p, out);
        } else if matches!(
            p.extension().and_then(|e| e.to_str()),
            Some("cpp" | "hpp" | "cu" | "inc")
        ) && let Ok(t) = std::fs::read_to_string(&p)
        {
            out.push_str(&t);
            out.push('\n');
        }
    }
}

/// `needle` appears in `hay` as a whole identifier.
///
/// Substring matching would answer the wrong question here: every
/// `attention_naive_bf16` is inside some `attention_naive_bf16_something`,
/// and a check that cannot tell them apart cannot fail usefully.
fn mentions_word(hay: &str, needle: &str) -> bool {
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    hay.match_indices(needle).any(|(i, _)| {
        let before = hay[..i].chars().next_back().is_none_or(|c| !ident(c));
        let after = hay[i + needle.len()..].chars().next().is_none_or(|c| !ident(c));
        before && after
    })
}

/// A row with no operands is not a row that takes none.
///
/// The distinction matters while the table is being filled a family at a
/// time: emitting an unstated row as a nullary `extern "C"` would generate a
/// call the compiler rejects for the wrong reason, and would make "this
/// family is done" indistinguishable from "this family is empty".
///
/// Asked of a SYNTHETIC row rather than of whichever family happens to be
/// empty today. It used to point at `attn`, and filling `attn` turned a
/// passing test into a failing one without anything being wrong — the check
/// is about `stated()`, so its subject should be too.
#[test]
fn an_unstated_row_is_skipped_rather_than_called_with_nothing() {
    let stated = kernels_cuda::rope::KERNELS
        .iter()
        .find(|k| !k.operands.is_empty())
        .expect("some rope row is stated");
    let row: &'static [KernelSig] =
        Vec::leak(vec![KernelSig { operands: &[], ..*stated }]);
    let shim = rope_shim(row);
    assert!(
        !shim.contains("extern \"C\""),
        "the row states no operands, so nothing should be emitted:\n{shim}"
    );
}

/// Corrupting a row must break the build — the mutation suite, answered
/// exactly.
///
/// Each case changes ONE thing a hand-written binding gets wrong, and every
/// one of them has to be caught. The last two are the interesting ones: they
/// are not type errors, they are an operand list of the right types in the
/// wrong ORDER, which is precisely the failure a `void*`-flattened ABI cannot
/// see and this one can.
#[test]
fn a_wrong_row_does_not_compile() {
    let base = kernels_cuda::rope::KERNELS
        .iter()
        .find(|k| k.symbol == "rope::qk_rmsnorm_rope_bf16")
        .expect("the pilot row");

    // `q_weight`/`k_weight` are `const void*`; `positions` is `const i32*`;
    // the extents are `int` and the two rates are `float`.
    let ops = base.operands;
    let swap = |i: usize, j: usize| {
        let mut v: Vec<Operand> = ops.to_vec();
        v.swap(i, j);
        v
    };
    let retype = |i: usize, ty: Ty| {
        let mut v: Vec<Operand> = ops.to_vec();
        v[i].ty = ty;
        v
    };

    let cases: Vec<(&str, Vec<Operand>)> = vec![
        ("a written buffer is claimed to be read-only", retype(0, Ty::Buf)),
        ("a read-only weight is claimed to be written", retype(2, Ty::BufMut)),
        ("positions loses its element type", retype(4, Ty::Buf)),
        ("an extent is widened to a float", retype(5, Ty::F32)),
        ("a rate is narrowed to an int", retype(9, Ty::I32)),
        ("the stream is dropped", ops[..ops.len() - 1].to_vec()),
        ("an operand is invented", {
            let mut v = ops.to_vec();
            v.insert(5, Operand { name: "extra", ty: Ty::I32, nullable: false, source: kernels::Source::Unbound });
            v
        }),
        ("q and k_weight trade places", swap(0, 3)),
        ("an extent and a rate trade places", swap(6, 9)),
    ];

    for (what, operands) in cases {
        let leaked: &'static [Operand] = Vec::leak(operands);
        let row: &'static [KernelSig] = Vec::leak(vec![KernelSig {
            name: base.name,
            symbol: base.symbol,
            whole: base.whole,
            needs: base.needs,
            lacks: base.lacks,
            sink: base.sink,
            in_place: base.in_place,
            depth_prefix_plan: base.depth_prefix_plan,
            operands: leaked,
            returns: base.returns,
            axes: base.axes,
        }]);
        assert!(
            compile(&rope_shim(row)).is_err(),
            "a row where {what} still compiled, so the proof is not watching"
        );
    }
}

/// The control: a change that is NOT a mistake must still compile.
///
/// Without this the test above passes for a build that is broken for some
/// unrelated reason, and every mutation registers as caught. Renaming an
/// operand is the right control here because the name is prose — the table
/// says so — so a rename must be invisible to the compiler while touching
/// exactly the text the mutations touch.
#[test]
fn renaming_an_operand_is_not_a_mistake() {
    let base = kernels_cuda::rope::KERNELS
        .iter()
        .find(|k| k.symbol == "rope::qk_rmsnorm_rope_bf16")
        .expect("the pilot row");
    let renamed: &'static [Operand] = Vec::leak(
        base.operands
            .iter()
            .enumerate()
            .map(|(i, o)| Operand {
                name: Box::leak(format!("arg{i}").into_boxed_str()),
                ..*o
            })
            .collect::<Vec<_>>(),
    );
    let row: &'static [KernelSig] = Vec::leak(vec![KernelSig {
        name: base.name,
        symbol: base.symbol,
        whole: base.whole,
        needs: base.needs,
        lacks: base.lacks,
        sink: base.sink,
        in_place: base.in_place,
        depth_prefix_plan: base.depth_prefix_plan,
        operands: renamed,
        returns: base.returns,
        axes: base.axes,
    }]);
    if let Err(err) = compile(&rope_shim(row)) {
        panic!("the control failed to compile, so the mutations prove nothing:\n{err}");
    }
}

/// The Rust bindings declare exactly what the C++ shim defines.
///
/// Both are generated from one row, so this cannot fail by drift; what it
/// pins is that the two emitters agree on the ENTRY POINT spelling, which is
/// the one string the linker matches on and the one thing neither compiler
/// checks.
#[test]
fn the_rust_bindings_name_the_symbols_the_shim_defines() {
    let shim = rope_shim(kernels_cuda::rope::KERNELS);
    let rs = kernels_cuda::abi::emit_rust_bindings(&[kernels_cuda::rope::KERNELS]);
    for k in kernels_cuda::rope::KERNELS {
        let entry = kernels_cuda::abi::entry_name(k.symbol);
        assert!(shim.contains(&format!("void {entry}(")), "{entry} not defined");
        assert!(rs.contains(&format!("fn {entry}(")), "{entry} not declared");
    }
}

// ---------------------------------------------------------------------------
// Records: the operands that are neither a scalar nor a pointer.
// ---------------------------------------------------------------------------

/// Every mirrored record, with the offsets its Rust side computes.
fn records() -> Vec<Record> {
    vec![
        kernels_cuda::record!(KvCacheLayerView => "::pie_cuda_driver::KvCacheLayerView" {
            layer, source_layer, num_pages, page_size, num_kv_heads, head_dim,
            scheme, storage_dtype, block_size,
            k_pages, v_pages, k_scales, v_scales, k_bf16_pages, v_bf16_pages,
            k_env_min, k_env_max,
            hnd_layout, native_bf16,
        }),
        kernels_cuda::record!(AttentionWorkspaceView => "::pie_cuda_driver::AttentionWorkspaceView" {
            float_buffer, float_bytes, int_buffer, int_bytes, page_locked_int,
        }),
        kernels_cuda::record!(MlaCacheLayerView => "::pie_cuda_driver::MlaCacheLayerView" {
            layer, num_pages, page_size, kv_lora_rank, qk_rope_head_dim,
            ckv_pages, kpe_pages,
        }),
        kernels_cuda::record!(HopperPrefillPlan => "::pie_cuda_driver::kernels::attn::HopperPrefillPlan" {
            qo_tile_indices_offset, qo_indptr_offset, kv_indptr_offset,
            qo_len_offset, kv_len_offset, head_indices_offset,
            work_indptr_offset, batch_indices_offset,
            same_schedule_for_all_heads,
            total_tokens, num_requests, num_q_heads, num_kv_heads, head_dim,
        }),
        kernels_cuda::record!(YarnOriginalParams => "::pie_cuda_driver::kernels::attn::YarnOriginalParams" {
            factor, beta_fast, beta_slow, attention_factor, original_max_position,
        }),
    ]
}

/// The headers the mirrored records are declared in.
///
/// One list, used by every layout case including the mutation ones. That
/// matters more than it looks: those cases assert a TU fails to compile, so a
/// missing include would make them pass for the wrong reason — the mutation
/// would never be what broke it. When `records()` grew from one to five this
/// was the bug, and this is the shape that does not have it.
const MIRROR_HPPS: &[&str] = &[
    "attn/kv_cache_view.hpp",
    "attention_workspace_view.hpp",
    "attn/mla_cache_view.hpp",
    "attn/attention_flashinfer_hopper.hpp",
    "attn/mla_paged.hpp",
];

/// A `#[repr(C)]` mirror really does have the C++ record's layout.
///
/// This is the claim that decides whether a POD operand is a port or a
/// wrapper. If it holds, `KvCacheLayerView` crosses the boundary as itself —
/// no accessor shims, no field-by-field constructor, no copy — and every
/// other descriptor in the launcher surface is the same kind of thing.
#[test]
fn the_mirrors_have_the_layout_the_cpp_has() {
    let tu = kernels_cuda::abi::emit_layout_assertions(&records(), MIRROR_HPPS);
    if let Err(err) = compile(&tu) {
        panic!("a mirror disagrees with the C++ record:\n{err}\n--- tu ---\n{tu}");
    }
}

/// One way a mirror can drift from the record it claims to describe.
type Mutation = Box<dyn Fn(&mut Record)>;

/// And the proof notices when it stops being true.
///
/// The mutation suite for the layout claim. Each case is a way a mirror can
/// drift from a record that is edited on the other side of the boundary, and
/// the last is the one `sizeof` alone would miss: a member APPENDED to the
/// C++ lands in the tail padding an 8-aligned record already has, so size,
/// alignment and every existing offset all still agree. Only the member-count
/// binding catches it.
#[test]
fn a_wrong_mirror_does_not_compile() {
    let bad = |mutate: &dyn Fn(&mut Record)| {
        let mut rs = records();
        mutate(&mut rs[0]);
        compile(&kernels_cuda::abi::emit_layout_assertions(&rs, MIRROR_HPPS))
    };

    let cases: Vec<(&str, Mutation)> = vec![
        ("the record is one byte bigger", Box::new(|r: &mut Record| r.size += 1)),
        ("the record is over-aligned", Box::new(|r: &mut Record| r.align *= 2)),
        (
            "a field moves by one byte",
            Box::new(|r: &mut Record| r.fields[3].1 += 1),
        ),
        (
            "two fields of the same width trade places",
            Box::new(|r: &mut Record| {
                let (a, b) = (r.fields[9].1, r.fields[10].1);
                r.fields[9].1 = b;
                r.fields[10].1 = a;
            }),
        ),
        (
            "a field the C++ does not have is claimed",
            Box::new(|r: &mut Record| r.fields.push(("no_such_field", 0))),
        ),
        (
            "a field the C++ HAS is dropped",
            Box::new(|r: &mut Record| {
                r.fields.pop();
            }),
        ),
    ];

    for (what, mutate) in cases {
        assert!(
            bad(&*mutate).is_err(),
            "a mirror where {what} still compiled, so the proof is not watching"
        );
    }
}

/// The control for the layout proof: renaming the BINDINGS is not a mistake.
///
/// The member-count check binds positionally, so the names it invents carry
/// no claim. If changing them broke the build, the mutation above that drops
/// a field would be "caught" for the wrong reason.
#[test]
fn the_member_count_check_does_not_depend_on_field_names() {
    let mut rs = records();
    let n = rs[0].fields.len();
    for (i, f) in rs[0].fields.iter_mut().enumerate() {
        f.0 = Box::leak(format!("z{}", n - i).into_boxed_str());
    }
    // The offsetof asserts DO name fields, so drop them and keep the rest:
    // what is under test is the binding, not the offsets.
    let tu = kernels_cuda::abi::emit_layout_assertions(&rs, MIRROR_HPPS)
        .lines()
        .filter(|l| !l.contains("offsetof"))
        .filter(|l| !l.contains("is not at"))
        .collect::<Vec<_>>()
        .join("\n");
    if let Err(err) = compile(&tu) {
        panic!("the control failed, so the layout mutations prove nothing:\n{err}");
    }
}

/// The member-count binding is load-bearing, not decoration.
///
/// Dropping the last field is the case the docs on
/// `emit_layout_assertions` claim only the binding can see. That claim is
/// worth checking rather than asserting: here the same mutation is compiled
/// twice, once with the binding and once with it stripped, and the stripped
/// build has to SUCCEED. If it failed, `sizeof` would already have been
/// catching this and the binding would be ceremony.
#[test]
fn without_the_binding_a_dropped_field_would_go_unnoticed() {
    let mut rs = records();
    rs[0].fields.pop();
    let tu = kernels_cuda::abi::emit_layout_assertions(&rs, MIRROR_HPPS);
    assert!(compile(&tu).is_err(), "with the binding, this must fail");

    let without = tu
        .lines()
        .take_while(|l| !l.contains("Exactly"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        without.contains("sizeof") && without.contains("offsetof"),
        "the stripped translation unit must still make the other two claims"
    );
    if let Err(err) = compile(&without) {
        panic!("sizeof/offsetof already caught it, so the binding is ceremony:\n{err}");
    }
}

/// Every launcher the GENERATED dispatch calls is declared by a header
/// `execute.hpp` includes.
///
/// The generator emits a CALL and nothing else. Whether that call has a
/// declaration in scope is the including file's business, and the
/// including file is one: `model/declared/execute.hpp`, which pulls the
/// `.inc` in mid-function. So a row that starts generating drags a
/// header requirement with it, and until this test existed the only
/// thing that noticed was a CUDA build — which this crate's CI job does
/// not run.
///
/// It checks DIRECT includes, deliberately. A launcher reachable only
/// through some other header's include is a build that works by
/// accident, and the fix for a failure here is one line either way.
#[test]
fn every_generated_call_has_its_header_in_scope() {
    let dispatch = kernels_cuda::abi::emit_dispatch(
        &[
            kernels_cuda::attn::KERNELS,
            kernels_cuda::gemm::KERNELS,
            kernels_cuda::layout::KERNELS,
            kernels_cuda::mlp::KERNELS,
            kernels_cuda::moe::KERNELS,
            kernels_cuda::norm::KERNELS,
            kernels_cuda::quant::KERNELS,
            kernels_cuda::rope::KERNELS,
            kernels_cuda::sample::KERNELS,
            kernels_cuda::ssm::KERNELS,
        ],
        "c",
    );

    let execute = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../driver-cuda/csrc/src/model/declared/execute.hpp");
    let included = std::fs::read_to_string(&execute)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", execute.display()));

    // Every header under `kernels-cuda/csrc/src`, by the text it holds.
    let mut headers: Vec<(String, String)> = Vec::new();
    let mut stack = vec![csrc()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "hpp") {
                let rel = p
                    .strip_prefix(csrc())
                    .expect("under csrc")
                    .to_string_lossy()
                    .replace('\\', "/");
                if let Ok(text) = std::fs::read_to_string(&p) {
                    headers.push((rel, text));
                }
            }
        }
    }

    let mut missing: Vec<String> = Vec::new();
    for line in dispatch.lines() {
        let Some(at) = line.find("::pie_cuda_driver::kernels::") else { continue };
        let call = &line[at..];
        let Some(open) = call.find('(') else { continue };
        let path = &call[..open];
        let Some(name) = path.rsplit("::").next() else { continue };
        // A header DECLARES it if the name appears as a call-shaped
        // token; a header is IN SCOPE if `execute.hpp` includes it.
        let declared_in: Vec<&str> = headers
            .iter()
            .filter(|(_, text)| text.contains(&format!("{name}(")))
            .map(|(rel, _)| rel.as_str())
            .collect();
        if declared_in.is_empty() {
            // No header declares it at all — a different failure, and
            // the shim compile is what catches that one.
            continue;
        }
        if !declared_in
            .iter()
            .any(|rel| included.contains(&format!("#include \"{rel}\"")))
        {
            missing.push(format!("  {name} — declared in {declared_in:?}"));
        }
    }
    missing.sort();
    missing.dedup();
    assert!(
        missing.is_empty(),
        "the generated dispatch calls launchers `execute.hpp` has no \
         declaration for; add the header(s):\n{}",
        missing.join("\n")
    );
}
