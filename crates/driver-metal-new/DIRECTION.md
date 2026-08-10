# Direction: the model-compiler path, and what it retires

Decided 2026-08-10. **Read this before picking work out of `PARITY-BATCH.md`
or `CUTOVER.md`** — both were written against the older plan and describe work
that is no longer worth doing.

## The north star

`crates/model-compiler/DSL-DESIGN.md` states it in one line:

> **Nothing in the driver may choose a kernel.** A statement names the symbol
> it runs; the driver resolves weight names to pointers, resolves value ids to
> addresses, and calls. That is the whole of its job.

A traced fire is lowered by `model_compiler::lower` into a flat list of
`Launch` rectangles, each naming a kernel symbol and carrying its operands as
`Arg`s. The executor binds and dispatches. There is no per-family forward.

**Metal is going all in on this**, alongside CUDA.

## This is a seam that already exists, not a new architecture

`model-compiler` already depends on **both** kernel tables and already has the
backend it needs:

```
crates/model-compiler/Cargo.toml
    kernels       = { path = "../kernels" }
    kernels-cuda  = { path = "../kernels-cuda",  default-features = false }
    kernels-metal = { path = "../kernels-metal", default-features = false }

crates/model-compiler/src/kernels.rs
    pub enum Backend { Cuda, Metal }
    Backend::Metal => KERNELS_METAL
```

## Three legs, and only one of them is done

Going all in needs three things. They are independent and only the first is
finished.

### 1. The lowering — **done**

`Lowered` is backend-neutral by construction: `launches`, `kernels:
Vec<String>` (symbols, not function pointers), `arena_bytes`, `value_offset`.
Nothing in it is CUDA-shaped, and `Backend::Metal` resolves to `KERNELS_METAL`
today.

### 2. The Metal DSL text — **started, for one family**

`model-compiler` compiles a DSL, and **a text has to be written for Metal**:
`dsl::trace_metal(family, ..)` records `<family>.metal.<class>`, and the
symbols the body names must be Metal symbols. A CUDA text does not serve.

What exists: `crates/model/src/families/llama_like/forward/mod.rs`'s
`llama_like_metal_text`. Its own doc states the gaps, and they are the work:

* the **M>1 lane is a guess** — the driver's `MultiBatchPsos` carries split-k,
  fp16-precast, strided and bias variants behind a `kQmmMinBatch` gate; the
  text states one GEMM and one paged attention;
* `sdpa_*_d_256` **pins head_dim 256**, where the driver compiles other widths
  (`d_512` for gemma4);
* **no seams** — the adapter, the two observation taps and the boundaries the
  CUDA text states are absent, "because none of the machinery behind them
  exists on this backend yet";
* qk-norm and bias are stated as ordinary norms and are **untested** against
  what `declared_dag.hpp` expects.

What does not exist: a Metal text for any other family. `crates/model/src/
families/` holds only `llama_like`, while the Metal driver carries handwritten
forwards for llama, gemma4, gpt-oss and qwen. **Every one of those needs a text
before its handwritten forward can go.**

Related: `kernels-metal` has **98** `kernel!` rows against `kernels-cuda`'s
**226**. A symbol a text wants to name needs a row, because the row is where
the contract lives.

(`trace_metal`'s doc comment still says "nothing calls it, and the empty Metal
kernel table". Both were true when written and neither is now — one caller, 98
rows. Same staleness this crate's ledgers keep showing.)

### 3. The Metal executor — **not started**

The consumer. `driver-cuda-new/src/model/executor.rs` is the template — its own
doc calls it *"the family-independent replacement the flat list was designed
for: three resolution rules, stated once"*. `driver-metal-new` does not depend
on `model-compiler` at all today.

### Metal is not behind on kernel resolution — it is ahead

Worth stating because it is easy to assume otherwise. Both kernel crates are
the same shape: a `KERNELS` table of `KernelSig` rows built on the shared
`kernels` crate. The difference is how a symbol is reached:

* **Metal** resolves by **name string at runtime** — `Compiler::compile(context,
  source, function: &str)` builds a pipeline state from an entry-point name.
  A symbol the lowering states can be reached without the driver having been
  written to know it exists.
* **CUDA** reaches `pie_k_*` C symbols through a dispatch arm per kernel, which
  `executor.rs` says "grows kernel by kernel beside the bridge".

So the mechanism the north star needs is already in place on Metal. What is not
in place is that **the plans deciding which symbols to use are written per
family, by hand** — `psos_llama.rs`, `psos_gemma4.rs`, `psos_gptoss.rs`,
`psos_mb.rs`. Those are the driver choosing kernels, and they are what the
lowering replaces.

## What this retires

Roughly 8.5k lines across 21 files, plus the qwen path embedded in the shared
modules:

| retired | what it is |
|---|---|
| `batch/dispatch_{llama,gemma4,gptoss}.rs`, `dispatch_mb.rs` | per-family DAG builders — the handwritten forward |
| `batch/psos_{llama,gemma4,gptoss,mb}.rs`, `psos.rs` | per-family PSO plans — the driver choosing kernels |
| `batch/{llama,gemma4,gptoss}.rs`, `*_consts.rs`, `gptoss_solve.rs` | per-family geometry and constant walks |
| `metal/{llama,gemma4,gptoss}_{bind,step,engine}.rs`, `step.rs`, `step_mb.rs`, `bind.rs`, `bind_mb.rs` | per-family binds, steps and engines |
| **`forward.cpp` / `forward.hpp` (5393)** | **do not port it.** It is the family executor. It is replaced, not translated |

`PARITY-BATCH.md`'s remaining rows are almost entirely this executor and its
dependents. Those rows are now *obsolete rather than outstanding*, and the
ledger should be read with that in mind until it is rewritten.

## What survives, and it is most of the crate

Everything that is not a family:

| survives | why |
|---|---|
| all of `src/metal/` except the family files — context, device, heaps, pools, elastic, keepalive, encoder/stepper, pipeline compiler, archives, tables, timestamps, timing, residency, handle, ring, fire, fused, grouped, storage, paging | the substrate any executor needs. The lowering names symbols; this is what runs them |
| all of `src/pipeline/` | the PTIR channel-plane interpreter. A **different layer** from the model forward — prologue/epilogue shell stages, channels, readiness, the fire's plan. It is already model-agnostic and is not affected |
| `src/loader/`, `src/store/` | weights, KV pages, recurrent slots |
| `batch/` minus the family files — `schedule`, `mask`, `admit`, `member`, `marshal`, `sequence`, `paged_state`, `tickets`, `color`, `sizing`, `heap_budget`, `fit`, `logits`, `golden`, `timing`, `paging`, `fire_csr`, `abi` | the frame and fleet layer: who is in this fire, which pages, which slots, what fits. The lowering does not answer any of these |
| `src/facts.rs`, `shader.rs`, `tuning.rs`, `region.rs`, `bump.rs` | host-portable substrate |

The work of the last two days is in the surviving column. It was ported for the
C++'s reasons and it holds for the new ones, because none of it chooses a
kernel.

## The next step

The three legs can go in parallel, and the order that de-risks fastest is:

1. **The executor**, against `driver-cuda-new/src/model/executor.rs`. It is
   the smallest of the three and it makes the other two testable end to end.
   Metal's dispatch half should be **shorter than CUDA's**, because a symbol is
   a name here: where CUDA grows an arm per kernel, Metal can compile the entry
   point the lowering named and bind operands in the row's stated order. If
   that holds it is the argument for the whole approach, and it is worth
   proving before the texts are written against it.
2. **Close `llama_like`'s gaps**, in the order its doc lists them. The M>1 lane
   is the one that decides whether the text can replace the driver's
   `MultiBatchPsos` or only its decode step.
3. **A text per remaining family** — gemma4, gpt-oss, qwen — each retiring its
   handwritten forward as it lands, with the device smokes already in
   `tests/device_smoke.rs` as the equality check. Those smokes are
   token-exact against mlx_lm today, so a text is right when it reproduces
   them.

Add `model-compiler` as a dependency when step 1 starts.
