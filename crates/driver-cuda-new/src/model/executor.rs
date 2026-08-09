//! The executor's first half: binding a flat launch's operands
//! (retirement plan phase C).
//!
//! `model_compiler::lower` turns a traced fire into rectangles whose
//! operands are [`Arg`]s — an arena offset, a backend-named value, or a
//! weight name. The C++ declared executor binds those against `ws.*`
//! fields per family; THIS binder is the family-independent replacement
//! the flat list was designed for: three resolution rules, stated once.
//!
//! What binding is NOT: dispatch. A bound launch still has to reach its
//! `pie_k_*` entry with the operands in the row's own order, and that
//! per-kernel arm is the executor's other half — it grows kernel by
//! kernel beside the bridge. Splitting the two means the binder is pure
//! host logic, provable against a real lowered trace with no GPU and no
//! bridge in the build.

use std::ffi::c_void;

use model_compiler::lower::{Arg, Buffers, Launch, Lowered};
use model_compiler::trace::ValueId;

/// The frame's activation arena: one device block of
/// [`Lowered::arena_bytes`], allocated per fire (or reused across them —
/// the binder only ADDRESSES it).
#[derive(Debug, Clone, Copy)]
pub struct Frame {
    /// Device base of the arena.
    pub arena: *mut c_void,
    /// Its extent — [`Lowered::arena_bytes`] at allocation time. Offsets
    /// are checked against it, because an arena reused across fires can
    /// be SMALLER than the new fire needs, and a launch that addressed
    /// past it would corrupt whatever the allocator placed next.
    pub arena_bytes: usize,
}

/// Resolves the names the trace states against the driver's stores.
///
/// The one thing that stays per-family is a MAP rather than a switch —
/// `lower.rs`'s own words — and this is that map's seam. The live
/// implementation reads the loaded model's tensor store and the fire's
/// seam values; tests answer with sentinels.
pub trait Resolver {
    /// The device pointer for a weight the trace names
    /// (`layer.3.q_proj`), or `None` — which is DRIFT, not absence: a
    /// trace that names a weight the store lacks was traced against a
    /// different binding.
    fn weight(&mut self, name: &str) -> Option<*const c_void>;
    /// The device pointer for a backend-named value (the observed query,
    /// the logits — `Buffers::NAMED`).
    fn named(&mut self, value: ValueId) -> Option<*mut c_void>;
}

/// One resolved operand: where it is, and how wide one row is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundArg {
    /// The device address.
    pub ptr: *mut c_void,
    /// Elements per row, for the args that carry one ([`Arg::Arena`],
    /// [`Arg::Named`]); zero for a weight, whose extent is the tensor's.
    pub width: u32,
}

/// A launch with every operand resolved — what a dispatch arm consumes.
#[derive(Debug)]
pub struct BoundLaunch<'a> {
    /// The kernel's symbol, resolved through [`Lowered::kernels`].
    pub kernel: &'a str,
    /// The rectangle, in the op's own row space.
    pub rows: std::ops::Range<u32>,
    /// The layer range.
    pub layers: std::ops::Range<u16>,
    /// Operands in the trace's stated order: inputs, outputs, weights.
    pub args: Vec<BoundArg>,
}

/// Why a launch refused to bind. Every variant is a DRIFT diagnosis, not
/// a runtime condition — the C++ executor's `throw_drift` shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindRefusal {
    /// An arena operand addresses past the frame's arena.
    ArenaOutOfBounds {
        /// The offending offset.
        at: usize,
        /// What the frame actually holds.
        arena_bytes: usize,
    },
    /// The trace names a weight the resolver does not hold.
    UnknownWeight(String),
    /// The trace names a seam value the resolver does not bind.
    UnknownNamed(ValueId),
}

/// What one launch needs beyond its bound args: the op join.
///
/// `Launch::args` carries VALUES; the op the launch lowers carries the
/// rest — the weight its statement names and the accumulate flag — which
/// is exactly the `plan.weight_name(op)` read the C++ executor does. The
/// join is computed once per lowering, so the arms read a slot instead of
/// re-matching `OpKind` per fire.
#[derive(Debug, Clone, Default)]
pub struct LaunchSpec {
    /// The weight the op names, when it names one. Concrete — the trace
    /// is layer-unrolled, so this is `layer.3.q_proj`, never a template.
    pub weight: Option<String>,
    /// `Matmul::beta_one`: the residual fold. The launch then carries the
    /// accumulate target as its LAST arg (inputs, then outputs — the
    /// output aliases the residual input's bytes).
    pub beta_one: bool,
    /// The op's OUTPUT placements, resolved through `value_offset` — the
    /// values a launch writes that its args do not carry (the fused qkv's
    /// observed-query pin, the attention output the o_proj reads).
    pub outs: Vec<Arg>,
    /// The SECOND weight an op names, when it names two (`GdnPrep`'s
    /// `dt_bias` beside its `a_log`).
    pub weight2: Option<String>,
    /// The per-request store this op addresses (`OpKind::state_ref`) —
    /// how a GDN arm learns its state layer, the C++ executor's
    /// `op.param1` read.
    pub state: Option<model_compiler::trace::StateRef>,
    /// `RmsnormPerHead`'s head width: the launch's rows are token rows,
    /// the kernel's rows are `tokens * (width / head_dim)` of `head_dim`.
    pub per_head_dim: Option<u32>,
    /// `Rope`'s partial-rotary channel count, when the op states one.
    pub rope_partial: Option<u32>,
}

/// The per-launch op join over a whole lowering.
#[derive(Debug, Clone)]
pub struct DispatchPlan {
    specs: Vec<LaunchSpec>,
}

impl DispatchPlan {
    /// Join `lowered`'s launches with the ops that produced them.
    #[must_use]
    pub fn new(plan: &model_compiler::trace::ForwardPlan, lowered: &Lowered) -> Self {
        use model_compiler::trace::OpKind;
        use model_compiler::trace::Dim;
        let width_of = |v: ValueId| -> u32 {
            plan.values[v as usize]
                .shape
                .0
                .iter()
                .filter_map(|d| match d {
                    Dim::Const(w) => Some(*w),
                    _ => None,
                })
                .product::<u32>()
                .max(1)
        };
        let out_arg = |v: ValueId| -> Arg {
            match lowered.value_offset.get(v as usize) {
                Some(&at) if at != Buffers::NAMED => Arg::Arena { at, width: width_of(v) },
                _ => Arg::Named { value: v, width: width_of(v) },
            }
        };
        // A value-producing GUARD's outputs belong to every launch of its
        // regions (the region's launches "bind the same output buffer and
        // record no SSA outputs of their own" — the recurrence three-way).
        // Map each region op back to its owning guard, once.
        let mut guard_of: Vec<Option<usize>> = vec![None; plan.ops.len()];
        for (g, op) in plan.ops.iter().enumerate() {
            if let OpKind::Guard { arms, else_ops } = &op.kind {
                let span = arms.iter().map(|a| a.ops as usize).sum::<usize>()
                    + *else_ops as usize;
                for slot in guard_of.iter_mut().skip(g + 1).take(span) {
                    *slot = Some(g);
                }
            }
        }
        let specs = lowered
            .launches
            .iter()
            .map(|launch| {
                let op = &plan.ops[launch.op as usize];
                let out_values: &[ValueId] = if op.outputs.is_empty() {
                    guard_of[launch.op as usize]
                        .map_or(&[], |g| plan.ops[g].outputs.as_slice())
                } else {
                    &op.outputs
                };
                let outs: Vec<Arg> = out_values.iter().map(|&v| out_arg(v)).collect();
                let mut spec = match &op.kind {
                    OpKind::Embed { weight }
                    | OpKind::Rmsnorm { weight, .. }
                    | OpKind::RmsnormPerHead { weight, .. }
                    | OpKind::AddBias { weight }
                    | OpKind::RmsnormGated { weight }
                    | OpKind::CausalConv1d { weight, .. }
                    | OpKind::LmHead { weight } => LaunchSpec {
                        weight: Some(weight.clone()),
                        ..LaunchSpec::default()
                    },
                    OpKind::Matmul { weight, beta_one, .. } => LaunchSpec {
                        weight: Some(weight.clone()),
                        beta_one: *beta_one,
                        ..LaunchSpec::default()
                    },
                    OpKind::GdnPrep { a_log, dt_bias } => LaunchSpec {
                        weight: Some(a_log.clone()),
                        weight2: Some(dt_bias.clone()),
                        ..LaunchSpec::default()
                    },
                    // A lowered `Launch` states its weights as
                    // `Arg::Weight`s; the FIRST also rides the spec so
                    // constant-naming arms (`scale.*`) can read the name
                    // the bound pointer lost.
                    OpKind::Launch { weights, .. } if !weights.is_empty() => LaunchSpec {
                        weight: Some(weights[0].clone()),
                        weight2: weights.get(1).cloned(),
                        ..LaunchSpec::default()
                    },
                    _ => LaunchSpec::default(),
                };
                spec.outs = outs;
                spec.state = op.kind.state_ref();
                if let OpKind::RmsnormPerHead { head_dim, .. } = op.kind {
                    spec.per_head_dim = Some(head_dim);
                }
                if let OpKind::Rope { partial, .. } = op.kind {
                    spec.rope_partial = partial;
                }
                spec
            })
            .collect();
        Self { specs }
    }

    /// The spec for launch `i` — index-parallel with
    /// [`Lowered::launches`].
    #[must_use]
    pub fn spec(&self, i: usize) -> &LaunchSpec {
        &self.specs[i]
    }
}

/// FlashInfer's decode plan cache, owned across the bridge.
///
/// The C++ type is INCOMPLETE on purpose (`struct DecodePlanCache;`), so
/// this is a handle, never a layout — created by the hand-written extras
/// (`pie_x_make_decode_plan`, the factory's `release()`), destroyed by the
/// factory's own deleter. Plain [`Drop`] rather than the crate's explicit
/// `release(&mut ops)` pattern, deliberately: destruction is a pure host
/// `delete` with no CUDA ordering and no recorder seam — there is no
/// oracle that needs to see it.
#[cfg(feature = "bridge")]
#[derive(Debug)]
pub struct DecodePlan {
    cache: *mut c_void,
}

#[cfg(feature = "bridge")]
impl DecodePlan {
    /// A fresh, unplanned cache.
    #[must_use]
    pub fn new() -> Self {
        let cache = unsafe { crate::launch::ffi::pie_x_make_decode_plan() };
        assert!(!cache.is_null(), "make_decode_plan returned null");
        Self { cache }
    }

    /// The raw handle a dispatch arm passes as the `DecodePlanCache&`.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.cache
    }

    /// Where the plan's int arrays sit inside the workspace's
    /// `int_buffer`.
    pub fn set_int_base(&mut self, bytes: usize) {
        unsafe { crate::launch::ffi::pie_x_set_decode_plan_int_base(self.cache, bytes) };
    }

    /// Run FlashInfer's decode planner over the fire's HOST page indptr.
    ///
    /// The caller brackets this with the workspace's
    /// `begin_plan_update`/`end_plan_update`, exactly as the C++ does —
    /// the planner stages into the view's pinned slot.
    // Safe by design like the seam methods: the view's pointers are the
    // workspace's own, and the stream is the caller's live handle.
    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    pub fn plan_decode(
        &mut self,
        kv_page_indptr_h: &[u32],
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        workspace: crate::launch::AttentionWorkspaceView,
        stream: *mut c_void,
        enable_cuda_graph: bool,
        window_left: i32,
    ) {
        let num_requests =
            i32::try_from(kv_page_indptr_h.len() - 1).expect("request count fits i32");
        unsafe {
            crate::launch::ffi::pie_x_plan_attention_flashinfer_decode_bf16(
                self.cache,
                kv_page_indptr_h.as_ptr(),
                num_requests,
                num_q_heads,
                num_kv_heads,
                head_dim,
                page_size,
                workspace,
                stream,
                enable_cuda_graph,
                false,
                false,
                window_left,
            );
        }
    }
}

#[cfg(feature = "bridge")]
impl Default for DecodePlan {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "bridge")]
impl Drop for DecodePlan {
    fn drop(&mut self) {
        unsafe { crate::launch::ffi::pie_x_destroy_decode_plan(self.cache) };
    }
}

/// FlashInfer's prefill plan cache — [`DecodePlan`]'s twin, owned the same
/// way for the same reasons.
#[cfg(feature = "bridge")]
#[derive(Debug)]
pub struct PrefillPlan {
    cache: *mut c_void,
}

#[cfg(feature = "bridge")]
impl PrefillPlan {
    /// A fresh, unplanned cache.
    #[must_use]
    pub fn new() -> Self {
        let cache = unsafe { crate::launch::ffi::pie_x_make_prefill_plan() };
        assert!(!cache.is_null(), "make_prefill_plan returned null");
        Self { cache }
    }

    /// The raw handle a dispatch arm passes.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.cache
    }

    /// Run FlashInfer's prefill planner over the fire's HOST CSRs.
    ///
    /// Bracket with the workspace's plan-update fence, as with
    /// [`DecodePlan::plan_decode`].
    // Safe by design like the seam methods: the view's pointers are the
    // workspace's own, and the stream is the caller's live handle.
    #[allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
    pub fn plan_prefill(
        &mut self,
        qo_indptr_h: &[u32],
        kv_page_indptr_h: &[u32],
        kv_last_page_lens_h: &[u32],
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        workspace: crate::launch::AttentionWorkspaceView,
        stream: *mut c_void,
        enable_cuda_graph: bool,
        window_left: i32,
    ) {
        let num_requests =
            i32::try_from(qo_indptr_h.len() - 1).expect("request count fits i32");
        let total_tokens =
            i32::try_from(*qo_indptr_h.last().expect("a CSR has a last entry"))
                .expect("token count fits i32");
        unsafe {
            crate::launch::ffi::pie_x_plan_attention_flashinfer_prefill_bf16(
                self.cache,
                qo_indptr_h.as_ptr(),
                kv_page_indptr_h.as_ptr(),
                kv_last_page_lens_h.as_ptr(),
                total_tokens,
                num_requests,
                num_q_heads,
                num_kv_heads,
                head_dim,
                page_size,
                workspace,
                stream,
                enable_cuda_graph,
                window_left,
                false,
                false,
                true,
                false,
                false,
            );
        }
    }
}

#[cfg(feature = "bridge")]
impl Default for PrefillPlan {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "bridge")]
impl Drop for PrefillPlan {
    fn drop(&mut self) {
        unsafe { crate::launch::ffi::pie_x_destroy_prefill_plan(self.cache) };
    }
}

/// The scalar facts a dispatch arm reads beside its bound operands.
///
/// Everything else an arm needs is IN the launch: row counts from
/// `rows`, per-operand widths from the args. What remains is the
/// deployment's constants — the same values the C++ arms read off their
/// facts structs — and the per-fire handles.
#[cfg(feature = "bridge")]
#[derive(Debug, Clone)]
pub struct DispatchCtx {
    /// The fire's stream.
    pub stream: *mut c_void,
    /// The cuBLAS handle `gemm::act_x_w` routes through.
    pub cublas: *mut c_void,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
    /// Rope theta, for the table fill.
    pub rope_theta: f32,
    /// Head width, for the table fill.
    pub head_dim: i32,
    /// Vocabulary rows the embed weight holds.
    pub vocab: i32,
    /// The packed gate‖up order `chunked_swiglu` was bound with.
    pub gate_second: bool,
    /// GPT-J adjacent-pair rotation (`rope_interleave`), vs NeoX half/half.
    pub rope_interleaved: bool,
    /// The fire's token ids (device i32, one per row) — the embed's
    /// input, which is the backend's to provide rather than an arg.
    pub token_ids: *mut c_void,
    /// The fire's positions (device i32, one per row) — the rope table's
    /// input, provided the same way.
    pub positions: *mut c_void,
    /// gemma's FINAL logit softcap (`cap * tanh(x / cap)` over the
    /// logits) — the value behind the `attn::logit_softcap_bf16` launch,
    /// which the trace states only when the deployment configures it.
    pub final_logit_softcap: f32,
    /// gemma-4's per-layer embedding width (`ple_dim`) — what the PLE
    /// relay transpose divides its flat `[N, layers*dim]` row by. Zero
    /// on families without a PLE.
    pub ple_dim: i32,
    /// The scalar constants `norm::scalar_mul_bf16` launches name in
    /// their `scale.<name>` weight slot — `sqrt(hidden)` on the
    /// embedding, gemma's query pre-scale. Resolved here by NAME because
    /// a scale is a constant, not a tensor (the dsl's own words).
    pub scales: std::collections::BTreeMap<String, f32>,
}

/// The fire's attention context: what the attention arms need beyond
/// args and the op join — the planned cache, the workspace, the per-layer
/// KV views, and the fire's device-resident page CSRs and write
/// descriptors. The ENGINE'S half of a fire, assembled once.
#[cfg(feature = "bridge")]
#[derive(Debug, Clone)]
pub struct AttnCtx {
    /// The planned [`DecodePlan`]'s handle. Null on a pure-prefill fire.
    pub decode_plan: *mut c_void,
    /// The planned [`PrefillPlan`]'s handle. Null on a pure-decode fire.
    pub prefill_plan: *mut c_void,
    /// The workspace, as launchers take it.
    pub workspace: crate::launch::AttentionWorkspaceView,
    /// One KV view per layer, indexed by the launch's layer.
    pub layers: Vec<crate::launch::KvCacheLayerView>,
    /// Device page-index CSR.
    pub kv_page_indices_d: *const u32,
    /// Device page indptr.
    pub kv_page_indptr_d: *const u32,
    /// Device last-page lengths.
    pub kv_last_page_lens_d: *const u32,
    /// Device query indptr — prefill's token-rows-per-request CSR.
    pub qo_indptr_d: *const u32,
    /// HOST qo indptr — the planless prefill dispatch plans internally
    /// per fire and reads the CSR from the host. Null when no planless
    /// launch is stated.
    pub qo_indptr_h: *const u32,
    /// HOST kv page indptr, the planless dispatch's other host read.
    pub kv_page_indptr_h: *const u32,
    /// Requests in the fire (`indptr.len() - 1`).
    pub num_requests: i32,
    /// Pages the fire's CSR names — what the dequant staging walks.
    pub num_pages_in_batch: i32,
    /// `write_kv_to_pages`'s first-token scalar (the fire's write origin).
    pub first_token: i32,
    /// Per-row target page for this fire's KV append.
    pub w_page_d: *const u32,
    /// Per-row offset-in-page for the append.
    pub w_off_d: *const u32,
    /// Per-row validity for the append.
    pub row_valid_d: *const u8,
    /// The observed-query pin the fused qkv writes and the dispatch
    /// reads. A GUARD-owned value (the region's launches record no SSA
    /// outputs of their own), so it is fire context until the join learns
    /// to walk back to the guard op.
    pub q_out: *mut c_void,
    /// The attention output slot the o_proj reads — guard-owned like
    /// `q_out`, and one arena slot reused by every layer (liveness).
    pub o_out: *mut c_void,
    /// LSE scratch the decode dispatch writes.
    pub lse_out_d: *mut f32,
    /// Sliding-window extent, `-1` for none.
    pub window_left: i32,
    /// PER-LAYER window extents for alternating-window families
    /// (gemma's global/local schedule); empty means uniform
    /// [`Self::window_left`].
    pub window_left_by_layer: Vec<i32>,
    /// Logit soft cap, `0` for none.
    pub logits_soft_cap: f32,
    /// The attention scale (`1/sqrt(head_dim)` unless overridden).
    pub sm_scale: f32,
}

/// The fire's GDN context: what the linear-attention arms need beyond
/// args and the op join — the per-layer conv/recurrent state slabs, the
/// request→slot indirection, and the deployment's head geometry. The
/// C++ executor reads these off `RecurrentStateCache` + facts per launch;
/// here they are assembled once per fire, [`AttnCtx`]-style.
#[cfg(feature = "bridge")]
#[derive(Debug, Clone)]
pub struct GdnCtx {
    /// Key heads (compact, pre-GQA-repeat).
    pub k_h: i32,
    /// Value heads.
    pub v_h: i32,
    /// Key head width.
    pub k_d: i32,
    /// Value head width.
    pub v_d: i32,
    /// Conv channels (`2*K_h*K_d + V_h*V_d`).
    pub conv_dim: i32,
    /// Conv window width (`linear_conv_kernel_dim`).
    pub conv_k: i32,
    /// Device base of each MODEL layer's conv-state slab (slot 0); zero
    /// for layers with no linear-attention state.
    pub conv_state: Vec<u64>,
    /// Elements per conv slot (`conv_k * conv_dim`).
    pub conv_stride_elems: i64,
    /// Device base of each MODEL layer's recurrent-state slab (slot 0),
    /// in the store's own dtype (fp32 or bf16 — the deployment's
    /// `state_bf16` fact); zero for non-linear layers.
    pub recurrent_state: Vec<u64>,
    /// Elements per recurrent slot.
    pub state_stride_elems: i64,
    /// Device request→slot ids, one per request in the fire.
    pub slot_ids_d: *const i32,
    /// Whether this fire advances state (true for Decode/Prefill; the
    /// frozen-verify service classes pass false).
    pub write_state: bool,
}

/// Why a bound launch could not be dispatched.
#[cfg(feature = "bridge")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchRefusal {
    /// No arm exists for this kernel yet. The executor grows kernel by
    /// kernel, and an explicit refusal is what keeps a missing arm from
    /// reading as a covered launch.
    NoArm(String),
    /// The op join names a weight the resolver does not hold — the same
    /// drift [`BindRefusal::UnknownWeight`] diagnoses for stated args.
    UnknownWeight(String),
    /// The arm expected the op join to name a weight and it named none —
    /// the arm and the lowering disagree about the statement's shape.
    NoWeight(String),
    /// An attention arm ran without an [`AttnCtx`], or with one whose
    /// layer list does not cover the launch's layer.
    NoAttnCtx(String),
    /// A GDN arm ran without a [`GdnCtx`], or with one whose state
    /// vectors do not cover the launch's state layer.
    NoGdnCtx(String),
    /// An output placement failed to resolve — the join and the resolver
    /// disagree.
    Out(String),
    /// The arm and the lowering disagree about the operand count — a
    /// drift between the trace's statement and this arm's reading of it.
    ArgCount {
        /// The kernel whose arm refused.
        kernel: String,
        /// Operands the arm expects.
        expected: usize,
        /// Operands the launch bound.
        got: usize,
    },
}

/// Dispatch one bound launch through its `pie_k_*` entry.
///
/// The arms cover the anchor deployment's compute backbone — embed, the
/// rope table, rmsnorm, the quantized-dispatch GEMM, chunked swiglu.
/// Operand order inside each arm is the trace's stated order (inputs,
/// then outputs, then weights), which the numeric smoke verifies — a
/// swapped operand is wrong VALUES, not a type error, and only a check
/// against host math catches it.
///
/// # Errors
///
/// See [`DispatchRefusal`].
#[cfg(feature = "bridge")]
#[allow(clippy::too_many_lines)]
pub fn dispatch<R: Resolver>(
    bound: &BoundLaunch<'_>,
    spec: &LaunchSpec,
    frame: Frame,
    resolver: &mut R,
    ctx: &DispatchCtx,
    attn: Option<&AttnCtx>,
    gdn: Option<&GdnCtx>,
) -> Result<(), DispatchRefusal> {
    use crate::launch::ffi;

    // The GDN arms' shared reads: the ctx itself, and the launch's state
    // layer's slab out of one of its per-layer vectors.
    let gdn_ctx = || -> Result<&GdnCtx, DispatchRefusal> {
        gdn.ok_or_else(|| DispatchRefusal::NoGdnCtx(bound.kernel.to_string()))
    };
    let state_layer = || -> Result<usize, DispatchRefusal> {
        spec.state
            .map(|s| s.layer as usize)
            .ok_or_else(|| DispatchRefusal::NoGdnCtx(format!("{}: op states no layer", bound.kernel)))
    };
    let slab = |v: &[u64], layer: usize, what: &str| -> Result<*mut c_void, DispatchRefusal> {
        match v.get(layer) {
            Some(&base) if base != 0 => Ok(base as *mut c_void),
            _ => Err(DispatchRefusal::NoGdnCtx(format!(
                "{}: layer {layer} has no {what} slab",
                bound.kernel
            ))),
        }
    };

    let rows = i32::try_from(bound.rows.end - bound.rows.start).expect("row count fits i32");
    // The op join's output placements: what a guard-region launch binds
    // for the value the GUARD owns (the recurrence three-way's core out).
    let out_slot = |i: usize, resolver: &mut R| -> Result<BoundArg, DispatchRefusal> {
        let arg = spec
            .outs
            .get(i)
            .ok_or_else(|| DispatchRefusal::Out(format!("{}: no output {i}", bound.kernel)))?;
        resolve_arg(arg, frame, resolver)
            .map_err(|e| DispatchRefusal::Out(format!("{}: {e:?}", bound.kernel)))
    };
    let need = |n: usize| -> Result<(), DispatchRefusal> {
        if bound.args.len() == n {
            Ok(())
        } else {
            Err(DispatchRefusal::ArgCount {
                kernel: bound.kernel.to_string(),
                expected: n,
                got: bound.args.len(),
            })
        }
    };
    let weight = |resolver: &mut R| -> Result<*const c_void, DispatchRefusal> {
        let name = spec
            .weight
            .as_deref()
            .ok_or_else(|| DispatchRefusal::NoWeight(bound.kernel.to_string()))?;
        resolver
            .weight(name)
            .ok_or_else(|| DispatchRefusal::UnknownWeight(name.to_string()))
    };

    match bound.kernel {
        // args: [y]. The token ids are the fire's input and the weight is
        // the op's — both context, neither an arg.
        "layout::embed_bf16" => {
            need(1)?;
            let y = bound.args[0];
            let w = weight(resolver)?;
            unsafe {
                ffi::pie_k_layout_embed_bf16(
                    ctx.token_ids.cast_const().cast(),
                    w,
                    y.ptr,
                    rows,
                    i32::try_from(y.width).expect("hidden fits i32"),
                    ctx.vocab,
                    ctx.stream,
                );
            }
        }
        // args: [table]; positions are the fire's.
        "rope::rope_standard_table" => {
            need(1)?;
            let table = bound.args[0];
            unsafe {
                ffi::pie_k_rope_rope_standard_table(
                    ctx.positions.cast_const().cast(),
                    table.ptr.cast(),
                    rows,
                    ctx.head_dim,
                    ctx.rope_theta,
                    ctx.stream,
                );
            }
        }
        // args: [x, y]; the norm weight is the op's.
        "norm::rmsnorm_bf16" => {
            need(2)?;
            let (x, y) = (bound.args[0], bound.args[1]);
            let w = weight(resolver)?;
            unsafe {
                ffi::pie_k_norm_rmsnorm_bf16(
                    x.ptr,
                    w,
                    y.ptr,
                    rows,
                    i32::try_from(x.width).expect("hidden fits i32"),
                    ctx.rms_eps,
                    ctx.stream,
                );
            }
        }
        // args: [act, y] with beta 0, or [act, resid_in, y] with beta 1 —
        // the residual fold, where the output aliases the residual's
        // bytes and cuBLAS accumulates in place. M/K/N come from the
        // rectangle and the widths; the weight is the op's; the view is
        // raw bf16 until quantized deployments join.
        "gemm::act_x_w" => {
            let (act, y, beta) = if spec.beta_one {
                need(3)?;
                (bound.args[0], bound.args[2], 1.0f32)
            } else {
                need(2)?;
                (bound.args[0], bound.args[1], 0.0f32)
            };
            let w = weight(resolver)?;
            let view = super::weight_view::WeightView::raw(w, crate::dtype::DType::Bf16);
            unsafe {
                ffi::pie_k_gemm_act_x_w(
                    ctx.cublas,
                    act.ptr,
                    view,
                    y.ptr,
                    rows,
                    i32::try_from(y.width).expect("N fits i32"),
                    i32::try_from(act.width).expect("K fits i32"),
                    beta,
                    crate::dtype::DType::Bf16,
                    crate::dtype::DType::Bf16,
                );
            }
        }
        // args: [packed, y].
        "mlp::chunked_swiglu_bf16" => {
            need(2)?;
            let (packed, y) = (bound.args[0], bound.args[1]);
            unsafe {
                ffi::pie_k_mlp_chunked_swiglu_bf16(
                    packed.ptr,
                    y.ptr,
                    rows,
                    i32::try_from(y.width).expect("I fits i32"),
                    ctx.stream,
                    ctx.gate_second,
                );
            }
        }
        // args: [packed, rope_table, q_norm_w, k_norm_w]; the q output is
        // the observed-query PIN (outs[0], Named); the KV pages, CSRs and
        // write descriptors are the fire's ([`AttnCtx`]).
        "attn::qkv_decode_qk_norm_rope_write_kv_bf16" => {
            need(4)?;
            let a = attn
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = a
                .layers
                .get(bound.layers.start as usize)
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let (packed, table, qw, kw) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            unsafe {
                ffi::pie_k_attn_qkv_decode_qk_norm_rope_write_kv_bf16(
                    packed.ptr,
                    a.q_out,
                    layer.k_pages,
                    layer.v_pages,
                    qw.ptr,
                    kw.ptr,
                    ctx.positions.cast_const().cast(),
                    table.ptr.cast_const().cast(),
                    a.kv_page_indices_d,
                    a.kv_page_indptr_d,
                    a.kv_last_page_lens_d,
                    a.w_page_d,
                    a.w_off_d,
                    a.row_valid_d,
                    rows,
                    (i32::try_from(packed.width).expect("packed width")
                        - 2 * layer.num_kv_heads * layer.head_dim)
                        / layer.head_dim.max(1),
                    layer.num_kv_heads,
                    layer.head_dim,
                    layer.page_size,
                    layer.hnd_layout,
                    ctx.rope_theta,
                    ctx.rms_eps,
                    ctx.stream,
                );
            }
        }
        // args: [q (the pin)]; o is the op's arena output; the plan, the
        // workspace and the layer view are the fire's.
        "attn::dispatch_attention_flashinfer_decode" => {
            let a = attn
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = a
                .layers
                .get(bound.layers.start as usize)
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            // [q] with the output guard-owned (`AttnCtx::o_out`), or
            // [q, o] when the op records its output as an SSA arg.
            let (q, o) = match bound.args.len() {
                1 => (bound.args[0], a.o_out),
                2 => (bound.args[0], bound.args[1].ptr),
                got => {
                    return Err(DispatchRefusal::ArgCount {
                        kernel: bound.kernel.to_string(),
                        expected: 1,
                        got,
                    });
                }
            };
            unsafe {
                ffi::pie_k_attn_dispatch_attention_flashinfer_decode(
                    a.decode_plan.cast_const(),
                    q.ptr,
                    *layer,
                    o,
                    a.kv_page_indices_d,
                    a.kv_page_indptr_d,
                    a.kv_last_page_lens_d,
                    a.workspace,
                    ctx.stream,
                    a.window_left_by_layer
                        .get(bound.layers.start as usize)
                        .copied()
                        .unwrap_or(a.window_left),
                    a.logits_soft_cap,
                    a.sm_scale,
                    a.lse_out_d,
                );
            }
        }
        // args: [packed, q_raw, k_raw, v] — one input, then the op's THREE
        // outputs stated as args (SplitQkv's outputs are values).
        "attn::split_qkv_bf16" => {
            need(4)?;
            let (packed, q, k, v) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            unsafe {
                ffi::pie_k_attn_split_qkv_bf16(
                    packed.ptr,
                    q.ptr,
                    k.ptr,
                    v.ptr,
                    rows,
                    i32::try_from(q.width).expect("q width"),
                    i32::try_from(k.width).expect("kv width"),
                    ctx.stream,
                );
            }
        }
        // args: [q_in, k_in, q_out, k_out, q_norm_w, k_norm_w]. The KERNEL
        // is in-place on (q, k); the lowering assigned separate in/out
        // buffers, so the arm stages in→out with a d2d copy, then runs the
        // kernel over the outs — the only reading under which both the
        // row's signature and the launch's buffer assignment are honest.
        "rope::qk_rmsnorm_rope_bf16" => {
            need(6)?;
            let (q_in, k_in, q_out, k_out, qw, kw) = (
                bound.args[0],
                bound.args[1],
                bound.args[2],
                bound.args[3],
                bound.args[4],
                bound.args[5],
            );
            stage_d2d(ctx, &bound.rows, q_out, q_in);
            stage_d2d(ctx, &bound.rows, k_out, k_in);
            unsafe {
                ffi::pie_k_rope_qk_rmsnorm_rope_bf16(
                    q_out.ptr,
                    k_out.ptr,
                    qw.ptr,
                    kw.ptr,
                    ctx.positions.cast_const().cast(),
                    rows,
                    i32::try_from(q_out.width).expect("q width") / ctx.head_dim.max(1),
                    i32::try_from(k_out.width).expect("k width") / ctx.head_dim.max(1),
                    ctx.head_dim,
                    ctx.rope_theta,
                    ctx.rms_eps,
                    ctx.stream,
                );
            }
        }
        // args: [k_curr, v_curr]; the layer view, the CSRs and the fire
        // scalars are the fire's.
        "attn::write_kv_to_pages" => {
            need(2)?;
            let a = attn
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = a
                .layers
                .get(bound.layers.start as usize)
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let (k_curr, v_curr) = (bound.args[0], bound.args[1]);
            unsafe {
                ffi::pie_k_attn_write_kv_to_pages(
                    *layer,
                    k_curr.ptr,
                    v_curr.ptr,
                    a.qo_indptr_d,
                    a.kv_page_indices_d,
                    a.kv_page_indptr_d,
                    a.kv_last_page_lens_d,
                    rows,
                    a.num_requests,
                    ctx.stream,
                    a.row_valid_d,
                    a.first_token,
                );
            }
        }
        // args: [] — everything is the fire's. A no-op on a native cache,
        // and the arm still fires it: the launch is stated, so it runs.
        "attn::dequant_kv_cache_layer_to_bf16_active" => {
            need(0)?;
            let a = attn
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = a
                .layers
                .get(bound.layers.start as usize)
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            unsafe {
                ffi::pie_k_attn_dequant_kv_cache_layer_to_bf16_active(
                    *layer,
                    a.kv_page_indices_d,
                    a.num_pages_in_batch,
                    ctx.stream,
                );
            }
        }
        // args: [q]; o is guard-owned ([`AttnCtx::o_out`]); the pages are
        // the layer's bf16 MIRRORS — the native alias, the decode lesson.
        "attn::dispatch_attention_flashinfer_prefill_bf16" => {
            let a = attn
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = a
                .layers
                .get(bound.layers.start as usize)
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            // [q] with the output guard-owned, or [q, o] as SSA.
            let (q, o) = match bound.args.len() {
                1 => (bound.args[0], a.o_out),
                2 => (bound.args[0], bound.args[1].ptr),
                got => {
                    return Err(DispatchRefusal::ArgCount {
                        kernel: bound.kernel.to_string(),
                        expected: 1,
                        got,
                    });
                }
            };
            unsafe {
                ffi::pie_k_attn_dispatch_attention_flashinfer_prefill_bf16(
                    a.prefill_plan.cast_const(),
                    q.ptr,
                    layer.k_bf16_pages,
                    layer.v_bf16_pages,
                    o,
                    a.qo_indptr_d,
                    a.kv_page_indices_d,
                    a.kv_page_indptr_d,
                    a.kv_last_page_lens_d,
                    a.workspace,
                    ctx.stream,
                    a.logits_soft_cap,
                    a.sm_scale,
                    a.lse_out_d,
                );
            }
        }
        // args: [q_in, k_in, q_out, k_out] — the same staged-in-place
        // shape as `qk_rmsnorm_rope`: the kernel rotates (q, k) where they
        // lie, the lowering may assign fresh out buffers, so the arm
        // stages in→out then rotates the outs.
        "rope::rope_bf16" => {
            need(4)?;
            let (q_in, k_in, q_out, k_out) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            stage_d2d(ctx, &bound.rows, q_out, q_in);
            stage_d2d(ctx, &bound.rows, k_out, k_in);
            unsafe {
                ffi::pie_k_rope_rope_bf16(
                    q_out.ptr,
                    k_out.ptr,
                    ctx.positions.cast_const().cast(),
                    rows,
                    i32::try_from(q_out.width).expect("q width") / ctx.head_dim.max(1),
                    i32::try_from(k_out.width).expect("k width") / ctx.head_dim.max(1),
                    ctx.head_dim,
                    ctx.rope_theta,
                    ctx.stream,
                    ctx.rope_interleaved,
                );
            }
        }
        // args: [a, b, out] — out = a + b. The kernel is the in-place
        // `y += x` over flat elements, so: stage a→out, add b.
        "norm::residual_add_bf16" => {
            need(3)?;
            let (a_in, b_in, out_arg) = (bound.args[0], bound.args[1], bound.args[2]);
            stage_d2d(ctx, &bound.rows, out_arg, a_in);
            let n = (bound.rows.end - bound.rows.start) as usize * out_arg.width as usize;
            unsafe {
                ffi::pie_k_norm_residual_add_bf16(out_arg.ptr, b_in.ptr, n, ctx.stream);
            }
        }
        // args: [x, out] — out = x + bias, the bias being the op's weight.
        // The kernel is in-place, so: stage x→out, add.
        "norm::add_bias_bf16" => {
            need(2)?;
            let (x_in, out_arg) = (bound.args[0], bound.args[1]);
            let w = weight(resolver)?;
            stage_d2d(ctx, &bound.rows, out_arg, x_in);
            unsafe {
                ffi::pie_k_norm_add_bias_bf16(
                    out_arg.ptr,
                    w,
                    rows,
                    i32::try_from(out_arg.width).expect("dim"),
                    ctx.stream,
                );
            }
        }
        // ── The qwen3_5 hybrid's arms ────────────────────────────────
        // args: [x, y]. Whole-row for the block/final norms; the
        // per-head q/k norms are the same symbol over `tokens * heads`
        // rows of `head_dim` — the op join says which reading applies.
        "norm::rmsnorm_gemma_bf16" => {
            need(2)?;
            let (x, y) = (bound.args[0], bound.args[1]);
            let w = weight(resolver)?;
            let (num_rows, hidden) = match spec.per_head_dim {
                Some(d) => (
                    rows * (i32::try_from(x.width).expect("width") / i32::try_from(d).expect("d")),
                    i32::try_from(d).expect("head_dim fits i32"),
                ),
                None => (rows, i32::try_from(x.width).expect("hidden fits i32")),
            };
            unsafe {
                ffi::pie_k_norm_rmsnorm_gemma_bf16(
                    x.ptr, w, y.ptr, num_rows, hidden, ctx.rms_eps, ctx.stream,
                );
            }
        }
        // args: [q_in, k_in, q_out, k_out] — in-place pair, staged like
        // `rope::rope_bf16`; the rotary width is the op's statement.
        "rope::rope_partial_bf16" => {
            need(4)?;
            let (q_in, k_in, q_out, k_out) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            let rotary = spec.rope_partial.ok_or_else(|| {
                DispatchRefusal::NoArm(format!("{}: op states no rotary width", bound.kernel))
            })?;
            stage_d2d(ctx, &bound.rows, q_out, q_in);
            stage_d2d(ctx, &bound.rows, k_out, k_in);
            unsafe {
                ffi::pie_k_rope_rope_partial_bf16(
                    q_out.ptr,
                    k_out.ptr,
                    ctx.positions.cast_const().cast(),
                    rows,
                    i32::try_from(q_out.width).expect("q width") / ctx.head_dim.max(1),
                    i32::try_from(k_out.width).expect("k width") / ctx.head_dim.max(1),
                    ctx.head_dim,
                    i32::try_from(rotary).expect("rotary fits i32"),
                    ctx.rope_theta,
                    ctx.stream,
                );
            }
        }
        // args: [packed, q_out, gate_out] — the 2×-wide gated q pack's
        // per-head de-interleave.
        "layout::split_q_gate_bf16" => {
            need(3)?;
            let (packed, q_out, gate_out) = (bound.args[0], bound.args[1], bound.args[2]);
            unsafe {
                ffi::pie_k_layout_split_q_gate_bf16(
                    packed.ptr,
                    q_out.ptr,
                    gate_out.ptr,
                    rows,
                    i32::try_from(q_out.width).expect("q width") / ctx.head_dim.max(1),
                    ctx.head_dim,
                    ctx.stream,
                );
            }
        }
        // args: [x, gate] in place, or [x, gate, out] when the lowering
        // assigned distinct buffers — staged, the in-place contract.
        "mlp::sigmoid_gate_inplace_bf16" => {
            let (x, gate) = match bound.args.len() {
                2 => (bound.args[0], bound.args[1]),
                3 => {
                    let (x_in, gate, out) = (bound.args[0], bound.args[1], bound.args[2]);
                    stage_d2d(ctx, &bound.rows, out, x_in);
                    (out, gate)
                }
                got => {
                    return Err(DispatchRefusal::ArgCount {
                        kernel: bound.kernel.to_string(),
                        expected: 2,
                        got,
                    });
                }
            };
            let n = rows * i32::try_from(x.width).expect("width fits i32");
            unsafe {
                ffi::pie_k_mlp_sigmoid_gate_inplace_bf16(x.ptr, gate.ptr, n, ctx.stream);
            }
        }
        // args: [x, y, conv_weight]. The bias rides the conv binding
        // (`<name>_bias`, null when the checkpoint has none); the state
        // slab, slot indirection and window geometry are the fire's.
        "ssm::causal_conv1d_update_batched_bf16" => {
            need(3)?;
            let g = gdn_ctx()?;
            let layer = state_layer()?;
            let (x, y, w) = (bound.args[0], bound.args[1], bound.args[2]);
            let bias = spec
                .weight
                .as_deref()
                .and_then(|n| resolver.weight(&format!("{n}_bias")))
                .unwrap_or(std::ptr::null());
            let state = slab(&g.conv_state, layer, "conv")?;
            unsafe {
                ffi::pie_k_ssm_causal_conv1d_update_batched_bf16(
                    x.ptr.cast_const(),
                    w.ptr.cast_const(),
                    bias,
                    state,
                    g.slot_ids_d,
                    g.conv_stride_elems,
                    y.ptr,
                    rows,
                    g.conv_dim,
                    g.conv_k,
                    ctx.stream,
                );
            }
        }
        // args: [x, y, conv_weight] — the prefill walk over the fire's
        // qo CSR; requests come from the attention context.
        "ssm::causal_conv1d_prefill_batched_bf16" => {
            need(3)?;
            let g = gdn_ctx()?;
            let a = attn.ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = state_layer()?;
            let (x, y, w) = (bound.args[0], bound.args[1], bound.args[2]);
            let bias = spec
                .weight
                .as_deref()
                .and_then(|n| resolver.weight(&format!("{n}_bias")))
                .unwrap_or(std::ptr::null());
            let state = slab(&g.conv_state, layer, "conv")?;
            unsafe {
                ffi::pie_k_ssm_causal_conv1d_prefill_batched_bf16(
                    x.ptr.cast_const(),
                    w.ptr.cast_const(),
                    bias,
                    y.ptr,
                    state,
                    g.slot_ids_d,
                    a.qo_indptr_d,
                    g.conv_stride_elems,
                    a.num_requests,
                    g.conv_dim,
                    g.conv_k,
                    ctx.stream,
                    g.write_state,
                    std::ptr::null(),
                    std::ptr::null(),
                );
            }
        }
        // args: [qkv_post, a, b, q, k, v, g, beta] — three inputs, the
        // op's five fp32 results; `a_log` (fp32-widened) and `dt_bias`
        // are the op's two named parameters.
        "ssm::qwen_gdn_post_conv_prep_bf16" => {
            need(8)?;
            let g = gdn_ctx()?;
            let a_log = weight(resolver)?;
            let dt_name = spec
                .weight2
                .as_deref()
                .ok_or_else(|| DispatchRefusal::NoWeight(bound.kernel.to_string()))?;
            let dt_bias = resolver
                .weight(dt_name)
                .ok_or_else(|| DispatchRefusal::UnknownWeight(dt_name.to_string()))?;
            unsafe {
                ffi::pie_k_ssm_qwen_gdn_post_conv_prep_bf16(
                    bound.args[0].ptr.cast_const(),
                    bound.args[1].ptr.cast_const(),
                    bound.args[2].ptr.cast_const(),
                    a_log,
                    dt_bias,
                    bound.args[3].ptr.cast(),
                    bound.args[4].ptr.cast(),
                    bound.args[5].ptr.cast(),
                    bound.args[6].ptr.cast(),
                    bound.args[7].ptr.cast(),
                    rows,
                    g.k_h,
                    g.v_h,
                    g.k_d,
                    g.v_d,
                    g.conv_dim,
                    ctx.stream,
                );
            }
        }
        // args: [q, k, v, g, beta, out] — the decode recurrence against
        // the layer's bf16 state slab (the deployment's `state_bf16`
        // fact picked this symbol at trace time).
        "ssm::recurrent_gated_delta_step_batched_state_bf16" => {
            need(6)?;
            let g = gdn_ctx()?;
            let layer = state_layer()?;
            let state = slab(&g.recurrent_state, layer, "recurrent")?;
            unsafe {
                ffi::pie_k_ssm_recurrent_gated_delta_step_batched_state_bf16(
                    bound.args[0].ptr.cast_const().cast(),
                    bound.args[1].ptr.cast_const().cast(),
                    bound.args[2].ptr.cast_const().cast(),
                    bound.args[3].ptr.cast_const().cast(),
                    bound.args[4].ptr.cast_const().cast(),
                    state,
                    g.slot_ids_d,
                    g.state_stride_elems,
                    bound.args[5].ptr.cast(),
                    rows,
                    g.v_h,
                    g.k_d,
                    g.v_d,
                    ctx.stream,
                );
            }
        }
        // args: [q, k, v, g, beta] — the fp32-STATE FLA prefill
        // recurrence (the `state_bf16: false` deployments' text).
        "ssm::chunk_gated_delta_prefill_batched" => {
            need(5)?;
            let g = gdn_ctx()?;
            let a = attn.ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = state_layer()?;
            let state = slab(&g.recurrent_state, layer, "recurrent")?;
            let core_out = out_slot(0, resolver)?;
            unsafe {
                ffi::pie_k_ssm_chunk_gated_delta_prefill_batched(
                    bound.args[0].ptr.cast_const().cast(),
                    bound.args[1].ptr.cast_const().cast(),
                    bound.args[2].ptr.cast_const().cast(),
                    bound.args[3].ptr.cast_const().cast(),
                    bound.args[4].ptr.cast_const().cast(),
                    state.cast(),
                    g.slot_ids_d,
                    a.qo_indptr_d,
                    g.state_stride_elems,
                    core_out.ptr.cast(),
                    a.num_requests,
                    g.k_h,
                    g.v_h,
                    g.k_d,
                    g.v_d,
                    ctx.stream,
                    g.write_state,
                    std::ptr::null(),
                    std::ptr::null(),
                );
            }
        }
        // args: [q, k, v, g, beta, out] — the chunked FLA prefill
        // recurrence over the fire's qo CSR.
        "ssm::chunk_gated_delta_prefill_batched_state_bf16" => {
            need(5)?;
            let g = gdn_ctx()?;
            let a = attn.ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = state_layer()?;
            let state = slab(&g.recurrent_state, layer, "recurrent")?;
            // The core output is the GUARD's value, not an SSA arg of
            // this region launch — the join walked back to it.
            let core_out = out_slot(0, resolver)?;
            unsafe {
                ffi::pie_k_ssm_chunk_gated_delta_prefill_batched_state_bf16(
                    bound.args[0].ptr.cast_const().cast(),
                    bound.args[1].ptr.cast_const().cast(),
                    bound.args[2].ptr.cast_const().cast(),
                    bound.args[3].ptr.cast_const().cast(),
                    bound.args[4].ptr.cast_const().cast(),
                    state,
                    g.slot_ids_d,
                    a.qo_indptr_d,
                    g.state_stride_elems,
                    core_out.ptr.cast(),
                    a.num_requests,
                    g.k_h,
                    g.v_h,
                    g.k_d,
                    g.v_d,
                    ctx.stream,
                    g.write_state,
                    std::ptr::null(),
                    std::ptr::null(),
                );
            }
        }
        // args: [x, gate, y] — the GDN landing norm: per (row, value
        // head) over the trailing head width, weight fp32-widened.
        "norm::rmsnorm_gated_fp32_in_bf16" => {
            need(3)?;
            let g = gdn_ctx()?;
            let (x, gate, y) = (bound.args[0], bound.args[1], bound.args[2]);
            let w = weight(resolver)?;
            unsafe {
                ffi::pie_k_norm_rmsnorm_gated_fp32_in_bf16(
                    x.ptr.cast_const(),
                    gate.ptr.cast_const(),
                    w,
                    y.ptr,
                    rows * g.v_h,
                    g.v_d,
                    ctx.rms_eps,
                    ctx.stream,
                );
            }
        }
        // ── gemma's arms ─────────────────────────────────────────────
        // args: [x_in, x_out, scale-name] — `x *= s`, the constant named
        // in the weight slot, resolved through `DispatchCtx::scales`.
        "norm::scalar_mul_bf16" => {
            need(3)?;
            let (x_in, x_out) = (bound.args[0], bound.args[1]);
            let name = spec
                .weight
                .as_deref()
                .and_then(|n| n.strip_prefix("scale."))
                .ok_or_else(|| DispatchRefusal::NoWeight(bound.kernel.to_string()))?;
            let s = *ctx
                .scales
                .get(name)
                .ok_or_else(|| DispatchRefusal::UnknownWeight(format!("scale.{name}")))?;
            stage_d2d(ctx, &bound.rows, x_out, x_in);
            let n = (bound.rows.end - bound.rows.start) as usize * x_out.width as usize;
            unsafe {
                ffi::pie_k_norm_scalar_mul_bf16(x_out.ptr, s, n, ctx.stream);
            }
        }
        // args: [gate, up, y] — gemma's GeGLU (tanh approximation); the
        // lowering lands y on the gate's bytes (the kernel's in-place
        // contract), staged when it assigned distinct buffers.
        "mlp::geglu_tanh_bf16" => {
            need(3)?;
            let (gate, up, y) = (bound.args[0], bound.args[1], bound.args[2]);
            stage_d2d(ctx, &bound.rows, y, gate);
            let n = rows * i32::try_from(y.width).expect("width fits i32");
            unsafe {
                ffi::pie_k_mlp_geglu_tanh_bf16(y.ptr.cast_const(), up.ptr.cast_const(), y.ptr, n, ctx.stream);
            }
        }
        // args: [x_in, x_out] — `cap * tanh(x / cap)` over the logits;
        // the cap is the deployment's final-softcap fact.
        "attn::logit_softcap_bf16" => {
            need(2)?;
            let (x_in, x_out) = (bound.args[0], bound.args[1]);
            stage_d2d(ctx, &bound.rows, x_out, x_in);
            let n = (bound.rows.end - bound.rows.start) as usize * x_out.width as usize;
            unsafe {
                ffi::pie_k_attn_logit_softcap_bf16(
                    x_out.ptr,
                    ctx.final_logit_softcap,
                    n,
                    ctx.stream,
                );
            }
        }
        // args: [packed, q_out, q_norm, k_norm] — gemma-4's fused local
        // decode post: split the packed projection, norm q/k, rope them
        // (rounded), norm v, write k/v straight to the pages. Only the
        // query survives as a value.
        "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16" => {
            need(4)?;
            let a = attn
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = a
                .layers
                .get(bound.layers.start as usize)
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let (packed, q_out, qw, kw) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            unsafe {
                ffi::pie_k_attn_qkv_packed_qk_norm_rope_vnorm_write_kv_bf16(
                    packed.ptr.cast_const(),
                    q_out.ptr,
                    layer.k_pages,
                    layer.v_pages,
                    qw.ptr.cast_const(),
                    kw.ptr.cast_const(),
                    ctx.positions.cast_const().cast(),
                    a.kv_page_indices_d,
                    a.kv_page_indptr_d,
                    a.kv_last_page_lens_d,
                    a.row_valid_d,
                    rows,
                    i32::try_from(q_out.width).expect("q width") / layer.head_dim.max(1),
                    layer.num_kv_heads,
                    layer.head_dim,
                    layer.page_size,
                    layer.hnd_layout,
                    ctx.rope_theta,
                    ctx.rms_eps,
                    ctx.stream,
                );
            }
        }
        // The rounded fused norm+rope, BOTH shapes the driver reaches:
        // [q, k, q_norm, k_norm] — the local pair, in place — and
        // [q_in, q_out, q_norm] — a KV-SHARED layer's Q-ONLY form, which
        // the driver reaches by passing `num_kv_heads = 0`, never by a
        // generic rope.
        "rope::qk_rmsnorm_rope_bf16_rounded" => {
            let a = attn
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = a
                .layers
                .get(bound.layers.start as usize)
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let (q, k, qw, kw, kv_heads) = match bound.args.len() {
                4 => (
                    bound.args[0],
                    bound.args[1].ptr,
                    bound.args[2],
                    bound.args[3].ptr.cast_const(),
                    layer.num_kv_heads,
                ),
                3 => {
                    let (q_in, q_out, qw) = (bound.args[0], bound.args[1], bound.args[2]);
                    stage_d2d(ctx, &bound.rows, q_out, q_in);
                    (q_out, std::ptr::null_mut(), qw, std::ptr::null(), 0)
                }
                got => {
                    return Err(DispatchRefusal::ArgCount {
                        kernel: bound.kernel.to_string(),
                        expected: 4,
                        got,
                    });
                }
            };
            unsafe {
                ffi::pie_k_rope_qk_rmsnorm_rope_bf16_rounded(
                    q.ptr,
                    k,
                    qw.ptr.cast_const(),
                    kw,
                    ctx.positions.cast_const().cast(),
                    rows,
                    i32::try_from(q.width).expect("q width") / layer.head_dim.max(1),
                    kv_heads,
                    layer.head_dim,
                    ctx.rope_theta,
                    ctx.rms_eps,
                    ctx.stream,
                );
            }
        }
        // args: [q, o] — the PLANLESS flashinfer prefill (plans
        // internally per fire; reads the host CSR mirrors).
        "attn::attention_flashinfer_prefill" => {
            need(2)?;
            let a = attn
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = a
                .layers
                .get(bound.layers.start as usize)
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let (q, o) = (bound.args[0], bound.args[1]);
            unsafe {
                ffi::pie_k_attn_attention_flashinfer_prefill(
                    q.ptr.cast_const(),
                    *layer,
                    o.ptr,
                    a.qo_indptr_d,
                    a.kv_page_indices_d,
                    a.kv_page_indptr_d,
                    a.kv_last_page_lens_d,
                    a.qo_indptr_h,
                    a.kv_page_indptr_h,
                    rows,
                    a.num_requests,
                    i32::try_from(q.width).expect("q width") / layer.head_dim.max(1),
                    a.workspace,
                    ctx.stream,
                    a.window_left_by_layer
                        .get(bound.layers.start as usize)
                        .copied()
                        .unwrap_or(a.window_left),
                    a.logits_soft_cap,
                    a.sm_scale,
                    a.lse_out_d,
                );
            }
        }
        // args: [q, o] — the naive paged prefill, for the head dims
        // flashinfer's prefill template refuses (gemma-4's 512).
        "attn::attention_naive_paged" => {
            need(2)?;
            let a = attn
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let layer = a
                .layers
                .get(bound.layers.start as usize)
                .ok_or_else(|| DispatchRefusal::NoAttnCtx(bound.kernel.to_string()))?;
            let (q, o) = (bound.args[0], bound.args[1]);
            unsafe {
                ffi::pie_k_attn_attention_naive_paged(
                    q.ptr.cast_const(),
                    *layer,
                    o.ptr,
                    a.qo_indptr_d,
                    a.kv_page_indices_d,
                    a.kv_page_indptr_d,
                    a.kv_last_page_lens_d,
                    rows,
                    a.num_requests,
                    a.num_pages_in_batch,
                    i32::try_from(q.width).expect("q width") / layer.head_dim.max(1),
                    ctx.stream,
                    a.window_left_by_layer
                        .get(bound.layers.start as usize)
                        .copied()
                        .unwrap_or(a.window_left),
                    a.sm_scale,
                );
            }
        }
        // ── gemma-4's arms ───────────────────────────────────────────
        // args: [src, dst] — the PLE relay: `[N, layers*dim]` transposed
        // to `[layers, N, dim]` so each layer reads a contiguous slice.
        "layout::transpose_bf16_nld_to_lnd" => {
            need(2)?;
            let (src, dst) = (bound.args[0], bound.args[1]);
            if ctx.ple_dim <= 0 {
                return Err(DispatchRefusal::NoArm(format!(
                    "{}: the fire states no ple_dim",
                    bound.kernel
                )));
            }
            unsafe {
                ffi::pie_k_layout_transpose_bf16_nld_to_lnd(
                    src.ptr.cast_const().cast(),
                    dst.ptr.cast(),
                    rows,
                    i32::try_from(src.width).expect("width") / ctx.ple_dim,
                    ctx.ple_dim,
                    ctx.stream,
                );
            }
        }
        // args: [x, hidden_in, hidden_out, norm_out, w, next_w] — FOUR
        // statements in one launch: norm x, land on the stream, scale,
        // norm THAT with the next block's weight. The scale is 1 at the
        // attention landing; the PLE landing carries the layer's own
        // scalar, resolved from `DispatchCtx::scales` by the weight's
        // name (the C++ reads `layer_scalar_value` the same way).
        "norm::rmsnorm_residual_add_scale_rmsnorm_bf16" => {
            need(6)?;
            let (x, hid_in, hid_out, norm_out, w, next_w) = (
                bound.args[0],
                bound.args[1],
                bound.args[2],
                bound.args[3],
                bound.args[4],
                bound.args[5],
            );
            let scale = spec
                .weight
                .as_deref()
                .filter(|n| n.ends_with("ple_norm"))
                .map_or(1.0, |n| ctx.scales.get(n).copied().unwrap_or(1.0));
            stage_d2d(ctx, &bound.rows, hid_out, hid_in);
            unsafe {
                ffi::pie_k_norm_rmsnorm_residual_add_scale_rmsnorm_bf16(
                    x.ptr.cast_const(),
                    w.ptr.cast_const(),
                    hid_out.ptr,
                    scale,
                    next_w.ptr.cast_const(),
                    norm_out.ptr,
                    rows,
                    i32::try_from(x.width).expect("hidden fits i32"),
                    ctx.rms_eps,
                    ctx.stream,
                );
            }
        }
        // args: [x, hidden_in, hidden_out, w] — the two-statement form:
        // norm x, land on the stream (gemma-4's post-feedforward norm).
        "norm::rmsnorm_residual_add_bf16" => {
            need(4)?;
            let (x, hid_in, hid_out, w) =
                (bound.args[0], bound.args[1], bound.args[2], bound.args[3]);
            stage_d2d(ctx, &bound.rows, hid_out, hid_in);
            unsafe {
                ffi::pie_k_norm_rmsnorm_residual_add_bf16(
                    x.ptr.cast_const(),
                    w.ptr.cast_const(),
                    hid_out.ptr,
                    rows,
                    i32::try_from(x.width).expect("hidden fits i32"),
                    ctx.rms_eps,
                    ctx.stream,
                );
            }
        }
        // args: [packed, y] — GeGLU over the packed gate‖up bank.
        "mlp::chunked_geglu_tanh_bf16" => {
            need(2)?;
            let (packed, y) = (bound.args[0], bound.args[1]);
            unsafe {
                ffi::pie_k_mlp_chunked_geglu_tanh_bf16(
                    packed.ptr.cast_const(),
                    y.ptr,
                    rows,
                    i32::try_from(y.width).expect("I fits i32"),
                    ctx.stream,
                    ctx.gate_second,
                );
            }
        }
        // args: [x, y] — the weightless per-head V-norm (`v / rms(v)`).
        "norm::rmsnorm_no_scale_bf16" => {
            need(2)?;
            let (x, y) = (bound.args[0], bound.args[1]);
            let (num_rows, hidden) = match spec.per_head_dim {
                Some(d) => (
                    rows * (i32::try_from(x.width).expect("width") / i32::try_from(d).expect("d")),
                    i32::try_from(d).expect("head_dim fits i32"),
                ),
                None => (rows, i32::try_from(x.width).expect("hidden fits i32")),
            };
            unsafe {
                ffi::pie_k_norm_rmsnorm_no_scale_bf16(
                    x.ptr.cast_const(),
                    y.ptr,
                    num_rows,
                    hidden,
                    ctx.rms_eps,
                    ctx.stream,
                );
            }
        }
        other => return Err(DispatchRefusal::NoArm(other.to_string())),
    }
    Ok(())
}

/// Stage `src` into `dst` (device-to-device) when the lowering assigned an
/// in-place kernel distinct in/out buffers — the executor's half of that
/// contract, shared by the rope and elementwise-add arms.
#[cfg(feature = "bridge")]
fn stage_d2d(ctx: &DispatchCtx, rows: &std::ops::Range<u32>, dst: BoundArg, src: BoundArg) {
    if dst.ptr != src.ptr {
        use cudarc::runtime::sys::{cudaError, cudaMemcpyAsync, cudaMemcpyKind};
        let bytes = (rows.end - rows.start) as usize * src.width as usize * 2;
        let code = unsafe {
            cudaMemcpyAsync(
                dst.ptr,
                src.ptr.cast_const(),
                bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                ctx.stream.cast(),
            )
        };
        assert!(code == cudaError::cudaSuccess, "d2d stage: {code:?}");
    }
}

/// A store-backed [`Resolver`]: the per-family MAP, productized. The
/// loader (or a test) fills it; the executor asks it.
#[derive(Debug, Default)]
pub struct MapResolver {
    weights: std::collections::BTreeMap<String, *const c_void>,
    named: std::collections::BTreeMap<ValueId, *mut c_void>,
}

impl MapResolver {
    /// An empty map — every ask is a drift until something is inserted.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind a weight name to its device tensor.
    pub fn insert_weight(&mut self, name: impl Into<String>, ptr: *const c_void) {
        self.weights.insert(name.into(), ptr);
    }

    /// Bind a pinned seam value to its buffer.
    pub fn insert_named(&mut self, value: ValueId, ptr: *mut c_void) {
        self.named.insert(value, ptr);
    }
}

impl Resolver for MapResolver {
    fn weight(&mut self, name: &str) -> Option<*const c_void> {
        self.weights.get(name).copied()
    }
    fn named(&mut self, value: ValueId) -> Option<*mut c_void> {
        self.named.get(&value).copied()
    }
}

/// Why a fire's walk stopped.
#[cfg(feature = "bridge")]
#[derive(Debug)]
pub struct RunRefusal {
    /// Which launch refused.
    pub launch: usize,
    /// Its kernel.
    pub kernel: String,
    /// The refusal itself.
    pub why: RunRefusalKind,
}

/// The two ways a launch refuses.
#[cfg(feature = "bridge")]
#[derive(Debug)]
pub enum RunRefusalKind {
    /// Binding refused — see [`BindRefusal`].
    Bind(BindRefusal),
    /// Dispatch refused — see [`DispatchRefusal`].
    Dispatch(DispatchRefusal),
}

/// Execute one fire: bind and dispatch every launch of the lowering, in
/// order. The walk the full-decode smoke proved, as the executor's entry.
///
/// # Errors
///
/// The first refusing launch, with its index and kernel — a drift
/// diagnosis, never a runtime condition to retry.
#[cfg(feature = "bridge")]
pub fn run<R: Resolver>(
    lowered: &Lowered,
    dplan: &DispatchPlan,
    frame: Frame,
    resolver: &mut R,
    ctx: &DispatchCtx,
    attn: Option<&AttnCtx>,
    gdn: Option<&GdnCtx>,
) -> Result<usize, RunRefusal> {
    for (i, launch) in lowered.launches.iter().enumerate() {
        let kernel = || lowered.kernels[launch.kernel as usize].clone();
        let bound = bind(lowered, launch, frame, resolver).map_err(|e| RunRefusal {
            launch: i,
            kernel: kernel(),
            why: RunRefusalKind::Bind(e),
        })?;
        dispatch(&bound, dplan.spec(i), frame, resolver, ctx, attn, gdn).map_err(|e| {
            RunRefusal { launch: i, kernel: kernel(), why: RunRefusalKind::Dispatch(e) }
        })?;
    }
    Ok(lowered.launches.len())
}

/// Resolve one [`Arg`] — the three rules, shared by [`bind`] and by the
/// arms that resolve an op's OUTPUT placements from the join.
///
/// # Errors
///
/// See [`BindRefusal`].
pub fn resolve_arg<R: Resolver>(
    arg: &Arg,
    frame: Frame,
    resolver: &mut R,
) -> Result<BoundArg, BindRefusal> {
    Ok(match arg {
        Arg::Arena { at, width } => {
            if *at >= frame.arena_bytes {
                return Err(BindRefusal::ArenaOutOfBounds {
                    at: *at,
                    arena_bytes: frame.arena_bytes,
                });
            }
            BoundArg {
                ptr: unsafe { frame.arena.cast::<u8>().add(*at) }.cast(),
                width: *width,
            }
        }
        Arg::Named { value, width } => BoundArg {
            ptr: resolver
                .named(*value)
                .ok_or(BindRefusal::UnknownNamed(*value))?,
            width: *width,
        },
        Arg::Weight(name) => {
            // `scale.` marks a CONSTANT riding the name slot — "a binder
            // never looks for it" (`dsl::cuda::scalar_mul`). The value
            // reaches the arm through `DispatchCtx::scales`; the operand
            // slot binds a dangling sentinel so the launch's arity holds.
            if name.starts_with("scale.") {
                BoundArg { ptr: std::ptr::NonNull::<c_void>::dangling().as_ptr(), width: 0 }
            } else {
                BoundArg {
                    ptr: resolver
                        .weight(name)
                        .ok_or_else(|| BindRefusal::UnknownWeight(name.clone()))?
                        .cast_mut(),
                    width: 0,
                }
            }
        }
    })
}

/// Bind one launch's operands against the frame and the resolver.
///
/// # Errors
///
/// See [`BindRefusal`] — each names the drift it diagnoses.
pub fn bind<'a, R: Resolver>(
    lowered: &'a Lowered,
    launch: &Launch,
    frame: Frame,
    resolver: &mut R,
) -> Result<BoundLaunch<'a>, BindRefusal> {
    let mut args = Vec::with_capacity(launch.args.len());
    for arg in &lowered.args[launch.args.start as usize..launch.args.end as usize] {
        args.push(resolve_arg(arg, frame, resolver)?);
    }
    Ok(BoundLaunch {
        kernel: &lowered.kernels[launch.kernel as usize],
        rows: launch.rows.clone(),
        layers: launch.layers.clone(),
        args,
    })
}
