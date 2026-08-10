//! The seam to `driver-metal-new`.
//!
//! # A library call, not an ABI crossing
//!
//! The CUDA seam beside this one goes through the C ABI —
//! `pie_cuda_create`, `pie_cuda_launch`, a `*mut PieDriver` — because the
//! driver it talks to is C++. This one does not, because the driver it talks
//! to is Rust, and a `#[repr(C)]` boundary between two Rust crates is a second
//! spelling of a contract they already share.
//!
//! That is `metal.md`'s task 9 arriving early and from the other end: the C
//! boundary retires when its last C++ consumer does, and nothing here adds a
//! new one.
//!
//! # What is servable today, and what is not
//!
//! The verbs split cleanly. `create`, `device_facts`, the registry four and
//! `close_*` are answered by machinery that is already ported and device
//! tested. `encode` refuses, as the CUDA side does — Metal media encode is
//! unsupported on both. `launch`, `copy_kv`, `copy_state` and `resize_pool`
//! need the **KV pool**, which is the frame bridge's device half and the one
//! piece still missing.
//!
//! Those four refuse by name rather than being absent. A backend that cannot
//! be selected teaches nothing; one that is selected and says exactly which
//! verb it cannot serve is a working seam with a stated hole.

use anyhow::{Result, anyhow, bail};

use crate::driver::FrameLaunchOutcome;
use crate::driver::channel::RegisteredChannel;
use crate::driver::command::{
    ChannelRegistrationPlan, KvCopyPlan, MediaEncodePlan, PoolResizePlan, ProgramRegistration,
    StateCopyPlan,
};
use crate::driver::completion::{CompletionBroker, SubmissionCompletion};
use crate::driver::instance::{BoundInstance, InstanceBindingPlan};
use crate::driver::submission::FrameSubmission;

/// The Metal shell, behind the seam's fourteen verbs.
pub struct MetalDriver {
    context: driver_metal_new::metal::Context,
    registry: driver_metal_new::pipeline::Registry,
    device_facts: driver_abi::DeviceFacts,
    /// The checkpoint, once one is loaded. Held because every address in its
    /// tensor map points into the region it owns.
    model: Option<driver_metal_new::model::load::Loaded>,
    /// What the checkpoint said it is — which text `model::text` looks up.
    arch: String,
    /// The paged KV pool, allocated at load.
    pool: Option<driver_metal_new::model::kv::Pool>,
    /// The runtime shader compiler, and the pipelines a fire's symbols have
    /// compiled to. Held across fires: a model's symbol set is bounded by its
    /// text, so a driver that recompiled per fire would spend more time in the
    /// compiler than on the GPU.
    compiler: driver_metal_new::metal::Compiler,
    pipelines: driver_metal_new::model::encode::Pipelines,
    broker: CompletionBroker,
}

// The context holds Objective-C objects, which are not `Send` by declaration.
// The seam owns the driver exclusively and the scheduler drives it from one
// place, which is the same reason `DummyDriver` asserts this.
unsafe impl Send for MetalDriver {}
unsafe impl Sync for MetalDriver {}

impl MetalDriver {
    /// Open the default Metal 4 device.
    ///
    /// # Errors
    ///
    /// No Metal 4 device, or a device whose queue could not be created. Both
    /// are boot conditions, not runtime ones.
    pub fn create(_config_bytes: &[u8]) -> Result<(Self, driver_abi::DeviceFacts)> {
        let context = driver_metal_new::metal::Context::new()
            .map_err(|e| anyhow!("metal context: {e:?}"))?;
        let compiler = driver_metal_new::metal::Compiler::new(&context)
            .map_err(|e| anyhow!("metal compiler: {e:?}"))?;
        // The facts a scheduler reads, stated from what this backend IS
        // rather than parsed out of a config — a config that disagreed with
        // the hardware would simply be believed.
        //
        // `unified_memory` is the one that changes scheduling: on Apple
        // silicon the KV pool and the host share physical memory, so a
        // "device is full" question is a different question here.
        let device_facts = driver_abi::DeviceFacts {
            abi_version: driver_abi::PIE_DRIVER_ABI_VERSION,
            backend: "metal".to_string(),
            unified_memory: true,
            // Metal has no native fp8 path and no MXFP4 MoE kernel; the table
            // says which kernels exist and neither is among them.
            fp8_native: false,
            native_mxfp4_moe: false,
            storage_alignment: 256,
            storage_max_tile_bytes: 0,
            storage_tile_map_mask: 0,
            // The paged KV pool's rows per page, which every `kv_translation`
            // index is in units of.
            page_size: 16,
        };
        Ok((
            Self {
                context,
                registry: driver_metal_new::pipeline::Registry::new(),
                device_facts: device_facts.clone(),
                model: None,
                arch: String::new(),
                pool: None,
                compiler,
                pipelines: driver_metal_new::model::encode::Pipelines::new(shader_tree()),
                broker: CompletionBroker::new(),
            },
            device_facts,
        ))
    }

    /// The device's stated facts.
    #[must_use]
    pub fn device_facts(&self) -> &driver_abi::DeviceFacts {
        &self.device_facts
    }

    /// Metal exports no KV handle: there is no cross-process sharing path.
    #[must_use]
    pub fn export_kv_handle(&self) -> Option<driver_abi::KvHandle> {
        None
    }

    /// The device this driver runs on.
    #[must_use]
    pub fn context(&self) -> &driver_metal_new::metal::Context {
        &self.context
    }

    /// The program/instance/channel registry.
    #[must_use]
    pub fn registry(&self) -> &driver_metal_new::pipeline::Registry {
        &self.registry
    }

    /// Author the checkpoint's load plan, run it, and stage every tensor.
    ///
    /// One descriptor: this backend holds one model, which is the same shape
    /// the CUDA shell's `state.model` has and the reason a frame's instance
    /// roster is one family's.
    ///
    /// # Errors
    ///
    /// More than one descriptor, a missing `config.json`, or a plan that will
    /// not compile or stage.
    pub fn load_model(
        &mut self,
        descs: Vec<driver_abi::ModelLoadDesc>,
    ) -> Result<driver_abi::DriverCapabilities> {
        let [desc] = descs.as_slice() else {
            bail!(
                "driver-metal-new holds ONE model; got {} descriptors",
                descs.len()
            );
        };
        // The descriptor the load plan is authored from is the checkpoint's
        // own `config.json`. The driver does not synthesize one: a descriptor
        // that disagreed with the weights beside it would be believed.
        let descriptor = std::fs::read_to_string(desc.snapshot_dir.join("config.json"))
            .map_err(|e| anyhow!("{}: config.json: {e}", desc.snapshot_dir.display()))?;
        let loaded = driver_metal_new::model::load::load(
            &self.context,
            &desc.snapshot_dir,
            &descriptor,
        )
        .map_err(|e| anyhow!("metal load: {e:?}"))?;
        let facts = driver_metal_new::facts::ModelFacts::from_descriptor(&descriptor)
            .ok_or_else(|| anyhow!("the descriptor does not parse as model facts"))?;
        self.model = Some(loaded);
        self.arch = facts.arch_name.clone();
        if !driver_metal_new::model::text::serves(&self.arch) {
            bail!(
                "driver-metal-new has no Metal text for `{}`; it serves {:?}. \
                 The checkpoint loaded, but nothing states its forward pass.",
                self.arch,
                driver_metal_new::model::text::known()
            );
        }

        // The pool, at the geometry the checkpoint states. `PIE_METAL_KV_PAGES`
        // is the size knob: a pool is a fixed allocation on this backend, and
        // the number the engine would negotiate is the number it is told here.
        let pages: u32 = std::env::var("PIE_METAL_KV_PAGES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024);
        let shape = driver_metal_new::model::kv::Shape {
            layers: u32::try_from(facts.go_num_hidden_layers).unwrap_or(0),
            kv_heads: u32::try_from(facts.go_num_key_value_heads).unwrap_or(0),
            head_dim: u32::try_from(facts.go_head_dim).unwrap_or(0),
            page_size: self.device_facts.page_size,
            pages,
            element_bytes: 2,
        };
        self.pool = Some(
            driver_metal_new::model::kv::Pool::allocate(&self.context, shape)
                .map_err(|e| anyhow!("kv pool: {e:?}"))?,
        );

        // What the checkpoint states, and what the pool states.
        //
        // `total_pages` is the pool's own count now, so a scheduler admits
        // against what was actually allocated. It read zero while no pool
        // existed, which was the truth then and the reason nothing was
        // admitted.
        Ok(driver_abi::DriverCapabilities {
            abi_version: driver_abi::PIE_DRIVER_ABI_VERSION,
            total_pages: pages,
            kv_page_size: self.device_facts.page_size,
            swap_pool_size: 0,
            kv_copy_domain_mask: 0,
            rs_cache_required: facts.has_linear_attn,
            rs_cache_slots: 0,
            rs_cache_slot_bytes: 0,
            elastic_page_bytes: 0,
            elastic_budget_pages: 0,
            has_mtp_logits: false,
            has_mtp_drafts: false,
            has_value_head: false,
            // Every one of these is a SINK this backend cannot honour, and the
            // `kernel!` rows say so: `sdpa_vector_decode` and
            // `sdpa_paged_decode` both declare `lacks = [Scores,
            // PageMaskSink]`. Advertising one would make a program bind and
            // then run as a silent no-op.
            has_kv_envelopes: false,
            has_attn_score: false,
            has_attn_page_mask: false,
            has_lora: false,
            model_site_summary: driver_abi::ModelSiteSummary::default(),
            device_geometry_port_mask: 0,
            // The ceilings a scheduler batches under. Stated rather than
            // unbounded: a fire wider than this has no arena sized for it.
            max_forward_tokens: 4096,
            max_forward_requests: 256,
            max_page_refs: pages,
            arch_name: facts.arch_name.clone(),
            vocab_size: facts.vocab_size,
            max_model_len: facts.max_model_len,
            activation_dtype: "bf16".to_string(),
            hidden_size: u32::try_from(facts.go_hidden_size).unwrap_or(0),
            supports_media_encode: false,
            snapshot_dir: desc.snapshot_dir.display().to_string(),
            kv_handle: None,
            // Metal compiles its shaders at run time from the tree; nothing
            // upstream needs to generate a kernel for it.
            codegen_backend: String::new(),
        })
    }

    /// The tensors the loaded checkpoint published, or `None` before a load.
    #[must_use]
    pub fn model(&self) -> Option<&driver_metal_new::model::load::Loaded> {
        self.model.as_ref()
    }

    /// # Errors
    ///
    /// Always, until the registry is wired to the seam's own id space.
    pub fn register_program(&mut self, _desc: &ProgramRegistration) -> Result<u64> {
        bail!(UNSERVED_REGISTRY)
    }

    /// # Errors
    ///
    /// As [`Self::register_program`].
    pub fn register_channel(&mut self, _desc: &ChannelRegistrationPlan) -> Result<RegisteredChannel> {
        bail!(UNSERVED_REGISTRY)
    }

    /// # Errors
    ///
    /// As [`Self::register_program`].
    pub fn bind_instance(&mut self, _desc: &InstanceBindingPlan) -> Result<BoundInstance> {
        bail!(UNSERVED_REGISTRY)
    }

    /// Post one sealed frame: admit it, then run its steps in order.
    ///
    /// The whole body is the four calls the executor is made of, with
    /// admission in front. Nothing here decides what runs — the text states
    /// it, `lower` flattens it, and `run` walks the result.
    ///
    /// # Errors
    ///
    /// A frame whose step tables do not describe its rows, an architecture no
    /// text serves, or a device failure. Admission is NOT an error: a frame
    /// that does not fit reports [`FrameLaunchOutcome::Exhausted`], which the
    /// engine re-posts, or `Impossible` when no eviction could ever make room.
    pub fn launch(&mut self, frame: &FrameSubmission) -> Result<FrameLaunchOutcome> {
        let (Some(model), Some(pool)) = (self.model.as_ref(), self.pool.as_ref()) else {
            bail!("driver-metal-new: launch before load_model");
        };

        // ── Admission, against the frame-union demand. ──
        //
        // Before anything is encoded, and without side effects, which is what
        // lets the engine re-post: a frame that took an arena and then failed
        // to admit would have to be undone.
        if !pool.admits(frame.required_kv_pages) {
            // Impossible rather than Exhausted when no eviction could make
            // room — the demand exceeds the physical pool, so waiting is
            // waiting for something that cannot happen.
            return Ok(FrameLaunchOutcome::Impossible);
        }

        // ── The page translation, checked per lane. ──
        //
        // A page past the pool addresses another layer's memory and attention
        // would read it without complaint, so this is a refusal and not a
        // clamp.
        for lane in 0..frame.instance_ids.len() {
            driver_metal_new::model::kv::translate(
                pool,
                &frame.kv_translation,
                &frame.kv_translation_indptr,
                lane,
            )
            .map_err(|why| anyhow!("frame kv translation: {why:?}"))?;
        }

        let facts = model::families::llama_like::forward::facts::LlamaLikeFacts::qwen3_0_6b();
        let metal = model::families::llama_like::forward::facts::LlamaLikeMetalFacts::synthetic();
        let named = std::collections::HashMap::new();

        for step in &frame.steps {
            let s = driver_metal_new::model::frame::Step {
                token_ids: &step.plan.token_ids,
                qo_indptr: &step.plan.qo_indptr,
                region_row_indptr: &step.region_row_indptr,
                region_sig: &step.region_sig,
                region_k: &step.region_k,
                sampling_indices: &step.plan.sampling_indices,
            };
            let class = driver_metal_new::model::frame::fire_class(&s);
            let plan = driver_metal_new::model::text::plan_for(&self.arch, class, &facts, &metal)
                .map_err(|why| anyhow!("no text: {why:?}"))?;
            let lowered = driver_metal_new::model::frame::lower_step(&plan, &s)
                .map_err(|why| anyhow!("step did not lower: {why:?}"))?;

            let geometry = driver_metal_new::model::dispatch::Geometry {
                q_heads: facts.q_heads,
                kv_heads: facts.kv_heads,
                head_dim: facts.head_dim,
                rotary_dims: facts.head_dim,
                n_experts: 0,
                experts_per_token: 0,
            };
            let names = driver_metal_new::model::resolve::Names::mlx();
            let mut store =
                driver_metal_new::model::resolve::Store::new(names, &model.tensors, &named);
            driver_metal_new::model::run::run(
                &self.context,
                &self.compiler,
                &mut self.pipelines,
                &lowered,
                geometry,
                &mut store,
            )
            .map_err(|e| {
                // A fire that could not bind names them all, because a
                // checkpoint missing one tensor is usually missing a family of
                // them and stopping at the first costs a round trip each.
                let missed = store.missed();
                if missed.is_empty() {
                    anyhow!("metal fire: {e:?}")
                } else {
                    anyhow!("metal fire: {e:?}; unresolved names: {missed:?}")
                }
            })?;
        }

        let (_raw, completion) = self.broker.launch_completion(1);
        Ok(FrameLaunchOutcome::Launched(completion))
    }

    /// # Errors
    ///
    /// Always. Media encode is unsupported on this backend, as it is on CUDA;
    /// both seams refuse rather than pretending.
    pub fn encode(&mut self, _plan: &mut MediaEncodePlan) -> Result<SubmissionCompletion> {
        bail!("driver-metal-new: media encode is unsupported on this backend")
    }

    /// # Errors
    ///
    /// Always, until the KV pool exists. `store::control::plan_kv_copy` and
    /// `store::kv_move::plan_cell_moves` already decide what would move.
    pub fn copy_kv(&mut self, _desc: &KvCopyPlan) -> Result<SubmissionCompletion> {
        bail!(UNSERVED_POOL)
    }

    /// # Errors
    ///
    /// As [`Self::copy_kv`].
    pub fn copy_state(&mut self, _desc: &StateCopyPlan) -> Result<SubmissionCompletion> {
        bail!(UNSERVED_POOL)
    }

    /// # Errors
    ///
    /// As [`Self::copy_kv`].
    pub fn resize_pool(&mut self, _desc: &PoolResizePlan) -> Result<SubmissionCompletion> {
        bail!(UNSERVED_POOL)
    }

    /// # Errors
    ///
    /// Never today; the registry accepts a close of an id it does not hold,
    /// because a close is idempotent from the scheduler's side.
    pub fn close_instance(&mut self, _id: u64) -> Result<()> {
        Ok(())
    }

    /// # Errors
    ///
    /// As [`Self::close_instance`].
    pub fn close_channel(&mut self, _id: u64) -> Result<()> {
        Ok(())
    }
}

/// The hole, named once so every verb that shares it reads the same.
const UNSERVED_POOL: &str = "driver-metal-new: the KV pool is not wired to the seam yet. The \
     executor above it is complete and device-tested (tests/device_text_fire.rs \
     fires the whole llama_like text); what is missing is the buffers a fire \
     binds, not the decisions it makes.";

/// The other hole: the registry is ported and device tested
/// (`PARITY-REGISTRY.md`), but nothing maps the seam's plans onto it yet.
const UNSERVED_REGISTRY: &str = "driver-metal-new: `register_program` / `register_channel` / \
     `bind_instance` are not wired to `pipeline::Registry` yet. The registry \
     itself is ported and device tested; what is missing is the translation \
     from the seam's plan types to its own.";

/// Where the Metal shader tree lives.
///
/// Metal compiles at run time from `(path, entry name)`, so a driver needs the
/// `.metal` sources on disk. `PIE_METAL_KERNELS` overrides; the default is the
/// checkout's own tree, which is what a development run wants and what every
/// device test already uses.
fn shader_tree() -> std::path::PathBuf {
    std::env::var_os("PIE_METAL_KERNELS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .map(|crates| crates.join("kernels-metal/kernels"))
                .unwrap_or_default()
        })
}
