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
    #[allow(dead_code)]
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

        // What the checkpoint states, and zero for what the pool would state.
        //
        // `total_pages: 0` is not a placeholder — it is the truth, and it is
        // the one field that keeps this honest. A scheduler reading it admits
        // nothing, which is exactly right for a backend whose KV pool does not
        // exist: a non-zero guess here would be admitted against and the fire
        // would fail at launch instead of at admission.
        Ok(driver_abi::DriverCapabilities {
            abi_version: driver_abi::PIE_DRIVER_ABI_VERSION,
            total_pages: 0,
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
            max_forward_tokens: 0,
            max_forward_requests: 0,
            max_page_refs: 0,
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

    /// Post one sealed frame.
    ///
    /// # Errors
    ///
    /// Always, until the KV pool exists. Everything ABOVE the pool is done and
    /// tested — `model::frame::lower_step` turns this submission's step into
    /// rows and rectangles, and `model::run::run` turns those into a command
    /// buffer — so what is missing is the buffers a fire binds, not the
    /// decisions it makes.
    pub fn launch(&mut self, _frame: &FrameSubmission) -> Result<FrameLaunchOutcome> {
        bail!(UNSERVED_POOL)
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
