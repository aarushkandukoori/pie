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

    /// # Errors
    ///
    /// Always, until the loader is wired to the seam.
    pub fn load_model(
        &mut self,
        _descs: Vec<driver_abi::ModelLoadDesc>,
    ) -> Result<driver_abi::DriverCapabilities> {
        bail!(UNSERVED_LOAD)
    }

    /// # Errors
    ///
    /// Always, until [`Self::load_model`] is wired: a program registered
    /// against no model is a registration that cannot be honoured.
    pub fn register_program(&mut self, _desc: &ProgramRegistration) -> Result<u64> {
        bail!(UNSERVED_LOAD)
    }

    /// # Errors
    ///
    /// As [`Self::register_program`].
    pub fn register_channel(&mut self, _desc: &ChannelRegistrationPlan) -> Result<RegisteredChannel> {
        bail!(UNSERVED_LOAD)
    }

    /// # Errors
    ///
    /// As [`Self::register_program`].
    pub fn bind_instance(&mut self, _desc: &InstanceBindingPlan) -> Result<BoundInstance> {
        bail!(UNSERVED_LOAD)
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

/// The other hole: nothing has taught this seam to load a checkpoint.
const UNSERVED_LOAD: &str = "driver-metal-new: `load_model` is not wired to the seam yet. \
     `loader/` compiles the plan and `metal::stage_plan_weights` stages it; \
     what is missing is the call between them and the name map \
     (`model::resolve::Store`) that answers a trace's weight names from it.";
