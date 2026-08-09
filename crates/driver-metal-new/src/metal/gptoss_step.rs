//! The gpt-oss M=1 step: bound once, fired per token.
//!
//! Three of the four bind passes are the SHARED ones — the Go* kinds
//! carry their weight names in `weight_binds`, the sink attention reads
//! the ring at the plain SDPA's slots, and the all-full-attention view
//! ([`gptoss_decode_geometry`]) gives every layer its KV pair — so this
//! step owns only the assembly order and the family's consts walk. What
//! it must NOT forget is the YaRN table's keepalive: the rope dispatches
//! hold its GPU address in their argument tables, and a dropped handle is
//! a use-after-free the fault reporter attributes to an innocent rope.

use crate::batch::{
    Dispatch, GptOssGeometry, ScratchSchedule, build_gptoss_dag, gptoss_decode_geometry,
};
use crate::tuning::Tuning;
use crate::{Error, Result};

use super::bind::{ConstSlots, StepPsos, bind_decode_dag, bind_scratch, encode_decode_step};
use super::context::Context;
use super::encoder::Stepper;
use super::gptoss_bind::bind_gptoss_consts;
use super::handle::Handle;
use super::storage::DecodeStorage;
use super::tables::Tables;
use super::timing::Timing;

/// One bound gpt-oss decode step.
#[derive(Debug)]
pub struct GptOssStep {
    /// The dispatch list, golden-surface order.
    pub dag: Vec<Dispatch>,
    /// Per-ordinal argument tables.
    pub tables: Tables,
    /// The const-slot cache.
    pub consts: ConstSlots,
    /// The compiled pipelines (from `gptoss_step_plan`).
    pub psos: StepPsos,
    /// Whether every barrier is forced (the debug lever).
    pub force_barriers: bool,
    /// The YaRN table the rope tables point into.
    _freqs: Handle,
}

impl GptOssStep {
    /// Build the DAG, bind everything, and hold the result ready.
    ///
    /// The passes run in the shared step's order — weights/state/IO,
    /// scratch, constants — so a prepared step is never half-bound.
    ///
    /// # Errors
    ///
    /// Any bind refusal, or a scratch schedule that does not cover this
    /// DAG.
    pub fn prepare(
        context: &Context,
        storage: &DecodeStorage,
        g: &GptOssGeometry,
        tuning: &Tuning,
        schedule: &ScratchSchedule,
        psos: StepPsos,
        max_ctx: u32,
    ) -> Result<Self> {
        let dag = build_gptoss_dag(g, true);
        if schedule.per_dispatch.len() != dag.len() {
            return Err(Error::Create {
                what: "gpt-oss step",
                message: format!(
                    "the scratch schedule covers {} dispatches, the DAG has {}",
                    schedule.per_dispatch.len(),
                    dag.len()
                ),
            });
        }
        let shared = gptoss_decode_geometry(g);
        let mut tables = Tables::new();
        let mut consts = ConstSlots::new();
        bind_decode_dag(context, &mut tables, storage, &dag, &shared, false)?;
        bind_scratch(context, &mut tables, storage, schedule)?;
        let freqs = bind_gptoss_consts(
            context,
            &mut tables,
            &mut consts,
            &dag,
            g,
            tuning,
            max_ctx,
            1,
            0,
        )?;
        Ok(GptOssStep {
            dag,
            tables,
            consts,
            psos,
            force_barriers: false,
            _freqs: freqs,
        })
    }

    /// Encode and run the whole DAG as one command buffer.
    ///
    /// # Errors
    ///
    /// A kind with no compiled pipeline, or a command buffer that does
    /// not retire clean.
    pub fn fire(&self, stepper: &mut Stepper<'_>) -> Result<Timing> {
        self.fire_prefix(stepper, self.dag.len())
    }

    /// Encode and run only `[0, end)` of the DAG — the bisect's stage
    /// probe: with the ordinary recycled pool, the LAST dispatch's output
    /// slot still holds its value when the prefix retires, so a truncated
    /// fire reads any stage without the no-recycle allocation.
    ///
    /// # Errors
    ///
    /// As [`fire`](Self::fire).
    pub fn fire_prefix(&self, stepper: &mut Stepper<'_>, end: usize) -> Result<Timing> {
        stepper.run(|encoder| {
            encode_decode_step(
                encoder,
                &self.tables,
                &self.dag,
                &self.psos,
                self.force_barriers,
                0,
                end,
            )
        })
    }
}
