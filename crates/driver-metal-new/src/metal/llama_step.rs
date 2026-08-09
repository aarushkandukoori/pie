//! The llama M=1 step: bound once, fired per token.
//!
//! Three of the four bind passes are the SHARED ones — the kinds carry
//! their weight names in `weight_binds`, the attention reads the ring at
//! the plain SDPA's slots, and [`llama_decode_geometry`] puts every
//! layer on the attention path — so this step owns only the assembly
//! order and the family's consts walk, exactly as gpt-oss's does. The
//! llama3 frequency table's keepalive is optional where YaRN's was not:
//! a geometric-series checkpoint has no table, and `None` says so
//! rather than a dummy allocation pretending to be one.

use crate::batch::{
    Dispatch, LlamaGeometry, ScratchSchedule, build_llama_dag, llama_decode_geometry,
};
use crate::tuning::Tuning;
use crate::{Error, Result};

use super::bind::{ConstSlots, StepPsos, bind_decode_dag, bind_scratch, encode_decode_step};
use super::context::Context;
use super::encoder::Stepper;
use super::handle::Handle;
use super::llama_bind::bind_llama_consts;
use super::storage::DecodeStorage;
use super::tables::Tables;
use super::timing::Timing;

/// One bound llama decode step.
#[derive(Debug)]
pub struct LlamaStep {
    /// The dispatch list, golden-surface order.
    pub dag: Vec<Dispatch>,
    /// Per-ordinal argument tables.
    pub tables: Tables,
    /// The const-slot cache.
    pub consts: ConstSlots,
    /// The compiled pipelines (from `llama_step_plan`).
    pub psos: StepPsos,
    /// Whether every barrier is forced (the debug lever).
    pub force_barriers: bool,
    /// The llama3 frequency table the rope tables point into, when the
    /// geometry carries one.
    _freqs: Option<Handle>,
}

impl LlamaStep {
    /// Build the DAG, bind everything, and hold the result ready.
    ///
    /// # Errors
    ///
    /// Any bind refusal, or a scratch schedule that does not cover this
    /// DAG.
    pub fn prepare(
        context: &Context,
        storage: &DecodeStorage,
        g: &LlamaGeometry,
        tuning: &Tuning,
        schedule: &ScratchSchedule,
        psos: StepPsos,
        max_ctx: u32,
    ) -> Result<Self> {
        let dag = build_llama_dag(g, tuning, true);
        if schedule.per_dispatch.len() != dag.len() {
            return Err(Error::Create {
                what: "llama step",
                message: format!(
                    "the scratch schedule covers {} dispatches, the DAG has {}",
                    schedule.per_dispatch.len(),
                    dag.len()
                ),
            });
        }
        let shared = llama_decode_geometry(g);
        let mut tables = Tables::new();
        let mut consts = ConstSlots::new();
        bind_decode_dag(context, &mut tables, storage, &dag, &shared, false)?;
        bind_scratch(context, &mut tables, storage, schedule)?;
        let freqs = bind_llama_consts(
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
        Ok(LlamaStep {
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
    /// probe, as on the other family steps.
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
