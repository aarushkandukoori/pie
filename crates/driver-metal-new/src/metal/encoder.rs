//! Encoding a step, committing it, and waiting for it with a bound.
//!
//! A step is one command buffer: an allocator reset, an encoder, some number
//! of dispatches, and a wait on a shared event. The shape is the C++ shell's
//! `encode_one_command_buffer` and `await_event`, with the two things that
//! were comments there made into types here.
//!
//! # The wait has a bound, and running out of it is terminal
//!
//! `waitUntilSignaledValue:` with no timeout is how the C++ shell spent
//! twenty-two minutes silent inside a bare retry loop. The bound here is
//! deliberately far past any real step -- the slowest measured on this
//! machine, a 192-token prefill through a 30B mixture, is about 200 ms -- so
//! reaching it does not mean "slow", it means the GPU is not coming back.
//!
//! What happens then is the part worth stating: nothing. A command buffer
//! that has not signalled may still be executing, so its allocator cannot be
//! reset and the heap it reads cannot be freed. There is no recovery, so
//! [`Stepper`] latches into a wedged state and refuses every later step
//! rather than papering over it with a retry.

use std::ptr::NonNull;
use std::time::Duration;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4CommandQueue, MTL4ComputeCommandEncoder, MTL4VisibilityOptions,
    MTLComputePipelineState, MTLDevice, MTLSharedEvent, MTLSize, MTLStages,
};

use super::context::{Context, describe};
use super::heap::Slot;
use super::tables::Tables;
use crate::error::{Error, Result};

/// How long one probe of the completion wait lasts.
///
/// Split into probes rather than one long timeout so that a step which is
/// merely slow can be COUNTED as slow the moment it passes the first probe,
/// while the total is what decides to give up.
const WAIT_PROBE: Duration = Duration::from_secs(5);

/// How many probes before the step is declared not coming back.
///
/// Sixty seconds total, two orders of magnitude past the slowest real step.
const WAIT_PROBES: u32 = 12;

/// What a dispatch's barrier makes visible to the next one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Visibility {
    /// Order execution only, without flushing caches.
    ///
    /// The default, and measured to be the correct one: the placement heap is
    /// L2-coherent within a single encoder on this UMA part, so the consumer
    /// of a producer's heap write sees it without an explicit flush. Both the
    /// device-flush and execution-only sweeps landed within noise of each
    /// other, so the cheaper one is not a gamble taken for speed.
    #[default]
    ExecutionOnly,
    /// Flush caches to the device coherence point.
    Device,
}

impl From<Visibility> for MTL4VisibilityOptions {
    fn from(v: Visibility) -> Self {
        match v {
            Visibility::ExecutionOnly => Self::None,
            Visibility::Device => Self::Device,
        }
    }
}

/// The buffer addresses one dispatch reads.
///
/// A table, not a list of `setBuffer:` calls: MTL4 binds by GPU address, and
/// an address outlives the encoder it was bound in. That is what lets a table
/// be built once, before any step, and reused by a byte-identical command
/// buffer every token -- which is the whole reason the encode cost of a step
/// is flat in the number of tokens.
pub struct ArgumentTable {
    table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    capacity: usize,
}

impl ArgumentTable {
    /// A table with room for `capacity` buffer bindings.
    pub fn new(context: &Context, capacity: usize) -> Result<Self> {
        let descriptor = MTL4ArgumentTableDescriptor::new();
        descriptor.setMaxBufferBindCount(capacity);
        let table = context
            .device()
            .newArgumentTableWithDescriptor_error(&descriptor)
            .map_err(|e| Error::Create {
                what: "MTL4ArgumentTable",
                message: describe(&e),
            })?;
        Ok(Self { table, capacity })
    }

    /// How many bindings this table holds.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bind `slot` at `index`.
    ///
    /// Out of range is an error rather than a silent no-op. Metal's own
    /// behaviour past `maxBufferBindCount` is not a diagnostic, and a binding
    /// that did not happen surfaces as a kernel reading address zero -- which
    /// on this driver means a kernel reading whatever the last step left.
    pub fn bind(&self, index: usize, slot: &Slot<'_>) -> Result<()> {
        self.bind_address(index, slot.gpu_address())
    }

    /// Bind a raw GPU address at `index`.
    pub fn bind_address(&self, index: usize, address: u64) -> Result<()> {
        if index >= self.capacity {
            return Err(Error::Create {
                what: "argument table binding",
                message: format!("index {index} past the table's {} bindings", self.capacity),
            });
        }
        // SAFETY: `address` is a GPU address obtained from a buffer that the
        // heap keeps alive, and `index` is within the bind count the table was
        // created with. Metal validates neither.
        unsafe { self.table.setAddress_atIndex(address, index) };
        Ok(())
    }
}

impl std::fmt::Debug for ArgumentTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArgumentTable")
            .field("capacity", &self.capacity)
            .finish_non_exhaustive()
    }
}

/// The per-dispatch surface, live only while a step is being encoded.
pub struct StepEncoder<'a> {
    encoder: &'a ProtocolObject<dyn MTL4ComputeCommandEncoder>,
    /// The thread limit of the pipeline currently set, or 0 if none is.
    max_threads: usize,
}

impl StepEncoder<'_> {
    /// Set the pipeline the next dispatch runs.
    pub fn set_pipeline(&mut self, pipeline: &ProtocolObject<dyn MTLComputePipelineState>) {
        self.encoder.setComputePipelineState(pipeline);
        self.max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    }

    /// Set the table the next dispatch reads its addresses from.
    pub fn set_argument_table(&mut self, table: &ArgumentTable) {
        self.encoder.setArgumentTable(Some(&table.table));
    }

    /// Set the table built for `ordinal`, or refuse.
    ///
    /// A miss is an error rather than a skipped call. Skipping it leaves the
    /// PREVIOUS dispatch's table bound, so the kernel runs to completion over
    /// another dispatch's buffers and the step reports success -- the same
    /// failure shape as a dispatch with no pipeline.
    pub fn set_argument_table_for(&mut self, tables: &Tables, ordinal: u32) -> Result<()> {
        self.set_argument_table(tables.expect(ordinal)?);
        Ok(())
    }

    /// Dispatch `threads`, in threadgroups of `threadgroup`.
    ///
    /// Refuses a threadgroup wider than the pipeline allows. Metal does not:
    /// the dispatch is simply not performed, its output keeps whatever it
    /// held, and the step reports success. That is how a model that answers
    /// nonsense passes every check -- which is exactly how it went unnoticed
    /// in the C++ shell, where this is a printf that fires once per pipeline.
    pub fn dispatch(&mut self, threads: [usize; 3], threadgroup: [usize; 3]) -> Result<()> {
        if self.max_threads == 0 {
            return Err(Error::Create {
                what: "dispatch",
                message: "no pipeline is set; the dispatch would run the previous kernel or none"
                    .to_string(),
            });
        }
        let per_group = threadgroup[0] * threadgroup[1] * threadgroup[2];
        if per_group == 0 {
            return Err(Error::Create {
                what: "dispatch",
                message: "a threadgroup of no threads runs nothing".to_string(),
            });
        }
        if per_group > self.max_threads {
            return Err(Error::Create {
                what: "dispatch",
                message: format!(
                    "{per_group} threads a threadgroup, and the pipeline allows {}; \
                     Metal would skip this dispatch and report success",
                    self.max_threads
                ),
            });
        }
        self.encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: threads[0],
                height: threads[1],
                depth: threads[2],
            },
            MTLSize {
                width: threadgroup[0],
                height: threadgroup[1],
                depth: threadgroup[2],
            },
        );
        Ok(())
    }

    /// Order the next dispatch after the ones already encoded.
    pub fn barrier(&mut self, visibility: Visibility) {
        self.encoder
            .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                MTLStages::Dispatch,
                MTLStages::Dispatch,
                visibility.into(),
            );
    }
}

/// Runs steps against a context, one at a time.
///
/// Synchronous: [`Stepper::run`] does not return until the GPU has signalled.
/// The allocator pair is still alternated, because the parity is what a
/// pipelined version needs and a synchronous one that ignores it would have to
/// grow the state back when it stops being synchronous.
pub struct Stepper<'ctx> {
    context: &'ctx Context,
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    /// The event value the last committed step signals.
    committed: u64,
    /// Set once a wait ran out. There is no way back; see the module docs.
    wedged: bool,
}

impl<'ctx> Stepper<'ctx> {
    /// Build a stepper for `context`.
    pub fn new(context: &'ctx Context) -> Result<Self> {
        let event = context.device().newSharedEvent().ok_or(Error::Create {
            what: "MTLSharedEvent",
            message: String::new(),
        })?;
        Ok(Self {
            context,
            event,
            committed: 0,
            wedged: false,
        })
    }

    /// How many steps have been committed.
    #[must_use]
    pub const fn steps(&self) -> u64 {
        self.committed
    }

    /// Whether a wait ran out, after which nothing more will run.
    #[must_use]
    pub const fn is_wedged(&self) -> bool {
        self.wedged
    }

    /// Encode one step, commit it, and wait for it.
    ///
    /// `encode` is handed the live encoder. Its error is returned as-is and
    /// the command buffer is still closed -- an encoder abandoned mid-step
    /// leaves Metal holding an open command buffer against an allocator this
    /// type is about to reset.
    pub fn run<F>(&mut self, encode: F) -> Result<()>
    where
        F: FnOnce(&mut StepEncoder<'_>) -> Result<()>,
    {
        if self.wedged {
            return Err(Error::Create {
                what: "step",
                message: "this context was abandoned after a completion wait ran out".to_string(),
            });
        }

        // Safe because this stepper is synchronous: the work drawn from this
        // allocator was waited for two steps ago. The parity is what makes
        // that sentence still true when the wait moves off the commit path.
        let allocator: &ProtocolObject<dyn MTL4CommandAllocator> =
            self.context.allocator(self.committed as usize);
        allocator.reset();

        let command_buffer = self
            .context
            .device()
            .newCommandBuffer()
            .ok_or(Error::Create {
                what: "MTL4CommandBuffer",
                message: String::new(),
            })?;
        command_buffer.beginCommandBufferWithAllocator(allocator);
        // The heap was made resident once; this is what tells THIS command
        // buffer to use that set. Without it every address the argument table
        // holds is a page the GPU has not been told to keep.
        command_buffer.useResidencySet(self.context.residency());

        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(Error::Create {
                what: "MTL4ComputeCommandEncoder",
                message: String::new(),
            })?;

        let mut step = StepEncoder {
            encoder: &encoder,
            max_threads: 0,
        };
        let encoded = encode(&mut step);

        // Closed on both paths. See the doc comment: an abandoned encoder
        // outlives the allocator reset that comes next.
        encoder.endEncoding();
        command_buffer.endCommandBuffer();
        encoded?;

        let value = self.committed + 1;
        let mut buffers = [NonNull::from(&*command_buffer)];
        // SAFETY: the pointer is to a live array of exactly one command
        // buffer, which outlives the call; `commit` takes it by reference and
        // does not retain the array.
        unsafe {
            self.context
                .queue()
                .commit_count(NonNull::from(&mut buffers).cast(), 1);
        }
        self.context
            .queue()
            .signalEvent_value(ProtocolObject::from_ref(&*self.event), value);
        self.committed = value;

        self.await_value(value)
    }

    /// Wait for the event to reach `value`, or wedge.
    fn await_value(&mut self, value: u64) -> Result<()> {
        let probe_ms = u64::try_from(WAIT_PROBE.as_millis()).unwrap_or(u64::MAX);
        for _ in 0..WAIT_PROBES {
            if self.event.waitUntilSignaledValue_timeoutMS(value, probe_ms) {
                return Ok(());
            }
        }
        self.wedged = true;
        Err(Error::Create {
            what: "step",
            message: format!(
                "the GPU did not reach event {value} within {} ms; this context is abandoned \
                 because its command buffers may still be running",
                probe_ms * u64::from(WAIT_PROBES)
            ),
        })
    }
}

impl std::fmt::Debug for Stepper<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stepper")
            .field("steps", &self.committed)
            .field("wedged", &self.wedged)
            .finish_non_exhaustive()
    }
}
