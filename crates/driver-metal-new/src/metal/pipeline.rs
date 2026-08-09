//! Turning `.metal` text into a compute pipeline state.
//!
//! Runtime compilation, not a prebuilt `.metallib`. The box this driver was
//! developed on has CommandLineTools and no Xcode, so there is no offline
//! `metal` compiler to produce one -- and the AOT path in the C++ shell's
//! CMake is gated off by default for exactly that reason. Handing
//! `newLibraryWithSource:` a string is the path that is always available.
//!
//! # Compilation is serialised, and not by choice
//!
//! Two threads compiling at once corrupt the process heap. This is not a
//! race in this crate: the trap lands inside `libsystem_malloc`, reached
//! from `_MTLCreateComputePipelineScriptFromDescriptor` by way of
//! `-[NSTaggedPointerString UTF8String]`, on a libdispatch queue owned by
//! `AGXG13XFamilyCompiler`. Nothing of ours is on that stack.
//!
//! It was found by bisection and is reproducible on demand. Eight threads
//! each compiling one trivial kernel and then doing nothing but host-side
//! `malloc` traffic abort in three runs out of six; the same eight threads
//! with the compile behind a mutex abort in none. The malloc traffic is the
//! part that misleads -- the corruption happens during the compile, but a
//! process that exits promptly afterwards never touches the damaged freelist
//! and looks healthy. That is why a test binary crashes in whichever test
//! happens to allocate next rather than in the one that compiled, and why
//! `--test-threads=1` "fixes" it.
//!
//! So [`GATE`] holds every pipeline creation to one at a time, process-wide
//! -- one lock for all compilers, because the damaged heap is the process's,
//! not any one compiler's. The cost is throughput at load time only: the
//! serving path compiles nothing. If a future OS makes this safe, the
//! evidence to re-test with is above.
//!
//! # The language version is a property of the driver
//!
//! `MTLCompileOptions` defaults to an older MSL standard. Under that default
//! `<metal_tensor>` and the MetalPerformancePrimitives tensor ops are simply
//! not visible, so a kernel that uses them fails to compile with an error
//! about an unknown identifier rather than about a dialect. The C++ shell
//! learned to pin `MTLLanguageVersion4_0` in ONE place after setting it at
//! some call sites and not others; [`Compiler::compile`] is that place here,
//! and there is no way to ask for a pipeline that skips it.

use std::path::Path;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSString, NSURL};
use objc2_metal::{
    MTL4Compiler, MTL4CompilerDescriptor, MTL4CompilerTaskOptions,
    MTL4ComputePipelineDescriptor, MTL4LibraryFunctionDescriptor,
    MTL4PipelineDataSetSerializer, MTL4PipelineDataSetSerializerConfiguration,
    MTL4PipelineDataSetSerializerDescriptor, MTLCompileOptions, MTLComputePipelineState, MTLDevice,
    MTLLanguageVersion, MTLLibrary,
};

use super::archive::{Archives, MAX_AGE};
use super::context::{Context, describe};
use crate::error::{Error, Result};
use crate::shader::{Batch, Request};

/// The MSL dialect every kernel in this driver is compiled as.
///
/// See the module docs: this is not a default, and a kernel compiled without
/// it fails in a way that does not mention the dialect.
const LANGUAGE_VERSION: MTLLanguageVersion = MTLLanguageVersion::Version4_0;

/// Held across every compilation in the process. See the module docs.
///
/// A plain `Mutex<()>`, and deliberately not a `OnceLock`-guarded compiler
/// singleton: the thing that must not overlap is the compilation, not the
/// ownership of a compiler. Poisoning is ignored, because a panic while
/// compiling leaves Metal's heap no worse than it already is and refusing
/// every later compile would turn one failure into a dead process.
static GATE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The runtime shader compiler.
///
/// One per context. Kept out of [`Context`] because it is not needed to
/// encode a step -- a driver that loaded prebuilt pipelines would have a
/// context and no compiler -- and because a type that owns a device object
/// and also compiles text is two types.
pub struct Compiler {
    compiler: Retained<ProtocolObject<dyn MTL4Compiler>>,
    /// Collects the binaries of everything this compiler builds.
    ///
    /// Attached at creation and never detached, because it cannot be: the
    /// serializer is a property of the compiler descriptor, so a compiler
    /// that might later want to write an archive has to have been collecting
    /// from its first pipeline. Collection is cheap; the write is not, and
    /// the write is explicit.
    serializer: Retained<ProtocolObject<dyn MTL4PipelineDataSetSerializer>>,
    /// Where archives are looked up and written.
    archives: Archives,
}

impl Compiler {
    /// Create the compiler for `context`'s device.
    pub fn new(context: &Context) -> Result<Self> {
        Self::with_archives(context, Archives::discover())
    }

    /// [`Compiler::new`] against an explicit cache location.
    ///
    /// Tests use this so that they neither read nor write the developer's own
    /// cache, and a caller that wants no cache passes `Archives::new(None)`.
    pub fn with_archives(context: &Context, archives: Archives) -> Result<Self> {
        let serializer_descriptor = MTL4PipelineDataSetSerializerDescriptor::new();
        // Binaries, not descriptors. A descriptor archive is what the offline
        // generator consumes; what a cold start needs back is the compiled
        // code, and asking for the wrong one produces an archive that loads
        // and then accelerates nothing.
        serializer_descriptor
            .setConfiguration(MTL4PipelineDataSetSerializerConfiguration::CaptureBinaries);
        let serializer = context
            .device()
            .newPipelineDataSetSerializerWithDescriptor(&serializer_descriptor);

        let descriptor = MTL4CompilerDescriptor::new();
        descriptor.setPipelineDataSetSerializer(Some(&serializer));
        let compiler = context
            .device()
            .newCompilerWithDescriptor_error(&descriptor)
            .map_err(|e| Error::Create {
                what: "MTL4Compiler",
                message: describe(&e),
            })?;
        Ok(Self {
            compiler,
            serializer,
            archives,
        })
    }

    /// Where this compiler caches compiled pipelines.
    #[must_use]
    pub fn archives(&self) -> &Archives {
        &self.archives
    }

    /// Compile `source` and build the pipeline for its `function` entry point.
    ///
    /// Three failures, kept apart because they have three different remedies:
    /// the source did not compile, it compiled but exports no such entry
    /// point, or the pipeline itself was rejected. The middle one is the one
    /// worth separating -- a misspelled entry point otherwise arrives as
    /// Metal's own message about a nil function, which names neither the
    /// spelling that was asked for nor the ones that exist.
    ///
    /// No archive. One source compiled on its own has no batch to be keyed
    /// as, and keying it by itself would fill the cache with an archive per
    /// call. [`Compiler::compile_batch`] is the cached path.
    pub fn compile(
        &self,
        context: &Context,
        source: &str,
        function: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        // Held across both halves. The library build is part of the same
        // compilation and there is no evidence separating the two.
        let _gate = GATE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let library = build_library(context, source, function)?;
        self.build_pipeline(&library, function, None)
    }

    /// Build every pipeline in `requests`, reusing libraries and the archive.
    ///
    /// Positional: one result per request, in order, and a request that fails
    /// fails alone. A load asking for thirty kernels should not lose the
    /// twenty-nine that built because one source has a typo -- and the caller
    /// needs to know WHICH one, which a single `Result` for the batch cannot
    /// say.
    ///
    /// Two things make this faster than the same calls one at a time. Each
    /// distinct file becomes an `MTLLibrary` once even when several entry
    /// points share it, and on a second run the pipelines are fetched from
    /// the archive named by [`Batch::key`] instead of being built at all.
    ///
    /// The archive is written only on a miss, and only when every request
    /// built. A partial archive would be served back on the next run as if it
    /// were the whole batch, and the requests missing from it would be
    /// compiled -- silently, so the cost would never be attributed.
    ///
    /// Compilation is serial, and that is not this crate's choice twice over:
    /// see the module docs for the heap corruption that forces the gate, and
    /// the C++ shell's own note that driving Metal's compiler service from
    /// extra threads measured no faster.
    pub fn compile_batch(&self, context: &Context, requests: &[Request]) -> Compiled {
        if requests.is_empty() {
            return Compiled {
                pipelines: Vec::new(),
                archive: Archived::Skipped,
            };
        }
        let batch = Batch::load(requests);
        let path = self.archives.path(batch.key(self.salt(context)));

        let _gate = GATE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        // A hit means every pipeline below is fetched rather than built. A
        // miss leaves `lookup` empty and the archive is written at the end.
        let lookup = path.as_deref().and_then(|path| self.lookup(context, path));

        // Stage one: one library per distinct source file.
        let libraries: Vec<Result<Retained<ProtocolObject<dyn MTLLibrary>>>> = (0..batch
            .paths()
            .len())
            .map(|index| match batch.source(index) {
                Some(Ok(source)) => build_library(context, source, ""),
                Some(Err(error)) => Err(clone_read_error(error, &batch.paths()[index])),
                None => unreachable!("index came from paths()"),
            })
            .collect();

        // Stage two: every pipeline off those libraries.
        let built: Vec<_> = (0..batch.len())
            .map(|index| {
                let (library, function) = batch.request(index).expect("index came from len()");
                match &libraries[library] {
                    Ok(library) => self.build_pipeline(library, function, lookup.as_deref()),
                    Err(error) => Err(clone_error(error)),
                }
            })
            .collect();

        let archive = match (lookup.is_some(), path.as_deref()) {
            (true, _) => Archived::Hit,
            (false, None) => Archived::Disabled,
            // A partial archive would be served back on the next run as if it
            // were the whole batch. Skipping the write costs one slow start;
            // writing it costs every start after this one.
            (false, Some(_)) if !built.iter().all(Result::is_ok) => Archived::Skipped,
            (false, Some(path)) => match self.write(path) {
                Ok(()) => Archived::Written,
                Err(error) => Archived::Failed(error),
            },
        };
        Compiled {
            pipelines: built,
            archive,
        }
    }

    /// Fetch the archive at `path`, as compiler task options that look in it.
    ///
    /// `None` covers both "not there" and "there but unusable". They are the
    /// same to the caller: compile, and be prepared to write. An unusable
    /// archive is not reported, because the only thing a caller could do with
    /// the report is what happens anyway.
    fn lookup(&self, context: &Context, path: &Path) -> Option<Retained<MTL4CompilerTaskOptions>> {
        if !path.exists() {
            return None;
        }
        let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
        let archive = context.device().newArchiveWithURL_error(&url).ok()?;
        let options = MTL4CompilerTaskOptions::new();
        options.setLookupArchives(Some(&NSArray::from_retained_slice(&[archive])));
        Some(options)
    }

    /// Write everything compiled so far to `path`.
    ///
    /// The pruning is here rather than at startup because this is the only
    /// moment the directory is known to have just grown.
    fn write(&self, path: &Path) -> Result<()> {
        let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
        self.serializer
            .serializeAsArchiveAndFlushToURL_error(&url)
            .map_err(|error| Error::Create {
                what: "pipeline archive",
                message: format!("writing {}: {}", path.display(), describe(&error)),
            })?;
        self.archives.prune(MAX_AGE);
        Ok(())
    }

    /// What the key must include that the sources do not say.
    ///
    /// The GPU, because a binary compiled for one is not valid on another and
    /// a cache directory can be shared over a network home. The language
    /// version, because the same text compiled as two dialects is two
    /// different sets of binaries.
    fn salt(&self, context: &Context) -> u64 {
        context.device().registryID() ^ (LANGUAGE_VERSION.0 as u64).rotate_left(32)
    }

    /// Build one pipeline off `library`, looking in `task` if there is one.
    fn build_pipeline(
        &self,
        library: &ProtocolObject<dyn MTLLibrary>,
        function: &str,
        task: Option<&MTL4CompilerTaskOptions>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        // Asked before the descriptor is built, so the message can list what
        // the library DOES export. After the pipeline call it is too late:
        // the library is still in hand but the error is already Metal's.
        let exported: Vec<String> = library
            .functionNames()
            .iter()
            .map(|name| name.to_string())
            .collect();
        if !exported.iter().any(|name| name == function) {
            return Err(Error::Compile {
                function: function.to_string(),
                message: format!(
                    "the source compiled but exports no such entry point; it exports [{}]",
                    exported.join(", ")
                ),
            });
        }

        let name = NSString::from_str(function);
        let function_descriptor = MTL4LibraryFunctionDescriptor::new();
        function_descriptor.setName(Some(&name));
        function_descriptor.setLibrary(Some(library));

        let pipeline_descriptor = MTL4ComputePipelineDescriptor::new();
        pipeline_descriptor.setComputeFunctionDescriptor(Some(&function_descriptor));
        // The entry point's name, carried on the pipeline. Per-dispatch
        // tracing has nothing else to report: a pipeline is an opaque object
        // and the DAG ordinal names a position rather than a kernel.
        pipeline_descriptor.setLabel(Some(&name));

        self.compiler
            .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(
                &pipeline_descriptor,
                task,
            )
            .map_err(|e| Error::Compile {
                function: function.to_string(),
                message: describe(&e),
            })
    }
}

/// What [`Compiler::compile_batch`] built, and what it did about the cache.
#[derive(Debug)]
pub struct Compiled {
    /// One result per request, positionally.
    pub pipelines: Vec<Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
    /// What became of the archive for this batch.
    pub archive: Archived,
}

impl Compiled {
    /// The pipelines, if every request built.
    ///
    /// For the caller that wants a batch to be all or nothing. The first
    /// failure is returned and the rest are dropped, which is why the field
    /// is public: a loader reporting which kernels are broken wants them all.
    pub fn all(self) -> Result<Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>> {
        self.pipelines.into_iter().collect()
    }
}

/// What one batch did about its archive.
///
/// Returned rather than logged. A cache that never writes and a cache that is
/// always hit look identical from outside -- both compile nothing on the
/// second run only if the first one worked -- so the difference has to be a
/// value the caller can assert on. The C++ shell reaches for the same thing
/// with a `bool* cache_hit` out-parameter on one of its four compile
/// functions.
#[derive(Debug)]
pub enum Archived {
    /// The archive was found and every pipeline came out of it.
    Hit,
    /// It was not there; the batch was compiled and the archive written.
    Written,
    /// There is no cache directory. See [`Archives`].
    Disabled,
    /// Nothing was written, because not every request built. A partial
    /// archive is worse than none: it would be served back as complete.
    Skipped,
    /// The write was attempted and refused. The pipelines are still built --
    /// what is lost is that the next run will be slow too.
    Failed(Error),
}

impl Archived {
    /// Whether the pipelines came out of an archive rather than a compiler.
    #[must_use]
    pub fn is_hit(&self) -> bool {
        matches!(self, Self::Hit)
    }
}

/// Turn one source into a library, pinned to this driver's dialect.
///
/// `function` only names the failure. A library is a translation unit and a
/// batch builds one for several entry points, so there is not always a single
/// function to blame -- an empty name says so rather than picking one.
fn build_library(
    context: &Context,
    source: &str,
    function: &str,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(LANGUAGE_VERSION);
    context
        .device()
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
        .map_err(|e| Error::Compile {
            function: function.to_string(),
            message: describe(&e),
        })
}

/// Restate a read failure that several requests have to share.
///
/// `std::io::Error` is not `Clone` and the batch has one error for a file
/// that any number of requests named. Restating it keeps the path and the
/// text, which is everything a caller reads out of it.
fn clone_read_error(error: &Error, path: &Path) -> Error {
    Error::Compile {
        function: String::new(),
        message: format!("{}: {error}", path.display()),
    }
}

/// The same, for a library failure shared by several requests.
fn clone_error(error: &Error) -> Error {
    Error::Compile {
        function: String::new(),
        message: error.to_string(),
    }
}

impl std::fmt::Debug for Compiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compiler").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRIVIAL: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void fill_ones(device float* out [[buffer(0)]],
                      uint gid [[thread_position_in_grid]]) {
    out[gid] = 1.0f;
}
";

    fn compiler() -> Option<(Context, Compiler)> {
        let context = match Context::new() {
            Ok(c) => c,
            Err(Error::NoDevice) => return None,
            Err(e) => panic!("context: {e}"),
        };
        let compiler = Compiler::new(&context).expect("compiler");
        Some((context, compiler))
    }

    #[test]
    fn a_trivial_kernel_compiles() {
        let Some((context, compiler)) = compiler() else {
            return;
        };
        let pso = compiler
            .compile(&context, TRIVIAL, "fill_ones")
            .expect("compiles");
        assert!(
            pso.maxTotalThreadsPerThreadgroup() > 0,
            "a real pipeline reports a threadgroup limit"
        );
    }

    /// The dialect is pinned by the driver, so a source that names an MSL 4.0
    /// header compiles without the caller asking for anything. Under the
    /// default standard this fails at the include.
    #[test]
    fn the_pinned_dialect_reaches_the_msl_4_headers() {
        let Some((context, compiler)) = compiler() else {
            return;
        };
        let source = r"
#include <metal_stdlib>
#include <metal_tensor>
using namespace metal;
kernel void touch(device float* out [[buffer(0)]],
                  uint gid [[thread_position_in_grid]]) {
    out[gid] = 0.0f;
}
";
        compiler
            .compile(&context, source, "touch")
            .expect("<metal_tensor> is visible only under MSL 4.0");
    }

    #[test]
    fn a_syntax_error_names_the_function_and_says_what_metal_said() {
        let Some((context, compiler)) = compiler() else {
            return;
        };
        let err = compiler
            .compile(&context, "kernel void broken( {", "broken")
            .expect_err("that is not MSL");
        match err {
            Error::Compile { function, message } => {
                assert_eq!(function, "broken");
                assert!(!message.is_empty(), "Metal's diagnostic is not dropped");
            }
            other => panic!("expected Compile, got {other}"),
        }
    }

    /// The failure this variant exists for: the source is fine and the name is
    /// not. Metal reports it as a nil function, naming neither the spelling
    /// asked for nor the ones that exist.
    #[test]
    fn a_missing_entry_point_lists_the_ones_that_exist() {
        let Some((context, compiler)) = compiler() else {
            return;
        };
        let err = compiler
            .compile(&context, TRIVIAL, "fill_zeroes")
            .expect_err("no such entry point");
        match err {
            Error::Compile { function, message } => {
                assert_eq!(function, "fill_zeroes");
                assert!(
                    message.contains("fill_ones"),
                    "the message must list what does exist: {message}"
                );
            }
            other => panic!("expected Compile, got {other}"),
        }
    }

    /// A library exports its KERNELS, not its functions. Asking for a plain
    /// helper is therefore the missing-entry-point path rather than a
    /// pipeline failure, and it is answered before Metal is asked.
    #[test]
    fn a_plain_function_is_not_an_entry_point() {
        let Some((context, compiler)) = compiler() else {
            return;
        };
        let source = r"
#include <metal_stdlib>
using namespace metal;
float helper(float x) { return x * 2.0f; }
kernel void real(device float* out [[buffer(0)]],
                 uint gid [[thread_position_in_grid]]) {
    out[gid] = helper(1.0f);
}
";
        let err = compiler
            .compile(&context, source, "helper")
            .expect_err("a helper is not an entry point");
        match err {
            Error::Compile { message, .. } => assert!(
                message.contains("real"),
                "the message lists the kernel that IS exported: {message}"
            ),
            other => panic!("expected Compile, got {other}"),
        }
    }
}
