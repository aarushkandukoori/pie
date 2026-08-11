//! C-ABI shim embedding the Pie engine in an iOS app.
//!
//! Mirrors the boot path of `pie run` (server/src/cli/run_cmd.rs): load a
//! config TOML, boot a one-shot engine via `serve::start_engine`, drive one
//! inferlet over the local WebSocket with `pie-client`, collect its output,
//! and tear the engine down. The whole transcript comes back as a single
//! heap-allocated C string the caller must release with `pie_ios_free`.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;

use anyhow::{Context, Result};
use pie_client::client::{Client, ProcessEvent};
use pie_server::{config, serve};

/// Boot Pie, run one inferlet to completion, shut down, return transcript.
///
/// All arguments are NUL-terminated UTF-8 paths/strings. Returns a malloc'd
/// C string (transcript, or a message starting with "PIE ERROR:").
///
/// # Safety
/// All pointers must be valid NUL-terminated C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_ios_run(
    config_path: *const c_char,
    wasm_path: *const c_char,
    manifest_path: *const c_char,
    inferlet_id: *const c_char,
    input_json: *const c_char,
) -> *mut c_char {
    let arg = |p: *const c_char| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    let out = run_impl(
        &arg(config_path),
        &arg(wasm_path),
        &arg(manifest_path),
        &arg(inferlet_id),
        &arg(input_json),
    )
    .unwrap_or_else(|e| format!("PIE ERROR: {e:#}"));
    CString::new(out.replace('\0', ""))
        .expect("NULs stripped above")
        .into_raw()
}

/// Release a string returned by `pie_ios_run`.
///
/// # Safety
/// `s` must be a pointer previously returned by `pie_ios_run` (or null).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_ios_free(s: *mut c_char) {
    if !s.is_null() {
        drop(unsafe { CString::from_raw(s) });
    }
}

/// The engine boots once per process (global tracing subscriber and
/// driver channels are process-wide) and stays warm across runs — the
/// natural shape for an app embedding Pie anyway.
struct EngineGlobals {
    runtime: tokio::runtime::Runtime,
    url: String,
    token: String,
}

static ENGINE: std::sync::OnceLock<EngineGlobals> = std::sync::OnceLock::new();
static ENGINE_INIT: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn engine_globals(config_path: &str) -> Result<&'static EngineGlobals> {
    if let Some(g) = ENGINE.get() {
        return Ok(g);
    }
    let _guard = ENGINE_INIT.lock().expect("engine init lock");
    if let Some(g) = ENGINE.get() {
        return Ok(g);
    }

    let mut cfg = config::Config::from_toml_file(Path::new(config_path))
        .with_context(|| format!("loading config {config_path}"))?;
    // Port 0: let the OS pick, same as `pie run`'s one-shot engine.
    cfg.server.port = 0;

    let runtime = serve::build_runtime(&cfg)?;
    let engine = runtime
        .block_on(serve::start_engine(cfg))
        .context("start_engine")?;
    let url = engine.url.clone();
    let token = engine.token.clone();
    // Keep the engine (driver threads, server task) alive for the process
    // lifetime — iOS reclaims the process wholesale. EngineHandle holds
    // raw pointers and isn't Sync, so it can't live in the static.
    std::mem::forget(engine);

    let _ = ENGINE.set(EngineGlobals {
        runtime,
        url,
        token,
    });
    Ok(ENGINE.get().expect("just set"))
}

fn run_impl(
    config_path: &str,
    wasm: &str,
    manifest: &str,
    inferlet: &str,
    input: &str,
) -> Result<String> {
    let g = engine_globals(config_path)?;
    let mut log = format!("engine up at {}\n", g.url);
    let result = g.runtime.block_on(drive(
        &g.url, &g.token, wasm, manifest, inferlet, input, &mut log,
    ));
    match result {
        Ok(()) => Ok(log),
        Err(e) => {
            log.push_str(&format!("\nERROR: {e:#}\n"));
            Ok(log)
        }
    }
}

async fn drive(
    url: &str,
    token: &str,
    wasm: &str,
    manifest: &str,
    inferlet: &str,
    input: &str,
    log: &mut String,
) -> Result<()> {
    let client = Client::connect(url).await.context("ws connect")?;
    client.auth_by_token(token).await.context("auth")?;
    client
        .add_program(Path::new(wasm), Path::new(manifest), true)
        .await
        .context("add_program")?;
    log.push_str("inferlet installed\n");

    let mut process = client
        .launch_process(inferlet.to_string(), input.to_string(), true, None)
        .await
        .context("launch_process")?;
    log.push_str(&format!("launched (pid={})\n----\n", process.id()));

    loop {
        match process.recv().await.context("event recv")? {
            ProcessEvent::Stdout(s) | ProcessEvent::Stderr(s) => log.push_str(&s),
            ProcessEvent::Return(s) => {
                log.push_str(&format!("----\nreturn: {s}\n"));
                return Ok(());
            }
            ProcessEvent::Error(e) => anyhow::bail!("inferlet errored: {e}"),
            _ => {}
        }
    }
}
