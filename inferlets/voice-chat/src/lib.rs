//! One spoken turn of a voice conversation.
//!
//! The app calls this once per utterance. Everything that makes the turn
//! cheap lives here rather than in the client: the conversation's KV state
//! stays inside the engine under a named snapshot, so turn *n* prefills
//! only the new user message instead of replaying the whole transcript.
//! That is the difference between a phone assistant that answers in a
//! beat and one that thinks for ten seconds before opening its mouth.
//!
//! Contract with the client:
//!
//! - **stdout** carries speakable text only — reasoning blocks never
//!   reach it, so the caller can hand chunks straight to a speech
//!   synthesizer without hearing the model reason out loud.
//! - **the return value** is a JSON object with the full reply plus this
//!   turn's KV accounting, which keeps the numbers off the spoken channel.
//!
//! ```json
//! {"text": "...", "reused": 312, "new_prefill": 14, "generated": 41,
//!  "resumed": true, "note": ""}
//! ```

use std::io::{self, Write};

use inferlet::{Context, Result, chat, model::Model, runtime, sample::Sampler};
use serde::Deserialize;

/// A closed, empty reasoning block.
///
/// Qwen3-style chat templates prefill exactly this when thinking is
/// disabled, but Pie's `Context::cue()` fills only the generation header,
/// which leaves the decision to the model. Left to itself on a resumed
/// context it opens a `<think>` block and never closes it, burying the
/// entire reply inside a region that must not be spoken. Prefilling the
/// closed block puts generation outside any reasoning region from the
/// first token, which makes the turn deterministic and saves the tokens
/// the model would have spent thinking.
const CLOSED_REASONING_BLOCK: &str = "<think>\n\n</think>\n\n";

#[derive(Deserialize)]
struct Input {
    /// What the user just said.
    text: String,

    /// Named KV snapshot carrying this conversation across turns.
    #[serde(default = "default_session")]
    session: String,

    /// Start over: drop the snapshot before this turn.
    #[serde(default)]
    reset: bool,

    /// Applied only when the conversation starts.
    #[serde(default = "default_system")]
    system: String,

    #[serde(default = "default_max_tokens")]
    max_tokens: usize,

    #[serde(default = "default_temperature")]
    temperature: f32,

    #[serde(default = "default_top_p")]
    top_p: f32,

    /// Let the model reason before answering. Off by default: reasoning
    /// is latency the listener pays for and cannot hear.
    #[serde(default)]
    think: bool,
}

fn default_session() -> String {
    "voice-session".into()
}
fn default_max_tokens() -> usize {
    120
}
fn default_temperature() -> f32 {
    0.7
}
fn default_top_p() -> f32 {
    0.95
}
fn default_system() -> String {
    "You are a voice assistant. Your replies are read aloud, so keep them \
     to one or two short spoken sentences. Never use markdown, lists, code \
     blocks, or emoji. Write numbers and symbols the way a person would say \
     them."
        .into()
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let model_name = runtime::models()
        .first()
        .cloned()
        .ok_or("No models available")?;
    let model = Model::load(&model_name)?;

    if input.reset {
        let _ = Context::delete(&model, &input.session);
    }

    // `take` rather than `open` where possible: taking consumes the
    // snapshot, so a long conversation holds exactly one instead of
    // accumulating a fork per turn. `take` is contention-aware and fails
    // on drivers that report no spare pages, so `open` is the fallback —
    // the conversation matters more than the tidiness of the snapshot
    // table, and the name is freed before saving either way.
    let (mut ctx, resumed, note) = match Context::take(&model, &input.session) {
        Ok(ctx) => (ctx, true, String::new()),
        Err(take_error) => match Context::open(&model, &input.session) {
            Ok(ctx) => (
                ctx,
                true,
                format!("take failed, opened instead: {take_error}"),
            ),
            Err(_) => {
                let mut ctx = Context::new(&model)?;
                ctx.system(&input.system);
                (ctx, false, String::new())
            }
        },
    };

    let reused = ctx.seq_len();

    ctx.user(input.text.trim()).cue();
    if !input.think {
        ctx.append(&model.tokenizer().encode(CLOSED_REASONING_BLOCK));
    }
    let new_prefill = (ctx.seq_len() + ctx.buffer().len() as u32).saturating_sub(reused);

    // Scoped so the Generator's borrow of `ctx` ends before the snapshot
    // save below.
    let (raw, streamed, generated) = stream_turn(
        &mut ctx,
        &model,
        input.max_tokens,
        input.temperature,
        input.top_p,
    )
    .await?;

    // `Done` hands back the whole turn at once, so re-derive the spoken
    // text from the raw transcript when the stream ended that way.
    let mut spoken = streamed;
    let recovered = speakable_text(&raw);
    if recovered.trim() != spoken.trim() {
        spoken = recovered;
    }

    // Close the assistant turn. The generator truncates the stop token
    // rather than appending it, so without this the snapshot ends on an
    // unterminated assistant message and the next turn's user block gets
    // spliced straight onto it.
    ctx.seal();

    // Commit the working pages before snapshotting, or the reply this
    // turn just generated would be missing from the next one.
    ctx.flush().await?;
    // `save` refuses to overwrite, and the name is still taken whenever
    // this turn resumed via `open` rather than `take`. Dropping it first
    // makes the save unconditional instead of dependent on how the
    // context was obtained.
    let _ = Context::delete(&model, &input.session);
    ctx.save(&input.session)?;

    Ok(format!(
        "{{\"text\":{},\"reused\":{},\"new_prefill\":{},\"generated\":{},\"resumed\":{},\"note\":{}}}",
        json_string(spoken.trim()),
        reused,
        new_prefill,
        generated,
        resumed,
        json_string(&note)
    ))
}

/// Generates one assistant turn, streaming speakable text to stdout.
///
/// Returns `(raw transcript, streamed speakable text, tokens generated)`.
async fn stream_turn(
    ctx: &mut Context,
    model: &Model,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
) -> Result<(String, String, usize)> {
    let mut generator = ctx
        .generate(Sampler::TopP {
            temperature,
            p: top_p,
        })
        .max_tokens(max_tokens)
        .stop(&chat::stop_tokens(model));

    let mut decoder = chat::Decoder::new(model);
    let mut stripper = ReasoningStripper::new();
    let mut raw = String::new();
    let mut spoken = String::new();
    let mut generated = 0usize;

    while let Some(step) = generator.next()? {
        let out = step.execute().await?;
        if out.tokens.is_empty() {
            continue;
        }
        generated += out.tokens.len();

        match decoder.feed(&out.tokens)? {
            chat::Event::Delta(delta) => {
                raw.push_str(&delta);
                // Only speakable text reaches stdout, and it is flushed
                // per delta so the synthesizer starts on sentence one
                // instead of waiting for the turn to finish.
                let visible = stripper.process(&delta);
                if !visible.is_empty() {
                    spoken.push_str(&visible);
                    print!("{}", visible);
                    let _ = io::stdout().flush();
                }
            }
            chat::Event::Done(text) => {
                raw = text;
                break;
            }
            _ => {}
        }
    }

    // Release the holdback. Without this the last few characters of every
    // reply — held back in case they were the start of a `<think>` tag —
    // are never streamed, so the synthesizer clips the final word.
    let tail = stripper.flush();
    if !tail.is_empty() {
        spoken.push_str(&tail);
        print!("{}", tail);
        let _ = io::stdout().flush();
    }

    Ok((raw, spoken, generated))
}

/// The part of a raw assistant turn that should be said out loud.
///
/// With the closed block prefilled there is usually nothing to strip. The
/// two fallbacks matter anyway: a model that emits its own complete block
/// (everything after the last `</think>`), and — the case that actually
/// bites — one that opens a block and never closes it, where the reply is
/// the block's contents and dropping it would leave silence.
fn speakable_text(raw: &str) -> String {
    if let Some(index) = raw.rfind("</think>") {
        return raw[index + "</think>".len()..].to_string();
    }
    if let Some(index) = raw.find("<think>") {
        return raw[index + "<think>".len()..].to_string();
    }
    raw.to_string()
}

/// Minimal JSON string escaping — the inferlet has no serde_json and one
/// field does not justify pulling it in.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Largest char boundary at or below `idx`. Token deltas are arbitrary
/// UTF-8, so the holdback below cannot slice on raw byte offsets.
fn floor_boundary(s: &str, mut idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Removes `<think>…</think>` from a token stream without ever emitting a
/// partial tag: text is held back until it is certain no tag straddles the
/// chunk boundary.
struct ReasoningStripper {
    in_think: bool,
    pending: String,
}

impl ReasoningStripper {
    fn new() -> Self {
        Self {
            in_think: false,
            pending: String::new(),
        }
    }

    fn process(&mut self, delta: &str) -> String {
        self.pending.push_str(delta);
        let mut out = String::new();
        loop {
            if self.in_think {
                if let Some(idx) = self.pending.find("</think>") {
                    self.pending = self.pending.split_off(idx + "</think>".len());
                    self.in_think = false;
                    continue;
                }
                // Keep the tail that could be a partial "</think>".
                let cut = floor_boundary(&self.pending, self.pending.len().saturating_sub(7));
                if cut > 0 {
                    self.pending = self.pending.split_off(cut);
                }
                break;
            }
            if let Some(idx) = self.pending.find("<think>") {
                out.push_str(&self.pending[..idx]);
                self.pending = self.pending.split_off(idx + "<think>".len());
                self.in_think = true;
                continue;
            }
            // Same idea for a partial "<think>".
            let safe = floor_boundary(&self.pending, self.pending.len().saturating_sub(6));
            if safe > 0 {
                let head: String = self.pending.drain(..safe).collect();
                out.push_str(&head);
            }
            break;
        }
        out
    }

    /// Everything still held back at the end of the turn.
    ///
    /// Returns nothing while inside an unterminated block — that text is
    /// reasoning as far as this decoder can tell, and `speakable_text`
    /// recovers it from the raw transcript if it turns out to be a reply.
    fn flush(&mut self) -> String {
        if self.in_think {
            return String::new();
        }
        std::mem::take(&mut self.pending)
    }
}
