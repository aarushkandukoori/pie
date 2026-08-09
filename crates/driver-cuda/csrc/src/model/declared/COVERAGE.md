# What the declared drive covers, and what the hand-written pass still serves

D3 says "delete the hand-written forward passes, family by family". This
file is what stopped that from being a deletion, and it is written from
the callers' own eligibility tests rather than from an estimate.

**The declared drive is not a replacement for the hand-written pass. It
is a replacement for a SUBSET of the fires that pass serves, and every
family's caller says so in as many words** — gemma-4's reads
"Eligibility is an ANSWER, not an error", and the shape is
`if (eligible && declared(...)) return; hand_pass(...);`.

So the hand pass is two things at once: the parity gate's reference, and
the FALLBACK. Deleting it removes both.

## What each family's declared drive refuses

Read off `*_model.cpp`'s `declared_eligible` and the executors' own
`return false` sites.

| family | the fire is refused when |
|---|---|
| gemma-4 | a custom mask, a stage hook, **any multimodal input (images, clips, precomputed embeddings)**, a lora adapter, a row-decode-shaped fire, `tp_size > 1`, or a deployment whose PLE buffers / cache format do not match |
| gpt-oss | a custom mask, a stage hook, `tp_size > 1`, or `routes > max_routes` (the fused MXFP4 leg's admission bound) |
| qwen3.5 | a fire with no class (legacy slot-less harness fires, live-fact mismatches), and the MoE arc unless `PIE_DECLARED_MOE` |
| llama_like | nine named `DeclineReason`s: `NoPlan` (TP, **quantized projections**, non-standard rope), `WriteDescMissing`, `SlidingWindow`, `PaddedHeadNarrowing`, `UnionPrefill`, `TruncatedAxisUnstated`, `FusedQkvUnstaged`, `BandedPlanMissing` |

Several of those are not fire properties that could be closed one at a
time. `NoPlan` is a DSL vocabulary question — a deployment the text
never traced. Multimodal gemma-4 has no declared statement at all.

## What deleting the hand pass would remove

Every fire in the table above would have nothing to run. In particular
gemma-4's vision and audio paths, every `tp_size > 1` deployment, and
llama_like's quantized-projection deployments.

## The other half: paths no gate reaches

Even inside what the declared drive DOES serve, four kinds of path are
exercised by no A/B here, so the hand pass is their only reference:

| path | why no gate reaches it |
|------|------------------------|
| qwen3.5's MoE leg | 35B-A3B is ~67G of bf16 against a 46G card |
| llama_like's post-norm branches | want olmo2; the gate runs qwen3-0.6b |
| llama_like's semantic rope and per-head norm | the gate's model states the FUSED `qk_rmsnorm_rope` instead |
| llama_like's hook sites and lora correction | the harness attaches neither — `hooked=0`, `lora=0` on all 52 fires |

`which_op_kinds_each_family_states` prints the first three from the
text; the fourth comes from `PIE_DECLARED_FORWARD_TRACE`.

## What D3 can actually be

Three things, in order, none of which is "delete the file":

1. **Close the declines that are closeable.** Each one names a piece of
   work — a statement the text does not carry, a prepare-side plan that
   is not stamped. That is where the 17k lines actually go.
2. **Give the unreachable paths a gate,** or record that they will not
   get one. A deployment nobody can load is not covered by a green run
   on a different one.
3. **Then delete, per family, at a commit whose gate run is named in
   the deletion commit** — because after it, the comparison cannot be
   made again.

Deleting first and verifying later is the one order that does not work
here: the thing being deleted is the verifier.
