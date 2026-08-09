#pragma once

// SHARED OP ARMS — the executor's body, for the ops whose execution is
// already family-blind.
//
// The audit that started this merge found 13 of 23 op kinds present in
// both family executors, with the bodies differing only by OPERAND
// CONVENTION (which workspace buffer plays each role) and by the weights
// struct — never by arithmetic. Step 1 removed the weights difference (a
// binder); step 2 removed the walk's; this file is where an arm lands the
// moment its operands stop being a family's private convention.
//
// It starts with the arms that were ALREADY identical, character for
// character, in both executors — the strongest possible evidence that the
// executor wanted to be one file. The rest follow as the SSA value arena
// (the trace already carries `inputs`/`outputs`; what it does not carry is
// a buffer, because buffer assignment is a backend job that was written as
// family convention) replaces the routing conventions.

#include <cstdint>
#include <stdexcept>
#include <string>

#include "attn/split_packed.hpp"
#include "layout/embed.hpp"
#include "layout/gather_rows.hpp"
#include "norm/residual_add.hpp"
#include "norm/rmsnorm.hpp"
#include "mlp/swiglu.hpp"
#include "model/declared/value_arena.hpp"
#include "model/workspace.hpp"

namespace pie_cuda_driver::model::declared {

// `Swiglu`: the packed-bank form when the MLP's gate/up matmul landed in
// the fused bank, the two-buffer form otherwise. `gate_up_used_fused` is
// the Matmul arm's own decision, carried forward — the trace states ONE
// packed matmul either way (see the binder's `gate_up`).
//
// Both executors held this arm character-for-character identical.
// `dst` is the traced value's slot once the caller has moved this island
// onto the arena; a caller that has not keeps passing its convention.
inline void arm_swiglu(Workspace& ws,
                       bool gate_up_used_fused,
                       void* dst,
                       int n,
                       int intermediate,
                       cudaStream_t stream) {
    if (gate_up_used_fused) {
        kernels::mlp::chunked_swiglu_bf16(
            ws.gate_up_fused.data(), dst, n, intermediate, stream);
    } else {
        kernels::mlp::swiglu_bf16(
            ws.gate.data(), ws.up.data(), dst, n * intermediate, stream);
    }
}

// ── the arms that read their operands off the plan ─────────────────
//
// D1's shape, one arm at a time. An arm lands here the moment its body
// stops mentioning a workspace field: what is left is the statement's
// operands, its widths, and the fire's row count, none of which is a
// family's to know.
//
// The two guards below travel with them, because both were written per
// executor and both catch real defects. `need` refuses a short operand
// span -- indexing past one does not fault, it reads the NEXT
// statement's operands and hands the arm a plausible pointer to the
// wrong buffer. `row_width` is a value's trailing dims, which is what
// `Hq`, `Hk`, `I`, `H` and the rest were spelling.

// WHAT EVERY ARM TAKES, and nothing more.
//
// The five arms below had converged on the same first four parameters —
// the plan to read the statement from, the arena to resolve its
// operands in, the rectangle's row count, and the stream. That is the
// shape D1 is heading for: one context, the statement, and whatever the
// FAMILY has to add.
//
// `win_start` is here rather than in the one arm that reads it because
// it is the same kind of fact as `rows`: both describe the RECTANGLE a
// launch covers, and a driver that walked rectangles rather than ops
// would hand the pair to every arm without asking which cares.
//
// What is NOT here is the family's half — a weight pointer, `eps`, a
// vocabulary size, a quantization descriptor. Those stay explicit
// arguments, one per arm, because making them a bag would hide which
// arm needs which, and the whole point of the exercise is that the list
// gets shorter as the trace states more.
struct ArmCtx {
    const pie_forward::ForwardPlan& plan;
    ValueArena& values;
    /// Rows of the rectangle this launch covers.
    int rows;
    /// Its first row, in the fire's row space. Zero for the plain form.
    int win_start;
    cudaStream_t stream;
};

inline void need(const pie_forward::ForwardPlan::IdSpan& span,
                 std::size_t n, const char* what) {
    if (span.size < n) {
        throw std::runtime_error(
            std::string("declared arm: ") + what + " states " +
            std::to_string(span.size) + " operands, needs " +
            std::to_string(n));
    }
}

inline int row_width(const pie_forward::ForwardPlan& plan,
                     std::uint32_t id) {
    const auto& val = plan.value(id);
    std::uint32_t out = 1;
    for (std::uint32_t d = 1; d < val.rank; ++d) {
        if (val.dims[d].kind != pie_forward::PieForwardDimKind::Const) {
            return 0;
        }
        out *= val.dims[d].value;
    }
    return static_cast<int>(out);
}

// `SplitQkv`: one packed bank into three. Identical in gemma-4 and
// qwen3.5 once both read their operands off the plan; llama_like's is
// this plus a row WINDOW, which is the rectangle's and so stays a
// parameter rather than a second arm.
//
// `win_start` offsets each operand by whole rows -- the peel's tail
// splits the hook-visible rows at their absolute offsets, so the
// full-N consumers see one contiguous buffer. Zero is the plain form.
inline void arm_split_qkv(const ArmCtx& c,
                          const pie_forward::PieForwardOp& op) {
    const auto& plan = c.plan;
    auto& values = c.values;
    const int rows = c.rows;
    const int win_start = c.win_start;
    const auto stream = c.stream;
    const auto ins = plan.inputs(op);
    const auto outs = plan.outputs(op);
    need(ins, 1, "split_qkv inputs");
    need(outs, 3, "split_qkv outputs");
    const int q_w = row_width(plan, outs[0]);
    const int kv_w = row_width(plan, outs[1]);
    const auto row = [&](void* base, int width) -> void* {
        return static_cast<std::uint16_t*>(base) +
               static_cast<std::size_t>(win_start) *
                   static_cast<std::size_t>(width);
    };
    kernels::attn::split_qkv_bf16(
        row(values.slot(ins[0]), row_width(plan, ins[0])),
        row(values.slot(outs[0]), q_w),
        row(values.slot(outs[1]), kv_w),
        row(values.slot(outs[2]), kv_w),
        rows, q_w, kv_w, stream);
}

// `Embed`: the token table into the residual stream. All four executors
// hold this identically once their operands come off the plan — only
// the WEIGHT lookup differs, because each family binds its tensors
// through its own store, so the resolved pointer is a parameter.
//
// `token_ids` stays a driver input: the fire's tokens are not a traced
// value.
inline void arm_embed(const ArmCtx& c,
                      const pie_forward::PieForwardOp& op,
                      const std::int32_t* token_ids,
                      const void* table,
                      int vocab) {
    const auto& plan = c.plan;
    const auto outs = plan.outputs(op);
    need(outs, 1, "embed outputs");
    kernels::layout::embed_bf16(token_ids, table, c.values.slot(outs[0]),
                                c.rows, row_width(plan, outs[0]), vocab,
                                c.stream);
}

// `residual_add`: `x += residual`, landing on operand 0 — the `kernel!`
// row aliases the result over it, so the destination is the OUTPUT's
// slot and the addend is operand 1. Both spellings that reach here
// (llama_like's post-norm landing, gpt-oss's MoE landing) are this.
inline void arm_residual_add(const ArmCtx& c,
                             const pie_forward::PieForwardOp& op) {
    const auto& plan = c.plan;
    auto& values = c.values;
    const auto ins = plan.inputs(op);
    const auto outs = plan.outputs(op);
    need(ins, 2, "residual add inputs");
    need(outs, 1, "residual add outputs");
    kernels::norm::residual_add_bf16(
        values.slot(outs[0]), values.slot(ins[1]),
        static_cast<std::size_t>(c.rows) *
            static_cast<std::size_t>(row_width(plan, outs[0])),
        c.stream);
}

// `Rmsnorm`: the row norm, with the WEIGHT FOLD chosen by the variant
// the statement carries. Gemma folds `(1 + w)` instead of `w` — different
// arithmetic, so a different kernel, but the same signature and the same
// row space, and the variant rides on the wire (`op.param0`).
//
// That is what makes this arm family-blind rather than nearly so: the
// fork is a fact of the STATEMENT, not of the executor, and three of the
// four had hard-coded their deployment's answer to it.
//
// `eps` stays a parameter. It is a config number the trace does not
// carry, which is the same reason the weight pointer is one.
//
// So does `gemma_fold`, and that is the migration showing: a SEMANTIC
// `Rmsnorm` makes its caller read the variant off `op.param0`, while a
// text that states `norm::rmsnorm_gemma_bf16` makes its caller pass
// what the registry already matched. Same arm, and only one of the two
// callers is choosing.
inline void arm_rmsnorm(const ArmCtx& c,
                        const pie_forward::PieForwardOp& op,
                        const void* weight,
                        float eps,
                        bool gemma_fold) {
    const auto& plan = c.plan;
    auto& values = c.values;
    const int rows = c.rows;
    const auto stream = c.stream;
    const auto ins = plan.inputs(op);
    const auto outs = plan.outputs(op);
    need(ins, 1, "rmsnorm inputs");
    need(outs, 1, "rmsnorm outputs");
    const int width = row_width(plan, ins[0]);
    if (gemma_fold) {
        kernels::norm::rmsnorm_gemma_bf16(values.slot(ins[0]), weight,
                                          values.slot(outs[0]), rows, width,
                                          eps, stream);
    } else {
        kernels::norm::rmsnorm_bf16(values.slot(ins[0]), weight,
                                    values.slot(outs[0]), rows, width, eps,
                                    stream);
    }
}

// The EPILOGUE's compaction, which is the half of `LmHead` every
// executor spells the same way.
//
// A fire whose sampled rows are a strict subset gathers them before the
// projection; anything else multiplies every row. The gather's
// destination belongs to the LOWERING, not to a workspace and not to a
// traced value — see `ValueArena::epilogue_gather` — and the caller
// passes it because only the caller knows whether its executor built
// `flat` before the arms or after.
//
// Returns the activation the projection should read and writes the row
// count through `rows`. The GEMM itself stays with the caller: three of
// the four resolve their head weight differently enough (a name, a
// bound tensor, a quantized view) that passing the result back is
// clearer than passing the resolver in.
inline const void* arm_epilogue_gather(const ArmCtx& c,
                                       const pie_forward::PieForwardOp& op,
                                       void* gathered,
                                       const std::int32_t* logit_row_indices,
                                       int num_logit_rows,
                                       int* rows) {
    const auto& plan = c.plan;
    auto& values = c.values;
    const auto stream = c.stream;
    const auto ins = plan.inputs(op);
    need(ins, 1, "lm_head inputs");
    const void* input = values.slot(ins[0]);
    if (logit_row_indices == nullptr || num_logit_rows <= 0 ||
        num_logit_rows >= *rows) {
        return input;
    }
    if (gathered == nullptr) {
        throw std::runtime_error(
            "declared arm: the epilogue compacts rows but the lowering "
            "reserved no scratch for it");
    }
    kernels::layout::gather_bf16_rows(
        static_cast<const std::uint16_t*>(input), logit_row_indices,
        static_cast<std::uint16_t*>(gathered), num_logit_rows,
        row_width(plan, ins[0]), stream);
    *rows = num_logit_rows;
    return gathered;
}

}  // namespace pie_cuda_driver::model::declared
