#pragma once

// THE SHARED SWITCH — one arm per symbol, for every symbol whose
// execution is the same wherever it is stated.
//
// There were four switches, one per family executor, and a census of
// their cases found most of each already family-blind: 13 of
// llama_like's 24, 17 of gemma-4's, 17 of gpt-oss's. "Family-blind"
// means the body mentions no workspace field, no weights struct and no
// config — what is left is the statement's operands, the rectangle's
// rows, and the fire's own inputs.
//
// Those are not four arms. They are one arm written four times, and
// where they differed they differed by ACCIDENT rather than by family:
// llama_like's `WriteKvToPages` takes a device-window form under
// hook-graph capture and gemma-4's does not, which is a property of the
// FIRE (is there a peel?) and not of the model. So the window joins the
// context and one body serves both.
//
// [`execute_shared`] returns false for a symbol it does not own, and the
// caller's own switch runs. That residue is what is left of a family
// executor, and it shrinks as arms land here — which is the measure
// this file exists to make, rather than a claim it exists to support.

#include <cstdint>
#include <stdexcept>

#include "attention_workspace.hpp"
#include "attn/head_dim_pad.hpp"
#include "attn/kv_paged.hpp"
#include "gemm/gemm.hpp"
#include "mlp/swiglu.hpp"
#include "rope/rope.hpp"
#include "store/kv_cache.hpp"
#include "model/declared/arms.hpp"
#include "model/declared/registry.hpp"
#include "model/declared/value_arena.hpp"
#include "model/declared/weights.hpp"

namespace pie_cuda_driver::model::declared {

// WHAT A SHARED ARM MAY READ, and nothing more.
//
// [`ArmCtx`] is the inner half — the plan, the arena, the rectangle.
// This adds what a LAUNCH needs beyond an arm: the fire's own inputs,
// the binder, and the handles a kernel is given rather than told.
//
// The head geometry is here for the reason `eps` is a parameter to
// `arm_rmsnorm`: it is config the trace does not carry, and a row width
// divided by a head count is a carving only once you already know one
// of them. What is NOT here is anything a family knows and another
// does not — the moment a field would be one family's, the arm it
// serves is not shared and belongs in that family's residue.
struct ExecCtx {
    // The inner half, verbatim.
    ArmCtx arm;

    const WeightBinder& wb;
    KvCache& cache;
    AttentionWorkspace& attn_ws;
    kernels::gemm::CublasHandle& cublas;

    // The fire's inputs.
    const std::int32_t* positions = nullptr;
    const std::uint32_t* qo_indptr = nullptr;
    const std::uint32_t* kv_page_indices = nullptr;
    const std::uint32_t* kv_page_indptr = nullptr;
    const std::uint32_t* kv_last_page_lens = nullptr;
    const std::uint8_t* row_valid = nullptr;
    // The explicit KV write's descriptors; null on a page-derived fire.
    const std::uint32_t* w_page_d = nullptr;
    const std::uint32_t* w_off_d = nullptr;
    int num_requests = 0;

    // The PEEL's device window, and which face this launch serves.
    // Null and false are the plain form, which is why one body covers
    // both: whether a fire has a peel is the fire's property.
    const std::uint32_t* peel_window_d = nullptr;
    bool peel_tail = false;

    // Config the trace does not carry.
    float eps = 0.f;
    float rope_theta = 0.f;
    int num_q_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    // The width the attention kernels run at; equal to `head_dim` where
    // nothing pads.
    int head_dim_kernel = 0;

    // The layer this launch names, from the statement's state mark.
    int layer = 0;
};

// Run `op`'s arm if this file owns its symbol.
//
// Returns false when the symbol needs the family's own half — which is
// an answer, not a failure: `resolve_kernel` already refused anything
// the registry does not know, so a false here means "stated, and this
// family executes it its own way".
inline bool execute_shared(const ExecCtx& c,
                           const pie_forward::PieForwardOp& op) {
    const auto& plan = c.arm.plan;
    auto& values = c.arm.values;
    const int N = c.arm.rows;
    const auto stream = c.arm.stream;
    const auto aux = plan.aux_names(op);
    const auto ins = plan.inputs(op);
    const auto outs = plan.outputs(op);

    // One weight, required, by the name the statement gave.
    const auto one_weight = [&](const char* what) -> const DeviceTensor& {
        if (aux.size != 1) {
            throw std::runtime_error(
                std::string("declared arm: a stated ") + what + " names " +
                std::to_string(aux.size) + " weights, wants 1");
        }
        return c.wb.require(plan.name(aux[0]));
    };

    switch (resolve_kernel(plan.weight_name(op))) {
    // ── the row norms ──────────────────────────────────────────────
    //
    // All four executors already called `arm_rmsnorm` here; the fold
    // comes from the SYMBOL the registry matched, which is the
    // difference between binding and choosing.
    case Kernel::RmsnormRow:
    case Kernel::RmsnormRowGemma:
        arm_rmsnorm(c.arm, op, one_weight("row norm").data(), c.eps,
                    resolve_kernel(plan.weight_name(op)) ==
                        Kernel::RmsnormRowGemma);
        return true;

    // ── the rotation ───────────────────────────────────────────────
    //
    // Which one is the symbol; the partial one's width rides the
    // statement's params. Rope rewrites q and k where they lie and the
    // `kernel!` rows state that alias, so the outputs name the buffers
    // their operands already sit in.
    //
    // A Q-ONLY site (gemma-4's) states ONE operand and one result, and
    // reaches the same launcher with `num_kv_heads = 0` — the arity is
    // the statement's, so nothing here decides it.
    case Kernel::RopeFull:
    case Kernel::RopePartial: {
        need(outs, 1, "rope outputs");
        const bool q_only = outs.size < 2;
        void* const rq = values.slot(outs[0]);
        void* const rk = q_only ? rq : values.slot(outs[1]);
        const int kv_heads = q_only ? 0 : c.num_kv_heads;
        if (resolve_kernel(plan.weight_name(op)) == Kernel::RopePartial) {
            const auto ps = plan.aux_params(op);
            if (ps.size < 1) {
                throw std::runtime_error(
                    "declared arm: a partial rotation states no rotary "
                    "width");
            }
            kernels::rope::rope_partial_bf16(
                rq, rk, c.positions, N, c.num_q_heads, kv_heads, c.head_dim,
                static_cast<int>(ps[0]), c.rope_theta, stream);
        } else {
            kernels::rope::rope_bf16(
                rq, rk, c.positions, N, c.num_q_heads, kv_heads, c.head_dim,
                c.rope_theta, stream);
        }
        return true;
    }

    // ── the head-dim staging ───────────────────────────────────────
    case Kernel::PadHeadDim:
    case Kernel::StripHeadDim: {
        need(ins, 1, "head-dim staging inputs");
        need(outs, 1, "head-dim staging outputs");
        const auto& rv = plan.value(outs[0]);
        if (rv.rank != 3) {
            throw std::runtime_error(
                "declared arm: a head-dim staging result states rank " +
                std::to_string(rv.rank) + ", wants [Tokens, heads, dim]");
        }
        const int heads = static_cast<int>(rv.dims[1].value);
        if (resolve_kernel(plan.weight_name(op)) == Kernel::PadHeadDim) {
            kernels::attn::pad_head_dim_bf16(
                values.slot(ins[0]), values.slot(outs[0]), N, heads,
                c.head_dim, c.head_dim_kernel, stream);
        } else {
            kernels::attn::strip_head_dim_bf16(
                values.slot(ins[0]), values.slot(outs[0]), N, heads,
                c.head_dim, c.head_dim_kernel, stream);
        }
        return true;
    }

    // ── the KV write ───────────────────────────────────────────────
    //
    // The device-window form is the PEEL's, and a peel is a property of
    // the fire: llama_like's hook-graph captures carry one and gemma-4's
    // fires do not, which is why one body serves both and the fork reads
    // the context rather than the family.
    case Kernel::WriteKvToPages: {
        need(ins, 2, "kv write inputs");
        auto kv_view = c.cache.layer_view(c.layer);
        if (c.peel_window_d != nullptr && c.peel_tail) {
            kernels::attn::write_kv_to_pages_bf16_devwin(
                kv_view, values.slot(ins[0]), values.slot(ins[1]),
                c.qo_indptr, c.kv_page_indices, c.kv_page_indptr,
                c.kv_last_page_lens, c.peel_window_d, N, c.num_requests,
                stream, c.row_valid);
            return true;
        }
        kernels::attn::write_kv_to_pages(
            kv_view, values.slot(ins[0]), values.slot(ins[1]), c.qo_indptr,
            c.kv_page_indices, c.kv_page_indptr, c.kv_last_page_lens, N,
            c.num_requests, stream, c.row_valid, /*first_token=*/0);
        return true;
    }

    // ── the MLP activation ─────────────────────────────────────────
    //
    // The chunked form reads one packed operand, the pair form two, and
    // each statement carries the operands it reads (2d).
    case Kernel::ChunkedSwiglu:
    case Kernel::Swiglu: {
        const bool pair =
            resolve_kernel(plan.weight_name(op)) == Kernel::Swiglu;
        need(ins, pair ? 2 : 1, "swiglu inputs");
        need(outs, 1, "swiglu outputs");
        void* const dst = values.slot(outs[0]);
        const int width = row_width(plan, outs[0]);
        if (pair) {
            kernels::mlp::swiglu_bf16(values.slot(ins[0]),
                                      values.slot(ins[1]), dst, N * width,
                                      stream);
        } else {
            kernels::mlp::chunked_swiglu_bf16(values.slot(ins[0]), dst, N,
                                              width, stream);
        }
        return true;
    }

    // ── the residual landing ───────────────────────────────────────
    case Kernel::ResidualAdd:
        arm_residual_add(c.arm, op);
        return true;

    // ── the collectives ────────────────────────────────────────────
    //
    // Whether the result is the operand's own bytes is the `kernel!`
    // row's alias pair, which the host honoured when it assigned
    // addresses — so both spellings are one call and the two slots are
    // simply equal or not.
    case Kernel::AllReduce:
    case Kernel::AllReduceOut:
    case Kernel::ResidualAddRmsnorm:
        return false;  // the communicator is the deployment's, not this
                       // file's; llama_like binds it. See its residue.

    // ── the weight representation axis ─────────────────────────────
    case Kernel::MatmulTensorScaled:
    case Kernel::MatmulChannelScaled:
    case Kernel::MatmulGroupedScaled:
    case Kernel::MatmulMxfp4Marlin: {
        if (aux.size < 2 || aux.size > 3) {
            throw std::runtime_error(
                "declared arm: a scaled projection names " +
                std::to_string(aux.size) +
                " weights, wants 2 (W, scales) or 3 (+ zeros)");
        }
        const auto matched = resolve_kernel(plan.weight_name(op));
        const ScaledRepr repr =
            matched == Kernel::MatmulTensorScaled  ? ScaledRepr::PerTensor
            : matched == Kernel::MatmulChannelScaled ? ScaledRepr::PerChannel
            : matched == Kernel::MatmulGroupedScaled ? ScaledRepr::PerGroup
                                                     : ScaledRepr::Mxfp4Marlin;
        arm_scaled_matmul(c.arm, op, repr, c.cublas.handle(),
                          c.wb.require(plan.name(aux[0])),
                          c.wb.require(plan.name(aux[1])),
                          aux.size == 3 ? &c.wb.require(plan.name(aux[2]))
                                        : nullptr,
                          // A quantized projection never folds its
                          // residual: `try_fold_residual` refuses a
                          // `Launch`, so the landing is a stated add.
                          0.f);
        return true;
    }

    default:
        return false;
    }
}

}  // namespace pie_cuda_driver::model::declared
