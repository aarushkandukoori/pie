#include "model/gemma4/declared_forward.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "kernels/gather_rows.hpp"
#include "kernels/residual_add.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/scalar_mul.hpp"
#include "kernels/softcap.hpp"
#include "kernels/split_packed.hpp"
#include "kernels/swiglu.hpp"
#include "kernels/embed.hpp"
#include "kernels/kv_paged.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/attention_naive_paged.hpp"
#include "ops/gemm.hpp"
#include "model/declared/value_arena.hpp"
#include <string>
#include <string_view>

namespace pie_cuda_driver::model {

namespace {

// The launcher registry — every kernel a gemma-4 class trace may STATE,
// one enum value per symbol. Deliberately EXHAUSTIVE against the traced
// decode plan: `gemma4_validate_stated_kernels` walks the plan at load
// and a symbol outside this list is a model-load failure, so this list
// and `family::gemma4_cuda` are two spellings of one vocabulary.
enum class G4Kernel {
    QkvPackedPost,
    QkRmsnormRopeRounded,
    RopeQOnly,
    RopeQOnlyPartial,
    RmsnormNoScale,
    WriteKvToPages,
    AttnFlashinferDecode,
    AttnFlashinferPrefill,
    AttnNaivePaged,
    GegluTanh,
    ChunkedGegluTanh,
    NormResidualScaleNorm,
    NormResidualAdd,
    ScalarMul,
    TransposeNldToLnd,
    LogitSoftcap,
    ResidualAdd,
    // The `attn.qv` seam's construct. In the registry because the trace
    // STATES it -- the seam is real and its position rule is checked --
    // and a symbol the registry does not know refuses the whole plan at
    // load. Its arm throws: gemma-4 has no adapter support on either
    // side, hand-written or declared, so a fire that reached it would be
    // a lowering bug. `row.lora` is hard 0 below and arc 2 declines lora
    // fires, which is what keeps it unreachable.
    LoraQkvCorrection,
};

G4Kernel resolve_g4_kernel(std::string_view k) {
    if (k == "launch_qkv_packed_qk_norm_rope_vnorm_write_kv_bf16")
        return G4Kernel::QkvPackedPost;
    if (k == "launch_qk_rmsnorm_rope_bf16_rounded")
        return G4Kernel::QkRmsnormRopeRounded;
    if (k == "launch_rope_bf16") return G4Kernel::RopeQOnly;
    if (k == "launch_rope_partial_bf16") return G4Kernel::RopeQOnlyPartial;
    if (k == "launch_rmsnorm_no_scale_bf16") return G4Kernel::RmsnormNoScale;
    if (k == "launch_write_kv_to_pages") return G4Kernel::WriteKvToPages;
    if (k == "dispatch_attention_flashinfer_decode")
        return G4Kernel::AttnFlashinferDecode;
    // gemma-4's prefill fires the PLAN-FREE wrapper, not the dispatch
    // llama_like states — one call apart in C++, a whole contract apart
    // in the declaration, so the symbols differ and this registry has
    // only the one gemma-4 actually says.
    if (k == "ops::launch_attention_flashinfer_prefill")
        return G4Kernel::AttnFlashinferPrefill;
    if (k == "ops::launch_attention_naive_paged")
        return G4Kernel::AttnNaivePaged;
    if (k == "launch_geglu_tanh_bf16") return G4Kernel::GegluTanh;
    if (k == "launch_chunked_geglu_tanh_bf16") return G4Kernel::ChunkedGegluTanh;
    if (k == "launch_rmsnorm_residual_add_scale_rmsnorm_bf16")
        return G4Kernel::NormResidualScaleNorm;
    if (k == "launch_rmsnorm_residual_add_bf16") return G4Kernel::NormResidualAdd;
    if (k == "launch_scalar_mul_bf16") return G4Kernel::ScalarMul;
    if (k == "launch_transpose_bf16_nld_to_lnd")
        return G4Kernel::TransposeNldToLnd;
    if (k == "launch_logit_softcap_bf16") return G4Kernel::LogitSoftcap;
    if (k == "launch_residual_add_bf16") return G4Kernel::ResidualAdd;
    if (k == "pie_lora_qkv_correction") return G4Kernel::LoraQkvCorrection;
    throw std::runtime_error(
        "declared gemma4: stated kernel '" + std::string(k) +
        "' is not in this executor's registry (the trace and the driver "
        "drifted)");
}

}  // namespace

void gemma4_validate_stated_kernels(const pie_forward::ForwardPlan& plan) {
    const std::size_t n = plan.op_count();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& op = plan.op(i);
        if (op.kind != pie_forward::PieForwardOpKind::Launch) continue;
        (void)resolve_g4_kernel(plan.weight_name(op));
    }
}

}  // namespace pie_cuda_driver::model

namespace pie_cuda_driver::model {

namespace {

using pie_forward::PieForwardOp;
using pie_forward::PieForwardOpKind;

[[noreturn]] void throw_drift(const std::string& what) {
    throw std::runtime_error("declared gemma4: " + what +
                             " (the trace and the driver drifted)");
}

// A plan weight name split into layer and field — llama_like's parse.
struct ParsedName {
    int layer = -1;
    std::string_view field;
};

ParsedName parse_name(std::string_view name) {
    constexpr std::string_view prefix = "layer.";
    if (name.substr(0, prefix.size()) != prefix) return {-1, name};
    const std::size_t dot = name.find('.', prefix.size());
    if (dot == std::string_view::npos) throw_drift("weight name '" +
                                                   std::string(name) + "'");
    int layer = 0;
    for (std::size_t i = prefix.size(); i < dot; ++i) {
        if (name[i] < '0' || name[i] > '9') {
            throw_drift("weight name '" + std::string(name) + "'");
        }
        layer = layer * 10 + (name[i] - '0');
    }
    return {layer, name.substr(dot + 1)};
}

// The binder. gemma-4's trace names its weights after the driver's own
// fields, so this is a map and not a translation.
const DeviceTensor* bind(const Gemma4Weights& w, std::string_view name) {
    const ParsedName nm = parse_name(name);
    if (nm.layer < 0) {
        if (nm.field == "embed") return w.embed;
        if (nm.field == "embed_per_layer") return w.embed_per_layer;
        if (nm.field == "ple_model_proj") return w.ple_model_proj;
        if (nm.field == "ple_model_norm") return w.ple_model_norm;
        if (nm.field == "final_norm") return w.final_norm;
        if (nm.field == "lm_head") return w.lm_head;
        throw_drift("unknown model weight '" + std::string(name) + "'");
    }
    if (nm.layer >= static_cast<int>(w.layers.size())) {
        throw_drift("weight names layer " + std::to_string(nm.layer));
    }
    const Gemma4LayerWeights& l = w.layers[static_cast<std::size_t>(nm.layer)];
    if (nm.field == "attn_norm") return l.attn_norm_pre;
    if (nm.field == "post_attn_norm") return l.attn_norm_post;
    if (nm.field == "pre_ffw_norm") return l.mlp_norm_pre;
    if (nm.field == "post_ffw_norm") return l.mlp_norm_post;
    if (nm.field == "qkv") return l.qkv_proj_fused;
    if (nm.field == "q_proj") return l.q_proj;
    if (nm.field == "k_proj") return l.k_proj;
    if (nm.field == "v_proj") return l.v_proj;
    if (nm.field == "o_proj") return l.o_proj;
    if (nm.field == "q_norm") return l.q_norm;
    if (nm.field == "k_norm") return l.k_norm;
    if (nm.field == "gate_up") return l.gate_up_proj_fused;
    // The unfused pair, for a deployment without the packed bank (E2B).
    if (nm.field == "gate_proj") return l.gate_proj;
    if (nm.field == "up_proj") return l.up_proj;
    if (nm.field == "down") return l.down_proj;
    if (nm.field == "ple_gate") return l.ple_input_gate;
    if (nm.field == "ple_proj") return l.ple_projection;
    if (nm.field == "ple_norm") return l.ple_norm;
    throw_drift("unknown layer weight '" + std::string(name) + "'");
}

const DeviceTensor& require(const Gemma4Weights& w, std::string_view name) {
    const DeviceTensor* t = bind(w, name);
    if (t == nullptr) {
        throw std::runtime_error("declared gemma4: weight '" +
                                 std::string(name) +
                                 "' is named by the trace but not bound");
    }
    return *t;
}

}  // namespace

std::string gemma4_validate_stated_weights(
    const pie_forward::ForwardPlan& plan, const Gemma4Weights& w) {
    // The name-resolution DRY WALK, qwen3_5's precedent. Without it an
    // unbound weight is discovered by the first fire and takes the whole
    // MODEL LOAD down; with it the plan simply declines and the
    // hand-written pass runs. That difference is what makes arming this
    // drive by default safe on a geometry nobody has booted yet — E2B
    // needed exactly this treatment three times.
    const auto resolves = [&](std::string_view name) {
        if (name.empty()) return true;
        // Names the executor does NOT resolve to a tensor: `scale.*` is a
        // CONSTANT riding in the weight slot so the arm can tell four
        // identical launches apart.
        if (name.rfind("scale.", 0) == 0) return true;
        try {
            return bind(w, name) != nullptr;
        } catch (const std::exception&) {
            return false;
        }
    };
    for (std::size_t i = 0; i < plan.op_count(); ++i) {
        const auto& op = plan.op(i);
        // On a LAUNCH op the weight slot holds the KERNEL SYMBOL, not a
        // weight — the arms read `aux_names` for the weights. Checking
        // the symbol here declined every deployment for a bogus reason,
        // which a fault-injection run caught: the drive was silently off
        // and the parity gate still said PASS, because both sides were
        // then the hand-written pass.
        const std::string_view name =
            op.kind == pie_forward::PieForwardOpKind::Launch
                ? std::string_view{}
                : plan.weight_name(op);
        if (!resolves(name)) {
            return "weight '" + std::string(name) + "' unresolvable";
        }
        if (op.kind == pie_forward::PieForwardOpKind::Launch) {
            const auto aux = plan.aux_names(op);
            for (std::size_t j = 0; j < aux.size; ++j) {
                const std::string_view a = plan.name(aux[j]);
                if (!resolves(a)) {
                    return "weight '" + std::string(a) + "' unresolvable";
                }
            }
        }
    }
    return {};
}

bool gemma4_forward_declared(
    const Gemma4DeclaredPlan& declared,
    const Gemma4Weights& w,
    const HfConfig& cfg,
    const Gemma4ForwardCfg& fwd_cfg,
    Workspace& ws,
    Gemma4MoeMlpWorkspace& moe_ws,
    KvCache& cache,
    AttentionWorkspace& attn_ws,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    bool is_pure_decode,
    const std::uint8_t* row_valid_d,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows)
{
    if (!declared.usable) return false;
    // WHICH CLASS. `use_decode_path` is the hand-written pass's own test
    // and this mirrors it, `force_prefill_path` included — a deployment
    // forced onto the prefill kernels must reach the PREFILL class here
    // or the drive would fire a decode dispatch the hand pass never
    // would.
    const bool decode_class = is_pure_decode && !fwd_cfg.force_prefill_path;
    const pie_forward::ForwardPlan& plan =
        decode_class ? declared.decode : declared.prefill;
    if (!decode_class && (qo_indptr_h == nullptr || kv_page_indptr_h == nullptr)) {
        // The prefill class's two dispatches both read host indptrs.
        return false;
    }
    // Say ONCE per class that this drive took a fire of it. Without this
    // line "the output is coherent" is evidence about the hand-written
    // pass as easily as about this one — an eligibility gate that
    // silently answers false looks exactly like a drive that works.
    {
        static std::atomic<bool> said[2] = {{false}, {false}};
        const std::size_t slot = decode_class ? 0 : 1;
        if (!said[slot].exchange(true)) {
            std::fprintf(stderr,
                         "[declared-gemma4] first %s fire: N=%d R=%d ops=%zu\n",
                         decode_class ? "DECODE" : "PREFILL",
                         total_tokens, num_requests, plan.op_count());
        }
    }

    const int N = total_tokens;
    const int R = num_requests;
    const int H = cfg.hidden_size;
    // The NARROW MLP width. A double-wide deployment (E2B) doubles it
    // on the KV-shared layers, so every MLP arm reads the layer's own
    // width — `w.per_layer_intermediate` is what the binder measured off
    // the gate_proj tensor, and it is the same number the trace baked in.
    const int I = cfg.intermediate_size;
    const int V = cfg.vocab_size;
    const int L = cfg.num_hidden_layers;
    const int ple_dim = cfg.gemma_hidden_size_per_layer_input;
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = cublas.stream();

    // The PLE relay's two buffers. Without them the prologue has nowhere
    // to land, and the hand-written pass allocates on the fly — a shape
    // this drive does not reproduce, so it declines instead.
    if (moe_ws.ple_token.empty() || moe_ws.ple_proj.empty()) return false;
    if (!cache.format().is_native_bf16()) return false;

    // The value the previous statement produced, by slot. gemma-4's
    // buffers are the hand-written pass's, so the drive threads them the
    // way that pass does rather than allocating an arena: `ws.y` is the
    // residual stream, `ws.norm_x` the block scratch.
    void* per_layer_token = moe_ws.ple_token.data();
    void* per_layer_proj = moe_ws.ple_proj.data();
    int lm_head_rows = N;

    // Layer state the arms need, refreshed as the drive walks into a
    // layer. The op's own `layer` field carries it, so nothing here
    // counts.
    int cur_layer = -1;
    bool cur_full = false;
    bool cur_shared = false;
    int cur_d = 0;
    int cur_hq = 0;
    int cur_hk = 0;
    int cur_inter = 0;
    const auto enter = [&](int l) {
        if (l < 0 || l == cur_layer) return;
        cur_layer = l;
        cur_full = w.layers[static_cast<std::size_t>(l)].is_full;
        cur_shared = w.layers[static_cast<std::size_t>(l)].is_shared;
        cur_d = w.per_layer_head_dim[static_cast<std::size_t>(l)];
        cur_hq = cfg.num_attention_heads * cur_d;
        cur_hk = w.per_layer_num_kv_heads[static_cast<std::size_t>(l)] * cur_d;
        cur_inter =
            (static_cast<std::size_t>(l) < w.per_layer_intermediate.size())
                ? w.per_layer_intermediate[static_cast<std::size_t>(l)]
                : I;
    };

    // The FULL layers' partial-rotary width, the driver's derivation.
    const auto rotary_of = [&](int l) {
        const float f =
            w.per_layer_partial_rotary_factor[static_cast<std::size_t>(l)];
        const int d = w.per_layer_head_dim[static_cast<std::size_t>(l)];
        return std::max(2, 2 * static_cast<int>(0.5f * f * d));
    };

    const auto gemm = [&](const void* act, std::string_view weight, void* out,
                          int m, int n, int k, float beta) {
        ops::gemm_act_x_wt_bf16(cublas.handle(), act, require(w, weight).data(),
                                out, m, n, k, beta);
    };

    // Declared HERE so the arms can capture it; filled after the
    // lowering exists, which is the only thing that has to come first.
    declared::ValueArena values;

    // A traced value's ROW WIDTH: the product of every dim but the
    // leading one, which is the row axis. This is the number the arms
    // used to carry as `cur_hq`, `cur_hk`, `cur_inter`, `L * ple_dim`
    // and `H` — per-layer bookkeeping the executor maintained beside a
    // declaration that already said it.
    //
    // Returns 0 when a dim after the first is not a constant, which
    // happens: a rank-3 value whose middle axis is `Tokens`
    // (`[N, L, ple_dim]`) has no fixed row width at all. No arm below
    // asks for one of those, and the ones that would are gated in
    // `model/tests/arena_soundness.rs` by name.
    // An operand span is a VIEW into the plan's flat id array, so
    // indexing past its end reads the next statement's operands and
    // hands an arm a plausible pointer to the wrong buffer. Every arm
    // that takes a fixed arity states it here instead.
    const auto need = [&](const auto& span, std::size_t n, const char* what) {
        if (span.size < n) {
            throw std::runtime_error(
                std::string("declared gemma4: ") + what + " states " +
                std::to_string(span.size) + " operands, needs " +
                std::to_string(n));
        }
    };

    const auto row_width = [&](std::uint32_t id) {
        const auto& val = plan.value(id);
        std::uint32_t out = 1;
        for (std::uint32_t d = 1; d < val.rank; ++d) {
            if (val.dims[d].kind != pie_forward::PieForwardDimKind::Const) {
                return 0;
            }
            out *= val.dims[d].value;
        }
        return static_cast<int>(out);
    };

    const auto execute_op = [&](const PieForwardOp& op) {
        enter(op.layer);
        switch (op.kind) {
        case PieForwardOpKind::Embed: {
            // ISLAND (value arena). Two sites differing only in WHERE
            // the rows land and how wide they are -- both the trace's.
            // `token_ids` stays a driver input: it is the fire's, not a
            // traced value.
            const std::string_view name = plan.weight_name(op);
            const auto outs = plan.outputs(op);
            need(outs, 1, "embed outputs");
            kernels::launch_embed_bf16(token_ids, require(w, name).data(),
                                       values.slot(outs[0]), N,
                                       row_width(outs[0]), V, stream);
            break;
        }
        case PieForwardOpKind::Matmul: {
            // ISLAND (value arena). Twelve branches told apart by the
            // WEIGHT NAME used to sit here, and every one of them chose
            // buffers and widths the trace already states: the operands
            // are `op.inputs[0]` and `op.outputs[0]`, and a GEMM's two
            // extents are those values' row widths. Reading the name to
            // rediscover them was the family convention doing work the
            // declaration had already done.
            //
            // The widths come off the value descriptors rather than the
            // `cur_*` per-layer bookkeeping, which is the same number by
            // a shorter route — a traced value's trailing dims ARE its
            // row width, and for these statements that is `cur_hq`,
            // `2 * cur_inter`, `L * ple_dim` and the rest, per layer,
            // without the executor tracking any of it.
            //
            // The `throw_drift` on an unrecognised field goes with them,
            // and is not a guard lost: it fired when the DECLARATION
            // named a matmul this arm had no placement for, and there is
            // no placement left to lack. A weight that does not exist
            // still refuses, one line down, where `gemm` requires it.
            const std::string_view name = plan.weight_name(op);
            const auto ins = plan.inputs(op);
            const auto outs = plan.outputs(op);
            need(ins, 1, "matmul inputs");
            need(outs, 1, "matmul outputs");
            gemm(values.slot(ins[0]), name, values.slot(outs[0]),
                 N, row_width(outs[0]), row_width(ins[0]), 0.f);
            break;
        }
        case PieForwardOpKind::Rmsnorm: {
            // ISLAND (value arena). Both sites — layer 0's `attn_norm`
            // (every later layer's input norm arrives fused into the
            // previous layer's PLE landing) and the epilogue's
            // `final_norm` — ran the SAME call with the same buffers,
            // told apart only to be treated identically. The operands
            // and the width are the trace's.
            const std::string_view name = plan.weight_name(op);
            const auto ins = plan.inputs(op);
            const auto outs = plan.outputs(op);
            need(ins, 1, "rmsnorm inputs");
            need(outs, 1, "rmsnorm outputs");
            kernels::launch_rmsnorm_bf16(
                values.slot(ins[0]), require(w, name).data(),
                values.slot(outs[0]), N, row_width(ins[0]), eps, stream);
            break;
        }
        case PieForwardOpKind::RmsnormPerHead: {
            // ISLAND (value arena), and three branches become one.
            //
            // They differed in nothing but their extents, and the
            // extents were re-derived per site from config: `N * L` by
            // `ple_dim`, `N * heads` by `cur_d`, `N * (hk / d)` by
            // `cur_d`. All three are the same statement — split the
            // value's row into HEAD-WIDE rows — and the head width is
            // `param0`, which the op has carried the whole time.
            const std::string_view name = plan.weight_name(op);
            const auto ins = plan.inputs(op);
            const auto outs = plan.outputs(op);
            need(ins, 1, "per-head norm inputs");
            need(outs, 1, "per-head norm outputs");
            const int head = static_cast<int>(op.param0);
            if (head <= 0) {
                throw_drift("per-head norm on '" + std::string(name) +
                            "' states no head width");
            }
            kernels::launch_rmsnorm_bf16(
                values.slot(ins[0]), require(w, name).data(),
                values.slot(outs[0]), N * (row_width(ins[0]) / head), head,
                eps, stream);
            break;
        }
        case PieForwardOpKind::SplitQkv: {
            const auto ins = plan.inputs(op);
            const auto outs = plan.outputs(op);
            need(ins, 1, "split_qkv inputs");
            need(outs, 3, "split_qkv outputs");
            kernels::launch_split_qkv_bf16(
                values.slot(ins[0]), values.slot(outs[0]),
                values.slot(outs[1]), values.slot(outs[2]),
                N, row_width(outs[0]), row_width(outs[1]), stream);
            break;
        }
        case PieForwardOpKind::Rope: {
            // Only the FULL layers reach the semantic kind: sliding
            // layers state the fused rounded pair instead.
            //
            // The head COUNTS stay read from config: rope needs the
            // rotation's head geometry, and a value's row width divided
            // by the head dim is that geometry only once you know the
            // head dim, which this op does not state. The BUFFERS are
            // the trace's, which is the half that was convention.
            const auto outs = plan.outputs(op);
            need(outs, 2, "rope outputs");
            kernels::launch_rope_partial_bf16(
                values.slot(outs[0]), values.slot(outs[1]), positions, N,
                cfg.num_attention_heads, cur_hk / cur_d, cur_d,
                static_cast<int>(op.param1),
                w.per_layer_rope_theta[static_cast<std::size_t>(cur_layer)],
                stream);
            break;
        }
        case PieForwardOpKind::LmHead: {
            const std::string_view name = plan.weight_name(op);
            const void* input = ws.norm_x.data();
            int rows = N;
            if (logit_row_indices_d != nullptr && num_logit_rows > 0 &&
                num_logit_rows < N) {
                kernels::launch_gather_bf16_rows(
                    static_cast<const std::uint16_t*>(ws.norm_x.data()),
                    logit_row_indices_d,
                    static_cast<std::uint16_t*>(ws.norm_y.data()),
                    num_logit_rows, H, stream);
                input = ws.norm_y.data();
                rows = num_logit_rows;
            }
            lm_head_rows = rows;
            gemm(input, name, ws.logits.data(), rows, V, H, 0.f);
            break;
        }
        case PieForwardOpKind::Launch: {
            const std::string_view sym = plan.weight_name(op);
            const auto names = plan.aux_names(op);
            const auto aux = [&](std::size_t i) {
                return plan.name(names[i]);
            };
            switch (resolve_g4_kernel(sym)) {
            case G4Kernel::ScalarMul: {
                const std::string_view which = aux(0);
                // ISLAND (value arena). Four sites that named four
                // buffers and four element counts to apply one scalar.
                // The buffer and the count are the value's; only the
                // SCALAR is a declared fact, and it stays read by name.
                const auto outs = plan.outputs(op);
                need(outs, 1, "scalar_mul outputs");
                float by;
                if (which == "scale.sqrt_hidden") {
                    by = std::sqrt(static_cast<float>(H));
                } else if (which == "scale.sqrt_ple_dim") {
                    by = std::sqrt(static_cast<float>(ple_dim));
                } else if (which == "scale.rsqrt_hidden") {
                    by = 1.0f / std::sqrt(static_cast<float>(H));
                } else if (which == "scale.rsqrt_2") {
                    by = 1.0f / std::sqrt(2.0f);
                } else {
                    throw_drift("scale '" + std::string(which) + "'");
                }
                kernels::launch_scalar_mul_bf16(
                    values.slot(outs[0]), by,
                    static_cast<std::size_t>(N) * row_width(outs[0]), stream);
                break;
            }
            case G4Kernel::LoraQkvCorrection:
                // Unreachable by construction: the `HasLora` guard's
                // then-region needs a lora row, and gemma-4 states none.
                // Loud rather than silent -- an adapter dropped without a
                // word is the failure this whole arc exists to prevent.
                throw std::runtime_error(
                    "declared gemma4: lora correction reached, but gemma-4 "
                    "has no adapter support on either side (arc 2 should "
                    "have declined this fire)");
            case G4Kernel::ResidualAdd:
                kernels::launch_residual_add_bf16(
                    per_layer_proj, per_layer_token,
                    static_cast<std::size_t>(N) * L * ple_dim, stream);
                break;
            case G4Kernel::TransposeNldToLnd:
                kernels::launch_transpose_bf16_nld_to_lnd(
                    static_cast<const std::uint16_t*>(per_layer_proj),
                    static_cast<std::uint16_t*>(per_layer_token),
                    N, L, ple_dim, stream);
                break;
            case G4Kernel::QkvPackedPost: {
                auto kv_view = cache.layer_view(cur_layer);
                kernels::launch_qkv_packed_qk_norm_rope_vnorm_write_kv_bf16(
                    ws.qkv_fused.data(), ws.q.data(),
                    kv_view.k_pages, kv_view.v_pages,
                    require(w, aux(0)).data(), require(w, aux(1)).data(),
                    positions, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, row_valid_d, N,
                    cfg.num_attention_heads, cur_hk / cur_d, cur_d,
                    cache.page_size(), kv_view.hnd_layout,
                    w.per_layer_rope_theta[static_cast<std::size_t>(cur_layer)],
                    eps, stream);
                break;
            }
            case G4Kernel::QkRmsnormRopeRounded: {
                const bool q_only = names.size == 1;
                kernels::launch_qk_rmsnorm_rope_bf16_rounded(
                    ws.q.data(), ws.k.data(), require(w, aux(0)).data(),
                    q_only ? nullptr : require(w, aux(1)).data(),
                    positions, N, cfg.num_attention_heads,
                    q_only ? 0 : cur_hk / cur_d, cur_d,
                    w.per_layer_rope_theta[static_cast<std::size_t>(cur_layer)],
                    eps, stream);
                break;
            }
            case G4Kernel::RopeQOnlyPartial:
                kernels::launch_rope_partial_bf16(
                    ws.q.data(), ws.q.data(), positions, N,
                    cfg.num_attention_heads, /*num_kv_heads=*/0, cur_d,
                    rotary_of(cur_layer),
                    w.per_layer_rope_theta[static_cast<std::size_t>(cur_layer)],
                    stream);
                break;
            case G4Kernel::RmsnormNoScale:
                kernels::launch_rmsnorm_no_scale_bf16(
                    ws.v.data(), ws.v.data(), N * (cur_hk / cur_d), cur_d,
                    eps, stream);
                break;
            case G4Kernel::WriteKvToPages: {
                auto kv_view = cache.layer_view(cur_layer);
                // The fourth argument is `qo_indptr`, not an optional.
                // It was passed as nullptr on an assumption, and the
                // kernel dereferenced it — the illegal access this
                // drive faulted with on its first live fire.
                kernels::launch_write_kv_to_pages(
                    kv_view, ws.k.data(), ws.v.data(),
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, N, R, stream, row_valid_d);
                break;
            }
            case G4Kernel::AttnFlashinferDecode: {
                auto kv_view = cache.layer_view(cur_layer);
                ops::DecodePlanCache* p =
                    (cur_full ? moe_ws.decode_plan_full
                              : moe_ws.decode_plan_sliding).get();
                ops::DecodePlanCachePtr owned;
                if (p == nullptr) {
                    owned = ops::make_decode_plan();
                    p = owned.get();
                    ops::plan_attention_flashinfer_decode(
                        *p, kv_page_indptr_h, R, cfg.num_attention_heads,
                        cur_hk / cur_d, cur_d, cache.page_size(), attn_ws,
                        stream, /*enable_cuda_graph=*/true,
                        /*full_attention_variant=*/cur_full,
                        cache.hnd_layout());
                }
                ops::dispatch_attention_flashinfer_decode(
                    *p, values.slot(plan.inputs(op)[0]), kv_view,
                    values.slot(plan.outputs(op)[0]),
                    kv_page_indices, kv_page_indptr, kv_last_page_lens,
                    attn_ws, stream,
                    w.per_layer_window_left[static_cast<std::size_t>(cur_layer)],
                    /*logits_soft_cap=*/0.f, /*sm_scale=*/1.0f);
                break;
            }
            case G4Kernel::ChunkedGegluTanh:
                kernels::launch_chunked_geglu_tanh_bf16(
                    ws.gate_up_fused.data(), ws.gate.data(), N, cur_inter, stream);
                break;
            case G4Kernel::GegluTanh: {
                // TWO sites for one kernel, told apart by the WIDTH the
                // op declares — not by a counter. The PLE gate is
                // `ple_dim` wide and its "up" operand is this layer's
                // slice of the relay; the unfused MLP is `cur_inter`
                // wide and its operands are the two projections.
                const auto out = plan.outputs(op);
                const auto& val = plan.value(out[0]);
                const std::uint32_t width =
                    val.dims[val.rank - 1].value;
                if (static_cast<int>(width) == ple_dim) {
                    const auto* signal =
                        static_cast<const std::uint16_t*>(per_layer_token) +
                        static_cast<std::size_t>(cur_layer) * N * ple_dim;
                    kernels::launch_geglu_tanh_bf16(
                        ws.norm_x.data(), signal, ws.norm_x.data(),
                        N * ple_dim, stream);
                } else {
                    kernels::launch_geglu_tanh_bf16(
                        ws.gate.data(), ws.up.data(), ws.gate.data(),
                        N * cur_inter, stream);
                }
                break;
            }
            case G4Kernel::NormResidualScaleNorm: {
                const std::string_view first = aux(0);
                const ParsedName nm = parse_name(first);
                // Two sites: the attention landing (norm_x -> y, then the
                // MLP's input norm) and the PLE landing (norm_y -> y, then
                // the NEXT layer's input norm).
                const bool ple = nm.field == "ple_norm";
                // The PLE landing carries the layer's own scalar; the
                // attention landing carries 1. The declaration does not
                // state it — it is a per-layer load-time constant the
                // executor reads the way the hand-written pass does.
                const float scale =
                    ple ? w.layers[static_cast<std::size_t>(cur_layer)]
                              .layer_scalar_value
                        : 1.f;
                const auto ins = plan.inputs(op);
                const auto outs = plan.outputs(op);
                need(ins, 1, "norm-residual-scale-norm inputs");
                need(outs, 2, "norm-residual-scale-norm outputs");
                // `(landed, mlp_in)`: the stream, then the normed
                // activation. The `ple ? norm_y : norm_x` input choice
                // goes with them -- it was this family naming which
                // scratch the previous statement had landed in.
                kernels::launch_rmsnorm_residual_add_scale_rmsnorm_bf16(
                    values.slot(ins[0]), require(w, first).data(),
                    values.slot(outs[0]), scale,
                    require(w, aux(1)).data(), values.slot(outs[1]),
                    N, H, eps, stream);
                break;
            }
            case G4Kernel::NormResidualAdd: {
                // The `ple_norm` test that used to live here chose
                // between two input scratches. The trace names the
                // input, so there is nothing left to tell apart.
                const std::string_view first = aux(0);
                const auto ins = plan.inputs(op);
                const auto outs = plan.outputs(op);
                need(ins, 1, "norm-residual-add inputs");
                need(outs, 1, "norm-residual-add outputs");
                kernels::launch_rmsnorm_residual_add_bf16(
                    values.slot(ins[0]), require(w, first).data(),
                    values.slot(outs[0]), N, H, eps, stream);
                break;
            }
            case G4Kernel::LogitSoftcap:
                kernels::launch_logit_softcap_bf16(
                    ws.logits.data(), fwd_cfg.final_logit_softcap,
                    static_cast<std::size_t>(lm_head_rows) * V, stream);
                break;
            case G4Kernel::AttnFlashinferPrefill: {
                auto kv_view = cache.layer_view(cur_layer);
                ops::launch_attention_flashinfer_prefill(
                    values.slot(plan.inputs(op)[0]), kv_view,
                    values.slot(plan.outputs(op)[0]),
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, qo_indptr_h, kv_page_indptr_h,
                    N, R, cfg.num_attention_heads, attn_ws, stream,
                    w.per_layer_window_left[static_cast<std::size_t>(cur_layer)],
                    /*logits_soft_cap=*/0.f, /*sm_scale=*/1.0f);
                break;
            }
            case G4Kernel::AttnNaivePaged: {
                auto kv_view = cache.layer_view(cur_layer);
                // `num_pages_in_batch` is the host indptr's LAST entry —
                // the fire's page count, not the layer's and not the
                // cache's.
                ops::launch_attention_naive_paged(
                    values.slot(plan.inputs(op)[0]), kv_view,
                    values.slot(plan.outputs(op)[0]),
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, N, R,
                    static_cast<int>(kv_page_indptr_h[R]),
                    cfg.num_attention_heads, stream,
                    w.per_layer_window_left[static_cast<std::size_t>(cur_layer)],
                    /*sm_scale=*/1.0f, /*logits_soft_cap=*/0.f,
                    /*lse_out=*/nullptr);
                break;
            }
            }
            break;
        }
        case PieForwardOpKind::HookSite:
            // gemma-4's sites are OBSERVATION-only, and arc 2 admits no
            // hooked fire at all (`in.stage_hooks == nullptr` is an
            // eligibility term above), so there is never a program to
            // invoke here. The op is still STATED, because the seam is
            // real and its position is checked -- the arm is what makes
            // the trace executable rather than a load failure.
            //
            // A fire that arrived here WITH hooks would be an eligibility
            // bug, not a site to serve; when gemma-4 grows hook support
            // this is the arm that gains the invoke, next to qwen3_5's.
            break;
        default:
            throw_drift("op kind " +
                        std::to_string(static_cast<std::uint32_t>(op.kind)) +
                        " has no emission rule");
        }
    };

    // Build the fire's rows, lower them, execute the list.
    std::vector<pie_forward::PieForwardRow> rows(static_cast<std::size_t>(N));
    for (int r = 0; r < N; ++r) {
        auto& row = rows[static_cast<std::size_t>(r)];
        row.multi_token = decode_class ? 0 : 1;
        row.custom_mask = 0;
        row.hooked = 0;
        row.lora = 0;
        row.write_desc = 0;
        row.wants_scores = 0;
        row.samples = 1;
        row._pad = 0;
        row.depth_k = -1;
    }
    if (logit_row_indices_d != nullptr && num_logit_rows > 0 &&
        num_logit_rows < N) {
        for (int r = num_logit_rows; r < N; ++r) {
            rows[static_cast<std::size_t>(r)].samples = 0;
        }
    }
    const pie_forward::PieForwardLowered flat =
        plan.lower(rows.data(), rows.size());
    if (flat.uncovered != pie_forward::PieForwardUncovered::None) return false;

    // THE PIN PASS (`model/declared/value_arena.hpp`): this family's
    // buffer convention, stated ONCE.
    //
    // Every arm below used to carry a piece of it — "the normed
    // activation is `ws.norm_x`", "the geglu lands in `ws.gate`" — which
    // is why an arm could not be shared with a family spelling the same
    // role differently (qwen3_5 norms into `ws.norm_x` where llama_like
    // uses `ws.norm_y`). Collected here, an arm asks the arena by VALUE
    // ID and never learns whose convention it is serving.
    //
    // The bytes do not move. Each entry names the buffer that op's arm
    // writes today, so a converted arm addresses exactly what it
    // addressed before and the family A/B is a real comparison rather
    // than a re-baselining. Host-assigned offsets take over per island,
    // as each island's pins come out.
    values.reset_pins_only(plan.value_count());
    values.bind_offsets(ws.declared_values.data(),
                        ws.declared_values.nbytes(), flat);
    {
        const std::size_t op_count = plan.op_count();
        for (std::size_t i = 0; i < op_count; ++i) {
            const PieForwardOp& op = plan.op(i);
            const auto outs = plan.outputs(op);
            if (outs.size == 0) continue;
            const auto pin = [&](std::size_t which, void* ptr) {
                if (which < outs.size) values.pin(outs[which], ptr);
            };
            switch (op.kind) {
            case PieForwardOpKind::Embed:
                pin(0, plan.weight_name(op) == "embed" ? ws.y.data()
                                                       : per_layer_token);
                break;
            case PieForwardOpKind::Matmul: {
                const ParsedName nm = parse_name(plan.weight_name(op));
                if (nm.field == "ple_model_proj")   pin(0, per_layer_proj);
                else if (nm.field == "qkv")         pin(0, ws.qkv_fused.data());
                else if (nm.field == "q_proj")      pin(0, ws.q.data());
                else if (nm.field == "k_proj")      pin(0, ws.k.data());
                else if (nm.field == "v_proj")      pin(0, ws.v.data());
                else if (nm.field == "o_proj")      pin(0, ws.norm_x.data());
                else if (nm.field == "gate_up")     pin(0, ws.gate_up_fused.data());
                else if (nm.field == "gate_proj")   pin(0, ws.gate.data());
                else if (nm.field == "up_proj")     pin(0, ws.up.data());
                else if (nm.field == "down")        pin(0, ws.norm_x.data());
                else if (nm.field == "ple_gate")    pin(0, ws.norm_x.data());
                else if (nm.field == "ple_proj")    pin(0, ws.norm_y.data());
                break;
            }
            case PieForwardOpKind::Rmsnorm:
                // Both sites (`attn_norm`, `final_norm`) norm the stream
                // into the same scratch.
                pin(0, ws.norm_x.data());
                break;
            case PieForwardOpKind::RmsnormPerHead: {
                const ParsedName nm = parse_name(plan.weight_name(op));
                if (nm.field == "ple_model_norm") pin(0, per_layer_proj);
                else if (nm.field == "q_norm")    pin(0, ws.q.data());
                else if (nm.field == "k_norm")    pin(0, ws.k.data());
                break;
            }
            case PieForwardOpKind::SplitQkv:
                pin(0, ws.q.data());
                pin(1, ws.k.data());
                pin(2, ws.v.data());
                break;
            case PieForwardOpKind::Rope:
                pin(0, ws.q.data());
                pin(1, ws.k.data());
                break;
            case PieForwardOpKind::LmHead:
                pin(0, ws.logits.data());
                break;
            case PieForwardOpKind::Launch: {
                const auto names = plan.aux_names(op);
                const auto aux = [&](std::size_t j) { return plan.name(names[j]); };
                switch (resolve_g4_kernel(plan.weight_name(op))) {
                case G4Kernel::ScalarMul: {
                    const std::string_view which = aux(0);
                    if (which == "scale.sqrt_hidden")        pin(0, ws.y.data());
                    else if (which == "scale.sqrt_ple_dim")  pin(0, per_layer_token);
                    else                                     pin(0, per_layer_proj);
                    break;
                }
                case G4Kernel::ResidualAdd:       pin(0, per_layer_proj); break;
                case G4Kernel::TransposeNldToLnd: pin(0, per_layer_token); break;
                case G4Kernel::QkvPackedPost:     pin(0, ws.q.data()); break;
                case G4Kernel::QkRmsnormRopeRounded:
                    pin(0, ws.q.data());
                    pin(1, ws.k.data());
                    break;
                case G4Kernel::RopeQOnlyPartial:
                case G4Kernel::RopeQOnly:         pin(0, ws.q.data()); break;
                case G4Kernel::RmsnormNoScale:    pin(0, ws.v.data()); break;
                case G4Kernel::AttnFlashinferDecode:
                case G4Kernel::AttnFlashinferPrefill:
                case G4Kernel::AttnNaivePaged:    pin(0, ws.attn_out.data()); break;
                case G4Kernel::ChunkedGegluTanh:  pin(0, ws.gate.data()); break;
                case G4Kernel::GegluTanh: {
                    // TWO sites for one kernel, told apart by the WIDTH
                    // the op declares — the same test the arm makes.
                    const auto& val = plan.value(outs[0]);
                    const std::uint32_t width = val.dims[val.rank - 1].value;
                    pin(0, static_cast<int>(width) == ple_dim ? ws.norm_x.data()
                                                              : ws.gate.data());
                    break;
                }
                case G4Kernel::NormResidualScaleNorm:
                    // `(landed, mlp_in)` in the declaration: the stream
                    // first, the normed activation second.
                    pin(0, ws.y.data());
                    pin(1, ws.norm_x.data());
                    break;
                case G4Kernel::NormResidualAdd:   pin(0, ws.y.data()); break;
                case G4Kernel::LogitSoftcap:      pin(0, ws.logits.data()); break;
                default: break;
                }
                break;
            }
            default:
                break;
            }
        }
    }

    std::size_t next_site = 0;
    std::size_t at = 0;
    while (at < flat.launches_len || next_site < flat.structural_len) {
        const bool site_first =
            at >= flat.launches_len ||
            (next_site < flat.structural_len &&
             flat.structural[next_site].at_op < flat.launches[at].at_op);
        if (site_first) {
            execute_op(plan.op(flat.structural[next_site].at_op));
            ++next_site;
            continue;
        }
        const std::uint32_t at_op = flat.launches[at].at_op;
        while (at < flat.launches_len && flat.launches[at].at_op == at_op) ++at;
        execute_op(plan.op(at_op));
    }
    return true;
}

}  // namespace pie_cuda_driver::model
