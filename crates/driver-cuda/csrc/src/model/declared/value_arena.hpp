#pragma once

// The SSA VALUE ARENA — where a traced value's buffer comes from.
//
// The trace is pure SSA: `rmsnorm(x: &Val) -> Val`, and every op carries
// its `inputs` / `outputs` as value ids. What it deliberately does NOT
// carry is a buffer, because choosing one is a BACKEND job. Both family
// executors did that job as convention — "the normed activation lives in
// `ws.norm_y`" here, "in `ws.norm_x`" there — and since an arm then has to
// know whose convention it is serving, the executor could not be one file.
// (Measured at the start of this merge: the SSA edges are read 1 and 0
// times across the two executors; workspace fields are read 82 and 110.)
//
// This is the other answer: a value gets a slot, the arm asks for the slot
// by value id, and which physical bytes those are stops being anyone's
// convention.
//
// WHO CHOOSES THE BYTES: the HOST does, and this file no longer does.
//
// It used to. It walked the plan for last-use, kept a free list, and
// bump-allocated on first ask — the same job `Buffers::assign` does in
// `model-compiler/src/lower.rs`. Two allocators over one plan have to
// agree byte-for-byte forever, and these two already did not: this copy
// predates `Select` (a value that IS a window of another's bytes), the
// `kernel!` in-place table (an output that IS the operand it accumulates
// into) and `Dim::MoeAlignedRoutes` (a padded extent it cannot size). On
// a text using any of the three it placed or sized a value differently,
// and silently, because an allocator that hands back a plausible pointer
// reports nothing.
//
// So the host assigns and this maps. `PieForwardLowered::value_offsets`
// is a byte offset per value id, `arena_bytes` is the block they need,
// and `slot()` is an add. Everything the old allocator knew — liveness,
// reuse, alignment, the union of values that share bytes — is decided
// once on the host, where it is tested (`model/tests/arena_soundness.rs`
// walks a write trace per family and fails if any value lands on bytes a
// later op still reads). A mapper cannot notice an overlapping
// assignment, which is exactly why the overlap has to be impossible
// there rather than caught here.
//
// CAPTURE SAFETY comes out of the same change for free. A decode body
// runs inside `cudaStreamBeginCapture`, so it must allocate nothing and
// a value must land at the same address on every fire. That used to hold
// because the ask ORDER was deterministic — an emergent property, true
// until an arm asked in a different order. Now the address is a function
// of the plan and the fire's extents, so it is structural.
//
// PINS stay, for the values the host declines to place: the buffer some
// machinery OUTSIDE the traced ops reaches by name — LoRA captures the
// normed activation's pointer at fire setup, hook sites observe the
// query buffer, the sampler reads the logits. The host marks those
// `NAMED` off the SEAM statements, so which values they are is stated in
// the declaration rather than listed per family, and the pass that binds
// them is the only family-shaped thing left here.
//
// MIGRATION: arms move onto the arena one dataflow island at a time. An
// arm that has not moved keeps its convention, and the two coexist —
// which is only sound while every producer and consumer of a given value
// have moved TOGETHER. That is why the islands are converted whole.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "pie_forward.h"
#include "pie_forward/plan.hpp"

using pie_forward::PieForwardLowered;
using pie_forward::PieForwardValue;

namespace pie_cuda_driver::model::declared {

class ValueArena {
   public:
    // `offset[v] == kNamed` is the host declining to place a value: its
    // bytes are the backend's to bind, via `pin`.
    static constexpr std::size_t kNamed = static_cast<std::size_t>(-1);

    // `block` is workspace-owned memory, allocated once, outside capture;
    // `flat` is the lowering that says where in it every value lives.
    //
    // Refuses rather than truncates when the block is too small. An arena
    // that quietly hands out addresses past its end is a corruption whose
    // first symptom is somewhere else entirely, and the block is sized by
    // a workspace formula that nothing previously checked against the
    // plan's actual need.
    void reset(void* block, std::size_t capacity,
               const PieForwardLowered& flat) {
        if (flat.arena_bytes > capacity) {
            throw std::runtime_error(
                "declared value arena: this fire needs " +
                std::to_string(flat.arena_bytes) +
                " bytes of activation block, but the workspace holds " +
                std::to_string(capacity) +
                " — the arena may not grow inside a captured body");
        }
        block_ = static_cast<std::uint8_t*>(block);
        capacity_ = capacity;
        want_ = flat.arena_bytes;
        offsets_ = flat.value_offsets;
        count_ = flat.value_offsets_len;
        pinned_.assign(count_, nullptr);
    }

    // The same arena for an executor that has not moved onto the flat
    // list yet: every value is the backend's to bind, so `slot()` serves
    // exactly the pins and names any other value as unbound.
    //
    // This is the honest shape of a half-migrated executor. It walks OPS
    // rather than rectangles, so it holds no lowering to read offsets
    // from, and lowering per fire just to get one would put a plan walk
    // on the hot path for buffers it is not yet using. What it must not
    // do is assign its own — that is the second allocator this file just
    // stopped being.
    void reset_pins_only(std::size_t value_count) {
        block_ = nullptr;
        capacity_ = 0;
        offsets_ = nullptr;
        count_ = value_count;
        pinned_.assign(count_, nullptr);
    }

    // Point an already-pinned arena at a lowering's table, for the leg
    // that gets its rectangles after the pin pass has run. Keeps the
    // pins, which is what makes the two coexist.
    void bind_offsets(void* block, std::size_t capacity,
                      const PieForwardLowered& flat) {
        // NO capacity precheck here, unlike `reset`. The block is sized
        // for the islands that have MOVED (`ws.declared_values` is one
        // or two values wide, and its comment says every further island
        // widens it), while `arena_bytes` is the whole plan's peak. A
        // half-migrated executor asks for a handful of values and must
        // not be refused for the ones it never touches; `slot` bounds-
        // checks the ones it does.
        block_ = static_cast<std::uint8_t*>(block);
        want_ = flat.arena_bytes;
        capacity_ = capacity;
        offsets_ = flat.value_offsets;
        if (flat.value_offsets_len > count_) {
            count_ = flat.value_offsets_len;
            pinned_.resize(count_, nullptr);
        }
    }

    // PIN a value the host declined to place. Declared once per family by
    // a pass over the plan, so an ARM still just asks by value id and
    // stays family-blind; the convention lives in the pass, not in 82
    // scattered arm sites.
    //
    // A pin WINS over the host's table where both speak, and that is the
    // migration rule rather than a conflict: an arm that has not moved
    // still writes `ws.norm_y` by convention, so its consumers have to
    // read those bytes and not the ones the lowering set aside. The two
    // agree again when the island moves, because the arm stops naming a
    // workspace field and the pin for that value goes away with it.
    void pin(std::uint32_t value_id, void* ptr) {
        if (value_id >= count_ || ptr == nullptr) return;
        pinned_[value_id] = ptr;
    }

    // The bytes one value lives in — an add, plus the pin table for the
    // values the host left to the backend.
    void* slot(std::uint32_t value_id) const {
        if (value_id >= count_) {
            throw std::runtime_error(
                "declared value arena: value id " +
                std::to_string(value_id) + " is outside the plan's table");
        }
        // The pin first — see `pin` for why it outranks the table.
        if (pinned_[value_id] != nullptr) return pinned_[value_id];
        const std::size_t at = offsets_ == nullptr ? kNamed : offsets_[value_id];
        if (at == kNamed) {
            throw std::runtime_error(
                "declared value arena: value " + std::to_string(value_id) +
                " is one the lowering left to the backend, and no pin pass "
                "bound it");
        }
        if (block_ == nullptr || at >= capacity_) {
            throw std::runtime_error(
                "declared value arena: value " + std::to_string(value_id) +
                " sits at offset " + std::to_string(at) + ", past the " +
                std::to_string(capacity_) +
                "-byte block — this plan's arena wants " +
                std::to_string(want_) +
                " bytes, so `ws.declared_values` is sized for fewer islands "
                "than are asking");
        }
        return block_ + at;
    }

    // Overload kept for the call sites mid-migration, which pass the
    // value descriptor because the arena used to need it for SIZING. It
    // does not any more — the host sized it — so the descriptor is
    // ignored and the sites drop it as their island moves.
    void* slot(std::uint32_t value_id, const PieForwardValue&) const {
        return slot(value_id);
    }

   private:
    std::uint8_t* block_ = nullptr;
    std::size_t capacity_ = 0;
    // What the whole plan's arena wants, for the refusal message: the
    // block being short is a SIZING fact, and naming the target is the
    // difference between a fix and a bisect.
    std::size_t want_ = 0;
    const std::size_t* offsets_ = nullptr;
    std::size_t count_ = 0;
    std::vector<void*> pinned_;
};

}  // namespace pie_cuda_driver::model::declared
