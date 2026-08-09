//! The types a launcher takes that are neither scalars nor pointers.
//!
//! One `#[repr(C)]` mirror per C++ record, and nothing else. In particular
//! **no driver state lives here**: a record earns a place in this module only
//! by being kernel vocabulary — something a launcher reads and forgets.
//! `AttentionWorkspace` and the plan caches look like they belong and do not;
//! they are scratch pools and caches the DRIVER owns, they are C++ objects
//! only because they are currently defined on the kernel side of the line,
//! and the answer for them is to move rather than to be mirrored here.
//!
//! [`KvCacheLayerView`] is the opposite case and the reason this module
//! exists. Its C++ header says why it was written: *"a neutral per-layer KV
//! descriptor, so a cache owner in the driver can expose its storage to a
//! kernel without this crate importing `store/`"*. It carries no ownership,
//! outlives no call, and is pure description — which is exactly what makes a
//! `#[repr(C)]` mirror the whole of the port rather than a wrapper over one.
//!
//! ## Why a mirror is safe to write here at all
//!
//! Because it is checked. `tests/launch_abi.rs` reads each mirror's own
//! layout with `offset_of!`, bakes those numbers into a generated C++
//! translation unit as `static_assert`s, and compiles it against the real
//! header. Size, alignment, every field offset and the member COUNT all have
//! to agree or the file does not build. No number below was written by hand
//! and none is checked by inspection.

use core::ffi::{c_int, c_void};

use crate::dtype::DType;

/// How a KV cache stores its pages.
///
/// Discriminants are the C++ enum's and are load-bearing for the same reason
/// [`DType`]'s are: the value crosses the boundary as the one byte the C++
/// declares (`enum class KvCacheScheme : std::uint8_t`), so agreeing on the
/// numbers is what makes it a cast instead of a translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KvCacheScheme {
    /// Stored as the model's own dtype; no scales.
    Native = 0,
    /// FP8 with one scale for the whole tensor.
    Fp8PerTensor = 1,
    /// INT8 with a scale per (token, head).
    Int8PerTokenHead = 2,
    /// FP8 with a scale per (token, head).
    Fp8PerTokenHead = 3,
    /// FP4 with a scale per block.
    Fp4Block = 4,
}

/// One layer's KV storage, as a kernel sees it.
///
/// Field order is the C++'s and may not be rearranged — the mirror is checked
/// positionally, so a reordering is a build failure rather than a subtle one,
/// but it is still a reordering of a shared ABI and not a local choice.
///
/// The two envelope pointers are null unless envelopes were explicitly
/// enabled on the cache; [`Self::has_envelopes`] is the C++'s own predicate.
/// They are `*mut u16` rather than opaque because the C++ types them that way
/// — `std::uint16_t*` — and dropping that would make the mirror describe less
/// than the header does.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct KvCacheLayerView {
    /// Which layer this describes.
    pub layer: c_int,
    /// Where its pages actually live, for a KV-shared layer.
    pub source_layer: c_int,
    /// Pages in this layer's pool.
    pub num_pages: c_int,
    /// Tokens per page.
    pub page_size: c_int,
    /// KV heads, after any GQA grouping.
    pub num_kv_heads: c_int,
    /// Channels per head.
    pub head_dim: c_int,
    /// How the pages are stored, and therefore whether the scale planes
    /// below are meaningful.
    pub scheme: KvCacheScheme,
    /// The element type the pages actually hold, which is the model's dtype
    /// only under [`KvCacheScheme::Native`].
    pub storage_dtype: DType,
    /// The quantisation block, for the schemes that have one.
    pub block_size: c_int,
    /// The K pages.
    pub k_pages: *mut c_void,
    /// The V pages.
    pub v_pages: *mut c_void,
    /// K's scale plane; null under [`KvCacheScheme::Native`].
    pub k_scales: *mut c_void,
    /// V's scale plane; null under [`KvCacheScheme::Native`].
    pub v_scales: *mut c_void,
    /// A bf16 shadow of K, for the kernels that cannot dequantise inline.
    pub k_bf16_pages: *mut c_void,
    /// A bf16 shadow of V, for the same reason.
    pub v_bf16_pages: *mut c_void,
    /// Quest per-page key envelopes, `[num_pages, num_kv_heads, head_dim]`
    /// bf16 each. Null unless envelopes were enabled on the cache.
    pub k_env_min: *mut u16,
    /// The other envelope plane; see [`Self::has_envelopes`].
    pub k_env_max: *mut u16,
    /// Pages are `[..., num_kv_heads, page_size, head_dim]` rather than
    /// `[..., page_size, num_kv_heads, head_dim]`.
    pub hnd_layout: bool,
    /// Storage is the model's own bf16; [`Self::is_native_bf16`] reads it.
    pub native_bf16: bool,
}

impl KvCacheLayerView {
    /// Both envelope planes are present.
    ///
    /// The C++ predicate, kept rather than left to the caller: it tests BOTH
    /// pointers, and a caller that checked one would be right on every cache
    /// that exists and wrong on the one that half-allocates.
    pub fn has_envelopes(&self) -> bool {
        !self.k_env_min.is_null() && !self.k_env_max.is_null()
    }

    /// Storage is the model's own bf16, so no dequantisation step applies.
    pub fn is_native_bf16(&self) -> bool {
        self.native_bf16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pointer-carrying mirror is neither `Send` nor `Sync` by accident, so
    /// say what it is: a borrowed description of device memory the caller
    /// owns. Nothing here keeps the pages alive.
    #[test]
    fn a_view_is_a_borrow_and_owns_nothing() {
        let v = KvCacheLayerView {
            layer: 0,
            source_layer: 0,
            num_pages: 4,
            page_size: 16,
            num_kv_heads: 2,
            head_dim: 64,
            scheme: KvCacheScheme::Native,
            storage_dtype: DType::Bf16,
            block_size: 0,
            k_pages: core::ptr::null_mut(),
            v_pages: core::ptr::null_mut(),
            k_scales: core::ptr::null_mut(),
            v_scales: core::ptr::null_mut(),
            k_bf16_pages: core::ptr::null_mut(),
            v_bf16_pages: core::ptr::null_mut(),
            k_env_min: core::ptr::null_mut(),
            k_env_max: core::ptr::null_mut(),
            hnd_layout: false,
            native_bf16: true,
        };
        let copy = v;
        assert_eq!(copy.num_pages, 4);
        assert!(copy.is_native_bf16());
    }

    /// `has_envelopes` needs BOTH, which is the C++'s rule and not an
    /// obvious one — a half-allocated cache is exactly what it guards.
    #[test]
    fn one_envelope_plane_is_not_envelopes() {
        let mut v = KvCacheLayerView {
            layer: 0,
            source_layer: 0,
            num_pages: 1,
            page_size: 1,
            num_kv_heads: 1,
            head_dim: 1,
            scheme: KvCacheScheme::Native,
            storage_dtype: DType::Bf16,
            block_size: 0,
            k_pages: core::ptr::null_mut(),
            v_pages: core::ptr::null_mut(),
            k_scales: core::ptr::null_mut(),
            v_scales: core::ptr::null_mut(),
            k_bf16_pages: core::ptr::null_mut(),
            v_bf16_pages: core::ptr::null_mut(),
            k_env_min: core::ptr::null_mut(),
            k_env_max: core::ptr::null_mut(),
            hnd_layout: false,
            native_bf16: true,
        };
        assert!(!v.has_envelopes());
        let mut cell: u16 = 0;
        v.k_env_min = &mut cell;
        assert!(!v.has_envelopes(), "one plane is not enough");
        v.k_env_max = &mut cell;
        assert!(v.has_envelopes());
    }
}
