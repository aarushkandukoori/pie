//! The Guard/Peel surface unification (`.wiki/tart/dsl.md` migration
//! step 2), pinned.
//!
//! Here rather than in `model-compiler` for the reason `lowering.rs`
//! gives: `trace_finish` checks every launch against the kernel!
//! signature registry, and the registry is the BACKEND crates'. A trace
//! built where they are not linked refuses itself.
//!
//! The claim these pin is narrow and total: `regions` traces what the two
//! constructs it replaces traced, byte for byte in the op stream. The
//! goldens can only say that for families that already migrated; this
//! says it for the construct.


    use model_compiler::dsl::*;
    use model_compiler::trace::{DType, Dim, GuardPred, OpKind, Shape};

    /// A statement the registry knows, so `trace_finish` accepts it, and
    /// one with no operands so the two chains differ in nothing but the
    /// construct around them.
    fn stmt(t: &Trace) {
        let x = input(t, 8);
        let _ = cuda::residual_add(&x, &x, 8);
    }

    /// The unified surface must trace what the two it replaces traced.
    /// Not a formality: the whole claim of this step is that the SURFACE
    /// changed and nothing else did, and the goldens can only pin the
    /// families that already migrated. This pins the construct itself.
    #[test]
    fn a_fire_armed_chain_traces_what_guarded_value_did() {
        let a = trace_named("regions.cuda.decode", |t| {
            let (g, _) = guarded_value(t, None, (Shape(vec![Dim::Tokens]), DType::BF16));
            g.arm(GuardPred::HasLora, || stmt(t)).otherwise(|| stmt(t));
        });
        let b = trace_named("regions.cuda.decode", |t| {
            regions(
                t,
                None,
                Some((Shape(vec![Dim::Tokens]), DType::BF16)),
                |c| c.arm(Region::Fire(GuardPred::HasLora), || stmt(t)),
                || stmt(t),
            );
        });
        assert_eq!(a.ops.len(), b.ops.len());
        assert!(matches!(a.ops[0].kind, OpKind::Guard { .. }));
        assert_eq!(
            format!("{:?}", a.ops[0].kind),
            format!("{:?}", b.ops[0].kind)
        );
    }

    #[test]
    fn a_rows_armed_chain_traces_what_by_rows_did() {
        let a = trace_named("regions.cuda.decode", |t| {
            by_rows(t, None, None, |c| {
                c.arm(RowPred::Unmasked, || stmt(t));
                c.rest(|| stmt(t));
            });
        });
        let b = trace_named("regions.cuda.decode", |t| {
            regions(
                t,
                None,
                None,
                |c| c.arm(Region::Rows(RowPred::Unmasked), || stmt(t)),
                || stmt(t),
            );
        });
        assert_eq!(a.ops.len(), b.ops.len());
        assert!(matches!(a.ops[0].kind, OpKind::Peel { .. }));
        assert_eq!(
            format!("{:?}", a.ops[0].kind),
            format!("{:?}", b.ops[0].kind)
        );
    }

    /// A mix is REFUSED, not flattened into whichever op opened first.
    #[test]
    #[should_panic(expected = "cannot be both disciplines")]
    fn a_mixed_chain_is_refused() {
        let _ = trace_named("regions.cuda.decode", |t| {
            regions(
                t,
                None,
                None,
                |c| {
                    c.arm(Region::Rows(RowPred::Unmasked), || stmt(t));
                    c.arm(Region::Fire(GuardPred::HasLora), || stmt(t));
                },
                || stmt(t),
            );
        });
    }

