use lbfgs_rs::infra::math::kernel::vec_scale;
use lbfgs_rs::shared::types::primitives::ScalarType;

// ── Helpers ──

/// Naive scalar reference implementation: out[i] = scalar * x[i].
/// Order matches `vec_scale_ansi` (scalar first).
fn scale_oracle(x: &[ScalarType], scalar: ScalarType, out: &mut [ScalarType]) {
    for i in 0..x.len() {
        out[i] = scalar * x[i];
    }
}

/// Generate vector [0, 1, 2, ..., len-1].
fn arange(len: usize) -> Vec<ScalarType> {
    (0..len).map(|i| i as ScalarType).collect()
}

/// Generate a vector filled with `len` copies of `value`.
fn fill(len: usize, value: ScalarType) -> Vec<ScalarType> {
    vec![value; len]
}

/// Tolerance for floating-point comparisons.
fn epsilon() -> ScalarType {
    if cfg!(feature = "f32") {
        1e-6 as ScalarType
    } else {
        1e-12 as ScalarType
    }
}

// T0 — Input validation / Contract (C01, C02)

/// C01 — In debug builds, a length mismatch triggers a panic (`debug_assert_eq!`).
/// In release builds, `debug_assert_eq!` disappears, so this test is cfg-gated to avoid triggering UB.
#[test]
#[should_panic(expected = "vector length mismatch")]
#[cfg(debug_assertions)]
fn t0_1_length_mismatch_panics_in_debug() {
    let x = vec![1.0 as ScalarType; 4];
    let mut out = vec![0.0 as ScalarType; 3];
    vec_scale(&x, 1.0 as ScalarType, &mut out);
}

/// C02 — Empty x and empty out must not panic.
#[test]
fn t0_2_empty_vectors_no_panic() {
    let mut out: Vec<ScalarType> = vec![];
    vec_scale(&[], 2.0 as ScalarType, &mut out);
    assert_eq!(out.len(), 0);
}

// T1 — Scalar value coverage (S01-S08)
mod scalar_values {
    use super::*;

    /// Check `vec_scale` on a small fixed vector against the oracle.
    fn check(scalar: ScalarType) {
        let x = vec![
            1.5 as ScalarType,
            -2.0 as ScalarType,
            3.0 as ScalarType,
            0.0 as ScalarType,
        ];
        let mut out = vec![0.0 as ScalarType; x.len()];
        let mut expected = vec![0.0 as ScalarType; x.len()];
        scale_oracle(&x, scalar, &mut expected);
        vec_scale(&x, scalar, &mut out);
        for (i, (&a, &e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= epsilon(),
                "scalar={scalar}, idx={i}: got {a}, expected {e}, diff {}",
                (a - e).abs()
            );
        }
    }

    #[test]
    fn t1_1_scalar_zero() {
        check(0.0 as ScalarType);
    } // S01
    #[test]
    fn t1_2_scalar_one() {
        check(1.0 as ScalarType);
    } // S02
    #[test]
    fn t1_3_scalar_neg_one() {
        check(-1.0 as ScalarType);
    } // S03
    #[test]
    fn t1_4_scalar_two() {
        check(2.0 as ScalarType);
    } // S04
    #[test]
    fn t1_5_scalar_half() {
        check(0.5 as ScalarType);
    } // S05
    #[test]
    fn t1_6_scalar_neg_half() {
        check(-0.5 as ScalarType);
    } // S06
    #[test]
    fn t1_7_scalar_large() {
        check(1e10 as ScalarType);
    } // S07
    #[test]
    fn t1_8_scalar_tiny() {
        check(1e-10 as ScalarType);
    } // S08

    /// S02-specific: scalar = 1.0 must act as the identity function.
    #[test]
    fn t1_9_scalar_one_is_identity() {
        let x = arange(16);
        let mut out = vec![0.0 as ScalarType; 16];
        vec_scale(&x, 1.0 as ScalarType, &mut out);
        assert_eq!(out, x);
    }

    /// S01-specific: scalar = 0.0 must zero everything (common "clear buffer" usage).
    #[test]
    fn t1_10_scalar_zero_zeroes_all() {
        let x = arange(20);
        let mut out = vec![9.0 as ScalarType; 20];
        vec_scale(&x, 0.0 as ScalarType, &mut out);
        assert_eq!(out, vec![0.0 as ScalarType; 20]);
    }

    /// S03-specific: scalar = -1.0 must negate every element.
    #[test]
    fn t1_11_scalar_neg_one_negates() {
        let x = arange(16);
        let mut out = vec![0.0 as ScalarType; 16];
        vec_scale(&x, -1.0 as ScalarType, &mut out);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, -(i as ScalarType), "idx={i}: negation wrong, got {v}");
        }
    }
}

// T2 — Boundary alignment (L01-L13, three-tier tail handling)
mod boundary_alignment {
    use super::*;

    /// Assert `vec_scale` result matches oracle for `arange(len) * scalar`.
    fn assert_scale(len: usize, scalar: ScalarType) {
        let x = arange(len);
        let mut out = vec![0.0 as ScalarType; len];
        let mut expected = vec![0.0 as ScalarType; len];
        scale_oracle(&x, scalar, &mut expected);
        vec_scale(&x, scalar, &mut out);
        for (i, (&a, &e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= epsilon(),
                "len={len}, scalar={scalar}, idx={i}: got {a}, expected {e}, diff {}",
                (a - e).abs()
            );
        }
    }

    fn s() -> ScalarType {
        1.5 as ScalarType
    }

    #[test]
    fn t2_1_len_0() {
        assert_scale(0, s());
    } // L01: empty
    #[test]
    fn t2_2_len_1() {
        assert_scale(1, s());
    } // L02: pure scalar tail
    #[test]
    fn t2_3_len_3() {
        assert_scale(3, s());
    } // L03: < 1 SIMD lane

    /// L04 — exactly 1 SIMD lane (f32: 4, f64: 2)
    #[test]
    fn t2_4_one_lane() {
        let n = if cfg!(feature = "f32") { 4 } else { 2 };
        assert_scale(n, s());
    }

    /// L05 — f32-only: 1 lane + 1 scalar tail (len=5)
    #[test]
    #[cfg(feature = "f32")]
    fn t2_5_f32_one_lane_plus_one_scalar() {
        assert_scale(5, s());
    }

    /// L06 — f32-only: 1 lane + 3 scalar tail (len=7)
    #[test]
    #[cfg(feature = "f32")]
    fn t2_6_f32_one_lane_plus_three_scalar() {
        assert_scale(7, s());
    }

    /// L07 — len=15: f32 is 3 lanes + 3 scalars; f64 is 7 lanes + 1 scalar
    #[test]
    fn t2_7_len_15() {
        assert_scale(15, s());
    }

    /// L08 — exactly 1 unroll block (f32: 16, f64: 4)
    #[test]
    fn t2_8_one_block() {
        let n = if cfg!(feature = "f32") { 16 } else { 4 };
        assert_scale(n, s());
    }

    /// L09 — 1 block + 1 scalar (f32: 17, f64: 5)
    #[test]
    fn t2_9_one_block_plus_one_scalar() {
        let n = if cfg!(feature = "f32") { 17 } else { 5 };
        assert_scale(n, s());
    }

    /// L10 — f32-only: 1 block + 1 lane, 0 scalar (len=20)
    #[test]
    #[cfg(feature = "f32")]
    fn t2_10_f32_one_block_plus_one_lane() {
        assert_scale(20, s());
    }

    /// L10 — f64-only: 1 block + 1 lane, 0 scalar (len=6)
    #[test]
    #[cfg(feature = "f64")]
    fn t2_11_f64_one_block_plus_one_lane() {
        assert_scale(6, s());
    }

    /// L11 — 1 block + 1 lane + 1 scalar (f32: 21, f64: 7) — hits all three tail tiers
    #[test]
    fn t2_12_three_tier_tail() {
        let n = if cfg!(feature = "f32") { 21 } else { 7 };
        assert_scale(n, s());
    }

    /// L12 — 2 full blocks (f32: 32, f64: 8)
    #[test]
    fn t2_13_two_blocks() {
        let n = if cfg!(feature = "f32") { 32 } else { 8 };
        assert_scale(n, s());
    }

    /// L13 — Mixed: multiple blocks + multiple lanes + scalars
    #[test]
    fn t2_14_len_100() {
        assert_scale(100, s());
    }

    /// Extra test — large scale to catch cumulative drift
    #[test]
    fn t2_15_len_4096() {
        assert_scale(4096, s());
    }
}

// T3 — Input content coverage (V01-V06, non-special values)
mod input_content {
    use super::*;

    /// Run `vec_scale` on the given vector and scalar, then assert against oracle.
    fn run_with(x: &[ScalarType], scalar: ScalarType) -> Vec<ScalarType> {
        let mut out = vec![0.0 as ScalarType; x.len()];
        let mut expected = vec![0.0 as ScalarType; x.len()];
        scale_oracle(x, scalar, &mut expected);
        vec_scale(x, scalar, &mut out);
        for (i, (&a, &e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= epsilon(),
                "idx={i}: got {a}, expected {e}, diff {}",
                (a - e).abs()
            );
        }
        out
    }

    /// V01 — All-zero input
    #[test]
    fn t3_1_all_zeros() {
        let out = run_with(&fill(16, 0.0 as ScalarType), 2.0 as ScalarType);
        assert_eq!(out, vec![0.0 as ScalarType; 16]);
    }

    /// V02 — All-ones input: output == scalar broadcast
    #[test]
    fn t3_2_all_ones() {
        let out = run_with(&fill(16, 1.0 as ScalarType), 3.0 as ScalarType);
        assert_eq!(out, vec![3.0 as ScalarType; 16]);
    }

    /// V03 — All-negative input
    #[test]
    fn t3_3_all_negatives() {
        let x: Vec<ScalarType> = (0..16).map(|i| -((i + 1) as ScalarType)).collect();
        let out = run_with(&x, 2.0 as ScalarType);
        let expected: Vec<ScalarType> = x.iter().map(|&v| v * 2.0 as ScalarType).collect();
        assert_eq!(out, expected);
    }

    /// V04 — Alternating signs [1, -1, 1, -1, ...]
    #[test]
    fn t3_4_alternating_sign() {
        let x: Vec<ScalarType> = (0..16)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 } as ScalarType)
            .collect();
        let out = run_with(&x, 2.0 as ScalarType);
        let expected: Vec<ScalarType> = x.iter().map(|&v| v * 2.0 as ScalarType).collect();
        assert_eq!(out, expected);
    }

    /// V05 — Monotonically increasing [1, 2, 3, ..., n]
    #[test]
    fn t3_5_monotonic_increase() {
        let x = arange(20);
        let out = run_with(&x, 0.5 as ScalarType);
        let expected: Vec<ScalarType> = x.iter().map(|&v| v * 0.5 as ScalarType).collect();
        assert_eq!(out, expected);
    }

    /// V06 — Mixed magnitude [1e-10, 1e10, 1.0, -1.0]
    #[test]
    fn t3_6_mixed_magnitude() {
        let x = vec![
            1e-10 as ScalarType,
            1e10 as ScalarType,
            1.0 as ScalarType,
            -1.0 as ScalarType,
        ];
        let out = run_with(&x, 2.0 as ScalarType);
        // Use relative tolerance here — magnitude differences make absolute tolerance unsuitable
        let expected = vec![
            2e-10 as ScalarType,
            2e10 as ScalarType,
            2.0 as ScalarType,
            -2.0 as ScalarType,
        ];
        for (i, (&a, &e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= e.abs() * epsilon(),
                "idx={i}: got {a}, expected {e}"
            );
        }
    }
}

// T4 — Numeric precision (P01-P04)
mod numeric_precision {
    use super::*;

    /// P01 — General case: element-wise match against oracle for n=1000 pseudo-random vector
    #[test]
    fn t4_1_matches_oracle_general() {
        let n = 1000;
        let x: Vec<ScalarType> = (0..n)
            .map(|i| ((i as f64) * 1.234 + 0.5).sin() as ScalarType)
            .collect();
        let scalar = 1.7 as ScalarType;
        let mut out = vec![0.0 as ScalarType; n];
        let mut expected = vec![0.0 as ScalarType; n];
        scale_oracle(&x, scalar, &mut expected);
        vec_scale(&x, scalar, &mut out);
        for (i, (&a, &e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= epsilon(),
                "idx={i}: got {a}, expected {e}, diff {}",
                (a - e).abs()
            );
        }
    }

    /// P02 — Extreme mixed values (subnormals + large numbers) — check SIMD vs scalar path consistency
    #[test]
    fn t4_2_extreme_mixed_values() {
        let x = vec![
            1e-30 as ScalarType,
            1e30 as ScalarType,
            -1e-30 as ScalarType,
            -1e30 as ScalarType,
        ];
        let scalar = 1e5 as ScalarType;
        let mut out = vec![0.0 as ScalarType; 4];
        let mut expected = vec![0.0 as ScalarType; 4];
        scale_oracle(&x, scalar, &mut expected);
        vec_scale(&x, scalar, &mut out);
        for (i, (&a, &e)) in out.iter().zip(expected.iter()).enumerate() {
            // For extreme values, use relative tolerance (absolute tolerance would fail)
            let rel = if e.abs() > 1.0 as ScalarType {
                (a - e).abs() / e.abs()
            } else {
                (a - e).abs()
            };
            assert!(
                rel < epsilon() * 100.0,
                "extreme idx={i}: got {a}, expected {e}"
            );
        }
    }

    /// P03 — Integer-representable values × integer scalar: require strict equality (no floating-point error)
    #[test]
    fn t4_3_integer_values_strict_equal() {
        let n = 64;
        let x: Vec<ScalarType> = (0..n).map(|i| (i as i32) as ScalarType).collect();
        let scalar = 4_i32 as ScalarType;
        let mut out = vec![0.0 as ScalarType; n];
        vec_scale(&x, scalar, &mut out);
        let expected: Vec<ScalarType> = (0..n).map(|i| (i * 4) as ScalarType).collect();
        assert_eq!(out, expected);
    }

    /// P04 — Round-trip consistency: first ×2 then ×0.5 must return the original vector (within tolerance)
    #[test]
    fn t4_4_round_trip_consistency() {
        let n = 100;
        let x = arange(n);
        let mut buf = vec![0.0 as ScalarType; n];
        vec_scale(&x, 2.0 as ScalarType, &mut buf);
        let mut back = vec![0.0 as ScalarType; n];
        vec_scale(&buf, 0.5 as ScalarType, &mut back);
        for (i, (&a, &b)) in back.iter().zip(x.iter()).enumerate() {
            assert!(
                (a - b).abs() <= epsilon(),
                "round-trip idx={i}: got {a}, expected {b}"
            );
        }
    }
}

// T5 — IEEE 754 special values (S09-S11, V07-V09)
mod special_values {
    use super::*;

    /// S09 — scalar = NaN: all outputs must be NaN
    #[test]
    fn t5_1_scalar_nan_propagates() {
        let x = vec![1.0 as ScalarType, 2.0 as ScalarType, 3.0 as ScalarType];
        let mut out = vec![0.0 as ScalarType; 3];
        vec_scale(&x, ScalarType::NAN, &mut out);
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_nan(), "scalar=NaN: idx={i} expected NaN, got {v}");
        }
    }

    /// S10 — scalar = +Inf: positive×Inf=+Inf, negative×Inf=-Inf, 0×Inf=NaN
    #[test]
    fn t5_2_scalar_pos_inf_propagates() {
        let x = vec![1.0 as ScalarType, -2.0 as ScalarType, 0.0 as ScalarType];
        let mut out = vec![0.0 as ScalarType; 3];
        vec_scale(&x, ScalarType::INFINITY, &mut out);
        assert!(
            out[0].is_infinite() && out[0] > 0.0,
            "idx=0: +1*Inf should be +Inf, got {}",
            out[0]
        );
        assert!(
            out[1].is_infinite() && out[1] < 0.0,
            "idx=1: -2*Inf should be -Inf, got {}",
            out[1]
        );
        assert!(
            out[2].is_nan(),
            "idx=2: 0*Inf should be NaN, got {}",
            out[2]
        );
    }

    /// S11 — scalar = -Inf
    #[test]
    fn t5_3_scalar_neg_inf_propagates() {
        let x = vec![1.0 as ScalarType, -2.0 as ScalarType, 0.0 as ScalarType];
        let mut out = vec![0.0 as ScalarType; 3];
        vec_scale(&x, ScalarType::NEG_INFINITY, &mut out);
        assert!(
            out[0].is_infinite() && out[0] < 0.0,
            "idx=0: 1*(-Inf) should be -Inf, got {}",
            out[0]
        );
        assert!(
            out[1].is_infinite() && out[1] > 0.0,
            "idx=1: -2*(-Inf) should be +Inf, got {}",
            out[1]
        );
        assert!(
            out[2].is_nan(),
            "idx=2: 0*(-Inf) should be NaN, got {}",
            out[2]
        );
    }

    /// V07 — x contains NaN
    #[test]
    fn t5_4_x_nan_propagates() {
        let x = vec![ScalarType::NAN, 1.0 as ScalarType, 2.0 as ScalarType];
        let mut out = vec![0.0 as ScalarType; 3];
        vec_scale(&x, 2.0 as ScalarType, &mut out);
        assert!(
            out[0].is_nan(),
            "idx=0: NaN*2 should be NaN, got {}",
            out[0]
        );
        assert_eq!(out[1], 2.0 as ScalarType);
        assert_eq!(out[2], 4.0 as ScalarType);
    }

    /// V08 — x contains +Inf and -Inf
    #[test]
    fn t5_5_x_inf_propagates() {
        let x = vec![
            ScalarType::INFINITY,
            ScalarType::NEG_INFINITY,
            1.0 as ScalarType,
        ];
        let mut out = vec![0.0 as ScalarType; 3];
        vec_scale(&x, 2.0 as ScalarType, &mut out);
        assert!(
            out[0].is_infinite() && out[0] > 0.0,
            "idx=0: +Inf*2 should be +Inf, got {}",
            out[0]
        );
        assert!(
            out[1].is_infinite() && out[1] < 0.0,
            "idx=1: -Inf*2 should be -Inf, got {}",
            out[1]
        );
        assert_eq!(out[2], 2.0 as ScalarType);
    }

    /// V09 — x mixed 0 and NaN, scalar is Inf
    /// 0 * Inf = NaN; NaN * Inf = NaN — all outputs must be NaN
    #[test]
    fn t5_6_zero_and_nan_mixed() {
        let x = vec![
            0.0 as ScalarType,
            ScalarType::NAN,
            0.0 as ScalarType,
            ScalarType::NAN,
        ];
        let mut out = vec![0.0 as ScalarType; 4];
        vec_scale(&x, ScalarType::INFINITY, &mut out);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.is_nan(),
                "idx={i}: 0*Inf or NaN*Inf should be NaN, got {v}"
            );
        }
    }

    /// Negative zero preservation: -0.0 * 1.0 = -0.0
    #[test]
    fn t5_7_negative_zero_preserved() {
        let x = vec![-0.0 as ScalarType, 0.0 as ScalarType, 1.0 as ScalarType];
        let mut out = vec![0.0 as ScalarType; 3];
        vec_scale(&x, 1.0 as ScalarType, &mut out);
        assert!(
            out[0].is_sign_negative() && out[0] == 0.0,
            "idx=0: -0*1 should be -0, got {} (sign_neg={})",
            out[0],
            out[0].is_sign_negative()
        );
        assert!(out[1] == 0.0, "idx=1: 0*1 should be 0, got {}", out[1]);
        assert_eq!(out[2], 1.0 as ScalarType);
    }
}

// T6 — Aliasing / in-place behavior (A01-A02)
mod aliasing_inplace {
    use super::*;

    /// A01 — out and x are the same slice (in-place scaling)
    /// `vec_scale` reads `x[i]` before writing `out[i]` — no read-modify-write dependency —
    /// so aliasing is safe. This test locks down that contract.
    #[test]
    fn t6_1_inplace_aliasing_safe() {
        let n = 100;
        let mut data = arange(n);
        let original = data.clone();

        // SAFETY: pass the same memory as x and out via raw pointers, bypassing the borrow checker.
        // `vec_scale` is specified to read `x[i]` before writing `out[i]` for any given index,
        // so overlapping slices are safe.
        let ptr = data.as_mut_ptr();
        let len = data.len();
        let shared_x = unsafe { std::slice::from_raw_parts(ptr as *const ScalarType, len) };
        let shared_out = unsafe { std::slice::from_raw_parts_mut(ptr, len) };

        vec_scale(shared_x, 2.0 as ScalarType, shared_out);

        for (i, &v) in data.iter().enumerate() {
            assert_eq!(
                v,
                original[i] * 2.0 as ScalarType,
                "alias bug at idx={i}: got {v}, expected {}",
                original[i] * 2.0 as ScalarType
            );
        }
    }

    /// A02 — out and x are independent buffers: x must remain untouched
    #[test]
    fn t6_2_independent_buffers_no_crosstalk() {
        let n = 16;
        let x = arange(n);
        let mut out = fill(n, -1.0 as ScalarType);
        let mut expected = vec![0.0 as ScalarType; n];
        scale_oracle(&x, 3.0 as ScalarType, &mut expected);
        vec_scale(&x, 3.0 as ScalarType, &mut out);
        assert_eq!(out, expected);
        // x must remain untouched
        assert_eq!(x, arange(n));
    }
}

// T7 — Cross-path consistency (X01-X02)
mod consistency {
    use super::*;

    /// X01 — 100 consecutive calls with the same input produce identical results
    /// (detects register state leaks or use of uninitialized memory)
    #[test]
    fn t7_1_consecutive_calls_deterministic() {
        let n = 256;
        let x = arange(n);
        let scalar = 1.5 as ScalarType;
        let mut first = vec![0.0 as ScalarType; n];
        vec_scale(&x, scalar, &mut first);
        for _ in 0..100 {
            let mut out = vec![0.0 as ScalarType; n];
            vec_scale(&x, scalar, &mut out);
            assert_eq!(
                out, first,
                "non-deterministic result across consecutive calls"
            );
        }
    }

    /// X02 — SIMD path vs ANSI oracle
    /// (follows project convention: `vec_scale_ansi` is private, so we use a local naive oracle
    /// as the reference — same as `test_vec_dot.rs` and `test_vec_scaled_add_inplace.rs`)
    #[test]
    fn t7_2_matches_scalar_oracle_random() {
        let n = 1000;
        let x: Vec<ScalarType> = (0..n)
            .map(|i| ((i as f64) * 0.9876 + 1.234).sin() as ScalarType)
            .collect();
        let scalar = -2.5 as ScalarType;
        let mut out = vec![0.0 as ScalarType; n];
        let mut expected = vec![0.0 as ScalarType; n];
        scale_oracle(&x, scalar, &mut expected);
        vec_scale(&x, scalar, &mut out);
        for (i, (&a, &e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= epsilon(),
                "idx={i}: SIMD {a} != scalar oracle {e}, diff {}",
                (a - e).abs()
            );
        }
    }
}

// T8 — L-BFGS integration scenarios (M01-M05)
mod lbfgs_integration {
    use super::*;

    /// M01 — Gradient scaled by learning rate
    #[test]
    fn t8_1_gradient_scaled_by_lr() {
        let n = 1024;
        let grad: Vec<ScalarType> = (0..n)
            .map(|i| ((i as f64) * 0.01).sin() as ScalarType)
            .collect();
        let lr = 0.01 as ScalarType;
        let mut scaled_grad = vec![0.0 as ScalarType; n];
        vec_scale(&grad, lr, &mut scaled_grad);
        for (i, (&g, &s)) in grad.iter().zip(scaled_grad.iter()).enumerate() {
            assert!(
                (s - g * lr).abs() <= epsilon(),
                "idx={i}: scaled_grad {s} != grad*lr {}",
                g * lr
            );
        }
    }

    /// M02 — Weight decay (in-place × decay factor < 1)
    /// Simulates the common `w *= decay` L-BFGS update step.
    #[test]
    fn t8_2_weight_decay_inplace() {
        let n = 512;
        let mut w = arange(n);
        let original = w.clone();
        let decay = 0.99 as ScalarType;

        // SAFETY: in-place aliasing — see t6_1 rationale
        let ptr = w.as_mut_ptr();
        let len = w.len();
        let shared_x = unsafe { std::slice::from_raw_parts(ptr as *const ScalarType, len) };
        let shared_out = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        vec_scale(shared_x, decay, shared_out);

        for (i, &v) in w.iter().enumerate() {
            assert!(
                (v - original[i] * decay).abs() <= epsilon(),
                "weight decay idx={i}: got {v}, expected {}",
                original[i] * decay
            );
        }
    }

    /// M03 — Zero a buffer (scalar = 0)
    #[test]
    fn t8_3_zero_buffer() {
        let n = 256;
        let x = arange(n);
        let mut zeros = vec![9.0 as ScalarType; n];
        vec_scale(&x, 0.0 as ScalarType, &mut zeros);
        assert_eq!(zeros, vec![0.0 as ScalarType; n]);
    }

    /// M04 — Feature normalization (long vector × 1/max)
    #[test]
    fn t8_4_feature_normalization() {
        let n = 784; // MNIST-like
        let features = arange(n);
        let max_val = (n - 1) as ScalarType;
        let inv_max = 1.0 as ScalarType / max_val;
        let mut normalized = vec![0.0 as ScalarType; n];
        vec_scale(&features, inv_max, &mut normalized);
        // Boundary cases are easy to verify
        assert_eq!(normalized[0], 0.0 as ScalarType);
        assert!(
            (normalized[n - 1] - 1.0 as ScalarType).abs() <= epsilon(),
            "last element should be 1.0, got {}",
            normalized[n - 1]
        );
    }

    /// M05 — Common ML dimensions: 4, 8, 16, 32, 64, 128, 256, 784, 1024, 4096
    /// Covers IRIS features (4), MNIST (784), typical hidden layers (128/256/1024), etc.
    #[test]
    fn t8_5_common_ml_dimensions() {
        for &n in &[4usize, 8, 16, 32, 64, 128, 256, 784, 1024, 4096] {
            let x = arange(n);
            let scalar = 0.5 as ScalarType;
            let mut out = vec![0.0 as ScalarType; n];
            let mut expected = vec![0.0 as ScalarType; n];
            scale_oracle(&x, scalar, &mut expected);
            vec_scale(&x, scalar, &mut out);
            for (i, (&a, &e)) in out.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (a - e).abs() <= epsilon(),
                    "n={n}, idx={i}: got {a}, expected {e}, diff {}",
                    (a - e).abs()
                );
            }
        }
    }
}
