use lbfgs_rs::infra::math::ops_neon::vec_scaled_add;
use lbfgs_rs::shared::numeric::ScalarType;

// ── Helpers ──

/// Naive scalar reference implementation: out[i] = x[i] * scalar + y[i].
/// Order matches `vec_scaled_add_ansi`.
fn scaled_add_oracle(
    x: &[ScalarType],
    y: &[ScalarType],
    scalar: ScalarType,
    out: &mut [ScalarType],
) {
    for i in 0..x.len() {
        out[i] = x[i] * scalar + y[i];
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

// T0 — Input validation / Contract (C01-C05)

/// C01 — x longer than y: triggers panic in debug builds.
#[test]
#[should_panic(expected = "x and y must have the same length")]
#[cfg(debug_assertions)]
fn t0_1_x_longer_than_y_panics() {
    let x = vec![1.0 as ScalarType; 4];
    let y = vec![1.0 as ScalarType; 3];
    let mut out = vec![0.0 as ScalarType; 4];
    vec_scaled_add(&x, &y, 1.0 as ScalarType, &mut out);
}

/// C02 — x shorter than y: triggers panic in debug builds.
#[test]
#[should_panic(expected = "x and y must have the same length")]
#[cfg(debug_assertions)]
fn t0_2_x_shorter_than_y_panics() {
    let x = vec![1.0 as ScalarType; 3];
    let y = vec![1.0 as ScalarType; 4];
    let mut out = vec![0.0 as ScalarType; 3];
    vec_scaled_add(&x, &y, 1.0 as ScalarType, &mut out);
}

/// C03 — out shorter than x: triggers panic in debug builds.
#[test]
#[should_panic(expected = "output vector must have the same length as input")]
#[cfg(debug_assertions)]
fn t0_3_out_shorter_than_x_panics() {
    let x = vec![1.0 as ScalarType; 4];
    let y = vec![1.0 as ScalarType; 4];
    let mut out = vec![0.0 as ScalarType; 3];
    vec_scaled_add(&x, &y, 1.0 as ScalarType, &mut out);
}

/// C04 — out longer than x: triggers panic in debug builds.
#[test]
#[should_panic(expected = "output vector must have the same length as input")]
#[cfg(debug_assertions)]
fn t0_4_out_longer_than_x_panics() {
    let x = vec![1.0 as ScalarType; 3];
    let y = vec![1.0 as ScalarType; 3];
    let mut out = vec![0.0 as ScalarType; 4];
    vec_scaled_add(&x, &y, 1.0 as ScalarType, &mut out);
}

/// C05 — All vectors empty: must not panic.
#[test]
fn t0_5_all_empty_no_panic() {
    let mut out: Vec<ScalarType> = vec![];
    vec_scaled_add(&[], &[], 2.0 as ScalarType, &mut out);
    assert_eq!(out.len(), 0);
}

// T1 — Scalar value coverage (S01-S11)
mod scalar_values {
    use super::*;

    /// Check `vec_scaled_add` on a small fixed x and y against the oracle.
    fn check(scalar: ScalarType) {
        let x = vec![
            1.5 as ScalarType,
            -2.0 as ScalarType,
            3.0 as ScalarType,
            0.0 as ScalarType,
        ];
        let y = vec![
            0.5 as ScalarType,
            1.0 as ScalarType,
            -1.0 as ScalarType,
            2.0 as ScalarType,
        ];
        let mut out = vec![0.0 as ScalarType; x.len()];
        let mut expected = vec![0.0 as ScalarType; x.len()];
        scaled_add_oracle(&x, &y, scalar, &mut expected);
        vec_scaled_add(&x, &y, scalar, &mut out);
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

    /// S01-specific: scalar = 0.0 => out == y (scalar term vanishes).
    #[test]
    fn t1_9_scalar_zero_yields_y() {
        let n = 16;
        let x = arange(n);
        let y = fill(n, 5.0 as ScalarType);
        let mut out = vec![0.0 as ScalarType; n];
        vec_scaled_add(&x, &y, 0.0 as ScalarType, &mut out);
        assert_eq!(out, y);
    }

    /// S02-specific: scalar = 1.0 => out == x + y (most common FMA form).
    #[test]
    fn t1_10_scalar_one_is_addition() {
        let n = 16;
        let x = arange(n);
        let y = fill(n, 10.0 as ScalarType);
        let mut out = vec![0.0 as ScalarType; n];
        vec_scaled_add(&x, &y, 1.0 as ScalarType, &mut out);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, (i as ScalarType) + 10.0 as ScalarType, "idx={i}");
        }
    }

    /// S03-specific: scalar = -1.0 => out == y - x (difference vector).
    #[test]
    fn t1_11_scalar_neg_one_is_subtraction() {
        let n = 16;
        let x = arange(n);
        let y = fill(n, 100.0 as ScalarType);
        let mut out = vec![0.0 as ScalarType; n];
        vec_scaled_add(&x, &y, -1.0 as ScalarType, &mut out);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 100.0 as ScalarType - (i as ScalarType), "idx={i}");
        }
    }

    /// S09 — scalar = NaN: all outputs must be NaN
    #[test]
    fn t1_12_scalar_nan_propagates() {
        let x = vec![1.0 as ScalarType, 2.0 as ScalarType, 3.0 as ScalarType];
        let y = vec![1.0 as ScalarType, 2.0 as ScalarType, 3.0 as ScalarType];
        let mut out = vec![0.0 as ScalarType; 3];
        vec_scaled_add(&x, &y, ScalarType::NAN, &mut out);
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_nan(), "scalar=NaN: idx={i} expected NaN, got {v}");
        }
    }

    /// S10 — scalar = +Inf: positive×Inf=+Inf, negative×Inf=-Inf, 0×Inf=NaN, then + y
    #[test]
    fn t1_13_scalar_pos_inf_propagates() {
        let x = vec![1.0 as ScalarType, -2.0 as ScalarType, 0.0 as ScalarType];
        let y = vec![1.0 as ScalarType, 1.0 as ScalarType, 1.0 as ScalarType];
        let mut out = vec![0.0 as ScalarType; 3];
        vec_scaled_add(&x, &y, ScalarType::INFINITY, &mut out);
        assert!(
            out[0].is_infinite() && out[0] > 0.0,
            "idx=0: +1*Inf+1 should be +Inf, got {}",
            out[0]
        );
        assert!(
            out[1].is_infinite() && out[1] < 0.0,
            "idx=1: -2*Inf+1 should be -Inf, got {}",
            out[1]
        );
        assert!(
            out[2].is_nan(),
            "idx=2: 0*Inf+1 should be NaN, got {}",
            out[2]
        );
    }

    /// S11 — scalar = -Inf
    #[test]
    fn t1_14_scalar_neg_inf_propagates() {
        let x = vec![1.0 as ScalarType, -2.0 as ScalarType, 0.0 as ScalarType];
        let y = vec![1.0 as ScalarType, 1.0 as ScalarType, 1.0 as ScalarType];
        let mut out = vec![0.0 as ScalarType; 3];
        vec_scaled_add(&x, &y, ScalarType::NEG_INFINITY, &mut out);
        assert!(
            out[0].is_infinite() && out[0] < 0.0,
            "idx=0: 1*(-Inf)+1 should be -Inf, got {}",
            out[0]
        );
        assert!(
            out[1].is_infinite() && out[1] > 0.0,
            "idx=1: -2*(-Inf)+1 should be +Inf, got {}",
            out[1]
        );
        assert!(
            out[2].is_nan(),
            "idx=2: 0*(-Inf)+1 should be NaN, got {}",
            out[2]
        );
    }
}

// T2 — Length boundary / three-tier tail handling (L01-L14)
mod boundary_alignment {
    use super::*;

    /// Assert `vec_scaled_add` matches oracle for `arange(len) * scalar + fill(len, 1.0)`.
    fn assert_scaled_add(len: usize, scalar: ScalarType) {
        let x = arange(len);
        let y = fill(len, 1.0 as ScalarType);
        let mut out = vec![0.0 as ScalarType; len];
        let mut expected = vec![0.0 as ScalarType; len];
        scaled_add_oracle(&x, &y, scalar, &mut expected);
        vec_scaled_add(&x, &y, scalar, &mut out);
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
        assert_scaled_add(0, s());
    } // L01: empty
    #[test]
    fn t2_2_len_1() {
        assert_scaled_add(1, s());
    } // L02: pure scalar tail
    #[test]
    fn t2_3_len_3() {
        assert_scaled_add(3, s());
    } // L03: < 1 SIMD lane

    /// L04 — exactly 1 SIMD lane (f32: 4, f64: 2)
    #[test]
    fn t2_4_one_lane() {
        let n = if cfg!(feature = "f32") { 4 } else { 2 };
        assert_scaled_add(n, s());
    }

    /// L05 — f32-only: 1 lane + 1 scalar tail (len=5)
    #[test]
    #[cfg(feature = "f32")]
    fn t2_5_f32_one_lane_plus_one_scalar() {
        assert_scaled_add(5, s());
    }

    /// L06 — f32-only: 1 lane + 3 scalar tail (len=7)
    #[test]
    #[cfg(feature = "f32")]
    fn t2_6_f32_one_lane_plus_three_scalar() {
        assert_scaled_add(7, s());
    }

    /// L07 — len=15: f32 is 3 lanes + 3 scalars; f64 is 7 lanes + 1 scalar
    #[test]
    fn t2_7_len_15() {
        assert_scaled_add(15, s());
    }

    /// L08 — exactly 1 unroll block (f32: 16, f64: 4)
    #[test]
    fn t2_8_one_block() {
        let n = if cfg!(feature = "f32") { 16 } else { 4 };
        assert_scaled_add(n, s());
    }

    /// L09 — 1 block + 1 scalar (f32: 17, f64: 5)
    #[test]
    fn t2_9_one_block_plus_one_scalar() {
        let n = if cfg!(feature = "f32") { 17 } else { 5 };
        assert_scaled_add(n, s());
    }

    /// L10 — f32-only: 1 block + 1 lane, 0 scalar (len=20)
    #[test]
    #[cfg(feature = "f32")]
    fn t2_10_f32_one_block_plus_one_lane() {
        assert_scaled_add(20, s());
    }

    /// L10 — f64-only: 1 block + 1 lane, 0 scalar (len=6)
    #[test]
    #[cfg(feature = "f64")]
    fn t2_11_f64_one_block_plus_one_lane() {
        assert_scaled_add(6, s());
    }

    /// L11 — 1 block + 1 lane + 1 scalar (f32: 21, f64: 7) — hits all three tail tiers
    #[test]
    fn t2_12_three_tier_tail() {
        let n = if cfg!(feature = "f32") { 21 } else { 7 };
        assert_scaled_add(n, s());
    }

    /// L12 — 2 full blocks (f32: 32, f64: 8)
    #[test]
    fn t2_13_two_blocks() {
        let n = if cfg!(feature = "f32") { 32 } else { 8 };
        assert_scaled_add(n, s());
    }

    /// L13 — Mixed: multiple blocks + multiple lanes + scalars
    #[test]
    fn t2_14_len_100() {
        assert_scaled_add(100, s());
    }

    /// L14 — Large scale to catch cumulative drift
    #[test]
    fn t2_15_len_4096() {
        assert_scaled_add(4096, s());
    }
}

// T3 — x vector content coverage (V01-V09)
mod input_content_x {
    use super::*;

    /// Run `vec_scaled_add` on the given x, y, scalar and assert against oracle.
    fn run_with(x: &[ScalarType], y: &[ScalarType], scalar: ScalarType) -> Vec<ScalarType> {
        let mut out = vec![0.0 as ScalarType; x.len()];
        let mut expected = vec![0.0 as ScalarType; x.len()];
        scaled_add_oracle(x, y, scalar, &mut expected);
        vec_scaled_add(x, y, scalar, &mut out);
        for (i, (&a, &e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= epsilon(),
                "idx={i}: got {a}, expected {e}, diff {}",
                (a - e).abs()
            );
        }
        out
    }

    /// V01 — x all zeros: out == y
    #[test]
    fn t3_1_x_all_zeros() {
        let y = fill(16, 7.0 as ScalarType);
        let out = run_with(&fill(16, 0.0 as ScalarType), &y, 2.0 as ScalarType);
        assert_eq!(out, y);
    }

    /// V02 — x all ones: out[i] == scalar + y[i]
    #[test]
    fn t3_2_x_all_ones() {
        let y = fill(16, 3.0 as ScalarType);
        let out = run_with(&fill(16, 1.0 as ScalarType), &y, 2.0 as ScalarType);
        assert_eq!(out, fill(16, 5.0 as ScalarType));
    }

    /// V03 — x all negatives
    #[test]
    fn t3_3_x_all_negatives() {
        let x: Vec<ScalarType> = (0..16).map(|i| -((i + 1) as ScalarType)).collect();
        let y = fill(16, 1.0 as ScalarType);
        let out = run_with(&x, &y, 2.0 as ScalarType);
        let expected: Vec<ScalarType> = x
            .iter()
            .map(|&v| v * 2.0 as ScalarType + 1.0 as ScalarType)
            .collect();
        assert_eq!(out, expected);
    }

    /// V04 — x alternating signs [1, -1, 1, -1, ...]
    #[test]
    fn t3_4_x_alternating_sign() {
        let x: Vec<ScalarType> = (0..16)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 } as ScalarType)
            .collect();
        let y = fill(16, 0.5 as ScalarType);
        let out = run_with(&x, &y, 2.0 as ScalarType);
        let expected: Vec<ScalarType> = x
            .iter()
            .map(|&v| v * 2.0 as ScalarType + 0.5 as ScalarType)
            .collect();
        assert_eq!(out, expected);
    }

    /// V05 — x monotonically increasing [1, 2, 3, ..., n]
    #[test]
    fn t3_5_x_monotonic_increase() {
        let x = arange(20);
        let y = fill(20, 1.0 as ScalarType);
        let out = run_with(&x, &y, 0.5 as ScalarType);
        let expected: Vec<ScalarType> = x
            .iter()
            .map(|&v| v * 0.5 as ScalarType + 1.0 as ScalarType)
            .collect();
        assert_eq!(out, expected);
    }

    /// V06 — x mixed magnitude [1e-10, 1e10, 1.0, -1.0]
    #[test]
    fn t3_6_x_mixed_magnitude() {
        let x = vec![
            1e-10 as ScalarType,
            1e10 as ScalarType,
            1.0 as ScalarType,
            -1.0 as ScalarType,
        ];
        let y = fill(4, 1.0 as ScalarType);
        let out = run_with(&x, &y, 2.0 as ScalarType);
        let expected = vec![
            2e-10 as ScalarType + 1.0 as ScalarType,
            2e10 as ScalarType + 1.0 as ScalarType,
            3.0 as ScalarType,
            -1.0 as ScalarType,
        ];
        for (i, (&a, &e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= e.abs() * epsilon(),
                "idx={i}: got {a}, expected {e}"
            );
        }
    }

    /// V07 — x contains NaN
    #[test]
    fn t3_7_x_nan_propagates() {
        let x = vec![ScalarType::NAN, 1.0 as ScalarType, 2.0 as ScalarType];
        let y = fill(3, 1.0 as ScalarType);
        let mut out = vec![0.0 as ScalarType; 3];
        vec_scaled_add(&x, &y, 2.0 as ScalarType, &mut out);
        assert!(
            out[0].is_nan(),
            "idx=0: NaN*2+1 should be NaN, got {}",
            out[0]
        );
        assert_eq!(out[1], 3.0 as ScalarType);
        assert_eq!(out[2], 5.0 as ScalarType);
    }

    /// V08 — x contains +Inf and -Inf
    #[test]
    fn t3_8_x_inf_propagates() {
        let x = vec![
            ScalarType::INFINITY,
            ScalarType::NEG_INFINITY,
            1.0 as ScalarType,
        ];
        let y = fill(3, 1.0 as ScalarType);
        let mut out = vec![0.0 as ScalarType; 3];
        vec_scaled_add(&x, &y, 2.0 as ScalarType, &mut out);
        assert!(
            out[0].is_infinite() && out[0] > 0.0,
            "idx=0: +Inf*2+1 should be +Inf, got {}",
            out[0]
        );
        assert!(
            out[1].is_infinite() && out[1] < 0.0,
            "idx=1: -Inf*2+1 should be -Inf, got {}",
            out[1]
        );
        assert_eq!(out[2], 3.0 as ScalarType);
    }

    /// V09 — x mixed 0 and NaN, scalar finite
    /// 0 * scalar = 0 (when scalar is finite), NaN * scalar = NaN
    #[test]
    fn t3_9_x_zero_and_nan_mixed() {
        let x = vec![
            0.0 as ScalarType,
            ScalarType::NAN,
            0.0 as ScalarType,
            ScalarType::NAN,
        ];
        let y = fill(4, 1.0 as ScalarType);
        let mut out = vec![0.0 as ScalarType; 4];
        vec_scaled_add(&x, &y, 2.0 as ScalarType, &mut out);
        assert_eq!(
            out[0], 1.0 as ScalarType,
            "idx=0: 0*2+1 should be 1, got {}",
            out[0]
        );
        assert!(
            out[1].is_nan(),
            "idx=1: NaN*2+1 should be NaN, got {}",
            out[1]
        );
        assert_eq!(
            out[2], 1.0 as ScalarType,
            "idx=2: 0*2+1 should be 1, got {}",
            out[2]
        );
        assert!(
            out[3].is_nan(),
            "idx=3: NaN*2+1 should be NaN, got {}",
            out[3]
        );
    }
}

// T4 — y vector content coverage (Y01-Y09) — unique to vec_scaled_add
mod y_content {
    use super::*;

    /// Run on the given x, y, scalar and assert against oracle.
    fn run_with(x: &[ScalarType], y: &[ScalarType], scalar: ScalarType) -> Vec<ScalarType> {
        let mut out = vec![0.0 as ScalarType; x.len()];
        let mut expected = vec![0.0 as ScalarType; x.len()];
        scaled_add_oracle(x, y, scalar, &mut expected);
        vec_scaled_add(x, y, scalar, &mut out);
        for (i, (&a, &e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= epsilon(),
                "idx={i}: got {a}, expected {e}, diff {}",
                (a - e).abs()
            );
        }
        out
    }

    /// Y01 — y all zeros: degenerates to vec_scale semantics, out == x * scalar
    #[test]
    fn t4_1_y_all_zeros() {
        let x = arange(16);
        let out = run_with(&x, &fill(16, 0.0 as ScalarType), 2.0 as ScalarType);
        let expected: Vec<ScalarType> = x.iter().map(|&v| v * 2.0 as ScalarType).collect();
        assert_eq!(out, expected);
    }

    /// Y02 — y all ones: out[i] == x[i] * scalar + 1
    #[test]
    fn t4_2_y_all_ones() {
        let x = arange(16);
        let out = run_with(&x, &fill(16, 1.0 as ScalarType), 2.0 as ScalarType);
        let expected: Vec<ScalarType> = x
            .iter()
            .map(|&v| v * 2.0 as ScalarType + 1.0 as ScalarType)
            .collect();
        assert_eq!(out, expected);
    }

    /// Y03 — y all negatives
    #[test]
    fn t4_3_y_all_negatives() {
        let x = arange(16);
        let y: Vec<ScalarType> = (0..16).map(|i| -((i + 1) as ScalarType)).collect();
        let out = run_with(&x, &y, 2.0 as ScalarType);
        let expected: Vec<ScalarType> = (0..16)
            .map(|i| (i as ScalarType) * 2.0 as ScalarType - ((i + 1) as ScalarType))
            .collect();
        assert_eq!(out, expected);
    }

    /// Y04 — y alternating signs
    #[test]
    fn t4_4_y_alternating_sign() {
        let x = arange(16);
        let y: Vec<ScalarType> = (0..16)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 } as ScalarType)
            .collect();
        let out = run_with(&x, &y, 2.0 as ScalarType);
        let expected: Vec<ScalarType> = (0..16)
            .map(|i| {
                (i as ScalarType) * 2.0 as ScalarType
                    + if i % 2 == 0 { 1.0 } else { -1.0 } as ScalarType
            })
            .collect();
        assert_eq!(out, expected);
    }

    /// Y05 — y monotonically increasing
    #[test]
    fn t4_5_y_monotonic_increase() {
        let x = fill(20, 2.0 as ScalarType);
        let y = arange(20);
        let out = run_with(&x, &y, 0.5 as ScalarType);
        let expected: Vec<ScalarType> = y.iter().map(|&v| 1.0 as ScalarType + v).collect();
        assert_eq!(out, expected);
    }

    /// Y06 — y contains NaN: addition propagates NaN
    #[test]
    fn t4_6_y_nan_propagates() {
        let x = vec![1.0 as ScalarType, 2.0 as ScalarType, 3.0 as ScalarType];
        let y = vec![ScalarType::NAN, 2.0 as ScalarType, 3.0 as ScalarType];
        let mut out = vec![0.0 as ScalarType; 3];
        vec_scaled_add(&x, &y, 2.0 as ScalarType, &mut out);
        assert!(
            out[0].is_nan(),
            "idx=0: 1*2+NaN should be NaN, got {}",
            out[0]
        );
        assert_eq!(out[1], 6.0 as ScalarType);
        assert_eq!(out[2], 9.0 as ScalarType);
    }

    /// Y07 — y contains +Inf / -Inf
    #[test]
    fn t4_7_y_inf_propagates() {
        let x = vec![1.0 as ScalarType, -1.0 as ScalarType, 0.0 as ScalarType];
        let y = vec![
            ScalarType::INFINITY,
            ScalarType::NEG_INFINITY,
            ScalarType::INFINITY,
        ];
        let mut out = vec![0.0 as ScalarType; 3];
        vec_scaled_add(&x, &y, 2.0 as ScalarType, &mut out);
        assert!(
            out[0].is_infinite() && out[0] > 0.0,
            "idx=0: 2+Inf should be +Inf, got {}",
            out[0]
        );
        assert!(
            out[1].is_infinite() && out[1] < 0.0,
            "idx=1: -2+(-Inf) should be -Inf, got {}",
            out[1]
        );
        assert!(
            out[2].is_infinite() && out[2] > 0.0,
            "idx=2: 0+Inf should be +Inf, got {}",
            out[2]
        );
    }

    /// Y08 — x*scalar and y have vastly different magnitudes (large eats small)
    #[test]
    fn t4_8_y_magnitude_dominance() {
        let n = 16;
        let x = fill(n, 1e-10 as ScalarType);
        let y = fill(n, 1e10 as ScalarType);
        let out = run_with(&x, &y, 1.0 as ScalarType);
        // x*scalar = 1e-10, y = 1e10 → result ≈ 1e10 (1e-10 is absorbed)
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - 1e10 as ScalarType).abs() <= 1e10 as ScalarType * epsilon(),
                "idx={i}: large eats small, got {v}, expected ~1e10"
            );
        }
    }

    /// Y09 — Catastrophic cancellation: x*scalar and y are close in magnitude with opposite signs
    #[test]
    fn t4_9_y_catastrophic_cancellation() {
        let n = 16;
        let x: Vec<ScalarType> = (0..n).map(|i| (1e10 + (i as f64)) as ScalarType).collect();
        let y: Vec<ScalarType> = (0..n).map(|_| -1e10 as ScalarType).collect();
        let out = run_with(&x, &y, 1.0 as ScalarType);
        // Result = i (0..16), but floating-point cancellation may cause precision loss
        // Compare SIMD with oracle using relative tolerance (both should agree)
        let expected: Vec<ScalarType> = (0..n).map(|i| (i as f64) as ScalarType).collect();
        for (i, (&a, &e)) in out.iter().zip(expected.iter()).enumerate() {
            let tol = if e.abs() > 1.0 as ScalarType {
                e.abs() * epsilon()
            } else {
                epsilon() * 100.0 // small residual after cancellation, relax absolute tolerance
            };
            assert!(
                (a - e).abs() <= tol,
                "cancellation idx={i}: got {a}, expected ~{e}"
            );
        }
    }
}

// T5 — Numeric precision (P01-P05)
mod numeric_precision {
    use super::*;

    /// P01 — General case: element-wise match against oracle for n=1000 pseudo-random vector
    #[test]
    fn t5_1_matches_oracle_general() {
        let n = 1000;
        let x: Vec<ScalarType> = (0..n)
            .map(|i| ((i as f64) * 1.234 + 0.5).sin() as ScalarType)
            .collect();
        let y: Vec<ScalarType> = (0..n)
            .map(|i| ((i as f64) * 0.567 + 1.5).cos() as ScalarType)
            .collect();
        let scalar = 1.7 as ScalarType;
        let mut out = vec![0.0 as ScalarType; n];
        let mut expected = vec![0.0 as ScalarType; n];
        scaled_add_oracle(&x, &y, scalar, &mut expected);
        vec_scaled_add(&x, &y, scalar, &mut out);
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
    fn t5_2_extreme_mixed_values() {
        let x = vec![
            1e-30 as ScalarType,
            1e30 as ScalarType,
            -1e-30 as ScalarType,
            -1e30 as ScalarType,
        ];
        let y = vec![
            1e30 as ScalarType,
            1e-30 as ScalarType,
            -1e30 as ScalarType,
            -1e-30 as ScalarType,
        ];
        let scalar = 1e5 as ScalarType;
        let mut out = vec![0.0 as ScalarType; 4];
        let mut expected = vec![0.0 as ScalarType; 4];
        scaled_add_oracle(&x, &y, scalar, &mut expected);
        vec_scaled_add(&x, &y, scalar, &mut out);
        for (i, (&a, &e)) in out.iter().zip(expected.iter()).enumerate() {
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

    /// P03 — Integer-representable values × integer scalar + integer y: require strict equality
    #[test]
    fn t5_3_integer_values_strict_equal() {
        let n = 64;
        let x: Vec<ScalarType> = (0..n).map(|i| (i as i32) as ScalarType).collect();
        let y: Vec<ScalarType> = (0..n).map(|i| ((i * 10) as i32) as ScalarType).collect();
        let scalar = 4_i32 as ScalarType;
        let mut out = vec![0.0 as ScalarType; n];
        vec_scaled_add(&x, &y, scalar, &mut out);
        let expected: Vec<ScalarType> =
            (0..n).map(|i| ((i * 4) + (i * 10)) as ScalarType).collect();
        assert_eq!(out, expected);
    }

    /// P04 — Consecutive call consistency: two calls with same input produce identical results
    #[test]
    fn t5_4_consecutive_call_consistency() {
        let n = 100;
        let x = arange(n);
        let y = fill(n, 5.0 as ScalarType);
        let scalar = 1.5 as ScalarType;
        let mut out1 = vec![0.0 as ScalarType; n];
        vec_scaled_add(&x, &y, scalar, &mut out1);
        let mut out2 = vec![0.0 as ScalarType; n];
        vec_scaled_add(&x, &y, scalar, &mut out2);
        assert_eq!(
            out1, out2,
            "consecutive calls must produce identical results"
        );
    }

    /// P05 — FMA precision tolerance: NEON vfmaq single rounding vs ANSI double rounding may differ by 1 ULP
    /// Uses 0.1 (inexact in binary floating-point) to trigger FMA-sensitive scenario.
    /// Does not require strict equality, only requires matching oracle within tolerance.
    #[test]
    fn t5_5_fma_precision_tolerance() {
        // 0.1 is inexact in binary floating-point; 0.1*0.1 rounds differently from 0.01.
        // FMA path: y + x*scalar with single rounding
        // Non-FMA path (oracle): x*scalar rounds first, then + y rounds again
        // The two paths may differ by 1 ULP — use tolerance instead of strict equality.
        let n = 100;
        let x = fill(n, 0.1 as ScalarType);
        let y = fill(n, -0.01 as ScalarType);
        let scalar = 0.1 as ScalarType;
        let mut out = vec![0.0 as ScalarType; n];
        let mut expected = vec![0.0 as ScalarType; n];
        scaled_add_oracle(&x, &y, scalar, &mut expected);
        vec_scaled_add(&x, &y, scalar, &mut out);
        for (i, (&a, &e)) in out.iter().zip(expected.iter()).enumerate() {
            // Result is close to 0, use absolute tolerance (relaxed to accommodate 1 ULP difference)
            assert!(
                (a - e).abs() <= epsilon(),
                "FMA-sensitive idx={i}: got {a}, oracle {e}, diff {}",
                (a - e).abs()
            );
        }
    }
}

// T6 — Memory aliasing / in-place behavior (A01-A05) — unique complexity of vec_scaled_add
mod aliasing {
    use super::*;

    /// A01 — out == x (in-place x)
    /// The implementation must read x[i] and y[i] before writing out[i].
    #[test]
    fn t6_1_out_equals_x() {
        let n = 100;
        let mut data = arange(n);
        let original = data.clone();
        let y = fill(n, 5.0 as ScalarType);
        let scalar = 2.0 as ScalarType;

        // SAFETY: pass the same memory as both x and out, bypassing the borrow checker.
        // `vec_scaled_add` is specified to read x[i] and y[i] before writing out[i] for any given index.
        let ptr = data.as_mut_ptr();
        let len = data.len();
        let shared_x = unsafe { std::slice::from_raw_parts(ptr as *const ScalarType, len) };
        let shared_out = unsafe { std::slice::from_raw_parts_mut(ptr, len) };

        vec_scaled_add(shared_x, &y, scalar, shared_out);

        for (i, &v) in data.iter().enumerate() {
            let expected = original[i] * scalar + y[i];
            assert!(
                (v - expected).abs() <= epsilon(),
                "alias out==x bug at idx={i}: got {v}, expected {expected}"
            );
        }
    }

    /// A02 — out == y (in-place y)
    #[test]
    fn t6_2_out_equals_y() {
        let n = 100;
        let x = arange(n);
        let mut data = fill(n, 5.0 as ScalarType);
        let original = data.clone();
        let scalar = 2.0 as ScalarType;

        // SAFETY: same as t6_1, out==y aliasing. Implementation reads y[i] before writing out[i].
        let ptr = data.as_mut_ptr();
        let len = data.len();
        let shared_y = unsafe { std::slice::from_raw_parts(ptr as *const ScalarType, len) };
        let shared_out = unsafe { std::slice::from_raw_parts_mut(ptr, len) };

        vec_scaled_add(&x, shared_y, scalar, shared_out);

        for (i, &v) in data.iter().enumerate() {
            let expected = x[i] * scalar + original[i];
            assert!(
                (v - expected).abs() <= epsilon(),
                "alias out==y bug at idx={i}: got {v}, expected {expected}"
            );
        }
    }

    /// A03 — x == y (both input slices share the same memory, out is independent)
    /// out[i] = x[i]*scalar + x[i] = x[i] * (scalar + 1)
    #[test]
    fn t6_3_x_equals_y() {
        let n = 64;
        let data = arange(n);
        let mut out = vec![0.0 as ScalarType; n];
        let scalar = 3.0 as ScalarType;

        // SAFETY: x and y point to the same read-only memory, out is independent. No write conflict.
        let ptr = data.as_ptr();
        let len = data.len();
        let shared_x = unsafe { std::slice::from_raw_parts(ptr, len) };
        let shared_y = unsafe { std::slice::from_raw_parts(ptr, len) };

        vec_scaled_add(shared_x, shared_y, scalar, &mut out);

        for (i, &v) in out.iter().enumerate() {
            let expected = (i as ScalarType) * (scalar + 1.0 as ScalarType);
            assert!(
                (v - expected).abs() <= epsilon(),
                "alias x==y bug at idx={i}: got {v}, expected {expected}"
            );
        }
    }

    /// A04 — out == x == y (all three slices share the same memory)
    /// out[i] = x[i]*scalar + x[i] = x[i] * (scalar + 1)
    #[test]
    fn t6_4_all_three_equal() {
        let n = 64;
        let mut data = arange(n);
        let original = data.clone();
        let scalar = 3.0 as ScalarType;

        // SAFETY: all three slices share the same pointer. Implementation reads x[i] and y[i] before writing out[i].
        let ptr = data.as_mut_ptr();
        let len = data.len();
        let shared_x = unsafe { std::slice::from_raw_parts(ptr as *const ScalarType, len) };
        let shared_y = unsafe { std::slice::from_raw_parts(ptr as *const ScalarType, len) };
        let shared_out = unsafe { std::slice::from_raw_parts_mut(ptr, len) };

        vec_scaled_add(shared_x, shared_y, scalar, shared_out);

        for (i, &v) in data.iter().enumerate() {
            let expected = original[i] * (scalar + 1.0 as ScalarType);
            assert!(
                (v - expected).abs() <= epsilon(),
                "alias all-equal bug at idx={i}: got {v}, expected {expected}"
            );
        }
    }

    /// A05 — out is fully independent from x/y: verify x and y are not modified
    #[test]
    fn t6_5_independent_buffers_no_crosstalk() {
        let n = 16;
        let x = arange(n);
        let y = fill(n, 3.0 as ScalarType);
        let x_clone = x.clone();
        let y_clone = y.clone();
        let mut out = fill(n, -1.0 as ScalarType);
        let mut expected = vec![0.0 as ScalarType; n];
        scaled_add_oracle(&x, &y, 2.0 as ScalarType, &mut expected);
        vec_scaled_add(&x, &y, 2.0 as ScalarType, &mut out);
        for (i, (&a, &e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() <= epsilon(), "idx={i}: got {a}, expected {e}");
        }
        // x and y must remain untouched
        assert_eq!(x, x_clone, "x was modified");
        assert_eq!(y, y_clone, "y was modified");
    }
}

// T7 — Cross-path consistency (X01-X03)
mod consistency {
    use super::*;

    /// X01 — 100 consecutive calls with the same input produce identical results
    #[test]
    fn t7_1_consecutive_calls_deterministic() {
        let n = 256;
        let x = arange(n);
        let y = fill(n, 3.0 as ScalarType);
        let scalar = 1.5 as ScalarType;
        let mut first = vec![0.0 as ScalarType; n];
        vec_scaled_add(&x, &y, scalar, &mut first);
        for _ in 0..100 {
            let mut out = vec![0.0 as ScalarType; n];
            vec_scaled_add(&x, &y, scalar, &mut out);
            assert_eq!(
                out, first,
                "non-deterministic result across consecutive calls"
            );
        }
    }

    /// X02 — SIMD path vs ANSI oracle (n=1000 random)
    #[test]
    fn t7_2_matches_scalar_oracle_random() {
        let n = 1000;
        let x: Vec<ScalarType> = (0..n)
            .map(|i| ((i as f64) * 0.9876 + 1.234).sin() as ScalarType)
            .collect();
        let y: Vec<ScalarType> = (0..n)
            .map(|i| ((i as f64) * 0.4321 + 0.789).cos() as ScalarType)
            .collect();
        let scalar = -2.5 as ScalarType;
        let mut out = vec![0.0 as ScalarType; n];
        let mut expected = vec![0.0 as ScalarType; n];
        scaled_add_oracle(&x, &y, scalar, &mut expected);
        vec_scaled_add(&x, &y, scalar, &mut out);
        for (i, (&a, &e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= epsilon(),
                "idx={i}: SIMD {a} != scalar oracle {e}, diff {}",
                (a - e).abs()
            );
        }
    }

    /// X03 — Long vector (n=10000) SIMD vs ANSI oracle, precision drift control
    #[test]
    fn t7_3_long_vector_consistency() {
        let n = 10000;
        let x: Vec<ScalarType> = (0..n)
            .map(|i| ((i as f64) * 0.0001).sin() as ScalarType)
            .collect();
        let y: Vec<ScalarType> = (0..n)
            .map(|i| ((i as f64) * 0.0002).cos() as ScalarType)
            .collect();
        let scalar = 1.234 as ScalarType;
        let mut out = vec![0.0 as ScalarType; n];
        let mut expected = vec![0.0 as ScalarType; n];
        scaled_add_oracle(&x, &y, scalar, &mut expected);
        vec_scaled_add(&x, &y, scalar, &mut out);
        for (i, (&a, &e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= epsilon(),
                "long-vec idx={i}: got {a}, expected {e}, diff {}",
                (a - e).abs()
            );
        }
    }
}

// T8 — L-BFGS integration scenarios (M01-M05)
mod lbfgs_integration {
    use super::*;

    /// M01 — Parameter update: x = d * stepsize + xp
    /// This is the core call inside BacktrackingLineSearch::search.
    #[test]
    fn t8_1_param_update() {
        let n = 1024;
        let d: Vec<ScalarType> = (0..n)
            .map(|i| ((i as f64) * 0.01).sin() as ScalarType)
            .collect();
        let xp = arange(n);
        let stepsize = 0.01 as ScalarType;
        let mut x = vec![0.0 as ScalarType; n];
        vec_scaled_add(&d, &xp, stepsize, &mut x);
        for (i, (&xi, &di)) in x.iter().zip(d.iter()).enumerate() {
            let expected = di * stepsize + (i as ScalarType);
            assert!(
                (xi - expected).abs() <= epsilon(),
                "param update idx={i}: got {xi}, expected {expected}"
            );
        }
    }

    /// M02 — Momentum update: v = grad * lr + v_prev
    #[test]
    fn t8_2_momentum_update() {
        let n = 512;
        let grad: Vec<ScalarType> = (0..n)
            .map(|i| ((i as f64) * 0.05).sin() as ScalarType)
            .collect();
        let v_prev: Vec<ScalarType> = (0..n)
            .map(|i| ((i as f64) * 0.03).cos() as ScalarType)
            .collect();
        let lr = 0.001 as ScalarType;
        let mut v = vec![0.0 as ScalarType; n];
        vec_scaled_add(&grad, &v_prev, lr, &mut v);
        for (i, (&vi, &gi)) in v.iter().zip(grad.iter()).enumerate() {
            let expected = gi * lr + v_prev[i];
            assert!(
                (vi - expected).abs() <= epsilon(),
                "momentum idx={i}: got {vi}, expected {expected}"
            );
        }
    }

    /// M03 — Nesterov projection: x_tmp = d * momentum + x
    #[test]
    fn t8_3_nesterov_projection() {
        let n = 256;
        let d = arange(n);
        let x: Vec<ScalarType> = (0..n)
            .map(|i| ((i as f64) * 0.1).sin() as ScalarType)
            .collect();
        let momentum = 0.9 as ScalarType;
        let mut x_tmp = vec![0.0 as ScalarType; n];
        vec_scaled_add(&d, &x, momentum, &mut x_tmp);
        for (i, (&xi, &di)) in x_tmp.iter().zip(d.iter()).enumerate() {
            let expected = di * momentum + x[i];
            assert!(
                (xi - expected).abs() <= epsilon(),
                "nesterov idx={i}: got {xi}, expected {expected}"
            );
        }
    }

    /// M04 — Residual accumulation: r = A*x + b (linear algebra context)
    #[test]
    fn t8_4_residual_accumulation() {
        let n = 128;
        let x_col = arange(n);
        let b = fill(n, 0.5 as ScalarType);
        let a_val = 2.5 as ScalarType;
        let mut r = vec![0.0 as ScalarType; n];
        vec_scaled_add(&x_col, &b, a_val, &mut r);
        for (i, (&ri, &xi)) in r.iter().zip(x_col.iter()).enumerate() {
            let expected = xi * a_val + 0.5 as ScalarType;
            assert!(
                (ri - expected).abs() <= epsilon(),
                "residual idx={i}: got {ri}, expected {expected}"
            );
        }
    }

    /// M05 — Common ML dimensions: 4, 8, 16, 32, 64, 128, 256, 784, 1024, 4096
    #[test]
    fn t8_5_common_ml_dimensions() {
        for &n in &[4usize, 8, 16, 32, 64, 128, 256, 784, 1024, 4096] {
            let x = arange(n);
            let y = fill(n, 1.0 as ScalarType);
            let scalar = 0.5 as ScalarType;
            let mut out = vec![0.0 as ScalarType; n];
            let mut expected = vec![0.0 as ScalarType; n];
            scaled_add_oracle(&x, &y, scalar, &mut expected);
            vec_scaled_add(&x, &y, scalar, &mut out);
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
