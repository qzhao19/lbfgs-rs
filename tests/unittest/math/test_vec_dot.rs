use lbfgs_rs::infra::math::kernel::vec_dot;
use lbfgs_rs::shared::types::primitives::ScalarType;

// ── Helper utilities ──

/// Naive scalar dot product — serves as the ground truth oracle.
fn dot_oracle(x: &[ScalarType], y: &[ScalarType]) -> ScalarType {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

/// Generate a vector [0, 1, 2, ..., len-1].
fn arange(len: usize) -> Vec<ScalarType> {
    (0..len).map(|i| i as ScalarType).collect()
}

/// Generate a vector of `len` copies of `value`.
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

/// Compare two floats with relative tolerance (suitable for f32 large accumulations).
fn approx_eq(got: ScalarType, expected: ScalarType) -> bool {
    let abs_diff = (got - expected).abs();
    let scale = expected.abs().max(1.0 as ScalarType);
    let rel_tol = if cfg!(feature = "f32") {
        5e-6 as ScalarType // ~35 ULP — generous for 10000 FMA accumulations
    } else {
        1e-12 as ScalarType
    };
    abs_diff < scale * rel_tol
}

// T0 — Input validation
// TODO: Enable these tests after adding `assert_eq!` to `vec_dot`.
// Currently the function uses only `debug_assert_eq!` (ANSI path) and
// no check (NEON path). Unequal lengths may cause UB in release mode.
//
// #[test]
// #[should_panic(expected = "length mismatch")]
// fn t0_1_x_longer_than_y() {
//     let x = vec![1.0 as ScalarType; 5];
//     let y = vec![1.0 as ScalarType; 3];
//     vec_dot(&x, &y);
// }
//
// #[test]
// #[should_panic(expected = "length mismatch")]
// fn t0_2_y_longer_than_x() {
//     let x = vec![1.0 as ScalarType; 3];
//     let y = vec![1.0 as ScalarType; 5];
//     vec_dot(&x, &y);
// }

/// T0.3 — Empty vectors should return zero.
#[test]
fn t0_3_both_empty() {
    assert_eq!(vec_dot(&[], &[]), 0.0 as ScalarType);
}

// T1 — Basic correctness
mod basic_correctness {
    use super::*;

    #[test]
    fn t1_1_zero_vector() {
        let x = fill(4, 0.0 as ScalarType);
        let y = fill(4, 0.0 as ScalarType);
        assert_eq!(vec_dot(&x, &y), 0.0 as ScalarType);
    }

    #[test]
    fn t1_2_unit_vector() {
        let x = vec![
            1.0 as ScalarType,
            0.0 as ScalarType,
            0.0 as ScalarType,
            0.0 as ScalarType,
        ];
        let y = vec![
            1.0 as ScalarType,
            0.0 as ScalarType,
            0.0 as ScalarType,
            0.0 as ScalarType,
        ];
        assert_eq!(vec_dot(&x, &y), 1.0 as ScalarType);
    }

    #[test]
    fn t1_3_orthogonal() {
        let x = vec![1.0 as ScalarType, 0.0 as ScalarType];
        let y = vec![0.0 as ScalarType, 1.0 as ScalarType];
        assert_eq!(vec_dot(&x, &y), 0.0 as ScalarType);
    }

    #[test]
    fn t1_4_known_result() {
        let x = vec![1.0 as ScalarType, 2.0 as ScalarType, 3.0 as ScalarType];
        let y = vec![4.0 as ScalarType, 5.0 as ScalarType, 6.0 as ScalarType];
        assert_eq!(vec_dot(&x, &y), 32.0 as ScalarType);
    }

    #[test]
    fn t1_5_negatives() {
        let x = vec![-1.0 as ScalarType, 2.0 as ScalarType];
        let y = vec![3.0 as ScalarType, -4.0 as ScalarType];
        assert_eq!(vec_dot(&x, &y), -11.0 as ScalarType);
    }

    #[test]
    fn t1_6_fractional() {
        let x = vec![0.5 as ScalarType, 0.25 as ScalarType];
        let y = vec![2.0 as ScalarType, 4.0 as ScalarType];
        assert_eq!(vec_dot(&x, &y), 2.0 as ScalarType);
    }
}

// T2 — Boundary & alignment (loop-unrolling critical paths)
mod boundary_alignment {
    use super::*;

    /// Assert vec_dot result matches oracle within epsilon.
    fn assert_dot(len: usize) {
        let x = arange(len);
        let y: Vec<ScalarType> = x
            .iter()
            .map(|&v| v * 2.0 as ScalarType + 1.0 as ScalarType)
            .collect();
        let expected = dot_oracle(&x, &y);
        let got = vec_dot(&x, &y);
        assert!(
            approx_eq(got, expected),
            "len={len}: got {got}, expected {expected}, diff {}",
            (got - expected).abs()
        );
    }

    #[test]
    fn t2_1_len_0() {
        assert_dot(0);
    }
    #[test]
    fn t2_2_len_1() {
        assert_dot(1);
    } // pure scalar
    #[test]
    fn t2_3_len_3() {
        assert_dot(3);
    } // < 1 SIMD lane (f32: <4)

    /// T2.4 — Exactly 1 SIMD lane (f32: 4, f64: 2)
    #[test]
    fn t2_4_one_lane() {
        let n = if cfg!(feature = "f32") { 4 } else { 2 };
        assert_dot(n);
    }

    /// T2.12 — f32 only: 1 lane + 1 scalar remainder (len=5)
    #[test]
    #[cfg(feature = "f32")]
    fn t2_12_f32_one_lane_plus_one_scalar() {
        assert_dot(5);
    }

    /// T2.13 — f32 only: 2 lanes, 0 blocks, 0 scalar → pure while-loop path (len=8)
    #[test]
    #[cfg(feature = "f32")]
    fn t2_13_f32_two_lanes_no_remainder() {
        assert_dot(8);
    }

    #[test]
    fn t2_5_len_15() {
        assert_dot(15);
    } // 3 lanes + 3 scalar (f32); 1 lane + 1 scalar (f64)
    #[test]
    fn t2_6_len_16() {
        assert_dot(16);
    } // 1 unrolled block (f32); 4 blocks (f64)

    /// T2.7 — 1 block + 1 scalar (f32: 17, f64: 5)
    #[test]
    fn t2_7_one_block_plus_one_scalar() {
        let n = if cfg!(feature = "f32") { 17 } else { 5 };
        assert_dot(n);
    }

    /// T2.14 — f32 only: 1 block + 1 lane, 0 scalar (len=20)
    #[test]
    #[cfg(feature = "f32")]
    fn t2_14_f32_one_block_one_lane() {
        assert_dot(20);
    }

    /// T2.15 — f64 only: 1 block + 1 lane, 0 scalar (len=6)
    #[test]
    #[cfg(feature = "f64")]
    fn t2_15_f64_one_block_one_lane() {
        assert_dot(6);
    }

    /// T2.8 — 1 block + incomplete lanes + scalar (f32: 31, f64: 7)
    #[test]
    fn t2_8_mixed_remainder() {
        let n = if cfg!(feature = "f32") { 31 } else { 7 };
        assert_dot(n);
    }

    #[test]
    fn t2_9_len_32() {
        assert_dot(32);
    } // 2 blocks (f32); 8 blocks (f64)
    #[test]
    fn t2_10_len_100() {
        assert_dot(100);
    } // multi-block + remainder
    #[test]
    fn t2_11_len_1024() {
        assert_dot(1024);
    } // large scale
}

// T3 — Numerical stability (L-BFGS scenarios)
mod numeric_stability {
    use super::*;

    #[test]
    fn t3_1_large_values_no_overflow() {
        let big = 1e10 as ScalarType;
        let x = vec![big, big];
        let y = vec![big, big];
        let result = vec_dot(&x, &y);
        assert!(result.is_finite(), "overflow detected: {result}");
    }

    #[test]
    fn t3_2_small_values_no_underflow() {
        let tiny = 1e-10 as ScalarType;
        let x = vec![tiny, tiny];
        let y = vec![tiny, tiny];
        let result = vec_dot(&x, &y);
        assert!(result > 0.0 as ScalarType, "underflow to zero: {result}");
    }

    #[test]
    fn t3_3_mixed_scale() {
        let big = 1e10 as ScalarType;
        let tiny = 1e-10 as ScalarType;
        let x = vec![big, tiny];
        let y = vec![tiny, big];
        let result = vec_dot(&x, &y);
        let expected = 2.0 as ScalarType;
        assert!(
            (result - expected).abs() < epsilon(),
            "mixed scale: got {result}, expected {expected}"
        );
    }

    #[test]
    fn t3_4_accumulation_cancellation() {
        let x = vec![
            1.0 as ScalarType,
            -1.0 as ScalarType,
            1.0 as ScalarType,
            -1.0 as ScalarType,
        ];
        let y = vec![
            1.0 as ScalarType,
            1.0 as ScalarType,
            1.0 as ScalarType,
            1.0 as ScalarType,
        ];
        assert_eq!(vec_dot(&x, &y), 0.0 as ScalarType);
    }

    #[test]
    fn t3_5_long_sequence() {
        let len = 10_000;
        let x = arange(len);
        let y: Vec<ScalarType> = x.iter().map(|&v| v * 0.5 as ScalarType).collect();
        let got = vec_dot(&x, &y);
        let expected = dot_oracle(&x, &y);
        assert!(
            approx_eq(got, expected),
            "n=10000: got {got}, expected {expected}, diff {}",
            (got - expected).abs()
        );
    }
}

// T4 — SIMD vs scalar consistency (critical regression suite)
mod simd_vs_scalar {
    use super::*;

    /// Assert vec_dot matches oracle for random vectors of given length.
    fn assert_consistent(len: usize) {
        // Use a deterministic pseudo-random sequence (not true random)
        let x: Vec<ScalarType> = (0..len)
            .map(|i| {
                let v = (i as f64 * 1.234 + 0.567).sin();
                v as ScalarType
            })
            .collect();
        let y: Vec<ScalarType> = (0..len)
            .map(|i| {
                let v = (i as f64 * 3.456 + 1.789).cos();
                v as ScalarType
            })
            .collect();

        let got = vec_dot(&x, &y);
        let expected = dot_oracle(&x, &y);
        assert!(
            (got - expected).abs() < epsilon(),
            "len={len}: NEON result {got} != scalar {expected}, diff {}",
            (got - expected).abs()
        );
    }

    /// T4.1 — 1 unrolled block (f32: 16, f64: 4)
    #[test]
    fn t4_1_one_block() {
        let n = if cfg!(feature = "f32") { 16 } else { 4 };
        assert_consistent(n);
    }

    /// T4.2 — 1 block + 1 scalar remainder (f32: 17, f64: 5)
    #[test]
    fn t4_2_one_block_plus_scalar() {
        let n = if cfg!(feature = "f32") { 17 } else { 5 };
        assert_consistent(n);
    }

    #[test]
    fn t4_3_random_100() {
        assert_consistent(100);
    }

    #[test]
    fn t4_4_extreme_mixed_values() {
        let x = vec![
            ScalarType::MAX,
            ScalarType::MIN,
            0.0 as ScalarType,
            1.0 as ScalarType,
            -1.0 as ScalarType,
        ];
        let y = vec![
            1.0 as ScalarType,
            1.0 as ScalarType,
            1.0 as ScalarType,
            0.0 as ScalarType,
            0.0 as ScalarType,
        ];
        let got = vec_dot(&x, &y);
        let expected = dot_oracle(&x, &y);
        // For extreme values, use relative or absolute tolerance
        let diff = (got - expected).abs();
        let rel = if expected.abs() > 1.0 as ScalarType {
            diff / expected.abs()
        } else {
            diff
        };
        assert!(
            rel < epsilon() * 100.0,
            "extreme: got {got}, expected {expected}, diff {diff}"
        );
    }
}

// T5 — Special floating-point values (IEEE 754)
mod special_values {
    use super::*;

    #[test]
    fn t5_1_zero_times_inf_is_nan() {
        let x = vec![0.0 as ScalarType, 0.0 as ScalarType];
        let y = vec![ScalarType::INFINITY, 1.0 as ScalarType];
        let result = vec_dot(&x, &y);
        assert!(result.is_nan(), "0*INF should be NaN, got {result}");
    }

    #[test]
    fn t5_2_inf_times_zero_is_nan() {
        let x = vec![ScalarType::INFINITY, 0.0 as ScalarType];
        let y = vec![0.0 as ScalarType, 1.0 as ScalarType];
        let result = vec_dot(&x, &y);
        assert!(result.is_nan(), "INF*0 should be NaN, got {result}");
    }

    #[test]
    fn t5_3_negative_zero() {
        let x = vec![-0.0_f64 as ScalarType, 1.0 as ScalarType];
        let y = vec![1.0 as ScalarType, 1.0 as ScalarType];
        // -0.0 * 1.0 + 1.0 * 1.0 = -0.0 + 1.0 = 1.0
        assert_eq!(vec_dot(&x, &y), 1.0 as ScalarType);
    }

    #[test]
    fn t5_4_all_nan() {
        let v = vec![ScalarType::NAN; 4];
        let result = vec_dot(&v, &v);
        assert!(result.is_nan(), "NaN dot NaN should be NaN, got {result}");
    }
}

// T7 — L-BFGS integration scenarios
mod lbfgs_integration {
    use super::*;

    /// T7.1 — Weight vector · feature row (simulates LogLoss::evaluate_with_gradient).
    /// w: model weights, x: one row of features → y_hat = w · x
    #[test]
    fn t7_1_weight_dot_feature_row() {
        let n = 1000;
        let w: Vec<ScalarType> = (0..n)
            .map(|i| (i as f64 * 0.001 - 0.5).sin() as ScalarType)
            .collect();
        let x: Vec<ScalarType> = (0..n)
            .map(|i| (i as f64 * 0.003 + 0.1).cos() as ScalarType)
            .collect();

        let y_hat = vec_dot(&w, &x);
        let expected = dot_oracle(&w, &x);
        assert!(
            approx_eq(y_hat, expected),
            "weight·feature mismatch: got {y_hat}, expected {expected}, diff {}",
            (y_hat - expected).abs()
        );
    }

    /// T7.2 — Gradient · direction (simulates L-BFGS two-loop recursion).
    /// g: gradient, d: search direction → curvature = g · d
    #[test]
    fn t7_2_gradient_dot_direction() {
        let n = 1000;
        let g: Vec<ScalarType> = (0..n)
            .map(|i| (i as f64 * 0.002).sin() as ScalarType)
            .collect();
        let d: Vec<ScalarType> = (0..n)
            .map(|i| (i as f64 * 0.002 + 1.5).cos() as ScalarType)
            .collect();

        let curvature = vec_dot(&g, &d);
        let expected = dot_oracle(&g, &d);
        assert!(
            approx_eq(curvature, expected),
            "weight·feature mismatch: got {curvature}, expected {expected}, diff {}",
            (curvature - expected).abs()
        );
    }

    /// T7.3 — 100 consecutive calls (detects register state leakage).
    #[test]
    fn t7_3_consecutive_calls_no_state_leak() {
        let n = 256;
        let x = arange(n);
        let y: Vec<ScalarType> = x.iter().rev().copied().collect();

        let first = vec_dot(&x, &y);
        for _ in 0..100 {
            let result = vec_dot(&x, &y);
            assert!(
                (result - first).abs() < epsilon(),
                "state leak detected: first={first}, got={result}"
            );
        }
    }
}
