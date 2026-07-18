use lbfgs_rs::infra::math::kernel::vec_scaled_add_inplace;
use lbfgs_rs::shared::types::primitives::ScalarType;

// ── Helper utilities ──

/// Naive scalar reference implementation: acc[i] += src[i] × scalar.
fn vecadd_oracle(src: &[ScalarType], scalar: ScalarType, acc: &mut [ScalarType]) {
    for i in 0..src.len() {
        acc[i] += src[i] * scalar;
    }
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

// T0 — Input validation
//
// TODO: Enable T0.1/T0.2 after adding `assert_eq!` to `vec_scaled_add_inplace`.
// Currently only `debug_assert_eq!` in ansi path, no check in NEON path.

#[test]
fn t0_3_empty_vectors() {
    let mut acc: Vec<ScalarType> = vec![];
    vec_scaled_add_inplace(&[], 2.0 as ScalarType, &mut acc);
    assert_eq!(acc.len(), 0);
}

#[test]
fn t0_4_scalar_zero_acc_unchanged() {
    let src = vec![1.0 as ScalarType, 2.0 as ScalarType, 3.0 as ScalarType];
    let mut acc = vec![10.0 as ScalarType, 20.0 as ScalarType, 30.0 as ScalarType];
    let expected = acc.clone();
    vec_scaled_add_inplace(&src, 0.0 as ScalarType, &mut acc);
    assert_eq!(acc, expected);
}

// T1 — Basic correctness
mod basic_correctness {
    use super::*;

    #[test]
    fn t1_1_basic_accumulate_scalar_one() {
        let src = vec![4.0 as ScalarType, 5.0 as ScalarType, 6.0 as ScalarType];
        let mut acc = vec![1.0 as ScalarType, 2.0 as ScalarType, 3.0 as ScalarType];
        vec_scaled_add_inplace(&src, 1.0 as ScalarType, &mut acc);
        assert_eq!(
            acc,
            vec![5.0 as ScalarType, 7.0 as ScalarType, 9.0 as ScalarType]
        );
    }

    #[test]
    fn t1_2_scalar_two() {
        let src = vec![4.0 as ScalarType, 5.0 as ScalarType, 6.0 as ScalarType];
        let mut acc = vec![1.0 as ScalarType, 2.0 as ScalarType, 3.0 as ScalarType];
        vec_scaled_add_inplace(&src, 2.0 as ScalarType, &mut acc);
        assert_eq!(
            acc,
            vec![9.0 as ScalarType, 12.0 as ScalarType, 15.0 as ScalarType]
        );
    }

    #[test]
    fn t1_3_src_all_zeros() {
        let src = fill(4, 0.0 as ScalarType);
        let mut acc = vec![
            1.0 as ScalarType,
            2.0 as ScalarType,
            3.0 as ScalarType,
            4.0 as ScalarType,
        ];
        let expected = acc.clone();
        vec_scaled_add_inplace(&src, 5.0 as ScalarType, &mut acc);
        assert_eq!(acc, expected);
    }

    #[test]
    fn t1_4_acc_all_zeros() {
        let src = vec![4.0 as ScalarType, 5.0 as ScalarType, 6.0 as ScalarType];
        let mut acc = vec![0.0 as ScalarType; 3];
        vec_scaled_add_inplace(&src, 1.0 as ScalarType, &mut acc);
        assert_eq!(
            acc,
            vec![4.0 as ScalarType, 5.0 as ScalarType, 6.0 as ScalarType]
        );
    }

    #[test]
    fn t1_5_negative_scalar() {
        let src = vec![1.0 as ScalarType, 2.0 as ScalarType, 3.0 as ScalarType];
        let mut acc = vec![10.0 as ScalarType, 20.0 as ScalarType, 30.0 as ScalarType];
        vec_scaled_add_inplace(&src, -1.0 as ScalarType, &mut acc);
        assert_eq!(
            acc,
            vec![9.0 as ScalarType, 18.0 as ScalarType, 27.0 as ScalarType]
        );
    }

    #[test]
    fn t1_6_fractional_scalar() {
        let src = vec![2.0 as ScalarType, 4.0 as ScalarType, 8.0 as ScalarType];
        let mut acc = vec![0.0 as ScalarType; 3];
        vec_scaled_add_inplace(&src, 0.5 as ScalarType, &mut acc);
        assert_eq!(
            acc,
            vec![1.0 as ScalarType, 2.0 as ScalarType, 4.0 as ScalarType]
        );
    }

    #[test]
    fn t1_7_scalar_one_large_vector() {
        let n = 1000;
        let src = arange(n);
        let mut acc = arange(n); // acc = [0,1,2,...], src = [0,1,2,...], scalar=1.0 → acc = [0,2,4,...]
        let mut expected = acc.clone();
        vecadd_oracle(&src, 1.0 as ScalarType, &mut expected);
        vec_scaled_add_inplace(&src, 1.0 as ScalarType, &mut acc);
        assert_eq!(acc, expected);
    }
}

// T2 — Boundary & alignment (loop-unrolling critical paths)
mod boundary_alignment {
    use super::*;

    /// Assert vec_scaled_add_inplace result matches oracle for given length and scalar.
    fn assert_vecadd(len: usize, scalar: ScalarType) {
        let src = arange(len);
        let init: Vec<ScalarType> = (0..len).map(|i| (i as ScalarType) * 0.5).collect();

        let mut acc = init.clone();
        let mut expected = init.clone();

        vec_scaled_add_inplace(&src, scalar, &mut acc);
        vecadd_oracle(&src, scalar, &mut expected);

        for (i, (a, e)) in acc.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < epsilon(),
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
        let mut acc = vec![];
        vec_scaled_add_inplace(&[], s(), &mut acc);
        assert_eq!(acc.len(), 0);
    }
    #[test]
    fn t2_2_len_1() {
        assert_vecadd(1, s());
    }
    #[test]
    fn t2_3_len_3() {
        assert_vecadd(3, s());
    }

    #[test]
    fn t2_4_one_lane() {
        let n = if cfg!(feature = "f32") { 4 } else { 2 };
        assert_vecadd(n, s());
    }

    #[test]
    #[cfg(feature = "f32")]
    fn t2_5_f32_one_lane_plus_one_scalar() {
        assert_vecadd(5, s());
    }

    #[test]
    #[cfg(feature = "f32")]
    fn t2_6_f32_two_lanes() {
        assert_vecadd(8, s());
    }

    #[test]
    fn t2_7_len_15() {
        assert_vecadd(15, s());
    }

    #[test]
    fn t2_8_one_block() {
        let n = if cfg!(feature = "f32") { 16 } else { 4 };
        assert_vecadd(n, s());
    }

    #[test]
    fn t2_9_one_block_plus_one_scalar() {
        let n = if cfg!(feature = "f32") { 17 } else { 5 };
        assert_vecadd(n, s());
    }

    #[test]
    #[cfg(feature = "f32")]
    fn t2_10_f32_one_block_one_lane() {
        assert_vecadd(20, s());
    }

    #[test]
    #[cfg(feature = "f64")]
    fn t2_11_f64_one_block_one_lane() {
        assert_vecadd(6, s());
    }

    #[test]
    fn t2_12_mixed_remainder() {
        let n = if cfg!(feature = "f32") { 31 } else { 7 };
        assert_vecadd(n, s());
    }

    #[test]
    fn t2_13_len_32() {
        assert_vecadd(32, s());
    }
    #[test]
    fn t2_14_len_100() {
        assert_vecadd(100, s());
    }
    #[test]
    fn t2_15_len_1024() {
        assert_vecadd(1024, s());
    }
}

// T3 — In-place mutation correctness (vec_scaled_add_inplace-specific)
mod inplace_mutation {
    use super::*;

    #[test]
    fn t3_1_consecutive_calls_accumulate() {
        let src_a = vec![1.0 as ScalarType, 0.0 as ScalarType];
        let src_b = vec![0.0 as ScalarType, 2.0 as ScalarType];
        let src_c = vec![3.0 as ScalarType, 3.0 as ScalarType];
        let mut acc = vec![10.0 as ScalarType, 10.0 as ScalarType];

        vec_scaled_add_inplace(&src_a, 2.0 as ScalarType, &mut acc);
        vec_scaled_add_inplace(&src_b, 3.0 as ScalarType, &mut acc);
        vec_scaled_add_inplace(&src_c, 1.0 as ScalarType, &mut acc);

        // 10 + 1·2 = 12; 12 + 2·3 = 18; 18 + 3·1 = 21
        // 10 + 0·2 = 10; 10 + 2·3 = 16; 16 + 3·1 = 19
        assert_eq!(acc, vec![15.0 as ScalarType, 19.0 as ScalarType]);
    }

    #[test]
    fn t3_2_acc_equals_src() {
        // acc[i] += acc[i] * s  <=>  acc[i] *= (1 + s)
        // If acc and src are the same slice, vec_scaled_add_inplace must read
        // the original values before any writes happen.
        let scalar = 3.0 as ScalarType;
        let mut data = vec![
            1.0 as ScalarType,
            2.0 as ScalarType,
            3.0 as ScalarType,
            4.0 as ScalarType,
        ];

        // Compute expected: each element *= (1 + scalar)
        let expected: Vec<ScalarType> = data
            .iter()
            .map(|&v| v * (1.0 as ScalarType + scalar))
            .collect();

        // Use raw pointer to pass same memory as src and acc (bypass borrow checker)
        let ptr = data.as_mut_ptr();
        let len = data.len();
        // SAFETY: src and acc point to the same slice, but `vec_scaled_add_inplace` is
        // specified to read `src` before writing to `acc` for any given index.
        // We rely on `vec_scaled_add_inplace` respecting element-wise read-before-write order.
        let shared_src = unsafe { std::slice::from_raw_parts(ptr as *const ScalarType, len) };
        let shared_acc = unsafe { std::slice::from_raw_parts_mut(ptr, len) };

        vec_scaled_add_inplace(shared_src, scalar, shared_acc);

        assert_eq!(data, expected, "alias bug: acc != src * (1+scalar)");
    }

    #[test]
    fn t3_4_no_register_state_leak() {
        let n = 128;
        let src = arange(n);
        let acc = fill(n, 1.0 as ScalarType);

        let mut first_result = acc.clone();
        vec_scaled_add_inplace(&src, 0.5 as ScalarType, &mut first_result);

        // Reset and call again — result must be identical
        for _ in 0..50 {
            let mut acc_copy = fill(n, 1.0 as ScalarType);
            vec_scaled_add_inplace(&src, 0.5 as ScalarType, &mut acc_copy);
            for (i, (&a, &f)) in acc_copy.iter().zip(first_result.iter()).enumerate() {
                assert!(
                    (a - f).abs() < epsilon(),
                    "state leak at iteration: idx={i}, got {a}, expected {f}"
                );
            }
        }
    }
}

// T4 — SIMD vs scalar consistency
mod simd_vs_scalar {
    use super::*;

    fn assert_consistent(len: usize, scalar: ScalarType) {
        let src: Vec<ScalarType> = (0..len)
            .map(|i| ((i as f64) * 1.234 + 0.567).sin() as ScalarType)
            .collect();
        let init: Vec<ScalarType> = (0..len)
            .map(|i| ((i as f64) * 3.456 + 1.789).cos() as ScalarType)
            .collect();

        let mut acc_neon = init.clone();
        let mut acc_ansi = init.clone();

        vec_scaled_add_inplace(&src, scalar, &mut acc_neon);
        vecadd_oracle(&src, scalar, &mut acc_ansi);

        for (i, (&a, &e)) in acc_neon.iter().zip(acc_ansi.iter()).enumerate() {
            assert!(
                (a - e).abs() < epsilon(),
                "len={len}, idx={i}: neon={a}, ansi={e}, diff={}",
                (a - e).abs()
            );
        }
    }

    #[test]
    fn t4_1_one_block() {
        let n = if cfg!(feature = "f32") { 16 } else { 4 };
        assert_consistent(n, 2.5 as ScalarType);
    }

    #[test]
    fn t4_2_one_block_plus_scalar() {
        let n = if cfg!(feature = "f32") { 17 } else { 5 };
        assert_consistent(n, -1.5 as ScalarType);
    }

    #[test]
    fn t4_3_random_100() {
        assert_consistent(100, 0.75 as ScalarType);
    }

    #[test]
    fn t4_4_extreme_mixed_values() {
        let src = vec![
            ScalarType::MAX,
            ScalarType::MIN,
            0.0 as ScalarType,
            1.0 as ScalarType,
        ];
        let mut acc = vec![
            1.0 as ScalarType,
            0.0 as ScalarType,
            ScalarType::MAX,
            ScalarType::MIN,
        ];
        let mut expected = acc.clone();

        vec_scaled_add_inplace(&src, 0.5 as ScalarType, &mut acc);
        vecadd_oracle(&src, 0.5 as ScalarType, &mut expected);

        for (i, (&a, &e)) in acc.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            // Relax tolerance for extreme values
            let rel = if e.abs() > 1.0 as ScalarType {
                diff / e.abs()
            } else {
                diff
            };
            assert!(
                rel < epsilon() * 100.0,
                "idx={i}: neon={a} vs ansi={e}, diff={diff}"
            );
        }
    }
}

// T5 — Special floating-point values
mod special_values {
    use super::*;

    #[test]
    fn t5_1_src_inf_scalar_zero() {
        let src = vec![ScalarType::INFINITY, 1.0 as ScalarType];
        let mut acc = vec![5.0 as ScalarType, 5.0 as ScalarType];
        vec_scaled_add_inplace(&src, 0.0 as ScalarType, &mut acc);
        // 5.0 + INF·0 = 5.0 + NaN = NaN
        assert!(acc[0].is_nan());
        // 5.0 + 1.0·0 = 5.0  (other elements not affected)
        assert_eq!(acc[1], 5.0 as ScalarType);
    }

    #[test]
    fn t5_2_acc_inf_src_zero() {
        let src = vec![0.0 as ScalarType, 1.0 as ScalarType];
        let mut acc = vec![ScalarType::INFINITY, 5.0 as ScalarType];
        vec_scaled_add_inplace(&src, 0.0 as ScalarType, &mut acc);
        // INF + 0·0 = INF (well-defined)
        assert!(acc[0].is_infinite());
        assert!(acc[0].is_sign_positive());
        // 5.0 + 1.0·0 = 5.0
        assert_eq!(acc[1], 5.0 as ScalarType);
    }

    #[test]
    fn t5_3_scalar_nan_broadcast() {
        let src = vec![1.0 as ScalarType; 4];
        let mut acc = vec![0.0 as ScalarType; 4];
        vec_scaled_add_inplace(&src, ScalarType::NAN, &mut acc);
        for &v in &acc {
            assert!(v.is_nan(), "expected NaN, got {v}");
        }
    }

    #[test]
    fn t5_4_negative_zero_src() {
        let src = vec![-0.0_f64 as ScalarType, 1.0 as ScalarType];
        let mut acc = vec![42.0 as ScalarType, 42.0 as ScalarType];
        vec_scaled_add_inplace(&src, 3.0 as ScalarType, &mut acc);
        // -0.0 * 3.0 = -0.0 (sign preserved)
        // 42.0 + (-0.0) = 42.0
        assert_eq!(acc[0], 42.0 as ScalarType);
        assert_eq!(acc[1], 45.0 as ScalarType);
    }
}

// T6 — L-BFGS gradient accumulation scenarios
mod lbfgs_integration {
    use super::*;

    #[test]
    fn t6_1_batch_gradient_accumulation() {
        // Simulate evaluate_with_gradient: for each sample i,
        // grad[j] += feature_row[j] * dloss_i
        let n_samples = 5;
        let n_features = 4;

        // Feature matrix (row-major): 5 samples × 4 features
        let features: Vec<Vec<ScalarType>> = vec![
            vec![
                1.0 as ScalarType,
                0.0 as ScalarType,
                2.0 as ScalarType,
                0.0 as ScalarType,
            ],
            vec![
                0.0 as ScalarType,
                1.0 as ScalarType,
                0.0 as ScalarType,
                3.0 as ScalarType,
            ],
            vec![
                2.0 as ScalarType,
                2.0 as ScalarType,
                0.0 as ScalarType,
                0.0 as ScalarType,
            ],
            vec![
                1.0 as ScalarType,
                0.0 as ScalarType,
                1.0 as ScalarType,
                1.0 as ScalarType,
            ],
            vec![
                0.0 as ScalarType,
                3.0 as ScalarType,
                0.0 as ScalarType,
                2.0 as ScalarType,
            ],
        ];
        let dlosses: Vec<ScalarType> = vec![
            0.1 as ScalarType,
            -0.2 as ScalarType,
            0.15 as ScalarType,
            -0.05 as ScalarType,
            0.3 as ScalarType,
        ];

        // Accumulate via vec_scaled_add_inplace
        let mut grad = vec![0.0 as ScalarType; n_features];
        for i in 0..n_samples {
            vec_scaled_add_inplace(&features[i], dlosses[i], &mut grad);
        }

        // Manually compute expected gradient
        let mut expected = vec![0.0 as ScalarType; n_features];
        vecadd_oracle(&features[0], dlosses[0], &mut expected);
        vecadd_oracle(&features[1], dlosses[1], &mut expected);
        vecadd_oracle(&features[2], dlosses[2], &mut expected);
        vecadd_oracle(&features[3], dlosses[3], &mut expected);
        vecadd_oracle(&features[4], dlosses[4], &mut expected);

        for (i, (&a, &e)) in grad.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < epsilon(),
                "batch gradient idx={i}: got {a}, expected {e}"
            );
        }
    }

    #[test]
    fn t6_3_regularization_then_loss_gradient() {
        // Simulate grad already has L2 regularization term, then add loss gradient
        let n = 100;
        let src = arange(n);

        let mut grad = vec![0.0 as ScalarType; n];
        // Add L2 regularization: grad[j] += w[j] * lambda
        for j in 0..n {
            grad[j] = (j as ScalarType) * 0.01 as ScalarType;
        }

        // Save regularization contribution
        let reg_only = grad.clone();

        // Add loss gradient via vec_scaled_add_inplace
        let dloss = 0.5 as ScalarType;
        vec_scaled_add_inplace(&src, dloss, &mut grad);

        // Verify: grad = reg_only + src * dloss
        for j in 0..n {
            let expected = reg_only[j] + src[j] * dloss;
            assert!(
                (grad[j] - expected).abs() < epsilon(),
                "j={j}: got {}, expected {expected}",
                grad[j]
            );
        }
    }

    #[test]
    fn t6_4_large_scale_accumulation() {
        // 100 features × 1000 samples — no panic, results correct
        let n_features = 100;
        let n_samples = 1000;

        let samples: Vec<Vec<ScalarType>> = (0..n_samples)
            .map(|s| {
                (0..n_features)
                    .map(|f| ((s * n_features + f) as f64 % 7.0 - 3.0) as ScalarType)
                    .collect()
            })
            .collect();
        let dlosses: Vec<ScalarType> = (0..n_samples)
            .map(|i| ((i as f64) * 0.01 - 5.0).sin() as ScalarType)
            .collect();

        // vec_scaled_add_inplace path
        let mut grad_vecadd = vec![0.0 as ScalarType; n_features];
        for i in 0..n_samples {
            vec_scaled_add_inplace(&samples[i], dlosses[i], &mut grad_vecadd);
        }

        // Oracle path
        let mut grad_oracle = vec![0.0 as ScalarType; n_features];
        for i in 0..n_samples {
            vecadd_oracle(&samples[i], dlosses[i], &mut grad_oracle);
        }

        for (i, (&a, &e)) in grad_vecadd.iter().zip(grad_oracle.iter()).enumerate() {
            let diff = (a - e).abs();
            let rel = if e.abs() > 1.0 as ScalarType {
                diff / e.abs()
            } else {
                diff
            };
            assert!(
                rel < epsilon() * 10.0,
                "large scale idx={i}: vec_scaled_add_inplace={a}, oracle={e}, diff={diff}"
            );
        }
    }
}

// T7 — Numerical stability (long accumulation)
mod numeric_stability {
    use super::*;

    #[test]
    fn t7_1_cumulative_error_micro_increments() {
        // Accumulate 10000 tiny increments — check error vs oracle
        let n = 10000;
        let src = fill(n, 2.0 as ScalarType);
        let tiny = 0.0001 as ScalarType;

        let mut acc = vec![0.0 as ScalarType; 1];
        let mut expected = vec![0.0 as ScalarType; 1];

        for _ in 0..n {
            vec_scaled_add_inplace(&src[..1], tiny, &mut acc);
            vecadd_oracle(&src[..1], tiny, &mut expected);
        }

        let diff = (acc[0] - expected[0]).abs();
        let rel = diff / expected[0].abs();
        // Allow 1e-10 relative error for f64, 1e-4 for f32
        let tolerance = if cfg!(feature = "f32") {
            1e-4 as ScalarType
        } else {
            1e-10 as ScalarType
        };
        assert!(
            rel < tolerance,
            "cumulative error: neon={}, oracle={}, rel_err={rel}",
            acc[0],
            expected[0]
        );
    }

    #[test]
    fn t7_2_neon_vs_ansi_long_sequence() {
        let len = 5000;
        let src: Vec<ScalarType> = (0..len)
            .map(|i| ((i as f64) * 0.001).sin() as ScalarType)
            .collect();
        let scalar = 0.314 as ScalarType;
        let init: Vec<ScalarType> = (0..len)
            .map(|i| ((i as f64) * 0.002).cos() as ScalarType)
            .collect();

        let mut acc_vecadd = init.clone();
        let mut acc_oracle = init.clone();

        vec_scaled_add_inplace(&src, scalar, &mut acc_vecadd);
        vecadd_oracle(&src, scalar, &mut acc_oracle);

        let mut max_rel_err = 0.0_f64;
        for i in 0..len {
            let diff = (acc_vecadd[i] as f64 - acc_oracle[i] as f64).abs();
            let denom = (acc_oracle[i] as f64).abs().max(1e-10);
            let rel_err = diff / denom;
            if rel_err > max_rel_err {
                max_rel_err = rel_err;
            }
        }

        let tolerance = if cfg!(feature = "f32") {
            1e-4_f64
        } else {
            1e-12_f64
        };
        assert!(
            max_rel_err < tolerance,
            "long sequence max relative error {max_rel_err} exceeds {tolerance}"
        );
    }
}
