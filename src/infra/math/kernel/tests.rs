//! Covers, for every kernel where applicable:
//! 0. input validation
//! 1. basic correctness
//! 2. boundary / alignment (scalar · lane · block · remainder)
//! 3. numeric stability
//! 4. SIMD path vs scalar oracle consistency
//! 5. special IEEE-754 values
//! 6. L-BFGS usage scenarios
//! 7. in-place behaviour

use super::vec_ops::{
    vec_diff, vec_dot, vec_ncpy, vec_norm2, vec_scale_inplace, vec_scaled_add,
    vec_scaled_add_inplace,
};
use crate::shared::numeric::ScalarType;

// ── Shared helpers ───

fn arange(len: usize) -> Vec<ScalarType> {
    (0..len).map(|i| i as ScalarType).collect()
}

fn fill(len: usize, value: ScalarType) -> Vec<ScalarType> {
    vec![value; len]
}

/// Deterministic pseudo-random in [-1, 1].
fn prand(len: usize, seed: f64) -> Vec<ScalarType> {
    (0..len)
        .map(|i| ((i as f64 * 1.234 + seed).sin()) as ScalarType)
        .collect()
}

fn epsilon() -> ScalarType {
    if cfg!(feature = "f32") {
        1e-5 as ScalarType
    } else {
        1e-12 as ScalarType
    }
}

/// Relative comparison tolerant of multi-accumulator FMA ordering.
fn approx_eq(got: ScalarType, expected: ScalarType) -> bool {
    if got.is_nan() && expected.is_nan() {
        return true;
    }
    let abs_diff = (got - expected).abs();
    let scale = expected.abs().max(1.0 as ScalarType);
    let rel_tol = if cfg!(feature = "f32") {
        5e-5 as ScalarType
    } else {
        1e-12 as ScalarType
    };
    abs_diff < scale * rel_tol
}

fn assert_slice_eq(got: &[ScalarType], expected: &[ScalarType], ctx: &str) {
    assert_eq!(got.len(), expected.len(), "{ctx}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        if g.is_nan() && e.is_nan() {
            continue;
        }
        assert!(
            approx_eq(g, e),
            "{ctx} idx={i}: got {g}, expected {e}, diff {}",
            (g - e).abs()
        );
    }
}

/// Lengths that hit scalar / 1-lane / 1-block / mixed-remainder / large paths
/// for both f32 (lane=4, block=16) and f64 (lane=2, block=4).
fn alignment_lengths() -> &'static [usize] {
    &[
        0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 20, 31, 32, 64, 100, 256, 1024,
    ]
}

// ── Scalar oracles (ground truth, same order as *_ansi) ─────────────────────

fn oracle_dot(x: &[ScalarType], y: &[ScalarType]) -> ScalarType {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

fn oracle_diff(x: &[ScalarType], y: &[ScalarType]) -> Vec<ScalarType> {
    x.iter().zip(y).map(|(a, b)| a - b).collect()
}

fn oracle_ncpy(x: &[ScalarType]) -> Vec<ScalarType> {
    x.iter().map(|&v| -v).collect()
}

fn oracle_norm2(x: &[ScalarType], squared: bool) -> ScalarType {
    let s: ScalarType = x.iter().map(|&v| v * v).sum();
    if squared {
        s
    } else {
        s.sqrt()
    }
}

fn oracle_scale(x: &[ScalarType], scalar: ScalarType) -> Vec<ScalarType> {
    x.iter().map(|&v| v * scalar).collect()
}

fn oracle_scaled_add(x: &[ScalarType], y: &[ScalarType], s: ScalarType) -> Vec<ScalarType> {
    x.iter().zip(y).map(|(a, b)| a * s + b).collect()
}

fn oracle_scaled_add_inplace(
    src: &[ScalarType],
    scalar: ScalarType,
    acc: &[ScalarType],
) -> Vec<ScalarType> {
    acc.iter().zip(src).map(|(a, s)| a + s * scalar).collect()
}

// 0. Input validation

mod validation {
    use super::*;

    #[test]
    #[should_panic(expected = "length mismatch")]
    #[cfg(debug_assertions)]
    fn dot_length_mismatch() {
        vec_dot(&[1.0 as ScalarType; 4], &[1.0 as ScalarType; 3]);
    }

    #[test]
    fn dot_empty_is_zero() {
        assert_eq!(vec_dot(&[], &[]), 0.0 as ScalarType);
    }

    #[test]
    #[should_panic(expected = "x and y must have the same length")]
    #[cfg(debug_assertions)]
    fn diff_xy_mismatch() {
        let mut out = vec![0.0 as ScalarType; 4];
        vec_diff(&[1.0 as ScalarType; 4], &[1.0 as ScalarType; 3], &mut out);
    }

    #[test]
    #[should_panic(expected = "out must have the same length as x and y")]
    #[cfg(debug_assertions)]
    fn diff_out_mismatch() {
        let mut out = vec![0.0 as ScalarType; 3];
        vec_diff(&[1.0 as ScalarType; 4], &[1.0 as ScalarType; 4], &mut out);
    }

    #[test]
    fn diff_empty_ok() {
        let mut out: Vec<ScalarType> = vec![];
        vec_diff(&[], &[], &mut out);
        assert!(out.is_empty());
    }

    #[test]
    #[should_panic(expected = "vector length mismatch")]
    #[cfg(debug_assertions)]
    fn ncpy_length_mismatch() {
        let mut out = vec![0.0 as ScalarType; 3];
        vec_ncpy(&[1.0 as ScalarType; 4], &mut out);
    }

    #[test]
    fn ncpy_empty_ok() {
        let mut out: Vec<ScalarType> = vec![];
        vec_ncpy(&[], &mut out);
        assert!(out.is_empty());
    }

    #[test]
    #[should_panic(expected = "x must be non-empty")]
    #[cfg(debug_assertions)]
    fn norm2_empty_panics() {
        vec_norm2(&[], false);
    }

    #[test]
    #[should_panic(expected = "x must be non-empty")]
    #[cfg(debug_assertions)]
    fn scale_inplace_empty_panics() {
        let mut x: Vec<ScalarType> = vec![];
        vec_scale_inplace(&mut x, 2.0 as ScalarType);
    }

    #[test]
    #[should_panic(expected = "x and y must have the same length")]
    #[cfg(debug_assertions)]
    fn scaled_add_xy_mismatch() {
        let mut out = vec![0.0 as ScalarType; 4];
        vec_scaled_add(
            &[1.0 as ScalarType; 4],
            &[1.0 as ScalarType; 3],
            1.0 as ScalarType,
            &mut out,
        );
    }

    #[test]
    #[should_panic(expected = "output vector must have the same length as input")]
    #[cfg(debug_assertions)]
    fn scaled_add_out_mismatch() {
        let mut out = vec![0.0 as ScalarType; 3];
        vec_scaled_add(
            &[1.0 as ScalarType; 4],
            &[1.0 as ScalarType; 4],
            1.0 as ScalarType,
            &mut out,
        );
    }

    #[test]
    fn scaled_add_empty_ok() {
        let mut out: Vec<ScalarType> = vec![];
        vec_scaled_add(&[], &[], 2.0 as ScalarType, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn scaled_add_inplace_empty_ok() {
        let mut acc: Vec<ScalarType> = vec![];
        vec_scaled_add_inplace(&[], 2.0 as ScalarType, &mut acc);
        assert!(acc.is_empty());
    }

    #[test]
    fn scaled_add_inplace_scalar_zero_unchanged() {
        let src = vec![1.0 as ScalarType, 2.0, 3.0];
        let mut acc = vec![10.0 as ScalarType, 20.0, 30.0];
        let expected = acc.clone();
        vec_scaled_add_inplace(&src, 0.0 as ScalarType, &mut acc);
        assert_eq!(acc, expected);
    }
}

// 1. Basic correctness

mod basic_correctness {
    use super::*;

    #[test]
    fn dot_known_cases() {
        assert_eq!(
            vec_dot(
                &[1.0 as ScalarType, 2.0, 3.0],
                &[4.0 as ScalarType, 5.0, 6.0]
            ),
            32.0 as ScalarType
        );
        assert_eq!(
            vec_dot(&[1.0 as ScalarType, 0.0], &[0.0 as ScalarType, 1.0]),
            0.0 as ScalarType
        );
        assert_eq!(
            vec_dot(&[-1.0 as ScalarType, 2.0], &[3.0 as ScalarType, -4.0]),
            -11.0 as ScalarType
        );
        assert_eq!(
            vec_dot(&[0.5 as ScalarType, 0.25], &[2.0 as ScalarType, 4.0]),
            2.0 as ScalarType
        );
        assert_eq!(
            vec_dot(&fill(4, 0.0 as ScalarType), &fill(4, 1.0 as ScalarType)),
            0.0 as ScalarType
        );
    }

    #[test]
    fn diff_known_cases() {
        let x = vec![5.0 as ScalarType, 4.0, 3.0, 2.0];
        let y = vec![1.0 as ScalarType, 1.0, 1.0, 1.0];
        let mut out = vec![0.0 as ScalarType; 4];
        vec_diff(&x, &y, &mut out);
        assert_eq!(out, vec![4.0 as ScalarType, 3.0, 2.0, 1.0]);

        // x == y → 0
        vec_diff(&x, &x, &mut out);
        assert_eq!(out, fill(4, 0.0 as ScalarType));

        // x - 0 → x
        let z = fill(4, 0.0 as ScalarType);
        vec_diff(&x, &z, &mut out);
        assert_eq!(out, x);

        // 0 - y → -y
        vec_diff(&z, &y, &mut out);
        assert_eq!(out, oracle_ncpy(&y));
    }

    #[test]
    fn ncpy_known_cases() {
        let x = vec![1.0 as ScalarType, -2.0, 0.5, -0.0_f64 as ScalarType];
        let mut out = vec![0.0 as ScalarType; x.len()];
        vec_ncpy(&x, &mut out);
        assert_slice_eq(&out, &oracle_ncpy(&x), "ncpy");

        // double negation is identity
        let mut back = vec![0.0 as ScalarType; x.len()];
        vec_ncpy(&out, &mut back);
        assert_slice_eq(&back, &x, "ncpy²");
    }

    #[test]
    fn norm2_known_cases() {
        assert_eq!(vec_norm2(&[0.0 as ScalarType; 4], false), 0.0 as ScalarType);
        assert_eq!(
            vec_norm2(&[3.0 as ScalarType, 4.0], false),
            5.0 as ScalarType
        );
        assert_eq!(
            vec_norm2(&[3.0 as ScalarType, 4.0], true),
            25.0 as ScalarType
        );
        // sign-invariant
        let x = vec![1.0 as ScalarType, -2.0, 3.0, -4.0];
        let neg: Vec<_> = x.iter().map(|&v| -v).collect();
        assert!(approx_eq(vec_norm2(&x, false), vec_norm2(&neg, false)));
        // squared flag consistent with sqrt
        let n = vec_norm2(&x, false);
        let n2 = vec_norm2(&x, true);
        assert!(approx_eq(n * n, n2));
    }

    #[test]
    fn scale_inplace_known_scalars() {
        for &s in &[0.0 as ScalarType, 1.0, -1.0, 2.0, 0.5, -0.5, 1e6, 1e-6] {
            let mut x = vec![1.5 as ScalarType, -2.0, 0.25, 4.0];
            let expected = oracle_scale(&x, s);
            vec_scale_inplace(&mut x, s);
            assert_slice_eq(&x, &expected, &format!("scale s={s}"));
        }
    }

    #[test]
    fn scaled_add_known_scalars() {
        let x = vec![1.5 as ScalarType, -2.0, 0.25, 4.0];
        let y = vec![10.0 as ScalarType, 20.0, 30.0, 40.0];
        for &s in &[0.0 as ScalarType, 1.0, -1.0, 2.0, 0.5, -0.5] {
            let mut out = vec![0.0 as ScalarType; x.len()];
            vec_scaled_add(&x, &y, s, &mut out);
            assert_slice_eq(&out, &oracle_scaled_add(&x, &y, s), &format!("axpy s={s}"));
        }
        // s=0 → y; s=1 → x+y; s=-1 → y-x
        let mut out = vec![0.0 as ScalarType; x.len()];
        vec_scaled_add(&x, &y, 0.0 as ScalarType, &mut out);
        assert_eq!(out, y);
    }

    #[test]
    fn scaled_add_inplace_known() {
        let src = vec![1.0 as ScalarType, 2.0, 3.0, 4.0];
        let mut acc = vec![10.0 as ScalarType, 20.0, 30.0, 40.0];
        let expected = oracle_scaled_add_inplace(&src, 2.0 as ScalarType, &acc);
        vec_scaled_add_inplace(&src, 2.0 as ScalarType, &mut acc);
        assert_slice_eq(&acc, &expected, "axpy_inplace");

        // src zeros leave acc unchanged
        let z = fill(4, 0.0 as ScalarType);
        let before = acc.clone();
        vec_scaled_add_inplace(&z, 5.0 as ScalarType, &mut acc);
        assert_eq!(acc, before);
    }
}

// 2. Boundary & alignment  +  4. SIMD vs scalar oracle
//    (one loop covers lane / block / remainder / large for every kernel)

mod boundary_and_oracle {
    use super::*;

    fn check_all_at(len: usize) {
        let x = prand(len, 0.567);
        let y = prand(len, 1.789);
        let s = 1.75 as ScalarType;

        // vec_dot
        {
            let got = vec_dot(&x, &y);
            let exp = oracle_dot(&x, &y);
            assert!(
                approx_eq(got, exp),
                "dot len={len}: got {got}, expected {exp}, diff {}",
                (got - exp).abs()
            );
        }

        // vec_diff
        {
            let mut out = vec![0.0 as ScalarType; len];
            vec_diff(&x, &y, &mut out);
            assert_slice_eq(&out, &oracle_diff(&x, &y), &format!("diff len={len}"));
        }

        // vec_ncpy
        {
            let mut out = vec![0.0 as ScalarType; len];
            vec_ncpy(&x, &mut out);
            assert_slice_eq(&out, &oracle_ncpy(&x), &format!("ncpy len={len}"));
        }

        // vec_norm2 (skip empty — contract requires non-empty)
        if len > 0 {
            for squared in [false, true] {
                let got = vec_norm2(&x, squared);
                let exp = oracle_norm2(&x, squared);
                assert!(
                    approx_eq(got, exp),
                    "norm2(squared={squared}) len={len}: got {got}, expected {exp}"
                );
            }
        }

        // vec_scale_inplace (skip empty)
        if len > 0 {
            let mut buf = x.clone();
            vec_scale_inplace(&mut buf, s);
            assert_slice_eq(
                &buf,
                &oracle_scale(&x, s),
                &format!("scale_inplace len={len}"),
            );
        }

        // vec_scaled_add
        {
            let mut out = vec![0.0 as ScalarType; len];
            vec_scaled_add(&x, &y, s, &mut out);
            assert_slice_eq(
                &out,
                &oracle_scaled_add(&x, &y, s),
                &format!("scaled_add len={len}"),
            );
        }

        // vec_scaled_add_inplace
        {
            let mut acc = y.clone();
            vec_scaled_add_inplace(&x, s, &mut acc);
            assert_slice_eq(
                &acc,
                &oracle_scaled_add_inplace(&x, s, &y),
                &format!("scaled_add_inplace len={len}"),
            );
        }
    }

    #[test]
    fn all_alignment_lengths() {
        for &n in alignment_lengths() {
            check_all_at(n);
        }
    }

    /// f32/f64-specific one-block and one-block+scalar lengths.
    #[test]
    fn block_boundaries() {
        let one_block = if cfg!(feature = "f32") { 16 } else { 4 };
        let one_block_plus = one_block + 1;
        let one_lane = if cfg!(feature = "f32") { 4 } else { 2 };
        for n in [one_lane, one_block, one_block_plus, one_block + one_lane] {
            check_all_at(n);
        }
    }

    #[test]
    fn long_vector_4096() {
        check_all_at(4096);
    }
}

// 3. Numeric stability

mod numeric_stability {
    use super::*;

    #[test]
    fn large_values_finite() {
        let big = 1e10 as ScalarType;
        let x = vec![big, big, big, big];
        let y = vec![big, -big, big, -big];

        assert!(vec_dot(&x, &x).is_finite());
        assert!(vec_norm2(&x, false).is_finite());

        let mut out = vec![0.0 as ScalarType; 4];
        vec_diff(&x, &y, &mut out);
        assert!(out.iter().all(|v| v.is_finite()));

        vec_scaled_add(&x, &y, 0.5 as ScalarType, &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn small_values_no_underflow_to_zero() {
        let tiny = 1e-10 as ScalarType;
        let x = fill(4, tiny);
        assert!(
            vec_dot(&x, &x) > 0.0 as ScalarType,
            "dot underflowed to zero"
        );
        assert!(
            vec_norm2(&x, false) > 0.0 as ScalarType,
            "norm2 underflowed to zero"
        );
    }

    #[test]
    fn mixed_scale() {
        let big = 1e8 as ScalarType;
        let tiny = 1e-8 as ScalarType;
        let x = vec![big, tiny];
        let y = vec![tiny, big];
        // big*tiny + tiny*big = 2
        assert!(approx_eq(vec_dot(&x, &y), 2.0 as ScalarType));
        assert!(approx_eq(vec_norm2(&x, false), oracle_norm2(&x, false)));
    }

    #[test]
    fn cancellation_dot() {
        let x = vec![1.0 as ScalarType, -1.0, 1.0, -1.0];
        let y = fill(4, 1.0 as ScalarType);
        assert_eq!(vec_dot(&x, &y), 0.0 as ScalarType);
    }

    #[test]
    fn long_accumulation_10000() {
        let n = 10_000;
        let x = arange(n);
        let y: Vec<_> = x.iter().map(|&v| v * 0.5 as ScalarType).collect();
        assert!(approx_eq(vec_dot(&x, &y), oracle_dot(&x, &y)));
        assert!(approx_eq(vec_norm2(&x, false), oracle_norm2(&x, false)));
    }

    #[test]
    fn catastrophic_cancellation_diff() {
        // x ≈ y at large magnitude → tiny difference must survive
        let base = 1e8 as ScalarType;
        let x: Vec<_> = (0..32)
            .map(|i| base + (i as ScalarType) * 1e-3 as ScalarType)
            .collect();
        let y = fill(32, base);
        let mut out = vec![0.0 as ScalarType; 32];
        vec_diff(&x, &y, &mut out);
        assert_slice_eq(&out, &oracle_diff(&x, &y), "cancel-diff");
    }

    #[test]
    fn cumulative_axpy_inplace() {
        // Many micro-increments must stay close to oracle (FMA path).
        let n = 256;
        let src = prand(n, 0.3);
        let mut acc = fill(n, 0.0 as ScalarType);
        let mut ref_acc = fill(n, 0.0 as ScalarType);
        let step = 1e-3 as ScalarType;
        for _ in 0..100 {
            vec_scaled_add_inplace(&src, step, &mut acc);
            ref_acc = oracle_scaled_add_inplace(&src, step, &ref_acc);
        }
        assert_slice_eq(&acc, &ref_acc, "cumul-axpy");
    }
}

// 5. Special floating-point values (IEEE 754)

mod special_values {
    use super::*;

    #[test]
    fn nan_propagates() {
        let nan_x = vec![ScalarType::NAN, 1.0 as ScalarType, 2.0, 3.0];
        let y = fill(4, 1.0 as ScalarType);

        assert!(vec_dot(&nan_x, &y).is_nan());
        assert!(vec_norm2(&nan_x, false).is_nan());
        assert!(vec_norm2(&nan_x, true).is_nan());

        let mut out = vec![0.0 as ScalarType; 4];
        vec_diff(&nan_x, &y, &mut out);
        assert!(out[0].is_nan());

        vec_ncpy(&nan_x, &mut out);
        assert!(out[0].is_nan());

        let mut s = nan_x.clone();
        vec_scale_inplace(&mut s, 2.0 as ScalarType);
        assert!(s[0].is_nan());

        vec_scaled_add(&nan_x, &y, 1.0 as ScalarType, &mut out);
        assert!(out[0].is_nan());

        let mut acc = y.clone();
        vec_scaled_add_inplace(&nan_x, 1.0 as ScalarType, &mut acc);
        assert!(acc[0].is_nan());
    }

    #[test]
    fn inf_behaviour() {
        let x = vec![ScalarType::INFINITY, 1.0 as ScalarType];
        assert!(vec_norm2(&x, true).is_infinite());
        assert!(vec_norm2(&x, false).is_infinite());

        // 0 * Inf = NaN in IEEE 754 for the product itself
        let z = vec![0.0 as ScalarType, 0.0];
        let inf = vec![ScalarType::INFINITY, 1.0 as ScalarType];
        assert!(vec_dot(&z, &inf).is_nan());

        // pure Inf via ncpy → -Inf
        let mut out = vec![0.0 as ScalarType; 2];
        vec_ncpy(&[ScalarType::INFINITY, 1.0 as ScalarType], &mut out);
        assert!(out[0].is_infinite() && out[0].is_sign_negative());
        assert_eq!(out[1], -1.0 as ScalarType);
    }

    #[test]
    fn negative_zero() {
        let x = vec![-0.0_f64 as ScalarType, 1.0 as ScalarType];
        let y = vec![1.0 as ScalarType, 1.0 as ScalarType];
        // (-0)*1 + 1*1 = 1
        assert_eq!(vec_dot(&x, &y), 1.0 as ScalarType);
        // (-0)² + 1² = 1
        assert_eq!(vec_norm2(&x, true), 1.0 as ScalarType);
    }

    #[test]
    fn scale_scalar_nan_and_inf() {
        let mut x = vec![1.0 as ScalarType, 2.0, 3.0, 4.0];
        vec_scale_inplace(&mut x, ScalarType::NAN);
        assert!(x.iter().all(|v| v.is_nan()));

        let mut x = vec![1.0 as ScalarType, -2.0, 0.0, 3.0];
        vec_scale_inplace(&mut x, ScalarType::INFINITY);
        assert!(x[0].is_infinite() && x[0].is_sign_positive());
        assert!(x[1].is_infinite() && x[1].is_sign_negative());
        // 0 * Inf = NaN
        assert!(x[2].is_nan());
    }

    /// vec_norm2 of [0, Inf] is +Inf, NOT NaN — there is no 0·Inf path
    #[test]
    fn norm2_zero_and_inf_yields_inf() {
        let x = vec![0.0 as ScalarType, ScalarType::INFINITY];
        assert!(vec_norm2(&x, true).is_infinite());
        assert!(vec_norm2(&x, false).is_infinite());
    }

    /// (-0.0)² + 1.0 == 1.0 (sign of zero does not leak through squaring).
    #[test]
    fn norm2_negative_zero_squared() {
        let x = vec![-0.0_f64 as ScalarType, 1.0 as ScalarType];
        assert_eq!(vec_norm2(&x, true), 1.0 as ScalarType);
    }

    /// All-NaN input → NaN on both `squared` flags.
    #[test]
    fn norm2_all_nan_both_flags() {
        let v = vec![ScalarType::NAN; 4];
        assert!(vec_norm2(&v, false).is_nan());
        assert!(vec_norm2(&v, true).is_nan());
    }

    /// acc[i] += INF·0 → NaN (FMA drops into the NaN sink).
    #[test]
    fn scaled_add_inplace_inf_times_zero_is_nan() {
        let src = vec![ScalarType::INFINITY, 1.0 as ScalarType];
        let mut acc = vec![5.0 as ScalarType, 5.0 as ScalarType];
        vec_scaled_add_inplace(&src, 0.0 as ScalarType, &mut acc);
        assert!(acc[0].is_nan(), "5 + Inf·0 should be NaN, got {}", acc[0]);
        assert_eq!(acc[1], 5.0 as ScalarType);
    }

    /// INF + 0·0 == INF (signed infinite survives the FMA chain).
    #[test]
    fn scaled_add_inplace_acc_inf_src_zero() {
        let src = vec![0.0 as ScalarType, 1.0 as ScalarType];
        let mut acc = vec![ScalarType::INFINITY, 5.0 as ScalarType];
        vec_scaled_add_inplace(&src, 0.0 as ScalarType, &mut acc);
        assert!(acc[0].is_infinite() && acc[0].is_sign_positive());
        assert_eq!(acc[1], 5.0 as ScalarType);
    }

    /// scalar=NaN broadcasts to every lane of the output vector.
    #[test]
    fn scaled_add_inplace_scalar_nan_broadcasts() {
        let src = vec![1.0 as ScalarType; 8];
        let mut acc = vec![0.0 as ScalarType; 8];
        vec_scaled_add_inplace(&src, ScalarType::NAN, &mut acc);
        assert!(acc.iter().all(|v| v.is_nan()));
    }

    /// -0.0 · 3 + 42 == 42 (signed-zero is preserved through FMA but does not flip 42).
    #[test]
    fn scaled_add_inplace_negative_zero_src_sign() {
        let src = vec![-0.0_f64 as ScalarType, 1.0 as ScalarType];
        let mut acc = vec![42.0 as ScalarType, 42.0 as ScalarType];
        vec_scaled_add_inplace(&src, 3.0 as ScalarType, &mut acc);
        assert_eq!(
            acc[0], 42.0 as ScalarType,
            "42 + (-0)*3 should be 42, got {}",
            acc[0]
        );
        assert_eq!(acc[1], 45.0 as ScalarType);
    }

    /// -(-0.0) == +0.0 (sign bit flips on NEON `vnegq`).
    #[test]
    fn ncpy_negative_zero_sign_flip() {
        let x = vec![-0.0_f64 as ScalarType, 0.0 as ScalarType];
        let mut out = vec![0.0 as ScalarType; 2];
        vec_ncpy(&x, &mut out);
        assert!(
            !out[0].is_sign_negative() && out[0] == 0.0 as ScalarType,
            "idx=0: -(-0) should be +0, got {:?} (sign_neg={})",
            out[0],
            out[0].is_sign_negative()
        );
    }

    /// dot of [MAX, MIN, 0, 1, -1] vs [1, 1, 1, 0, 0]: SIMD vs oracle within 100·ε.
    #[test]
    fn dot_extreme_values_vs_oracle() {
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
        let exp = oracle_dot(&x, &y);
        let rel = if exp.abs() > 1.0 as ScalarType {
            (got - exp).abs() / exp.abs()
        } else {
            (got - exp).abs()
        };
        assert!(
            rel < epsilon() * 100.0,
            "dot extreme: got {got}, exp {exp}, rel {rel}"
        );
    }
}

// 7. In-place behaviour & aliasing safety

mod inplace {
    use super::*;

    #[test]
    fn scale_inplace_mutates_only_target() {
        let mut x = arange(32);
        let original = x.clone();
        vec_scale_inplace(&mut x, 3.0 as ScalarType);
        assert_slice_eq(&x, &oracle_scale(&original, 3.0 as ScalarType), "scale");
    }

    #[test]
    fn scale_inplace_compose() {
        let mut x = arange(64);
        let original = x.clone();
        vec_scale_inplace(&mut x, 2.0 as ScalarType);
        vec_scale_inplace(&mut x, 0.5 as ScalarType);
        assert_slice_eq(&x, &original, "scale 2 then 0.5");
    }

    #[test]
    fn scaled_add_inplace_accumulates() {
        let src = arange(32);
        let mut acc = fill(32, 1.0 as ScalarType);
        let mut expected = acc.clone();
        for s in [1.0 as ScalarType, 2.0, -0.5] {
            vec_scaled_add_inplace(&src, s, &mut acc);
            expected = oracle_scaled_add_inplace(&src, s, &expected);
        }
        assert_slice_eq(&acc, &expected, "triple-axpy");
    }

    #[test]
    fn scaled_add_inplace_acc_equals_src() {
        // Aliasing-style: acc starts as a clone of src (distinct buffers).
        let src = prand(64, 2.2);
        let mut acc = src.clone();
        let expected = oracle_scaled_add_inplace(&src, 1.0 as ScalarType, &src);
        vec_scaled_add_inplace(&src, 1.0 as ScalarType, &mut acc);
        assert_slice_eq(&acc, &expected, "acc≡src clone");
    }

    #[test]
    fn ncpy_round_trip() {
        let x = prand(128, 0.9);
        let mut tmp = vec![0.0 as ScalarType; x.len()];
        let mut back = vec![0.0 as ScalarType; x.len()];
        vec_ncpy(&x, &mut tmp);
        vec_ncpy(&tmp, &mut back);
        assert_slice_eq(&back, &x, "ncpy round-trip");
    }

    #[test]
    fn diff_out_equals_x_buffer_independent() {
        // out is a separate buffer initially equal to x; result must be x-y
        // written into out without reading stale out.
        let x = prand(48, 0.1);
        let y = prand(48, 0.2);
        let mut out = x.clone();
        vec_diff(&x, &y, &mut out);
        assert_slice_eq(&out, &oracle_diff(&x, &y), "diff out←x clone");
    }

    #[test]
    fn scaled_add_out_equals_y_buffer_independent() {
        let x = prand(48, 0.3);
        let y = prand(48, 0.4);
        let mut out = y.clone();
        vec_scaled_add(&x, &y, 2.0 as ScalarType, &mut out);
        assert_slice_eq(
            &out,
            &oracle_scaled_add(&x, &y, 2.0 as ScalarType),
            "axpy out←y clone",
        );
    }

    #[test]
    fn consecutive_calls_deterministic() {
        let n = 256;
        let x = prand(n, 1.1);
        let y = prand(n, 2.2);

        let d0 = vec_dot(&x, &y);
        let n0 = vec_norm2(&x, false);
        for _ in 0..50 {
            assert!(approx_eq(vec_dot(&x, &y), d0));
            assert!(approx_eq(vec_norm2(&x, false), n0));
        }

        let mut out0 = vec![0.0 as ScalarType; n];
        let mut out1 = vec![0.0 as ScalarType; n];
        vec_diff(&x, &y, &mut out0);
        vec_diff(&x, &y, &mut out1);
        assert_eq!(out0, out1);
    }
}

// 6. L-BFGS usage scenarios (cross-kernel compositions)

mod lbfgs_scenarios {
    use super::*;

    /// s = x - xp, y = g - gp, ys = y·s  (correction-pair construction).
    #[test]
    fn correction_pair_s_y_ys() {
        let n = 512;
        let xp = prand(n, 0.01);
        let gp = prand(n, 0.02);
        let step = 0.01 as ScalarType;
        let d = prand(n, 0.03);
        let x: Vec<_> = (0..n).map(|i| xp[i] + step * d[i]).collect();
        let g: Vec<_> = (0..n).map(|i| gp[i] + 0.1 as ScalarType * d[i]).collect();

        let mut s = vec![0.0 as ScalarType; n];
        let mut y = vec![0.0 as ScalarType; n];
        vec_diff(&x, &xp, &mut s);
        vec_diff(&g, &gp, &mut y);

        // s ≈ step * d
        for i in 0..n {
            assert!(
                approx_eq(s[i], step * d[i]),
                "s idx={i}: got {}, expected {}",
                s[i],
                step * d[i]
            );
        }

        let ys = vec_dot(&y, &s);
        assert!(approx_eq(ys, oracle_dot(&y, &s)));
        assert!(ys.is_finite());
    }

    /// d = -g  (initial search direction via vec_ncpy).
    #[test]
    fn negative_gradient_direction() {
        let g = prand(1024, 0.5);
        let mut d = vec![0.0 as ScalarType; g.len()];
        vec_ncpy(&g, &mut d);
        assert_slice_eq(&d, &oracle_ncpy(&g), "d=-g");

        // stepsize = 1 / ||d||
        let dnorm = vec_norm2(&d, false);
        assert!(dnorm > 0.0 as ScalarType);
        let step = 1.0 as ScalarType / dnorm;
        assert!(step.is_finite());
    }

    /// ||g|| / max(1, ||x||) convergence ratio, and yy = ||y||² ≡ y·y.
    #[test]
    fn gradient_norm_and_yy() {
        let n = 1000;
        let x = prand(n, 0.001);
        let g = prand(n, 0.003);

        let gnorm = vec_norm2(&g, false);
        let xnorm = vec_norm2(&x, false);
        assert!(gnorm > 0.0 && xnorm > 0.0);
        let _ratio = gnorm / xnorm.max(1.0 as ScalarType);

        let y = prand(512, 0.7);
        let yy_norm = vec_norm2(&y, true);
        let yy_dot = vec_dot(&y, &y);
        assert!(approx_eq(yy_norm, yy_dot));
    }

    /// x ← x + step * d   (line-search trial point via vec_scaled_add).
    #[test]
    fn linesearch_trial_point() {
        let n = 256;
        let xp = prand(n, 1.0);
        let d = prand(n, 2.0);
        let step = 0.25 as ScalarType;
        let mut x = vec![0.0 as ScalarType; n];
        vec_scaled_add(&d, &xp, step, &mut x);
        assert_slice_eq(&x, &oracle_scaled_add(&d, &xp, step), "x = xp + step*d");
    }

    /// Two-loop recursion primitives:
    ///   d -= alpha * y     and     d += coef * s
    ///   d *= (ys/yy)
    #[test]
    fn two_loop_primitives() {
        let n = 128;
        let s = prand(n, 0.1);
        let y = prand(n, 0.2);
        let mut d = prand(n, 0.3);

        let alpha = 0.5 as ScalarType;
        let expected_fwd = oracle_scaled_add_inplace(&y, -alpha, &d);
        vec_scaled_add_inplace(&y, -alpha, &mut d);
        assert_slice_eq(&d, &expected_fwd, "d -= α y");

        let ys = vec_dot(&y, &s);
        let yy = vec_norm2(&y, true);
        let scale = ys / yy;
        let expected_scale = oracle_scale(&d, scale);
        vec_scale_inplace(&mut d, scale);
        assert_slice_eq(&d, &expected_scale, "d *= ys/yy");

        let coef = 0.3 as ScalarType;
        let expected_bwd = oracle_scaled_add_inplace(&s, coef, &d);
        vec_scaled_add_inplace(&s, coef, &mut d);
        assert_slice_eq(&d, &expected_bwd, "d += coef s");
    }

    /// Weight · feature row (LogLoss path) + gradient · direction.
    #[test]
    fn weight_dot_feature_and_curvature() {
        let n = 1000;
        let w = prand(n, 0.001);
        let x = prand(n, 0.003);
        let g = prand(n, 0.002);
        let d = prand(n, 1.5);

        assert!(approx_eq(vec_dot(&w, &x), oracle_dot(&w, &x)));
        assert!(approx_eq(vec_dot(&g, &d), oracle_dot(&g, &d)));
    }

    /// Common ML dims across all kernels.
    #[test]
    fn common_ml_dimensions() {
        for &n in &[4usize, 8, 16, 32, 64, 128, 256, 784, 1024, 4096] {
            let x = arange(n);
            let y = fill(n, 1.0 as ScalarType);
            let s = 0.5 as ScalarType;

            assert!(approx_eq(vec_dot(&x, &y), oracle_dot(&x, &y)));

            let mut out = vec![0.0 as ScalarType; n];
            vec_diff(&x, &y, &mut out);
            assert_slice_eq(&out, &oracle_diff(&x, &y), &format!("diff n={n}"));

            vec_ncpy(&x, &mut out);
            assert_slice_eq(&out, &oracle_ncpy(&x), &format!("ncpy n={n}"));

            assert!(approx_eq(vec_norm2(&x, false), oracle_norm2(&x, false)));

            let mut buf = x.clone();
            vec_scale_inplace(&mut buf, s);
            assert_slice_eq(&buf, &oracle_scale(&x, s), &format!("scale n={n}"));

            vec_scaled_add(&x, &y, s, &mut out);
            assert_slice_eq(&out, &oracle_scaled_add(&x, &y, s), &format!("axpy n={n}"));

            let mut acc = y.clone();
            vec_scaled_add_inplace(&x, s, &mut acc);
            assert_slice_eq(
                &acc,
                &oracle_scaled_add_inplace(&x, s, &y),
                &format!("axpy_ip n={n}"),
            );
        }
    }
}
