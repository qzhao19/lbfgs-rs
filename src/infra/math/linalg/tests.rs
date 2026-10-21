//! Unit tests for the triangular solves in `triangular_solve`.

use super::triangular_solve::{solve_upper_triangular, solve_upper_triangular_transpose};
use crate::shared::numeric::ScalarType;

// ── Baseline (oracle) implementations ──────────────────────────────────
// The pre-vec_ops scalar loops, verbatim.  Any edit here must be
// re-anchored against the hand-derived fixtures below.

fn solve_upper_triangular_scalar(
    u: &[ScalarType],
    m: usize,
    k: usize,
    b: &mut [ScalarType],
) -> bool {
    debug_assert!(k <= m);

    for i in 0..k {
        if u[i * m + i] == 0.0 {
            return false;
        }
    }

    for i in (0..k).rev() {
        let mut sum = b[i];
        for j in (i + 1)..k {
            // u_{ii}x_i = b_i − Σ_{j=i+1}^{k} u_{ij}x_j
            sum -= u[i * m + j] * b[j];
        }
        b[i] = sum / u[i * m + i];
    }

    true
}

fn solve_upper_triangular_transpose_scalar(
    u: &[ScalarType],
    m: usize,
    k: usize,
    b: &mut [ScalarType],
) -> bool {
    debug_assert!(k <= m);

    for i in 0..k {
        if u[i * m + i] == 0.0 {
            return false;
        }
    }

    for i in 0..k {
        let mut sum = b[i];
        for j in 0..i {
            // (b_i − Σ_{j=0}^{i−1} u_{ji} x_j)
            sum -= u[j * m + i] * b[j];
        }
        b[i] = sum / u[i * m + i];
    }
    true
}

// ── Helpers ────────────────────────────────────────────────────────────

fn assert_close(actual: ScalarType, expected: ScalarType, msg: &str) {
    let tol = 1e4 * ScalarType::EPSILON * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tol,
        "{}: expected {}, got {}",
        msg,
        expected,
        actual
    );
}

/// Run the production solve and the scalar baseline on identical copies
/// of `b`; require the same status and, on success, the same solution
/// within rounding.
fn assert_matches_baseline(
    u: &[ScalarType],
    m: usize,
    k: usize,
    b: &[ScalarType],
    solve: fn(&[ScalarType], usize, usize, &mut [ScalarType]) -> bool,
    baseline: fn(&[ScalarType], usize, usize, &mut [ScalarType]) -> bool,
) {
    let mut b_opt = b.to_vec();
    let mut b_ref = b.to_vec();

    let ok = solve(u, m, k, &mut b_opt);
    let ok_ref = baseline(u, m, k, &mut b_ref);
    assert_eq!(ok, ok_ref, "success/failure status must agree");

    if ok_ref {
        for i in 0..k {
            assert_close(
                b_opt[i],
                b_ref[i],
                &format!("solution component {i} must match the baseline"),
            );
        }
    } else {
        // Contract: on failure neither implementation may touch `b`.
        assert_eq!(b_opt, b.to_vec(), "production b untouched on failure");
        assert_eq!(b_ref, b.to_vec(), "baseline b untouched on failure");
    }
}

/// Deterministic "random" matrix (no RNG dependency): varied off-diagonal
/// entries (|u| ≤ 2) with a guaranteed-nonzero, mildly dominant diagonal
/// (2 ≤ u ≤ 6), keeping the sweeps well-conditioned.
fn gen_upper(m: usize) -> Vec<ScalarType> {
    let mut u = vec![0.0 as ScalarType; m * m];
    for i in 0..m {
        for j in i..m {
            if i == j {
                u[i * m + j] = (((i + 1) * 13) % 5) as ScalarType + 2.0;
            } else {
                u[i * m + j] = (((i + 1) * 31 + (j + 1) * 17) % 5) as ScalarType - 2.0;
            }
        }
    }
    u
}

fn gen_rhs(m: usize) -> Vec<ScalarType> {
    (0..m)
        .map(|i| (((i + 1) * 5) % 7) as ScalarType - 3.0)
        .collect()
}

// ── Fixtures (anchor the baseline) ─────────────────────────────────────

#[test]
fn upper_solve_matches_hand_derived_fixture() {
    // U = [2 1 3; 0 4 5; 0 0 6], b = (2, 4, 6):
    //   x2 = 6/6 = 1
    //   x1 = (4 − 5·x2)/4 = −1/4
    //   x0 = (2 − 1·x1 − 3·x2)/2 = −3/8
    let u = [2.0 as ScalarType, 1.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0];
    let b = [2.0 as ScalarType, 4.0, 6.0];

    let mut x = b;
    assert!(solve_upper_triangular(&u, 3, 3, &mut x));
    assert_close(x[2], 1.0, "x2 = 6/6");
    assert_close(x[1], -0.25, "x1 = (4 − 5)/4");
    assert_close(x[0], -0.375, "x0 = (2 + 0.25 − 3)/2");

    let mut x = b;
    assert!(solve_upper_triangular_scalar(&u, 3, 3, &mut x));
    assert_close(x[2], 1.0, "baseline x2");
    assert_close(x[1], -0.25, "baseline x1");
    assert_close(x[0], -0.375, "baseline x0");
}

#[test]
fn upper_solve_transpose_matches_hand_derived_fixture() {
    // Same U; U' is lower triangular:
    //   x0 = 2/2 = 1
    //   x1 = (4 − u01·x0)/u11 = (4 − 1)/4 = 3/4
    //   x2 = (6 − u02·x0 − u12·x1)/u22 = (6 − 3 − 15/4)/6 = −1/8
    let u = [2.0 as ScalarType, 1.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0];
    let b = [2.0 as ScalarType, 4.0, 6.0];

    let mut x = b;
    assert!(solve_upper_triangular_transpose(&u, 3, 3, &mut x));
    assert_close(x[0], 1.0, "x0 = 2/2");
    assert_close(x[1], 0.75, "x1 = (4 − 1)/4");
    assert_close(x[2], -0.125, "x2 = (6 − 3 − 3.75)/6");

    let mut x = b;
    assert!(solve_upper_triangular_transpose_scalar(&u, 3, 3, &mut x));
    assert_close(x[0], 1.0, "baseline x0");
    assert_close(x[1], 0.75, "baseline x1");
    assert_close(x[2], -0.125, "baseline x2");
}

#[test]
fn solves_read_only_leading_block() {
    // m = 4 buffer, k = 2: only the leading 2×2 block may be consulted.
    // The zero at buffer entry (2, 2) would fail a diagonal scan running
    // to m — its acceptance proves the scan stops at k.
    let mut u = vec![0.0 as ScalarType; 16];
    u[0] = 2.0;
    u[1] = 1.0; // block row 0: [2 1]
    u[5] = 4.0; // block row 1: [0 4]

    // U·x = b with b = (4, 8):  x1 = 8/4 = 2, x0 = (4 − 1·2)/2 = 1.
    let b = [4.0 as ScalarType, 8.0];
    let mut x = b;
    assert!(solve_upper_triangular(&u, 4, 2, &mut x));
    assert_close(x[1], 2.0, "x1 = 8/4");
    assert_close(x[0], 1.0, "x0 = (4 − 2)/2");
    let mut x = b;
    assert!(solve_upper_triangular_scalar(&u, 4, 2, &mut x));
    assert_close(x[1], 2.0, "baseline x1");
    assert_close(x[0], 1.0, "baseline x0");

    // U'·x = b with b = (2, 6):  x0 = 2/2 = 1, x1 = (6 − 1·x0)/4 = 5/4.
    let b = [2.0 as ScalarType, 6.0];
    let mut x = b;
    assert!(solve_upper_triangular_transpose(&u, 4, 2, &mut x));
    assert_close(x[0], 1.0, "x0 = 2/2");
    assert_close(x[1], 1.25, "x1 = (6 − 1)/4");
    let mut x = b;
    assert!(solve_upper_triangular_transpose_scalar(&u, 4, 2, &mut x));
    assert_close(x[0], 1.0, "baseline x0");
    assert_close(x[1], 1.25, "baseline x1");
}

// ── Contract cases ──────────────────────────────────────────────────────

#[test]
fn solves_reject_zero_diagonal_without_touching_b() {
    // U = [1 2; 0 0]: the (1,1) entry is exactly zero → every
    // implementation must fail *before* modifying b.
    let u = [1.0 as ScalarType, 2.0, 0.0, 0.0];
    let b = [1.0 as ScalarType, 1.0];

    let mut x = b;
    assert!(!solve_upper_triangular(&u, 2, 2, &mut x));
    assert_eq!(x, b, "production b untouched on failure");
    let mut x = b;
    assert!(!solve_upper_triangular_scalar(&u, 2, 2, &mut x));
    assert_eq!(x, b, "baseline b untouched on failure");

    let mut x = b;
    assert!(!solve_upper_triangular_transpose(&u, 2, 2, &mut x));
    assert_eq!(x, b, "production b untouched on failure");
    let mut x = b;
    assert!(!solve_upper_triangular_transpose_scalar(&u, 2, 2, &mut x));
    assert_eq!(x, b, "baseline b untouched on failure");
}

#[test]
fn solves_handle_degenerate_blocks() {
    // k = 0: vacuous solve (mirrors col = 0 upstream); b untouched, true.
    let u = vec![0.0 as ScalarType; 4];
    let b = [3.0 as ScalarType];
    let mut x = b;
    assert!(solve_upper_triangular(&u, 2, 0, &mut x));
    assert!(solve_upper_triangular_transpose(&u, 2, 0, &mut x));
    assert_eq!(x, b, "k = 0 must not touch b");

    // k = 1: the row segment and the b tail are both empty — the SIMD
    // vec_dot / vec_scaled_add_inplace paths must no-op on empty slices.
    // The zero at buffer entry (1, 1) proves the scan stops at k = 1.
    let u = [4.0 as ScalarType, 0.0, 0.0, 0.0];
    let b = [8.0 as ScalarType];

    let mut x = b;
    assert!(solve_upper_triangular(&u, 2, 1, &mut x));
    assert_close(x[0], 2.0, "x0 = 8/4");
    assert_matches_baseline(
        &u,
        2,
        1,
        &b,
        solve_upper_triangular,
        solve_upper_triangular_scalar,
    );

    let mut x = b;
    assert!(solve_upper_triangular_transpose(&u, 2, 1, &mut x));
    assert_close(x[0], 2.0, "x0 = 8/4");
    assert_matches_baseline(
        &u,
        2,
        1,
        &b,
        solve_upper_triangular_transpose,
        solve_upper_triangular_transpose_scalar,
    );
}

// ── Differential sweep ─────────────────────────────────────────────────

#[test]
fn solves_match_baseline_across_generated_matrices() {
    // Deterministic pseudo-random upper matrices of every size m ≤ 6,
    // solved for every leading block k ≤ m, in both directions: the
    // vec_ops implementation must track the scalar baseline everywhere.
    for m in 1..=6 {
        let u = gen_upper(m);
        let b = gen_rhs(m);
        for k in 1..=m {
            assert_matches_baseline(
                &u,
                m,
                k,
                &b,
                solve_upper_triangular,
                solve_upper_triangular_scalar,
            );
            assert_matches_baseline(
                &u,
                m,
                k,
                &b,
                solve_upper_triangular_transpose,
                solve_upper_triangular_transpose_scalar,
            );
        }
    }
}
