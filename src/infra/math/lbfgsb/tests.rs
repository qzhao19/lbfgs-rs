//! Unit tests for the Generalized Cauchy Point port (`cauchy` / `middle_mat_vec_prod` /
//! `hpsolb`). Expected values below were hand-derived from the quadratic
//! model Q(x + s) = g's + 1/2·s'Bs and cross-checked against the Fortran
//! control flow.

use super::cauchy::{cauchy, hpsolb, CauchyError, CauchyWorkspace};
use super::middle_mat::middle_mat_vec_prod;

use crate::infra::math::bfgs_mat::LimitedMemBfgsMat;
use crate::shared::numeric::ScalarType;

fn assert_close(actual: ScalarType, expected: ScalarType, msg: &str) {
    // Tolerance scales with the compile-time precision (~1e4 ulp of the
    // reference magnitude): f64 stays tight at ~2e-12, while f32
    // (eps ≈ 1.2e-7) still passes after sqrt/div round-off.
    let tol = 1e4 * ScalarType::EPSILON * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tol,
        "{}: expected {}, got {}",
        msg,
        expected,
        actual
    );
}

#[test]
fn hpsolb_extracts_minimum_to_last_slot() {
    // Fortran t = (3, 1, 2), iorder = (1, 2, 3): after heap-build + extract
    // the least breakpoint (1, variable 2) sits in slot n.
    let mut t = [3.0 as ScalarType, 1.0, 2.0];
    let mut iorder = [1usize, 2, 3];
    hpsolb(3, &mut t, &mut iorder, 0);
    assert_eq!(iorder[2], 2, "least breakpoint belongs to variable 2");
    assert_close(t[2], 1.0, "least breakpoint value at t(n)");
    // The remaining slots hold {2, 3} in heap order.
    let mut rest = [t[0], t[1]];
    rest.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_close(rest[0], 2.0, "remaining heap smaller element");
    assert_close(rest[1], 3.0, "remaining heap larger element");
}

#[test]
fn middle_mat_vec_prod_with_zero_columns_is_noop() {
    let mat = LimitedMemBfgsMat::new(2, 3); // col = 0
    let v: [ScalarType; 2] = [1.0, 2.0];
    let mut p: [ScalarType; 2] = [9.0, 9.0];
    assert!(middle_mat_vec_prod(&mat, &v, &mut p));
    assert_eq!(p[0], 9.0, "p untouched when col == 0");
    assert_eq!(p[1], 9.0, "p untouched when col == 0");
}

#[test]
fn middle_mat_vec_prod_one_column_matches_inverse_middle_matrix() {
    // col = 1: the middle matrix is M = diag(-D, theta*S'S) with
    // D = s'y = 2 and theta*s's = 3 * 2 = 6, so
    //   M^{-1}·v = (-v1/2, v2/6) = (-2, 1) for v = (4, 6).
    let mut mat = LimitedMemBfgsMat::new(1, 2);
    // Pair 1: s = (1, 1), y = (2, 0) → s'y = 2.
    mat.update(&[1.0, 1.0], &[2.0, 0.0]);
    mat.theta = 3.0;
    mat.sy = [2.0].to_vec();
    mat.wt = [(6.0 as ScalarType).sqrt()].to_vec(); // chol(theta*s's = 6)

    let v: [ScalarType; 2] = [4.0, 6.0];
    let mut p: [ScalarType; 2] = [0.0, 0.0];
    assert!(middle_mat_vec_prod(&mat, &v, &mut p));
    assert_close(p[0], -2.0, "p1 = -v1/D");
    assert_close(p[1], 1.0, "p2 = v2/(theta*S'S)");
}

#[test]
fn cauchy_unconstrained_identity_model() {
    // B = theta*I (col = 0), no bounds: along d = -g the quadratic model has
    // its minimiser at t* = 1/theta, so the GCP is x - g/theta.
    let n = 2;
    let m = 4;
    let mut mat = LimitedMemBfgsMat::new(m, n);
    mat.theta = 3.0;

    let x = [1.0 as ScalarType, 2.0];
    let l = [0.0 as ScalarType; 2];
    let u = [0.0 as ScalarType; 2];
    let nbd = [0u8; 2];
    let g = [1.0 as ScalarType, -2.0];
    let mut iwhere = [-1i32; 2];
    let mut xcp = [0.0 as ScalarType; 2];
    let mut ws = CauchyWorkspace::new(n, m);

    let nseg = cauchy(
        &x,
        &l,
        &u,
        &nbd,
        &g,
        &mat,
        &mut iwhere,
        &mut xcp,
        &mut ws,
        2.0,
        1e-15,
    )
    .unwrap();

    assert_close(xcp[0], 1.0 - 1.0 / 3.0, "xcp0 = x0 - g0/theta");
    assert_close(xcp[1], 2.0 + 2.0 / 3.0, "xcp1 = x1 - g1/theta");
    assert_eq!(nseg, 1);
    assert_eq!(iwhere, [-1, -1], "free variables stay free");
}

#[test]
fn cauchy_fixes_variable_at_first_breakpoint() {
    // x = (0, 0), l = (-0.5, -2), u = (1, 1), g = (1, 1), theta = 1, col = 0.
    // d = (-1, -1) with breakpoints t = (0.5, 2); the initial minimiser is
    // dtm = 1, so variable 1 is pinned at its lower bound at t = 0.5 and
    // the minimiser of the remaining segment lies dtm = 0.5 past it:
    //   xcp = (-0.5, -1), iwhere = (1, 0), nseg = 2.
    let n = 2;
    let m = 4;
    let mut mat = LimitedMemBfgsMat::new(m, n);
    mat.theta = 1.0;

    let x = [0.0 as ScalarType; 2];
    let l = [-0.5 as ScalarType, -2.0];
    let u = [1.0 as ScalarType; 2];
    let nbd = [2u8; 2];
    let g = [1.0 as ScalarType; 2];
    let mut iwhere = [0i32; 2];
    let mut xcp = [0.0 as ScalarType; 2];
    let mut ws = CauchyWorkspace::new(n, m);

    let nseg = cauchy(
        &x,
        &l,
        &u,
        &nbd,
        &g,
        &mat,
        &mut iwhere,
        &mut xcp,
        &mut ws,
        0.5,
        1e-15,
    )
    .unwrap();

    assert_close(xcp[0], -0.5, "x0 pinned at its lower bound");
    assert_close(xcp[1], -1.0, "x1 moved by tsum = 1 along d");
    assert_eq!(iwhere[0], 1, "variable 1 fixed at lower bound");
    assert_eq!(iwhere[1], 0, "variable 2 still free");
    assert_eq!(nseg, 2);
}

#[test]
fn cauchy_all_variables_pinned_at_bounds() {
    // theta = 0.1 makes the initial minimiser dtm = 10, beyond every
    // breakpoint (t = 3, 0.5, 1 for lower bounds at distance 3, 0.5, 1):
    // all three variables get pinned at their lower bounds and the routine
    // exits through the all-fixed path (Fortran lines 1620-1625), calling
    // hpsolb on the way. xcp = l, iwhere = (1, 1, 1), nseg = 3.
    let n = 3;
    let m = 4;
    let mut mat = LimitedMemBfgsMat::new(m, n);
    mat.theta = 0.1;

    let x = [0.0 as ScalarType; 3];
    let l = [-3.0 as ScalarType, -0.5, -1.0];
    let u = [3.0 as ScalarType; 3];
    let nbd = [2u8; 3];
    let g = [1.0 as ScalarType; 3];
    let mut iwhere = [0i32; 3];
    let mut xcp = [0.0 as ScalarType; 3];
    let mut ws = CauchyWorkspace::new(n, m);

    let nseg = cauchy(
        &x,
        &l,
        &u,
        &nbd,
        &g,
        &mat,
        &mut iwhere,
        &mut xcp,
        &mut ws,
        1.0,
        1e-15,
    )
    .unwrap();

    for i in 0..3 {
        assert_close(xcp[i], l[i], "xcp pinned at lower bound");
        assert_eq!(iwhere[i], 1, "variable pinned at lower bound");
    }
    assert_eq!(nseg, 3);
}

#[test]
fn cauchy_with_memory_matches_quadratic_model() {
    // col = 1 with s = (1, 0), y = (2, 0), theta = 1, unbounded x = (0, 0),
    // g = (1, 1):
    //   W'd = (Y'd, theta*S'd) = (-2, -1),  M = diag(-D, theta*S'S)
    // with D = s'y = 2 and theta*S'S = 1, so middle_mat_vec_prod gives M^{-1}·(W'd) = (1, -1)
    // and   f2 = theta*g'g + d'W·M^{-1}·W'd = 2 + 1 = 3.
    // The exact minimiser along d = -g is t* = g'g / f2 = 2/3, hence
    //   xcp = t*·d = (-2/3, -2/3)   and   c = W'(xcp - x) = t*·(W'd).
    let n = 2;
    let m = 1;
    let mut mat = LimitedMemBfgsMat::new(m, n);
    // Pair 1: s = (1, 0), y = (2, 0) → s'y = 2.
    mat.update(&[1.0, 0.0], &[2.0, 0.0]);
    mat.theta = 1.0;
    mat.sy = [2.0].to_vec(); // s'y
    mat.wt = [1.0].to_vec(); // chol(theta*s's = 1)

    let x = [0.0 as ScalarType; 2];
    let l = [0.0 as ScalarType; 2];
    let u = [0.0 as ScalarType; 2];
    let nbd = [0u8; 2];
    let g = [1.0 as ScalarType, 1.0];
    let mut iwhere = [-1i32; 2];
    let mut xcp = [0.0 as ScalarType; 2];
    let mut ws = CauchyWorkspace::new(n, m);

    let nseg = cauchy(
        &x,
        &l,
        &u,
        &nbd,
        &g,
        &mat,
        &mut iwhere,
        &mut xcp,
        &mut ws,
        1.0,
        1e-15,
    )
    .unwrap();

    assert_close(xcp[0], -2.0 / 3.0, "xcp0 = t*·d0");
    assert_close(xcp[1], -2.0 / 3.0, "xcp1 = t*·d1");
    assert_close(ws.c[0], -4.0 / 3.0, "c0 = t*·(W'd)_1");
    assert_close(ws.c[1], -2.0 / 3.0, "c1 = t*·(W'd)_2");
    assert_eq!(nseg, 1);
}

#[test]
fn cauchy_reports_singular_system() {
    // A zero diagonal in wt makes the triangular solve inside middle_mat_vec_prod fail;
    // Fortran flags this via info != 0 and mainlb refreshes the memory.
    let n = 2;
    let m = 1;
    let mut mat = LimitedMemBfgsMat::new(m, n);
    // Pair 1: s = (1, 0), y = (2, 0) → s'y = 2.
    mat.update(&[1.0, 0.0], &[2.0, 0.0]);
    mat.sy = [2.0].to_vec();
    mat.wt = [0.0].to_vec(); // singular

    let x = [0.0 as ScalarType; 2];
    let l = [0.0 as ScalarType; 2];
    let u = [0.0 as ScalarType; 2];
    let nbd = [0u8; 2];
    let g = [1.0 as ScalarType, 1.0];
    let mut iwhere = [-1i32; 2];
    let mut xcp = [0.0 as ScalarType; 2];
    let mut ws = CauchyWorkspace::new(n, m);

    let err = cauchy(
        &x,
        &l,
        &u,
        &nbd,
        &g,
        &mat,
        &mut iwhere,
        &mut xcp,
        &mut ws,
        1.0,
        1e-15,
    );
    assert_eq!(err, Err(CauchyError::SingularSystem));
}
