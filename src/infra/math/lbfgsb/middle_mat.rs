use crate::infra::math::bfgs_mat::LimitedMemBfgsMat;
use crate::infra::math::linalg::triangular_solve::{
    solve_upper_triangular, solve_upper_triangular_transpose,
};
use crate::shared::numeric::ScalarType;

pub(crate) fn middle_mat_vec_prod(
    mat: &LimitedMemBfgsMat,
    v: &[ScalarType],
    p: &mut [ScalarType],
) -> bool {
    let col = mat.col();
    let m = mat.capacity;
    debug_assert!(col <= m);
    if col == 0 {
        return true;
    }
    let sy = &mat.sy;
    let wt = &mat.wt;

    // ── PART I: solve [ D^(1/2)      0 ] [ p1 ]   [ v1 ]
    //                [ -L*D^(-1/2)    J ] [ p2 ] = [ v2 ]
    // Fortran lines 1178-1194.

    // Solve J·p2 = v2 + L*D^{-1}·v1.  [p(col+1) = v(col+1)]
    p[col] = v[col];
    for i in 2..=col {
        // sum = Σ_{k=1}^{i-1} sy(i,k)·v(k)/sy(k,k)
        let mut sum: ScalarType = 0.0;
        for k in 1..i {
            sum += sy[(i - 1) + (k - 1) * m] * v[k - 1] / sy[(k - 1) + (k - 1) * m];
        }
        p[col + i - 1] = v[col + i - 1] + sum;
    }
    // Solve the triangular system.  [dtrsl job = 11]
    if !solve_upper_triangular_transpose(wt, m, col, &mut p[col..2 * col]) {
        return false;
    }

    // Solve D^(1/2)·p1 = v1.
    for i in 1..=col {
        p[i - 1] = v[i - 1] / sy[(i - 1) + (i - 1) * m].sqrt();
    }

    // ── PART II: solve [ -D^(1/2)   D^(-1/2)*L' ] [ p1 ]   [ p1 ]
    //                [  0         J'            ] [ p2 ] = [ p2 ]

    // Solve J'·p2 = p2.  [dtrsl job = 01]
    if !solve_upper_triangular(wt, m, col, &mut p[col..2 * col]) {
        return false;
    }

    // p1 = -D^{-1/2}·(p1 - D^{-1/2}·L'·p2) = -D^{-1/2}·p1 + D^{-1}·L'·p2.
    for i in 1..=col {
        p[i - 1] = -p[i - 1] / sy[(i - 1) + (i - 1) * m].sqrt();
    }
    for i in 1..=col {
        // sum = Σ_{k=i+1}^{col} sy(k,i)·p(col+k)/sy(i,i)
        let mut sum: ScalarType = 0.0;
        for k in (i + 1)..=col {
            sum += sy[(k - 1) + (i - 1) * m] * p[col + k - 1] / sy[(i - 1) + (i - 1) * m];
        }
        p[i - 1] += sum;
    }

    true
}
