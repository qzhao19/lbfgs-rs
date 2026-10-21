use crate::infra::math::vec_ops::{vec_dot, vec_scaled_add_inplace};
use crate::shared::numeric::ScalarType;

/// Solve `U·x = b` in place, where `U` is the upper triangle of the
/// row-major `m × m`, "solve U*x = b, t upper triangular"
///
/// Returns `false` when a diagonal entry of `U` is exactly zero
pub(crate) fn solve_upper_triangular(
    u: &[ScalarType],
    m: usize,
    k: usize,
    b: &mut [ScalarType],
) -> bool {
    debug_assert!(k <= m);

    // 1. Check the diagonal for zeros
    for i in 0..k {
        if u[i * m + i] == 0.0 {
            return false;
        }
    }

    // 2. Back substitution: x_i = (b_i − <row_i, x_{i+1..k}>) / u_{ii}.
    for i in (0..k).rev() {
        let row = &u[i * m + (i + 1)..i * m + k];
        let tail = &b[(i + 1)..k];
        let sum = b[i] - vec_dot(row, tail);
        b[i] = sum / u[i * m + i];
    }

    true
}

/// Solve `U'·x = b` in place, where `U` is the upper triangle of the
/// row-major `m × m`, "solve trans(U)*x = b, t upper triangular"
/// `U'` is lower triangular.
///
/// Returns `false` when a diagonal entry of `U` is exactly zero.
pub(crate) fn solve_upper_triangular_transpose(
    u: &[ScalarType],
    m: usize,
    k: usize,
    b: &mut [ScalarType],
) -> bool {
    debug_assert!(k <= m);

    // 1. Check the diagonal for zeros
    for i in 0..k {
        if u[i * m + i] == 0.0 {
            return false;
        }
    }

    // 2. Solve U'x = b

    for j in 0..k {
        let xj = b[j] / u[j * m + j];
        b[j] = xj;
        let row = &u[j * m + (j + 1)..j * m + k];
        vec_scaled_add_inplace(row, -xj, &mut b[(j + 1)..k]);
    }

    true
}
