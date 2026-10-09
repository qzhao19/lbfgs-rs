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

    // 2. Solve Ux = b
    for i in (0..k).rev() {
        let mut sum = b[i];
        for j in (i + 1)..k {
            // $u_{ii}x_i = b_i - \sum_{j=i+1}^{k} u_{ij}x_j$
            sum -= u[i * m + j] * b[j];
        }
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
    for i in 0..k {
        let mut sum = b[i];
        for j in 0..i {
            // $(b_i - \sum_{j=0}^{i-1} u_{ji}\, x_j)$
            sum -= u[j * m + i] * b[j];
        }
        b[i] = sum / u[i * m + i];
    }
    true
}
