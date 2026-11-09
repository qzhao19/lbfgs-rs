use crate::shared::numeric::ScalarType;

/// Cholesky factorization `A = Rᵀ·R` with `R` upper triangular.
/// Factorize the leading `k × k` block of the row-major `m × m` array `a`
/// into `Rᵀ·R`, overwriting the upper triangle with `R` (LINPACK `dpofa`,
/// column algorithm). `a(i, j)` lives at `a[(i-1)*m + (j-1)]`.
///
/// Returns `false` when a pivot is not positive — `dpofa` sets `info = j`
/// for the first column whose `a(j,j) − Σ t² ` is `<= 0`; on such failure
/// this routine leaves the columns up to `j` partially overwritten, exactly
/// like LINPACK, and the caller treats the matrix as not positive definite
/// (`formk` reports `info = -1 / -2` and `mainlb` refreshes the memory).
pub(crate) fn cholesky_upper(a: &mut [ScalarType], m: usize, k: usize) -> bool {
    debug_assert!(k <= m);
    for j in 1..=k {
        let mut s: ScalarType = 0.0;
        for c in 1..j {
            // t = (a(c, j) − Σ_{i<c} R(i,c)·R(i,j)) / R(c,c)
            let mut t = a[(c - 1) * m + (j - 1)];
            for i in 1..c {
                t -= a[(i - 1) * m + (c - 1)] * a[(i - 1) * m + (j - 1)];
            }
            t /= a[(c - 1) * m + (c - 1)];
            a[(c - 1) * m + (j - 1)] = t;
            s += t * t;
        }
        s = a[(j - 1) * m + (j - 1)] - s;
        if s <= 0.0 {
            return false;
        }
        a[(j - 1) * m + (j - 1)] = s.sqrt();
    }
    true
}
