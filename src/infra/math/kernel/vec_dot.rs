#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f32",
    not(target_os = "macos")
))]
use super::neon::float_impl;

#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f64",
    not(target_os = "macos")
))]
use super::neon::double_impl;

use crate::shared::numeric::ScalarType;

///  Scalar dot product
#[inline]
fn vec_dot_ansi(x: &[ScalarType], y: &[ScalarType]) -> ScalarType {
    let len: usize = x.len();
    let mut sum = 0.0 as ScalarType;
    for i in 0..len {
        sum = sum + (x[i] * y[i]);
    }

    return sum;
}

// ── Dispatch wrapper ──

/// Compute the dot product of two vectors.
pub(crate) fn vec_dot(x: &[ScalarType], y: &[ScalarType]) -> ScalarType {
    debug_assert_eq!(x.len(), y.len(), "vector length mismatch");

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f64",
        not(target_os = "macos")
    ))]
    unsafe {
        return double_impl::vec_dot_double_impl(x, y);
    }

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f32",
        not(target_os = "macos")
    ))]
    unsafe {
        return float_impl::vec_dot_float_impl(x, y);
    }

    #[cfg(any(
        not(all(target_arch = "aarch64", feature = "neon")),
        target_os = "macos"
    ))]
    return vec_dot_ansi(x, y);
}
