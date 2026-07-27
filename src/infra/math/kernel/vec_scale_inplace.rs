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

#[inline]
fn vec_scale_inplace_ansi(x: &mut [ScalarType], scalar: ScalarType) {
    let len: usize = x.len();
    for i in 0..len {
        x[i] *= scalar;
    }
}

// ── Dispatch wrapper ──

/// Compute x[i] = scalar * x[i] in place.
pub(crate) fn vec_scale_inplace(x: &mut [ScalarType], scalar: ScalarType) {
    debug_assert!(!x.is_empty(), "x must be non-empty");

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f64",
        not(target_os = "macos")
    ))]
    unsafe {
        double_impl::vec_scale_inplace_double_impl(x, scalar);
    }

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f32",
        not(target_os = "macos")
    ))]
    unsafe {
        float_impl::vec_scale_inplace_float_impl(x, scalar);
    }

    #[cfg(any(
        not(all(target_arch = "aarch64", feature = "neon")),
        target_os = "macos"
    ))]
    vec_scale_inplace_ansi(x, scalar);
}
