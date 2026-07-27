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
fn vec_scale_ansi(x: &[ScalarType], scalar: ScalarType, out: &mut [ScalarType]) {
    let len: usize = x.len();
    for i in 0..len {
        out[i] = scalar * x[i];
    }
}

// ── Dispatch wrapper ──

/// Compute out[i] = x[i] * scalar.
pub(crate) fn vec_scale(x: &[ScalarType], scalar: ScalarType, out: &mut [ScalarType]) {
    debug_assert_eq!(x.len(), out.len(), "vector length mismatch");

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f64",
        not(target_os = "macos")
    ))]
    unsafe {
        double_impl::vec_scale_double_impl(x, scalar, out);
    }

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f32",
        not(target_os = "macos")
    ))]
    unsafe {
        float_impl::vec_scale_float_impl(x, scalar, out);
    }

    #[cfg(any(
        not(all(target_arch = "aarch64", feature = "neon")),
        target_os = "macos"
    ))]
    vec_scale_ansi(x, scalar, out);
}
