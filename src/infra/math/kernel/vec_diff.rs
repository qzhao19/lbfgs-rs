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

/// Compute out[i] = x[i] - y[i].
#[inline]
fn vec_diff_ansi(x: &[ScalarType], y: &[ScalarType], out: &mut [ScalarType]) {
    let len: usize = x.len();
    for i in 0..len {
        out[i] = x[i] - y[i];
    }
}

// ── Dispatch wrappers ──

/// Compute out[i] = x[i] - y[i].
pub(crate) fn vec_diff(x: &[ScalarType], y: &[ScalarType], out: &mut [ScalarType]) {
    debug_assert_eq!(x.len(), y.len(), "x and y must have the same length");
    debug_assert_eq!(
        x.len(),
        out.len(),
        "out must have the same length as x and y"
    );

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f64",
        not(target_os = "macos")
    ))]
    unsafe {
        double_impl::vec_diff_double_impl(x, y, out);
    }

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f32",
        not(target_os = "macos")
    ))]
    unsafe {
        float_impl::vec_diff_float_impl(x, y, out);
    }

    #[cfg(any(
        not(all(target_arch = "aarch64", feature = "neon")),
        target_os = "macos"
    ))]
    vec_diff_ansi(x, y, out);
}
