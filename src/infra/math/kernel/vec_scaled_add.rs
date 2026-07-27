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

/// Compute out[i] = x[i] * scalar + y[i]
#[inline]
fn vec_scaled_add_ansi(
    x: &[ScalarType],
    y: &[ScalarType],
    scalar: ScalarType,
    out: &mut [ScalarType],
) {
    let len: usize = x.len();
    for i in 0..len {
        out[i] = x[i] * scalar + y[i];
    }
}

// ── Dispatch wrapper ──

/// Compute out[i] = x[i] * scalar + y[i].
pub(crate) fn vec_scaled_add(
    x: &[ScalarType],
    y: &[ScalarType],
    scalar: ScalarType,
    out: &mut [ScalarType],
) {
    debug_assert_eq!(x.len(), y.len(), "x and y must have the same length");
    debug_assert_eq!(
        x.len(),
        out.len(),
        "output vector must have the same length as input"
    );

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f64",
        not(target_os = "macos")
    ))]
    unsafe {
        double_impl::vec_scaled_add_double_impl(x, y, scalar, out);
    }

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f32",
        not(target_os = "macos")
    ))]
    unsafe {
        float_impl::vec_scaled_add_float_impl(x, y, scalar, out);
    }

    #[cfg(any(
        not(all(target_arch = "aarch64", feature = "neon")),
        target_os = "macos"
    ))]
    vec_scaled_add_ansi(x, y, scalar, out);
}
