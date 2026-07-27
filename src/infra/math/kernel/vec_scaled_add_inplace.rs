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

/// Scalar fused multiply-add: acc[i] += src[i] × scalar.
#[inline]
fn vec_scaled_add_inplace_ansi(src: &[ScalarType], scalar: ScalarType, acc: &mut [ScalarType]) {
    let len: usize = src.len();
    for i in 0..len {
        acc[i] = acc[i] + src[i] * scalar;
    }
}

// ── Dispatch wrapper ──

/// Element-wise fused multiply-add: acc[i] += src[i] × scalar.
pub(crate) fn vec_scaled_add_inplace(
    src: &[ScalarType],
    scalar: ScalarType,
    acc: &mut [ScalarType],
) {
    debug_assert_eq!(src.len(), acc.len(), "vector length mismatch");

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f64",
        not(target_os = "macos")
    ))]
    unsafe {
        double_impl::vec_scaled_add_inplace_double_impl(src, scalar, acc);
    }

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f32",
        not(target_os = "macos")
    ))]
    unsafe {
        float_impl::vec_scaled_add_inplace_float_impl(src, scalar, acc);
    }

    #[cfg(any(
        not(all(target_arch = "aarch64", feature = "neon")),
        target_os = "macos"
    ))]
    vec_scaled_add_inplace_ansi(src, scalar, acc);
}
