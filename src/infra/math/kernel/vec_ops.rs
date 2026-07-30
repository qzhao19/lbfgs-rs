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

/// Compute out[i] = -x[i].
#[inline]
fn vec_ncpy_ansi(x: &[ScalarType], out: &mut [ScalarType]) {
    let len: usize = x.len();
    for i in 0..len {
        out[i] = -x[i];
    }
}

/// ||x|| = Σ x[i]², optionally square-rooted.
#[inline]
fn vec_norm2_ansi(x: &[ScalarType], squared: bool) -> ScalarType {
    let len: usize = x.len();
    let mut sum: ScalarType = 0.0 as ScalarType;
    for i in 0..len {
        sum = sum + x[i] * x[i];
    }
    if squared {
        sum
    } else {
        sum.sqrt()
    }
}

/// x[i] *= scalar
#[inline]
fn vec_scale_inplace_ansi(x: &mut [ScalarType], scalar: ScalarType) {
    let len: usize = x.len();
    for i in 0..len {
        x[i] *= scalar;
    }
}

/// out[i] = x[i] * scalar + y[i]
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

/// Scalar fused multiply-add: acc[i] += src[i] × scalar.
#[inline]
fn vec_scaled_add_inplace_ansi(src: &[ScalarType], scalar: ScalarType, acc: &mut [ScalarType]) {
    let len: usize = src.len();
    for i in 0..len {
        acc[i] = acc[i] + src[i] * scalar;
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

/// Compute out[i] = -x[i].
pub(crate) fn vec_ncpy(x: &[ScalarType], out: &mut [ScalarType]) {
    debug_assert_eq!(x.len(), out.len(), "vector length mismatch");

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f64",
        not(target_os = "macos")
    ))]
    unsafe {
        double_impl::vec_ncpy_double_impl(x, out);
    }

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f32",
        not(target_os = "macos")
    ))]
    unsafe {
        float_impl::vec_ncpy_float_impl(x, out);
    }

    #[cfg(any(
        not(all(target_arch = "aarch64", feature = "neon")),
        target_os = "macos"
    ))]
    vec_ncpy_ansi(x, out);
}

/// Compute the 2-norm of `x`.
///
/// - `squared == true`  → returns `Σ x[i]²`  (squared L2 norm)
/// - `squared == false` → returns `sqrt(Σ x[i]²)`  (the L2 norm itself)
pub(crate) fn vec_norm2(x: &[ScalarType], squared: bool) -> ScalarType {
    debug_assert!(!x.is_empty(), "x must be non-empty");

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f64",
        not(target_os = "macos")
    ))]
    unsafe {
        return double_impl::vec_norm2_double_impl(x, squared);
    }

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f32",
        not(target_os = "macos")
    ))]
    unsafe {
        return float_impl::vec_norm2_float_impl(x, squared);
    }

    #[cfg(any(
        not(all(target_arch = "aarch64", feature = "neon")),
        target_os = "macos"
    ))]
    return vec_norm2_ansi(x, squared);
}

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
