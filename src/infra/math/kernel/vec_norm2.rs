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

/// ANSI scalar path: Σ x[i]², optionally square-rooted.
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

// ── Dispatch wrapper ──

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
