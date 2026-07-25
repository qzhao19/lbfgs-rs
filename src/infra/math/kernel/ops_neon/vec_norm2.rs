use crate::shared::numeric::ScalarType;

#[cfg(all(target_arch = "aarch64", feature = "neon", not(target_os = "macos")))]
use std::arch::aarch64::*;

#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f32",
    not(target_os = "macos")
))]
use crate::shared::arch::simd::{FTYPE_LANES, FTYPE_UNROLL};

#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f64",
    not(target_os = "macos")
))]
use crate::shared::arch::simd::{DTYPE_LANES, DTYPE_UNROLL};

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

#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f32",
    not(target_os = "macos")
))]
#[inline]
unsafe fn vec_norm2_float_impl(x: &[f32], squared: bool) -> f32 {
    let len: usize = x.len();
    let x_ptr = x.as_ptr();

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let step: usize = FTYPE_LANES * FTYPE_UNROLL;
    let chunks: usize = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        // Each load-process operation is independent.
        let x0 = vld1q_f32(x_ptr.add(i));
        acc0 = vfmaq_f32(acc0, x0, x0);
        i += FTYPE_LANES;

        let x1 = vld1q_f32(x_ptr.add(i));
        acc1 = vfmaq_f32(acc1, x1, x1);
        i += FTYPE_LANES;

        let x2 = vld1q_f32(x_ptr.add(i));
        acc2 = vfmaq_f32(acc2, x2, x2);
        i += FTYPE_LANES;

        let x3 = vld1q_f32(x_ptr.add(i));
        acc3 = vfmaq_f32(acc3, x3, x3);
        i += FTYPE_LANES;
    }

    // Handling remaining blocks that are less than 4-way
    // but still fill one vector width.
    while i + FTYPE_LANES <= len {
        let xv = vld1q_f32(x_ptr.add(i));
        acc0 = vfmaq_f32(acc0, xv, xv);
        i += FTYPE_LANES;
    }

    // Merge 4 registers
    let acc01 = vaddq_f32(acc0, acc1);
    let acc23 = vaddq_f32(acc2, acc3);
    let acc = vaddq_f32(acc01, acc23);

    // Horizontal sum via vaddvq_f32
    let mut sum = vaddvq_f32(acc);

    // Handle remaining elements that are less than one vector width
    while i < len {
        sum += *x_ptr.add(i) * *x_ptr.add(i);
        i += 1;
    }

    if squared {
        sum
    } else {
        sum.sqrt()
    }
}

#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f64",
    not(target_os = "macos")
))]
#[inline]
unsafe fn vec_norm2_double_impl(x: &[f64], squared: bool) -> f64 {
    let len = x.len();
    let x_ptr = x.as_ptr();

    let mut acc0 = vdupq_n_f64(0.0);
    let mut acc1 = vdupq_n_f64(0.0);

    let step = DTYPE_LANES * DTYPE_UNROLL;
    let chunks = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        let x0 = vld1q_f64(x_ptr.add(i));
        acc0 = vfmaq_f64(acc0, x0, x0);
        i += DTYPE_LANES;

        let x1 = vld1q_f64(x_ptr.add(i));
        acc1 = vfmaq_f64(acc1, x1, x1);
        i += DTYPE_LANES;
    }

    while i + DTYPE_LANES <= len {
        let xv = vld1q_f64(x_ptr.add(i));
        acc0 = vfmaq_f64(acc0, xv, xv);
        i += DTYPE_LANES;
    }

    let acc = vaddq_f64(acc0, acc1);
    let mut sum = vaddvq_f64(acc);

    while i < len {
        sum += *x_ptr.add(i) * *x_ptr.add(i);
        i += 1;
    }

    if squared {
        sum
    } else {
        sum.sqrt()
    }
}

// ── Dispatch wrappers ──

/// Compute the 2-norm of `x`.
///
/// - `squared == true`  → returns `Σ x[i]²`  (squared L2 norm)
/// - `squared == false` → returns `sqrt(Σ x[i]²)`  (the L2 norm itself)
pub fn vec_norm2(x: &[ScalarType], squared: bool) -> ScalarType {
    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f64",
        not(target_os = "macos")
    ))]
    unsafe {
        return vec_norm2_double_impl(x, squared);
    }

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f32",
        not(target_os = "macos")
    ))]
    unsafe {
        return vec_norm2_float_impl(x, squared);
    }

    #[cfg(any(
        not(all(target_arch = "aarch64", feature = "neon")),
        target_os = "macos"
    ))]
    return vec_norm2_ansi(x, squared);
}
