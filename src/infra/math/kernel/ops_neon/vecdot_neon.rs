use crate::shared::types::primitives::ScalarType;

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
use std::arch::aarch64::*;

#[cfg(all(target_arch = "aarch64", feature = "neon", feature = "f32"))]
use crate::shared::constants::{FTYPE_LANES, FTYPE_UNROLL};

#[cfg(all(target_arch = "aarch64", feature = "neon", feature = "f64"))]
use crate::shared::constants::{DTYPE_LANES, DTYPE_UNROLL};

///  Scalar dot product
#[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
#[inline]
fn vecdot_ansi(x: &[ScalarType], y: &[ScalarType]) -> ScalarType {
    debug_assert_eq!(x.len(), y.len(), "vector length mismatch");
    let len: usize = x.len();
    let mut sum = 0.0 as ScalarType;
    for i in 0..len {
        sum = sum + (x[i] * y[i]);
    }

    return sum;
}

/// NEON-accelerated dot product for f32.
#[cfg(all(target_arch = "aarch64", feature = "neon", feature = "f32"))]
#[inline]
unsafe fn vecdot_neon_float_impl(x: &[f32], y: &[f32]) -> f32 {
    let len: usize = x.len();

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let xptr = x.as_ptr();
    let yptr = y.as_ptr();

    let step: usize = FTYPE_LANES * FTYPE_UNROLL;
    let chunks: usize = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        let x0 = vld1q_f32(xptr.add(i));
        let y0 = vld1q_f32(yptr.add(i));
        acc0 = vfmaq_f32(acc0, x0, y0);
        i += FTYPE_LANES;

        let x1 = vld1q_f32(xptr.add(i));
        let y1 = vld1q_f32(yptr.add(i));
        acc1 = vfmaq_f32(acc1, x1, y1);
        i += FTYPE_LANES;

        let x2 = vld1q_f32(xptr.add(i));
        let y2 = vld1q_f32(yptr.add(i));
        acc2 = vfmaq_f32(acc2, x2, y2);
        i += FTYPE_LANES;

        let x3 = vld1q_f32(xptr.add(i));
        let y3 = vld1q_f32(yptr.add(i));
        acc3 = vfmaq_f32(acc3, x3, y3);
        i += FTYPE_LANES;
    }

    // Handling remaining blocks that are less than 4-way
    // but still fill one vector width.
    while i + FTYPE_LANES <= len {
        let xv = vld1q_f32(xptr.add(i));
        let yv = vld1q_f32(yptr.add(i));
        acc0 = vfmaq_f32(acc0, xv, yv);
        i += FTYPE_LANES;
    }

    // Merge 4 registers
    let acc01 = vaddq_f32(acc0, acc1);
    let acc23 = vaddq_f32(acc2, acc3);
    let acc = vaddq_f32(acc01, acc23);

    // Horizontal sum via vaddvq_f32
    let mut sum = vaddvq_f32(acc);

    // Handle remaining elements at the end
    // that are less than one vector width
    while i < len {
        sum += *xptr.add(i) * *yptr.add(i);
        i += 1;
    }

    return sum;
}

/// NEON-accelerated dot product for f64
#[cfg(all(target_arch = "aarch64", feature = "neon", feature = "f64"))]
#[inline]
unsafe fn vecdot_neon_double_impl(x: &[f64], y: &[f64]) -> f64 {
    let len = x.len();

    let mut acc0 = vdupq_n_f64(0.0);
    let mut acc1 = vdupq_n_f64(0.0);

    let xptr = x.as_ptr();
    let yptr = y.as_ptr();

    let step = DTYPE_LANES * DTYPE_UNROLL;
    let chunks = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        let x0 = vld1q_f64(xptr.add(i));
        let y0 = vld1q_f64(yptr.add(i));
        acc0 = vfmaq_f64(acc0, x0, y0);
        i += DTYPE_LANES;

        let x1 = vld1q_f64(xptr.add(i));
        let y1 = vld1q_f64(yptr.add(i));
        acc1 = vfmaq_f64(acc1, x1, y1);
        i += DTYPE_LANES;
    }

    while i + DTYPE_LANES <= len {
        let xv = vld1q_f64(xptr.add(i));
        let yv = vld1q_f64(yptr.add(i));
        acc0 = vfmaq_f64(acc0, xv, yv);
        i += DTYPE_LANES;
    }

    let acc = vaddq_f64(acc0, acc1);
    let mut sum = vaddvq_f64(acc);

    while i < len {
        sum += *xptr.add(i) * *yptr.add(i);
        i += 1;
    }

    return sum;
}

// ── Dispatch wrappers ──

/// Compute the dot product of two vectors.
/// Automatically selects NEON or scalar path at compile time.
pub fn vecdot(x: &[ScalarType], y: &[ScalarType]) -> ScalarType {
    // NEON f64 path
    #[cfg(all(target_arch = "aarch64", feature = "neon", feature = "f64"))]
    unsafe {
        return vecdot_neon_double_impl(x, y);
    }
    // NEON f32 path
    #[cfg(all(target_arch = "aarch64", feature = "neon", feature = "f32"))]
    unsafe {
        return vecdot_neon_float_impl(x, y);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
    return vecdot_ansi(x, y);
}
