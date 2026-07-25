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

#[inline]
fn vec_scale_ansi(x: &[ScalarType], scalar: ScalarType, out: &mut [ScalarType]) {
    let len: usize = x.len();
    for i in 0..len {
        out[i] = scalar * x[i];
    }
}

#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f32",
    not(target_os = "macos")
))]
unsafe fn vec_scale_float_impl(x: &[f32], scalar: f32, out: &mut [f32]) {
    let len: usize = x.len();

    // Scalar broadcast to vector register
    let scalar_v = vdupq_n_f32(scalar);

    let x_ptr = x.as_ptr();
    let out_ptr = out.as_mut_ptr();

    let step: usize = FTYPE_LANES * FTYPE_UNROLL;
    let chunks: usize = len / step;

    let mut i: usize = 0usize;

    for _ in 0..chunks {
        let x0 = vld1q_f32(x_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vmulq_f32(x0, scalar_v));
        i += FTYPE_LANES;

        let x1 = vld1q_f32(x_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vmulq_f32(x1, scalar_v));
        i += FTYPE_LANES;

        let x2 = vld1q_f32(x_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vmulq_f32(x2, scalar_v));
        i += FTYPE_LANES;

        let x3 = vld1q_f32(x_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vmulq_f32(x3, scalar_v));
        i += FTYPE_LANES;
    }

    // Handle remaining bloack that are less than 4-way
    // but still fill one vector
    while i + FTYPE_LANES <= len {
        let xv = vld1q_f32(x_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vmulq_f32(xv, scalar_v));
        i += FTYPE_LANES;
    }

    // Handle remaining block that are less than one vector
    while i < len {
        *out_ptr.add(i) = *x_ptr.add(i) * scalar;
        i += 1;
    }
}

#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f64",
    not(target_os = "macos")
))]
#[inline]
unsafe fn vec_scale_double_impl(x: &[f64], scalar: f64, out: &mut [f64]) {
    let len: usize = x.len();
    let scalar_v = vdupq_n_f64(scalar);

    let x_ptr = x.as_ptr();
    let out_ptr = out.as_mut_ptr();

    let step = DTYPE_LANES * DTYPE_UNROLL;
    let chunks = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        let x0 = vld1q_f64(x_ptr.add(i));
        vst1q_f64(out_ptr.add(i), vmulq_f64(x0, scalar_v));
        i += DTYPE_LANES;

        let x1 = vld1q_f64(x_ptr.add(i));
        vst1q_f64(out_ptr.add(i), vmulq_f64(x1, scalar_v));
        i += DTYPE_LANES;
    }

    while i + DTYPE_LANES <= len {
        let xv = vld1q_f64(x_ptr.add(i));
        vst1q_f64(out_ptr.add(i), vmulq_f64(xv, scalar_v));
        i += DTYPE_LANES;
    }

    while i < len {
        *out_ptr.add(i) = *x_ptr.add(i) * scalar;
        i += 1;
    }
}

// ── Dispatch wrappers ──

/// Compute out[i] = x[i] * scalar.
pub fn vec_scale(x: &[ScalarType], scalar: ScalarType, out: &mut [ScalarType]) {
    debug_assert_eq!(x.len(), out.len(), "vector length mismatch");

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f64",
        not(target_os = "macos")
    ))]
    unsafe {
        vec_scale_double_impl(x, scalar, out);
    }

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f32",
        not(target_os = "macos")
    ))]
    unsafe {
        vec_scale_float_impl(x, scalar, out);
    }

    #[cfg(any(
        not(all(target_arch = "aarch64", feature = "neon")),
        target_os = "macos"
    ))]
    vec_scale_ansi(x, scalar, out);
}
