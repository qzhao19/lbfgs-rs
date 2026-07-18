use crate::shared::types::primitives::ScalarType;

#[cfg(all(target_arch = "aarch64", feature = "neon", not(target_os = "macos")))]
use std::arch::aarch64::*;

#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f32",
    not(target_os = "macos")
))]
use crate::shared::constants::simd_params::{FTYPE_LANES, FTYPE_UNROLL};

#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f64",
    not(target_os = "macos")
))]
use crate::shared::constants::simd_params::{DTYPE_LANES, DTYPE_UNROLL};

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

#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f32",
    not(target_os = "macos")
))]
#[inline]
unsafe fn vec_scaled_add_float_impl(x: &[f32], y: &[f32], scalar: f32, out: &mut [f32]) {
    let len: usize = x.len();

    // Scalar broadcast to vector register
    let scalar_v = vdupq_n_f32(scalar);

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();
    let out_ptr = out.as_mut_ptr();

    let step: usize = FTYPE_LANES * FTYPE_UNROLL;
    let chunks: usize = len / step;

    let mut i: usize = 0usize;

    for _ in 0..chunks {
        // Each read-process-write operation is independent,
        // allowing the CPU's load/memory units
        // FMA pipeline to be scheduled in parallel.
        let x0 = vld1q_f32(x_ptr.add(i));
        let y0 = vld1q_f32(y_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vfmaq_f32(y0, x0, scalar_v));
        i += FTYPE_LANES;

        let x1 = vld1q_f32(x_ptr.add(i));
        let y1 = vld1q_f32(y_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vfmaq_f32(y1, x1, scalar_v));
        i += FTYPE_LANES;

        let x2 = vld1q_f32(x_ptr.add(i));
        let y2 = vld1q_f32(y_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vfmaq_f32(y2, x2, scalar_v));
        i += FTYPE_LANES;

        let x3 = vld1q_f32(x_ptr.add(i));
        let y3 = vld1q_f32(y_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vfmaq_f32(y3, x3, scalar_v));
        i += FTYPE_LANES;
    }

    // Handling remaining blocks that are less than 4-way
    // but still fill one vector width.
    while i + FTYPE_LANES <= len {
        let xv = vld1q_f32(x_ptr.add(i));
        let yv = vld1q_f32(y_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vfmaq_f32(yv, xv, scalar_v));
        i += FTYPE_LANES;
    }

    while i < len {
        *out_ptr.add(i) = *x_ptr.add(i) * scalar + *y_ptr.add(i);
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
unsafe fn vec_scaled_add_double_impl(x: &[f64], y: &[f64], scalar: f64, out: &mut [f64]) {
    let len = x.len();

    let scalar_v = vdupq_n_f64(scalar);

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();
    let out_ptr = out.as_mut_ptr();

    let step = DTYPE_LANES * DTYPE_UNROLL;
    let chunks = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        let x0 = vld1q_f64(x_ptr.add(i));
        let y0 = vld1q_f64(y_ptr.add(i));
        vst1q_f64(out_ptr.add(i), vfmaq_f64(y0, x0, scalar_v));
        i += DTYPE_LANES;

        let x1 = vld1q_f64(x_ptr.add(i));
        let y1 = vld1q_f64(y_ptr.add(i));
        vst1q_f64(out_ptr.add(i), vfmaq_f64(y1, x1, scalar_v));
        i += DTYPE_LANES;
    }

    while i + DTYPE_LANES <= len {
        let xv = vld1q_f64(x_ptr.add(i));
        let yv = vld1q_f64(y_ptr.add(i));
        vst1q_f64(out_ptr.add(i), vfmaq_f64(yv, xv, scalar_v));
        i += DTYPE_LANES;
    }

    while i < len {
        *out_ptr.add(i) = *x_ptr.add(i) * scalar + *y_ptr.add(i);
        i += 1;
    }
}

// ── Dispatch wrappers ──

/// Compute out[i] = x[i] * scalar + y[i].
pub fn vec_scaled_add(
    x: &[ScalarType],
    y: &[ScalarType],
    scalar: ScalarType,
    out: &mut [ScalarType],
) {
    assert_eq!(x.len(), y.len(), "x and y must have the same length");
    assert_eq!(
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
        vec_scaled_add_double_impl(x, y, scalar, out);
    }

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f32",
        not(target_os = "macos")
    ))]
    unsafe {
        vec_scaled_add_float_impl(x, y, scalar, out);
    }

    #[cfg(any(
        not(all(target_arch = "aarch64", feature = "neon")),
        target_os = "macos"
    ))]
    vec_scaled_add_ansi(x, y, scalar, out);
}
