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

/// Compute out[i] = x[i] - y[i].
#[inline]
fn vec_diff_ansi(x: &[ScalarType], y: &[ScalarType], out: &mut [ScalarType]) {
    let len: usize = x.len();
    for i in 0..len {
        out[i] = x[i] - y[i];
    }
}

#[cfg(all(
    target_arch = "aarch64",
    feature = "neon",
    feature = "f32",
    not(target_os = "macos")
))]
#[inline]
unsafe fn vec_diff_float_impl(x: &[f32], y: &[f32], out: &mut [f32]) {
    let len: usize = x.len();

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
        vst1q_f32(out_ptr.add(i), vsubq_f32(x0, y0));
        i += FTYPE_LANES;

        let x1 = vld1q_f32(x_ptr.add(i));
        let y1 = vld1q_f32(y_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vsubq_f32(x1, y1));
        i += FTYPE_LANES;

        let x2 = vld1q_f32(x_ptr.add(i));
        let y2 = vld1q_f32(y_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vsubq_f32(x2, y2));
        i += FTYPE_LANES;

        let x3 = vld1q_f32(x_ptr.add(i));
        let y3 = vld1q_f32(y_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vsubq_f32(x3, y3));
        i += FTYPE_LANES;
    }

    // Handling remaining blocks that are less than 4-way
    // but still fill one vector width.
    while i + FTYPE_LANES <= len {
        let xv = vld1q_f32(x_ptr.add(i));
        let yv = vld1q_f32(y_ptr.add(i));
        vst1q_f32(out_ptr.add(i), vsubq_f32(xv, yv));
        i += FTYPE_LANES;
    }

    // Handle remaining block that are less than one vector width
    while i < len {
        *out_ptr.add(i) = *x_ptr.add(i) - *y_ptr.add(i);
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
unsafe fn vec_diff_double_impl(x: &[f64], y: &[f64], out: &mut [f64]) {
    let len = x.len();

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();
    let out_ptr = out.as_mut_ptr();

    let step = DTYPE_LANES * DTYPE_UNROLL;
    let chunks = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        let x0 = vld1q_f64(x_ptr.add(i));
        let y0 = vld1q_f64(y_ptr.add(i));
        vst1q_f64(out_ptr.add(i), vsubq_f64(x0, y0));
        i += DTYPE_LANES;

        let x1 = vld1q_f64(x_ptr.add(i));
        let y1 = vld1q_f64(y_ptr.add(i));
        vst1q_f64(out_ptr.add(i), vsubq_f64(x1, y1));
        i += DTYPE_LANES;
    }

    while i + DTYPE_LANES <= len {
        let xv = vld1q_f64(x_ptr.add(i));
        let yv = vld1q_f64(y_ptr.add(i));
        vst1q_f64(out_ptr.add(i), vsubq_f64(xv, yv));
        i += DTYPE_LANES;
    }

    while i < len {
        *out_ptr.add(i) = *x_ptr.add(i) - *y_ptr.add(i);
        i += 1;
    }
}

// ── Dispatch wrappers ──

/// Compute out[i] = x[i] - y[i].
pub fn vec_diff(x: &[ScalarType], y: &[ScalarType], out: &mut [ScalarType]) {
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
        vec_diff_double_impl(x, y, out);
    }

    #[cfg(all(
        target_arch = "aarch64",
        feature = "neon",
        feature = "f32",
        not(target_os = "macos")
    ))]
    unsafe {
        vec_diff_float_impl(x, y, out);
    }

    #[cfg(any(
        not(all(target_arch = "aarch64", feature = "neon")),
        target_os = "macos"
    ))]
    vec_diff_ansi(x, y, out);
}
