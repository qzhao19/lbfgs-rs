#[cfg(all(target_arch = "x86_64", feature = "sse", feature = "f64"))]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "x86_64", feature = "sse", feature = "f64"))]
use crate::shared::arch::{DTYPE_LANES, DTYPE_UNROLL};

#[cfg(all(target_arch = "x86_64", feature = "sse", feature = "f64"))]
#[inline]
pub(crate) unsafe fn vec_diff_double_impl(x: &[f64], y: &[f64], out: &mut [f64]) {
    let len = x.len();

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();
    let out_ptr = out.as_mut_ptr();

    let step = DTYPE_LANES * DTYPE_UNROLL;
    let chunks = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        let x0 = _mm_loadu_pd(x_ptr.add(i));
        let y0 = _mm_loadu_pd(y_ptr.add(i));
        _mm_storeu_pd(out_ptr.add(i), _mm_sub_pd(x0, y0));
        i += DTYPE_LANES;

        let x1 = _mm_loadu_pd(x_ptr.add(i));
        let y1 = _mm_loadu_pd(y_ptr.add(i));
        _mm_storeu_pd(out_ptr.add(i), _mm_sub_pd(x1, y1));
        i += DTYPE_LANES;
    }

    while i + DTYPE_LANES <= len {
        let xv = _mm_loadu_pd(x_ptr.add(i));
        let yv = _mm_loadu_pd(y_ptr.add(i));
        _mm_storeu_pd(out_ptr.add(i), _mm_sub_pd(xv, yv));
        i += DTYPE_LANES;
    }

    while i < len {
        *out_ptr.add(i) = *x_ptr.add(i) - *y_ptr.add(i);
        i += 1;
    }
}

#[cfg(all(target_arch = "x86_64", feature = "sse", feature = "f64"))]
#[inline]
pub(crate) unsafe fn vec_dot_double_impl(x: &[f64], y: &[f64]) -> f64 {
    let len = x.len();

    let mut acc0 = _mm_setzero_pd();
    let mut acc1 = _mm_setzero_pd();

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();

    let step = DTYPE_LANES * DTYPE_UNROLL;
    let chunks = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        let x0 = _mm_loadu_pd(x_ptr.add(i));
        let y0 = _mm_loadu_pd(y_ptr.add(i));
        acc0 = _mm_add_pd(acc0, _mm_mul_pd(x0, y0));
        i += DTYPE_LANES;

        let x1 = _mm_loadu_pd(x_ptr.add(i));
        let y1 = _mm_loadu_pd(y_ptr.add(i));
        acc1 = _mm_add_pd(acc1, _mm_mul_pd(x1, y1));
        i += DTYPE_LANES;
    }

    while i + DTYPE_LANES <= len {
        let xv = _mm_loadu_pd(x_ptr.add(i));
        let yv = _mm_loadu_pd(y_ptr.add(i));
        acc0 = _mm_add_pd(acc0, _mm_mul_pd(xv, yv));
        i += DTYPE_LANES;
    }

    let acc = _mm_add_pd(acc0, acc1);
    // SSE2 has no fused horizontal add for doubles; extract low/high lanes.
    let hi = _mm_unpackhi_pd(acc, acc);
    let mut sum = _mm_cvtsd_f64(acc) + _mm_cvtsd_f64(hi);

    while i < len {
        sum += *x_ptr.add(i) * *y_ptr.add(i);
        i += 1;
    }

    return sum;
}

#[cfg(all(target_arch = "x86_64", feature = "sse", feature = "f64"))]
#[inline]
pub(crate) unsafe fn vec_ncpy_double_impl(x: &[f64], out: &mut [f64]) {
    let len = x.len();

    let x_ptr = x.as_ptr();
    let out_ptr = out.as_mut_ptr();

    // Sign-bit mask: XOR-ing flips the sign, equivalent to vnegq_f64.
    let sign_mask = _mm_set1_pd(-0.0f64);

    let step = DTYPE_LANES * DTYPE_UNROLL;
    let chunks = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        let x0 = _mm_loadu_pd(x_ptr.add(i));
        _mm_storeu_pd(out_ptr.add(i), _mm_xor_pd(x0, sign_mask));
        i += DTYPE_LANES;

        let x1 = _mm_loadu_pd(x_ptr.add(i));
        _mm_storeu_pd(out_ptr.add(i), _mm_xor_pd(x1, sign_mask));
        i += DTYPE_LANES;
    }

    while i + DTYPE_LANES <= len {
        let xv = _mm_loadu_pd(x_ptr.add(i));
        _mm_storeu_pd(out_ptr.add(i), _mm_xor_pd(xv, sign_mask));
        i += DTYPE_LANES;
    }

    while i < len {
        *out_ptr.add(i) = -*x_ptr.add(i);
        i += 1;
    }
}

#[cfg(all(target_arch = "x86_64", feature = "sse", feature = "f64"))]
#[inline]
pub(crate) unsafe fn vec_norm2_double_impl(x: &[f64], squared: bool) -> f64 {
    let len = x.len();
    let x_ptr = x.as_ptr();

    let mut acc0 = _mm_setzero_pd();
    let mut acc1 = _mm_setzero_pd();

    let step = DTYPE_LANES * DTYPE_UNROLL;
    let chunks = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        let x0 = _mm_loadu_pd(x_ptr.add(i));
        acc0 = _mm_add_pd(acc0, _mm_mul_pd(x0, x0));
        i += DTYPE_LANES;

        let x1 = _mm_loadu_pd(x_ptr.add(i));
        acc1 = _mm_add_pd(acc1, _mm_mul_pd(x1, x1));
        i += DTYPE_LANES;
    }

    while i + DTYPE_LANES <= len {
        let xv = _mm_loadu_pd(x_ptr.add(i));
        acc0 = _mm_add_pd(acc0, _mm_mul_pd(xv, xv));
        i += DTYPE_LANES;
    }

    let acc = _mm_add_pd(acc0, acc1);
    let hi = _mm_unpackhi_pd(acc, acc);
    let mut sum = _mm_cvtsd_f64(acc) + _mm_cvtsd_f64(hi);

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

#[cfg(all(target_arch = "x86_64", feature = "sse", feature = "f64"))]
#[inline]
pub(crate) unsafe fn vec_scale_inplace_double_impl(x: &mut [f64], scalar: f64) {
    let len = x.len();

    let scalar_v = _mm_set1_pd(scalar);

    let x_ptr = x.as_mut_ptr();

    let step = DTYPE_LANES * DTYPE_UNROLL;
    let chunks = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        let x0 = _mm_loadu_pd(x_ptr.add(i));
        _mm_storeu_pd(x_ptr.add(i), _mm_mul_pd(x0, scalar_v));
        i += DTYPE_LANES;

        let x1 = _mm_loadu_pd(x_ptr.add(i));
        _mm_storeu_pd(x_ptr.add(i), _mm_mul_pd(x1, scalar_v));
        i += DTYPE_LANES;
    }

    while i + DTYPE_LANES <= len {
        let xv = _mm_loadu_pd(x_ptr.add(i));
        _mm_storeu_pd(x_ptr.add(i), _mm_mul_pd(xv, scalar_v));
        i += DTYPE_LANES;
    }

    while i < len {
        *x_ptr.add(i) *= scalar;
        i += 1;
    }
}

#[cfg(all(target_arch = "x86_64", feature = "sse", feature = "f64"))]
#[inline]
pub(crate) unsafe fn vec_scaled_add_double_impl(
    x: &[f64],
    y: &[f64],
    scalar: f64,
    out: &mut [f64],
) {
    let len = x.len();

    let scalar_v = _mm_set1_pd(scalar);

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();
    let out_ptr = out.as_mut_ptr();

    let step = DTYPE_LANES * DTYPE_UNROLL;
    let chunks = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        let x0 = _mm_loadu_pd(x_ptr.add(i));
        let y0 = _mm_loadu_pd(y_ptr.add(i));
        _mm_storeu_pd(out_ptr.add(i), _mm_add_pd(y0, _mm_mul_pd(x0, scalar_v)));
        i += DTYPE_LANES;

        let x1 = _mm_loadu_pd(x_ptr.add(i));
        let y1 = _mm_loadu_pd(y_ptr.add(i));
        _mm_storeu_pd(out_ptr.add(i), _mm_add_pd(y1, _mm_mul_pd(x1, scalar_v)));
        i += DTYPE_LANES;
    }

    while i + DTYPE_LANES <= len {
        let xv = _mm_loadu_pd(x_ptr.add(i));
        let yv = _mm_loadu_pd(y_ptr.add(i));
        _mm_storeu_pd(out_ptr.add(i), _mm_add_pd(yv, _mm_mul_pd(xv, scalar_v)));
        i += DTYPE_LANES;
    }

    while i < len {
        *out_ptr.add(i) = *x_ptr.add(i) * scalar + *y_ptr.add(i);
        i += 1;
    }
}

#[cfg(all(target_arch = "x86_64", feature = "sse", feature = "f64"))]
#[inline]
pub(crate) unsafe fn vec_scaled_add_inplace_double_impl(src: &[f64], scalar: f64, acc: &mut [f64]) {
    let len = src.len();

    let scalar_v = _mm_set1_pd(scalar);

    let acc_ptr = acc.as_mut_ptr();
    let src_ptr = src.as_ptr();

    let step = DTYPE_LANES * DTYPE_UNROLL;
    let chunks = len / step;

    let mut i: usize = 0usize;
    for _ in 0..chunks {
        let a0 = _mm_loadu_pd(acc_ptr.add(i));
        let s0 = _mm_loadu_pd(src_ptr.add(i));
        _mm_storeu_pd(acc_ptr.add(i), _mm_add_pd(a0, _mm_mul_pd(s0, scalar_v)));
        i += DTYPE_LANES;

        let a1 = _mm_loadu_pd(acc_ptr.add(i));
        let s1 = _mm_loadu_pd(src_ptr.add(i));
        _mm_storeu_pd(acc_ptr.add(i), _mm_add_pd(a1, _mm_mul_pd(s1, scalar_v)));
        i += DTYPE_LANES;
    }

    while i + DTYPE_LANES <= len {
        let av = _mm_loadu_pd(acc_ptr.add(i));
        let sv = _mm_loadu_pd(src_ptr.add(i));
        _mm_storeu_pd(acc_ptr.add(i), _mm_add_pd(av, _mm_mul_pd(sv, scalar_v)));
        i += DTYPE_LANES;
    }

    while i < len {
        *acc_ptr.add(i) += *src_ptr.add(i) * scalar;
        i += 1;
    }
}
