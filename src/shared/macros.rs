// ── SIMD dispatch macros ─────────────────────────────────────────────────
//
// These macros eliminate the cfg-boilerplate that would otherwise repeat 7×
// per dispatch wrapper (NEON-f64, NEON-f32, SSE-f64, SSE-f32, AVX2-f64,
// AVX2-f32, ANSI fallback).
//
// `simd_dispatch_stmt!`  — emits `unsafe { ... }` statement blocks for void
//                          kernels (diff/ncpy/scale_inplace/scaled_add*).
// `simd_dispatch_ret!`  — emits `unsafe { return ... }` expression blocks
//                          for kernels returning `ScalarType` (dot/norm2).
//
// Backends (in order): NEON-f64 → NEON-f32 → SSE-f64 → SSE-f32 →
//                      AVX2-f64 → AVX2-f32 → ANSI.
// The ANSI branch fires only when no listed SIMD backend is enabled; its cfg
// predicate must enumerate every backend listed above, otherwise an enabled
// backend would co-fire with ANSI and produce duplicate `return` / unreachable
// code (compile error).
//
// `$call_dbl` / `$call_f32` are the backend impl fn names (resolved via the
// `double_impl::` / `float_impl::` module aliasing introduced at the top of
// this file); the macro emits e.g. `double_impl::vec_dot_double_impl(args)`.
//
// NOTE: macro intentionally does NOT emit debug_asserts — those differ per
// kernel and stay in the wrapper body for readability (§3: don't hide
// per-function intent behind macro indirection).

macro_rules! simd_dispatch_stmt {
    ($call_dbl:ident, $call_f32:ident, $ansi:expr $(, $arg:expr)*) => {
        #[cfg(all(
            target_arch = "aarch64",
            feature = "neon",
            feature = "f64",
            not(target_os = "macos")
        ))]
        unsafe {
            double_impl::$call_dbl($($arg),*);
        }

        #[cfg(all(
            target_arch = "aarch64",
            feature = "neon",
            feature = "f32",
            not(target_os = "macos")
        ))]
        unsafe {
            float_impl::$call_f32($($arg),*);
        }

        #[cfg(all(target_arch = "x86_64", feature = "sse", feature = "f64"))]
        unsafe {
            double_impl::$call_dbl($($arg),*);
        }

        #[cfg(all(target_arch = "x86_64", feature = "sse", feature = "f32"))]
        unsafe {
            float_impl::$call_f32($($arg),*);
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "f64"))]
        unsafe {
            double_impl::$call_dbl($($arg),*);
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "f32"))]
        unsafe {
            float_impl::$call_f32($($arg),*);
        }

        #[cfg(not(any(
            all(
                target_arch = "aarch64",
                feature = "neon",
                not(target_os = "macos")
            ),
            all(target_arch = "x86_64", feature = "sse"),
            all(target_arch = "x86_64", feature = "avx2")
        )))]
        $ansi($($arg),*);
    };
}

macro_rules! simd_dispatch_ret {
    ($call_dbl:ident, $call_f32:ident, $ansi:expr $(, $arg:expr)*) => {
        #[cfg(all(
            target_arch = "aarch64",
            feature = "neon",
            feature = "f64",
            not(target_os = "macos")
        ))]
        unsafe {
            return double_impl::$call_dbl($($arg),*);
        }

        #[cfg(all(
            target_arch = "aarch64",
            feature = "neon",
            feature = "f32",
            not(target_os = "macos")
        ))]
        unsafe {
            return float_impl::$call_f32($($arg),*);
        }

        #[cfg(all(target_arch = "x86_64", feature = "sse", feature = "f64"))]
        unsafe {
            return double_impl::$call_dbl($($arg),*);
        }

        #[cfg(all(target_arch = "x86_64", feature = "sse", feature = "f32"))]
        unsafe {
            return float_impl::$call_f32($($arg),*);
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "f64"))]
        unsafe {
            return double_impl::$call_dbl($($arg),*);
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx2", feature = "f32"))]
        unsafe {
            return float_impl::$call_f32($($arg),*);
        }

        #[cfg(not(any(
            all(
                target_arch = "aarch64",
                feature = "neon",
                not(target_os = "macos")
            ),
            all(target_arch = "x86_64", feature = "sse"),
            all(target_arch = "x86_64", feature = "avx2")
        )))]
        return $ansi($($arg),*);
    };
}

// ── Macro re-exports ──

pub(crate) use simd_dispatch_ret;
pub(crate) use simd_dispatch_stmt;
