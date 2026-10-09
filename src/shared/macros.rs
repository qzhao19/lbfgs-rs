// ── SIMD dispatch macros ──

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
