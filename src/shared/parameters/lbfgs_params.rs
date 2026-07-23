use super::linesearch_params::LineSearchParam;
use crate::shared::enums::lbfgs_options::{LineSearchPolicy, LossType};
use crate::shared::types::primitives::ScalarType;

/// Top-level parameters for an L-BFGS optimisation run.
pub struct LbfgsParams {
    /// Loss function used by the optimiser.
    pub loss: LossType,

    /// Line-search strategy used inside each outer iteration.
    pub linesearch_policy: LineSearchPolicy,

    // ── Convergence criteria ──
    /// Function-value convergence threshold:
    /// |f' - f| / max(1, |f|) < delta
    pub delta: ScalarType,

    /// Gradient convergence threshold:
    /// ||g(x)||_inf / max(1, ||x||_inf) < epsilon
    pub epsilon: ScalarType,

    // ── Outer-loop control ──
    /// Maximum number of L-BFGS outer iterations.
    pub max_iters: usize,

    /// Number of correction pairs retained for the inverse-Hessian
    /// approximation (the "m" in L-BFGS).
    pub mem_size: usize,

    /// Number of past iterations used by the delta-based convergence test
    /// (the function-value drop rate window).
    pub past: usize,

    // ── Line-search sub-parameters ──
    /// Parameters forwarded to the chosen line-search strategy.
    pub linesearch: LineSearchParam,
}

impl LbfgsParams {
    pub fn default() -> Self {
        Self {
            // Default
            loss: LossType::LogLoss,
            linesearch_policy: LineSearchPolicy::Backtracking,

            // Convergence thresholds.
            // |f' - f| / max(1, |f|) < delta
            delta: 1e-5,
            // ||g(x)||_inf / max(1, ||x||_inf) < epsilon
            epsilon: 1e-5,

            // Outer loop.
            max_iters: 100,

            // m = 8: standard L-BFGS correction-pair count.
            mem_size: 8,

            // Set past >= 1 to enable |f' - f|/max(1,|f|) < delta.
            // only the gradient (epsilon) test is active by default.
            past: 3,

            linesearch: LineSearchParam::default(),
        }
    }
}
