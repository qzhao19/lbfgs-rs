use super::numeric::ScalarType;

/// Acceptance condition for the line search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LineSearchCondition {
    Armijo,
    Wolfe,
    StrongWolfe,
}

/// Parameters controlling line-search behavior in L-BFGS.
#[derive(Clone, Copy)]
pub(crate) struct LineSearchParam {
    /// Step reduction factor (decrease until condition met).
    pub dec_factor: ScalarType,

    /// Step increase factor (enlarge when step too small).
    pub inc_factor: ScalarType,

    /// Tolerance for the line-search acceptance condition.
    pub ftol: ScalarType,

    /// Wolfe curvature coefficient (only used when condition == Wolfe).
    pub wolfe: ScalarType,

    /// Maximum allowed step size.
    pub max_stepsize: ScalarType,

    /// Minimum allowed step size.
    pub min_stepsize: ScalarType,

    /// Maximum line-search iterations.
    pub max_linesearch_iters: usize,

    /// Maximum line-search trials per iteration.
    pub max_searches: usize,

    /// Acceptance condition: Armijo or Wolfe.
    pub condition: LineSearchCondition,
}

impl LineSearchParam {
    pub fn default() -> Self {
        Self {
            dec_factor: 0.5,
            inc_factor: 2.1,
            ftol: 1e-4,
            wolfe: 0.9,
            max_stepsize: 1e+20,
            min_stepsize: 1e-20,
            max_linesearch_iters: 10,
            max_searches: 20,
            condition: LineSearchCondition::Wolfe,
        }
    }
}

/// Loss function variants available to the L-BFGS driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LossType {
    /// Logistic loss (binary classification, labels ±1).
    LogLoss,
}

/// Line-search strategy variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LineSearchPolicy {
    Backtracking,
    Bracketing,
}

/// Top-level parameters for an L-BFGS optimisation run.
pub(crate) struct LbfgsParams {
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
    pub linesearch_params: LineSearchParam,

    ///
    pub verbose: bool,
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

            linesearch_params: LineSearchParam::default(),

            verbose: false,
        }
    }
}
