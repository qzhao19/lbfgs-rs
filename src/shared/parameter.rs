use super::exception::LbfgsError;
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

/// Parse a condition string into [`LineSearchCondition`]. Case-insensitive.
fn parse_condition(s: &str) -> Result<LineSearchCondition, String> {
    match s.to_ascii_lowercase().as_str() {
        "armijo" => Ok(LineSearchCondition::Armijo),
        "wolfe" => Ok(LineSearchCondition::Wolfe),
        "strongwolfe" | "strong-wolfe" => Ok(LineSearchCondition::StrongWolfe),
        _ => Err(format!("unknown condition: {}", s)),
    }
}

/// User-facing optimisation arguments.
///
/// Mirrors the fields of [`LbfgsParams`], which are passed
/// as top-level arguments to [`crate::LBFGS::new`].
///
/// Every field is optional; fields left as `None` take the value from
/// [`LbfgsParams::default`]. Construct via struct-update syntax:
///
/// ```ignore
/// let args = OptimizeArgs {
///     max_iters: Some(20),
///     condition: Some("wolfe".to_string()),
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Debug, Default)]
pub struct OptimizeArgs {
    // ── Outer-loop convergence criteria ──
    /// Function-value convergence threshold: |f' - f| / max(1, |f|) < delta.
    pub delta: Option<ScalarType>,
    /// Gradient convergence threshold: ||g(x)|| / max(1, ||x||) < epsilon.
    pub epsilon: Option<ScalarType>,

    // ── Outer-loop control ──
    /// Maximum number of L-BFGS outer iterations.
    pub max_iters: Option<usize>,
    /// Number of correction pairs retained for the inverse-Hessian
    /// approximation (the "m" in L-BFGS).
    pub mem_size: Option<usize>,
    /// Number of past iterations used by the delta-based convergence test.
    pub past: Option<usize>,

    // ── Line-search sub-parameters (flattened) ──
    /// Step reduction factor.
    pub dec_factor: Option<ScalarType>,
    /// Step increase factor.
    pub inc_factor: Option<ScalarType>,
    /// Tolerance for the line-search acceptance condition.
    pub ftol: Option<ScalarType>,
    /// Wolfe curvature coefficient (used when `condition` is `Wolfe`).
    pub wolfe: Option<ScalarType>,
    /// Maximum allowed step size.
    pub max_stepsize: Option<ScalarType>,
    /// Minimum allowed step size.
    pub min_stepsize: Option<ScalarType>,
    /// Maximum line-search iterations.
    pub max_linesearch_iters: Option<usize>,
    /// Maximum line-search trials per iteration.
    pub max_searches: Option<usize>,
    /// Acceptance condition string: `"armijo"` | `"wolfe"` | `"strongwolfe"`
    /// (case-insensitive).
    pub condition: Option<String>,
}

impl OptimizeArgs {
    /// Validate every set field against its admissible range.
    pub(crate) fn validate(&self) -> Result<(), LbfgsError> {
        // ── Outer-loop convergence criteria ──
        if let Some(v) = self.epsilon {
            if v < 0.0 {
                return Err(LbfgsError::InvalidEpsilon);
            }
        }
        if let Some(v) = self.delta {
            if v < 0.0 {
                return Err(LbfgsError::InvalidDelta);
            }
        }

        // ── Outer-loop control ──
        if let Some(v) = self.mem_size {
            if v == 0 {
                return Err(LbfgsError::InvalidMemSize);
            }
        }

        // ── Line-search sub-parameters ──
        if let Some(v) = self.dec_factor {
            if v <= 0.0 || v >= 1.0 {
                return Err(LbfgsError::InvalidDecFactor);
            }
        }
        if let Some(v) = self.inc_factor {
            if v <= 1.0 {
                // No InvalidIncFactor variant; use InvalidParameters as fallback.
                return Err(LbfgsError::InvalidParameters);
            }
        }
        if let Some(v) = self.wolfe {
            if v <= 0.0 || v >= 1.0 {
                return Err(LbfgsError::InvalidWolfe);
            }
        }
        if let Some(v) = self.max_stepsize {
            if v <= 0.0 {
                return Err(LbfgsError::InvalidMaxStepsize);
            }
        }
        if let Some(v) = self.min_stepsize {
            if v <= 0.0 {
                return Err(LbfgsError::InvalidMinStepsize);
            }
        }
        if let Some(v) = self.max_linesearch_iters {
            if v == 0 {
                return Err(LbfgsError::InvalidMaxLineSearchIters);
            }
        }
        if let Some(v) = self.max_searches {
            if v == 0 {
                // No InvalidMaxSearches variant; use InvalidParameters as fallback.
                return Err(LbfgsError::InvalidParameters);
            }
        }

        Ok(())
    }

    /// Merge into a fully-populated [`LbfgsParams`], starting from
    /// [`LbfgsParams::default`] and overriding any fields the caller set.
    pub(crate) fn to_lbfgs_params(
        self,
        loss: LossType,
        linesearch_policy: LineSearchPolicy,
        verbose: bool,
    ) -> Result<LbfgsParams, String> {
        let mut params = LbfgsParams::default();
        params.loss = loss;
        params.linesearch_policy = linesearch_policy;
        params.verbose = verbose;

        if let Some(v) = self.delta {
            params.delta = v;
        }
        if let Some(v) = self.epsilon {
            params.epsilon = v;
        }
        if let Some(v) = self.max_iters {
            params.max_iters = v;
        }
        if let Some(v) = self.mem_size {
            params.mem_size = v;
        }
        if let Some(v) = self.past {
            params.past = v;
        }

        if let Some(v) = self.dec_factor {
            params.linesearch_params.dec_factor = v;
        }
        if let Some(v) = self.inc_factor {
            params.linesearch_params.inc_factor = v;
        }
        if let Some(v) = self.ftol {
            params.linesearch_params.ftol = v;
        }
        if let Some(v) = self.wolfe {
            params.linesearch_params.wolfe = v;
        }
        if let Some(v) = self.max_stepsize {
            params.linesearch_params.max_stepsize = v;
        }
        if let Some(v) = self.min_stepsize {
            params.linesearch_params.min_stepsize = v;
        }
        if let Some(v) = self.max_linesearch_iters {
            params.linesearch_params.max_linesearch_iters = v;
        }
        if let Some(v) = self.max_searches {
            params.linesearch_params.max_searches = v;
        }
        if let Some(s) = self.condition {
            params.linesearch_params.condition = parse_condition(&s)?;
        }

        Ok(params)
    }
}
