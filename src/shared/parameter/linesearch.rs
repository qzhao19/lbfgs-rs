use crate::shared::numeric::scalar::ScalarType;

/// Acceptance condition for the line search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineSearchCondition {
    Armijo,
    Wolfe,
    StrongWolfe,
}

/// Parameters controlling line-search behavior in L-BFGS.
#[derive(Clone, Copy)]
pub struct LineSearchParam {
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
