use crate::shared::types::primitives::ScalarType;

/// Parameters controlling line-search behavior in L-BFGS.
pub struct LineSearchParamType {
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
    pub max_iters: usize,

    /// Maximum line-search trials per iteration.
    pub max_searches: usize,

    /// Acceptance condition: Armijo or Wolfe.
    pub condition: LineSearchCondition,
}

/// Acceptance condition for the line search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineSearchCondition {
    Armijo,
    Wolfe,
}
