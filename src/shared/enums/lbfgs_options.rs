/// Loss function variants available to the L-BFGS driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LossType {
    /// Logistic loss (binary classification, labels ±1).
    LogLoss,
}

/// Line-search strategy variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineSearchPolicy {
    Backtracking,
    Bracketing,
}

/// Acceptance condition for the line search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineSearchCondition {
    Armijo,
    Wolfe,
    StrongWolfe,
}
