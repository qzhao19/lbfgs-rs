/// Success termination status of the L-BFGS outer loop.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LbfgsStatus {
    /// L-BFGS reaches convergence.
    Convergence = 0,
    /// L-BFGS satisfies stopping criteria.
    Stop = 1,
    /// The iteration has been canceled by the monitor callback.
    Canceled = 2,
    /// ALREADY_MINIMIZED
    AlreadyMinimized = 3,
}

impl LbfgsStatus {
    pub fn status_message(&self) -> &'static str {
        match self {
            LbfgsStatus::Convergence => "convergence",
            LbfgsStatus::Stop => "stop",
            LbfgsStatus::Canceled => "cancelled",
            LbfgsStatus::AlreadyMinimized => "already minimized",
        }
    }
}

/// Error codes returned by L-BFGS routines
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LbfgsError {
    /// Unknown error.
    UnknownError = -1024,
    /// Invalid number of variables specified.
    InvalidN,
    /// Invalid parameter mem_size specified.
    InvalidMemSize,
    /// Invalid parameter epsilon specified.
    InvalidEpsilon,
    /// Invalid parameter past specified.
    InvalidPast,
    /// Invalid parameter delta specified.
    InvalidDelta,
    /// Invalid parameter min_stepsize specified.
    InvalidMinStepsize,
    /// Invalid parameter max_stepsize specified.
    InvalidMaxStepsize,
    /// Invalid parameter dec_factor specified.
    InvalidDecFactor,
    /// Invalid parameter wolfe specified.
    InvalidWolfe,
    /// Invalid parameter machine_prec specified.
    InvalidMachinePrec,
    /// Invalid parameter max_linesearch_iters specified.
    InvalidMaxLineSearchIters,
    /// The function value became NaN or Inf.
    InvalidFuncVal,
    /// The line-search step became smaller than min_stepsize.
    MinimumStep,
    /// The line-search step became larger than max_stepsize.
    MaximumStep,
    /// Line search reaches the maximum, assumptions not satisfied or precision not achievable.
    MaximumLineSearchIteration,
    /// The algorithm routine reaches the maximum number of iterations.
    MaximumIteration,
    /// Relative search interval width is at least machine_prec.
    WidthTooSmall,
    /// A logic error (negative line-search step) occurred.
    InvalidParameters,
    /// The current search direction increases the cost function value.
    IncreaseGradient,
}

impl LbfgsError {
    pub fn error_message(&self) -> &'static str {
        match self {
            LbfgsError::UnknownError => "unknown error",
            LbfgsError::InvalidN => "invalid number of variables specified",
            LbfgsError::InvalidMemSize => "invalid parameter mem_size specified",
            LbfgsError::InvalidEpsilon => "invalid parameter epsilon specified",
            LbfgsError::InvalidPast => "invalid parameter past specified",
            LbfgsError::InvalidDelta => "invalid parameter delta specified",
            LbfgsError::InvalidMinStepsize => "invalid parameter min_stepsize specified",
            LbfgsError::InvalidMaxStepsize => "invalid parameter max_stepsize specified",
            LbfgsError::InvalidDecFactor => "invalid parameter dec_factor specified",
            LbfgsError::InvalidWolfe => "invalid parameter wolfe specified",
            LbfgsError::InvalidMachinePrec => "invalid parameter machine_prec specified",
            LbfgsError::InvalidMaxLineSearchIters => {
                "invalid parameter max_linesearch_iters specified"
            }
            LbfgsError::InvalidFuncVal => "the function value became NaN or Inf",
            LbfgsError::MinimumStep => "the line-search step became smaller than min_stepsize",
            LbfgsError::MaximumStep => "the line-search step became larger than max_stepsize",
            LbfgsError::MaximumLineSearchIteration => {
                "line search reaches the maximum, assumptions not satisfied or precision not achievable"
            }
            LbfgsError::MaximumIteration => {
                "the algorithm routine reaches the maximum number of iterations"
            }
            LbfgsError::WidthTooSmall => {
                "relative search interval width is at least machine_prec"
            }
            LbfgsError::InvalidParameters => {
                "a logic error (negative line-search step) occurred"
            }
            LbfgsError::IncreaseGradient => {
                "the current search direction increases the cost function value"
            }
        }
    }
}
