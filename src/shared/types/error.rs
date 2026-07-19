/// Success termination status of the L-BFGS outer loop.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LbfgsStatus {
    /// L-BFGS reaches convergence.
    Convergence = 0,
    /// L-BFGS satisfies stopping criteria.
    Stop = 1,
    /// The iteration has been canceled by the monitor callback.
    Canceled = 2,
}

/// Error codes returned by L-BFGS routines
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LbfgsError {
    /// Unknown error.
    UnknownError = -1024,
    /// Invalid number of variables specified.
    InvalidN,
    /// Invalid parameter mem_size specified.
    InvalidMemSize,
    /// Invalid parameter g_epsilon specified.
    InvalidGEpsilon,
    /// Invalid parameter past specified.
    InvalidTestPeriod,
    /// Invalid parameter delta specified.
    InvalidDelta,
    /// Invalid parameter min_step specified.
    InvalidMinStep,
    /// Invalid parameter max_step specified.
    InvalidMaxStep,
    /// Invalid parameter f_dec_coeff specified.
    InvalidFDecCoeff,
    /// Invalid parameter s_curv_coeff specified.
    InvalidSCurvCoeff,
    /// Invalid parameter machine_prec specified.
    InvalidMachinePrec,
    /// Invalid parameter max_linesearch specified.
    InvalidMaxLineSearch,
    /// The function value became NaN or Inf.
    InvalidFuncVal,
    /// The line-search step became smaller than min_step.
    MinimumStep,
    /// The line-search step became larger than max_step.
    MaximumStep,
    /// Line search reaches the maximum, assumptions not satisfied or precision not achievable.
    MaximumLineSearch,
    /// The algorithm routine reaches the maximum number of iterations.
    MaximumIteration,
    /// Relative search interval width is at least machine_prec.
    WidthTooSmall,
    /// A logic error (negative line-search step) occurred.
    InvalidParameters,
    /// The current search direction increases the cost function value.
    IncreaseGradient,
}
