use crate::shared::types::error::LbfgsError;
use crate::shared::types::primitives::{FeatureType, ScalarType};

pub trait LineSearch {
    /// Returns the number of function evaluations on success.
    fn search(
        &mut self,
        xp: &[FeatureType],
        gp: &[FeatureType],
        d: &[FeatureType],
        x: &mut [FeatureType],
        g: &mut [FeatureType],
        fx: &mut ScalarType,
        stepsize: &mut ScalarType,
    ) -> Result<usize, LbfgsError>;
}
