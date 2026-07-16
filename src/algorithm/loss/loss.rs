use crate::data::dataset::Dataset;
use crate::shared::types::primitives::{FeatureType, LabelType, ScalarType};

pub trait LossFunc {
    /// Compute loss for a single sample.
    fn evaluate(&self, y_pred: FeatureType, y_true: LabelType) -> ScalarType;

    /// Compute gradient of loss w.r.t. prediction.
    fn derivate(&self, y_pred: FeatureType, y_true: LabelType) -> ScalarType;

    /// Compute total loss and accumulate gradients over the entire dataset.
    /// Returns total loss value.
    fn evaluate_with_gradient(
        &mut self,
        dataset: &dyn Dataset,
        w: &[FeatureType],
        grad: &mut [FeatureType],
    ) -> ScalarType;
}
