use crate::data::dataset::Dataset;
use crate::shared::numeric::ScalarType;

pub(crate) trait LossFunc {
    /// Compute total loss and accumulate gradients over the entire dataset.
    /// Returns total loss value.
    fn evaluate_with_gradient(
        &mut self,
        dataset: &dyn Dataset,
        w: &[ScalarType],
        grad: &mut [ScalarType],
    ) -> ScalarType;
}
