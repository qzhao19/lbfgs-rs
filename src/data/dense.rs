use crate::shared::types::primitives::{FeatureType, LabelType};
use super::dataset::Dataset;

#[derive(Clone)]
pub struct DenseDataset {
    /// Column-major matrix：X[r, c] = x_data[r + c * nrows]
    x_data: Vec<FeatureType>,
    y_data: Vec<LabelType>,
    nrows: usize,
    ncols: usize,
    /// row-major cache：cache[r * ncols + c] = X[r, c]
    x_row_cache: Option<Vec<FeatureType>>,
}

