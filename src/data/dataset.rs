use crate::shared::types::primitives::{FeatureType, LabelType};

pub trait Dataset {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;

    // Write the i-th row of features into the buffer.
    fn x_row_into(&self, i: usize, row: &mut [FeatureType]);

    // Return i-th index
    fn y_row(&self, i: usize) -> LabelType;
}
