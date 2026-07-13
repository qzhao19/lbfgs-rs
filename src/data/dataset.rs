use crate::shared::types::primitives::{FeatureType, LabelType};

pub trait Dataset {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    
    // Fill `buffer` with the i‑th instance row.
    fn fill_x_row(&self, i: usize, buf: &mut [FeatureType]);

    // Fill `buffer` with the j‑th feature col.
    fn fill_x_col(&self, j: usize, buff: &mut [FeatureType]);

    // Return i-th index
    fn y_row(&self, i: usize) -> LabelType;
}
