#![allow(dead_code)]

use super::dataset::Dataset;
use crate::shared::numeric::{FeatureType, LabelType};

pub(crate) struct DenseDataset {
    /// Row-major matrix：X[r, c] = x_data[r * ncols + c]
    x_data: Vec<FeatureType>,
    y_data: Vec<LabelType>,
    nrows: usize,
    ncols: usize,

    /// Column-major cache: cache[c * nrows + r] = X[r, c]
    x_col_cache: Option<Vec<FeatureType>>,
}

impl DenseDataset {
    /// Use Vec to contruct dataset
    pub fn new(
        x_data: Vec<FeatureType>,
        y_data: Vec<LabelType>,
        nrows: usize,
        ncols: usize,
        enable_cache: bool,
    ) -> Result<Self, String> {
        if x_data.len() < nrows * ncols {
            return Err(format!(
                "x_data length {} < nrows * ncols ({})",
                x_data.len(),
                nrows * ncols
            ));
        }

        if y_data.len() < nrows {
            return Err(format!(
                "y_data length {} < nrows ({})",
                y_data.len(),
                nrows
            ));
        }

        let x_col_cache = if enable_cache {
            Some(build_col_cache(&x_data, nrows, ncols))
        } else {
            None
        };

        return Ok(Self {
            x_data,
            y_data,
            nrows,
            ncols,
            x_col_cache,
        });
    }

    #[inline]
    pub fn x_row(&self, i: usize) -> &[FeatureType] {
        debug_assert!(i < self.nrows, "row index {} out of range", i);
        return &self.x_data[i * self.ncols..(i + 1) * self.ncols];
    }

    #[inline]
    pub fn x_col(&self, j: usize) -> Vec<FeatureType> {
        debug_assert!(j < self.ncols, "column index {} out of range", j);
        match &self.x_col_cache {
            Some(cache) => cache[j * self.nrows..(j + 1) * self.nrows].to_vec(),
            None => {
                let mut col = Vec::with_capacity(self.nrows);
                for r in 0..self.nrows {
                    col.push(self.x_data[r * self.ncols + j]);
                }
                col
            }
        }
    }

    /// Fill buffer with the i-th feature row.
    #[inline]
    pub fn fill_x_row(&self, i: usize, buf: &mut [FeatureType]) {
        buf.copy_from_slice(self.x_row(i));
    }

    #[inline]
    pub fn fill_x_col(&self, j: usize, buf: &mut [FeatureType]) {
        debug_assert!(j < self.ncols, "column index {} out of range", j);
        match &self.x_col_cache {
            Some(cache) => {
                buf.copy_from_slice(&cache[j * self.nrows..(j + 1) * self.nrows]);
            }
            None => {
                for r in 0..self.nrows {
                    buf[r] = self.x_data[r * self.ncols + j];
                }
            }
        }
    }

    #[inline]
    pub fn y_row(&self, i: usize) -> LabelType {
        debug_assert!(i < self.nrows, "row index {} out of range", i);
        self.y_data[i]
    }

    pub fn x_data(&self) -> &[FeatureType] {
        return &self.x_data;
    }
    pub fn y_data(&self) -> &[LabelType] {
        return &self.y_data;
    }
    pub fn nrows(&self) -> usize {
        return self.nrows;
    }
    pub fn ncols(&self) -> usize {
        return self.ncols;
    }
    pub fn is_cache_enabled(&self) -> bool {
        return self.x_col_cache.is_some();
    }
}

impl Dataset for DenseDataset {
    fn nrows(&self) -> usize {
        return self.nrows;
    }
    fn ncols(&self) -> usize {
        return self.ncols;
    }

    fn fill_x_row(&self, i: usize, buf: &mut [FeatureType]) {
        self.fill_x_row(i, buf);
    }

    fn fill_x_col(&self, j: usize, buf: &mut [FeatureType]) {
        self.fill_x_col(j, buf);
    }

    fn y_row(&self, i: usize) -> LabelType {
        return self.y_row(i);
    }
}

fn build_col_cache(x_data: &[FeatureType], nrows: usize, ncols: usize) -> Vec<FeatureType> {
    let total: usize = nrows * ncols;
    let mut cache: Vec<FeatureType> = vec![0.0 as FeatureType; total];

    for r in 0..nrows {
        // Sequentially read the entire row
        let x_row_ptr: usize = r * ncols;
        for c in 0..ncols {
            cache[c * nrows + r] = x_data[x_row_ptr + c];
        }
    }
    return cache;
}
