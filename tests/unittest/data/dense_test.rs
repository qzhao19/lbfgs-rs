use lbfgs_rs::data::dataset::Dataset;
use lbfgs_rs::data::dense::DenseDataset;
use lbfgs_rs::shared::types::primitives::{FeatureType, LabelType};

/// Build a 3×2 row-major dataset: [[1,2], [3,4], [5,6]]
fn rectangular_3x2(cache: bool) -> DenseDataset {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y = vec![10.0, 20.0, 30.0];
    DenseDataset::new(x, y, 3, 2, cache).unwrap()
}

/// Build a 2×2 row-major dataset: [[1,2], [3,4]]
fn square_2x2(cache: bool) -> DenseDataset {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![0.0, 1.0];
    DenseDataset::new(x, y, 2, 2, cache).unwrap()
}

mod construction {
    use super::*;

    #[test]
    fn t1_1_square_no_cache() {
        let ds = square_2x2(false);
        assert_eq!(ds.nrows(), 2);
        assert_eq!(ds.ncols(), 2);
        assert!(!ds.is_cache_enabled());
    }

    #[test]
    fn t1_2_rectangular_with_cache() {
        let ds = rectangular_3x2(true);
        assert_eq!(ds.nrows(), 3);
        assert_eq!(ds.ncols(), 2);
        assert!(ds.is_cache_enabled());
    }

    #[test]
    fn t1_3_single_row_many_features() {
        let x: Vec<FeatureType> = (0..100).map(|i| i as FeatureType).collect();
        let y = vec![1.0];
        let ds = DenseDataset::new(x, y, 1, 100, false).unwrap();
        assert_eq!(ds.nrows(), 1);
        assert_eq!(ds.ncols(), 100);
    }

    #[test]
    fn t1_4_many_samples_single_feature() {
        let x: Vec<FeatureType> = (0..100).map(|i| i as FeatureType).collect();
        let y: Vec<LabelType> = (0..100).map(|_| 1.0).collect();
        let ds = DenseDataset::new(x, y, 100, 1, false).unwrap();
        assert_eq!(ds.nrows(), 100);
        assert_eq!(ds.ncols(), 1);
        assert_eq!(ds.x_row(42), &[42.0]);
    }

    #[test]
    fn t1_5_x_data_too_small() {
        let x = vec![1.0, 2.0];
        let y = vec![0.0, 1.0];
        assert!(DenseDataset::new(x, y, 2, 2, false).is_err());
    }

    #[test]
    fn t1_6_x_data_exact_size() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0];
        let ds = DenseDataset::new(x, y, 2, 2, false).unwrap();
        assert_eq!(ds.x_row(0), &[1.0, 2.0]);
    }

    #[test]
    fn t1_7_y_data_too_small() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0];
        assert!(DenseDataset::new(x, y, 2, 2, false).is_err());
    }

    #[test]
    fn t1_8_empty_dataset() {
        let ds = DenseDataset::new(vec![], vec![], 0, 0, false).unwrap();
        assert_eq!(ds.nrows(), 0);
        assert_eq!(ds.ncols(), 0);
    }

    #[test]
    fn t1_9_cache_flag() {
        let with = square_2x2(true);
        let without = square_2x2(false);
        assert!(with.is_cache_enabled());
        assert!(!without.is_cache_enabled());
    }
}

mod metadata {
    use super::*;

    #[test]
    fn t2_1_nrows_square_and_rect() {
        assert_eq!(square_2x2(false).nrows(), 2);
        assert_eq!(rectangular_3x2(false).nrows(), 3);
    }

    #[test]
    fn t2_2_ncols_square_and_rect() {
        assert_eq!(square_2x2(false).ncols(), 2);
        assert_eq!(rectangular_3x2(false).ncols(), 2);
    }

    #[test]
    fn t2_3_t2_4_cache_flag_reflects_enable_cache() {
        assert!(square_2x2(true).is_cache_enabled());
        assert!(!square_2x2(false).is_cache_enabled());
    }

    #[test]
    fn t2_5_x_data_slice() {
        let ds = rectangular_3x2(false);
        let slice = ds.x_data();
        assert_eq!(slice.len(), 6);
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn t2_6_y_data_slice() {
        let ds = rectangular_3x2(false);
        let slice = ds.y_data();
        assert_eq!(slice.len(), 3);
        assert_eq!(slice, &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn t2_7_data_slice_matches_construction_input() {
        let x = vec![7.0, 8.0, 9.0, 10.0];
        let y = vec![100.0, 200.0];
        let ds = DenseDataset::new(x.clone(), y.clone(), 2, 2, false).unwrap();
        assert_eq!(ds.x_data(), &x[..]);
        assert_eq!(ds.y_data(), &y[..]);
    }
}

mod row_access {
    use super::*;

    #[test]
    fn t3_1_first_row_square() {
        let ds = square_2x2(false);
        assert_eq!(ds.x_row(0), &[1.0, 2.0]);
    }

    #[test]
    fn t3_2_last_row_square() {
        let ds = square_2x2(false);
        assert_eq!(ds.x_row(1), &[3.0, 4.0]);
    }

    #[test]
    fn t3_3_all_rows_rectangular() {
        let ds = rectangular_3x2(false);
        let expected = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        for i in 0..ds.nrows() {
            assert_eq!(ds.x_row(i), expected[i].as_slice());
        }
    }

    #[test]
    fn t3_4_fill_x_row_no_cache() {
        let ds = rectangular_3x2(false);
        let mut buf = vec![0.0; 2];
        ds.fill_x_row(1, &mut buf);
        assert_eq!(buf, vec![3.0, 4.0]);
    }

    #[test]
    fn t3_5_fill_x_row_with_cache() {
        let ds = rectangular_3x2(true);
        let mut buf = vec![0.0; 2];
        ds.fill_x_row(1, &mut buf);
        assert_eq!(buf, vec![3.0, 4.0]);
    }

    #[test]
    fn t3_6_fill_x_row_consistency_all_rows() {
        let ds = rectangular_3x2(false);
        let mut buf = vec![0.0; 2];
        for i in 0..ds.nrows() {
            ds.fill_x_row(i, &mut buf);
            assert_eq!(&buf, ds.x_row(i), "row {i} mismatch");
        }
    }
}

mod col_access {
    use super::*;

    #[test]
    fn t4_1_fill_x_col_square_with_cache() {
        let ds = square_2x2(true);
        let mut buf = vec![0.0; 2];
        ds.fill_x_col(0, &mut buf);
        assert_eq!(buf, vec![1.0, 3.0]);
        ds.fill_x_col(1, &mut buf);
        assert_eq!(buf, vec![2.0, 4.0]);
    }

    #[test]
    fn t4_2_fill_x_col_square_no_cache() {
        let ds = square_2x2(false);
        let mut buf = vec![0.0; 2];
        ds.fill_x_col(0, &mut buf);
        assert_eq!(buf, vec![1.0, 3.0]);
        ds.fill_x_col(1, &mut buf);
        assert_eq!(buf, vec![2.0, 4.0]);
    }

    #[test]
    fn t4_3_fill_x_col_rectangular_with_cache() {
        let ds = rectangular_3x2(true);
        let mut buf = vec![0.0; 3];
        ds.fill_x_col(0, &mut buf);
        assert_eq!(buf, vec![1.0, 3.0, 5.0]);
        ds.fill_x_col(1, &mut buf);
        assert_eq!(buf, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn t4_4_fill_x_col_rectangular_no_cache() {
        // P0 CRITICAL: regression for r*nrows + j bug
        let ds = rectangular_3x2(false);
        let mut buf = vec![0.0; 3];
        ds.fill_x_col(0, &mut buf);
        assert_eq!(buf, vec![1.0, 3.0, 5.0]);
        ds.fill_x_col(1, &mut buf);
        assert_eq!(buf, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn t4_5_first_and_last_column() {
        let ds = square_2x2(false);
        let mut buf = vec![0.0; 2];
        ds.fill_x_col(0, &mut buf);
        assert_eq!(buf, vec![1.0, 3.0]);
        ds.fill_x_col(1, &mut buf);
        assert_eq!(buf, vec![2.0, 4.0]);
    }

    #[test]
    fn t4_6_last_column_rectangular() {
        let ds = rectangular_3x2(false);
        let mut buf = vec![0.0; 3];
        ds.fill_x_col(1, &mut buf);
        assert_eq!(buf, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn t4_7_buffer_exact_size() {
        let ds = rectangular_3x2(true);
        let mut buf = vec![0.0; 3];
        ds.fill_x_col(0, &mut buf);
        assert_eq!(buf, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn t4_8_x_col_vec_with_cache() {
        let ds = rectangular_3x2(true);
        assert_eq!(ds.x_col(0), vec![1.0, 3.0, 5.0]);
        assert_eq!(ds.x_col(1), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn t4_9_x_col_vec_no_cache() {
        let ds = rectangular_3x2(false);
        assert_eq!(ds.x_col(0), vec![1.0, 3.0, 5.0]);
        assert_eq!(ds.x_col(1), vec![2.0, 4.0, 6.0]);
    }
}

mod trait_polymorphism {
    use super::*;

    #[test]
    fn t5_1_trait_nrows_ncols() {
        let ds = rectangular_3x2(false);
        let d: &dyn Dataset = &ds;
        assert_eq!(d.nrows(), 3);
        assert_eq!(d.ncols(), 2);
    }

    #[test]
    fn t5_2_trait_fill_x_row() {
        let ds = rectangular_3x2(false);
        let d: &dyn Dataset = &ds;
        let mut buf = vec![0.0; 2];
        d.fill_x_row(1, &mut buf);
        assert_eq!(buf, vec![3.0, 4.0]);
    }

    #[test]
    fn t5_3_trait_fill_x_col_with_and_without_cache() {
        {
            let ds = rectangular_3x2(false);
            let d: &dyn Dataset = &ds;
            let mut buf = vec![0.0; 3];
            d.fill_x_col(0, &mut buf);
            assert_eq!(buf, vec![1.0, 3.0, 5.0]);
        }
        {
            let ds = rectangular_3x2(true);
            let d: &dyn Dataset = &ds;
            let mut buf = vec![0.0; 3];
            d.fill_x_col(1, &mut buf);
            assert_eq!(buf, vec![2.0, 4.0, 6.0]);
        }
    }

    #[test]
    fn t5_4_trait_y_row() {
        let ds = rectangular_3x2(false);
        let d: &dyn Dataset = &ds;
        assert_eq!(d.y_row(0), 10.0);
        assert_eq!(d.y_row(1), 20.0);
        assert_eq!(d.y_row(2), 30.0);
    }

    #[test]
    fn t5_5_trait_vs_concrete_consistency() {
        let ds = rectangular_3x2(true);
        let d: &dyn Dataset = &ds;
        let mut buf_trait = vec![0.0; 3];
        let mut buf_concrete = vec![0.0; 3];

        d.fill_x_col(0, &mut buf_trait);
        ds.fill_x_col(0, &mut buf_concrete);
        assert_eq!(buf_trait, buf_concrete);

        d.fill_x_col(1, &mut buf_trait);
        ds.fill_x_col(1, &mut buf_concrete);
        assert_eq!(buf_trait, buf_concrete);

        let mut row_trait = vec![0.0; 2];
        let mut row_concrete = vec![0.0; 2];
        for i in 0..ds.nrows() {
            d.fill_x_row(i, &mut row_trait);
            ds.fill_x_row(i, &mut row_concrete);
            assert_eq!(row_trait, row_concrete, "row {i}");
        }
    }
}

mod numeric_precision {
    use super::*;

    #[test]
    fn t6_1_f64_exact_roundtrip() {
        let values = vec![1.0, -0.5, 1e-10, 0.0];
        let n = values.len();
        let ds = DenseDataset::new(values.clone(), values.clone(), n, 1, false).unwrap();
        for i in 0..n {
            assert_eq!(ds.x_row(i)[0], values[i]);
            assert_eq!(ds.y_row(i), values[i]);
        }
    }

    #[test]
    #[cfg(feature = "f32")]
    fn t6_2_f32_exact_roundtrip() {
        let values: Vec<f32> = vec![1.0, -0.5, 1e-7, 0.0];
        let n = values.len();
        let ds = DenseDataset::new(values.clone(), values.clone(), n, 1, false).unwrap();
        for i in 0..n {
            assert!((ds.x_row(i)[0] - values[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn t6_3_all_zeros_no_panic() {
        let x = vec![0.0; 100];
        let y = vec![0.0; 10];
        let ds = DenseDataset::new(x, y, 10, 10, true).unwrap();
        let mut buf = vec![1.0; 10];
        ds.fill_x_row(0, &mut buf);
        assert_eq!(buf, vec![0.0; 10]);
        ds.fill_x_col(0, &mut buf);
        assert_eq!(buf, vec![0.0; 10]);
        assert_eq!(ds.x_col(5), vec![0.0; 10]);
    }
}
