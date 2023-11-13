pub mod dimenion;
pub mod distance;
pub mod kernel;

use crate::config::Output;
use crate::data::{write_parquet_col, Result};
use crate::model::dimenion::Dimension;

pub struct Weave {
    pub dimensions: Vec<Dimension>,
    values: Vec<f32>,
    lens: (usize, usize),
    output: Output,
}

impl Weave {
    pub fn new(
        dimensions: Vec<Dimension>,
        values: Vec<f32>,
        lens: (usize, usize),
        output: Output,
    ) -> Self {
        Self {
            dimensions,
            values,
            lens,
            output,
        }
    }

    fn compute_weighted_avg_for(&mut self, i: usize) -> f32 {
        let mut weight: Vec<f32> = vec![1.0; self.lens.0];
        for dim in &mut self.dimensions {
            dim.update_weight(i, &mut weight);
        }
        let s: f32 = weight.iter().sum();
        self.values
            .iter()
            .zip(weight.iter())
            .map(|(x, w)| x * w / s)
            .sum()
    }

    pub fn compute_weighted_avg(&mut self) -> Vec<f32> {
        (0..self.lens.1)
            .map(|i| self.compute_weighted_avg_for(i))
            .collect()
    }

    pub fn run(&mut self) -> Result<()> {
        let weighted_avg = self.compute_weighted_avg();
        write_parquet_col::<f32>(&self.output.path, &self.output.values, &weighted_avg)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        dimenion::{Coords, CoordsData, Dimension, DimensionKind},
        distance::{Distance, EuclideanFn},
        kernel::{ExponentialFn, Kernel, TricubicFn},
        *,
    };
    use crate::data::Matrix;

    fn setup() -> Weave {
        // dimension 0
        let d0 = Distance::Euclidean(EuclideanFn);
        let k0 = Kernel::Exponential(ExponentialFn::new(1.0));
        let c0 = Coords::F32(CoordsData {
            data: Matrix::new(vec![0_f32, 1_f32], 1),
            pred: Matrix::new(vec![0_f32], 1),
        });
        let t0 = DimensionKind::Generic;
        let dim0 = Dimension::new(d0, k0, c0, t0);

        // dimension 1
        let d1 = Distance::Euclidean(EuclideanFn);
        let k1 = Kernel::Tricubic(TricubicFn::new(1.0, 0.5));
        let c1 = Coords::F32(CoordsData {
            data: Matrix::new(vec![0_f32, 1_f32], 1),
            pred: Matrix::new(vec![0_f32], 1),
        });
        let t1 = DimensionKind::Generic;
        let dim1 = Dimension::new(d1, k1, c1, t1);

        let values = vec![1_f32, 1_f32];
        let output = Output {
            path: "example/result.parquet".to_string(),
            values: "prediction".to_string(),
        };

        Weave {
            dimensions: vec![dim0, dim1],
            values,
            lens: (2, 1),
            output,
        }
    }

    #[test]
    fn test_compute_weighted_avg() {
        let mut model = setup();
        let my_avg = model.compute_weighted_avg();
        let tr_avg = vec![1_f32];
        assert_eq!(my_avg, tr_avg);
    }
}
