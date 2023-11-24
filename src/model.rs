pub mod dimenion;
pub mod distance;
pub mod kernel;

use crate::{config::Output, model::dimenion::Dimension};

pub struct Weave {
    pub dimensions: Vec<Dimension>,
    pub lens: (usize, usize),
    pub values: Vec<f32>,
    pub output: Output,
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

    pub fn avg_for(&self, i: usize) -> f32 {
        let mut weight: Vec<f32> = vec![1.0; self.lens.0];
        for dim in &self.dimensions {
            dim.update_weight(i, &mut weight);
        }
        let s: f32 = weight.iter().sum();
        self.values
            .iter()
            .zip(weight.iter())
            .map(|(x, w)| x * w / s)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        dimenion::{Coords, CoordsData, Dimension, DimensionKind},
        kernel::{ExponentialFn, Kernel, TricubicFn},
        *,
    };
    use crate::data::types::Matrix;

    fn setup() -> Weave {
        // dimension 0
        let k0 = Kernel::Exponential(ExponentialFn::new(1.0));
        let c0 = Coords::F32(CoordsData {
            data: Matrix::new(vec![0_f32, 1_f32], 1),
            pred: Matrix::new(vec![0_f32], 1),
        });
        let t0 = DimensionKind::Generic;
        let dim0 = Dimension::new(k0, c0, t0);

        // dimension 1
        let k1 = Kernel::Tricubic(TricubicFn::new(1.0, 0.5));
        let c1 = Coords::F32(CoordsData {
            data: Matrix::new(vec![0_f32, 1_f32], 1),
            pred: Matrix::new(vec![0_f32], 1),
        });
        let t1 = DimensionKind::Generic;
        let dim1 = Dimension::new(k1, c1, t1);

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
        let model = setup();
        let my_avg = model.avg_for(0);
        let tr_avg = 1_f32;
        assert_eq!(my_avg, tr_avg);
    }
}
