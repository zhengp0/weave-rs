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
        dimenion::{Dimension, DimensionHandle},
        kernel::{Exponential, Tricubic},
        *,
    };
    use crate::data::types::Matrix;

    fn setup() -> Weave {
        // dimension 0
        let dim0 = Dimension::GenericExponential(DimensionHandle::new(
            Exponential::new(1.0),
            Matrix::new(vec![0_f32, 1_f32], 1),
            Matrix::new(vec![0_f32], 1),
        ));

        // dimension 1
        let dim1 = Dimension::GenericTricubic(DimensionHandle::new(
            Tricubic::new(1.0, 0.5),
            Matrix::new(vec![0_f32, 1_f32], 1),
            Matrix::new(vec![0_f32], 1),
        ));

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
