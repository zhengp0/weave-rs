pub mod dimenion;
pub mod distance;
pub mod kernel;

use crate::model::dimenion::Dimension;

pub struct Weave {
    pub dimensions: Vec<Box<dyn Dimension>>,
    values: Vec<f32>,
    lens: (usize, usize),
}

impl Weave {
    fn compute_weighted_avg_for(&self, i: usize) -> f32 {
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

    pub fn compute_weighted_avg(&self) -> Vec<f32> {
        (0..self.lens.1)
            .map(|i| self.compute_weighted_avg_for(i))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        dimenion::{Coords, DimensionInfo, Generic},
        distance::Euclidean,
        kernel::{Exponential, Tricubic},
        *,
    };

    fn setup() -> Weave {
        // dimension 0
        let d0 = Euclidean;
        let k0 = Exponential::new(1.0);
        let c0 = Coords {
            data: vec![vec![0_f32], vec![1_f32]],
            pred: vec![vec![0_f32]],
        };
        let dim0 = Generic(DimensionInfo::new(d0, k0, c0));

        // dimension 1
        let d1 = Euclidean;
        let k1 = Tricubic::new(1.0, 0.5);
        let c1 = Coords {
            data: vec![vec![0_f32], vec![1_f32]],
            pred: vec![vec![0_f32]],
        };
        let dim1 = Generic(DimensionInfo::new(d1, k1, c1));

        let values = vec![1_f32, 1_f32];

        Weave {
            dimensions: vec![Box::new(dim0), Box::new(dim1)],
            values,
            lens: (2, 1),
        }
    }

    #[test]
    fn test_compute_weighted_avg() {
        let model = setup();
        let my_avg = model.compute_weighted_avg();
        let tr_avg = vec![1_f32];
        assert_eq!(my_avg, tr_avg);
    }
}
