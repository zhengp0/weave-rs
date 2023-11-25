use super::kernel::{DepthCODEm, Exponential, Kernel, Tricubic};
use crate::data::types::Matrix;

pub trait GenericWorker {
    fn update_weight(&self, i: usize, weight: &mut [f32]);
}

pub trait CategoricalWorker {
    fn update_weight(&self, i: usize, weight: &mut [f32]);
}

pub trait AdaptiveWorker {
    fn update_weight(&self, i: usize, weight: &mut [f32]);
}

pub struct DimensionHandle<K: Kernel> {
    kernel: K,
    coord_data: Matrix<K::CType>,
    coord_pred: Matrix<K::CType>,
}
impl<K: Kernel> DimensionHandle<K> {
    pub fn new(kernel: K, coord_data: Matrix<K::CType>, coord_pred: Matrix<K::CType>) -> Self {
        Self {
            kernel,
            coord_data,
            coord_pred,
        }
    }
}

impl<K: Kernel> GenericWorker for DimensionHandle<K> {
    fn update_weight(&self, i: usize, weight: &mut [f32]) {
        let x = self.coord_data.rows().nth(i).unwrap();
        self.coord_pred
            .rows()
            .zip(weight.iter_mut())
            .for_each(|(y, w)| *w *= self.kernel.kernel(x, y))
    }
}

impl CategoricalWorker for DimensionHandle<DepthCODEm> {
    fn update_weight(&self, i: usize, weight: &mut [f32]) {
        let x = self.coord_data.rows().nth(i).unwrap();
        let mut weight_sum: Vec<f32> = vec![0.0; self.kernel.maxlvl as usize + 1];

        let distance: Vec<i32> = self
            .coord_pred
            .rows()
            .zip(weight.iter())
            .map(|(y, w)| {
                let d = self.kernel.distance(x, y);
                weight_sum[d as usize] += w;
                d
            })
            .collect();

        distance
            .iter()
            .zip(weight.iter_mut())
            .map(|(d, w)| (&weight_sum[*d as usize], d, w))
            .filter(|(s, ..)| **s > 0.0)
            .for_each(|(s, d, w)| {
                *w /= s;
                *w *= self.kernel.kernel_from_distance(d);
            });
    }
}

impl AdaptiveWorker for DimensionHandle<Tricubic> {
    fn update_weight(&self, i: usize, weight: &mut [f32]) {
        let x = self.coord_data.rows().nth(i).unwrap();
        let distance: Vec<f32> = self
            .coord_pred
            .rows()
            .map(|y| self.kernel.distance(x, y))
            .collect();
        let radius = distance
            .iter()
            .max_by(|x, y| x.partial_cmp(y).unwrap())
            .map(|x| x + 1.0)
            .unwrap();
        let kernel = Tricubic {
            radius,
            ..self.kernel
        };
        distance
            .iter()
            .zip(weight.iter_mut())
            .for_each(|(d, w)| *w *= kernel.kernel_from_distance(d));
    }
}

pub enum Dimension {
    GenericExponential(DimensionHandle<Exponential>),
    GenericTricubic(DimensionHandle<Tricubic>),
    GenericDepthCODEm(DimensionHandle<DepthCODEm>),
    CategoricalDepthCODEm(DimensionHandle<DepthCODEm>),
    AdaptiveTricubic(DimensionHandle<Tricubic>),
}

impl Dimension {
    pub fn update_weight(&self, i: usize, weight: &mut [f32]) {
        match self {
            Self::GenericExponential(handle) => GenericWorker::update_weight(handle, i, weight),
            Self::GenericTricubic(handle) => GenericWorker::update_weight(handle, i, weight),
            Self::GenericDepthCODEm(handle) => GenericWorker::update_weight(handle, i, weight),
            Self::CategoricalDepthCODEm(handle) => {
                CategoricalWorker::update_weight(handle, i, weight)
            }
            Self::AdaptiveTricubic(handle) => AdaptiveWorker::update_weight(handle, i, weight),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generic_update_weight() {
        let handle = DimensionHandle::new(
            DepthCODEm::new(0.5, 3),
            Matrix::new(vec![0, 1, 2], 3),
            Matrix::new(vec![0, 1, 2, 0, 1, 8, 0, 6, 7, 3, 4, 5], 3),
        );
        let mut my_weight: Vec<f32> = vec![1.0; 4];
        GenericWorker::update_weight(&handle, 0, &mut my_weight);
        let ok_weight = vec![0.5, 0.25, 0.25, 0.0];
        assert_eq!(my_weight, ok_weight);
    }

    #[test]
    fn test_categorical_update_weight() {
        let handle = DimensionHandle::new(
            DepthCODEm::new(0.5, 3),
            Matrix::new(vec![0, 1, 2], 3),
            Matrix::new(vec![0, 1, 2, 0, 1, 8, 0, 6, 7, 3, 4, 5], 3),
        );
        let mut my_weight: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        CategoricalWorker::update_weight(&handle, 0, &mut my_weight);
        let ok_weight = vec![0.5, 0.25, 0.25, 0.0];
        assert_eq!(my_weight, ok_weight);
    }
}
