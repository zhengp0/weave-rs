use crate::data::Matrix;
use crate::model::{distance::Distance, kernel::Kernel};
use serde::Deserialize;

pub struct CoordsData<T> {
    pub data: Matrix<T>,
    pub pred: Matrix<T>,
}

pub enum Coords {
    I32(CoordsData<i32>),
    F32(CoordsData<f32>),
}

#[derive(Deserialize)]
pub enum DimensionKind {
    Generic,
    Categorical,
    Adaptive,
}

pub struct Dimension {
    pub distance: Distance,
    pub kernel: Kernel,
    pub coords: Coords,
    pub kind: DimensionKind,
}

impl Dimension {
    pub fn new(distance: Distance, kernel: Kernel, coords: Coords, kind: DimensionKind) -> Self {
        Self {
            distance,
            kernel,
            coords,
            kind,
        }
    }

    pub fn update_weight(&mut self, i: usize, weight: &mut Vec<f32>) {
        match self {
            Self {
                distance: Distance::Euclidean(distance_fn),
                kernel: Kernel::Exponential(kernel_fn),
                coords: Coords::F32(coords),
                kind: DimensionKind::Generic,
            } => {
                let x = coords.data.rows().nth(i).unwrap();
                let y_iter = coords.pred.rows();
                for (y, w) in y_iter.zip(weight.iter_mut()) {
                    *w *= kernel_fn.call(&distance_fn.call(x, y));
                }
            }
            Self {
                distance: Distance::Euclidean(distance_fn),
                kernel: Kernel::Tricubic(kernel_fn),
                coords: Coords::F32(coords),
                kind: DimensionKind::Generic,
            } => {
                let x = coords.data.rows().nth(i).unwrap();
                let y_iter = coords.pred.rows();
                for (y, w) in y_iter.zip(weight.iter_mut()) {
                    *w *= kernel_fn.call(&distance_fn.call(x, y));
                }
            }
            Self {
                distance: Distance::Tree(distance_fn),
                kernel: Kernel::DepthCODEm(kernel_fn),
                coords: Coords::I32(coords),
                kind: DimensionKind::Generic,
            } => {
                let x = coords.data.rows().nth(i).unwrap();
                let y_iter = coords.pred.rows();
                for (y, w) in y_iter.zip(weight.iter_mut()) {
                    *w *= kernel_fn.call(&distance_fn.call(x, y));
                }
            }
            Self {
                distance: Distance::Tree(distance_fn),
                kernel: Kernel::DepthCODEm(kernel_fn),
                coords: Coords::I32(coords),
                kind: DimensionKind::Categorical,
            } => {
                let x = coords.data.rows().nth(i).unwrap();
                let y_iter = coords.pred.rows();
                let mut weight_sum: Vec<f32> = vec![0.0; kernel_fn.maxlvl as usize + 1];
                let distance: Vec<i32> = y_iter
                    .zip(weight.iter())
                    .map(|(y, w)| {
                        let d = distance_fn.call(x, y);
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
                        *w *= kernel_fn.call(d);
                    });
            }
            Self {
                distance: Distance::Euclidean(distance_fn),
                kernel: Kernel::Tricubic(kernel_fn),
                coords: Coords::F32(coords),
                kind: DimensionKind::Adaptive,
            } => {
                let x = coords.data.rows().nth(i).unwrap();
                let y_iter = coords.pred.rows();
                let distance: Vec<f32> = y_iter.map(|y| distance_fn.call(x, y)).collect();
                kernel_fn.set_radius(
                    distance
                        .iter()
                        .max_by(|x, y| x.partial_cmp(y).unwrap())
                        .map(|x| x + 1.0)
                        .unwrap(),
                );
                for (d, w) in distance.iter().zip(weight.iter_mut()) {
                    *w *= kernel_fn.call(d);
                }
            }
            _ => panic!("cannot update weight"),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::model::{distance::TreeFn, kernel::DepthCODEmFn};

    use super::*;

    fn coords() -> Coords {
        Coords::I32(CoordsData {
            data: Matrix::new(vec![0, 1, 2], 3),
            pred: Matrix::new(vec![0, 1, 2, 0, 1, 8, 0, 6, 7, 3, 4, 5], 3),
        })
    }

    #[test]
    fn test_generic_update_weight() {
        let mut dimension = Dimension::new(
            Distance::Tree(TreeFn),
            Kernel::DepthCODEm(DepthCODEmFn::new(0.5, 3)),
            coords(),
            DimensionKind::Generic,
        );
        let mut my_weight: Vec<f32> = vec![1.0; 4];
        dimension.update_weight(0, &mut my_weight);
        let ok_weight = vec![0.5, 0.25, 0.25, 0.0];
        assert_eq!(my_weight, ok_weight);
    }

    #[test]
    fn test_categorical_update_weight() {
        let mut dimension = Dimension::new(
            Distance::Tree(TreeFn),
            Kernel::DepthCODEm(DepthCODEmFn::new(0.5, 3)),
            coords(),
            DimensionKind::Categorical,
        );
        let mut my_weight: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        dimension.update_weight(0, &mut my_weight);
        let ok_weight = vec![0.5, 0.25, 0.25, 0.0];
        assert_eq!(my_weight, ok_weight);
    }
}
