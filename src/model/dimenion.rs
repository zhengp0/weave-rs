use crate::model::{distance::Distance, kernel::Kernel};
use serde::Deserialize;
use std::collections::HashMap;

pub struct CoordsData<T> {
    pub data: Vec<Vec<T>>,
    pub pred: Vec<Vec<T>>,
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
                let x = &coords.data[i];
                let y_iter = coords.pred.iter();
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
                let x = &coords.data[i];
                let y_iter = coords.pred.iter();
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
                let x = &coords.data[i];
                let y_iter = coords.pred.iter();
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
                let x = &coords.data[i];
                let y_iter = coords.pred.iter();
                let mut norm_map = HashMap::new();
                let distance: Vec<i32> = y_iter.map(|y| distance_fn.call(x, y)).collect();

                for (d, w) in distance.iter().zip(weight.iter()) {
                    norm_map.entry(d).and_modify(|x| *x += w).or_insert(*w);
                }
                for (d, w) in distance.iter().zip(weight.iter_mut()) {
                    let s = norm_map[d];
                    if s > 0.0 {
                        *w /= s;
                        *w *= kernel_fn.call(d);
                    }
                }
            }
            Self {
                distance: Distance::Euclidean(distance_fn),
                kernel: Kernel::Tricubic(kernel_fn),
                coords: Coords::F32(coords),
                kind: DimensionKind::Adaptive,
            } => {
                let x = &coords.data[i];
                let y_iter = coords.pred.iter();
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
            data: vec![vec![0, 1, 2]],
            pred: vec![vec![0, 1, 2], vec![0, 1, 8], vec![0, 6, 7], vec![3, 4, 5]],
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
