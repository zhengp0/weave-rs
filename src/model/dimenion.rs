use crate::model::{distance::Distance, kernel::Kernel};
use std::{collections::HashMap, hash::Hash};

pub struct Coords<T> {
    pub data: Vec<Vec<T>>,
    pub pred: Vec<Vec<T>>,
}

pub struct DimensionInfo<D, K>
where
    D: Distance,
    K: Kernel<Input = D::Output>,
{
    pub distance: D,
    pub kernel: K,
    pub coords: Coords<D::Input>,
}

impl<D, K> DimensionInfo<D, K>
where
    D: Distance,
    K: Kernel<Input = D::Output>,
{
    pub fn new(distance: D, kernel: K, coords: Coords<D::Input>) -> Self {
        Self {
            distance,
            kernel,
            coords,
        }
    }
}

pub struct Generic<D, K>(pub DimensionInfo<D, K>)
where
    D: Distance,
    K: Kernel<Input = D::Output>;

pub struct Categorical<D, K>(pub DimensionInfo<D, K>)
where
    D: Distance,
    K: Kernel<Input = D::Output>,
    D::Output: Eq + Hash + Copy;

pub trait Dimension {
    fn update_weight(&self, i: usize, weight: &mut Vec<f32>);
}

impl<D, K> Dimension for Generic<D, K>
where
    D: Distance,
    K: Kernel<Input = D::Output>,
{
    fn update_weight(&self, i: usize, weight: &mut Vec<f32>) {
        let info = &self.0;
        let x = &info.coords.data[i];
        let y_iter = info.coords.pred.iter();
        for (y, w) in y_iter.zip(weight.iter_mut()) {
            *w *= self.0.kernel.call(&info.distance.call(x, y));
        }
    }
}

impl<D, K> Dimension for Categorical<D, K>
where
    D: Distance,
    K: Kernel<Input = D::Output>,
    D::Output: Eq + Hash + Copy,
{
    fn update_weight(&self, i: usize, weight: &mut Vec<f32>) {
        let info = &self.0;
        let x = &info.coords.data[i];
        let y_iter = info.coords.pred.iter();
        let mut norm_map = HashMap::new();
        let distance: Vec<D::Output> = y_iter.map(|y| info.distance.call(x, y)).collect();

        for (d, w) in distance.iter().zip(weight.iter()) {
            norm_map.entry(d).and_modify(|x| *x += w).or_insert(*w);
        }
        for (d, w) in distance.iter().zip(weight.iter_mut()) {
            *w *= info.kernel.call(d) / &norm_map[d];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{distance::Tree, kernel::DepthCODEm};

    fn info() -> DimensionInfo<Tree, DepthCODEm> {
        let distance = Tree;
        let kernel = DepthCODEm::new(0.5, 3);
        let coords = Coords {
            data: vec![vec![0, 1, 2]],
            pred: vec![vec![0, 1, 2], vec![0, 1, 8], vec![0, 6, 7], vec![3, 4, 5]],
        };
        DimensionInfo {
            distance,
            kernel,
            coords,
        }
    }

    #[test]
    fn test_generic_update_weight() {
        let dimension = Generic(info());
        let mut my_weight: Vec<f32> = vec![1.0; 4];
        dimension.update_weight(0, &mut my_weight);
        let ok_weight = vec![0.5, 0.25, 0.25, 0.0];
        assert_eq!(my_weight, ok_weight);
    }

    #[test]
    fn test_categorical_update_weight() {
        let dimension = Categorical(info());
        let mut my_weight: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        dimension.update_weight(0, &mut my_weight);
        let ok_weight = vec![0.5, 0.25, 0.25, 0.0];
        assert_eq!(my_weight, ok_weight);
    }
}
