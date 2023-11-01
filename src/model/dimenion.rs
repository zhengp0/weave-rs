use crate::model::{distance::Distance, kernel::Kernel};
use std::{collections::HashMap, hash::Hash};

struct Coords<T> {
    data: Vec<Vec<T>>,
    pred: Vec<Vec<T>>,
}

pub struct Dimension<D, K>
where
    D: Distance,
    K: Kernel<Input = D::Output>,
{
    pub name: String,
    distance: D,
    kernel: K,
    coords: Coords<D::Input>,
}

impl<D, K> Dimension<D, K>
where
    D: Distance,
    K: Kernel<Input = D::Output>,
{
    pub fn update_weight(&self, i: usize, weight: &mut Vec<f32>) {
        let x = &self.coords.data[i];
        let y_iter = self.coords.pred.iter();
        for (y, w) in y_iter.zip(weight.iter_mut()) {
            *w *= self.kernel.call(&self.distance.call(x, y));
        }
    }
}

impl<D, K> Dimension<D, K>
where
    D: Distance,
    K: Kernel<Input = D::Output>,
    D::Output: Eq + Hash + Copy,
{
    pub fn normalize_and_update_weight(&self, i: usize, weight: &mut Vec<f32>) {
        let x = &self.coords.data[i];
        let y_iter = self.coords.pred.iter();
        let mut norm_map = HashMap::new();
        let distance: Vec<D::Output> = y_iter.map(|y| self.distance.call(x, y)).collect();

        for (d, w) in distance.iter().zip(weight.iter()) {
            norm_map.entry(d).and_modify(|x| *x += w).or_insert(*w);
        }
        for (d, w) in distance.iter().zip(weight.iter_mut()) {
            *w *= self.kernel.call(d) / &norm_map[d];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{distance::Tree, kernel::DepthCODEm};

    fn dimension() -> Dimension<Tree, DepthCODEm> {
        let name = "loc".to_string();
        let distance = Tree;
        let kernel = DepthCODEm::new(0.5, 3);
        let coords = Coords {
            data: vec![vec![0, 1, 2]],
            pred: vec![vec![0, 1, 2], vec![0, 1, 8], vec![0, 6, 7], vec![3, 4, 5]],
        };
        Dimension {
            name,
            distance,
            kernel,
            coords,
        }
    }

    #[test]
    fn test_update_weight() {
        let dimension = dimension();
        let mut my_weight: Vec<f32> = vec![1.0; 4];
        dimension.update_weight(0, &mut my_weight);
        let ok_weight = vec![0.5, 0.25, 0.25, 0.0];
        assert_eq!(my_weight, ok_weight);
    }

    #[test]
    fn test_normalize_and_update_weight() {
        let dimension = dimension();
        let mut my_weight: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        dimension.normalize_and_update_weight(0, &mut my_weight);
        let ok_weight = vec![0.5, 0.25, 0.25, 0.0];
        assert_eq!(my_weight, ok_weight);
    }
}
