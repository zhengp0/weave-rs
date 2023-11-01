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
    pub key: Vec<String>,
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
        y_iter
            .map(|y| self.distance.call(x, y))
            .zip(weight.iter_mut())
            .for_each(|(d, w)| *w *= self.kernel.call(&d));
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
        let distance: Vec<D::Output> = y_iter
            .zip(weight.iter())
            .map(|(y, w)| {
                let d = self.distance.call(x, y);
                let s = norm_map.entry(d).or_insert(0_f32);
                *s += w;
                d
            })
            .collect();
        distance.iter().zip(weight.iter_mut()).for_each(|(d, w)| {
            let s = norm_map.get(d).unwrap();
            *w *= self.kernel.call(d) / s;
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{distance::Tree, kernel::DepthCODEm};

    fn dimension() -> Dimension<Tree, DepthCODEm> {
        let name = "loc".to_string();
        let key: Vec<String> = ["super_region_id", "region_id", "location_id"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let distance = Tree;
        let kernel = DepthCODEm::new(0.5, 3);
        let coords = Coords {
            data: vec![vec![0, 1, 2]],
            pred: vec![vec![0, 1, 2], vec![0, 1, 8], vec![0, 6, 7], vec![3, 4, 5]],
        };
        Dimension {
            name,
            key,
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
