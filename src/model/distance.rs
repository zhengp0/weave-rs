use serde::Deserialize;

#[derive(Deserialize)]
#[serde(tag = "kind")]
pub enum Distance {
    Euclidean(EuclideanFn),
    Tree(TreeFn),
}

#[derive(Deserialize)]
pub struct EuclideanFn;
impl EuclideanFn {
    pub fn call(&self, x: &[f32], y: &[f32]) -> f32 {
        (x[0] - y[0]).abs()
    }
}

#[derive(Deserialize)]
pub struct TreeFn;
impl TreeFn {
    pub fn call(&self, x: &[i32], y: &[i32]) -> i32 {
        let x_iter = x.iter().rev();
        let y_iter = y.iter().rev();
        x_iter.zip(y_iter).take_while(|(xi, yi)| xi != yi).count() as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean() {
        let distance_fn = EuclideanFn;
        let x: Vec<f32> = vec![0.0];
        let y: Vec<f32> = vec![1.0];

        let my_distance = distance_fn.call(&x, &y);
        let ok_distance = 1.0_f32;
        assert_eq!(my_distance, ok_distance);
    }

    #[test]
    fn test_tree() {
        let distance_fn = TreeFn;

        let x = vec![0, 1, 2];
        let y_vec = vec![vec![3, 4, 5], vec![0, 6, 7], vec![0, 1, 8], vec![0, 1, 2]];

        let my_distance: Vec<i32> = y_vec.iter().map(|y| distance_fn.call(&x, y)).collect();
        let ok_distance: Vec<i32> = vec![3, 2, 1, 0];
        assert_eq!(my_distance, ok_distance);
    }
}
