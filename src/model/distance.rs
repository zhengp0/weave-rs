pub trait Distance {
    type Input;
    type Output;

    fn call(&self, x: &Vec<Self::Input>, y: &Vec<Self::Input>) -> Self::Output;
}

pub struct Euclidean;

impl Distance for Euclidean {
    type Input = f32;
    type Output = f32;

    fn call(&self, x: &Vec<Self::Input>, y: &Vec<Self::Input>) -> Self::Output {
        (x[0] - y[0]).abs()
    }
}

pub struct Tree;

impl Distance for Tree {
    type Input = i32;
    type Output = i32;

    fn call(&self, x: &Vec<Self::Input>, y: &Vec<Self::Input>) -> Self::Output {
        let x_iter = x.iter().rev();
        let y_iter = y.iter().rev();
        x_iter.zip(y_iter).take_while(|(xi, yi)| xi != yi).count() as Self::Output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean() {
        let distance_fn = Euclidean;
        let x: Vec<f32> = vec![0.0];
        let y: Vec<f32> = vec![1.0];

        let my_distance = distance_fn.call(&x, &y);
        let ok_distance = 1.0_f32;
        assert_eq!(my_distance, ok_distance);
    }

    #[test]
    fn test_tree() {
        let distance_fn = Tree;

        let x = vec![0, 1, 2];
        let y_vec = vec![vec![3, 4, 5], vec![0, 6, 7], vec![0, 1, 8], vec![0, 1, 2]];

        let my_distance: Vec<i32> = y_vec.iter().map(|y| distance_fn.call(&x, y)).collect();
        let ok_distance: Vec<i32> = vec![3, 2, 1, 0];
        assert_eq!(my_distance, ok_distance);
    }
}
