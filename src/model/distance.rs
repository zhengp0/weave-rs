#[inline]
pub fn euclidean(x: &[f32], y: &[f32]) -> f32 {
    (x[0] - y[0]).abs()
}

#[inline]
pub fn hierarchical(x: &[i32], y: &[i32]) -> i32 {
    x.iter()
        .rev()
        .zip(y.iter().rev())
        .take_while(|(xi, yi)| xi != yi)
        .count() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean() {
        let x: Vec<f32> = vec![0.0];
        let y: Vec<f32> = vec![1.0];

        let my_distance = euclidean(&x, &y);
        let ok_distance = 1.0_f32;
        assert_eq!(my_distance, ok_distance);
    }

    #[test]
    fn test_hierarchical() {
        let x = vec![0, 1, 2];
        let y_vec = vec![vec![3, 4, 5], vec![0, 6, 7], vec![0, 1, 8], vec![0, 1, 2]];

        let my_distance: Vec<i32> = y_vec.iter().map(|y| hierarchical(&x, y)).collect();
        let ok_distance: Vec<i32> = vec![3, 2, 1, 0];
        assert_eq!(my_distance, ok_distance);
    }
}
