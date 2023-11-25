use crate::model::distance::{euclidean, tree};

pub trait Kernel {
    type CType;
    type DType;

    fn distance(&self, x: &[Self::CType], y: &[Self::CType]) -> Self::DType;
    fn kernel_from_distance(&self, d: &Self::DType) -> f32;
    fn kernel(&self, x: &[Self::CType], y: &[Self::CType]) -> f32 {
        self.kernel_from_distance(&self.distance(x, y))
    }
}

pub struct Exponential {
    pub radius: f32,
}
impl Exponential {
    pub fn new(radius: f32) -> Self {
        Self { radius }
    }
}
impl Kernel for Exponential {
    type CType = f32;
    type DType = f32;

    #[inline]
    fn distance(&self, x: &[Self::CType], y: &[Self::CType]) -> Self::DType {
        euclidean(x, y)
    }

    #[inline]
    fn kernel_from_distance(&self, d: &Self::DType) -> f32 {
        (-(d / self.radius)).exp()
    }
}

pub struct Tricubic {
    pub radius: f32,
    pub exponent: f32,
}
impl Tricubic {
    pub fn new(radius: f32, exponent: f32) -> Self {
        Self { radius, exponent }
    }
}
impl Kernel for Tricubic {
    type CType = f32;
    type DType = f32;

    #[inline]
    fn distance(&self, x: &[Self::CType], y: &[Self::CType]) -> Self::DType {
        euclidean(x, y)
    }

    #[inline]
    fn kernel_from_distance(&self, d: &Self::DType) -> f32 {
        let x = 1.0 - (d / self.radius).powf(self.exponent);
        x * x * x
    }
}

pub struct DepthCODEm {
    pub radius: f32,
    pub maxlvl: i32,
    one_minus_radius: f32,
    maxlvl_minus_one: i32,
}

impl DepthCODEm {
    pub fn new(radius: f32, maxlvl: i32) -> Self {
        let one_minus_radius = 1.0 - radius;
        let maxlvl_minus_one = maxlvl - 1;
        Self {
            radius,
            maxlvl,
            one_minus_radius,
            maxlvl_minus_one,
        }
    }
}
impl Kernel for DepthCODEm {
    type CType = i32;
    type DType = i32;

    #[inline]
    fn distance(&self, x: &[Self::CType], y: &[Self::CType]) -> Self::DType {
        tree(x, y)
    }

    #[inline]
    fn kernel_from_distance(&self, d: &Self::DType) -> f32 {
        if d >= &self.maxlvl {
            0.0
        } else {
            let mut result: f32 = (0..*d).map(|_| &self.one_minus_radius).product();
            if d < &self.maxlvl_minus_one {
                result *= &self.radius;
            }
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential() {
        let kernel = Exponential::new(1.0);

        let my_weight = kernel.kernel_from_distance(&1.0);
        let ok_weight = (-1.0_f32).exp();
        assert_eq!(my_weight, ok_weight);
    }

    #[test]
    fn test_tricubic() {
        let kernel = Tricubic::new(4.0, 0.5);

        let my_weight = kernel.kernel_from_distance(&1.0);
        let ok_weight = 0.125_f32;
        assert_eq!(my_weight, ok_weight);
    }

    #[test]
    fn test_depth_codem() {
        let kenerl = DepthCODEm::new(0.5, 3);
        let distance = vec![0, 1, 2, 3];

        let my_weight: Vec<f32> = distance
            .iter()
            .map(|d| kenerl.kernel_from_distance(d))
            .collect();
        let ok_weight: Vec<f32> = vec![0.5, 0.25, 0.25, 0.0];
        assert_eq!(my_weight, ok_weight);
    }
}
