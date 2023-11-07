use serde::Deserialize;
use std::convert::From;

#[derive(Deserialize)]
#[serde(tag = "kind")]
pub enum Kernel {
    Exponential(ExponentialFn),
    Tricubic(TricubicFn),
    DepthCODEm(DepthCODEmFn),
}

#[derive(Deserialize)]
pub struct ExponentialFn {
    pub radius: f32,
}
impl ExponentialFn {
    pub fn new(radius: f32) -> Self {
        Self { radius }
    }
    pub fn call(&self, d: &f32) -> f32 {
        (-(d / self.radius)).exp()
    }
}

#[derive(Deserialize)]
pub struct TricubicFn {
    pub radius: f32,
    pub exponent: f32,
}
impl TricubicFn {
    pub fn new(radius: f32, exponent: f32) -> Self {
        Self { radius, exponent }
    }
    pub fn call(&self, d: &f32) -> f32 {
        (1.0 - (d / self.radius).powf(self.exponent)).powi(3)
    }
}

#[derive(Deserialize)]
#[serde(from = "DepthCODEmFnNewArgs")]
pub struct DepthCODEmFn {
    pub radius: f32,
    pub maxlvl: i32,
    one_minus_radius: f32,
    maxlvl_minus_one: i32,
}

impl DepthCODEmFn {
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
    pub fn call(&self, d: &i32) -> f32 {
        let d = *d;
        if d >= self.maxlvl {
            0.0
        } else if d == self.maxlvl_minus_one {
            self.one_minus_radius.powi(d)
        } else {
            self.one_minus_radius.powi(d) * self.radius
        }
    }
}

#[derive(Deserialize)]
struct DepthCODEmFnNewArgs {
    radius: f32,
    maxlvl: i32,
}

impl From<DepthCODEmFnNewArgs> for DepthCODEmFn {
    fn from(value: DepthCODEmFnNewArgs) -> Self {
        let DepthCODEmFnNewArgs { radius, maxlvl } = value;
        Self::new(radius, maxlvl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential() {
        let kernel_fn = ExponentialFn::new(1.0);

        let my_weight = kernel_fn.call(&1.0);
        let ok_weight = (-1.0_f32).exp();
        assert_eq!(my_weight, ok_weight);
    }

    #[test]
    fn test_tricubic() {
        let kernel_fn = TricubicFn::new(4.0, 0.5);

        let my_weight = kernel_fn.call(&1.0);
        let ok_weight = 0.125_f32;
        assert_eq!(my_weight, ok_weight);
    }

    #[test]
    fn test_depth_codem() {
        let kenerl_fn = DepthCODEmFn::new(0.5, 3);
        let distance = vec![0, 1, 2, 3];

        let my_weight: Vec<f32> = distance.iter().map(|d| kenerl_fn.call(d)).collect();
        let ok_weight: Vec<f32> = vec![0.5, 0.25, 0.25, 0.0];
        assert_eq!(my_weight, ok_weight);
    }
}
