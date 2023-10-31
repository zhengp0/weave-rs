pub trait Kernel {
    type Input;

    fn call(&self, d: &Self::Input) -> f32;
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
    type Input = f32;

    fn call(&self, d: &Self::Input) -> f32 {
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
    type Input = f32;

    fn call(&self, d: &Self::Input) -> f32 {
        (1.0 - (d / self.radius).powf(self.exponent)).powi(3)
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
    type Input = i32;

    fn call(&self, d: &Self::Input) -> f32 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential() {
        let kernel_fn = Exponential::new(1.0);

        let my_weight = kernel_fn.call(&1.0);
        let ok_weight = (-1.0_f32).exp();
        assert_eq!(my_weight, ok_weight);
    }

    #[test]
    fn test_tricubic() {
        let kernel_fn = Tricubic::new(4.0, 0.5);

        let my_weight = kernel_fn.call(&1.0);
        let ok_weight = 0.125_f32;
        assert_eq!(my_weight, ok_weight);
    }

    #[test]
    fn test_depth_codem() {
        let kenerl_fn = DepthCODEm::new(0.5, 3);
        let distance = vec![0, 1, 2, 3];

        let my_weight: Vec<f32> = distance.iter().map(|d| kenerl_fn.call(d)).collect();
        let ok_weight: Vec<f32> = vec![0.5, 0.25, 0.25, 0.0];
        assert_eq!(my_weight, ok_weight);
    }
}
