use serde::{de, Deserialize};
use std::fmt;

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

impl<'de> de::Deserialize<'de> for DepthCODEmFn {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        enum Field {
            Radius,
            Maxlvl,
        }

        // This part could also be generated independently by:
        //
        //    #[derive(de::Deserialize)]
        //    #[serde(field_identifier, rename_all = "lowercase")]
        //    enum Field { Radius, Maxlvl }
        impl<'de> de::Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
            where
                D: de::Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> de::Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("`radius` or `maxlvl`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "radius" => Ok(Field::Radius),
                            "maxlvl" => Ok(Field::Maxlvl),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct DepthCODEmFnVisitor;

        impl<'de> de::Visitor<'de> for DepthCODEmFnVisitor {
            type Value = DepthCODEmFn;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct DepthCODEmFn")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<DepthCODEmFn, V::Error>
            where
                V: de::SeqAccess<'de>,
            {
                let radius = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let maxlvl = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                Ok(DepthCODEmFn::new(radius, maxlvl))
            }

            fn visit_map<V>(self, mut map: V) -> Result<DepthCODEmFn, V::Error>
            where
                V: de::MapAccess<'de>,
            {
                let mut radius = None;
                let mut maxlvl = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Radius => {
                            if radius.is_some() {
                                return Err(de::Error::duplicate_field("radius"));
                            }
                            radius = Some(map.next_value()?);
                        }
                        Field::Maxlvl => {
                            if maxlvl.is_some() {
                                return Err(de::Error::duplicate_field("maxlvl"));
                            }
                            maxlvl = Some(map.next_value()?);
                        }
                    }
                }
                let radius = radius.ok_or_else(|| de::Error::missing_field("radius"))?;
                let maxlvl = maxlvl.ok_or_else(|| de::Error::missing_field("maxlvl"))?;
                Ok(DepthCODEmFn::new(radius, maxlvl))
            }
        }

        const FIELDS: &'static [&'static str] = &["radius", "maxlvl"];
        deserializer.deserialize_struct("DepthCODEmFn", FIELDS, DepthCODEmFnVisitor)
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
