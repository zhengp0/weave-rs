use core::convert::From;
use std::fmt::Display;

pub trait Cast<T>: Sized {
    fn from(value: T) -> Self;
}

// convert bool
impl Cast<bool> for bool {
    fn from(value: bool) -> bool {
        value
    }
}

impl Cast<i32> for bool {
    fn from(value: i32) -> bool {
        value != 0
    }
}

impl Cast<i64> for bool {
    fn from(value: i64) -> bool {
        value != 0
    }
}

impl Cast<f32> for bool {
    fn from(value: f32) -> bool {
        value != 0.0
    }
}

impl Cast<f64> for bool {
    fn from(value: f64) -> bool {
        value != 0.0
    }
}

// convert i32
impl Cast<bool> for i32 {
    fn from(value: bool) -> i32 {
        <i32 as From<bool>>::from(value)
    }
}

impl Cast<i32> for i32 {
    fn from(value: i32) -> i32 {
        value
    }
}

impl Cast<i64> for i32 {
    fn from(value: i64) -> i32 {
        value as i32
    }
}

impl Cast<f32> for i32 {
    fn from(value: f32) -> i32 {
        value as i32
    }
}

impl Cast<f64> for i32 {
    fn from(value: f64) -> i32 {
        value as i32
    }
}

// convert i64
impl Cast<bool> for i64 {
    fn from(value: bool) -> i64 {
        <i64 as From<bool>>::from(value)
    }
}

impl Cast<i32> for i64 {
    fn from(value: i32) -> i64 {
        <i64 as From<i32>>::from(value)
    }
}

impl Cast<i64> for i64 {
    fn from(value: i64) -> i64 {
        value
    }
}

impl Cast<f32> for i64 {
    fn from(value: f32) -> i64 {
        value as i64
    }
}

impl Cast<f64> for i64 {
    fn from(value: f64) -> i64 {
        value as i64
    }
}

// convert f32
impl Cast<bool> for f32 {
    fn from(value: bool) -> f32 {
        <f32 as From<bool>>::from(value)
    }
}

impl Cast<i32> for f32 {
    fn from(value: i32) -> f32 {
        value as f32
    }
}

impl Cast<i64> for f32 {
    fn from(value: i64) -> f32 {
        value as f32
    }
}

impl Cast<f32> for f32 {
    fn from(value: f32) -> f32 {
        value
    }
}

impl Cast<f64> for f32 {
    fn from(value: f64) -> f32 {
        value as f32
    }
}

// convert f64
impl Cast<bool> for f64 {
    fn from(value: bool) -> f64 {
        <f64 as From<bool>>::from(value)
    }
}

impl Cast<i32> for f64 {
    fn from(value: i32) -> f64 {
        <f64 as From<i32>>::from(value)
    }
}

impl Cast<i64> for f64 {
    fn from(value: i64) -> f64 {
        value as f64
    }
}

impl Cast<f32> for f64 {
    fn from(value: f32) -> f64 {
        <f64 as From<f32>>::from(value)
    }
}

impl Cast<f64> for f64 {
    fn from(value: f64) -> f64 {
        value
    }
}

pub trait Number:
    Cast<bool> + Cast<i32> + Cast<i64> + Cast<f32> + Cast<f64> + Clone + Copy + Display
{
}

impl<T> Number for T where
    T: Cast<bool> + Cast<i32> + Cast<i64> + Cast<f32> + Cast<f64> + Clone + Copy + Display
{
}
