use core::convert::From;
use parquet::basic::Type;
use parquet::data_type::{BoolType, DataType, DoubleType, FloatType, Int32Type, Int64Type};
use std::{fmt::Display, slice::Chunks};

pub trait Cast<T>: Sized {
    fn from(value: T) -> Self;
}

// convert bool
impl Cast<bool> for bool {
    fn from(value: bool) -> Self {
        value
    }
}

impl Cast<i32> for bool {
    fn from(value: i32) -> Self {
        value != 0
    }
}

impl Cast<i64> for bool {
    fn from(value: i64) -> Self {
        value != 0
    }
}

impl Cast<f32> for bool {
    fn from(value: f32) -> Self {
        value != 0.0
    }
}

impl Cast<f64> for bool {
    fn from(value: f64) -> Self {
        value != 0.0
    }
}

// convert i32
impl Cast<bool> for i32 {
    fn from(value: bool) -> Self {
        <Self as From<bool>>::from(value)
    }
}

impl Cast<i32> for i32 {
    fn from(value: i32) -> Self {
        value
    }
}

impl Cast<i64> for i32 {
    fn from(value: i64) -> Self {
        value as Self
    }
}

impl Cast<f32> for i32 {
    fn from(value: f32) -> Self {
        value as Self
    }
}

impl Cast<f64> for i32 {
    fn from(value: f64) -> Self {
        value as Self
    }
}

// convert i64
impl Cast<bool> for i64 {
    fn from(value: bool) -> Self {
        <Self as From<bool>>::from(value)
    }
}

impl Cast<i32> for i64 {
    fn from(value: i32) -> Self {
        <Self as From<i32>>::from(value)
    }
}

impl Cast<i64> for i64 {
    fn from(value: i64) -> Self {
        value
    }
}

impl Cast<f32> for i64 {
    fn from(value: f32) -> Self {
        value as Self
    }
}

impl Cast<f64> for i64 {
    fn from(value: f64) -> Self {
        value as Self
    }
}

// convert f32
impl Cast<bool> for f32 {
    fn from(value: bool) -> Self {
        <Self as From<bool>>::from(value)
    }
}

impl Cast<i32> for f32 {
    fn from(value: i32) -> Self {
        value as Self
    }
}

impl Cast<i64> for f32 {
    fn from(value: i64) -> Self {
        value as Self
    }
}

impl Cast<f32> for f32 {
    fn from(value: f32) -> Self {
        value
    }
}

impl Cast<f64> for f32 {
    fn from(value: f64) -> Self {
        value as Self
    }
}

// convert f64
impl Cast<bool> for f64 {
    fn from(value: bool) -> Self {
        <Self as From<bool>>::from(value)
    }
}

impl Cast<i32> for f64 {
    fn from(value: i32) -> Self {
        <Self as From<i32>>::from(value)
    }
}

impl Cast<i64> for f64 {
    fn from(value: i64) -> Self {
        value as Self
    }
}

impl Cast<f32> for f64 {
    fn from(value: f32) -> Self {
        <Self as From<f32>>::from(value)
    }
}

impl Cast<f64> for f64 {
    fn from(value: f64) -> Self {
        value
    }
}

// parquet interface
pub trait ParquetDataType {
    type D: DataType<T = Self>;

    fn physical_type() -> Type {
        Self::D::get_physical_type()
    }
}

impl ParquetDataType for bool {
    type D = BoolType;
}

impl ParquetDataType for i32 {
    type D = Int32Type;
}

impl ParquetDataType for i64 {
    type D = Int64Type;
}

impl ParquetDataType for f32 {
    type D = FloatType;
}

impl ParquetDataType for f64 {
    type D = DoubleType;
}

pub trait Number:
    Cast<bool>
    + Cast<i32>
    + Cast<i64>
    + Cast<f32>
    + Cast<f64>
    + Clone
    + Copy
    + Display
    + ParquetDataType
{
}

impl<T> Number for T where
    T: Cast<bool>
        + Cast<i32>
        + Cast<i64>
        + Cast<f32>
        + Cast<f64>
        + Clone
        + Copy
        + Display
        + ParquetDataType
{
}

pub struct Matrix<T> {
    pub vec: Vec<T>,
    pub ncols: usize,
}

impl<T> Matrix<T> {
    pub fn new(vec: Vec<T>, ncols: usize) -> Self {
        if vec.len() % ncols != 0 {
            panic!(
                "matrix can't be created by vec.len: {}, ncols: {}",
                vec.len(),
                ncols
            );
        }
        Self { vec, ncols }
    }

    pub fn rows(&self) -> Chunks<T> {
        self.vec.chunks(self.ncols)
    }
}
