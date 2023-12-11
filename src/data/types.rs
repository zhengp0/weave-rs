use parquet::basic::Type;
use parquet::data_type::{BoolType, DataType, DoubleType, FloatType, Int32Type, Int64Type};
use parquet::record::Field;
use std::{
    fmt::{Debug, Display},
    slice::Chunks,
    sync::atomic::{AtomicU32, Ordering},
};

// parquet interface
pub trait ParquetDataType: Sized {
    type D: DataType<T = Self>;

    fn physical_type() -> Type {
        Self::D::get_physical_type()
    }

    fn from_field(field: &Field) -> Option<Self>;
}

impl ParquetDataType for bool {
    type D = BoolType;
    fn from_field(field: &Field) -> Option<Self> {
        match field {
            Field::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

impl ParquetDataType for i32 {
    type D = Int32Type;
    fn from_field(field: &Field) -> Option<Self> {
        match field {
            Field::Int(v) => Some(*v),
            _ => None,
        }
    }
}

impl ParquetDataType for i64 {
    type D = Int64Type;
    fn from_field(field: &Field) -> Option<Self> {
        match field {
            Field::Long(v) => Some(*v),
            _ => None,
        }
    }
}

impl ParquetDataType for f32 {
    type D = FloatType;
    fn from_field(field: &Field) -> Option<Self> {
        match field {
            Field::Float(v) => Some(*v),
            _ => None,
        }
    }
}

impl ParquetDataType for f64 {
    type D = DoubleType;
    fn from_field(field: &Field) -> Option<Self> {
        match field {
            Field::Double(v) => Some(*v),
            _ => None,
        }
    }
}

pub trait Number: Clone + Copy + Default + Display + ParquetDataType {}

impl<T> Number for T where T: Clone + Copy + Default + Display + ParquetDataType {}

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

    pub fn to_vec(self) -> Vec<T> {
        self.vec
    }
}

pub struct AtomicF32(AtomicU32);
impl AtomicF32 {
    pub fn new(val: f32) -> Self {
        Self(AtomicU32::new(val.to_bits()))
    }
    pub fn load(&self, order: Ordering) -> f32 {
        f32::from_bits(self.0.load(order))
    }
    pub fn store(&self, val: f32, order: Ordering) {
        self.0.store(val.to_bits(), order)
    }
}
impl Debug for AtomicF32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.load(Ordering::Relaxed), f)
    }
}
