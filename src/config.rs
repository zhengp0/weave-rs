use serde::Deserialize;
use std::fs;
use toml;

use crate::{
    data::{parquet::ParquetFileReader, types::Matrix},
    error::Result,
    model::{
        dimenion::{Dimension, DimensionHandle},
        kernel::{Exponential, Leveled, Tricubic},
        Weave,
    },
};

#[derive(Deserialize)]
pub struct WeaveBuilder {
    pub input: Input,
    pub output: Output,
    pub dimensions: Vec<DimensionBuilder>,
}

impl WeaveBuilder {
    pub fn from_toml(path: &str) -> Result<WeaveBuilder> {
        let file = fs::read_to_string(path)?;
        let builder = toml::from_str(&file)?;
        Ok(builder)
    }

    pub fn build(self) -> Weave {
        let data_reader = ParquetFileReader::new(&self.input.data.path).unwrap();
        let pred_reader = ParquetFileReader::new(&self.input.pred.path).unwrap();
        let dimensions: Vec<Dimension> = self
            .dimensions
            .into_iter()
            .map(|dim_builder| dim_builder.build(&data_reader, &pred_reader))
            .collect();
        let values = data_reader
            .read_cols::<f32>(&[self.input.data.values])
            .unwrap()
            .to_vec();
        let lens = (data_reader.nrow(), pred_reader.nrow());
        Weave::new(dimensions, values, lens, self.output)
    }
}

#[derive(Deserialize)]
pub struct Input {
    pub data: InputData,
    pub pred: InputPred,
}

#[derive(Deserialize)]
pub struct InputData {
    pub path: String,
    pub values: String,
}

#[derive(Deserialize)]
pub struct InputPred {
    pub path: String,
}

#[derive(Deserialize)]
pub struct Output {
    pub path: String,
    pub values: String,
}

#[derive(Deserialize)]
pub struct ExponentialBuilder {
    radius: f32,
}
impl ExponentialBuilder {
    pub fn build(self) -> Exponential {
        Exponential::new(self.radius)
    }
}

#[derive(Deserialize)]
pub struct TricubicBuilder {
    radius: Option<f32>,
    exponent: f32,
}
impl TricubicBuilder {
    fn build(self, coord_data: &Matrix<f32>, coord_pred: &Matrix<f32>) -> Tricubic {
        let radius = match self.radius {
            Some(x) => x,
            None => {
                let (data_min, data_max) = partialord_min_max(&coord_data);
                let (pred_min, pred_max) = partialord_min_max(&coord_pred);
                let (diff0, diff1) = (data_max - pred_min, pred_max - data_min);
                if diff0 > diff1 {
                    diff0 + 1.0
                } else {
                    diff1 + 1.0
                }
            }
        };
        Tricubic::new(radius, self.exponent)
    }
}

#[derive(Deserialize)]
pub struct LeveledBuilder {
    radius: f32,
}
impl LeveledBuilder {
    pub fn build(self, maxlvl: i32) -> Leveled {
        Leveled::new(self.radius, maxlvl)
    }
}

#[derive(Deserialize)]
#[serde(tag = "kind")]
pub enum DimensionBuilder {
    GenericExponential {
        kernel: ExponentialBuilder,
        coord: Vec<String>,
    },
    GenericTricubic {
        kernel: TricubicBuilder,
        coord: Vec<String>,
    },
    GenericLeveled {
        kernel: LeveledBuilder,
        coord: Vec<String>,
    },
    CategoricalLeveled {
        kernel: LeveledBuilder,
        coord: Vec<String>,
    },
    AdaptiveTricubic {
        kernel: TricubicBuilder,
        coord: Vec<String>,
    },
}

impl DimensionBuilder {
    pub fn build(
        self,
        data_reader: &ParquetFileReader,
        pred_reader: &ParquetFileReader,
    ) -> Dimension {
        match self {
            Self::GenericExponential { kernel, coord } => {
                let coord_data = data_reader.read_cols::<f32>(&coord).unwrap();
                let coord_pred = pred_reader.read_cols::<f32>(&coord).unwrap();
                let kernel = kernel.build();
                Dimension::GenericExponential(DimensionHandle::new(kernel, coord_data, coord_pred))
            }
            Self::GenericTricubic { kernel, coord } => {
                let coord_data = data_reader.read_cols::<f32>(&coord).unwrap();
                let coord_pred = pred_reader.read_cols::<f32>(&coord).unwrap();
                let kernel = kernel.build(&coord_data, &coord_pred);
                Dimension::GenericTricubic(DimensionHandle::new(kernel, coord_data, coord_pred))
            }
            Self::GenericLeveled { kernel, coord } => {
                let coord_data = data_reader.read_cols::<i32>(&coord).unwrap();
                let coord_pred = pred_reader.read_cols::<i32>(&coord).unwrap();
                let kernel = kernel.build(coord_data.ncols as i32);
                Dimension::GenericLeveled(DimensionHandle::new(kernel, coord_data, coord_pred))
            }
            Self::CategoricalLeveled { kernel, coord } => {
                let coord_data = data_reader.read_cols::<i32>(&coord).unwrap();
                let coord_pred = pred_reader.read_cols::<i32>(&coord).unwrap();
                let kernel = kernel.build(coord_data.ncols as i32);
                Dimension::CategoricalLeveled(DimensionHandle::new(kernel, coord_data, coord_pred))
            }
            Self::AdaptiveTricubic { kernel, coord } => {
                let coord_data = data_reader.read_cols::<f32>(&coord).unwrap();
                let coord_pred = pred_reader.read_cols::<f32>(&coord).unwrap();
                let kernel = kernel.build(&coord_data, &coord_pred);
                Dimension::AdaptiveTricubic(DimensionHandle::new(kernel, coord_data, coord_pred))
            }
        }
    }
}

fn partialord_min_max<T: PartialOrd>(coords: &Matrix<T>) -> (&T, &T) {
    let mut coords_iter = coords.vec.iter();
    let first = coords_iter.next().unwrap();
    let mut min_max = (first, first);

    coords_iter.for_each(|x| {
        if x < min_max.0 {
            min_max.0 = x;
        }
        if x > min_max.1 {
            min_max.1 = x;
        }
    });

    min_max
}
