use serde::Deserialize;
use std::{error, fs};
use toml;

use crate::{
    data::{read_parquet_col, read_parquet_cols, read_parquet_nrow},
    model::{
        dimenion::{Coords, CoordsData, Dimension, DimensionKind},
        distance::Distance,
        kernel::{DepthCODEmFn, ExponentialFn, Kernel, TricubicFn},
        Weave,
    },
};

#[derive(Deserialize)]
pub struct Config {
    pub input: Input,
    pub output: Output,
    pub dimensions: Vec<DimensionConfig>,
}

impl Config {
    pub fn from_file(path: &String) -> Result<Config, Box<dyn error::Error>> {
        let file = fs::read_to_string(path)?;
        let config = toml::from_str(&file)?;
        Ok(config)
    }

    pub fn into_weave(self) -> Weave {
        let dimensions: Vec<Dimension> = self
            .dimensions
            .into_iter()
            .map(|dim_config| dim_config.into_dimension(&self.input))
            .collect();
        let values =
            read_parquet_col::<f32>(&self.input.data.path, &self.input.data.values).unwrap();
        let lens = (
            read_parquet_nrow(&self.input.data.path).unwrap(),
            read_parquet_nrow(&self.input.pred.path).unwrap(),
        );
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
pub struct DimensionConfig {
    pub kind: DimensionKind,
    pub coords: Vec<String>,
    pub distance: Distance,
    pub kernel: KernelConfig,
}

#[derive(Deserialize)]
#[serde(tag = "kind")]
pub enum KernelConfig {
    Exponential { radius: f32 },
    Tricubic(TricubicConfig),
    DepthCODEm { radius: f32 },
}

#[derive(Deserialize)]
pub struct TricubicConfig {
    radius: Option<f32>,
    exponent: f32,
}

impl DimensionConfig {
    pub fn into_dimension(self, input: &Input) -> Dimension {
        let coords = match self.distance {
            Distance::Euclidean(_) => Coords::F32(CoordsData {
                data: read_parquet_cols::<f32>(&input.data.path, &self.coords).unwrap(),
                pred: read_parquet_cols::<f32>(&input.pred.path, &self.coords).unwrap(),
            }),
            Distance::Tree(_) => Coords::I32(CoordsData {
                data: read_parquet_cols::<i32>(&input.data.path, &self.coords).unwrap(),
                pred: read_parquet_cols::<i32>(&input.pred.path, &self.coords).unwrap(),
            }),
        };
        let kernel = match self.kernel {
            KernelConfig::Exponential { radius } => Kernel::Exponential(ExponentialFn::new(radius)),
            KernelConfig::Tricubic(inner) => {
                let radius = match inner.radius {
                    Some(x) => x,
                    None => {
                        let coords_data = match &coords {
                            Coords::F32(inner) => inner,
                            _ => panic!("wrong coords data type for tricubic kernel"),
                        };
                        let (data_min, data_max) = coords_min_max(&coords_data.data);
                        let (pred_min, pred_max) = coords_min_max(&coords_data.pred);
                        let (diff0, diff1) = (data_max - pred_min, pred_max - data_min);
                        if diff0 > diff1 {
                            diff0 + 1.0
                        } else {
                            diff1 + 1.0
                        }
                    }
                };
                Kernel::Tricubic(TricubicFn::new(radius, inner.exponent))
            }
            KernelConfig::DepthCODEm { radius } => {
                let maxlvl = self.coords.len() as i32;
                Kernel::DepthCODEm(DepthCODEmFn::new(radius, maxlvl))
            }
        };
        Dimension::new(self.distance, kernel, coords, self.kind)
    }
}

fn coords_min_max<T: PartialOrd>(coords: &Vec<Vec<T>>) -> (&T, &T) {
    let mut coords_iter = coords.iter().flatten();
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
