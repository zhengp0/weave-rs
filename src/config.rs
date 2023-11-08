use serde::Deserialize;
use std::{error, fs};
use toml;

use crate::{
    data::{read_parquet_col, read_parquet_cols, read_parquet_nrow},
    model::{
        dimenion::{Coords, CoordsData, Dimension, DimensionKind},
        distance::Distance,
        kernel::Kernel,
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
        Weave::new(dimensions, values, lens)
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
    pub kernel: Kernel,
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
        Dimension::new(self.distance, self.kernel, coords, self.kind)
    }
}
