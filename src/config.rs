use serde::Deserialize;
use std::{error, fs};
use toml;

use crate::{
    data::{read_parquet_col, read_parquet_cols, read_parquet_nrow, Matrix},
    model::{
        dimenion::{Coords, CoordsData, Dimension, DimensionKind},
        distance::Distance,
        kernel::{DepthCODEmFn, ExponentialFn, Kernel, TricubicFn},
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
    pub fn from_file(path: &str) -> Result<WeaveBuilder, Box<dyn error::Error>> {
        let file = fs::read_to_string(path)?;
        let builder = toml::from_str(&file)?;
        Ok(builder)
    }

    pub fn build(self) -> Weave {
        let dimensions: Vec<Dimension> = self
            .dimensions
            .into_iter()
            .map(|dim_builder| dim_builder.build(&self.input))
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
#[serde(tag = "kind")]
pub enum KernelBuilder {
    Exponential { radius: f32 },
    Tricubic(TricubicFnBuilder),
    DepthCODEm(DepthCODEmFnBuilder),
}

impl KernelBuilder {
    pub fn build(self, coords: &Coords) -> Kernel {
        match self {
            Self::Exponential { radius } => Kernel::Exponential(ExponentialFn::new(radius)),
            Self::Tricubic(builder) => Kernel::Tricubic(builder.build(coords)),
            Self::DepthCODEm(builder) => Kernel::DepthCODEm(builder.build(coords)),
        }
    }
}

#[derive(Deserialize)]
pub struct TricubicFnBuilder {
    radius: Option<f32>,
    exponent: f32,
}

impl TricubicFnBuilder {
    fn build(self, coords: &Coords) -> TricubicFn {
        let radius = match self.radius {
            Some(x) => x,
            None => {
                let coords_data = match coords {
                    Coords::F32(inner) => inner,
                    _ => panic!("wrong coords data type for Tricubic kernel"),
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
        TricubicFn::new(radius, self.exponent)
    }
}

#[derive(Deserialize)]
pub struct DepthCODEmFnBuilder {
    radius: f32,
}

impl DepthCODEmFnBuilder {
    pub fn build(self, coords: &Coords) -> DepthCODEmFn {
        let maxlvl = match coords {
            Coords::I32(inner) => inner.data.ncols as i32,
            _ => panic!("wrong coords data type for DepthCODEm kernel"),
        };
        DepthCODEmFn::new(self.radius, maxlvl)
    }
}

#[derive(Deserialize)]
pub struct DimensionBuilder {
    pub kind: DimensionKind,
    pub coords: Vec<String>,
    pub distance: Distance,
    pub kernel: KernelBuilder,
}

impl DimensionBuilder {
    pub fn build(self, input: &Input) -> Dimension {
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
        let kernel = self.kernel.build(&coords);
        Dimension::new(self.distance, kernel, coords, self.kind)
    }
}

fn coords_min_max<T: PartialOrd>(coords: &Matrix<T>) -> (&T, &T) {
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
