use serde::Deserialize;
use std::{error, fs, path};
use toml;

#[derive(Deserialize)]
pub struct Config {
    pub datasets: DatasetsConfig,
    pub datakeys: DataKeysConfig,
    pub dimensions: Vec<DimensionConfig>,
}

impl Config {
    pub fn from_file<P: AsRef<path::Path>>(path: P) -> Result<Config, Box<dyn error::Error>> {
        let file = fs::read_to_string(path)?;
        let config = toml::from_str(&file)?;
        Ok(config)
    }
}

#[derive(Deserialize)]
pub struct DatasetsConfig {
    pub data: path::PathBuf,
    pub pred: path::PathBuf,
}

#[derive(Deserialize)]
pub struct DataKeysConfig {
    pub values: String,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum DimensionConfig {
    Generic(DimensionInfoConfig),
    Categorical(DimensionInfoConfig),
}

impl DimensionConfig {
    pub fn as_inner(&self) -> &DimensionInfoConfig {
        match self {
            Self::Generic(info_config) => info_config,
            Self::Categorical(info_config) => info_config,
        }
    }
}

#[derive(Deserialize)]
pub struct DimensionInfoConfig {
    pub key: Vec<String>,
    pub distance: DistanceConfig,
    pub kernel: KernelConfig,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum DistanceConfig {
    Euclidean,
    Tree,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum KernelConfig {
    Exponential { radius: f32 },
    Tricubic { radius: f32, exponent: f32 },
    DepthCODEm { radius: f32, maxlvl: i32 },
}
