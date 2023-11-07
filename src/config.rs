use serde::Deserialize;
use std::{error, fs, path};
use toml;

use crate::model::{dimenion::DimensionKind, distance::Distance, kernel::Kernel};

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
pub struct DimensionConfig {
    pub kind: DimensionKind,
    pub key: Vec<String>,
    pub distance: Distance,
    pub kernel: Kernel,
}
