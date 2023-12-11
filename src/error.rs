use std::{error, result};
use thiserror;

pub type Result<T> = result::Result<T, Box<dyn error::Error>>;

#[derive(thiserror::Error, Debug)]
pub enum ColumnError {
    #[error("column `{0}` is missing")]
    ColumnMissing(String),
    #[error("column `{0}`'s type does not match")]
    TypeMismatch(String),
}
