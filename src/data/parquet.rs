use crate::{
    data::types::{Matrix, Number},
    error::{ColumnError, Result},
};
use parquet::{
    file::reader::{self, FileReader},
    schema::types::Type,
};
use std::{collections::HashMap, fs, sync::Arc};

pub struct ParquetFileReader(reader::SerializedFileReader<fs::File>);

impl ParquetFileReader {
    pub fn new(path: &str) -> Result<Self> {
        let file = fs::File::open(path)?;
        Ok(Self(reader::SerializedFileReader::new(file)?))
    }

    pub fn nrow(&self) -> usize {
        self.0.metadata().file_metadata().num_rows() as usize
    }

    pub fn cols(&self) -> impl Iterator<Item = &str> {
        self.0
            .metadata()
            .file_metadata()
            .schema()
            .get_fields()
            .iter()
            .map(|field| field.name())
    }

    fn build_projection(&self, cols: &[String]) -> Result<Type> {
        let schema = self.0.metadata().file_metadata().schema();
        let basic_info = schema.get_basic_info().clone();
        let mut field_map: HashMap<&str, Arc<Type>> = schema
            .get_fields()
            .iter()
            .map(|field| (field.name(), field.clone()))
            .collect();
        let mut fields: Vec<Arc<Type>> = Vec::new();
        for col in cols {
            let field = field_map
                .remove(col.as_str())
                .ok_or(ColumnError::ColumnMissing(col.clone()))?;
            fields.push(field);
        }
        Ok(Type::GroupType { basic_info, fields })
    }

    pub fn read_cols<T: Number>(&self, cols: &[String]) -> Result<Matrix<T>> {
        let projection = self.build_projection(cols)?;
        for field in projection.get_fields() {
            if field.get_physical_type() != T::physical_type() {
                return Err(Box::new(ColumnError::TypeMismatch(field.name().to_owned())));
            }
        }
        let vec = self
            .0
            .get_row_iter(Some(projection))?
            .flat_map(|row| {
                row.expect("err getting row")
                    .get_column_iter()
                    .map(|(_, field)| T::from_field(field).expect("err converting field"))
                    .collect::<Vec<_>>()
            })
            .collect();
        Ok(Matrix::new(vec, cols.len()))
    }
}
