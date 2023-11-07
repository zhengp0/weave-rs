use number::Number;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::{record::Field, schema::types::Type};
use std::{collections::HashMap, error::Error, fmt::Display, fs::File, sync::Arc};

pub mod number;

type GenResult<T> = Result<T, Box<dyn Error>>;

#[derive(Debug)]
enum DataError {
    ColumnMissingError(String),
}

impl Display for DataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::ColumnMissingError(col) => format!("missing column {}", col),
        };
        write!(f, "{}", message)
    }
}

impl Error for DataError {}

pub fn read_parquet_cols<T: Number>(path: &String, cols: &Vec<String>) -> GenResult<Vec<Vec<T>>> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let schema = reader.metadata().file_metadata().schema();
    let projection = build_projection(cols, schema)?;
    let rows = reader.get_row_iter(Some(projection))?;

    let mut values = Vec::<Vec<T>>::new();
    for row in rows {
        let item: Vec<T> = row?
            .get_column_iter()
            .map(|(_, field)| cast_field_to_number::<T>(field))
            .collect();
        values.push(item);
    }

    Ok(values)
}

fn build_projection(cols: &Vec<String>, schema: &Type) -> GenResult<Type> {
    let basic_info = schema.get_basic_info().clone();
    let field_map: HashMap<&str, &Arc<Type>> = schema
        .get_fields()
        .iter()
        .map(|field| (field.name(), field))
        .collect();

    let mut fields = Vec::<Arc<Type>>::new();
    for col in cols {
        let col = col.as_str();
        if !field_map.contains_key(col) {
            return Err(Box::new(DataError::ColumnMissingError(col.to_string())));
        }
        let field = (*field_map[col]).clone();
        fields.push(field);
    }
    Ok(Type::GroupType { basic_info, fields })
}

fn cast_field_to_number<D: Number>(field: &Field) -> D {
    match field {
        Field::Bool(v) => D::from(*v),
        Field::Int(v) => D::from(*v),
        Field::Long(v) => D::from(*v),
        Field::Float(v) => D::from(*v),
        Field::Double(v) => D::from(*v),
        _ => panic!("Cannot cast {:?} to preset types", field),
    }
}

pub fn read_parquet_nrow(path: &String) -> GenResult<usize> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    Ok(reader.metadata().file_metadata().num_rows() as usize)
}
