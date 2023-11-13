use number::Number;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::{properties::WriterProperties, writer::SerializedFileWriter};
use parquet::schema::parser::parse_message_type;
use parquet::{record::Field, schema::types::Type};
use std::slice::Chunks;
use std::{collections::HashMap, error::Error, fs::File, result, sync::Arc};

pub mod number;

pub type Result<T> = result::Result<T, Box<dyn Error>>;

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
}

pub fn read_parquet_cols<T: Number>(path: &str, colnames: &[String]) -> Result<Matrix<T>> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let schema = reader.metadata().file_metadata().schema();
    let projection = build_projection(colnames, schema)?;
    let vec: Vec<T> = reader
        .get_row_iter(Some(projection))?
        .flat_map(|row| {
            row.unwrap()
                .get_column_iter()
                .map(|(_, field)| cast_field_to_number::<T>(field))
                .collect::<Vec<_>>()
        })
        .collect();
    Ok(Matrix::new(vec, colnames.len()))
}

fn build_projection(colnames: &[String], schema: &Type) -> Result<Type> {
    let basic_info = schema.get_basic_info().clone();
    let field_map: HashMap<&str, &Arc<Type>> = schema
        .get_fields()
        .iter()
        .map(|field| (field.name(), field))
        .collect();

    let mut fields = Vec::<Arc<Type>>::new();
    for colname in colnames {
        let field = field_map
            .get(colname.as_str())
            .ok_or(format!("missing column {}", colname))?;
        fields.push((*field).clone());
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

pub fn read_parquet_nrow(path: &str) -> Result<usize> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    Ok(reader.metadata().file_metadata().num_rows() as usize)
}

pub fn read_parquet_col<T: Number>(path: &str, colname: &str) -> Result<Vec<T>> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let schema = reader.metadata().file_metadata().schema();
    let projection = build_projection(&[colname.to_string()], schema)?;
    let values = reader
        .get_row_iter(Some(projection))?
        .map(|row| {
            row.unwrap()
                .get_column_iter()
                .map(|(_, field)| cast_field_to_number::<T>(field))
                .next()
                .unwrap()
        })
        .collect();
    Ok(values)
}

pub fn write_parquet_col<T: Number>(path: &str, colname: &str, values: &[T]) -> Result<()> {
    let file = File::create(path)?;
    let message_type = format!(
        "message schema {{ REQUIRED {} {}; }}",
        T::physical_type(),
        colname
    );
    let schema = Arc::new(parse_message_type(&message_type)?);
    let properties = Arc::new(WriterProperties::builder().build());
    let mut writer = SerializedFileWriter::new(file, schema, properties)?;

    for value_chunk in values.chunks(writer.properties().data_page_row_count_limit()) {
        let mut row_group_writer = writer.next_row_group()?;
        let mut col_writer = row_group_writer
            .next_column()?
            .ok_or("trouble with colunm writer")?;
        col_writer
            .typed::<T::D>()
            .write_batch(value_chunk, None, None)?;
        col_writer.close()?;
        row_group_writer.close()?;
    }
    writer.close()?;
    Ok(())
}
