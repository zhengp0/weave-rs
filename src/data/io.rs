use crate::{data::types::Number, error::Result};
use parquet::file::{properties::WriterProperties, writer::SerializedFileWriter};
use parquet::schema::parser::parse_message_type;
use std::{fs::File, sync::Arc};

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
