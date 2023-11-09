use weavers::config::Config;
use weavers::data::{read_parquet_cols, read_parquet_nrow, write_parquet_col};

fn main() {
    // TODO: hanle command line argument more elegantly
    let args: Vec<String> = std::env::args().collect();
    let config = Config::from_file(&args[1]).expect("have trouble loading the file");

    println!(
        "datasets: data: {}, pred: {}",
        config.input.data.path, config.input.pred.path,
    );

    let coord =
        read_parquet_cols::<i32>(&config.input.data.path, &config.dimensions[2].coords).unwrap();
    println!("{:?}", coord);

    let nrow = read_parquet_nrow(&config.input.data.path).unwrap();
    println!("number of rows: {}", nrow);

    let _result = write_parquet_col::<f32>(
        &config.output.path,
        &config.output.values,
        &(0..100).map(|x| x as f32).collect(),
    )
    .unwrap();

    let _weave = config.into_weave();
}
