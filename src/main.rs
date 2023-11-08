use weavers::config::Config;
use weavers::data::{read_parquet_cols, read_parquet_nrow};

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
    println!("number of rows is {}", nrow);

    let _weave = config.into_weave();
}
