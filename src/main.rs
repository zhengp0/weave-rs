use weavers::config::Config;
use weavers::data::{read_parquet_cols, read_parquet_nrow};

fn main() {
    // TODO: hanle command line argument more elegantly
    let args: Vec<String> = std::env::args().collect();
    let config = Config::from_file(&args[1]).expect("have trouble loading the file");

    println!(
        "datasets: data: {}, pred: {}",
        config.datasets.data.display(),
        config.datasets.pred.display(),
    );

    let coord =
        read_parquet_cols::<_, i32>(&config.datasets.data, &config.dimensions[2].as_inner().key)
            .unwrap();
    println!("{:?}", coord);

    let nrow = read_parquet_nrow(&config.datasets.data).unwrap();
    println!("number of rows is {}", nrow);
}
