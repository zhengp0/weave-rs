use weavers::config::Config;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config = Config::from_file(&args[1]).expect("have trouble loading the file");

    println!(
        "datasets: data: {}, pred: {}",
        config.datasets.data.display(),
        config.datasets.pred.display(),
    );
}
