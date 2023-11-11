use weavers::config::Config;

fn main() {
    // TODO: hanle command line argument more elegantly
    let args: Vec<String> = std::env::args().collect();
    let config = Config::from_file(&args[1]).expect("have trouble loading the file");
    let mut weave = config.into_weave();
    weave.run().unwrap();
}
