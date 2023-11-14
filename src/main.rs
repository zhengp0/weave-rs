use std::sync::{Arc, Mutex};
use weavers::config::Config;
use weavers::threadpool::TaskManager;

fn main() {
    // TODO: hanle command line argument more elegantly
    let args: Vec<String> = std::env::args().collect();
    let config = Config::from_file(&args[1]).expect("have trouble loading the file");
    let weave = config.into_weave();

    // weave.run().unwrap();

    let weave = Arc::new(weave);
    let result = Arc::new(Mutex::new(vec![0.0_f32; weave.lens.1]));

    let manager = TaskManager::new(4, &weave, &result);
    manager.run();

    let result = result.lock().unwrap();
    println!("result: {:?}", result);
}
