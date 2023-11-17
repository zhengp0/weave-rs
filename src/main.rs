use weavers::app::Application;

fn main() {
    // TODO: hanle command line argument more elegantly
    let args: Vec<String> = std::env::args().collect();
    let app = Application::new().load_model(&args[1]).unwrap();

    let result = app.ave_multi_thread(4);
    println!("result: {:?}", result);
}
