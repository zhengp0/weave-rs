use crate::{
    config::WeaveBuilder, data::io::write_parquet_col, error::Result, model::Weave,
    threadpool::TaskManager,
};
use std::sync::{Arc, Mutex};

#[derive(Default)]
pub struct Application {
    pub model: Option<Arc<Weave>>,
}

impl Application {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load_model(mut self, path: &str) -> Result<Self> {
        if self.model.is_none() {
            self.model = Some(Arc::new(WeaveBuilder::from_toml(path)?.build()));
        }
        Ok(self)
    }

    pub fn avg_single_thread(&self) -> Vec<f32> {
        let weave = self.model.as_ref().unwrap();
        (0..weave.lens.1).map(|i| weave.avg_for(i)).collect()
    }

    pub fn ave_multi_thread(&self, num_threads: usize) -> Vec<f32> {
        let weave = self.model.as_ref().unwrap();
        let result = Arc::new(Mutex::new(vec![0.0_f32; weave.lens.1]));
        let manager = TaskManager::new(num_threads, weave, &result);

        for i in 0..weave.lens.1 {
            manager.execute(i);
        }
        manager.join();

        let result = result.lock().unwrap().to_vec();
        result
    }

    pub fn run(&self, num_threads: usize) -> Result<()> {
        let result = if num_threads > 0 {
            self.avg_single_thread()
        } else {
            self.ave_multi_thread(num_threads)
        };
        let weave = self.model.as_ref().unwrap();
        write_parquet_col(&weave.output.path, &weave.output.values, &result)?;
        Ok(())
    }
}
