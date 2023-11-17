use crate::{
    config::WeaveBuilder,
    data::{io::write_parquet_col, types::AtomicF32},
    error::Result,
    model::Weave,
};
use crossbeam_utils::thread;
use std::sync::{atomic::Ordering, mpsc, Mutex};

#[derive(Default)]
pub struct Application {
    pub model: Option<Weave>,
}

impl Application {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load_model(mut self, path: &str) -> Result<Self> {
        if self.model.is_none() {
            self.model = Some(WeaveBuilder::from_toml(path)?.build());
        }
        Ok(self)
    }

    pub fn avg_single_thread(&self) -> Vec<f32> {
        let weave = self.model.as_ref().unwrap();
        (0..weave.lens.1).map(|i| weave.avg_for(i)).collect()
    }

    pub fn avg_multi_thread(&self, num_threads: usize) -> Vec<f32> {
        let weave = self.model.as_ref().unwrap();
        let result: Vec<AtomicF32> = (0..weave.lens.1)
            .map(|_| AtomicF32::new(0.0 as f32))
            .collect();
        let (tx, rx) = mpsc::channel::<usize>();
        let rx = Mutex::new(rx);
        thread::scope(|scope| {
            for _ in 0..num_threads {
                scope.spawn(|_| loop {
                    let message = rx.lock().unwrap().recv();
                    match message {
                        Ok(i) => {
                            result[i].store(weave.avg_for(i), Ordering::Relaxed);
                        }
                        Err(_) => {
                            break;
                        }
                    }
                });
            }
            for i in 0..weave.lens.1 {
                tx.send(i).unwrap();
            }
            drop(tx);
        })
        .unwrap();

        result
            .into_iter()
            .map(|v| v.load(Ordering::Relaxed))
            .collect()
    }

    pub fn run(&self, num_threads: usize) -> Result<()> {
        let result = if num_threads > 0 {
            self.avg_single_thread()
        } else {
            self.avg_multi_thread(num_threads)
        };
        let weave = self.model.as_ref().unwrap();
        write_parquet_col(&weave.output.path, &weave.output.values, &result)?;
        Ok(())
    }
}
