use std::{
    sync::{mpsc, Arc, Mutex},
    thread,
};

use crate::model::Weave;

pub struct TaskManager {
    workers: Vec<Worker>,
    tx: Option<mpsc::Sender<usize>>,
    num_tasks: usize,
}

impl TaskManager {
    pub fn new(size: usize, source: &Arc<Weave>, result: &Arc<Mutex<Vec<f32>>>) -> Self {
        assert!(size > 0);

        let (tx_parent, rx_child) = mpsc::channel();
        let rx_child = Arc::new(Mutex::new(rx_child));

        let workers: Vec<Worker> = (0..size)
            .map(|_| {
                Worker::new(
                    Arc::clone(&rx_child),
                    Arc::clone(&source),
                    Arc::clone(&result),
                )
            })
            .collect();

        Self {
            workers,
            tx: Some(tx_parent),
            num_tasks: source.lens.1,
        }
    }

    pub fn execute(&self, i: usize) {
        self.tx.as_ref().unwrap().send(i).unwrap();
    }

    pub fn join(mut self) {
        drop(self.tx.take());

        for worker in &mut self.workers {
            if let Some(handle) = worker.handle.take() {
                handle.join().unwrap();
            }
        }
    }

    pub fn run(self) {
        for i in 0..self.num_tasks {
            self.execute(i);
        }
        self.join()
    }
}

struct Worker {
    handle: Option<thread::JoinHandle<()>>,
}

impl Worker {
    fn new(
        rx: Arc<Mutex<mpsc::Receiver<usize>>>,
        source: Arc<Weave>,
        result: Arc<Mutex<Vec<f32>>>,
    ) -> Self {
        let handle = thread::spawn(move || loop {
            let message = rx.lock().unwrap().recv();

            match message {
                Ok(i) => {
                    let x = source.compute_weighted_avg_for(i);
                    *result.lock().unwrap().iter_mut().nth(i).unwrap() = x;
                }
                Err(_) => {
                    break;
                }
            }
        });

        Self {
            handle: Some(handle),
        }
    }
}
