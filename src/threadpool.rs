use std::{
    sync::{mpsc, Arc, Mutex},
    thread,
};

use crate::model::Weave;

pub struct TaskManager {
    workers: Vec<Worker>,
    tx: Option<mpsc::Sender<usize>>,
}

impl TaskManager {
    pub fn new(size: usize, source: &Arc<Weave>, result: &Arc<Mutex<Vec<f32>>>) -> Self {
        assert!(size > 0);

        let (tx_parent, rx_child) = mpsc::channel();
        let rx_child = Arc::new(Mutex::new(rx_child));

        let workers: Vec<Worker> = (0..size)
            .map(|id| {
                Worker::new(
                    id,
                    Arc::clone(&rx_child),
                    Arc::clone(&source),
                    Arc::clone(&result),
                )
            })
            .collect();

        Self {
            workers,
            tx: Some(tx_parent),
        }
    }

    pub fn execute(&self, i: usize) {
        self.tx.as_ref().unwrap().send(i).unwrap();
    }

    pub fn join(mut self) {
        drop(self.tx.take());

        for worker in &mut self.workers {
            println!("Shutting down worker {}", worker.id);
            if let Some(handle) = worker.handle.take() {
                handle.join().unwrap();
            }
        }
    }
}

struct Worker {
    id: usize,
    handle: Option<thread::JoinHandle<()>>,
}

impl Worker {
    fn new(
        id: usize,
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
                    // println!("{:?}", result);
                    // println!("worker id: {}, i: {}, x: {}", id, i, x);
                }
                Err(_) => {
                    // println!("Worker {id} disconnected");
                    break;
                }
            }
        });

        Self {
            id,
            handle: Some(handle),
        }
    }
}
