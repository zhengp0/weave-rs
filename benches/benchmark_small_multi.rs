use criterion::{criterion_group, criterion_main, Criterion};
use std::sync::{Arc, Mutex};
use weavers::{config::WeaveBuilder, threadpool::TaskManager};

pub fn criterion_benchmark(c: &mut Criterion) {
    let path = "benches/small/config.toml".to_string();
    let builder = WeaveBuilder::from_file(&path).unwrap();
    let weave = builder.build();

    let source = Arc::new(weave);
    let result = Arc::new(Mutex::new(vec![0.0_f32; source.lens.1]));

    let mut group = c.benchmark_group("weave-compute_weighted_avg");
    group.sample_size(10);
    group.bench_function("small_multi", |b| {
        b.iter(|| {
            let manager = TaskManager::new(5, &source, &result);
            manager.run();
        })
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
