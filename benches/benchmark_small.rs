use criterion::{criterion_group, criterion_main, Criterion};
use weavers::config::Config;

pub fn criterion_benchmark(c: &mut Criterion) {
    let path = "benches/small/config.toml".to_string();
    let config = Config::from_file(&path).unwrap();
    let mut weave = config.into_weave();

    let mut group = c.benchmark_group("weave-compute_weighted_avg");
    group.sample_size(10);
    group.bench_function("small", |b| b.iter(|| weave.compute_weighted_avg()));
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
