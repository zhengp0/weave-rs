use criterion::{criterion_group, criterion_main, Criterion};
use weavers::config::WeaveBuilder;

pub fn criterion_benchmark(c: &mut Criterion) {
    let path = "benches/small/config.toml".to_string();
    let builder = WeaveBuilder::from_toml(&path).unwrap();
    let weave = builder.build();

    let mut group = c.benchmark_group("weave-compute_weighted_avg");
    group.sample_size(10);
    group.bench_function("small", |b| b.iter(|| weave.compute_weighted_avg()));
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
