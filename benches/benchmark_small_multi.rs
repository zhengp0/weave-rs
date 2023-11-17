use criterion::{criterion_group, criterion_main, Criterion};
use weavers::app::Application;

pub fn criterion_benchmark(c: &mut Criterion) {
    let path = "benches/small/config.toml";
    let app = Application::new().load_model(path).unwrap();

    let mut group = c.benchmark_group("weave-compute_weighted_avg");
    group.sample_size(10);
    group.bench_function("small_multi", |b| b.iter(|| app.avg_multi_thread(5)));
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
