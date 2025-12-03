#include <benchmark/benchmark.h>
#include <numeric/math/reduce.hpp>
#include <numeric/memory/array.hpp>

static void BandwidthSum(benchmark::State &state,
                         numeric::memory::MemoryType memory_type) {
  constexpr static numeric::dim_t N = 1ll * 1024 * 1024 * 1024 / sizeof(double);
  const numeric::memory::Shape<1> shape(N);
  numeric::memory::Array<double, 1> a(shape, memory_type);
  a = 1;
  for (auto _ : state) {
    const double sum = numeric::math::sum(a);
  }
  state.SetBytesProcessed(state.iterations() * N * sizeof(double));
}

BENCHMARK_CAPTURE(BandwidthSum, Host, numeric::memory::MemoryType::HOST);
#if NUMERIC_ENABLE_HIP
BENCHMARK_CAPTURE(BandwidthSum, Pinned, numeric::memory::MemoryType::PINNED);
BENCHMARK_CAPTURE(BandwidthSum, Device, numeric::memory::MemoryType::DEVICE);
#endif
