#include <benchmark/benchmark.h>
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_const_view.hpp>
#include <numeric/memory/array_op.hpp>
#include <numeric/memory/array_view.hpp>
#include <numeric/memory/copy.hpp>
#include <numeric/memory/memcpy.hpp>

static void BandwidthMemcpy(benchmark::State &state,
                            numeric::memory::MemoryType from,
                            numeric::memory::MemoryType to) {
  constexpr static numeric::dim_t N = 1ll * 1024 * 1024 * 1024 / sizeof(double);
  const numeric::memory::Shape<1> shape(N);
  numeric::memory::Array<double, 1> a(shape, from);
  numeric::memory::Array<double, 1> b(shape, to);
  a = 1;
  for (auto _ : state) {
    numeric::memory::memcpy(b, a);
  }
  state.SetBytesProcessed(state.iterations() * N * sizeof(double));
}

BENCHMARK_CAPTURE(BandwidthMemcpy, HostHost, numeric::memory::MemoryType::HOST,
                  numeric::memory::MemoryType::HOST);
#if NUMERIC_ENABLE_HIP
BENCHMARK_CAPTURE(BandwidthMemcpy, HostPinned,
                  numeric::memory::MemoryType::HOST,
                  numeric::memory::MemoryType::PINNED);
BENCHMARK_CAPTURE(BandwidthMemcpy, HostDevice,
                  numeric::memory::MemoryType::HOST,
                  numeric::memory::MemoryType::DEVICE);
BENCHMARK_CAPTURE(BandwidthMemcpy, PinnedHost,
                  numeric::memory::MemoryType::PINNED,
                  numeric::memory::MemoryType::HOST);
BENCHMARK_CAPTURE(BandwidthMemcpy, PinnedPinned,
                  numeric::memory::MemoryType::PINNED,
                  numeric::memory::MemoryType::PINNED);
BENCHMARK_CAPTURE(BandwidthMemcpy, PinnedDevice,
                  numeric::memory::MemoryType::PINNED,
                  numeric::memory::MemoryType::DEVICE);
BENCHMARK_CAPTURE(BandwidthMemcpy, DeviceHost,
                  numeric::memory::MemoryType::DEVICE,
                  numeric::memory::MemoryType::HOST);
BENCHMARK_CAPTURE(BandwidthMemcpy, DevicePinned,
                  numeric::memory::MemoryType::DEVICE,
                  numeric::memory::MemoryType::PINNED);
BENCHMARK_CAPTURE(BandwidthMemcpy, DeviceDevice,
                  numeric::memory::MemoryType::DEVICE,
                  numeric::memory::MemoryType::DEVICE);
#endif

static void BandwidthCopy(benchmark::State &state,
                          numeric::memory::MemoryType from,
                          numeric::memory::MemoryType to) {
  constexpr static numeric::dim_t N = 4ll * 1024 * 1024 / sizeof(double);
  const numeric::memory::Shape<1> shape(N);
  numeric::memory::Array<double, 1> a(shape, from);
  numeric::memory::Array<double, 1> b(shape, to);
  numeric::memory::Copyer<double, 1, numeric::memory::ArrayConstView<double, 1>>
      cpy(b, a);
  for (auto _ : state) {
    cpy(b, a);
  }
  state.SetBytesProcessed(state.iterations() * N * sizeof(double));
}

BENCHMARK_CAPTURE(BandwidthCopy, HostHost, numeric::memory::MemoryType::HOST,
                  numeric::memory::MemoryType::HOST);
#if NUMERIC_ENABLE_HIP
BENCHMARK_CAPTURE(BandwidthCopy, HostPinned, numeric::memory::MemoryType::HOST,
                  numeric::memory::MemoryType::PINNED);
BENCHMARK_CAPTURE(BandwidthCopy, HostDevice, numeric::memory::MemoryType::HOST,
                  numeric::memory::MemoryType::DEVICE);
BENCHMARK_CAPTURE(BandwidthCopy, PinnedHost,
                  numeric::memory::MemoryType::PINNED,
                  numeric::memory::MemoryType::HOST);
BENCHMARK_CAPTURE(BandwidthCopy, PinnedPinned,
                  numeric::memory::MemoryType::PINNED,
                  numeric::memory::MemoryType::PINNED);
BENCHMARK_CAPTURE(BandwidthCopy, PinnedDevice,
                  numeric::memory::MemoryType::PINNED,
                  numeric::memory::MemoryType::DEVICE);
BENCHMARK_CAPTURE(BandwidthCopy, DeviceHost,
                  numeric::memory::MemoryType::DEVICE,
                  numeric::memory::MemoryType::HOST);
BENCHMARK_CAPTURE(BandwidthCopy, DevicePinned,
                  numeric::memory::MemoryType::DEVICE,
                  numeric::memory::MemoryType::PINNED);
BENCHMARK_CAPTURE(BandwidthCopy, DeviceDevice,
                  numeric::memory::MemoryType::DEVICE,
                  numeric::memory::MemoryType::DEVICE);
#endif
