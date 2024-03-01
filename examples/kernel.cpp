#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric/hip/device.hpp>
#include <numeric/hip/kernel.hpp>
#include <numeric/hip/program.hpp>
#include <numeric/memory/array.hpp>
#include <string>
#include <vector>

std::string read_file(std::string_view path) {
  std::ifstream f(path.data());
  const std::string src(std::istreambuf_iterator<char>(f), {});
  return src;
}

static constexpr char kernel_src[] = R"(
  #include <numeric/config.hpp>
  #include <numeric/memory/layout.hpp>
  #include <numeric/memory/array_view.hpp>

  template<typename Scalar>
  __global__ void gpu_kernel(numeric::memory::ArrayView<Scalar, 2> a) {
    const size_t i = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    const size_t j = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i >= a.shape(0)) {
      return;
    }
    if (j >= a.shape(1)) {
      return;
    }
    a(i, j) = 100 * (1+hipThreadIdx_x) + (1+hipThreadIdx_y);
  }
)";

int main() {
  static constexpr size_t N = 16;

  numeric::hip::Device device;

  numeric::hip::Program program(kernel_src);
  program.add_compile_option("--std=c++17");
  program.instantiate_kernel<double>("gpu_kernel");
  auto kernel = program.get_kernel<double>("gpu_kernel");

  numeric::memory::Shape<2> a_layout(N, N);
  numeric::memory::Array<double, 2> a_device(
      a_layout, numeric::memory::MemoryType::DEVICE);
  kernel({N / 8, N / 8, 1, 8, 8, 1}, numeric::hip::Stream(device),
         a_device.view());
  device.sync();

  numeric::memory::Array<double, 2> a_host(a_layout);
  numeric::memory::memcpy(a_host, a_device);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      std::cout << a_host(i, j) << ' ';
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return 0;
}
