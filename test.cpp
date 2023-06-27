#include <hip/hiprtc.h>
#include <hip/hip_runtime_api.h>
#include <string>
#include <iostream>
#include <vector>
#include <numeric/hip/safe_call.hpp>
#include <numeric/hip/device.hpp>
#include <numeric/hip/program.hpp>
#include <numeric/hip/kernel.hpp>
#include <numeric/memory/memory_type.hpp>
#include <numeric/memory/allocator.hpp>


static constexpr char kernel_src[] = R"(
  template<typename Scalar>
  __global__ void gpu_kernel(Scalar *a, size_t N) {
    const size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i >= N) {
      return;
    }
    a[i] = hipThreadIdx_x;
  }
)";


int main() {
  static constexpr size_t N = 1000;

  numeric::hip::Device device;

  numeric::hip::Program program(kernel_src);
  program.add_compile_option("-I/usr/local/cuda-11.7/targets/x86_64-linux/include");
  program.instantiate_kernel<double>("gpu_kernel");
  auto kernel = program.get_kernel<double>("gpu_kernel");

  numeric::memory::Allocator<double> alloc_host(numeric::memory::MemoryType::HOST);
  numeric::memory::Allocator<double> alloc_device(numeric::memory::MemoryType::DEVICE);
  double *a_host = alloc_host.allocate(N);
  double *a_device = alloc_device.allocate(N);

  kernel(N/10, 1, 1, 10, 1, 1, 0, numeric::hip::Stream(device), a_device, N);
  device.sync();

  hipMemcpy(a_host, a_device, N*sizeof(double), hipMemcpyDeviceToHost);
  for (size_t i = 0 ; i < N ; ++i) {
    std::cout << a_host[i] << ' ';
  }
  std::cout << std::endl;

  alloc_host.deallocate(a_host, N);
  alloc_device.deallocate(a_device, N);

  return 0;
}
