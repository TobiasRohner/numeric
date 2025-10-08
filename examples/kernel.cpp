/**
 * @page numeric_hip_tutorial Getting Started with Numeric HIP
 *
 * @section intro_sec Introduction
 *
 * This tutorial demonstrates how to use the **Numeric HIP** library to write,
 * compile, and launch a GPU kernel using C++. It covers the full pipeline from
 * memory allocation to computation and result retrieval.
 *
 * You will learn how to:
 * - Write a simple GPU kernel
 * - Use `numeric::hip::Program` to compile device code at runtime
 * - Launch the kernel with proper grid/block configuration
 * - Allocate and copy memory between host and device
 * - Inspect the results
 *
 * @section full_example_sec Full Source Code
 *
 * Here is the complete source code for reference:
 *
 * @snippet kernel.cpp full_example
 *
 * @section breakdown_sec Code Breakdown
 *
 * Let's look at each piece of the code in detail.
 *
 * @subsection device_init_sec 1. Device Initialization
 *
 * The first step is to initialize the HIP device. This creates a handle that
 * will be used for memory management and kernel execution.
 *
 * @snippet kernel.cpp device_init
 *
 * @subsection kernel_compile_sec 2. Program and Kernel Compilation
 *
 * We define the kernel as a C++ string and compile it at runtime using the
 * `numeric::hip::Program` interface. This allows flexible and dynamic GPU
 * programming without the need for separate `.cu` or `.hip` files.
 *
 * - `add_compile_option` allows passing compiler flags (here, C++17).
 * - `instantiate_kernel` tells the program to compile a template instance.
 * - `get_kernel` fetches a callable wrapper for execution.
 *
 * @snippet kernel.cpp kernel_compilation
 *
 * @subsection memory_alloc_sec 3. Memory Allocation
 *
 * Here we create two 2D arrays:
 * - One on the **device** (`a_device`) to hold GPU-side data.
 * - One on the **host** (`a_host`) to retrieve results after computation.
 *
 * The memory layout is 2D with shape `N x N`. The arrays are strongly typed
 * and managed safely using `numeric::memory::Array`.
 *
 * @snippet kernel.cpp memory_allocation
 *
 * @subsection kernel_launch_sec 4. Kernel Launch
 *
 * The kernel is launched with a block/grid configuration:
 *
 * ```
 *   dim3 gridDim  = (N / 8, N / 8, 1);
 *   dim3 blockDim = (8, 8, 1);
 * ```
 *
 * This divides the work into 16 blocks of 64 threads (total 1024 threads).
 * Each thread calculates its `(i, j)` position and sets the corresponding
 * element in the 2D array using a unique formula.
 *
 * After the launch, `device.sync()` ensures the computation is completed.
 *
 * @snippet kernel.cpp kernel_launch
 *
 * @subsection copy_results_sec 5. Copying and Printing Results
 *
 * Once the kernel finishes, we copy the device memory to host memory using
 * `numeric::memory::memcpy`, which handles all memory-type awareness
 * internally.
 *
 * Finally, we iterate through the `a_host` array and print the results in a
 * 2D grid format. Each element should reflect the thread's contribution,
 * confirming successful computation.
 *
 * @snippet kernel.cpp device_to_host_copy
 * @snippet kernel.cpp print_output
 */

//! [full_example]
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric/hip/device.hpp>
#include <numeric/hip/kernel.hpp>
#include <numeric/hip/program.hpp>
#include <numeric/memory/array.hpp>
#include <string>
#include <vector>

// Kernel source code
static constexpr char kernel_src[] = R"(
  #include <numeric/config.hpp>
  #include <numeric/memory/layout.hpp>
  #include <numeric/memory/array_view.hpp>

  template<typename Scalar>
  __global__ void gpu_kernel(numeric::memory::ArrayView<Scalar, 2> a) {
    const size_t i = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    const size_t j = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i >= a.shape(0)) return;
    if (j >= a.shape(1)) return;
    a(i, j) = 100 * (1 + hipThreadIdx_x) + (1 + hipThreadIdx_y);
  }
)";

int main() {
  //! [device_init]
  static constexpr size_t N = 16;
  numeric::hip::Device device;
  //! [device_init]

  //! [kernel_compilation]
  numeric::hip::Program program(kernel_src);
  program.add_compile_option("--std=c++17");
  program.instantiate_kernel<double>("gpu_kernel");
  auto kernel = program.get_kernel<double>("gpu_kernel");
  //! [kernel_compilation]

  //! [memory_allocation]
  numeric::memory::Shape<2> a_layout(N, N);
  numeric::memory::Array<double, 2> a_device(
      a_layout, numeric::memory::MemoryType::DEVICE);
  numeric::memory::Array<double, 2> a_host(a_layout);
  //! [memory_allocation]

  //! [kernel_launch]
  kernel({N / 8, N / 8, 1, 8, 8, 1}, numeric::hip::Stream(device),
         a_device.view());
  device.sync();
  //! [kernel_launch]

  //! [device_to_host_copy]
  numeric::memory::memcpy(a_host, a_device);
  //! [device_to_host_copy]

  //! [print_output]
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      std::cout << a_host(i, j) << ' ';
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  //! [print_output]

  return 0;
}
//! [full_example]
