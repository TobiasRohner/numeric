#include <numeric/hip/kernel.hpp>
#include <numeric/hip/runtime.hpp>
#include <numeric/hip/safe_call.hpp>

namespace numeric::hip {

Kernel::Kernel() : module_(nullptr), kernel_(0) {}

Kernel::Kernel(const std::shared_ptr<Module> &module, std::string_view name)
    : module_(module) {
  NUMERIC_CHECK_HIP(
      hipModuleGetFunction(&kernel_, module_->module, name.data()));
  // NUMERIC_CHECK_HIP(hipFuncGetAttributes(&attributes_, kernel_)); // TODO:
  // This currently fails. Why?
  int attr;
  NUMERIC_CHECK_HIP(hipFuncGetAttribute(
      &attr, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel_));
  attributes_.maxThreadsPerBlock = attr;
  NUMERIC_CHECK_HIP(hipFuncGetAttribute(
      &attr, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel_));
  attributes_.sharedSizeBytes = attr;
  NUMERIC_CHECK_HIP(
      hipFuncGetAttribute(&attr, HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, kernel_));
  attributes_.constSizeBytes = attr;
  NUMERIC_CHECK_HIP(
      hipFuncGetAttribute(&attr, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel_));
  attributes_.localSizeBytes = attr;
  NUMERIC_CHECK_HIP(
      hipFuncGetAttribute(&attr, HIP_FUNC_ATTRIBUTE_NUM_REGS, kernel_));
  attributes_.numRegs = attr;
  NUMERIC_CHECK_HIP(
      hipFuncGetAttribute(&attr, HIP_FUNC_ATTRIBUTE_PTX_VERSION, kernel_));
  attributes_.ptxVersion = attr;
  NUMERIC_CHECK_HIP(
      hipFuncGetAttribute(&attr, HIP_FUNC_ATTRIBUTE_BINARY_VERSION, kernel_));
  attributes_.binaryVersion = attr;
  NUMERIC_CHECK_HIP(
      hipFuncGetAttribute(&attr, HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA, kernel_));
  attributes_.cacheModeCA = attr;
  NUMERIC_CHECK_HIP(hipFuncGetAttribute(
      &attr, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernel_));
  attributes_.maxDynamicSharedSizeBytes = attr;
  NUMERIC_CHECK_HIP(hipFuncGetAttribute(
      &attr, HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, kernel_));
  attributes_.preferredShmemCarveout = attr;
}

int Kernel::max_threads_per_block() const {
  return attributes_.maxThreadsPerBlock;
}

size_t Kernel::shared_size_bytes() const { return attributes_.sharedSizeBytes; }

size_t Kernel::const_size_bytes() const { return attributes_.constSizeBytes; }

size_t Kernel::local_size_bytes() const { return attributes_.localSizeBytes; }

int Kernel::num_regs() const { return attributes_.numRegs; }

int Kernel::ptx_version() const { return attributes_.ptxVersion; }

int Kernel::binary_version() const { return attributes_.binaryVersion; }

int Kernel::cache_mode() const { return attributes_.cacheModeCA; }

int Kernel::max_dynamic_shared_size_bytes() const {
  return attributes_.maxDynamicSharedSizeBytes;
}

int Kernel::preferred_shared_memory_carveout() const {
  return attributes_.preferredShmemCarveout;
}

int Kernel::suggested_block_size(size_t shared_mem_bytes) const {
  int grid_size, block_size;
  NUMERIC_CHECK_HIP(hipOccupancyMaxPotentialBlockSize(
      &grid_size, &block_size, kernel_, shared_mem_bytes, 0));
  return block_size;
}

int Kernel::max_active_blocks_per_sm(int block_size,
                                     size_t shared_mem_bytes) const {
  int blocks;
  NUMERIC_CHECK_HIP(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, kernel_, block_size, shared_mem_bytes));
  return blocks;
}

LaunchParams Kernel::launch_params_for_grid(const Device &device, int dim_x,
                                            int dim_y, int dim_z,
                                            size_t shared_mem_bytes) const {
  return launch_params_for_grid(device, dim_x, dim_y, dim_z,
                                [&](int) { return shared_mem_bytes; });
}

} // namespace numeric::hip
