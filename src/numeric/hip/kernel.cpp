#include <numeric/hip/kernel.hpp>
#include <numeric/hip/runtime.hpp>
#include <numeric/hip/safe_call.hpp>

namespace numeric::hip {

Kernel::Kernel() : module_(nullptr), kernel_(0) {}

Kernel::Kernel(const std::shared_ptr<Module> &module, std::string_view name)
    : module_(module) {
  NUMERIC_CHECK_HIP(
      hipModuleGetFunction(&kernel_, module_->module, name.data()));
}

int Kernel::max_threads_per_block() const {
  int value;
  NUMERIC_CHECK_HIP(hipFuncGetAttribute(
      &value, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel_));
  return value;
}

int Kernel::shared_size_bytes() const {
  int value;
  NUMERIC_CHECK_HIP(hipFuncGetAttribute(
      &value, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel_));
  return value;
}

int Kernel::const_size_bytes() const {
  int value;
  NUMERIC_CHECK_HIP(hipFuncGetAttribute(
      &value, HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, kernel_));
  return value;
}

int Kernel::local_size_bytes() const {
  int value;
  NUMERIC_CHECK_HIP(hipFuncGetAttribute(
      &value, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel_));
  return value;
}

int Kernel::num_regs() const {
  int value;
  NUMERIC_CHECK_HIP(
      hipFuncGetAttribute(&value, HIP_FUNC_ATTRIBUTE_NUM_REGS, kernel_));
  return value;
}

int Kernel::ptx_version() const {
  int value;
  NUMERIC_CHECK_HIP(
      hipFuncGetAttribute(&value, HIP_FUNC_ATTRIBUTE_PTX_VERSION, kernel_));
  return value;
}

int Kernel::binary_version() const {
  int value;
  NUMERIC_CHECK_HIP(
      hipFuncGetAttribute(&value, HIP_FUNC_ATTRIBUTE_BINARY_VERSION, kernel_));
  return value;
}

int Kernel::cache_mode() const {
  int value;
  NUMERIC_CHECK_HIP(
      hipFuncGetAttribute(&value, HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA, kernel_));
  return value;
}

int Kernel::max_dynamic_shared_size_bytes() const {
  int value;
  NUMERIC_CHECK_HIP(hipFuncGetAttribute(
      &value, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernel_));
  return value;
}

int Kernel::preferred_shared_memory_carveout() const {
  int value;
  NUMERIC_CHECK_HIP(hipFuncGetAttribute(
      &value, HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, kernel_));
  return value;
}

} // namespace numeric::hip
