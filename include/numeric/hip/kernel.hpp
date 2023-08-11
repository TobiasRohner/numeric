#ifndef NUMERIC_HIP_KERNEL_HPP_
#define NUMERIC_HIP_KERNEL_HPP_

#include <hip/hip_runtime_api.h>
#include <numeric/hip/module.hpp>
#include <numeric/hip/safe_call.hpp>
#include <numeric/hip/stream.hpp>
#include <string_view>

namespace numeric::hip {

class Kernel {
public:
  Kernel(const std::shared_ptr<Module> &module, std::string_view name);
  Kernel(const Kernel &) = default;
  Kernel(Kernel &&) = default;
  Kernel &operator=(const Kernel &) = default;
  Kernel &operator=(Kernel &&) = default;

  template <typename... Args>
  void operator()(unsigned grid_dim_x, unsigned grid_dim_y, unsigned grid_dim_z,
                  unsigned block_dim_x, unsigned block_dim_y,
                  unsigned block_dim_z, unsigned shared_mem_bytes,
                  const Stream &stream, Args &&...args) const {
    const void *argsp[] = {&args...};
    NUMERIC_CHECK_HIP(hipModuleLaunchKernel(
        kernel_, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y,
        block_dim_z, shared_mem_bytes, stream.id(), const_cast<void **>(argsp),
        NULL));
  }

private:
  std::shared_ptr<Module> module_;
  hipFunction_t kernel_;
};

} // namespace numeric::hip

#endif
