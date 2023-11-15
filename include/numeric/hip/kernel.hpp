#ifndef NUMERIC_HIP_KERNEL_HPP_
#define NUMERIC_HIP_KERNEL_HPP_

#include <numeric/hip/launch_params.hpp>
#include <numeric/hip/module.hpp>
#include <numeric/hip/runtime.hpp>
#include <numeric/hip/safe_call.hpp>
#include <numeric/hip/stream.hpp>
#include <string_view>

namespace numeric::hip {

class Kernel {
public:
  Kernel();
  Kernel(const std::shared_ptr<Module> &module, std::string_view name);
  Kernel(const Kernel &) = default;
  Kernel(Kernel &&) = default;
  Kernel &operator=(const Kernel &) = default;
  Kernel &operator=(Kernel &&) = default;

  template <typename... Args>
  void operator()(const LaunchParams &params, const Stream &stream,
                  Args &&...args) const {
    const void *argsp[] = {&args...};
    NUMERIC_CHECK_HIP(hipModuleLaunchKernel(
        kernel_, params.grid_dim_x, params.grid_dim_y, params.grid_dim_z,
        params.block_dim_x, params.block_dim_y, params.block_dim_z,
        params.shared_mem_bytes, stream.id(), const_cast<void **>(argsp),
        NULL));
  }

private:
  std::shared_ptr<Module> module_;
  hipFunction_t kernel_;
};

} // namespace numeric::hip

#endif
