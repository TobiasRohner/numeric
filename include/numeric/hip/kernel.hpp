#ifndef NUMERIC_HIP_KERNEL_HPP_
#define NUMERIC_HIP_KERNEL_HPP_

#include <numeric/hip/launch_params.hpp>
#include <numeric/hip/module.hpp>
#include <numeric/hip/runtime.hpp>
#include <numeric/hip/safe_call.hpp>
#include <numeric/hip/stream.hpp>
#include <string_view>

namespace numeric::hip {

/**
 * @brief Class representing a HIP compute kernel.
 */
class Kernel {
public:
  /**
   * @brief Default constructor.
   *
   * Constructs an empty Kernel object.
   */
  Kernel();

  /**
   * @brief Constructor with module and kernel name.
   *
   * Constructs a Kernel object with the specified module and kernel name.
   *
   * @param module Shared pointer to the module containing the kernel.
   * @param name Name of the kernel function.
   */
  Kernel(const std::shared_ptr<Module> &module, std::string_view name);

  Kernel(const Kernel &) = default;
  Kernel(Kernel &&) = default;
  Kernel &operator=(const Kernel &) = default;
  Kernel &operator=(Kernel &&) = default;

  /**
   * @brief Conversion operator to bool.
   *
   * @return True if the kernel is valid, false otherwise.
   */
  operator bool() const noexcept { return kernel_ != NULL; }

  /**
   * @brief Launches the kernel asynchronously.
   *
   * @tparam Args Argument types.
   * @param params Launch parameters.
   * @param stream Stream where the kernel will be launched.
   * @param args Arguments to be passed to the kernel.
   */
  template <typename... Args>
  void async(const LaunchParams &params, const Stream &stream,
             Args &&...args) const {
    const void *argsp[] = {&args...};
    NUMERIC_CHECK_HIP(hipModuleLaunchKernel(
        kernel_, params.grid_dim_x, params.grid_dim_y, params.grid_dim_z,
        params.block_dim_x, params.block_dim_y, params.block_dim_z,
        params.shared_mem_bytes, stream.id(), const_cast<void **>(argsp),
        NULL));
  }

  /**
   * @brief Launches the kernel synchronously.
   *
   * @tparam Args Argument types.
   * @param params Launch parameters.
   * @param stream Stream where the kernel will be launched.
   * @param args Arguments to be passed to the kernel.
   */
  template <typename... Args>
  void operator()(const LaunchParams &params, const Stream &stream,
                  Args &&...args) const {
    async(params, stream, std::forward<Args>(args)...);
    stream.sync();
  }

private:
  std::shared_ptr<Module> module_;
  hipFunction_t kernel_;
};

} // namespace numeric::hip

#endif
