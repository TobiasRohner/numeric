#ifndef NUMERIC_HIP_KERNEL_HPP_
#define NUMERIC_HIP_KERNEL_HPP_

#include <numeric/hip/launch_params.hpp>
#include <numeric/hip/module.hpp>
#include <numeric/hip/runtime.hpp>
#include <numeric/hip/safe_call.hpp>
#include <numeric/hip/stream.hpp>
#include <numeric/math/functions.hpp>
#include <numeric/meta/meta.hpp>
#include <numeric/utils/error.hpp>
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

  int max_threads_per_block() const;
  size_t shared_size_bytes() const;
  size_t const_size_bytes() const;
  size_t local_size_bytes() const;
  int num_regs() const;
  int ptx_version() const;
  int binary_version() const;
  int cache_mode() const;
  int max_dynamic_shared_size_bytes() const;
  int preferred_shared_memory_carveout() const;
  int suggested_block_size(size_t shared_mem_bytes) const;
  int max_active_blocks_per_sm(int block_size, size_t shared_mem_bytes) const;
  LaunchParams launch_params_for_grid(const Device &device, int dim_x,
                                      int dim_y, int dim_z,
                                      size_t shared_mem_bytes) const;

  template <typename UnaryFunction>
  int suggested_block_size(
      const UnaryFunction &block_size_to_shared_mem_bytes) const {
    int grid_size, block_size;
    NUMERIC_CHECK_HIP(hipOccupancyMaxPotentialBlockSizeVariableSMem(
        &grid_size, &block_size, kernel_, block_size_to_shared_mem_bytes, 0));
    return block_size;
  }

  template <
      typename UnaryFunction,
      typename = decltype(meta::declval<UnaryFunction>()(meta::declval<int>()))>
  LaunchParams launch_params_for_grid(
      const Device &device, int dim_x, int dim_y, int dim_z,
      const UnaryFunction &block_size_to_shared_mem_bytes) const {
    LaunchParams lp_opt;
    double score_opt = 0;
    const int min_block_dim_x = device.warp_size();
    const int max_block_dim_x =
        math::max(min_block_dim_x, math::min(dim_x, 1024));
    for (int block_dim_x = min_block_dim_x; block_dim_x <= max_block_dim_x;
         block_dim_x *= 2) {
      const int min_block_dim_y = 1;
      const int max_block_dim_y = math::max(
          min_block_dim_y, math::min(math::min(dim_y, 1024),
                                     max_threads_per_block() / block_dim_x));
      if (max_block_dim_y == 0) {
        continue;
      }
      for (int block_dim_y = min_block_dim_y; block_dim_y <= max_block_dim_y;
           block_dim_y *= 2) {
        const int min_block_dim_z = 1;
        const int max_block_dim_z = math::max(
            min_block_dim_z,
            math::min(math::min(dim_z, 1024),
                      max_threads_per_block() / (block_dim_x * block_dim_y)));
        if (max_block_dim_z == 0) {
          continue;
        }
        for (int block_dim_z = min_block_dim_z; block_dim_z <= max_block_dim_z;
             block_dim_z *= 2) {
          const int grid_dim_x = math::div_up(dim_x, block_dim_x);
          const int grid_dim_y = math::div_up(dim_y, block_dim_y);
          const int grid_dim_z = math::div_up(dim_z, block_dim_z);
          const int block_size = block_dim_x * block_dim_y * block_dim_z;
          const int grid_size = grid_dim_x * grid_dim_y * grid_dim_z;
          const size_t shared_mem_bytes =
              block_size_to_shared_mem_bytes(block_size);
          const double efficiency = static_cast<double>(dim_x * dim_y * dim_z) /
                                    (block_size * grid_size);
          const int blocks_per_sm =
              max_active_blocks_per_sm(block_size, shared_mem_bytes);
          const double occupancy =
              static_cast<double>(blocks_per_sm * block_size) /
              device.max_threads_per_sm();
          const double score = efficiency * occupancy;
          if (score > score_opt) {
            score_opt = score;
            lp_opt.grid_dim_x = grid_dim_x;
            lp_opt.grid_dim_y = grid_dim_y;
            lp_opt.grid_dim_z = grid_dim_z;
            lp_opt.block_dim_x = block_dim_x;
            lp_opt.block_dim_y = block_dim_y;
            lp_opt.block_dim_z = block_dim_z;
            lp_opt.shared_mem_bytes = shared_mem_bytes;
          }
        }
      }
    }
    return lp_opt;
  }

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
  hipFuncAttributes attributes_;
};

} // namespace numeric::hip

#endif
