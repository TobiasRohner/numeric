#ifndef NUMERIC_MATH_SUM_DEVICE_HPP_
#define NUMERIC_MATH_SUM_DEVICE_HPP_

#include <hip/hip_runtime_api.h>
#include <numeric/hip/kernel.hpp>
#include <numeric/math/functions.hpp>
#include <numeric/math/summer_impl.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/memory/memcpy.hpp>
#include <numeric/utils/error.hpp>
#include <numeric/utils/type_name.hpp>

namespace numeric::math {

namespace detail {

hip::Kernel sum_device_build_kernel(std::string_view src);
hip::Kernel sum_device_build_kernel_1d_contiguous_impl(std::string_view scalar);

template <typename Scalar> hip::Kernel sum_device_build_kernel_1d_contiguous() {
  static hip::Kernel kernel =
      sum_device_build_kernel_1d_contiguous_impl(utils::type_name<Scalar>());
  return kernel;
}

} // namespace detail

template <typename Src> class SumDevice : public SummerImpl<Src> {
  using super = SummerImpl<Src>;

public:
  using scalar_t = typename super::scalar_t;

  SumDevice(const hip::Device &device = hip::Device()) : device_(device) {
    static hip::Kernel shared_kernel =
        detail::sum_device_build_kernel(utils::type_name<Src>());
    kernel_general_ = shared_kernel;
    kernel_1d_ = detail::sum_device_build_kernel_1d_contiguous<scalar_t>();
  }

  virtual scalar_t operator()(const memory::ArrayBase<Src> &src) override {
    using sl = memory::Slice;
    const auto drv = src.derived();
    memory::Array<scalar_t, 1> buffer(memory::Shape<1>(buffer_size(drv.size())),
                                      memory::MemoryType::DEVICE);
    dim_t buffer_start_idx = 0;
    unsigned num_reduced = reduce(drv, buffer.raw() + buffer_start_idx);
    while (num_reduced > 1) {
      dim_t new_buffer_start_idx;
      if (buffer_start_idx > 0) {
        new_buffer_start_idx = 0;
      } else {
        new_buffer_start_idx = num_reduced;
      }
      num_reduced =
          reduce(buffer(sl(buffer_start_idx, buffer_start_idx + num_reduced)),
                 buffer.raw() + new_buffer_start_idx);
      buffer_start_idx = new_buffer_start_idx;
    }
    scalar_t result;
    memory::ArrayView<scalar_t, 1> result_view(&result, memory::Shape<1>(1),
                                               memory::MemoryType::HOST);
    memory::memcpy(result_view,
                   buffer(sl(buffer_start_idx, buffer_start_idx + 1)));
    return result;
  }

private:
  hip::Device device_;
  hip::Kernel kernel_general_;
  hip::Kernel kernel_1d_;

  dim_t buffer_size(dim_t size) const {
    const dim_t first_iter =
        device_.launch_params_for_grid(math::div_up(size, 2), 1, 1).grid_dim_x;
    const dim_t second_iter =
        device_.launch_params_for_grid(math::div_up(first_iter, 2), 1, 1)
            .grid_dim_x;
    return first_iter + second_iter;
  }

  template <typename S> unsigned reduce(const S &src, scalar_t *buffer) {
    if constexpr (meta::is_same_v<S, memory::ArrayView<scalar_t, 1>> ||
                  meta::is_same_v<S, memory::ArrayConstView<scalar_t, 1>>) {
      if constexpr (meta::is_same_v<Src, memory::ArrayView<scalar_t, 1>> ||
                    meta::is_same_v<Src, memory::ArrayConstView<scalar_t, 1>>) {
        if (src.stride(0) == 1) {
          return reduce_buffer(src, buffer);
        } else {
          return reduce_src(src, buffer);
        }
      } else {
        return reduce_buffer(src, buffer);
      }
    } else {
      return reduce_src(src, buffer);
    }
  }

  unsigned reduce_buffer(const memory::ArrayConstView<scalar_t, 1> &src,
                         scalar_t *buffer) {
    hip::LaunchParams lp =
        device_.launch_params_for_grid(math::div_up(src.size(), 2), 1, 1);
    lp.shared_mem_bytes =
        2 * device_.warp_size() *
        math::div_up(lp.block_dim_x, 2 * device_.warp_size()) *
        sizeof(scalar_t);
    kernel_1d_(lp, hip::Stream(device_), buffer, src.raw(), src.size());
    return lp.grid_dim_x;
  }

  unsigned reduce_src(const Src &src, scalar_t *buffer) {
    static bool warned = false;
    if (!warned) {
      std::cerr << "WARNING: Sum on the device currently copies data to a "
                   "buffer first!"
                << std::endl;
      warned = true;
    }
    memory::Array<scalar_t, memory::ArrayTraits<Src>::dim> input(
        src.shape(), memory::MemoryType::DEVICE);
    input = src;
    const numeric::memory::ArrayConstView<scalar_t, 1> flat_view(
        input.raw(), memory::Shape<1>(input.size()), input.memory_type());
    return reduce_buffer(flat_view, buffer);
  }
};

} // namespace numeric::math

#endif
