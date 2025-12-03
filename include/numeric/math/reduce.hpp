#ifndef NUMERIC_MATH_REDUCE_HPP_
#define NUMERIC_MATH_REDUCE_HPP_

#include <limits>
#include <numeric/memory/array_base.hpp>
#include <numeric/memory/array_traits.hpp>
#include <numeric/utils/forward.hpp>
#include <numeric/utils/lambda.hpp>
#include <string_view>
#if NUMERIC_ENABLE_HIP
#include <numeric/hip/kernel.hpp>
#include <numeric/math/functions.hpp>
#include <numeric/memory/array.hpp>
#include <numeric/memory/array_const_view.hpp>
#include <numeric/memory/array_view.hpp>
#include <numeric/utils/type_name.hpp>
#endif

namespace numeric::math {

namespace internal {

template <typename Src, typename Func>
typename memory::ArrayTraits<Src>::scalar_t reduce_host(const Src &src,
                                                        Func &&f) {
  typename memory::ArrayTraits<Src>::scalar_t value;
  if constexpr (memory::ArrayTraits<Src>::dim > 1) {
    value = reduce_host(src(0), utils::forward<Func>(f));
  } else {
    value = src(0);
  }
  for (dim_t i = 1; i < src.shape(0); ++i) {
    if constexpr (memory::ArrayTraits<Src>::dim > 1) {
      value = f(value, reduce_host(src(i), utils::forward<Func>(f)));
    } else {
      value = f(value, src(i));
    }
  }
  return value;
}

#if NUMERIC_ENABLE_HIP
hip::Kernel reduce_device_build_kernel(std::string_view src,
                                       std::string_view f);
hip::Kernel
reduce_device_build_kernel_1d_contiguous_impl(std::string_view scalar,
                                              std::string_view f);
hip::LaunchParams reduce_device_launch_params(const hip::Device &device,
                                              dim_t N, dim_t bytes_per_scalar);

template <typename Scalar, typename Func>
hip::Kernel
reduce_device_build_kernel_1d_contiguous(const utils::Lambda<Func> &f) {
  static hip::Kernel kernel = reduce_device_build_kernel_1d_contiguous_impl(
      utils::type_name<Scalar>(), f.source);
  return kernel;
}

template <typename Scalar, typename Func>
unsigned reduce_device_buffer(const memory::ArrayConstView<Scalar, 1> &src,
                              const utils::Lambda<Func> &f, Scalar *buffer,
                              Scalar identity, const hip::Device &device) {
  static hip::Kernel kernel =
      reduce_device_build_kernel_1d_contiguous<Scalar>(f);
  const hip::LaunchParams lp =
      reduce_device_launch_params(device, src.size(), sizeof(Scalar));
  kernel(lp, hip::Stream(device), buffer, src.raw(), src.size(), identity);
  return lp.grid_dim_x;
}

template <typename Src, typename Func>
unsigned
reduce_device_general(const Src &src, const utils::Lambda<Func> &f,
                      typename memory::ArrayTraits<Src>::scalar_t *buffer,
                      typename memory::ArrayTraits<Src>::scalar_t identity,
                      const hip::Device &device) {
  static bool warned = false;
  if (!warned) {
    std::cerr << "WARNING: Reduction on the device currently copies data to a "
                 "buffer first!"
              << std::endl;
    warned = true;
  }
  using scalar_t = typename memory::ArrayTraits<Src>::scalar_t;
  memory::Array<scalar_t, memory::ArrayTraits<Src>::dim> input(
      src.shape(), memory::MemoryType::DEVICE);
  input = src;
  const numeric::memory::ArrayConstView<scalar_t, 1> flat_view(
      input.raw(), memory::Shape<1>(input.size()), input.memory_type());
  return reduce_device_buffer(flat_view, f, buffer, identity, device);
}

template <typename Src, typename Func>
unsigned
reduce_device_step(const Src &src, const utils::Lambda<Func> &f,
                   typename memory::ArrayTraits<Src>::scalar_t *buffer,
                   typename memory::ArrayTraits<Src>::scalar_t identity,
                   const hip::Device &device) {
  using scalar_t = typename memory::ArrayTraits<Src>::scalar_t;
  if constexpr (meta::is_same_v<Src, memory::ArrayView<scalar_t, 1>> ||
                meta::is_same_v<Src, memory::ArrayConstView<scalar_t, 1>>) {
    if (src.stride(0) == 1) {
      return reduce_device_buffer(src, f, buffer, identity, device);
    } else {
      return reduce_device_general(src, f, buffer, identity, device);
    }
  } else {
    return reduce_device_general(src, f, buffer, identity, device);
  }
}

template <typename Src, typename Func>
typename memory::ArrayTraits<Src>::scalar_t
reduce_device(const Src &src, const utils::Lambda<Func> &f,
              typename memory::ArrayTraits<Src>::scalar_t identity,
              const hip::Device &device) {
  using sl = memory::Slice;
  using scalar_t = typename memory::ArrayTraits<Src>::scalar_t;
  const auto drv = src.derived();
  const dim_t buffer_size = [&](dim_t size) {
    const dim_t first_iter =
        reduce_device_launch_params(device, size, sizeof(scalar_t)).grid_dim_x;
    const dim_t second_iter =
        reduce_device_launch_params(device, first_iter, sizeof(scalar_t))
            .grid_dim_x;
    return first_iter + second_iter;
  }(drv.size());
  memory::Array<scalar_t, 1> buffer(memory::Shape<1>(buffer_size),
                                    memory::MemoryType::DEVICE);
  dim_t buffer_start_idx = 0;
  unsigned num_reduced = reduce_device_step(
      drv, f, buffer.raw() + buffer_start_idx, identity, device);
  while (num_reduced > 1) {
    dim_t new_buffer_start_idx;
    if (buffer_start_idx > 0) {
      new_buffer_start_idx = 0;
    } else {
      new_buffer_start_idx = num_reduced;
    }
    num_reduced = reduce_device_step(
        buffer(sl(buffer_start_idx, buffer_start_idx + num_reduced)), f,
        buffer.raw() + new_buffer_start_idx, identity, device);
    buffer_start_idx = new_buffer_start_idx;
  }
  scalar_t result;
  memory::ArrayView<scalar_t, 1> result_view(&result, memory::Shape<1>(1),
                                             memory::MemoryType::HOST);
  memory::memcpy(result_view,
                 buffer(sl(buffer_start_idx, buffer_start_idx + 1)));
  return result;
}
#endif

} // namespace internal

template <typename Src, typename Func>
typename memory::ArrayTraits<Src>::scalar_t
reduce(const memory::ArrayBase<Src> &src, const utils::Lambda<Func> &f,
       typename memory::ArrayTraits<Src>::scalar_t identity) {
  if (is_host_accessible(src.memory_type())) {
    return internal::reduce_host(src.derived(), f.f);
  }
#if NUMERIC_ENABLE_HIP
  else if (is_device_accessible(src.memory_type())) {
    hip::Device device;
    return internal::reduce_device(src.derived(), f, identity, device);
  }
#endif
  else {
    NUMERIC_ERROR("Unknown memory_type: \"{}\"", to_string(src.memory_type()));
  }
}

template <typename Src>
typename memory::ArrayTraits<Src>::scalar_t
sum(const memory::ArrayBase<Src> &src) {
  const auto f = NUMERIC_LAMBDA([](auto a, auto b) { return a + b; });
  return reduce(src.derived(), f, 0);
}

template <typename Src>
typename memory::ArrayTraits<Src>::scalar_t
min(const memory::ArrayBase<Src> &src) {
  using scalar_t = typename memory::ArrayTraits<Src>::scalar_t;
  const scalar_t identity = std::numeric_limits<scalar_t>::max();
  const auto f =
      NUMERIC_LAMBDA([](auto a, auto b) { return numeric::math::min(a, b); });
  return reduce(src.derived(), f, identity);
}

template <typename Src>
typename memory::ArrayTraits<Src>::scalar_t
max(const memory::ArrayBase<Src> &src) {
  using scalar_t = typename memory::ArrayTraits<Src>::scalar_t;
  const scalar_t identity = std::numeric_limits<scalar_t>::lowest();
  const auto f =
      NUMERIC_LAMBDA([](auto a, auto b) { return numeric::math::max(a, b); });
  return reduce(src.derived(), f, identity);
}

} // namespace numeric::math

#endif
