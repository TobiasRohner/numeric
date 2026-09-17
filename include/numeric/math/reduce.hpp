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
typename memory::ArrayTraits<Src>::scalar_t
reduce_host(const Src &src, Func &&f,
            typename memory::ArrayTraits<Src>::scalar_t identity) {
  typename memory::ArrayTraits<Src>::scalar_t value = identity;
  for (dim_t i = 0; i < src.shape(0); ++i) {
    if constexpr (memory::ArrayTraits<Src>::dim > 1) {
      f(&value, reduce_host(src(i), utils::forward<Func>(f), identity));
    } else {
      f(&value, src(i));
    }
  }
  return value;
}

#if NUMERIC_ENABLE_HIP
hip::Kernel reduce_device_build_kernel_impl(std::string_view src,
                                            std::string_view f,
                                            bool f_is_atomic);
hip::Kernel reduce_device_build_kernel_1d_contiguous_impl(
    std::string_view scalar, std::string_view f, bool f_is_atomic);
hip::LaunchParams reduce_device_launch_params(const hip::Device &device,
                                              dim_t N, dim_t bytes_per_scalar);

template <typename Src, typename Func>
hip::Kernel reduce_device_build_kernel(const utils::Lambda<Func> &f,
                                       bool f_is_atomic) {
  static hip::Kernel kernel = reduce_device_build_kernel_impl(
      utils::type_name<Src>(), f.source, f_is_atomic);
  return kernel;
}

template <typename Scalar, typename Func>
hip::Kernel
reduce_device_build_kernel_1d_contiguous(const utils::Lambda<Func> &f,
                                         bool f_is_atomic) {
  static hip::Kernel kernel = reduce_device_build_kernel_1d_contiguous_impl(
      utils::type_name<Scalar>(), f.source, f_is_atomic);
  return kernel;
}

template <typename Scalar, typename Func>
unsigned reduce_device_buffer(const memory::ArrayConstView<Scalar, 1> &src,
                              const utils::Lambda<Func> &f, Scalar *buffer,
                              Scalar identity, bool f_is_atomic,
                              const hip::Device &device) {
  static hip::Kernel kernel =
      reduce_device_build_kernel_1d_contiguous<Scalar>(f, f_is_atomic);
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
                      bool f_is_atomic, const hip::Device &device) {
  using scalar_t = typename memory::ArrayTraits<Src>::scalar_t;
  static hip::Kernel kernel = reduce_device_build_kernel<Src>(f, f_is_atomic);
  const hip::LaunchParams lp =
      reduce_device_launch_params(device, src.size(), sizeof(scalar_t));
  kernel(lp, hip::Stream(device), buffer, src, identity);
  return lp.grid_dim_x;
}

template <typename Src, typename Func>
unsigned
reduce_device_step(const Src &src, const utils::Lambda<Func> &f,
                   typename memory::ArrayTraits<Src>::scalar_t *buffer,
                   typename memory::ArrayTraits<Src>::scalar_t identity,
                   bool f_is_atomic, const hip::Device &device) {
  using scalar_t = typename memory::ArrayTraits<Src>::scalar_t;
  if constexpr (meta::is_same_v<Src, memory::ArrayView<scalar_t, 1>> ||
                meta::is_same_v<Src, memory::ArrayConstView<scalar_t, 1>>) {
    if (src.stride(0) == 1) {
      return reduce_device_buffer(src, f, buffer, identity, f_is_atomic,
                                  device);
    } else {
      return reduce_device_general(src, f, buffer, identity, f_is_atomic,
                                   device);
    }
  } else {
    return reduce_device_general(src, f, buffer, identity, f_is_atomic, device);
  }
}

template <typename Src, typename Func>
typename memory::ArrayTraits<Src>::scalar_t reduce_device(
    const Src &src, const utils::Lambda<Func> &f,
    typename memory::ArrayTraits<Src>::scalar_t identity, bool f_is_atomic,
    const hip::Device &device,
    memory::ArrayView<typename memory::ArrayTraits<Src>::scalar_t, 1> &buffer) {
  using sl = memory::Slice;
  using scalar_t = typename memory::ArrayTraits<Src>::scalar_t;
  const auto drv = src.derived();
  if (f_is_atomic) {
    buffer = identity;
    reduce_device_step(drv, f, buffer.raw(), identity, f_is_atomic, device);
    scalar_t result;
    memory::ArrayView<scalar_t, 1> result_view(&result, memory::Shape<1>(1),
                                               memory::MemoryType::HOST);
    memory::memcpy(result_view, buffer);
    return result;
  } else {
    dim_t buffer_start_idx = 0;
    unsigned num_reduced = reduce_device_step(
        drv, f, buffer.raw() + buffer_start_idx, identity, f_is_atomic, device);
    while (num_reduced > 1) {
      dim_t new_buffer_start_idx;
      if (buffer_start_idx > 0) {
        new_buffer_start_idx = 0;
      } else {
        new_buffer_start_idx = num_reduced;
      }
      num_reduced = reduce_device_step(
          buffer(sl(buffer_start_idx, buffer_start_idx + num_reduced)), f,
          buffer.raw() + new_buffer_start_idx, identity, f_is_atomic, device);
      buffer_start_idx = new_buffer_start_idx;
    }
    scalar_t result;
    memory::ArrayView<scalar_t, 1> result_view(&result, memory::Shape<1>(1),
                                               memory::MemoryType::HOST);
    memory::memcpy(result_view,
                   buffer(sl(buffer_start_idx, buffer_start_idx + 1)));
    return result;
  }
}
#endif

} // namespace internal

template <typename Src, typename Func> class Reduction {
public:
  using scalar_t = typename memory::ArrayTraits<Src>::scalar_t;

  Reduction(const memory::ArrayBase<Src> &src, const utils::Lambda<Func> &f,
            scalar_t identity, bool f_is_atomic = false)
      : src_(src.derived()), f_(f), identity_(identity),
        f_is_atomic_(f_is_atomic) {}

  scalar_t reduce() {
    if (is_host_accessible(src_.memory_type())) {
      return internal::reduce_host(src_, f_.f, identity_);
    }
#if NUMERIC_ENABLE_HIP
    else if (is_device_accessible(src_.memory_type())) {
      hip::Device device;
      return reduce(device);
    }
#endif
    else {
      NUMERIC_ERROR("Unsupported memory type: {}",
                    to_string(src_.memory_type()));
    }
  }

#if NUMERIC_ENABLE_HIP
  scalar_t reduce(const hip::Device &device) {
    if (!buffer_.raw()) {
      if (f_is_atomic_) {
        buffer_ = std::move(memory::Array<scalar_t, 1>(
            memory::Shape<1>(1), memory::MemoryType::DEVICE));
      } else {
        const dim_t buffer_size = [&](dim_t size) {
          const dim_t first_iter = internal::reduce_device_launch_params(
                                       device, size, sizeof(scalar_t))
                                       .grid_dim_x;
          const dim_t second_iter = internal::reduce_device_launch_params(
                                        device, first_iter, sizeof(scalar_t))
                                        .grid_dim_x;
          return first_iter + second_iter;
        }(src_.size());
        buffer_ = std::move(memory::Array<scalar_t, 1>(
            memory::Shape<1>(buffer_size), memory::MemoryType::DEVICE));
      }
    }
    return internal::reduce_device(src_, f_, identity_, f_is_atomic_, device,
                                   buffer_);
  }
#endif

private:
  memory::Array<scalar_t, 1> buffer_;
  const Src &src_;
  utils::Lambda<Func> f_;
  scalar_t identity_;
  bool f_is_atomic_;
};

template <typename Src, typename Func>
auto make_reduction(const memory::ArrayBase<Src> &src,
                    const utils::Lambda<Func> &f,
                    typename memory::ArrayTraits<Src>::scalar_t identity,
                    bool f_is_atomic = false) {
  return Reduction<Src, Func>(src, f, identity, f_is_atomic);
}

template <typename Src>
auto make_reduction_sum(const memory::ArrayBase<Src> &src) {
  static const auto f = NUMERIC_LAMBDA([](auto *a, auto b) {
    if constexpr (::numeric::is_host_compile) {
      *a += b;
    } else {
      atomicAdd(a, b);
    }
  });
  return make_reduction(src, f, 0, true);
}

template <typename Src>
auto make_reduction_min(const memory::ArrayBase<Src> &src) {
  using scalar_t = typename memory::ArrayTraits<Src>::scalar_t;
  const scalar_t identity = std::numeric_limits<scalar_t>::max();
  static const auto f = NUMERIC_LAMBDA([](auto *a, auto b) {
    if constexpr (::numeric::is_host_compile) {
      *a = numeric::math::min(*a, b);
    } else {
      atomicMin(a, b);
    }
  });
  return make_reduction(src, f, identity, true);
}

template <typename Src>
auto make_reduction_max(const memory::ArrayBase<Src> &src) {
  using scalar_t = typename memory::ArrayTraits<Src>::scalar_t;
  const scalar_t identity = std::numeric_limits<scalar_t>::lowest();
  static const auto f = NUMERIC_LAMBDA([](auto *a, auto b) {
    if constexpr (::numeric::is_host_compile) {
      *a = numeric::math::max(*a, b);
    } else {
      atomicMax(a, b);
    }
  });
  return make_reduction(src, f, identity, true);
}

template <typename Src, typename Func>
typename memory::ArrayTraits<Src>::scalar_t
reduce(const memory::ArrayBase<Src> &src, const utils::Lambda<Func> &f,
       typename memory::ArrayTraits<Src>::scalar_t identity,
       bool f_is_atomic = false) {
  auto reduction = make_reduction(src, f, identity, f_is_atomic);
  if (is_host_accessible(src.memory_type())) {
    return reduction.reduce();
  }
#if NUMERIC_ENABLE_HIP
  else if (is_device_accessible(src.memory_type())) {
    hip::Device device;
    return reduction.reduce(device);
  }
#endif
  else {
    NUMERIC_ERROR("Unknown memory_type: \"{}\"", to_string(src.memory_type()));
  }
}

template <typename Src>
typename memory::ArrayTraits<Src>::scalar_t
sum(const memory::ArrayBase<Src> &src) {
  auto reduction = make_reduction_sum(src);
  if (is_host_accessible(src.memory_type())) {
    return reduction.reduce();
  }
#if NUMERIC_ENABLE_HIP
  else if (is_device_accessible(src.memory_type())) {
    hip::Device device;
    return reduction.reduce(device);
  }
#endif
  else {
    NUMERIC_ERROR("Unknown memory_type: \"{}\"", to_string(src.memory_type()));
  }
}

template <typename Src>
typename memory::ArrayTraits<Src>::scalar_t
min(const memory::ArrayBase<Src> &src) {
  auto reduction = make_reduction_min(src);
  if (is_host_accessible(src.memory_type())) {
    return reduction.reduce();
  }
#if NUMERIC_ENABLE_HIP
  else if (is_device_accessible(src.memory_type())) {
    hip::Device device;
    return reduction.reduce(device);
  }
#endif
  else {
    NUMERIC_ERROR("Unknown memory_type: \"{}\"", to_string(src.memory_type()));
  }
}

template <typename Src>
typename memory::ArrayTraits<Src>::scalar_t
max(const memory::ArrayBase<Src> &src) {
  auto reduction = make_reduction_max(src);
  if (is_host_accessible(src.memory_type())) {
    return reduction.reduce();
  }
#if NUMERIC_ENABLE_HIP
  else if (is_device_accessible(src.memory_type())) {
    hip::Device device;
    return reduction.reduce(device);
  }
#endif
  else {
    NUMERIC_ERROR("Unknown memory_type: \"{}\"", to_string(src.memory_type()));
  }
}

} // namespace numeric::math

#endif
