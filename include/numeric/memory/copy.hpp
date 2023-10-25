#ifndef NUMERIC_MEMORY_COPY_HPP_
#define NUMERIC_MEMORY_COPY_HPP_

#include <numeric/config.hpp>
#include <numeric/memory/copy_kernels.hpp>
#include <numeric/utils/error.hpp>
#if NUMERIC_ENABLE_HIP
#include <numeric/memory/copy_device_device.hpp>
#endif

namespace numeric::memory {

template <typename Scalar, dim_t N, typename Src>
void copy_host(ArrayView<Scalar, N> dst, const ArrayBase<Src> &src) {
  copy_naive_elm(dst, src.derived());
}

template <typename Scalar, dim_t N, typename Src>
void copy_device(ArrayView<Scalar, N> dst, const ArrayBase<Src> &src) {
  CopyDeviceToDevice<Scalar, N, Src> cpy;
  cpy(dst, src);
}

template <typename Scalar, dim_t N, typename Src>
void copy(ArrayView<Scalar, N> dst, const ArrayBase<Src> &src) {
  if (is_host_accessible(dst.memory_type()) &&
      is_host_accessible(src.memory_type())) {
    copy_host(dst, src.derived());
#if NUMERIC_ENABLE_HIP
  } else if (is_device_accessible(dst.memory_type()) &&
             is_device_accessible(src.memory_type())) {
    copy_device(dst, src.derived());
#endif
  } else {
    NUMERIC_ERROR("Unsupported memory locations {} -> {}",
                  to_string(src.memory_type()), to_string(dst.memory_type()));
  }
}

} // namespace numeric::memory

#endif
