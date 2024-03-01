#ifndef NUMERIC_MEMORY_STRIDE_HPP_
#define NUMERIC_MEMORY_STRIDE_HPP_

#include <numeric/config.hpp>
#ifndef __HIP_DEVICE_COMPILE__
#include <iostream>
#endif

namespace numeric::memory {

template <dim_t N> class Stride {
public:
  NUMERIC_HOST_DEVICE Stride() {
    for (dim_t i = 0; i < N; ++i) {
      stride_[i] = 0;
    }
  }
  Stride(const Stride &) = default;
  Stride &operator=(const Stride &) = default;

  NUMERIC_HOST_DEVICE dim_t *raw() noexcept { return stride_; }
  NUMERIC_HOST_DEVICE const dim_t *raw() const noexcept { return stride_; }

  NUMERIC_HOST_DEVICE dim_t &operator[](size_t idx) noexcept {
    return stride_[idx];
  }

  NUMERIC_HOST_DEVICE dim_t operator[](size_t idx) const noexcept {
    return stride_[idx];
  }

  NUMERIC_HOST_DEVICE dim_t size() const noexcept {
    dim_t s = 1;
    for (size_t i = 0; i < N; ++i) {
      s *= stride_[i];
    }
    return s;
  }

private:
  dim_t stride_[N];
};

#ifndef __HIP_DEVICE_COMPILE__
template <dim_t N>
std::ostream &operator<<(std::ostream &os, const Stride<N> &stride) {
  os << "[" << stride[0];
  for (size_t d = 1; d < N; ++d) {
    os << ", " << stride[d];
  }
  os << ']';
  return os;
}
#endif

} // namespace numeric::memory

#endif
