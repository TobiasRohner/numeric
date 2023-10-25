#ifndef NUMERIC_MEMORY_LAYOUT_HPP_
#define NUMERIC_MEMORY_LAYOUT_HPP_

#include <numeric/config.hpp>
#ifndef __HIP_DEVICE_COMPILE__
#include <iostream>
#endif

namespace numeric::memory {

template <dim_t N> class Layout {
public:
  NUMERIC_HOST_DEVICE Layout() {
    for (dim_t i = 0; i < N; ++i) {
      shape_[i] = 0;
      stride_[i] = 0;
    }
  }
  template <typename... Ints>
  NUMERIC_HOST_DEVICE Layout(Ints... dims)
      : shape_{static_cast<dim_t>(dims)...} {
    static_assert(sizeof...(dims) == N,
                  "Wrong number of dimensions provided to Layout constructor");
    stride_[N - 1] = 1;
    for (dim_t i = 2; i <= N; ++i) {
      stride_[N - i] = shape_[N - i + 1] * stride_[N - i + 1];
    }
  }
  Layout(const Layout &) = default;
  Layout &operator=(const Layout &) = default;

  NUMERIC_HOST_DEVICE dim_t *shape() noexcept { return shape_; }
  NUMERIC_HOST_DEVICE const dim_t *shape() const noexcept { return shape_; }
  NUMERIC_HOST_DEVICE dim_t &shape(size_t idx) noexcept { return shape_[idx]; }
  NUMERIC_HOST_DEVICE dim_t shape(size_t idx) const noexcept {
    return shape_[idx];
  }

  NUMERIC_HOST_DEVICE dim_t *stride() noexcept { return stride_; }
  NUMERIC_HOST_DEVICE const dim_t *stride() const noexcept { return stride_; }
  NUMERIC_HOST_DEVICE dim_t &stride(size_t idx) noexcept {
    return stride_[idx];
  }
  NUMERIC_HOST_DEVICE dim_t stride(size_t idx) const noexcept {
    return stride_[idx];
  }

  NUMERIC_HOST_DEVICE dim_t size() const noexcept {
    dim_t s = 1;
    for (size_t i = 0; i < N; ++i) {
      s *= shape(i);
    }
    return s;
  }

private:
  dim_t shape_[N];
  dim_t stride_[N];
};

#ifndef __HIP_DEVICE_COMPILE__
template <dim_t N>
std::ostream &operator<<(std::ostream &os, const Layout<N> &layout) {
  os << "[(" << layout.shape(0) << ", " << layout.stride(0) << ')';
  for (size_t d = 1; d < N; ++d) {
    os << ", (" << layout.shape(d) << ", " << layout.stride(d) << ')';
  }
  os << ']';
  return os;
}
#endif

} // namespace numeric::memory

#endif
