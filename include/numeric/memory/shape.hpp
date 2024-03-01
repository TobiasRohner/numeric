#ifndef NUMERIC_MEMORY_SHAPE_HPP_
#define NUMERIC_MEMORY_SHAPE_HPP_

#include <numeric/config.hpp>
#ifndef __HIP_DEVICE_COMPILE__
#include <iostream>
#endif

namespace numeric::memory {

template <dim_t N> class Shape {
public:
  NUMERIC_HOST_DEVICE Shape() {
    for (dim_t i = 0; i < N; ++i) {
      shape_[i] = 0;
    }
  }

  template <typename... Ints>
  NUMERIC_HOST_DEVICE Shape(Ints... dims)
      : shape_{static_cast<dim_t>(dims)...} {
    static_assert(sizeof...(dims) == N,
                  "Wrong number of dimensions provided to Shape constructor");
  }
  Shape(const Shape &) = default;
  Shape &operator=(const Shape &) = default;

  NUMERIC_HOST_DEVICE dim_t *raw() noexcept { return shape_; }
  NUMERIC_HOST_DEVICE const dim_t *raw() const noexcept { return shape_; }

  NUMERIC_HOST_DEVICE dim_t &operator[](size_t idx) noexcept {
    return shape_[idx];
  }

  NUMERIC_HOST_DEVICE dim_t operator[](size_t idx) const noexcept {
    return shape_[idx];
  }

  NUMERIC_HOST_DEVICE dim_t size() const noexcept {
    dim_t s = 1;
    for (size_t i = 0; i < N; ++i) {
      s *= shape_[i];
    }
    return s;
  }

private:
  dim_t shape_[N];
};

#ifndef __HIP_DEVICE_COMPILE__
template <dim_t N>
std::ostream &operator<<(std::ostream &os, const Shape<N> &shape) {
  os << "[" << shape[0];
  for (size_t d = 1; d < N; ++d) {
    os << ", " << shape[d];
  }
  os << ']';
  return os;
}
#endif

} // namespace numeric::memory

#endif
