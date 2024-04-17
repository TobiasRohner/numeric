#ifndef NUMERIC_MEMORY_SHAPE_HPP_
#define NUMERIC_MEMORY_SHAPE_HPP_

#include <numeric/config.hpp>
#ifndef __HIP_DEVICE_COMPILE__
#include <iostream>
#endif

namespace numeric::memory {

/**
 * @brief Represents the shape of a multidimensional array.
 *
 * @tparam N Dimensionality of the shape.
 */
template <dim_t N> class Shape {
public:
  /**
   * @brief Default constructor.
   *
   * Initializes the shape with zeros.
   */
  NUMERIC_HOST_DEVICE Shape() {
    for (dim_t i = 0; i < N; ++i) {
      shape_[i] = 0;
    }
  }

  /**
   * @brief Constructor with variadic arguments.
   *
   * Constructs the shape with provided dimensions.
   *
   * @tparam Ints Parameter pack of dimensions.
   * @param dims Dimensions for the shape.
   */
  template <typename... Ints>
  NUMERIC_HOST_DEVICE Shape(Ints... dims)
      : shape_{static_cast<dim_t>(dims)...} {
    static_assert(sizeof...(dims) == N,
                  "Wrong number of dimensions provided to Shape constructor");
  }
  Shape(const Shape &) = default;
  Shape &operator=(const Shape &) = default;

  /**
   * @brief Returns a pointer to the raw shape data.
   *
   * @return Pointer to the raw shape data.
   */
  NUMERIC_HOST_DEVICE dim_t *raw() noexcept { return shape_; }

  /**
   * @brief Returns a const pointer to the raw shape data.
   *
   * @return Const pointer to the raw shape data.
   */
  NUMERIC_HOST_DEVICE const dim_t *raw() const noexcept { return shape_; }

  /**
   * @brief Accesses the dimension at the specified index.
   *
   * @param idx Index of the dimension.
   * @return Dimension at the specified index.
   */
  NUMERIC_HOST_DEVICE dim_t &operator[](size_t idx) noexcept {
    return shape_[idx];
  }

  /**
   * @brief Accesses the dimension at the specified index (const version).
   *
   * @param idx Index of the dimension.
   * @return Dimension at the specified index.
   */
  NUMERIC_HOST_DEVICE dim_t operator[](size_t idx) const noexcept {
    return shape_[idx];
  }

  /**
   * @brief Calculates the total size of the shape.
   *
   * @return Total size of the shape.
   */
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
