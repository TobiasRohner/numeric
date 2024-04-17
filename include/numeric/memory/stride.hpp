#ifndef NUMERIC_MEMORY_STRIDE_HPP_
#define NUMERIC_MEMORY_STRIDE_HPP_

#include <numeric/config.hpp>
#ifndef __HIP_DEVICE_COMPILE__
#include <iostream>
#endif

namespace numeric::memory {

/**
 * @brief Represents the stride of a multidimensional array.
 *
 * @tparam N Dimensionality of the stride.
 */
template <dim_t N> class Stride {
public:
  /**
   * @brief Default constructor.
   *
   * Initializes the stride with zeros.
   */
  NUMERIC_HOST_DEVICE Stride() {
    for (dim_t i = 0; i < N; ++i) {
      stride_[i] = 0;
    }
  }
  Stride(const Stride &) = default;
  Stride &operator=(const Stride &) = default;

  /**
   * @brief Returns a pointer to the raw stride data.
   *
   * @return Pointer to the raw stride data.
   */
  NUMERIC_HOST_DEVICE dim_t *raw() noexcept { return stride_; }

  /**
   * @brief Returns a const pointer to the raw stride data.
   *
   * @return Const pointer to the raw stride data.
   */
  NUMERIC_HOST_DEVICE const dim_t *raw() const noexcept { return stride_; }

  /**
   * @brief Accesses the stride at the specified index.
   *
   * @param idx Index of the stride.
   * @return Stride at the specified index.
   */
  NUMERIC_HOST_DEVICE dim_t &operator[](size_t idx) noexcept {
    return stride_[idx];
  }

  /**
   * @brief Accesses the stride at the specified index (const version).
   *
   * @param idx Index of the stride.
   * @return Stride at the specified index.
   */
  NUMERIC_HOST_DEVICE dim_t operator[](size_t idx) const noexcept {
    return stride_[idx];
  }

  /**
   * @brief Calculates the total size of the stride.
   *
   * @return Total size of the stride.
   */
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
