#ifndef NUMERIC_UTILS_FORWARD_HPP_
#define NUMERIC_UTILS_FORWARD_HPP_

#include <numeric/meta/meta.hpp>

namespace numeric::utils {

template <typename T>
NUMERIC_HOST_DEVICE constexpr T &&
forward(typename meta::remove_reference_t<T> &param) noexcept {
  return static_cast<T &&>(param);
}

template <typename T>
NUMERIC_HOST_DEVICE constexpr T &&
forward(typename meta::remove_reference_t<T> &&param) noexcept {
  static_assert(!meta::is_lvalue_reference_v<T>,
                "Cannot forward an rvalue as an lvalue");
  return static_cast<T &&>(param);
}

} // namespace numeric::utils

#endif
