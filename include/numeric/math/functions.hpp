#ifndef NUMERIC_MATH_FUNCTIONS_HPP_
#define NUMERIC_MATH_FUNCTIONS_HPP_

#include <numeric/config.hpp>

namespace numeric::math {

template <typename T> T min(T a, T b) { return a < b ? a : b; }

template <typename T> T max(T a, T b) { return a < b ? b : a; }

template <typename Lhs, typename Rhs>
NUMERIC_HOST_DEVICE auto div_up(Lhs lhs, Rhs rhs) {
  return (lhs + rhs - 1) / rhs;
}

} // namespace numeric::math

#endif
