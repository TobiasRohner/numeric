#ifndef NUMERIC_MATH_SUM_HPP_
#define NUMERIC_MATH_SUM_HPP_

#include <numeric/math/summer.hpp>

namespace numeric::math {

template <typename Src>
typename memory::ArrayTraits<Src>::scalar_t
sum(const memory::ArrayBase<Src> &src) {
  Summer<Src> summer(src.derived());
  return summer(src.derived());
}

} // namespace numeric::math

#endif
