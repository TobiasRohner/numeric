#ifndef NUMERIC_MATH_SUMMER_IMPL_HPP_
#define NUMERIC_MATH_SUMMER_IMPL_HPP_

#include <numeric/memory/array_base.hpp>
#include <numeric/memory/array_traits.hpp>

namespace numeric::math {

template <typename Src> class SummerImpl {
public:
  using scalar_t = typename memory::ArrayTraits<Src>::scalar_t;

  SummerImpl() = default;
  virtual ~SummerImpl() = default;

  virtual scalar_t operator()(const memory::ArrayBase<Src> &src) = 0;
};

} // namespace numeric::math

#endif
