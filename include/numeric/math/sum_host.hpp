#ifndef NUMERIC_MATH_SUM_HOST_HPP_
#define NUMERIC_MATH_SUM_HOST_HPP_

#include <numeric/math/summer_impl.hpp>

namespace numeric::math {

namespace detail {

template <typename Src>
typename memory::ArrayTraits<Src>::scalar_t sum_host(const Src &src) {
  typename memory::ArrayTraits<Src>::scalar_t value = 0;
  for (dim_t i = 0; i < src.shape(0); ++i) {
    if constexpr (memory::ArrayTraits<Src>::dim > 1) {
      value += sum_host(src(i));
    } else {
      value += src(i);
    }
  }
  return value;
}

} // namespace detail

template <typename Src> class SumHost : public SummerImpl<Src> {
  using super = SummerImpl<Src>;

public:
  using scalar_t = typename super::scalar_t;

  virtual scalar_t operator()(const memory::ArrayBase<Src> &src) override {
    return detail::sum_host(src.derived());
  }
};

} // namespace numeric::math

#endif
