#ifndef NUMERIC_MATH_NORM_HPP_
#define NUMERIC_MATH_NORM_HPP_

#include <numeric/math/sum.hpp>
#include <numeric/memory/array_base.hpp>
#include <numeric/memory/array_op.hpp>

namespace numeric::math::norm {

template <dim_t p, typename Derived>
decltype(auto) lp_p(const memory::ArrayBase<Derived> &x) {
  return sum(memory::pow<p>(memory::abs(x.derived())));
}

template <dim_t p, typename Derived>
decltype(auto) lp(const memory::ArrayBase<Derived> &x) {
  return std::pow(lp_p<p>(x), 1. / p);
}

template <typename Derived>
decltype(auto) l1(const memory::ArrayBase<Derived> &x) {
  return sum(memory::abs(x.derived()));
}

template <typename Derived>
decltype(auto) l2_squared(const memory::ArrayBase<Derived> &x) {
  return lp_p<2>(x);
}

template <typename Derived>
decltype(auto) l2(const memory::ArrayBase<Derived> &x) {
  return lp<2>(x);
}

} // namespace numeric::math::norm

#endif
