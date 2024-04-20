#ifndef NUMERIC_MATH_FES_BASIS_BASE_HPP_
#define NUMERIC_MATH_FES_BASIS_BASE_HPP_

#include <numeric/config.hpp>
#include <numeric/meta/type_tag.hpp>

namespace numeric::math::fes {

template <typename Derived> struct BasisBase {
  template <typename Element>
  static constexpr dim_t num_interior_basis_functions() {
    return Derived::num_interior_basis_functions(meta::type_tag<Element>{});
  }

  template <typename Element> static constexpr dim_t num_basis_functions() {
    return Derived::num_basis_functions(meta::type_tag<Element>{});
  }

  template <typename Element, typename Scalar>
  static void eval(Scalar *out, const Scalar *x) {
    return Derived::eval(out, x, meta::type_tag<Element>{});
  }

  template <typename Element, typename Scalar>
  static void
  gradient(Scalar (*out)[Element::dim == 0 ? dim_t(1) : Element::dim],
           const Scalar *x) {
    return Derived::gradient(out, x, meta::type_tag<Element>{});
  }
};

} // namespace numeric::math::fes

#endif
