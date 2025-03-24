#ifndef NUMERIC_MATH_FES_BASIS_L2_HPP_
#define NUMERIC_MATH_FES_BASIS_L2_HPP_

#include <numeric/math/basis_lagrange.hpp>
#include <numeric/math/fes/basis_base.hpp>
#include <numeric/mesh/elements.hpp>
#include <numeric/meta/type_tag.hpp>

namespace numeric::math::fes {

template <dim_t Order> struct BasisL2 : public BasisBase<BasisL2<Order>> {
  using super = BasisBase<BasisL2<Order>>;

  template <typename RefEl> using element_basis_t = BasisLagrange<RefEl, Order>;

  static constexpr dim_t order = Order;

  template <typename RefEl>
  static constexpr dim_t num_interior_basis_functions(meta::type_tag<RefEl>) {
    return element_basis_t<RefEl>::num_basis_functions;
  }

  template <typename RefEl>
  static constexpr dim_t num_basis_functions(meta::type_tag<RefEl> tt) {
    return num_interior_basis_functions(tt);
  }

  template <typename Scalar, typename RefEl>
  static void eval(Scalar *out, const Scalar *x, meta::type_tag<RefEl>) {
    element_basis_t<RefEl>::eval_basis(x, out);
  }

  template <typename Scalar, typename RefEl>
  static void gradient(Scalar (*out)[RefEl::dim == 0 ? dim_t(1) : RefEl::dim],
                       const Scalar *x, meta::type_tag<RefEl>) {
    element_basis_t<RefEl>::grad_basis(x, out);
  }

  using super::eval;
  using super::gradient;
  using super::num_basis_functions;
  using super::num_interior_basis_functions;
  using super::total_num_basis_functions;
};

} // namespace numeric::math::fes

#endif
