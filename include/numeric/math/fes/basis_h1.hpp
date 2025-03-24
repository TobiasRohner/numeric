#ifndef NUMERIC_MATH_FES_BASIS_H1_HPP_
#define NUMERIC_MATH_FES_BASIS_H1_HPP_

#include <numeric/math/basis_lagrange.hpp>
#include <numeric/math/fes/basis_base.hpp>
#include <numeric/mesh/elements.hpp>
#include <numeric/meta/type_tag.hpp>

namespace numeric::math::fes {

template <dim_t Order> struct BasisH1 : public BasisBase<BasisH1<Order>> {
  using super = BasisBase<BasisH1<Order>>;

  template <typename RefEl> using element_basis_t = BasisLagrange<RefEl, Order>;

  static constexpr dim_t order = Order;

  static constexpr dim_t
  num_interior_basis_functions(meta::type_tag<mesh::RefElPoint>) {
    return element_basis_t<mesh::RefElPoint>::num_basis_functions;
  }

  static constexpr dim_t
  num_interior_basis_functions(meta::type_tag<mesh::RefElSegment>) {
    return element_basis_t<mesh::RefElSegment>::num_basis_functions -
           2 * num_interior_basis_functions<mesh::RefElPoint>();
  }

  static constexpr dim_t
  num_interior_basis_functions(meta::type_tag<mesh::RefElTria>) {
    return element_basis_t<mesh::RefElTria>::num_basis_functions -
           3 * num_interior_basis_functions<mesh::RefElPoint>() -
           3 * num_interior_basis_functions<mesh::RefElSegment>();
  }

  static constexpr dim_t
  num_interior_basis_functions(meta::type_tag<mesh::RefElQuad>) {
    return element_basis_t<mesh::RefElQuad>::num_basis_functions -
           4 * num_interior_basis_functions<mesh::RefElPoint>() -
           4 * num_interior_basis_functions<mesh::RefElSegment>();
  }

  static constexpr dim_t
  num_interior_basis_functions(meta::type_tag<mesh::RefElTetra>) {
    return element_basis_t<mesh::RefElTetra>::num_basis_functions -
           4 * num_interior_basis_functions<mesh::RefElPoint>() -
           6 * num_interior_basis_functions<mesh::RefElSegment>() -
           4 * num_interior_basis_functions<mesh::RefElTria>();
  }

  static constexpr dim_t
  num_interior_basis_functions(meta::type_tag<mesh::RefElCube>) {
    return element_basis_t<mesh::RefElCube>::num_basis_functions -
           8 * num_interior_basis_functions<mesh::RefElPoint>() -
           12 * num_interior_basis_functions<mesh::RefElSegment>() -
           6 * num_interior_basis_functions<mesh::RefElQuad>();
  }

  template <typename RefEl>
  static constexpr dim_t num_basis_functions(meta::type_tag<RefEl>) {
    return element_basis_t<RefEl>::num_basis_functions;
  }

  template <typename Parent, typename Child>
  static constexpr dim_t num_basis_functions(dim_t /*subelement*/,
                                             meta::type_tag<Parent>,
                                             meta::type_tag<Child>) {
    return num_interior_basis_functions(meta::type_tag<Child>{});
  }

  template <typename Scalar, typename RefEl>
  static void eval(Scalar *out, const Scalar *x, meta::type_tag<RefEl>) {
    element_basis_t<RefEl>::eval_basis(x, out);
  }

  template <typename Scalar, typename RefEl>
  static void gradient(Scalar (*out)[RefEl::dim == 0 ? dim_t(1) : RefEl::dim],
                       const Scalar *x, meta::type_tag<RefEl> tag) {
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
