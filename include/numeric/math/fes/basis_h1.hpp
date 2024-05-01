#ifndef NUMERIC_MATH_FES_BASIS_H1_HPP_
#define NUMERIC_MATH_FES_BASIS_H1_HPP_

#include <numeric/math/fes/basis_base.hpp>
#include <numeric/math/fes/basis_l2.hpp>

namespace numeric::math::fes {

template <dim_t Order> struct BasisH1 : public BasisBase<BasisH1<Order>> {
  using super = BasisBase<BasisH1<Order>>;

  static constexpr dim_t order = Order;

  template <dim_t Ord>
  static constexpr dim_t
  num_interior_basis_functions(meta::type_tag<mesh::Point<Ord>>) {
    return 1;
  }

  template <dim_t Ord>
  static constexpr dim_t
  num_interior_basis_functions(meta::type_tag<mesh::Segment<Ord>>) {
    return order - 1;
  }

  template <dim_t Ord>
  static constexpr dim_t
  num_interior_basis_functions(meta::type_tag<mesh::Tria<Ord>>) {
    return (order - 1) * order / 2 - (order - 1);
  }

  template <dim_t Ord>
  static constexpr dim_t
  num_interior_basis_functions(meta::type_tag<mesh::Quad<Ord>>) {
    return (order - 1) * (order - 1);
  }

  template <dim_t Ord>
  static constexpr dim_t
  num_interior_basis_functions(meta::type_tag<mesh::Tetra<Ord>>) {
    return (order - 1) * order * (order + 1) / 6;
  }

  template <dim_t Ord>
  static constexpr dim_t
  num_interior_basis_functions(meta::type_tag<mesh::Cube<Ord>>) {
    return (order - 1) * (order - 1) * (order - 1);
  }

  template <dim_t Ord>
  static constexpr dim_t num_basis_functions(meta::type_tag<mesh::Point<Ord>>) {
    return 1;
  }

  template <dim_t Ord>
  static constexpr dim_t
  num_basis_functions(meta::type_tag<mesh::Segment<Ord>>) {
    return order + 1;
  }

  template <dim_t Ord>
  static constexpr dim_t num_basis_functions(meta::type_tag<mesh::Tria<Ord>>) {
    return (order + 1) * (order + 2) / 2;
  }

  template <dim_t Ord>
  static constexpr dim_t num_basis_functions(meta::type_tag<mesh::Quad<Ord>>) {
    return (order + 1) * (order + 1);
  }

  template <dim_t Ord>
  static constexpr dim_t num_basis_functions(meta::type_tag<mesh::Tetra<Ord>>) {
    return (order + 1) * (order + 2) * (order + 3) / 6;
  }

  template <dim_t Ord>
  static constexpr dim_t num_basis_functions(meta::type_tag<mesh::Cube<Ord>>) {
    return (order + 1) * (order + 1) * (order + 1);
  }

  template <dim_t Ord>
  static constexpr dim_t num_basis_functions(dim_t /*subelement*/,
                                             meta::type_tag<mesh::Segment<Ord>>,
                                             meta::type_tag<mesh::Point<Ord>>) {
    return 1;
  }

  template <dim_t Ord>
  static constexpr dim_t num_basis_functions(dim_t /*subelement*/,
                                             meta::type_tag<mesh::Tria<Ord>>,
                                             meta::type_tag<mesh::Point<Ord>>) {
    return 1;
  }

  template <dim_t Ord>
  static constexpr dim_t
  num_basis_functions(dim_t /*subelement*/, meta::type_tag<mesh::Tria<Ord>>,
                      meta::type_tag<mesh::Segment<Ord>>) {
    return order - 1;
  }

  template <dim_t Ord>
  static constexpr dim_t num_basis_functions(dim_t /*subelement*/,
                                             meta::type_tag<mesh::Quad<Ord>>,
                                             meta::type_tag<mesh::Point<Ord>>) {
    return 1;
  }

  template <dim_t Ord>
  static constexpr dim_t
  num_basis_functions(dim_t /*subelement*/, meta::type_tag<mesh::Quad<Ord>>,
                      meta::type_tag<mesh::Segment<Ord>>) {
    return order - 1;
  }

  template <dim_t Ord>
  static constexpr dim_t num_basis_functions(dim_t /*subelement*/,
                                             meta::type_tag<mesh::Tetra<Ord>>,
                                             meta::type_tag<mesh::Point<Ord>>) {
    return 1;
  }

  template <dim_t Ord>
  static constexpr dim_t
  num_basis_functions(dim_t /*subelement*/, meta::type_tag<mesh::Tetra<Ord>>,
                      meta::type_tag<mesh::Segment<Ord>>) {
    return order - 1;
  }

  template <dim_t Ord>
  static constexpr dim_t num_basis_functions(dim_t /*subelement*/,
                                             meta::type_tag<mesh::Tetra<Ord>>,
                                             meta::type_tag<mesh::Tria<Ord>>) {
    return (order - 1) * order / 2;
  }

  template <dim_t Ord>
  static constexpr dim_t num_basis_functions(dim_t /*subelement*/,
                                             meta::type_tag<mesh::Cube<Ord>>,
                                             meta::type_tag<mesh::Point<Ord>>) {
    return 1;
  }

  template <dim_t Ord>
  static constexpr dim_t
  num_basis_functions(dim_t /*subelement*/, meta::type_tag<mesh::Cube<Ord>>,
                      meta::type_tag<mesh::Segment<Ord>>) {
    return order - 1;
  }

  template <dim_t Ord>
  static constexpr dim_t num_basis_functions(dim_t /*subelement*/,
                                             meta::type_tag<mesh::Cube<Ord>>,
                                             meta::type_tag<mesh::Quad<Ord>>) {
    return (order - 1) * (order - 1);
  }

  template <typename Element, typename Scalar>
  static void eval(Scalar *out, const Scalar *x, meta::type_tag<Element> tag) {
    BasisL2<Order>::eval(out, x, tag);
  }

  template <typename Element, typename Scalar>
  static void
  gradient(Scalar (*out)[Element::dim == 0 ? dim_t(1) : Element::dim],
           const Scalar *x, meta::type_tag<Element> tag) {
    BasisL2<Order>::gradient(out, x, tag);
  }

  using super::eval;
  using super::gradient;
  using super::num_basis_functions;
  using super::num_interior_basis_functions;
  using super::total_num_basis_functions;
};

} // namespace numeric::math::fes

#endif
