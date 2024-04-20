#ifndef NUMERIC_MATH_FES_BASIS_L2_HPP_
#define NUMERIC_MATH_FES_BASIS_L2_HPP_

#include <numeric/math/fes/basis_base.hpp>
#include <numeric/mesh/elements.hpp>
#include <numeric/meta/type_tag.hpp>

namespace numeric::math::fes {

template <typename Derived> struct BasisL2Base : public BasisBase<Derived> {
  using super = BasisBase<Derived>;

  static constexpr dim_t order = Derived::order;

  template <dim_t Ord>
  static constexpr dim_t
  num_interior_basis_functions(meta::type_tag<mesh::Point<Ord>>) {
    return 1;
  }

  template <dim_t Ord>
  static constexpr dim_t
  num_interior_basis_functions(meta::type_tag<mesh::Segment<Ord>>) {
    return order + 1;
  }

  template <dim_t Ord>
  static constexpr dim_t
  num_interior_basis_functions(meta::type_tag<mesh::Tria<Ord>>) {
    return (order + 1) * (order + 2) / 2;
  }

  template <dim_t Ord>
  static constexpr dim_t
  num_interior_basis_functions(meta::type_tag<mesh::Quad<Ord>>) {
    return (order + 1) * (order + 1);
  }

  template <dim_t Ord>
  static constexpr dim_t
  num_interior_basis_functions(meta::type_tag<mesh::Tetra<Ord>>) {
    return (order + 1) * (order + 2) * (order + 3) / 6;
  }

  template <dim_t Ord>
  static constexpr dim_t
  num_interior_basis_functions(meta::type_tag<mesh::Cube<Ord>>) {
    return (order + 1) * (order + 1) * (order + 1);
  }

  template <typename Element>
  static constexpr dim_t num_basis_functions(meta::type_tag<Element> tt) {
    return num_interior_basis_functions(tt);
  }

  using super::eval;
  using super::gradient;
  using super::num_basis_functions;
  using super::num_interior_basis_functions;
};

template <dim_t Order> struct BasisL2 : public BasisL2Base<BasisL2<Order>> {
  static_assert(Order != Order,
                "L2 Basis is not supported for the given order");
};

template <> struct BasisL2<0> : public BasisL2Base<BasisL2<0>> {
  using super = BasisL2Base<BasisL2<0>>;

  static constexpr dim_t order = 0;

  using super::eval;
  using super::gradient;
  using super::num_basis_functions;
  using super::num_interior_basis_functions;

  template <typename Element, typename Scalar>
  static void eval(Scalar *out, const Scalar *x, meta::type_tag<Element>) {
    out[0] = 1;
  }

  template <typename Element, typename Scalar>
  static void
  gradient(Scalar (*out)[Element::dim == 0 ? dim_t(1) : Element::dim],
           const Scalar *x, meta::type_tag<Element>) {
    for (dim_t i = 0; i < Element::dim; ++i) {
      out[0][i] = 0;
    }
  }
};

template <> struct BasisL2<1> : public BasisL2Base<BasisL2<1>> {
  using super = BasisL2Base<BasisL2<1>>;

  static constexpr dim_t order = 1;

  using super::eval;
  using super::gradient;
  using super::num_basis_functions;
  using super::num_interior_basis_functions;

  template <typename Scalar, dim_t Ord>
  static void eval(Scalar *out, const Scalar *x,
                   meta::type_tag<mesh::Point<Ord>>) {
    out[0] = 1;
  }

  template <typename Scalar, dim_t Ord>
  static void gradient(Scalar (*out)[1], const Scalar *x,
                       meta::type_tag<mesh::Point<Ord>>) {}

  template <typename Scalar, dim_t Ord>
  static void eval(Scalar *out, const Scalar *x,
                   meta::type_tag<mesh::Segment<Ord>>) {
    out[0] = 1 - *x;
    out[1] = *x;
  }

  template <typename Scalar, dim_t Ord>
  static void gradient(Scalar (*out)[1], const Scalar *x,
                       meta::type_tag<mesh::Segment<Ord>>) {
    out[0][0] = -1;
    out[1][0] = 1;
  }

  template <typename Scalar, dim_t Ord>
  static void eval(Scalar *out, const Scalar *x,
                   meta::type_tag<mesh::Tria<Ord>>) {
    const Scalar l2 = x[0];
    const Scalar l3 = x[1];
    const Scalar l1 = 1 - l2 - l3;
    out[0] = l1;
    out[1] = l2;
    out[2] = l3;
  }

  template <typename Scalar, dim_t Ord>
  static void gradient(Scalar (*out)[2], const Scalar *x,
                       meta::type_tag<mesh::Tria<Ord>>) {
    out[0][0] = -1;
    out[0][1] = -1;
    out[1][0] = 1;
    out[1][1] = 0;
    out[2][0] = 0;
    out[2][1] = 1;
  }

  template <typename Scalar, dim_t Ord>
  static void eval(Scalar *out, const Scalar *x,
                   meta::type_tag<mesh::Quad<Ord>>) {
    const Scalar x1 = x[0];
    const Scalar x2 = x[1];
    out[0] = (1 - x1) * (1 - x2);
    out[1] = x1 * (1 - x2);
    out[2] = x1 * x2;
    out[3] = (1 - x1) * x2;
  }

  template <typename Scalar, dim_t Ord>
  static void gradient(Scalar (*out)[2], const Scalar *x,
                       meta::type_tag<mesh::Quad<Ord>>) {
    const Scalar x1 = x[0];
    const Scalar x2 = x[1];
    out[0][0] = -1 + x2;
    out[0][1] = -1 + x1;
    out[1][0] = 1 - x2;
    out[1][1] = -x1;
    out[2][0] = x2;
    out[2][1] = x1;
    out[3][0] = -x2;
    out[3][1] = 1 - x1;
  }

  template <typename Scalar, dim_t Ord>
  static void eval(Scalar *out, const Scalar *x,
                   meta::type_tag<mesh::Tetra<Ord>>) {
    const Scalar l2 = x[0];
    const Scalar l3 = x[1];
    const Scalar l4 = x[2];
    const Scalar l1 = 1 - l2 - l3 - l4;
    out[0] = l1;
    out[1] = l2;
    out[2] = l3;
    out[3] = l4;
  }

  template <typename Scalar, dim_t Ord>
  static void gradient(Scalar (*out)[3], const Scalar *x,
                       meta::type_tag<mesh::Tetra<Ord>>) {
    const Scalar l2 = x[0];
    const Scalar l3 = x[1];
    const Scalar l4 = x[2];
    const Scalar l1 = 1 - l2 - l3 - l4;
    out[0][0] = -1;
    out[0][1] = -1;
    out[0][2] = -1;
    out[1][0] = 1;
    out[1][1] = 0;
    out[1][2] = 0;
    out[2][0] = 0;
    out[2][1] = 1;
    out[2][2] = 0;
    out[3][0] = 0;
    out[3][1] = 0;
    out[3][2] = 1;
  }

  template <typename Scalar, dim_t Ord>
  static void eval(Scalar *out, const Scalar *x,
                   meta::type_tag<mesh::Cube<Ord>>) {
    const Scalar x1 = x[0];
    const Scalar x2 = x[1];
    const Scalar x3 = x[2];
    out[0] = (1 - x1) * (1 - x2) * (1 - x3);
    out[1] = x1 * (1 - x2) * (1 - x3);
    out[2] = x1 * x2 * (1 - x3);
    out[3] = (1 - x1) * x2 * (1 - x3);
    out[4] = (1 - x1) * (1 - x2) * x3;
    out[5] = x1 * (1 - x2) * x3;
    out[6] = x1 * x2 * x3;
    out[7] = (1 - x1) * x2 * x3;
  }

  template <typename Scalar, dim_t Ord>
  static void gradient(Scalar (*out)[3], const Scalar *x,
                       meta::type_tag<mesh::Cube<Ord>>) {
    const Scalar x1 = x[0];
    const Scalar x2 = x[1];
    const Scalar x3 = x[2];
    out[0][0] = -(1 - x2) * (1 - x3);
    out[0][1] = -(1 - x1) * (1 - x3);
    out[0][2] = -(1 - x1) * (1 - x2);
    out[1][0] = (1 - x2) * (1 - x3);
    out[1][1] = -x1 * (1 - x3);
    out[1][2] = -x1 * (1 - x2);
    out[2][0] = x2 * (1 - x3);
    out[2][1] = x1 * (1 - x3);
    out[2][2] = -x1 * x2;
    out[3][0] = -x2 * (1 - x3);
    out[3][1] = (1 - x1) * (1 - x3);
    out[3][2] = -(1 - x1) * x2;
    out[4][0] = -(1 - x2) * x3;
    out[4][1] = -(1 - x1) * x3;
    out[4][2] = (1 - x1) * (1 - x2);
    out[5][0] = (1 - x2) * x3;
    out[5][1] = -x1 * x3;
    out[5][2] = x1 * (1 - x2);
    out[6][0] = x2 * x3;
    out[6][1] = x1 * x3;
    out[6][2] = x1 * x2;
    out[7][0] = -x2 * x3;
    out[7][1] = (1 - x1) * x3;
    out[7][2] = (1 - x1) * x2;
  }
};

} // namespace numeric::math::fes

#endif
