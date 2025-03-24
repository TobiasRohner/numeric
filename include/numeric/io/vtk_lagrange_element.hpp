#ifndef NUMERIC_IO_VTK_LAGRANGE_ELEMENT_HPP_
#define NUMERIC_IO_VTK_LAGRANGE_ELEMENT_HPP_

#include <numeric/math/basis_lagrange.hpp>
#include <numeric/meta/meta.hpp>

namespace numeric::io {

template <typename RefEl, dim_t Order> struct VTKLagrangeElement {
  static constexpr dim_t order = Order;
  using ref_el_t = RefEl;
  static constexpr dim_t num_nodes =
      math::BasisLagrange<ref_el_t, order>::num_basis_functions;
  static constexpr unsigned char type = []() {
    if (meta::is_same_v<ref_el_t, mesh::RefElSegment>) {
      return 68;
    }
    if (meta::is_same_v<ref_el_t, mesh::RefElTria>) {
      return 69;
    }
    if (meta::is_same_v<ref_el_t, mesh::RefElQuad>) {
      return 70;
    }
    if (meta::is_same_v<ref_el_t, mesh::RefElTetra>) {
      return 71;
    }
    if (meta::is_same_v<ref_el_t, mesh::RefElCube>) {
      return 72;
    }
  }();

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    math::BasisLagrange<ref_el_t, order>::node(i, out);
  }
};

} // namespace numeric::io

#endif
