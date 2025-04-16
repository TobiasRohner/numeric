#ifndef NUMERIC_MATH_BASIS_LAGRANGE_CUBE_HPP_
#define NUMERIC_MATH_BASIS_LAGRANGE_CUBE_HPP_

#include <numeric/mesh/ref_el_cube.hpp>
#include <numeric/mesh/ref_el_point.hpp>
#include <numeric/mesh/ref_el_quad.hpp>
#include <numeric/mesh/ref_el_segment.hpp>
#include <numeric/mesh/ref_el_tetra.hpp>
#include <numeric/mesh/ref_el_tria.hpp>
#include <numeric/meta/meta.hpp>
#include <numeric/meta/type_tag.hpp>

namespace numeric::math {

template <dim_t Order> struct BasisLagrange<mesh::RefElCube, Order> {
  using ref_el_t = mesh::RefElCube;
  static constexpr dim_t order = 1;
  static constexpr dim_t num_basis_functions = 8;
  static constexpr dim_t num_interpolation_nodes = 8;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = x[2] - 1;
    const Scalar x1 = x[1] - 1;
    const Scalar x2 = x[0] - 1;
    return -coeffs[0] * x0 * x1 * x2 + coeffs[1] * x0 * x1 * x[0] -
           coeffs[2] * x0 * x[0] * x[1] + coeffs[3] * x0 * x2 * x[1] +
           coeffs[4] * x1 * x2 * x[2] - coeffs[5] * x1 * x[0] * x[2] +
           coeffs[6] * x[0] * x[1] * x[2] - coeffs[7] * x2 * x[1] * x[2];
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = x[1] - 1;
    const Scalar x2 = x[2] - 1;
    const Scalar x3 = x1 * x2;
    const Scalar x4 = x2 * x[1];
    const Scalar x5 = x1 * x[2];
    const Scalar x6 = x[1] * x[2];
    out[0] = -x0 * x3;
    out[1] = x3 * x[0];
    out[2] = -x4 * x[0];
    out[3] = x0 * x4;
    out[4] = x0 * x5;
    out[5] = -x5 * x[0];
    out[6] = x6 * x[0];
    out[7] = -x0 * x6;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = x[2] - 1;
    const Scalar x1 = x[1] - 1;
    const Scalar x2 = x[0] - 1;
    out[0] = -coeffs[0] * x0 * x1 + coeffs[1] * x0 * x1 -
             coeffs[2] * x0 * x[1] + coeffs[3] * x0 * x[1] +
             coeffs[4] * x1 * x[2] - coeffs[5] * x1 * x[2] +
             coeffs[6] * x[1] * x[2] - coeffs[7] * x[1] * x[2];
    out[1] = -coeffs[0] * x0 * x2 + coeffs[1] * x0 * x[0] -
             coeffs[2] * x0 * x[0] + coeffs[3] * x0 * x2 +
             coeffs[4] * x2 * x[2] - coeffs[5] * x[0] * x[2] +
             coeffs[6] * x[0] * x[2] - coeffs[7] * x2 * x[2];
    out[2] = -coeffs[0] * x1 * x2 + coeffs[1] * x1 * x[0] -
             coeffs[2] * x[0] * x[1] + coeffs[3] * x2 * x[1] +
             coeffs[4] * x1 * x2 - coeffs[5] * x1 * x[0] +
             coeffs[6] * x[0] * x[1] - coeffs[7] * x2 * x[1];
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[3]) {
    const Scalar x0 = x[1] - 1;
    const Scalar x1 = x[2] - 1;
    const Scalar x2 = x0 * x1;
    const Scalar x3 = x[0] - 1;
    const Scalar x4 = x1 * x3;
    const Scalar x5 = x0 * x3;
    const Scalar x6 = x1 * x[0];
    const Scalar x7 = x0 * x[0];
    const Scalar x8 = x1 * x[1];
    const Scalar x9 = x[0] * x[1];
    const Scalar x10 = x3 * x[1];
    const Scalar x11 = x0 * x[2];
    const Scalar x12 = x3 * x[2];
    const Scalar x13 = x[0] * x[2];
    const Scalar x14 = x[1] * x[2];
    out[0][0] = -x2;
    out[0][1] = -x4;
    out[0][2] = -x5;
    out[1][0] = x2;
    out[1][1] = x6;
    out[1][2] = x7;
    out[2][0] = -x8;
    out[2][1] = -x6;
    out[2][2] = -x9;
    out[3][0] = x8;
    out[3][1] = x4;
    out[3][2] = x10;
    out[4][0] = x11;
    out[4][1] = x12;
    out[4][2] = x5;
    out[5][0] = -x11;
    out[5][1] = -x13;
    out[5][2] = -x7;
    out[6][0] = x14;
    out[6][1] = x13;
    out[6][2] = x9;
    out[7][0] = -x14;
    out[7][1] = -x12;
    out[7][2] = -x10;
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[3];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
    out[2] = static_cast<Scalar>(idxs[2]) / order;
  }

  template <typename Scalar>
  static constexpr void interpolation_nodes(Scalar (*out)[3]) {
    out[0][0] = static_cast<Scalar>(0) / order;
    out[0][1] = static_cast<Scalar>(0) / order;
    out[0][2] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(1) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[1][2] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(1) / order;
    out[2][1] = static_cast<Scalar>(1) / order;
    out[2][2] = static_cast<Scalar>(0) / order;
    out[3][0] = static_cast<Scalar>(0) / order;
    out[3][1] = static_cast<Scalar>(1) / order;
    out[3][2] = static_cast<Scalar>(0) / order;
    out[4][0] = static_cast<Scalar>(0) / order;
    out[4][1] = static_cast<Scalar>(0) / order;
    out[4][2] = static_cast<Scalar>(1) / order;
    out[5][0] = static_cast<Scalar>(1) / order;
    out[5][1] = static_cast<Scalar>(0) / order;
    out[5][2] = static_cast<Scalar>(1) / order;
    out[6][0] = static_cast<Scalar>(1) / order;
    out[6][1] = static_cast<Scalar>(1) / order;
    out[6][2] = static_cast<Scalar>(1) / order;
    out[7][0] = static_cast<Scalar>(0) / order;
    out[7][1] = static_cast<Scalar>(1) / order;
    out[7][2] = static_cast<Scalar>(1) / order;
  }
  template <typename Scalar>
  static constexpr void interpolate(const Scalar *node_values, Scalar *coeffs) {
    for (dim_t i = 0; i < num_interpolation_nodes; ++i) {
      coeffs[i] = node_values[i];
    }
  }

  static constexpr void node_idxs(dim_t i, dim_t *out) {
    switch (i) {
    case 0:
      out[0] = 0;
      out[1] = 0;
      out[2] = 0;
      break;
    case 1:
      out[0] = 1;
      out[1] = 0;
      out[2] = 0;
      break;
    case 2:
      out[0] = 1;
      out[1] = 1;
      out[2] = 0;
      break;
    case 3:
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
      break;
    case 4:
      out[0] = 0;
      out[1] = 0;
      out[2] = 1;
      break;
    case 5:
      out[0] = 1;
      out[1] = 0;
      out[2] = 1;
      break;
    case 6:
      out[0] = 1;
      out[1] = 1;
      out[2] = 1;
      break;
    case 7:
      out[0] = 0;
      out[1] = 1;
      out[2] = 1;
      break;
    }
  }

  template <typename Element>
  static void subelement_node_idxs(dim_t subelement, dim_t *idxs) {
    subelement_node_idxs(subelement, idxs, meta::type_tag<Element>{});
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElPoint>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      break;
    case 1:
      idxs[0] = 1;
      break;
    case 2:
      idxs[0] = 2;
      break;
    case 3:
      idxs[0] = 3;
      break;
    case 4:
      idxs[0] = 4;
      break;
    case 5:
      idxs[0] = 5;
      break;
    case 6:
      idxs[0] = 6;
      break;
    case 7:
      idxs[0] = 7;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElSegment>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 1;
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      break;
    case 2:
      idxs[0] = 3;
      idxs[1] = 2;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 3;
      break;
    case 4:
      idxs[0] = 4;
      idxs[1] = 5;
      break;
    case 5:
      idxs[0] = 5;
      idxs[1] = 6;
      break;
    case 6:
      idxs[0] = 7;
      idxs[1] = 6;
      break;
    case 7:
      idxs[0] = 4;
      idxs[1] = 7;
      break;
    case 8:
      idxs[0] = 0;
      idxs[1] = 4;
      break;
    case 9:
      idxs[0] = 1;
      idxs[1] = 5;
      break;
    case 10:
      idxs[0] = 2;
      idxs[1] = 6;
      break;
    case 11:
      idxs[0] = 3;
      idxs[1] = 7;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTria>) {
    switch (subelement) {
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElQuad>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 7;
      idxs[3] = 4;
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 6;
      idxs[3] = 5;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 5;
      idxs[3] = 4;
      break;
    case 3:
      idxs[0] = 3;
      idxs[1] = 2;
      idxs[2] = 6;
      idxs[3] = 7;
      break;
    case 4:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 2;
      idxs[3] = 3;
      break;
    case 5:
      idxs[0] = 4;
      idxs[1] = 5;
      idxs[2] = 6;
      idxs[3] = 7;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTetra>) {
    switch (subelement) {
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElCube>) {
    switch (subelement) {
    default:
      break;
    }
  }
};

} // namespace numeric::math

#endif
