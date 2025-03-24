#ifndef NUMERIC_MATH_BASIS_LAGRANGE_TETRA_HPP_
#define NUMERIC_MATH_BASIS_LAGRANGE_TETRA_HPP_

#include <numeric/mesh/ref_el_cube.hpp>
#include <numeric/mesh/ref_el_point.hpp>
#include <numeric/mesh/ref_el_quad.hpp>
#include <numeric/mesh/ref_el_segment.hpp>
#include <numeric/mesh/ref_el_tetra.hpp>
#include <numeric/mesh/ref_el_tria.hpp>
#include <numeric/meta/meta.hpp>
#include <numeric/meta/type_tag.hpp>

namespace numeric::math {

template <> struct BasisLagrange<mesh::RefElTetra, 1> {
  using ref_el_t = mesh::RefElTetra;
  static constexpr dim_t order = 1;
  static constexpr dim_t num_basis_functions = 4;
  static constexpr dim_t num_interpolation_nodes = 4;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    return -coeffs[0] * (x[0] + x[1] + x[2] - 1) + coeffs[1] * x[0] +
           coeffs[2] * x[1] + coeffs[3] * x[2];
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    out[0] = -x[0] - x[1] - x[2] + 1;
    out[1] = x[0];
    out[2] = x[1];
    out[3] = x[2];
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    out[0] = -coeffs[0] + coeffs[1];
    out[1] = -coeffs[0] + coeffs[2];
    out[2] = -coeffs[0] + coeffs[3];
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[3]) {
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
    out[2][0] = static_cast<Scalar>(0) / order;
    out[2][1] = static_cast<Scalar>(1) / order;
    out[2][2] = static_cast<Scalar>(0) / order;
    out[3][0] = static_cast<Scalar>(0) / order;
    out[3][1] = static_cast<Scalar>(0) / order;
    out[3][2] = static_cast<Scalar>(1) / order;
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
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
      break;
    case 3:
      out[0] = 0;
      out[1] = 0;
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
      idxs[0] = 2;
      idxs[1] = 0;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 3;
      break;
    case 4:
      idxs[0] = 1;
      idxs[1] = 3;
      break;
    case 5:
      idxs[0] = 2;
      idxs[1] = 3;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTria>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 3;
      break;
    case 1:
      idxs[0] = 2;
      idxs[1] = 3;
      idxs[2] = 1;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 2;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 2;
      idxs[2] = 1;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElQuad>) {
    switch (subelement) {
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

template <> struct BasisLagrange<mesh::RefElTetra, 2> {
  using ref_el_t = mesh::RefElTetra;
  static constexpr dim_t order = 2;
  static constexpr dim_t num_basis_functions = 10;
  static constexpr dim_t num_interpolation_nodes = 10;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = 2 * x[0];
    const Scalar x1 = 2 * x[1];
    const Scalar x2 = x0 - 1;
    const Scalar x3 = 2 * x[2];
    const Scalar x4 = x[0] + x[1] + x[2] - 1;
    return coeffs[0] * x4 * (x1 + x2 + x3) + coeffs[1] * x2 * x[0] +
           coeffs[2] * x[1] * (x1 - 1) + coeffs[3] * x[2] * (x3 - 1) -
           2 * coeffs[4] * x0 * x4 + 2 * coeffs[5] * x0 * x[1] -
           2 * coeffs[6] * x1 * x4 - 2 * coeffs[7] * x3 * x4 +
           2 * coeffs[8] * x0 * x[2] + 2 * coeffs[9] * x1 * x[2];
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    const Scalar x0 = x[0] + x[1] + x[2] - 1;
    const Scalar x1 = 2 * x[1];
    const Scalar x2 = 2 * x[2];
    const Scalar x3 = 2 * x[0] - 1;
    const Scalar x4 = 4 * x[0];
    const Scalar x5 = 4 * x0;
    out[0] = x0 * (x1 + x2 + x3);
    out[1] = x3 * x[0];
    out[2] = x[1] * (x1 - 1);
    out[3] = x[2] * (x2 - 1);
    out[4] = -x0 * x4;
    out[5] = x4 * x[1];
    out[6] = -x5 * x[1];
    out[7] = -x5 * x[2];
    out[8] = x4 * x[2];
    out[9] = 4 * x[1] * x[2];
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = 4 * x[0];
    const Scalar x1 = x[2] - 1;
    const Scalar x2 = 4 * x[1];
    const Scalar x3 = -coeffs[6] * x2;
    const Scalar x4 = 4 * x[2];
    const Scalar x5 = coeffs[0] * (x0 + x2 + x4 - 3);
    const Scalar x6 = -coeffs[7] * x4 + x5;
    const Scalar x7 = -coeffs[4] * x0;
    out[0] = coeffs[1] * (x0 - 1) - 4 * coeffs[4] * (x1 + 2 * x[0] + x[1]) +
             coeffs[5] * x2 + coeffs[8] * x4 + x3 + x6;
    out[1] = coeffs[2] * (x2 - 1) + coeffs[5] * x0 -
             4 * coeffs[6] * (x1 + x[0] + 2 * x[1]) + coeffs[9] * x4 + x6 + x7;
    out[2] = coeffs[3] * (x4 - 1) -
             4 * coeffs[7] * (x[0] + x[1] + 2 * x[2] - 1) + coeffs[8] * x0 +
             coeffs[9] * x2 + x3 + x5 + x7;
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[3]) {
    const Scalar x0 = 4 * x[0];
    const Scalar x1 = 4 * x[1];
    const Scalar x2 = 4 * x[2];
    const Scalar x3 = x0 + x1 + x2 - 3;
    const Scalar x4 = x[2] - 1;
    const Scalar x5 = -x0;
    const Scalar x6 = -x1;
    const Scalar x7 = -x2;
    out[0][0] = x3;
    out[0][1] = x3;
    out[0][2] = x3;
    out[1][0] = x0 - 1;
    out[1][1] = 0;
    out[1][2] = 0;
    out[2][0] = 0;
    out[2][1] = x1 - 1;
    out[2][2] = 0;
    out[3][0] = 0;
    out[3][1] = 0;
    out[3][2] = x2 - 1;
    out[4][0] = -4 * x4 - 8 * x[0] - 4 * x[1];
    out[4][1] = x5;
    out[4][2] = x5;
    out[5][0] = x1;
    out[5][1] = x0;
    out[5][2] = 0;
    out[6][0] = x6;
    out[6][1] = -4 * x4 - 4 * x[0] - 8 * x[1];
    out[6][2] = x6;
    out[7][0] = x7;
    out[7][1] = x7;
    out[7][2] = 4 * (-x[0] - x[1] - 2 * x[2] + 1);
    out[8][0] = x2;
    out[8][1] = 0;
    out[8][2] = x0;
    out[9][0] = 0;
    out[9][1] = x2;
    out[9][2] = x1;
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
    out[1][0] = static_cast<Scalar>(2) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[1][2] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(0) / order;
    out[2][1] = static_cast<Scalar>(2) / order;
    out[2][2] = static_cast<Scalar>(0) / order;
    out[3][0] = static_cast<Scalar>(0) / order;
    out[3][1] = static_cast<Scalar>(0) / order;
    out[3][2] = static_cast<Scalar>(2) / order;
    out[4][0] = static_cast<Scalar>(1) / order;
    out[4][1] = static_cast<Scalar>(0) / order;
    out[4][2] = static_cast<Scalar>(0) / order;
    out[5][0] = static_cast<Scalar>(1) / order;
    out[5][1] = static_cast<Scalar>(1) / order;
    out[5][2] = static_cast<Scalar>(0) / order;
    out[6][0] = static_cast<Scalar>(0) / order;
    out[6][1] = static_cast<Scalar>(1) / order;
    out[6][2] = static_cast<Scalar>(0) / order;
    out[7][0] = static_cast<Scalar>(0) / order;
    out[7][1] = static_cast<Scalar>(0) / order;
    out[7][2] = static_cast<Scalar>(1) / order;
    out[8][0] = static_cast<Scalar>(1) / order;
    out[8][1] = static_cast<Scalar>(0) / order;
    out[8][2] = static_cast<Scalar>(1) / order;
    out[9][0] = static_cast<Scalar>(0) / order;
    out[9][1] = static_cast<Scalar>(1) / order;
    out[9][2] = static_cast<Scalar>(1) / order;
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
      out[0] = 2;
      out[1] = 0;
      out[2] = 0;
      break;
    case 2:
      out[0] = 0;
      out[1] = 2;
      out[2] = 0;
      break;
    case 3:
      out[0] = 0;
      out[1] = 0;
      out[2] = 2;
      break;
    case 4:
      out[0] = 1;
      out[1] = 0;
      out[2] = 0;
      break;
    case 5:
      out[0] = 1;
      out[1] = 1;
      out[2] = 0;
      break;
    case 6:
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
      break;
    case 7:
      out[0] = 0;
      out[1] = 0;
      out[2] = 1;
      break;
    case 8:
      out[0] = 1;
      out[1] = 0;
      out[2] = 1;
      break;
    case 9:
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
      idxs[2] = 4;
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 5;
      break;
    case 2:
      idxs[0] = 2;
      idxs[1] = 0;
      idxs[2] = 6;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 7;
      break;
    case 4:
      idxs[0] = 1;
      idxs[1] = 3;
      idxs[2] = 8;
      break;
    case 5:
      idxs[0] = 2;
      idxs[1] = 3;
      idxs[2] = 9;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTria>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 3;
      idxs[3] = 4;
      idxs[4] = 8;
      idxs[5] = 7;
      break;
    case 1:
      idxs[0] = 2;
      idxs[1] = 3;
      idxs[2] = 1;
      idxs[3] = 9;
      idxs[4] = 8;
      idxs[5] = 5;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 2;
      idxs[3] = 7;
      idxs[4] = 9;
      idxs[5] = 6;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 2;
      idxs[2] = 1;
      idxs[3] = 6;
      idxs[4] = 5;
      idxs[5] = 4;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElQuad>) {
    switch (subelement) {
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

template <> struct BasisLagrange<mesh::RefElTetra, 3> {
  using ref_el_t = mesh::RefElTetra;
  static constexpr dim_t order = 3;
  static constexpr dim_t num_basis_functions = 20;
  static constexpr dim_t num_interpolation_nodes = 20;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = 9 * x[1];
    const Scalar x1 = x[0] * x[2];
    const Scalar x2 = 3 * x[0];
    const Scalar x3 = x2 - 1;
    const Scalar x4 = (3.0 / 2.0) * x3;
    const Scalar x5 = 3 * x[2];
    const Scalar x6 = x5 - 1;
    const Scalar x7 = 3 * x[1];
    const Scalar x8 = x[1] * (x7 - 1);
    const Scalar x9 = (3.0 / 2.0) * x8;
    const Scalar x10 = x6 * x[2];
    const Scalar x11 = (3.0 / 2.0) * x[1];
    const Scalar x12 = x[0] + x[1] + x[2] - 1;
    const Scalar x13 = x12 * x[2];
    const Scalar x14 = 9 * x13;
    const Scalar x15 = x12 * x[0];
    const Scalar x16 = (3.0 / 2.0) * x13;
    const Scalar x17 = x2 - 2;
    const Scalar x18 = x17 + x5 + x7;
    const Scalar x19 = x12 * x18;
    return -1.0 / 2.0 * coeffs[0] * x19 * (x2 + x6 + x7) +
           3 * coeffs[10] * x16 * x18 - 3 * coeffs[11] * x16 * x6 +
           3 * coeffs[12] * x1 * x4 + (9.0 / 2.0) * coeffs[13] * x1 * x6 +
           3 * coeffs[14] * x9 * x[2] + 3 * coeffs[15] * x10 * x11 -
           3 * coeffs[16] * x14 * x[0] + 3 * coeffs[17] * x0 * x1 -
           3 * coeffs[18] * x14 * x[1] - 3 * coeffs[19] * x0 * x15 +
           (1.0 / 2.0) * coeffs[1] * x17 * x3 * x[0] +
           (1.0 / 2.0) * coeffs[2] * x8 * (x7 - 2) +
           (1.0 / 2.0) * coeffs[3] * x10 * (x5 - 2) +
           (9.0 / 2.0) * coeffs[4] * x15 * x18 - 3 * coeffs[5] * x15 * x4 +
           3 * coeffs[6] * x4 * x[0] * x[1] + 3 * coeffs[7] * x9 * x[0] -
           3 * coeffs[8] * x12 * x9 + 3 * coeffs[9] * x11 * x19;
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    const Scalar x0 = 3 * x[0];
    const Scalar x1 = x0 - 1;
    const Scalar x2 = 3 * x[1];
    const Scalar x3 = 3 * x[2];
    const Scalar x4 = x2 + x3;
    const Scalar x5 = x[0] + x[1] + x[2] - 1;
    const Scalar x6 = x0 - 2;
    const Scalar x7 = x5 * (x4 + x6);
    const Scalar x8 = x1 * x[0];
    const Scalar x9 = x[1] * (x2 - 1);
    const Scalar x10 = x[2] * (x3 - 1);
    const Scalar x11 = (9.0 / 2.0) * x[0];
    const Scalar x12 = (9.0 / 2.0) * x8;
    const Scalar x13 = (9.0 / 2.0) * x5;
    const Scalar x14 = (9.0 / 2.0) * x7;
    const Scalar x15 = 27 * x[0] * x[2];
    const Scalar x16 = 27 * x5 * x[1];
    out[0] = -1.0 / 2.0 * x7 * (x1 + x4);
    out[1] = (1.0 / 2.0) * x6 * x8;
    out[2] = (1.0 / 2.0) * x9 * (x2 - 2);
    out[3] = (1.0 / 2.0) * x10 * (x3 - 2);
    out[4] = x11 * x7;
    out[5] = -x12 * x5;
    out[6] = x12 * x[1];
    out[7] = x11 * x9;
    out[8] = -x13 * x9;
    out[9] = x14 * x[1];
    out[10] = x14 * x[2];
    out[11] = -x10 * x13;
    out[12] = x12 * x[2];
    out[13] = x10 * x11;
    out[14] = (9.0 / 2.0) * x9 * x[2];
    out[15] = (9.0 / 2.0) * x10 * x[1];
    out[16] = -x15 * x5;
    out[17] = x15 * x[1];
    out[18] = -x16 * x[2];
    out[19] = -x16 * x[0];
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = 3 * x[0];
    const Scalar x1 = x0 - 2;
    const Scalar x2 = x0 - 1;
    const Scalar x3 = 3 * x[2];
    const Scalar x4 = 3 * x[1];
    const Scalar x5 = x3 + x4;
    const Scalar x6 = -x1 - x5;
    const Scalar x7 = x[1] + x[2] - 1;
    const Scalar x8 = x7 + x[0];
    const Scalar x9 = -x8;
    const Scalar x10 = x6 * x9;
    const Scalar x11 = -x10;
    const Scalar x12 = (3.0 / 2.0) * coeffs[4];
    const Scalar x13 = x2 * x[0];
    const Scalar x14 = (3.0 / 2.0) * coeffs[5];
    const Scalar x15 = x7 + 2 * x[0];
    const Scalar x16 = 9 * x[2];
    const Scalar x17 = coeffs[16] * x16;
    const Scalar x18 = 9 * coeffs[19];
    const Scalar x19 = x18 * x[1];
    const Scalar x20 = 6 * x[0];
    const Scalar x21 = x20 - 1;
    const Scalar x22 = x3 - 1;
    const Scalar x23 = x4 - 1;
    const Scalar x24 = -x2 - x5;
    const Scalar x25 =
        (1.0 / 6.0) * coeffs[0] * (3 * x10 + x24 * x6 + 3 * x24 * x9);
    const Scalar x26 = 6 * x[1];
    const Scalar x27 = 6 * x[2];
    const Scalar x28 = -x20 - x26 - x27 + 5;
    const Scalar x29 = (3.0 / 2.0) * coeffs[10];
    const Scalar x30 = x22 * x[2];
    const Scalar x31 = (3.0 / 2.0) * coeffs[11];
    const Scalar x32 = x25 + x28 * x29 * x[2] + x30 * x31;
    const Scalar x33 = x23 * x[1];
    const Scalar x34 = (3.0 / 2.0) * coeffs[8];
    const Scalar x35 = (3.0 / 2.0) * coeffs[9];
    const Scalar x36 = x28 * x35 * x[1] + x33 * x34;
    const Scalar x37 = x4 - 2;
    const Scalar x38 = x[0] - 1;
    const Scalar x39 = x38 + 2 * x[1] + x[2];
    const Scalar x40 = x26 - 1;
    const Scalar x41 = x12 * x28 * x[0] + x13 * x14;
    const Scalar x42 = x3 - 2;
    const Scalar x43 = 9 * x38 + 9 * x[1] + 18 * x[2];
    const Scalar x44 = x27 - 1;
    out[0] = (9.0 / 2.0) * coeffs[12] * x21 * x[2] +
             (9.0 / 2.0) * coeffs[13] * x22 * x[2] +
             27 * coeffs[17] * x[1] * x[2] - 3 * coeffs[18] * x16 * x[1] +
             (1.0 / 2.0) * coeffs[1] * (x0 * x1 + x0 * x2 + x1 * x2) +
             (9.0 / 2.0) * coeffs[6] * x21 * x[1] +
             (9.0 / 2.0) * coeffs[7] * x23 * x[1] -
             3 * x12 * (x0 * x9 + x11 + x6 * x[0]) -
             3 * x14 * (x0 * x8 + x13 + x2 * x8) - 3 * x15 * x17 -
             3 * x15 * x19 - 3 * x32 - 3 * x36;
    out[1] = (9.0 / 2.0) * coeffs[14] * x40 * x[2] +
             (9.0 / 2.0) * coeffs[15] * x22 * x[2] +
             27 * coeffs[17] * x[0] * x[2] - 3 * coeffs[18] * x16 * x39 +
             (1.0 / 2.0) * coeffs[2] * (x23 * x37 + x23 * x4 + x37 * x4) +
             (9.0 / 2.0) * coeffs[6] * x2 * x[0] +
             (9.0 / 2.0) * coeffs[7] * x40 * x[0] - 3 * x17 * x[0] -
             3 * x18 * x39 * x[0] - 3 * x32 -
             3 * x34 * (x23 * x8 + x33 + x4 * x8) -
             3 * x35 * (x11 + x4 * x9 + x6 * x[1]) - 3 * x41;
    out[2] = (9.0 / 2.0) * coeffs[12] * x2 * x[0] +
             (9.0 / 2.0) * coeffs[13] * x44 * x[0] +
             (9.0 / 2.0) * coeffs[14] * x23 * x[1] +
             (9.0 / 2.0) * coeffs[15] * x44 * x[1] -
             3 * coeffs[16] * x43 * x[0] + 27 * coeffs[17] * x[0] * x[1] -
             3 * coeffs[18] * x43 * x[1] +
             (1.0 / 2.0) * coeffs[3] * (x22 * x3 + x22 * x42 + x3 * x42) -
             3 * x19 * x[0] - 3 * x25 - 3 * x29 * (x11 + x3 * x9 + x6 * x[2]) -
             3 * x31 * (x22 * x8 + x3 * x8 + x30) - 3 * x36 - 3 * x41;
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[3]) {
    const Scalar x0 = x[1] + x[2] - 1;
    const Scalar x1 = x0 + x[0];
    const Scalar x2 = 3 * x[0];
    const Scalar x3 = x2 - 2;
    const Scalar x4 = 3 * x[1];
    const Scalar x5 = 3 * x[2];
    const Scalar x6 = x4 + x5;
    const Scalar x7 = x3 + x6;
    const Scalar x8 = x1 * x7;
    const Scalar x9 = x2 - 1;
    const Scalar x10 = x6 + x9;
    const Scalar x11 =
        -3.0 / 2.0 * x1 * x10 - 1.0 / 2.0 * x10 * x7 - 3.0 / 2.0 * x8;
    const Scalar x12 = x4 - 2;
    const Scalar x13 = x4 - 1;
    const Scalar x14 = x5 - 2;
    const Scalar x15 = x5 - 1;
    const Scalar x16 = x1 * x2;
    const Scalar x17 = 6 * x[0];
    const Scalar x18 = 6 * x[1];
    const Scalar x19 = 6 * x[2];
    const Scalar x20 = x17 + x18 + x19 - 5;
    const Scalar x21 = (9.0 / 2.0) * x[0];
    const Scalar x22 = x20 * x21;
    const Scalar x23 = x9 * x[0];
    const Scalar x24 = (9.0 / 2.0) * x23;
    const Scalar x25 = -x24;
    const Scalar x26 = x17 - 1;
    const Scalar x27 = (9.0 / 2.0) * x[1];
    const Scalar x28 = x13 * x[1];
    const Scalar x29 = (9.0 / 2.0) * x28;
    const Scalar x30 = x18 - 1;
    const Scalar x31 = -x29;
    const Scalar x32 = x1 * x4;
    const Scalar x33 = x20 * x27;
    const Scalar x34 = (9.0 / 2.0) * x[2];
    const Scalar x35 = x20 * x34;
    const Scalar x36 = x1 * x5;
    const Scalar x37 = x15 * x[2];
    const Scalar x38 = (9.0 / 2.0) * x37;
    const Scalar x39 = -x38;
    const Scalar x40 = x19 - 1;
    const Scalar x41 = x0 + 2 * x[0];
    const Scalar x42 = 27 * x[2];
    const Scalar x43 = x42 * x[0];
    const Scalar x44 = x[0] - 1;
    const Scalar x45 = x44 + x[1] + 2 * x[2];
    const Scalar x46 = 27 * x[0];
    const Scalar x47 = x42 * x[1];
    const Scalar x48 = x46 * x[1];
    const Scalar x49 = x44 + 2 * x[1] + x[2];
    const Scalar x50 = 27 * x[1];
    out[0][0] = x11;
    out[0][1] = x11;
    out[0][2] = x11;
    out[1][0] =
        (1.0 / 2.0) * x2 * x3 + (1.0 / 2.0) * x2 * x9 + (1.0 / 2.0) * x3 * x9;
    out[1][1] = 0;
    out[1][2] = 0;
    out[2][0] = 0;
    out[2][1] = (1.0 / 2.0) * x12 * x13 + (1.0 / 2.0) * x12 * x4 +
                (1.0 / 2.0) * x13 * x4;
    out[2][2] = 0;
    out[3][0] = 0;
    out[3][1] = 0;
    out[3][2] = (1.0 / 2.0) * x14 * x15 + (1.0 / 2.0) * x14 * x5 +
                (1.0 / 2.0) * x15 * x5;
    out[4][0] = (9.0 / 2.0) * x16 + (9.0 / 2.0) * x7 * x[0] + (9.0 / 2.0) * x8;
    out[4][1] = x22;
    out[4][2] = x22;
    out[5][0] = -9.0 / 2.0 * x1 * x9 - 9.0 / 2.0 * x16 - 9.0 / 2.0 * x23;
    out[5][1] = x25;
    out[5][2] = x25;
    out[6][0] = x26 * x27;
    out[6][1] = x24;
    out[6][2] = 0;
    out[7][0] = x29;
    out[7][1] = x21 * x30;
    out[7][2] = 0;
    out[8][0] = x31;
    out[8][1] = -9.0 / 2.0 * x1 * x13 - 9.0 / 2.0 * x28 - 9.0 / 2.0 * x32;
    out[8][2] = x31;
    out[9][0] = x33;
    out[9][1] = (9.0 / 2.0) * x32 + (9.0 / 2.0) * x7 * x[1] + (9.0 / 2.0) * x8;
    out[9][2] = x33;
    out[10][0] = x35;
    out[10][1] = x35;
    out[10][2] = (9.0 / 2.0) * x36 + (9.0 / 2.0) * x7 * x[2] + (9.0 / 2.0) * x8;
    out[11][0] = x39;
    out[11][1] = x39;
    out[11][2] = -9.0 / 2.0 * x1 * x15 - 9.0 / 2.0 * x36 - 9.0 / 2.0 * x37;
    out[12][0] = x26 * x34;
    out[12][1] = 0;
    out[12][2] = x24;
    out[13][0] = x38;
    out[13][1] = 0;
    out[13][2] = x21 * x40;
    out[14][0] = 0;
    out[14][1] = x30 * x34;
    out[14][2] = x29;
    out[15][0] = 0;
    out[15][1] = x38;
    out[15][2] = x27 * x40;
    out[16][0] = -x41 * x42;
    out[16][1] = -x43;
    out[16][2] = -x45 * x46;
    out[17][0] = x47;
    out[17][1] = x43;
    out[17][2] = x48;
    out[18][0] = -x47;
    out[18][1] = -x42 * x49;
    out[18][2] = -x45 * x50;
    out[19][0] = -x41 * x50;
    out[19][1] = -x46 * x49;
    out[19][2] = -x48;
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
    out[1][0] = static_cast<Scalar>(3) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[1][2] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(0) / order;
    out[2][1] = static_cast<Scalar>(3) / order;
    out[2][2] = static_cast<Scalar>(0) / order;
    out[3][0] = static_cast<Scalar>(0) / order;
    out[3][1] = static_cast<Scalar>(0) / order;
    out[3][2] = static_cast<Scalar>(3) / order;
    out[4][0] = static_cast<Scalar>(1) / order;
    out[4][1] = static_cast<Scalar>(0) / order;
    out[4][2] = static_cast<Scalar>(0) / order;
    out[5][0] = static_cast<Scalar>(2) / order;
    out[5][1] = static_cast<Scalar>(0) / order;
    out[5][2] = static_cast<Scalar>(0) / order;
    out[6][0] = static_cast<Scalar>(2) / order;
    out[6][1] = static_cast<Scalar>(1) / order;
    out[6][2] = static_cast<Scalar>(0) / order;
    out[7][0] = static_cast<Scalar>(1) / order;
    out[7][1] = static_cast<Scalar>(2) / order;
    out[7][2] = static_cast<Scalar>(0) / order;
    out[8][0] = static_cast<Scalar>(0) / order;
    out[8][1] = static_cast<Scalar>(2) / order;
    out[8][2] = static_cast<Scalar>(0) / order;
    out[9][0] = static_cast<Scalar>(0) / order;
    out[9][1] = static_cast<Scalar>(1) / order;
    out[9][2] = static_cast<Scalar>(0) / order;
    out[10][0] = static_cast<Scalar>(0) / order;
    out[10][1] = static_cast<Scalar>(0) / order;
    out[10][2] = static_cast<Scalar>(1) / order;
    out[11][0] = static_cast<Scalar>(0) / order;
    out[11][1] = static_cast<Scalar>(0) / order;
    out[11][2] = static_cast<Scalar>(2) / order;
    out[12][0] = static_cast<Scalar>(2) / order;
    out[12][1] = static_cast<Scalar>(0) / order;
    out[12][2] = static_cast<Scalar>(1) / order;
    out[13][0] = static_cast<Scalar>(1) / order;
    out[13][1] = static_cast<Scalar>(0) / order;
    out[13][2] = static_cast<Scalar>(2) / order;
    out[14][0] = static_cast<Scalar>(0) / order;
    out[14][1] = static_cast<Scalar>(2) / order;
    out[14][2] = static_cast<Scalar>(1) / order;
    out[15][0] = static_cast<Scalar>(0) / order;
    out[15][1] = static_cast<Scalar>(1) / order;
    out[15][2] = static_cast<Scalar>(2) / order;
    out[16][0] = static_cast<Scalar>(1) / order;
    out[16][1] = static_cast<Scalar>(0) / order;
    out[16][2] = static_cast<Scalar>(1) / order;
    out[17][0] = static_cast<Scalar>(1) / order;
    out[17][1] = static_cast<Scalar>(1) / order;
    out[17][2] = static_cast<Scalar>(1) / order;
    out[18][0] = static_cast<Scalar>(0) / order;
    out[18][1] = static_cast<Scalar>(1) / order;
    out[18][2] = static_cast<Scalar>(1) / order;
    out[19][0] = static_cast<Scalar>(1) / order;
    out[19][1] = static_cast<Scalar>(1) / order;
    out[19][2] = static_cast<Scalar>(0) / order;
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
      out[0] = 3;
      out[1] = 0;
      out[2] = 0;
      break;
    case 2:
      out[0] = 0;
      out[1] = 3;
      out[2] = 0;
      break;
    case 3:
      out[0] = 0;
      out[1] = 0;
      out[2] = 3;
      break;
    case 4:
      out[0] = 1;
      out[1] = 0;
      out[2] = 0;
      break;
    case 5:
      out[0] = 2;
      out[1] = 0;
      out[2] = 0;
      break;
    case 6:
      out[0] = 2;
      out[1] = 1;
      out[2] = 0;
      break;
    case 7:
      out[0] = 1;
      out[1] = 2;
      out[2] = 0;
      break;
    case 8:
      out[0] = 0;
      out[1] = 2;
      out[2] = 0;
      break;
    case 9:
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
      break;
    case 10:
      out[0] = 0;
      out[1] = 0;
      out[2] = 1;
      break;
    case 11:
      out[0] = 0;
      out[1] = 0;
      out[2] = 2;
      break;
    case 12:
      out[0] = 2;
      out[1] = 0;
      out[2] = 1;
      break;
    case 13:
      out[0] = 1;
      out[1] = 0;
      out[2] = 2;
      break;
    case 14:
      out[0] = 0;
      out[1] = 2;
      out[2] = 1;
      break;
    case 15:
      out[0] = 0;
      out[1] = 1;
      out[2] = 2;
      break;
    case 16:
      out[0] = 1;
      out[1] = 0;
      out[2] = 1;
      break;
    case 17:
      out[0] = 1;
      out[1] = 1;
      out[2] = 1;
      break;
    case 18:
      out[0] = 0;
      out[1] = 1;
      out[2] = 1;
      break;
    case 19:
      out[0] = 1;
      out[1] = 1;
      out[2] = 0;
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
      idxs[2] = 4;
      idxs[3] = 5;
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 6;
      idxs[3] = 7;
      break;
    case 2:
      idxs[0] = 2;
      idxs[1] = 0;
      idxs[2] = 8;
      idxs[3] = 9;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 10;
      idxs[3] = 11;
      break;
    case 4:
      idxs[0] = 1;
      idxs[1] = 3;
      idxs[2] = 12;
      idxs[3] = 13;
      break;
    case 5:
      idxs[0] = 2;
      idxs[1] = 3;
      idxs[2] = 14;
      idxs[3] = 15;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTria>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 3;
      idxs[3] = 4;
      idxs[4] = 5;
      idxs[5] = 12;
      idxs[6] = 13;
      idxs[7] = 11;
      idxs[8] = 10;
      idxs[9] = 16;
      break;
    case 1:
      idxs[0] = 2;
      idxs[1] = 3;
      idxs[2] = 1;
      idxs[3] = 14;
      idxs[4] = 15;
      idxs[5] = 13;
      idxs[6] = 12;
      idxs[7] = 6;
      idxs[8] = 7;
      idxs[9] = 17;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 2;
      idxs[3] = 10;
      idxs[4] = 11;
      idxs[5] = 15;
      idxs[6] = 14;
      idxs[7] = 8;
      idxs[8] = 9;
      idxs[9] = 18;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 2;
      idxs[2] = 1;
      idxs[3] = 9;
      idxs[4] = 8;
      idxs[5] = 7;
      idxs[6] = 6;
      idxs[7] = 5;
      idxs[8] = 4;
      idxs[9] = 19;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElQuad>) {
    switch (subelement) {
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

template <> struct BasisLagrange<mesh::RefElTetra, 4> {
  using ref_el_t = mesh::RefElTetra;
  static constexpr dim_t order = 4;
  static constexpr dim_t num_basis_functions = 35;
  static constexpr dim_t num_interpolation_nodes = 35;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = 4 * x[1];
    const Scalar x1 = x0 - 1;
    const Scalar x2 = x[0] * x[2];
    const Scalar x3 = 4 * x[2];
    const Scalar x4 = x[2] * (x3 - 1);
    const Scalar x5 = x4 * x[0];
    const Scalar x6 = 4 * x[0];
    const Scalar x7 = x6 - 1;
    const Scalar x8 = x7 * x[0];
    const Scalar x9 = x8 * x[2];
    const Scalar x10 = x[0] + x[1] + x[2] - 1;
    const Scalar x11 = x10 * x[1];
    const Scalar x12 = 2 * x[0];
    const Scalar x13 = x12 - 1;
    const Scalar x14 = (1.0 / 2.0) * x4;
    const Scalar x15 = 2 * x[2];
    const Scalar x16 = x15 - 1;
    const Scalar x17 = x1 * x[1];
    const Scalar x18 = 2 * x[1] - 1;
    const Scalar x19 = (2.0 / 3.0) * x18;
    const Scalar x20 = x17 * x19;
    const Scalar x21 = x16 * x4;
    const Scalar x22 = (2.0 / 3.0) * x[1];
    const Scalar x23 = x10 * x6;
    const Scalar x24 = x23 * x[2];
    const Scalar x25 = x0 * x10;
    const Scalar x26 = x1 * x25;
    const Scalar x27 = x13 * x8;
    const Scalar x28 = x1 * x11;
    const Scalar x29 = (2.0 / 3.0) * x10;
    const Scalar x30 = x6 - 3;
    const Scalar x31 = x0 + x3 + x30;
    const Scalar x32 = x25 * x31;
    const Scalar x33 = (1.0 / 2.0) * x31;
    const Scalar x34 = x10 * x33;
    const Scalar x35 = x31 * (x12 + x15 + x18);
    const Scalar x36 = (2.0 / 3.0) * x35;
    const Scalar x37 = x10 * x36;
    return (1.0 / 3.0) * coeffs[0] * x10 * x35 * (x1 + x3 + x6) -
           8 * coeffs[10] * x19 * x28 + 8 * coeffs[11] * x28 * x33 -
           8 * coeffs[12] * x11 * x36 - 8 * coeffs[13] * x37 * x[2] +
           8 * coeffs[14] * x34 * x4 - 8 * coeffs[15] * x21 * x29 +
           (16.0 / 3.0) * coeffs[16] * x13 * x9 + 8 * coeffs[17] * x14 * x8 +
           (16.0 / 3.0) * coeffs[18] * x16 * x5 + 8 * coeffs[19] * x20 * x[2] +
           (1.0 / 3.0) * coeffs[1] * x27 * x30 + 8 * coeffs[20] * x14 * x17 +
           8 * coeffs[21] * x21 * x22 + 8 * coeffs[22] * x24 * x31 -
           8 * coeffs[23] * x24 * x7 - 8 * coeffs[24] * x23 * x4 +
           8 * coeffs[25] * x0 * x1 * x2 + 8 * coeffs[26] * x0 * x5 +
           8 * coeffs[27] * x0 * x9 + 8 * coeffs[28] * x32 * x[2] -
           8 * coeffs[29] * x25 * x4 +
           (1.0 / 3.0) * coeffs[2] * x17 * x18 * (x0 - 3) -
           8 * coeffs[30] * x26 * x[2] + 8 * coeffs[31] * x32 * x[0] -
           8 * coeffs[32] * x26 * x[0] - 8 * coeffs[33] * x25 * x8 -
           256 * coeffs[34] * x11 * x2 +
           (1.0 / 3.0) * coeffs[3] * x21 * (x3 - 3) -
           8 * coeffs[4] * x37 * x[0] + 8 * coeffs[5] * x34 * x8 -
           8 * coeffs[6] * x27 * x29 + 8 * coeffs[7] * x22 * x27 +
           4 * coeffs[8] * x17 * x8 + 8 * coeffs[9] * x20 * x[0];
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    const Scalar x0 = 4 * x[0];
    const Scalar x1 = x0 - 1;
    const Scalar x2 = 4 * x[1];
    const Scalar x3 = 4 * x[2];
    const Scalar x4 = x2 + x3;
    const Scalar x5 = 2 * x[1];
    const Scalar x6 = 2 * x[2];
    const Scalar x7 = 2 * x[0] - 1;
    const Scalar x8 = x[0] + x[1] + x[2] - 1;
    const Scalar x9 = x0 - 3;
    const Scalar x10 = x8 * (x4 + x9);
    const Scalar x11 = x10 * (x5 + x6 + x7);
    const Scalar x12 = x1 * x[0];
    const Scalar x13 = x12 * x7;
    const Scalar x14 = x2 - 1;
    const Scalar x15 = x14 * x[1];
    const Scalar x16 = x15 * (x5 - 1);
    const Scalar x17 = x3 - 1;
    const Scalar x18 = x17 * x[2];
    const Scalar x19 = x18 * (x6 - 1);
    const Scalar x20 = (16.0 / 3.0) * x[0];
    const Scalar x21 = x0 * x1;
    const Scalar x22 = (16.0 / 3.0) * x13;
    const Scalar x23 = (16.0 / 3.0) * x8;
    const Scalar x24 = x14 * x2;
    const Scalar x25 = (16.0 / 3.0) * x11;
    const Scalar x26 = 32 * x[2];
    const Scalar x27 = x26 * x[0];
    const Scalar x28 = x12 * x26;
    const Scalar x29 = 32 * x[0];
    const Scalar x30 = x18 * x29;
    const Scalar x31 = x10 * x[1];
    const Scalar x32 = x8 * x[1];
    const Scalar x33 = 32 * x32;
    const Scalar x34 = x15 * x8;
    out[0] = (1.0 / 3.0) * x11 * (x1 + x4);
    out[1] = (1.0 / 3.0) * x13 * x9;
    out[2] = (1.0 / 3.0) * x16 * (x2 - 3);
    out[3] = (1.0 / 3.0) * x19 * (x3 - 3);
    out[4] = -x11 * x20;
    out[5] = x10 * x21;
    out[6] = -x22 * x8;
    out[7] = x22 * x[1];
    out[8] = x15 * x21;
    out[9] = x16 * x20;
    out[10] = -x16 * x23;
    out[11] = x10 * x24;
    out[12] = -x25 * x[1];
    out[13] = -x25 * x[2];
    out[14] = x10 * x17 * x3;
    out[15] = -x19 * x23;
    out[16] = x22 * x[2];
    out[17] = x18 * x21;
    out[18] = x19 * x20;
    out[19] = (16.0 / 3.0) * x16 * x[2];
    out[20] = x18 * x24;
    out[21] = (16.0 / 3.0) * x19 * x[1];
    out[22] = x10 * x27;
    out[23] = -x28 * x8;
    out[24] = -x30 * x8;
    out[25] = x15 * x27;
    out[26] = x30 * x[1];
    out[27] = x28 * x[1];
    out[28] = x26 * x31;
    out[29] = -x18 * x33;
    out[30] = -x26 * x34;
    out[31] = x29 * x31;
    out[32] = -x29 * x34;
    out[33] = -x12 * x33;
    out[34] = -256 * x32 * x[0] * x[2];
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = 4 * x[0];
    const Scalar x1 = x0 - 1;
    const Scalar x2 = x0 - 3;
    const Scalar x3 = 4 * x[2];
    const Scalar x4 = 4 * x[1];
    const Scalar x5 = x3 + x4;
    const Scalar x6 = -x2 - x5;
    const Scalar x7 = x6 * x[0];
    const Scalar x8 = x[1] + x[2];
    const Scalar x9 = x8 + x[0] - 1;
    const Scalar x10 = -x9;
    const Scalar x11 = x0 * x10;
    const Scalar x12 = x10 * x6;
    const Scalar x13 = (1.0 / 2.0) * coeffs[5];
    const Scalar x14 = 2 * x[0];
    const Scalar x15 = x14 - 1;
    const Scalar x16 = x0 * x15;
    const Scalar x17 = x1 * x15;
    const Scalar x18 = x1 * x14;
    const Scalar x19 = 2 * x[1];
    const Scalar x20 = 2 * x[2];
    const Scalar x21 = -x15 - x19 - x20;
    const Scalar x22 = x12 * x21;
    const Scalar x23 = -x22;
    const Scalar x24 = (2.0 / 3.0) * coeffs[4];
    const Scalar x25 = x1 * x[0];
    const Scalar x26 = x15 * x25;
    const Scalar x27 = x0 * x9;
    const Scalar x28 = x1 * x9;
    const Scalar x29 = (2.0 / 3.0) * coeffs[6];
    const Scalar x30 = -x12;
    const Scalar x31 = x11 + x30 + x7;
    const Scalar x32 = x25 + x27 + x28;
    const Scalar x33 = x16 + x17 + x18;
    const Scalar x34 = x4 - 1;
    const Scalar x35 = x3 - 1;
    const Scalar x36 = 8 * x[0];
    const Scalar x37 = x36 - 1;
    const Scalar x38 = x15 + x8;
    const Scalar x39 = x3 * x35;
    const Scalar x40 = coeffs[28] * x4;
    const Scalar x41 = 8 * x[1];
    const Scalar x42 = 8 * x[2];
    const Scalar x43 = -x36 - x41 - x42 + 7;
    const Scalar x44 = x43 * x[2];
    const Scalar x45 = x35 * x[2];
    const Scalar x46 = coeffs[29] * x4;
    const Scalar x47 = x34 * x4;
    const Scalar x48 = 32 * coeffs[34] * x[2];
    const Scalar x49 = x20 - 1;
    const Scalar x50 = x19 - 1;
    const Scalar x51 = -x1 - x5;
    const Scalar x52 = x21 * x6;
    const Scalar x53 = 2 * x12;
    const Scalar x54 = x10 * x21;
    const Scalar x55 = 4 * x54;
    const Scalar x56 = (1.0 / 24.0) * coeffs[0] *
                       (4 * x22 + x51 * x52 + x51 * x53 + x51 * x55);
    const Scalar x57 = x52 + x53 + x55;
    const Scalar x58 = (2.0 / 3.0) * coeffs[13];
    const Scalar x59 = (1.0 / 2.0) * coeffs[14];
    const Scalar x60 = x45 * x49;
    const Scalar x61 = (2.0 / 3.0) * coeffs[15];
    const Scalar x62 = x43 * x45 * x59 + x56 + x57 * x58 * x[2] + x60 * x61;
    const Scalar x63 = (2.0 / 3.0) * coeffs[12];
    const Scalar x64 = x34 * x[1];
    const Scalar x65 = (1.0 / 2.0) * coeffs[11];
    const Scalar x66 = x50 * x64;
    const Scalar x67 = (2.0 / 3.0) * coeffs[10];
    const Scalar x68 = x43 * x64 * x65 + x57 * x63 * x[1] + x66 * x67;
    const Scalar x69 = x6 * x[1];
    const Scalar x70 = x10 * x4;
    const Scalar x71 = x4 - 3;
    const Scalar x72 = x4 * x50;
    const Scalar x73 = x34 * x50;
    const Scalar x74 = x19 * x34;
    const Scalar x75 = x4 * x9;
    const Scalar x76 = x34 * x9;
    const Scalar x77 = x30 + x69 + x70;
    const Scalar x78 = x64 + x75 + x76;
    const Scalar x79 = coeffs[31] * x0;
    const Scalar x80 = coeffs[32] * x0;
    const Scalar x81 = x72 + x73 + x74;
    const Scalar x82 = x41 - 1;
    const Scalar x83 = coeffs[22] * x0;
    const Scalar x84 = x0 * x1;
    const Scalar x85 = coeffs[23] * x84;
    const Scalar x86 = coeffs[24] * x0;
    const Scalar x87 = x50 + x[0] + x[2];
    const Scalar x88 = coeffs[33] * x84;
    const Scalar x89 = x13 * x25 * x43 + x24 * x57 * x[0] + x26 * x29;
    const Scalar x90 = x6 * x[2];
    const Scalar x91 = x10 * x3;
    const Scalar x92 = x3 - 3;
    const Scalar x93 = x3 * x49;
    const Scalar x94 = x35 * x49;
    const Scalar x95 = x20 * x35;
    const Scalar x96 = x3 * x9;
    const Scalar x97 = x35 * x9;
    const Scalar x98 = x30 + x90 + x91;
    const Scalar x99 = x45 + x96 + x97;
    const Scalar x100 = x93 + x94 + x95;
    const Scalar x101 = x42 - 1;
    const Scalar x102 = x49 + x[0] + x[1];
    out[0] =
        (16.0 / 3.0) * coeffs[16] * x33 * x[2] +
        4 * coeffs[17] * x35 * x37 * x[2] +
        (16.0 / 3.0) * coeffs[18] * x35 * x49 * x[2] +
        (1.0 / 3.0) * coeffs[1] * (x0 * x17 + x16 * x2 + x17 * x2 + x18 * x2) -
        8 * coeffs[22] * x3 * x31 - 8 * coeffs[23] * x3 * x32 -
        8 * coeffs[24] * x38 * x39 + 32 * coeffs[25] * x34 * x[1] * x[2] +
        32 * coeffs[26] * x35 * x[1] * x[2] +
        32 * coeffs[27] * x37 * x[1] * x[2] - 8 * coeffs[30] * x47 * x[2] -
        8 * coeffs[31] * x31 * x4 - 8 * coeffs[32] * x38 * x47 -
        8 * coeffs[33] * x32 * x4 + (16.0 / 3.0) * coeffs[7] * x33 * x[1] +
        4 * coeffs[8] * x34 * x37 * x[1] +
        (16.0 / 3.0) * coeffs[9] * x34 * x50 * x[1] -
        8 * x13 * (-x0 * x12 + x1 * x11 - x1 * x12 + x1 * x7) -
        8 * x24 * (x11 * x21 + x12 * x14 + x21 * x7 + x23) -
        8 * x29 * (x14 * x28 + x15 * x27 + x15 * x28 + x26) -
        8 * x38 * x48 * x[1] - 8 * x40 * x44 - 8 * x45 * x46 - 8 * x62 -
        8 * x68;
    out[1] =
        (16.0 / 3.0) * coeffs[19] * x81 * x[2] +
        4 * coeffs[20] * x35 * x82 * x[2] +
        (16.0 / 3.0) * coeffs[21] * x35 * x49 * x[2] +
        32 * coeffs[25] * x82 * x[0] * x[2] +
        32 * coeffs[26] * x35 * x[0] * x[2] +
        32 * coeffs[27] * x1 * x[0] * x[2] - 8 * coeffs[28] * x3 * x77 -
        8 * coeffs[29] * x39 * x87 +
        (1.0 / 3.0) * coeffs[2] *
            (x4 * x73 + x71 * x72 + x71 * x73 + x71 * x74) -
        8 * coeffs[30] * x3 * x78 + (16.0 / 3.0) * coeffs[7] * x1 * x15 * x[0] +
        4 * coeffs[8] * x1 * x82 * x[0] +
        (16.0 / 3.0) * coeffs[9] * x81 * x[0] - 8 * x44 * x83 - 8 * x45 * x86 -
        8 * x48 * x87 * x[0] - 8 * x62 -
        8 * x63 * (x12 * x19 + x23 + x4 * x54 + x52 * x[1]) -
        8 * x65 * (-x12 * x34 - x12 * x4 + x34 * x69 + x34 * x70) -
        8 * x67 * (x19 * x76 + x50 * x75 + x50 * x76 + x66) - 8 * x77 * x79 -
        8 * x78 * x80 - 8 * x85 * x[2] - 8 * x87 * x88 - 8 * x89;
    out[2] =
        (16.0 / 3.0) * coeffs[16] * x1 * x15 * x[0] +
        4 * coeffs[17] * x1 * x101 * x[0] +
        (16.0 / 3.0) * coeffs[18] * x100 * x[0] +
        (16.0 / 3.0) * coeffs[19] * x34 * x50 * x[1] +
        4 * coeffs[20] * x101 * x34 * x[1] +
        (16.0 / 3.0) * coeffs[21] * x100 * x[1] +
        32 * coeffs[25] * x34 * x[0] * x[1] +
        32 * coeffs[26] * x101 * x[0] * x[1] +
        32 * coeffs[27] * x1 * x[0] * x[1] - 8 * coeffs[30] * x102 * x47 -
        256 * coeffs[34] * x102 * x[0] * x[1] +
        (1.0 / 3.0) * coeffs[3] *
            (x3 * x94 + x92 * x93 + x92 * x94 + x92 * x95) -
        8 * x102 * x85 - 8 * x40 * x98 - 8 * x43 * x79 * x[1] - 8 * x46 * x99 -
        8 * x56 - 8 * x58 * (x12 * x20 + x23 + x3 * x54 + x52 * x[2]) -
        8 * x59 * (-x12 * x3 - x12 * x35 + x35 * x90 + x35 * x91) -
        8 * x61 * (x20 * x97 + x49 * x96 + x49 * x97 + x60) - 8 * x64 * x80 -
        8 * x68 - 8 * x83 * x98 - 8 * x86 * x99 - 8 * x88 * x[1] - 8 * x89;
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[3]) {
    const Scalar x0 = x[0] + x[1];
    const Scalar x1 = x0 + x[2] - 1;
    const Scalar x2 = 2 * x[1];
    const Scalar x3 = 2 * x[2];
    const Scalar x4 = 2 * x[0];
    const Scalar x5 = x4 - 1;
    const Scalar x6 = x2 + x3 + x5;
    const Scalar x7 = 4 * x[0];
    const Scalar x8 = x7 - 3;
    const Scalar x9 = 4 * x[1];
    const Scalar x10 = 4 * x[2];
    const Scalar x11 = x10 + x9;
    const Scalar x12 = x11 + x8;
    const Scalar x13 = x12 * x6;
    const Scalar x14 = x1 * x13;
    const Scalar x15 = x7 - 1;
    const Scalar x16 = x11 + x15;
    const Scalar x17 = x1 * x6;
    const Scalar x18 = 4 * x17;
    const Scalar x19 = x1 * x12;
    const Scalar x20 = 2 * x19;
    const Scalar x21 = (1.0 / 3.0) * x13 * x16 + (4.0 / 3.0) * x14 +
                       (1.0 / 3.0) * x16 * x18 + (1.0 / 3.0) * x16 * x20;
    const Scalar x22 = x5 * x7;
    const Scalar x23 = x15 * x5;
    const Scalar x24 = x15 * x4;
    const Scalar x25 = x9 - 3;
    const Scalar x26 = x2 - 1;
    const Scalar x27 = x26 * x9;
    const Scalar x28 = x9 - 1;
    const Scalar x29 = x26 * x28;
    const Scalar x30 = x2 * x28;
    const Scalar x31 = x10 - 3;
    const Scalar x32 = x3 - 1;
    const Scalar x33 = x10 * x32;
    const Scalar x34 = x10 - 1;
    const Scalar x35 = x32 * x34;
    const Scalar x36 = x3 * x34;
    const Scalar x37 = x13 + x18 + x20;
    const Scalar x38 = (16.0 / 3.0) * x[0];
    const Scalar x39 = -x37 * x38;
    const Scalar x40 = x1 * x15;
    const Scalar x41 = x12 * x[0];
    const Scalar x42 = 8 * x[0];
    const Scalar x43 = 8 * x[1];
    const Scalar x44 = 8 * x[2];
    const Scalar x45 = x42 + x43 + x44 - 7;
    const Scalar x46 = x15 * x7;
    const Scalar x47 = x45 * x46;
    const Scalar x48 = x23 * x38;
    const Scalar x49 = -x48;
    const Scalar x50 = x22 + x23 + x24;
    const Scalar x51 = (16.0 / 3.0) * x[1];
    const Scalar x52 = x42 - 1;
    const Scalar x53 = x28 * x9;
    const Scalar x54 = x43 - 1;
    const Scalar x55 = x29 * x51;
    const Scalar x56 = x27 + x29 + x30;
    const Scalar x57 = -x55;
    const Scalar x58 = x1 * x28;
    const Scalar x59 = x45 * x53;
    const Scalar x60 = x12 * x[1];
    const Scalar x61 = -x37 * x51;
    const Scalar x62 = (16.0 / 3.0) * x[2];
    const Scalar x63 = -x37 * x62;
    const Scalar x64 = x10 * x34;
    const Scalar x65 = x45 * x64;
    const Scalar x66 = x1 * x34;
    const Scalar x67 = x12 * x[2];
    const Scalar x68 = x35 * x62;
    const Scalar x69 = -x68;
    const Scalar x70 = x44 - 1;
    const Scalar x71 = x33 + x35 + x36;
    const Scalar x72 = x1 * x7;
    const Scalar x73 = x19 + x41 + x72;
    const Scalar x74 = 32 * x[2];
    const Scalar x75 = x74 * x[0];
    const Scalar x76 = x1 * x10;
    const Scalar x77 = x19 + x67 + x76;
    const Scalar x78 = 32 * x[0];
    const Scalar x79 = x15 * x[0];
    const Scalar x80 = x40 + x72 + x79;
    const Scalar x81 = x74 * x79;
    const Scalar x82 = x0 + x32;
    const Scalar x83 = 32 * x82;
    const Scalar x84 = x5 + x[1] + x[2];
    const Scalar x85 = x34 * x[2];
    const Scalar x86 = 32 * x85;
    const Scalar x87 = x78 * x85;
    const Scalar x88 = x66 + x76 + x85;
    const Scalar x89 = x28 * x[1];
    const Scalar x90 = x74 * x89;
    const Scalar x91 = x78 * x89;
    const Scalar x92 = 32 * x[1];
    const Scalar x93 = x85 * x92;
    const Scalar x94 = x78 * x[1];
    const Scalar x95 = x74 * x[1];
    const Scalar x96 = x79 * x92;
    const Scalar x97 = x1 * x9;
    const Scalar x98 = x19 + x60 + x97;
    const Scalar x99 = x26 + x[0] + x[2];
    const Scalar x100 = x58 + x89 + x97;
    const Scalar x101 = 256 * x[2];
    out[0][0] = x21;
    out[0][1] = x21;
    out[0][2] = x21;
    out[1][0] = (1.0 / 3.0) * x22 * x8 + (1.0 / 3.0) * x23 * x7 +
                (1.0 / 3.0) * x23 * x8 + (1.0 / 3.0) * x24 * x8;
    out[1][1] = 0;
    out[1][2] = 0;
    out[2][0] = 0;
    out[2][1] = (1.0 / 3.0) * x25 * x27 + (1.0 / 3.0) * x25 * x29 +
                (1.0 / 3.0) * x25 * x30 + (1.0 / 3.0) * x29 * x9;
    out[2][2] = 0;
    out[3][0] = 0;
    out[3][1] = 0;
    out[3][2] = (1.0 / 3.0) * x10 * x35 + (1.0 / 3.0) * x31 * x33 +
                (1.0 / 3.0) * x31 * x35 + (1.0 / 3.0) * x31 * x36;
    out[4][0] = -16.0 / 3.0 * x13 * x[0] - 16.0 / 3.0 * x14 -
                16.0 / 3.0 * x17 * x7 - 16.0 / 3.0 * x19 * x4;
    out[4][1] = x39;
    out[4][2] = x39;
    out[5][0] = 4 * x15 * x19 + 4 * x15 * x41 + 4 * x19 * x7 + 4 * x40 * x7;
    out[5][1] = x47;
    out[5][2] = x47;
    out[6][0] = -16.0 / 3.0 * x1 * x22 - 16.0 / 3.0 * x1 * x23 -
                16.0 / 3.0 * x23 * x[0] - 16.0 / 3.0 * x4 * x40;
    out[6][1] = x49;
    out[6][2] = x49;
    out[7][0] = x50 * x51;
    out[7][1] = x48;
    out[7][2] = 0;
    out[8][0] = x52 * x53;
    out[8][1] = x46 * x54;
    out[8][2] = 0;
    out[9][0] = x55;
    out[9][1] = x38 * x56;
    out[9][2] = 0;
    out[10][0] = x57;
    out[10][1] = -16.0 / 3.0 * x1 * x27 - 16.0 / 3.0 * x1 * x29 -
                 16.0 / 3.0 * x2 * x58 - 16.0 / 3.0 * x29 * x[1];
    out[10][2] = x57;
    out[11][0] = x59;
    out[11][1] = 4 * x19 * x28 + 4 * x19 * x9 + 4 * x28 * x60 + 4 * x58 * x9;
    out[11][2] = x59;
    out[12][0] = x61;
    out[12][1] = -16.0 / 3.0 * x13 * x[1] - 16.0 / 3.0 * x14 -
                 16.0 / 3.0 * x17 * x9 - 16.0 / 3.0 * x19 * x2;
    out[12][2] = x61;
    out[13][0] = x63;
    out[13][1] = x63;
    out[13][2] = -16.0 / 3.0 * x10 * x17 - 16.0 / 3.0 * x13 * x[2] -
                 16.0 / 3.0 * x14 - 16.0 / 3.0 * x19 * x3;
    out[14][0] = x65;
    out[14][1] = x65;
    out[14][2] = 4 * x10 * x19 + 4 * x10 * x66 + 4 * x19 * x34 + 4 * x34 * x67;
    out[15][0] = x69;
    out[15][1] = x69;
    out[15][2] = -16.0 / 3.0 * x1 * x33 - 16.0 / 3.0 * x1 * x35 -
                 16.0 / 3.0 * x3 * x66 - 16.0 / 3.0 * x35 * x[2];
    out[16][0] = x50 * x62;
    out[16][1] = 0;
    out[16][2] = x48;
    out[17][0] = x52 * x64;
    out[17][1] = 0;
    out[17][2] = x46 * x70;
    out[18][0] = x68;
    out[18][1] = 0;
    out[18][2] = x38 * x71;
    out[19][0] = 0;
    out[19][1] = x56 * x62;
    out[19][2] = x55;
    out[20][0] = 0;
    out[20][1] = x54 * x64;
    out[20][2] = x53 * x70;
    out[21][0] = 0;
    out[21][1] = x68;
    out[21][2] = x51 * x71;
    out[22][0] = x73 * x74;
    out[22][1] = x45 * x75;
    out[22][2] = x77 * x78;
    out[23][0] = -x74 * x80;
    out[23][1] = -x81;
    out[23][2] = -x79 * x83;
    out[24][0] = -x84 * x86;
    out[24][1] = -x87;
    out[24][2] = -x78 * x88;
    out[25][0] = x90;
    out[25][1] = x54 * x75;
    out[25][2] = x91;
    out[26][0] = x93;
    out[26][1] = x87;
    out[26][2] = x70 * x94;
    out[27][0] = x52 * x95;
    out[27][1] = x81;
    out[27][2] = x96;
    out[28][0] = x45 * x95;
    out[28][1] = x74 * x98;
    out[28][2] = x77 * x92;
    out[29][0] = -x93;
    out[29][1] = -x86 * x99;
    out[29][2] = -x88 * x92;
    out[30][0] = -x90;
    out[30][1] = -x100 * x74;
    out[30][2] = -x83 * x89;
    out[31][0] = x73 * x92;
    out[31][1] = x78 * x98;
    out[31][2] = x45 * x94;
    out[32][0] = -32 * x84 * x89;
    out[32][1] = -x100 * x78;
    out[32][2] = -x91;
    out[33][0] = -x80 * x92;
    out[33][1] = -32 * x79 * x99;
    out[33][2] = -x96;
    out[34][0] = -x101 * x84 * x[1];
    out[34][1] = -x101 * x99 * x[0];
    out[34][2] = -256 * x82 * x[0] * x[1];
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
    out[1][0] = static_cast<Scalar>(4) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[1][2] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(0) / order;
    out[2][1] = static_cast<Scalar>(4) / order;
    out[2][2] = static_cast<Scalar>(0) / order;
    out[3][0] = static_cast<Scalar>(0) / order;
    out[3][1] = static_cast<Scalar>(0) / order;
    out[3][2] = static_cast<Scalar>(4) / order;
    out[4][0] = static_cast<Scalar>(1) / order;
    out[4][1] = static_cast<Scalar>(0) / order;
    out[4][2] = static_cast<Scalar>(0) / order;
    out[5][0] = static_cast<Scalar>(2) / order;
    out[5][1] = static_cast<Scalar>(0) / order;
    out[5][2] = static_cast<Scalar>(0) / order;
    out[6][0] = static_cast<Scalar>(3) / order;
    out[6][1] = static_cast<Scalar>(0) / order;
    out[6][2] = static_cast<Scalar>(0) / order;
    out[7][0] = static_cast<Scalar>(3) / order;
    out[7][1] = static_cast<Scalar>(1) / order;
    out[7][2] = static_cast<Scalar>(0) / order;
    out[8][0] = static_cast<Scalar>(2) / order;
    out[8][1] = static_cast<Scalar>(2) / order;
    out[8][2] = static_cast<Scalar>(0) / order;
    out[9][0] = static_cast<Scalar>(1) / order;
    out[9][1] = static_cast<Scalar>(3) / order;
    out[9][2] = static_cast<Scalar>(0) / order;
    out[10][0] = static_cast<Scalar>(0) / order;
    out[10][1] = static_cast<Scalar>(3) / order;
    out[10][2] = static_cast<Scalar>(0) / order;
    out[11][0] = static_cast<Scalar>(0) / order;
    out[11][1] = static_cast<Scalar>(2) / order;
    out[11][2] = static_cast<Scalar>(0) / order;
    out[12][0] = static_cast<Scalar>(0) / order;
    out[12][1] = static_cast<Scalar>(1) / order;
    out[12][2] = static_cast<Scalar>(0) / order;
    out[13][0] = static_cast<Scalar>(0) / order;
    out[13][1] = static_cast<Scalar>(0) / order;
    out[13][2] = static_cast<Scalar>(1) / order;
    out[14][0] = static_cast<Scalar>(0) / order;
    out[14][1] = static_cast<Scalar>(0) / order;
    out[14][2] = static_cast<Scalar>(2) / order;
    out[15][0] = static_cast<Scalar>(0) / order;
    out[15][1] = static_cast<Scalar>(0) / order;
    out[15][2] = static_cast<Scalar>(3) / order;
    out[16][0] = static_cast<Scalar>(3) / order;
    out[16][1] = static_cast<Scalar>(0) / order;
    out[16][2] = static_cast<Scalar>(1) / order;
    out[17][0] = static_cast<Scalar>(2) / order;
    out[17][1] = static_cast<Scalar>(0) / order;
    out[17][2] = static_cast<Scalar>(2) / order;
    out[18][0] = static_cast<Scalar>(1) / order;
    out[18][1] = static_cast<Scalar>(0) / order;
    out[18][2] = static_cast<Scalar>(3) / order;
    out[19][0] = static_cast<Scalar>(0) / order;
    out[19][1] = static_cast<Scalar>(3) / order;
    out[19][2] = static_cast<Scalar>(1) / order;
    out[20][0] = static_cast<Scalar>(0) / order;
    out[20][1] = static_cast<Scalar>(2) / order;
    out[20][2] = static_cast<Scalar>(2) / order;
    out[21][0] = static_cast<Scalar>(0) / order;
    out[21][1] = static_cast<Scalar>(1) / order;
    out[21][2] = static_cast<Scalar>(3) / order;
    out[22][0] = static_cast<Scalar>(1) / order;
    out[22][1] = static_cast<Scalar>(0) / order;
    out[22][2] = static_cast<Scalar>(1) / order;
    out[23][0] = static_cast<Scalar>(2) / order;
    out[23][1] = static_cast<Scalar>(0) / order;
    out[23][2] = static_cast<Scalar>(1) / order;
    out[24][0] = static_cast<Scalar>(1) / order;
    out[24][1] = static_cast<Scalar>(0) / order;
    out[24][2] = static_cast<Scalar>(2) / order;
    out[25][0] = static_cast<Scalar>(1) / order;
    out[25][1] = static_cast<Scalar>(2) / order;
    out[25][2] = static_cast<Scalar>(1) / order;
    out[26][0] = static_cast<Scalar>(1) / order;
    out[26][1] = static_cast<Scalar>(1) / order;
    out[26][2] = static_cast<Scalar>(2) / order;
    out[27][0] = static_cast<Scalar>(2) / order;
    out[27][1] = static_cast<Scalar>(1) / order;
    out[27][2] = static_cast<Scalar>(1) / order;
    out[28][0] = static_cast<Scalar>(0) / order;
    out[28][1] = static_cast<Scalar>(1) / order;
    out[28][2] = static_cast<Scalar>(1) / order;
    out[29][0] = static_cast<Scalar>(0) / order;
    out[29][1] = static_cast<Scalar>(1) / order;
    out[29][2] = static_cast<Scalar>(2) / order;
    out[30][0] = static_cast<Scalar>(0) / order;
    out[30][1] = static_cast<Scalar>(2) / order;
    out[30][2] = static_cast<Scalar>(1) / order;
    out[31][0] = static_cast<Scalar>(1) / order;
    out[31][1] = static_cast<Scalar>(1) / order;
    out[31][2] = static_cast<Scalar>(0) / order;
    out[32][0] = static_cast<Scalar>(1) / order;
    out[32][1] = static_cast<Scalar>(2) / order;
    out[32][2] = static_cast<Scalar>(0) / order;
    out[33][0] = static_cast<Scalar>(2) / order;
    out[33][1] = static_cast<Scalar>(1) / order;
    out[33][2] = static_cast<Scalar>(0) / order;
    out[34][0] = static_cast<Scalar>(1) / order;
    out[34][1] = static_cast<Scalar>(1) / order;
    out[34][2] = static_cast<Scalar>(1) / order;
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
      out[0] = 4;
      out[1] = 0;
      out[2] = 0;
      break;
    case 2:
      out[0] = 0;
      out[1] = 4;
      out[2] = 0;
      break;
    case 3:
      out[0] = 0;
      out[1] = 0;
      out[2] = 4;
      break;
    case 4:
      out[0] = 1;
      out[1] = 0;
      out[2] = 0;
      break;
    case 5:
      out[0] = 2;
      out[1] = 0;
      out[2] = 0;
      break;
    case 6:
      out[0] = 3;
      out[1] = 0;
      out[2] = 0;
      break;
    case 7:
      out[0] = 3;
      out[1] = 1;
      out[2] = 0;
      break;
    case 8:
      out[0] = 2;
      out[1] = 2;
      out[2] = 0;
      break;
    case 9:
      out[0] = 1;
      out[1] = 3;
      out[2] = 0;
      break;
    case 10:
      out[0] = 0;
      out[1] = 3;
      out[2] = 0;
      break;
    case 11:
      out[0] = 0;
      out[1] = 2;
      out[2] = 0;
      break;
    case 12:
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
      break;
    case 13:
      out[0] = 0;
      out[1] = 0;
      out[2] = 1;
      break;
    case 14:
      out[0] = 0;
      out[1] = 0;
      out[2] = 2;
      break;
    case 15:
      out[0] = 0;
      out[1] = 0;
      out[2] = 3;
      break;
    case 16:
      out[0] = 3;
      out[1] = 0;
      out[2] = 1;
      break;
    case 17:
      out[0] = 2;
      out[1] = 0;
      out[2] = 2;
      break;
    case 18:
      out[0] = 1;
      out[1] = 0;
      out[2] = 3;
      break;
    case 19:
      out[0] = 0;
      out[1] = 3;
      out[2] = 1;
      break;
    case 20:
      out[0] = 0;
      out[1] = 2;
      out[2] = 2;
      break;
    case 21:
      out[0] = 0;
      out[1] = 1;
      out[2] = 3;
      break;
    case 22:
      out[0] = 1;
      out[1] = 0;
      out[2] = 1;
      break;
    case 23:
      out[0] = 2;
      out[1] = 0;
      out[2] = 1;
      break;
    case 24:
      out[0] = 1;
      out[1] = 0;
      out[2] = 2;
      break;
    case 25:
      out[0] = 1;
      out[1] = 2;
      out[2] = 1;
      break;
    case 26:
      out[0] = 1;
      out[1] = 1;
      out[2] = 2;
      break;
    case 27:
      out[0] = 2;
      out[1] = 1;
      out[2] = 1;
      break;
    case 28:
      out[0] = 0;
      out[1] = 1;
      out[2] = 1;
      break;
    case 29:
      out[0] = 0;
      out[1] = 1;
      out[2] = 2;
      break;
    case 30:
      out[0] = 0;
      out[1] = 2;
      out[2] = 1;
      break;
    case 31:
      out[0] = 1;
      out[1] = 1;
      out[2] = 0;
      break;
    case 32:
      out[0] = 1;
      out[1] = 2;
      out[2] = 0;
      break;
    case 33:
      out[0] = 2;
      out[1] = 1;
      out[2] = 0;
      break;
    case 34:
      out[0] = 1;
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
      idxs[2] = 4;
      idxs[3] = 5;
      idxs[4] = 6;
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 7;
      idxs[3] = 8;
      idxs[4] = 9;
      break;
    case 2:
      idxs[0] = 2;
      idxs[1] = 0;
      idxs[2] = 10;
      idxs[3] = 11;
      idxs[4] = 12;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 13;
      idxs[3] = 14;
      idxs[4] = 15;
      break;
    case 4:
      idxs[0] = 1;
      idxs[1] = 3;
      idxs[2] = 16;
      idxs[3] = 17;
      idxs[4] = 18;
      break;
    case 5:
      idxs[0] = 2;
      idxs[1] = 3;
      idxs[2] = 19;
      idxs[3] = 20;
      idxs[4] = 21;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTria>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 3;
      idxs[3] = 4;
      idxs[4] = 5;
      idxs[5] = 6;
      idxs[6] = 16;
      idxs[7] = 17;
      idxs[8] = 18;
      idxs[9] = 15;
      idxs[10] = 14;
      idxs[11] = 13;
      idxs[12] = 22;
      idxs[13] = 23;
      idxs[14] = 24;
      break;
    case 1:
      idxs[0] = 2;
      idxs[1] = 3;
      idxs[2] = 1;
      idxs[3] = 19;
      idxs[4] = 20;
      idxs[5] = 21;
      idxs[6] = 18;
      idxs[7] = 17;
      idxs[8] = 16;
      idxs[9] = 7;
      idxs[10] = 8;
      idxs[11] = 9;
      idxs[12] = 25;
      idxs[13] = 26;
      idxs[14] = 27;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 2;
      idxs[3] = 13;
      idxs[4] = 14;
      idxs[5] = 15;
      idxs[6] = 21;
      idxs[7] = 20;
      idxs[8] = 19;
      idxs[9] = 10;
      idxs[10] = 11;
      idxs[11] = 12;
      idxs[12] = 28;
      idxs[13] = 29;
      idxs[14] = 30;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 2;
      idxs[2] = 1;
      idxs[3] = 12;
      idxs[4] = 11;
      idxs[5] = 10;
      idxs[6] = 9;
      idxs[7] = 8;
      idxs[8] = 7;
      idxs[9] = 6;
      idxs[10] = 5;
      idxs[11] = 4;
      idxs[12] = 31;
      idxs[13] = 32;
      idxs[14] = 33;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElQuad>) {
    switch (subelement) {
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

template <> struct BasisLagrange<mesh::RefElTetra, 5> {
  using ref_el_t = mesh::RefElTetra;
  static constexpr dim_t order = 5;
  static constexpr dim_t num_basis_functions = 56;
  static constexpr dim_t num_interpolation_nodes = 56;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = 5 * x[1];
    const Scalar x1 = x[1] * (x0 - 1);
    const Scalar x2 = x1 * (x0 - 2);
    const Scalar x3 = x[0] * x[2];
    const Scalar x4 = 5 * x[2];
    const Scalar x5 = x[2] * (x4 - 1);
    const Scalar x6 = x5 * (x4 - 2);
    const Scalar x7 = x[0] * x[1];
    const Scalar x8 = 5 * x[0];
    const Scalar x9 = x8 - 2;
    const Scalar x10 = x8 - 1;
    const Scalar x11 = x10 * x[0];
    const Scalar x12 = x11 * x9;
    const Scalar x13 = x[1] * x[2];
    const Scalar x14 = 750 * x5;
    const Scalar x15 = x14 * x[0];
    const Scalar x16 = x5 * x[1];
    const Scalar x17 = 750 * x11;
    const Scalar x18 = 750 * x1;
    const Scalar x19 = x[0] + x[1] + x[2] - 1;
    const Scalar x20 = x11 * x19;
    const Scalar x21 = 7500 * x13;
    const Scalar x22 = x1 * x19;
    const Scalar x23 = x19 * x[0];
    const Scalar x24 = 50 * x2;
    const Scalar x25 = x2 * (x0 - 3);
    const Scalar x26 = 25 * x[0];
    const Scalar x27 = x8 - 3;
    const Scalar x28 = x12 * x27;
    const Scalar x29 = 25 * x[2];
    const Scalar x30 = 50 * x12;
    const Scalar x31 = 50 * x6;
    const Scalar x32 = x6 * (x4 - 3);
    const Scalar x33 = 25 * x[1];
    const Scalar x34 = 500 * x19;
    const Scalar x35 = x34 * x[2];
    const Scalar x36 = 500 * x23;
    const Scalar x37 = x34 * x[1];
    const Scalar x38 = x8 - 4;
    const Scalar x39 = x0 + x4;
    const Scalar x40 = x19 * (x38 + x39);
    const Scalar x41 = x40 * x[0];
    const Scalar x42 = 25 * x19;
    const Scalar x43 = x40 * x[2];
    const Scalar x44 = x40 * (x27 + x39);
    const Scalar x45 = 500 * x44;
    const Scalar x46 = 50 * x44;
    const Scalar x47 = x44 * (x39 + x9);
    const Scalar x48 = 25 * x47;
    return -1.0 / 24.0 * coeffs[0] * x47 * (x10 + x39) +
           (1.0 / 24.0) * coeffs[10] * x11 * x24 +
           (1.0 / 24.0) * coeffs[11] * x25 * x26 -
           1.0 / 24.0 * coeffs[12] * x25 * x42 +
           (1.0 / 24.0) * coeffs[13] * x24 * x40 -
           1.0 / 24.0 * coeffs[14] * x1 * x46 +
           (1.0 / 24.0) * coeffs[15] * x48 * x[1] +
           (1.0 / 24.0) * coeffs[16] * x48 * x[2] -
           1.0 / 24.0 * coeffs[17] * x46 * x5 +
           (1.0 / 24.0) * coeffs[18] * x31 * x40 -
           1.0 / 24.0 * coeffs[19] * x32 * x42 +
           (1.0 / 24.0) * coeffs[1] * x28 * x38 +
           (1.0 / 24.0) * coeffs[20] * x28 * x29 +
           (1.0 / 24.0) * coeffs[21] * x30 * x5 +
           (1.0 / 24.0) * coeffs[22] * x11 * x31 +
           (1.0 / 24.0) * coeffs[23] * x26 * x32 +
           (1.0 / 24.0) * coeffs[24] * x25 * x29 +
           (1.0 / 24.0) * coeffs[25] * x24 * x5 +
           (1.0 / 24.0) * coeffs[26] * x1 * x31 +
           (1.0 / 24.0) * coeffs[27] * x32 * x33 -
           1.0 / 24.0 * coeffs[28] * x3 * x45 -
           1.0 / 24.0 * coeffs[29] * x12 * x35 +
           (1.0 / 24.0) * coeffs[2] * x25 * (x0 - 4) -
           1.0 / 24.0 * coeffs[30] * x36 * x6 +
           (1.0 / 24.0) * coeffs[31] * x17 * x43 -
           1.0 / 24.0 * coeffs[32] * x14 * x20 +
           (1.0 / 24.0) * coeffs[33] * x15 * x40 +
           (125.0 / 6.0) * coeffs[34] * x2 * x3 +
           (125.0 / 6.0) * coeffs[35] * x6 * x7 +
           (125.0 / 6.0) * coeffs[36] * x12 * x13 +
           (1.0 / 24.0) * coeffs[37] * x1 * x15 +
           (1.0 / 24.0) * coeffs[38] * x16 * x17 +
           (1.0 / 24.0) * coeffs[39] * x11 * x18 * x[2] +
           (1.0 / 24.0) * coeffs[3] * x32 * (x4 - 4) -
           1.0 / 24.0 * coeffs[40] * x13 * x45 -
           1.0 / 24.0 * coeffs[41] * x37 * x6 -
           1.0 / 24.0 * coeffs[42] * x2 * x35 +
           (125.0 / 4.0) * coeffs[43] * x16 * x40 -
           1.0 / 24.0 * coeffs[44] * x14 * x22 +
           (1.0 / 24.0) * coeffs[45] * x18 * x43 -
           1.0 / 24.0 * coeffs[46] * x45 * x7 -
           1.0 / 24.0 * coeffs[47] * x2 * x36 -
           1.0 / 24.0 * coeffs[48] * x12 * x37 +
           (1.0 / 24.0) * coeffs[49] * x18 * x41 +
           (1.0 / 24.0) * coeffs[4] * x26 * x47 -
           1.0 / 24.0 * coeffs[50] * x18 * x20 +
           (1.0 / 24.0) * coeffs[51] * x17 * x40 * x[1] +
           (1.0 / 24.0) * coeffs[52] * x21 * x41 -
           1.0 / 24.0 * coeffs[53] * x20 * x21 -
           625.0 / 2.0 * coeffs[54] * x22 * x3 -
           625.0 / 2.0 * coeffs[55] * x16 * x23 -
           1.0 / 24.0 * coeffs[5] * x11 * x46 +
           (1.0 / 24.0) * coeffs[6] * x30 * x40 -
           1.0 / 24.0 * coeffs[7] * x28 * x42 +
           (1.0 / 24.0) * coeffs[8] * x28 * x33 +
           (1.0 / 24.0) * coeffs[9] * x1 * x30;
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    const Scalar x0 = 5 * x[0];
    const Scalar x1 = x0 - 1;
    const Scalar x2 = 5 * x[1];
    const Scalar x3 = 5 * x[2];
    const Scalar x4 = x2 + x3;
    const Scalar x5 = x0 - 2;
    const Scalar x6 = x0 - 3;
    const Scalar x7 = x[0] + x[1] + x[2] - 1;
    const Scalar x8 = x0 - 4;
    const Scalar x9 = x7 * (x4 + x8);
    const Scalar x10 = x9 * (x4 + x6);
    const Scalar x11 = x10 * (x4 + x5);
    const Scalar x12 = x1 * x[0];
    const Scalar x13 = x12 * x5;
    const Scalar x14 = x13 * x6;
    const Scalar x15 = x[1] * (x2 - 1);
    const Scalar x16 = x15 * (x2 - 2);
    const Scalar x17 = x16 * (x2 - 3);
    const Scalar x18 = x[2] * (x3 - 1);
    const Scalar x19 = x18 * (x3 - 2);
    const Scalar x20 = x19 * (x3 - 3);
    const Scalar x21 = (25.0 / 24.0) * x[0];
    const Scalar x22 = (25.0 / 12.0) * x12;
    const Scalar x23 = (25.0 / 12.0) * x13;
    const Scalar x24 = (25.0 / 24.0) * x14;
    const Scalar x25 = (25.0 / 24.0) * x7;
    const Scalar x26 = (25.0 / 12.0) * x9;
    const Scalar x27 = (25.0 / 12.0) * x10;
    const Scalar x28 = (25.0 / 24.0) * x11;
    const Scalar x29 = (125.0 / 6.0) * x[2];
    const Scalar x30 = x29 * x[0];
    const Scalar x31 = x13 * x29;
    const Scalar x32 = (125.0 / 6.0) * x[0];
    const Scalar x33 = x19 * x32;
    const Scalar x34 = (125.0 / 4.0) * x12;
    const Scalar x35 = x9 * x[2];
    const Scalar x36 = x18 * x34;
    const Scalar x37 = (125.0 / 4.0) * x18;
    const Scalar x38 = x37 * x[0];
    const Scalar x39 = x15 * x34;
    const Scalar x40 = x10 * x[1];
    const Scalar x41 = x7 * x[1];
    const Scalar x42 = (125.0 / 6.0) * x41;
    const Scalar x43 = x16 * x7;
    const Scalar x44 = x9 * x[1];
    const Scalar x45 = x15 * x7;
    const Scalar x46 = (125.0 / 4.0) * x15;
    const Scalar x47 = (625.0 / 2.0) * x[0];
    out[0] = -1.0 / 24.0 * x11 * (x1 + x4);
    out[1] = (1.0 / 24.0) * x14 * x8;
    out[2] = (1.0 / 24.0) * x17 * (x2 - 4);
    out[3] = (1.0 / 24.0) * x20 * (x3 - 4);
    out[4] = x11 * x21;
    out[5] = -x10 * x22;
    out[6] = x23 * x9;
    out[7] = -x24 * x7;
    out[8] = x24 * x[1];
    out[9] = x15 * x23;
    out[10] = x16 * x22;
    out[11] = x17 * x21;
    out[12] = -x17 * x25;
    out[13] = x16 * x26;
    out[14] = -x15 * x27;
    out[15] = x28 * x[1];
    out[16] = x28 * x[2];
    out[17] = -x18 * x27;
    out[18] = x19 * x26;
    out[19] = -x20 * x25;
    out[20] = x24 * x[2];
    out[21] = x18 * x23;
    out[22] = x19 * x22;
    out[23] = x20 * x21;
    out[24] = (25.0 / 24.0) * x17 * x[2];
    out[25] = (25.0 / 12.0) * x16 * x18;
    out[26] = (25.0 / 12.0) * x15 * x19;
    out[27] = (25.0 / 24.0) * x20 * x[1];
    out[28] = -x10 * x30;
    out[29] = -x31 * x7;
    out[30] = -x33 * x7;
    out[31] = x34 * x35;
    out[32] = -x36 * x7;
    out[33] = x38 * x9;
    out[34] = x16 * x30;
    out[35] = x33 * x[1];
    out[36] = x31 * x[1];
    out[37] = x15 * x38;
    out[38] = x36 * x[1];
    out[39] = x39 * x[2];
    out[40] = -x29 * x40;
    out[41] = -x19 * x42;
    out[42] = -x29 * x43;
    out[43] = x37 * x44;
    out[44] = -x37 * x45;
    out[45] = x35 * x46;
    out[46] = -x32 * x40;
    out[47] = -x32 * x43;
    out[48] = -x13 * x42;
    out[49] = x46 * x9 * x[0];
    out[50] = -x39 * x7;
    out[51] = x34 * x44;
    out[52] = x35 * x47 * x[1];
    out[53] = -625.0 / 2.0 * x12 * x41 * x[2];
    out[54] = -x45 * x47 * x[2];
    out[55] = -x18 * x41 * x47;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = 5 * x[0];
    const Scalar x1 = x0 - 4;
    const Scalar x2 = x0 - 3;
    const Scalar x3 = x0 - 2;
    const Scalar x4 = x0 * x3;
    const Scalar x5 = x2 * x4;
    const Scalar x6 = x0 - 1;
    const Scalar x7 = x0 * x6;
    const Scalar x8 = x2 * x7;
    const Scalar x9 = x3 * x6;
    const Scalar x10 = x0 * x9;
    const Scalar x11 = x2 * x9;
    const Scalar x12 = x[1] + x[2] - 1;
    const Scalar x13 = x12 + x[0];
    const Scalar x14 = x13 * x6;
    const Scalar x15 = x14 * x3;
    const Scalar x16 = x0 * x15;
    const Scalar x17 = 5 * x[2];
    const Scalar x18 = 5 * x[1];
    const Scalar x19 = x17 + x18;
    const Scalar x20 = x1 + x19;
    const Scalar x21 = x6 * x[0];
    const Scalar x22 = x21 * x3;
    const Scalar x23 = x0 * x13;
    const Scalar x24 = x23 * x3;
    const Scalar x25 = x0 * x14;
    const Scalar x26 = -x19 - x3;
    const Scalar x27 = -x19 - x2;
    const Scalar x28 = -x20;
    const Scalar x29 = x28 * x[0];
    const Scalar x30 = x27 * x29;
    const Scalar x31 = -x13;
    const Scalar x32 = x0 * x31;
    const Scalar x33 = x27 * x32;
    const Scalar x34 = x28 * x31;
    const Scalar x35 = x0 * x34;
    const Scalar x36 = x27 * x34;
    const Scalar x37 = x0 * x36;
    const Scalar x38 = x26 * x36;
    const Scalar x39 = -x38;
    const Scalar x40 = 25 * coeffs[4];
    const Scalar x41 = x2 * x22;
    const Scalar x42 = 25 * coeffs[7];
    const Scalar x43 = x29 * x6;
    const Scalar x44 = x32 * x6;
    const Scalar x45 = x34 * x6;
    const Scalar x46 = 50 * coeffs[5];
    const Scalar x47 = x10 + x11 + x5 + x8;
    const Scalar x48 = -x36;
    const Scalar x49 = x30 + x33 + x35 + x48;
    const Scalar x50 = 500 * x[2];
    const Scalar x51 = coeffs[28] * x50;
    const Scalar x52 = x15 + x22 + x24 + x25;
    const Scalar x53 = coeffs[29] * x50;
    const Scalar x54 = 500 * x[1];
    const Scalar x55 = coeffs[46] * x54;
    const Scalar x56 = coeffs[48] * x54;
    const Scalar x57 = -750 * x35 + 750 * x43 + 750 * x44 - 750 * x45;
    const Scalar x58 = coeffs[31] * x[2];
    const Scalar x59 = coeffs[51] * x[1];
    const Scalar x60 = x4 + x7 + x9;
    const Scalar x61 = x17 - 1;
    const Scalar x62 = x18 - 1;
    const Scalar x63 = x27 * x28;
    const Scalar x64 = x27 * x31;
    const Scalar x65 = 5 * x64;
    const Scalar x66 = 5 * x34;
    const Scalar x67 = x63 + x65 + x66;
    const Scalar x68 = coeffs[40] * x50;
    const Scalar x69 = x14 + x21 + x23;
    const Scalar x70 = x61 * x[2];
    const Scalar x71 = 750 * x70;
    const Scalar x72 = coeffs[32] * x71;
    const Scalar x73 = -x34;
    const Scalar x74 = x29 + x32 + x73;
    const Scalar x75 = x62 * x[1];
    const Scalar x76 = 750 * x75;
    const Scalar x77 = coeffs[50] * x76;
    const Scalar x78 = 7500 * x[2];
    const Scalar x79 = x78 * x[1];
    const Scalar x80 = 10 * x[0];
    const Scalar x81 = x80 - 1;
    const Scalar x82 = x18 - 2;
    const Scalar x83 = x17 - 2;
    const Scalar x84 = x18 - 3;
    const Scalar x85 = x17 - 3;
    const Scalar x86 = x70 * x83;
    const Scalar x87 = x12 + 2 * x[0];
    const Scalar x88 = 500 * x87;
    const Scalar x89 = coeffs[41] * x54;
    const Scalar x90 = x75 * x82;
    const Scalar x91 = coeffs[42] * x50;
    const Scalar x92 = coeffs[43] * x[1];
    const Scalar x93 = 10 * x[1];
    const Scalar x94 = 10 * x[2];
    const Scalar x95 = -x80 - x93 - x94 + 9;
    const Scalar x96 = x71 * x95;
    const Scalar x97 = coeffs[44] * x71;
    const Scalar x98 = coeffs[45] * x[2];
    const Scalar x99 = x76 * x95;
    const Scalar x100 = coeffs[54] * x75;
    const Scalar x101 = 7500 * x[1];
    const Scalar x102 = coeffs[55] * x70;
    const Scalar x103 = -x19 - x6;
    const Scalar x104 = x26 * x63;
    const Scalar x105 = x26 * x65;
    const Scalar x106 = x26 * x66;
    const Scalar x107 = 5 * x36;
    const Scalar x108 = coeffs[0] * (x103 * x104 + x103 * x105 + x103 * x106 +
                                     x103 * x107 + 5 * x38);
    const Scalar x109 = x104 + x105 + x106 + x107;
    const Scalar x110 = 25 * coeffs[16];
    const Scalar x111 = 50 * coeffs[17];
    const Scalar x112 = 25 * coeffs[19];
    const Scalar x113 = x85 * x86;
    const Scalar x114 = 50 * coeffs[18] * x86 * x95 + x108 +
                        x109 * x110 * x[2] + x111 * x67 * x70 + x112 * x113;
    const Scalar x115 = 25 * coeffs[15];
    const Scalar x116 = 50 * coeffs[14];
    const Scalar x117 = 25 * coeffs[12];
    const Scalar x118 = x84 * x90;
    const Scalar x119 = 50 * coeffs[13] * x90 * x95 + x109 * x115 * x[1] +
                        x116 * x67 * x75 + x117 * x118;
    const Scalar x120 = x18 - 4;
    const Scalar x121 = x18 * x82;
    const Scalar x122 = x121 * x84;
    const Scalar x123 = x18 * x62;
    const Scalar x124 = x123 * x84;
    const Scalar x125 = x62 * x82;
    const Scalar x126 = x125 * x18;
    const Scalar x127 = x125 * x84;
    const Scalar x128 = x13 * x62;
    const Scalar x129 = x128 * x82;
    const Scalar x130 = x129 * x18;
    const Scalar x131 = x13 * x18;
    const Scalar x132 = x131 * x82;
    const Scalar x133 = x128 * x18;
    const Scalar x134 = x18 * x64;
    const Scalar x135 = x18 * x34;
    const Scalar x136 = x18 * x36;
    const Scalar x137 = x34 * x62;
    const Scalar x138 = x122 + x124 + x126 + x127;
    const Scalar x139 = x134 + x135 + x48 + x63 * x[1];
    const Scalar x140 = x129 + x132 + x133 + x90;
    const Scalar x141 = 500 * x[0];
    const Scalar x142 = coeffs[47] * x141;
    const Scalar x143 = x28 * x[1];
    const Scalar x144 = x18 * x31;
    const Scalar x145 =
        -750 * x135 - 750 * x137 + 750 * x143 * x62 + 750 * x144 * x62;
    const Scalar x146 = coeffs[49] * x[0];
    const Scalar x147 = x121 + x123 + x125;
    const Scalar x148 = x67 * x[0];
    const Scalar x149 = x143 + x144 + x73;
    const Scalar x150 = x128 + x131 + x75;
    const Scalar x151 = 750 * x21;
    const Scalar x152 = x78 * x[0];
    const Scalar x153 = x93 - 1;
    const Scalar x154 = coeffs[30] * x141;
    const Scalar x155 = x[0] - 1;
    const Scalar x156 = x155 + 2 * x[1] + x[2];
    const Scalar x157 = 500 * x156;
    const Scalar x158 = x151 * x95;
    const Scalar x159 = coeffs[33] * x[0];
    const Scalar x160 = coeffs[53] * x21;
    const Scalar x161 = 7500 * x[0];
    const Scalar x162 = 50 * coeffs[6] * x22 * x95 + x109 * x40 * x[0] +
                        x21 * x46 * x67 + x41 * x42;
    const Scalar x163 = x17 - 4;
    const Scalar x164 = x17 * x83;
    const Scalar x165 = x164 * x85;
    const Scalar x166 = x17 * x61;
    const Scalar x167 = x166 * x85;
    const Scalar x168 = x61 * x83;
    const Scalar x169 = x168 * x17;
    const Scalar x170 = x168 * x85;
    const Scalar x171 = x13 * x61;
    const Scalar x172 = x171 * x83;
    const Scalar x173 = x17 * x172;
    const Scalar x174 = x13 * x17;
    const Scalar x175 = x174 * x83;
    const Scalar x176 = x17 * x171;
    const Scalar x177 = x17 * x64;
    const Scalar x178 = x17 * x34;
    const Scalar x179 = x17 * x36;
    const Scalar x180 = x34 * x61;
    const Scalar x181 = x165 + x167 + x169 + x170;
    const Scalar x182 = x177 + x178 + x48 + x63 * x[2];
    const Scalar x183 = x172 + x175 + x176 + x86;
    const Scalar x184 = x28 * x[2];
    const Scalar x185 = x17 * x31;
    const Scalar x186 =
        -750 * x178 - 750 * x180 + 750 * x184 * x61 + 750 * x185 * x61;
    const Scalar x187 = x164 + x166 + x168;
    const Scalar x188 = x184 + x185 + x73;
    const Scalar x189 = x171 + x174 + x70;
    const Scalar x190 = x101 * x[0];
    const Scalar x191 = x94 - 1;
    const Scalar x192 = x155 + x[1] + 2 * x[2];
    const Scalar x193 = 500 * x192;
    out[0] =
        (25.0 / 12.0) * coeffs[10] * x62 * x81 * x82 * x[1] +
        (25.0 / 24.0) * coeffs[11] * x62 * x82 * x84 * x[1] +
        (1.0 / 24.0) * coeffs[1] *
            (x0 * x11 + x1 * x10 + x1 * x11 + x1 * x5 + x1 * x8) +
        (25.0 / 24.0) * coeffs[20] * x47 * x[2] +
        (25.0 / 12.0) * coeffs[21] * x60 * x61 * x[2] +
        (25.0 / 12.0) * coeffs[22] * x61 * x81 * x83 * x[2] +
        (25.0 / 24.0) * coeffs[23] * x61 * x83 * x85 * x[2] -
        1.0 / 24.0 * coeffs[30] * x86 * x88 -
        1.0 / 24.0 * coeffs[33] * x71 * x74 +
        (125.0 / 6.0) * coeffs[34] * x62 * x82 * x[1] * x[2] +
        (125.0 / 6.0) * coeffs[35] * x61 * x83 * x[1] * x[2] +
        (125.0 / 6.0) * coeffs[36] * x60 * x[1] * x[2] +
        (125.0 / 4.0) * coeffs[37] * x61 * x62 * x[1] * x[2] +
        (125.0 / 4.0) * coeffs[38] * x61 * x81 * x[1] * x[2] +
        (125.0 / 4.0) * coeffs[39] * x62 * x81 * x[1] * x[2] -
        1.0 / 24.0 * coeffs[47] * x88 * x90 -
        1.0 / 24.0 * coeffs[49] * x74 * x76 -
        1.0 / 24.0 * coeffs[52] * x74 * x79 -
        1.0 / 24.0 * coeffs[53] * x69 * x79 +
        (25.0 / 12.0) * coeffs[6] *
            (x15 * x20 + x16 + x20 * x22 + x20 * x24 + x20 * x25) +
        (25.0 / 24.0) * coeffs[8] * x47 * x[1] +
        (25.0 / 12.0) * coeffs[9] * x60 * x62 * x[1] -
        1.0 / 24.0 * x100 * x78 * x87 - 1.0 / 24.0 * x101 * x102 * x87 -
        1.0 / 24.0 * x114 - 1.0 / 24.0 * x119 -
        1.0 / 24.0 * x40 * (x26 * x30 + x26 * x33 + x26 * x35 + x37 + x39) -
        1.0 / 24.0 * x42 * (x15 * x2 + x16 + x2 * x24 + x2 * x25 + x41) -
        1.0 / 24.0 * x46 *
            (x0 * x45 + x27 * x43 + x27 * x44 - x27 * x45 - x37) -
        1.0 / 24.0 * x49 * x51 - 1.0 / 24.0 * x49 * x55 -
        1.0 / 24.0 * x52 * x53 - 1.0 / 24.0 * x52 * x56 -
        1.0 / 24.0 * x57 * x58 - 1.0 / 24.0 * x57 * x59 -
        1.0 / 24.0 * x67 * x68 * x[1] - 1.0 / 24.0 * x69 * x72 -
        1.0 / 24.0 * x69 * x77 - 1.0 / 24.0 * x75 * x97 -
        1.0 / 24.0 * x86 * x89 - 1.0 / 24.0 * x90 * x91 -
        1.0 / 24.0 * x92 * x96 - 1.0 / 24.0 * x98 * x99;
    out[1] = (25.0 / 12.0) * coeffs[10] * x147 * x6 * x[0] +
             (25.0 / 24.0) * coeffs[11] * x138 * x[0] +
             (25.0 / 12.0) * coeffs[13] *
                 (x129 * x20 + x130 + x132 * x20 + x133 * x20 + x20 * x90) +
             (25.0 / 24.0) * coeffs[24] * x138 * x[2] +
             (25.0 / 12.0) * coeffs[25] * x147 * x61 * x[2] +
             (25.0 / 12.0) * coeffs[26] * x153 * x61 * x83 * x[2] +
             (25.0 / 24.0) * coeffs[27] * x61 * x83 * x85 * x[2] +
             (1.0 / 24.0) * coeffs[2] *
                 (x120 * x122 + x120 * x124 + x120 * x126 + x120 * x127 +
                  x127 * x18) +
             (125.0 / 6.0) * coeffs[34] * x147 * x[0] * x[2] +
             (125.0 / 6.0) * coeffs[35] * x61 * x83 * x[0] * x[2] +
             (125.0 / 6.0) * coeffs[36] * x3 * x6 * x[0] * x[2] +
             (125.0 / 4.0) * coeffs[37] * x153 * x61 * x[0] * x[2] +
             (125.0 / 4.0) * coeffs[38] * x6 * x61 * x[0] * x[2] +
             (125.0 / 4.0) * coeffs[39] * x153 * x6 * x[0] * x[2] -
             1.0 / 24.0 * coeffs[41] * x157 * x86 -
             1.0 / 24.0 * coeffs[43] * x149 * x71 -
             1.0 / 24.0 * coeffs[46] * x139 * x141 -
             1.0 / 24.0 * coeffs[48] * x157 * x22 -
             1.0 / 24.0 * coeffs[50] * x150 * x151 -
             1.0 / 24.0 * coeffs[51] * x149 * x151 -
             1.0 / 24.0 * coeffs[52] * x149 * x152 -
             1.0 / 24.0 * coeffs[54] * x150 * x152 +
             (25.0 / 24.0) * coeffs[8] * x2 * x3 * x6 * x[0] +
             (25.0 / 12.0) * coeffs[9] * x153 * x3 * x6 * x[0] -
             1.0 / 24.0 * x102 * x156 * x161 - 1.0 / 24.0 * x114 -
             1.0 / 24.0 * x115 *
                 (x104 * x[1] + x134 * x26 + x135 * x26 + x136 + x39) -
             1.0 / 24.0 * x116 *
                 (x123 * x64 - x136 + x137 * x18 - x36 * x62 + x63 * x75) -
             1.0 / 24.0 * x117 *
                 (x118 + x129 * x84 + x130 + x132 * x84 + x133 * x84) -
             1.0 / 24.0 * x139 * x68 - 1.0 / 24.0 * x140 * x142 -
             1.0 / 24.0 * x140 * x91 - 1.0 / 24.0 * x145 * x146 -
             1.0 / 24.0 * x145 * x98 - 1.0 / 24.0 * x148 * x51 -
             1.0 / 24.0 * x150 * x97 - 1.0 / 24.0 * x154 * x86 -
             1.0 / 24.0 * x156 * x160 * x78 - 1.0 / 24.0 * x158 * x58 -
             1.0 / 24.0 * x159 * x96 - 1.0 / 24.0 * x162 -
             1.0 / 24.0 * x21 * x72 - 1.0 / 24.0 * x22 * x53;
    out[2] = (25.0 / 12.0) * coeffs[18] *
                 (x172 * x20 + x173 + x175 * x20 + x176 * x20 + x20 * x86) +
             (25.0 / 24.0) * coeffs[20] * x2 * x3 * x6 * x[0] +
             (25.0 / 12.0) * coeffs[21] * x191 * x3 * x6 * x[0] +
             (25.0 / 12.0) * coeffs[22] * x187 * x6 * x[0] +
             (25.0 / 24.0) * coeffs[23] * x181 * x[0] +
             (25.0 / 24.0) * coeffs[24] * x62 * x82 * x84 * x[1] +
             (25.0 / 12.0) * coeffs[25] * x191 * x62 * x82 * x[1] +
             (25.0 / 12.0) * coeffs[26] * x187 * x62 * x[1] +
             (25.0 / 24.0) * coeffs[27] * x181 * x[1] -
             1.0 / 24.0 * coeffs[28] * x141 * x182 -
             1.0 / 24.0 * coeffs[29] * x193 * x22 -
             1.0 / 24.0 * coeffs[31] * x151 * x188 -
             1.0 / 24.0 * coeffs[32] * x151 * x189 +
             (125.0 / 6.0) * coeffs[34] * x62 * x82 * x[0] * x[1] +
             (125.0 / 6.0) * coeffs[35] * x187 * x[0] * x[1] +
             (125.0 / 6.0) * coeffs[36] * x3 * x6 * x[0] * x[1] +
             (125.0 / 4.0) * coeffs[37] * x191 * x62 * x[0] * x[1] +
             (125.0 / 4.0) * coeffs[38] * x191 * x6 * x[0] * x[1] +
             (125.0 / 4.0) * coeffs[39] * x6 * x62 * x[0] * x[1] +
             (1.0 / 24.0) * coeffs[3] *
                 (x163 * x165 + x163 * x167 + x163 * x169 + x163 * x170 +
                  x17 * x170) -
             1.0 / 24.0 * coeffs[40] * x182 * x54 -
             1.0 / 24.0 * coeffs[42] * x193 * x90 -
             1.0 / 24.0 * coeffs[44] * x189 * x76 -
             1.0 / 24.0 * coeffs[45] * x188 * x76 -
             1.0 / 24.0 * coeffs[52] * x188 * x190 -
             1.0 / 24.0 * coeffs[55] * x189 * x190 -
             1.0 / 24.0 * x100 * x161 * x192 - 1.0 / 24.0 * x101 * x160 * x192 -
             1.0 / 24.0 * x108 -
             1.0 / 24.0 * x110 *
                 (x104 * x[2] + x177 * x26 + x178 * x26 + x179 + x39) -
             1.0 / 24.0 * x111 *
                 (x166 * x64 + x17 * x180 - x179 - x36 * x61 + x63 * x70) -
             1.0 / 24.0 * x112 *
                 (x113 + x172 * x85 + x173 + x175 * x85 + x176 * x85) -
             1.0 / 24.0 * x119 - 1.0 / 24.0 * x142 * x90 -
             1.0 / 24.0 * x146 * x99 - 1.0 / 24.0 * x148 * x55 -
             1.0 / 24.0 * x154 * x183 - 1.0 / 24.0 * x158 * x59 -
             1.0 / 24.0 * x159 * x186 - 1.0 / 24.0 * x162 -
             1.0 / 24.0 * x183 * x89 - 1.0 / 24.0 * x186 * x92 -
             1.0 / 24.0 * x21 * x77 - 1.0 / 24.0 * x22 * x56;
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[3]) {
    const Scalar x0 = 5 * x[0];
    const Scalar x1 = x0 - 2;
    const Scalar x2 = 5 * x[1];
    const Scalar x3 = 5 * x[2];
    const Scalar x4 = x2 + x3;
    const Scalar x5 = x1 + x4;
    const Scalar x6 = x0 - 4;
    const Scalar x7 = x4 + x6;
    const Scalar x8 = x0 - 3;
    const Scalar x9 = x4 + x8;
    const Scalar x10 = x7 * x9;
    const Scalar x11 = x10 * x5;
    const Scalar x12 = x[0] + x[1] - 1;
    const Scalar x13 = x12 + x[2];
    const Scalar x14 = 5 * x13;
    const Scalar x15 = x0 - 1;
    const Scalar x16 = x15 + x4;
    const Scalar x17 = x10 * x14;
    const Scalar x18 = x13 * x7;
    const Scalar x19 = 5 * x18;
    const Scalar x20 = x19 * x5;
    const Scalar x21 = x14 * x9;
    const Scalar x22 = x21 * x5;
    const Scalar x23 = -1.0 / 24.0 * x11 * x14 - 1.0 / 24.0 * x11 * x16 -
                       1.0 / 24.0 * x16 * x17 - 1.0 / 24.0 * x16 * x20 -
                       1.0 / 24.0 * x16 * x22;
    const Scalar x24 = x0 * x1;
    const Scalar x25 = x24 * x8;
    const Scalar x26 = x0 * x15;
    const Scalar x27 = x26 * x8;
    const Scalar x28 = x1 * x15;
    const Scalar x29 = x0 * x28;
    const Scalar x30 = x28 * x8;
    const Scalar x31 = x2 - 4;
    const Scalar x32 = x2 - 3;
    const Scalar x33 = x2 - 2;
    const Scalar x34 = x2 * x33;
    const Scalar x35 = x32 * x34;
    const Scalar x36 = x2 - 1;
    const Scalar x37 = x2 * x36;
    const Scalar x38 = x32 * x37;
    const Scalar x39 = x33 * x36;
    const Scalar x40 = x2 * x39;
    const Scalar x41 = x32 * x39;
    const Scalar x42 = x3 - 4;
    const Scalar x43 = x3 - 3;
    const Scalar x44 = x3 - 2;
    const Scalar x45 = x3 * x44;
    const Scalar x46 = x43 * x45;
    const Scalar x47 = x3 - 1;
    const Scalar x48 = x3 * x47;
    const Scalar x49 = x43 * x48;
    const Scalar x50 = x44 * x47;
    const Scalar x51 = x3 * x50;
    const Scalar x52 = x43 * x50;
    const Scalar x53 = x10 * x13;
    const Scalar x54 = x0 * x53;
    const Scalar x55 = x0 * x18;
    const Scalar x56 = x0 * x13;
    const Scalar x57 = x56 * x9;
    const Scalar x58 = x11 * x13;
    const Scalar x59 = x11 + x17 + x20 + x22;
    const Scalar x60 = (25.0 / 24.0) * x[0];
    const Scalar x61 = x59 * x60;
    const Scalar x62 = x18 * x26;
    const Scalar x63 = x13 * x15;
    const Scalar x64 = x0 * x63;
    const Scalar x65 = x15 * x[0];
    const Scalar x66 = x10 + x19 + x21;
    const Scalar x67 = (25.0 / 12.0) * x65;
    const Scalar x68 = -x66 * x67;
    const Scalar x69 = x13 * x28;
    const Scalar x70 = x0 * x69;
    const Scalar x71 = x7 * x[0];
    const Scalar x72 = 10 * x[0];
    const Scalar x73 = 10 * x[1];
    const Scalar x74 = 10 * x[2];
    const Scalar x75 = x72 + x73 + x74 - 9;
    const Scalar x76 = x28 * x[0];
    const Scalar x77 = (25.0 / 12.0) * x76;
    const Scalar x78 = x75 * x77;
    const Scalar x79 = x30 * x60;
    const Scalar x80 = -x79;
    const Scalar x81 = x25 + x27 + x29 + x30;
    const Scalar x82 = (25.0 / 24.0) * x[1];
    const Scalar x83 = x24 + x26 + x28;
    const Scalar x84 = x36 * x[1];
    const Scalar x85 = (25.0 / 12.0) * x84;
    const Scalar x86 = x73 - 1;
    const Scalar x87 = x72 - 1;
    const Scalar x88 = x39 * x[1];
    const Scalar x89 = (25.0 / 12.0) * x88;
    const Scalar x90 = x34 + x37 + x39;
    const Scalar x91 = x41 * x82;
    const Scalar x92 = x35 + x38 + x40 + x41;
    const Scalar x93 = -x91;
    const Scalar x94 = x13 * x36;
    const Scalar x95 = x2 * x94;
    const Scalar x96 = x13 * x39;
    const Scalar x97 = x2 * x96;
    const Scalar x98 = x75 * x89;
    const Scalar x99 = x7 * x[1];
    const Scalar x100 = x18 * x37;
    const Scalar x101 = -x66 * x85;
    const Scalar x102 = x2 * x53;
    const Scalar x103 = x59 * x82;
    const Scalar x104 = x18 * x2;
    const Scalar x105 = x13 * x2;
    const Scalar x106 = x105 * x9;
    const Scalar x107 = (25.0 / 24.0) * x[2];
    const Scalar x108 = x107 * x59;
    const Scalar x109 = x3 * x53;
    const Scalar x110 = x18 * x3;
    const Scalar x111 = x13 * x3;
    const Scalar x112 = x111 * x9;
    const Scalar x113 = x47 * x[2];
    const Scalar x114 = (25.0 / 12.0) * x113;
    const Scalar x115 = -x114 * x66;
    const Scalar x116 = x18 * x48;
    const Scalar x117 = x13 * x47;
    const Scalar x118 = x117 * x3;
    const Scalar x119 = x50 * x[2];
    const Scalar x120 = (25.0 / 12.0) * x119;
    const Scalar x121 = x120 * x75;
    const Scalar x122 = x13 * x50;
    const Scalar x123 = x122 * x3;
    const Scalar x124 = x7 * x[2];
    const Scalar x125 = x107 * x52;
    const Scalar x126 = -x125;
    const Scalar x127 = x74 - 1;
    const Scalar x128 = x45 + x48 + x50;
    const Scalar x129 = x46 + x49 + x51 + x52;
    const Scalar x130 = x10 * x[0] + x53 + x55 + x57;
    const Scalar x131 = (125.0 / 6.0) * x[2];
    const Scalar x132 = x131 * x[0];
    const Scalar x133 = x10 * x[2] + x110 + x112 + x53;
    const Scalar x134 = (125.0 / 6.0) * x[0];
    const Scalar x135 = x13 * x24 + x64 + x69 + x76;
    const Scalar x136 = x131 * x76;
    const Scalar x137 = x12 + 2 * x[2];
    const Scalar x138 = x134 * x28;
    const Scalar x139 = x[2] - 1;
    const Scalar x140 = x139 + 2 * x[0] + x[1];
    const Scalar x141 = x131 * x50;
    const Scalar x142 = x132 * x50;
    const Scalar x143 = x118 + x119 + x122 + x13 * x45;
    const Scalar x144 = x15 * x18 + x55 + x64 + x65 * x7;
    const Scalar x145 = (125.0 / 4.0) * x[2];
    const Scalar x146 = x145 * x65;
    const Scalar x147 = x111 + x124 + x18;
    const Scalar x148 = (125.0 / 4.0) * x65;
    const Scalar x149 = x56 + x63 + x65;
    const Scalar x150 = (125.0 / 4.0) * x113;
    const Scalar x151 = x113 * x148;
    const Scalar x152 = x111 + x113 + x117;
    const Scalar x153 = x18 + x56 + x71;
    const Scalar x154 = (125.0 / 4.0) * x[0];
    const Scalar x155 = x113 * x154;
    const Scalar x156 = (125.0 / 4.0) * x110 + (125.0 / 4.0) * x118 +
                        (125.0 / 4.0) * x124 * x47 + (125.0 / 4.0) * x18 * x47;
    const Scalar x157 = x131 * x88;
    const Scalar x158 = x134 * x88;
    const Scalar x159 = x131 * x[1];
    const Scalar x160 = x159 * x50;
    const Scalar x161 = x134 * x[1];
    const Scalar x162 = x161 * x28;
    const Scalar x163 = x150 * x84;
    const Scalar x164 = x154 * x84;
    const Scalar x165 = (125.0 / 4.0) * x[1];
    const Scalar x166 = x113 * x165;
    const Scalar x167 = x165 * x65;
    const Scalar x168 = x145 * x84;
    const Scalar x169 = x148 * x84;
    const Scalar x170 = x10 * x[1] + x104 + x106 + x53;
    const Scalar x171 = (125.0 / 6.0) * x[1];
    const Scalar x172 = x139 + x[0] + 2 * x[1];
    const Scalar x173 = x13 * x34 + x88 + x95 + x96;
    const Scalar x174 = x171 * x39;
    const Scalar x175 = x105 + x18 + x99;
    const Scalar x176 = x105 + x84 + x94;
    const Scalar x177 = (125.0 / 4.0) * x84;
    const Scalar x178 = x104 + x18 * x36 + x36 * x99 + x95;
    const Scalar x179 = (625.0 / 2.0) * x[2];
    const Scalar x180 = x179 * x[1];
    const Scalar x181 = x179 * x[0];
    const Scalar x182 = (625.0 / 2.0) * x[1];
    const Scalar x183 = x182 * x[0];
    const Scalar x184 = (625.0 / 2.0) * x[0];
    out[0][0] = x23;
    out[0][1] = x23;
    out[0][2] = x23;
    out[1][0] = (1.0 / 24.0) * x0 * x30 + (1.0 / 24.0) * x25 * x6 +
                (1.0 / 24.0) * x27 * x6 + (1.0 / 24.0) * x29 * x6 +
                (1.0 / 24.0) * x30 * x6;
    out[1][1] = 0;
    out[1][2] = 0;
    out[2][0] = 0;
    out[2][1] = (1.0 / 24.0) * x2 * x41 + (1.0 / 24.0) * x31 * x35 +
                (1.0 / 24.0) * x31 * x38 + (1.0 / 24.0) * x31 * x40 +
                (1.0 / 24.0) * x31 * x41;
    out[2][2] = 0;
    out[3][0] = 0;
    out[3][1] = 0;
    out[3][2] = (1.0 / 24.0) * x3 * x52 + (1.0 / 24.0) * x42 * x46 +
                (1.0 / 24.0) * x42 * x49 + (1.0 / 24.0) * x42 * x51 +
                (1.0 / 24.0) * x42 * x52;
    out[4][0] = (25.0 / 24.0) * x11 * x[0] + (25.0 / 24.0) * x5 * x55 +
                (25.0 / 24.0) * x5 * x57 + (25.0 / 24.0) * x54 +
                (25.0 / 24.0) * x58;
    out[4][1] = x61;
    out[4][2] = x61;
    out[5][0] = -25.0 / 12.0 * x10 * x63 - 25.0 / 12.0 * x10 * x65 -
                25.0 / 12.0 * x54 - 25.0 / 12.0 * x62 - 25.0 / 12.0 * x64 * x9;
    out[5][1] = x68;
    out[5][2] = x68;
    out[6][0] = (25.0 / 12.0) * x18 * x24 + (25.0 / 12.0) * x18 * x28 +
                (25.0 / 12.0) * x28 * x71 + (25.0 / 12.0) * x62 +
                (25.0 / 12.0) * x70;
    out[6][1] = x78;
    out[6][2] = x78;
    out[7][0] = -25.0 / 24.0 * x13 * x25 - 25.0 / 24.0 * x13 * x30 -
                25.0 / 24.0 * x30 * x[0] - 25.0 / 24.0 * x64 * x8 -
                25.0 / 24.0 * x70;
    out[7][1] = x80;
    out[7][2] = x80;
    out[8][0] = x81 * x82;
    out[8][1] = x79;
    out[8][2] = 0;
    out[9][0] = x83 * x85;
    out[9][1] = x77 * x86;
    out[9][2] = 0;
    out[10][0] = x87 * x89;
    out[10][1] = x67 * x90;
    out[10][2] = 0;
    out[11][0] = x91;
    out[11][1] = x60 * x92;
    out[11][2] = 0;
    out[12][0] = x93;
    out[12][1] = -25.0 / 24.0 * x13 * x35 - 25.0 / 24.0 * x13 * x41 -
                 25.0 / 24.0 * x32 * x95 - 25.0 / 24.0 * x41 * x[1] -
                 25.0 / 24.0 * x97;
    out[12][2] = x93;
    out[13][0] = x98;
    out[13][1] = (25.0 / 12.0) * x100 + (25.0 / 12.0) * x18 * x34 +
                 (25.0 / 12.0) * x18 * x39 + (25.0 / 12.0) * x39 * x99 +
                 (25.0 / 12.0) * x97;
    out[13][2] = x98;
    out[14][0] = x101;
    out[14][1] = -25.0 / 12.0 * x10 * x84 - 25.0 / 12.0 * x10 * x94 -
                 25.0 / 12.0 * x100 - 25.0 / 12.0 * x102 -
                 25.0 / 12.0 * x9 * x95;
    out[14][2] = x101;
    out[15][0] = x103;
    out[15][1] = (25.0 / 24.0) * x102 + (25.0 / 24.0) * x104 * x5 +
                 (25.0 / 24.0) * x106 * x5 + (25.0 / 24.0) * x11 * x[1] +
                 (25.0 / 24.0) * x58;
    out[15][2] = x103;
    out[16][0] = x108;
    out[16][1] = x108;
    out[16][2] = (25.0 / 24.0) * x109 + (25.0 / 24.0) * x11 * x[2] +
                 (25.0 / 24.0) * x110 * x5 + (25.0 / 24.0) * x112 * x5 +
                 (25.0 / 24.0) * x58;
    out[17][0] = x115;
    out[17][1] = x115;
    out[17][2] = -25.0 / 12.0 * x10 * x113 - 25.0 / 12.0 * x10 * x117 -
                 25.0 / 12.0 * x109 - 25.0 / 12.0 * x116 -
                 25.0 / 12.0 * x118 * x9;
    out[18][0] = x121;
    out[18][1] = x121;
    out[18][2] = (25.0 / 12.0) * x116 + (25.0 / 12.0) * x123 +
                 (25.0 / 12.0) * x124 * x50 + (25.0 / 12.0) * x18 * x45 +
                 (25.0 / 12.0) * x18 * x50;
    out[19][0] = x126;
    out[19][1] = x126;
    out[19][2] = -25.0 / 24.0 * x118 * x43 - 25.0 / 24.0 * x123 -
                 25.0 / 24.0 * x13 * x46 - 25.0 / 24.0 * x13 * x52 -
                 25.0 / 24.0 * x52 * x[2];
    out[20][0] = x107 * x81;
    out[20][1] = 0;
    out[20][2] = x79;
    out[21][0] = x114 * x83;
    out[21][1] = 0;
    out[21][2] = x127 * x77;
    out[22][0] = x120 * x87;
    out[22][1] = 0;
    out[22][2] = x128 * x67;
    out[23][0] = x125;
    out[23][1] = 0;
    out[23][2] = x129 * x60;
    out[24][0] = 0;
    out[24][1] = x107 * x92;
    out[24][2] = x91;
    out[25][0] = 0;
    out[25][1] = x114 * x90;
    out[25][2] = x127 * x89;
    out[26][0] = 0;
    out[26][1] = x120 * x86;
    out[26][2] = x128 * x85;
    out[27][0] = 0;
    out[27][1] = x125;
    out[27][2] = x129 * x82;
    out[28][0] = -x130 * x131;
    out[28][1] = -x132 * x66;
    out[28][2] = -x133 * x134;
    out[29][0] = -x131 * x135;
    out[29][1] = -x136;
    out[29][2] = -x137 * x138;
    out[30][0] = -x140 * x141;
    out[30][1] = -x142;
    out[30][2] = -x134 * x143;
    out[31][0] = x144 * x145;
    out[31][1] = x146 * x75;
    out[31][2] = x147 * x148;
    out[32][0] = -x149 * x150;
    out[32][1] = -x151;
    out[32][2] = -x148 * x152;
    out[33][0] = x150 * x153;
    out[33][1] = x155 * x75;
    out[33][2] = x156 * x[0];
    out[34][0] = x157;
    out[34][1] = x132 * x90;
    out[34][2] = x158;
    out[35][0] = x160;
    out[35][1] = x142;
    out[35][2] = x128 * x161;
    out[36][0] = x159 * x83;
    out[36][1] = x136;
    out[36][2] = x162;
    out[37][0] = x163;
    out[37][1] = x155 * x86;
    out[37][2] = x127 * x164;
    out[38][0] = x166 * x87;
    out[38][1] = x151;
    out[38][2] = x127 * x167;
    out[39][0] = x168 * x87;
    out[39][1] = x146 * x86;
    out[39][2] = x169;
    out[40][0] = -x159 * x66;
    out[40][1] = -x131 * x170;
    out[40][2] = -x133 * x171;
    out[41][0] = -x160;
    out[41][1] = -x141 * x172;
    out[41][2] = -x143 * x171;
    out[42][0] = -x157;
    out[42][1] = -x131 * x173;
    out[42][2] = -x137 * x174;
    out[43][0] = x166 * x75;
    out[43][1] = x150 * x175;
    out[43][2] = x156 * x[1];
    out[44][0] = -x163;
    out[44][1] = -x150 * x176;
    out[44][2] = -x152 * x177;
    out[45][0] = x168 * x75;
    out[45][1] = x145 * x178;
    out[45][2] = x147 * x177;
    out[46][0] = -x130 * x171;
    out[46][1] = -x134 * x170;
    out[46][2] = -x161 * x66;
    out[47][0] = -x140 * x174;
    out[47][1] = -x134 * x173;
    out[47][2] = -x158;
    out[48][0] = -x135 * x171;
    out[48][1] = -x138 * x172;
    out[48][2] = -x162;
    out[49][0] = x153 * x177;
    out[49][1] = x154 * x178;
    out[49][2] = x164 * x75;
    out[50][0] = -x149 * x177;
    out[50][1] = -x148 * x176;
    out[50][2] = -x169;
    out[51][0] = x144 * x165;
    out[51][1] = x148 * x175;
    out[51][2] = x167 * x75;
    out[52][0] = x153 * x180;
    out[52][1] = x175 * x181;
    out[52][2] = x147 * x183;
    out[53][0] = -x149 * x180;
    out[53][1] = -x172 * x179 * x65;
    out[53][2] = -x137 * x182 * x65;
    out[54][0] = -x140 * x179 * x84;
    out[54][1] = -x176 * x181;
    out[54][2] = -x137 * x184 * x84;
    out[55][0] = -x113 * x140 * x182;
    out[55][1] = -x113 * x172 * x184;
    out[55][2] = -x152 * x183;
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
    out[1][0] = static_cast<Scalar>(5) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[1][2] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(0) / order;
    out[2][1] = static_cast<Scalar>(5) / order;
    out[2][2] = static_cast<Scalar>(0) / order;
    out[3][0] = static_cast<Scalar>(0) / order;
    out[3][1] = static_cast<Scalar>(0) / order;
    out[3][2] = static_cast<Scalar>(5) / order;
    out[4][0] = static_cast<Scalar>(1) / order;
    out[4][1] = static_cast<Scalar>(0) / order;
    out[4][2] = static_cast<Scalar>(0) / order;
    out[5][0] = static_cast<Scalar>(2) / order;
    out[5][1] = static_cast<Scalar>(0) / order;
    out[5][2] = static_cast<Scalar>(0) / order;
    out[6][0] = static_cast<Scalar>(3) / order;
    out[6][1] = static_cast<Scalar>(0) / order;
    out[6][2] = static_cast<Scalar>(0) / order;
    out[7][0] = static_cast<Scalar>(4) / order;
    out[7][1] = static_cast<Scalar>(0) / order;
    out[7][2] = static_cast<Scalar>(0) / order;
    out[8][0] = static_cast<Scalar>(4) / order;
    out[8][1] = static_cast<Scalar>(1) / order;
    out[8][2] = static_cast<Scalar>(0) / order;
    out[9][0] = static_cast<Scalar>(3) / order;
    out[9][1] = static_cast<Scalar>(2) / order;
    out[9][2] = static_cast<Scalar>(0) / order;
    out[10][0] = static_cast<Scalar>(2) / order;
    out[10][1] = static_cast<Scalar>(3) / order;
    out[10][2] = static_cast<Scalar>(0) / order;
    out[11][0] = static_cast<Scalar>(1) / order;
    out[11][1] = static_cast<Scalar>(4) / order;
    out[11][2] = static_cast<Scalar>(0) / order;
    out[12][0] = static_cast<Scalar>(0) / order;
    out[12][1] = static_cast<Scalar>(4) / order;
    out[12][2] = static_cast<Scalar>(0) / order;
    out[13][0] = static_cast<Scalar>(0) / order;
    out[13][1] = static_cast<Scalar>(3) / order;
    out[13][2] = static_cast<Scalar>(0) / order;
    out[14][0] = static_cast<Scalar>(0) / order;
    out[14][1] = static_cast<Scalar>(2) / order;
    out[14][2] = static_cast<Scalar>(0) / order;
    out[15][0] = static_cast<Scalar>(0) / order;
    out[15][1] = static_cast<Scalar>(1) / order;
    out[15][2] = static_cast<Scalar>(0) / order;
    out[16][0] = static_cast<Scalar>(0) / order;
    out[16][1] = static_cast<Scalar>(0) / order;
    out[16][2] = static_cast<Scalar>(1) / order;
    out[17][0] = static_cast<Scalar>(0) / order;
    out[17][1] = static_cast<Scalar>(0) / order;
    out[17][2] = static_cast<Scalar>(2) / order;
    out[18][0] = static_cast<Scalar>(0) / order;
    out[18][1] = static_cast<Scalar>(0) / order;
    out[18][2] = static_cast<Scalar>(3) / order;
    out[19][0] = static_cast<Scalar>(0) / order;
    out[19][1] = static_cast<Scalar>(0) / order;
    out[19][2] = static_cast<Scalar>(4) / order;
    out[20][0] = static_cast<Scalar>(4) / order;
    out[20][1] = static_cast<Scalar>(0) / order;
    out[20][2] = static_cast<Scalar>(1) / order;
    out[21][0] = static_cast<Scalar>(3) / order;
    out[21][1] = static_cast<Scalar>(0) / order;
    out[21][2] = static_cast<Scalar>(2) / order;
    out[22][0] = static_cast<Scalar>(2) / order;
    out[22][1] = static_cast<Scalar>(0) / order;
    out[22][2] = static_cast<Scalar>(3) / order;
    out[23][0] = static_cast<Scalar>(1) / order;
    out[23][1] = static_cast<Scalar>(0) / order;
    out[23][2] = static_cast<Scalar>(4) / order;
    out[24][0] = static_cast<Scalar>(0) / order;
    out[24][1] = static_cast<Scalar>(4) / order;
    out[24][2] = static_cast<Scalar>(1) / order;
    out[25][0] = static_cast<Scalar>(0) / order;
    out[25][1] = static_cast<Scalar>(3) / order;
    out[25][2] = static_cast<Scalar>(2) / order;
    out[26][0] = static_cast<Scalar>(0) / order;
    out[26][1] = static_cast<Scalar>(2) / order;
    out[26][2] = static_cast<Scalar>(3) / order;
    out[27][0] = static_cast<Scalar>(0) / order;
    out[27][1] = static_cast<Scalar>(1) / order;
    out[27][2] = static_cast<Scalar>(4) / order;
    out[28][0] = static_cast<Scalar>(1) / order;
    out[28][1] = static_cast<Scalar>(0) / order;
    out[28][2] = static_cast<Scalar>(1) / order;
    out[29][0] = static_cast<Scalar>(3) / order;
    out[29][1] = static_cast<Scalar>(0) / order;
    out[29][2] = static_cast<Scalar>(1) / order;
    out[30][0] = static_cast<Scalar>(1) / order;
    out[30][1] = static_cast<Scalar>(0) / order;
    out[30][2] = static_cast<Scalar>(3) / order;
    out[31][0] = static_cast<Scalar>(2) / order;
    out[31][1] = static_cast<Scalar>(0) / order;
    out[31][2] = static_cast<Scalar>(1) / order;
    out[32][0] = static_cast<Scalar>(2) / order;
    out[32][1] = static_cast<Scalar>(0) / order;
    out[32][2] = static_cast<Scalar>(2) / order;
    out[33][0] = static_cast<Scalar>(1) / order;
    out[33][1] = static_cast<Scalar>(0) / order;
    out[33][2] = static_cast<Scalar>(2) / order;
    out[34][0] = static_cast<Scalar>(1) / order;
    out[34][1] = static_cast<Scalar>(3) / order;
    out[34][2] = static_cast<Scalar>(1) / order;
    out[35][0] = static_cast<Scalar>(1) / order;
    out[35][1] = static_cast<Scalar>(1) / order;
    out[35][2] = static_cast<Scalar>(3) / order;
    out[36][0] = static_cast<Scalar>(3) / order;
    out[36][1] = static_cast<Scalar>(1) / order;
    out[36][2] = static_cast<Scalar>(1) / order;
    out[37][0] = static_cast<Scalar>(1) / order;
    out[37][1] = static_cast<Scalar>(2) / order;
    out[37][2] = static_cast<Scalar>(2) / order;
    out[38][0] = static_cast<Scalar>(2) / order;
    out[38][1] = static_cast<Scalar>(1) / order;
    out[38][2] = static_cast<Scalar>(2) / order;
    out[39][0] = static_cast<Scalar>(2) / order;
    out[39][1] = static_cast<Scalar>(2) / order;
    out[39][2] = static_cast<Scalar>(1) / order;
    out[40][0] = static_cast<Scalar>(0) / order;
    out[40][1] = static_cast<Scalar>(1) / order;
    out[40][2] = static_cast<Scalar>(1) / order;
    out[41][0] = static_cast<Scalar>(0) / order;
    out[41][1] = static_cast<Scalar>(1) / order;
    out[41][2] = static_cast<Scalar>(3) / order;
    out[42][0] = static_cast<Scalar>(0) / order;
    out[42][1] = static_cast<Scalar>(3) / order;
    out[42][2] = static_cast<Scalar>(1) / order;
    out[43][0] = static_cast<Scalar>(0) / order;
    out[43][1] = static_cast<Scalar>(1) / order;
    out[43][2] = static_cast<Scalar>(2) / order;
    out[44][0] = static_cast<Scalar>(0) / order;
    out[44][1] = static_cast<Scalar>(2) / order;
    out[44][2] = static_cast<Scalar>(2) / order;
    out[45][0] = static_cast<Scalar>(0) / order;
    out[45][1] = static_cast<Scalar>(2) / order;
    out[45][2] = static_cast<Scalar>(1) / order;
    out[46][0] = static_cast<Scalar>(1) / order;
    out[46][1] = static_cast<Scalar>(1) / order;
    out[46][2] = static_cast<Scalar>(0) / order;
    out[47][0] = static_cast<Scalar>(1) / order;
    out[47][1] = static_cast<Scalar>(3) / order;
    out[47][2] = static_cast<Scalar>(0) / order;
    out[48][0] = static_cast<Scalar>(3) / order;
    out[48][1] = static_cast<Scalar>(1) / order;
    out[48][2] = static_cast<Scalar>(0) / order;
    out[49][0] = static_cast<Scalar>(1) / order;
    out[49][1] = static_cast<Scalar>(2) / order;
    out[49][2] = static_cast<Scalar>(0) / order;
    out[50][0] = static_cast<Scalar>(2) / order;
    out[50][1] = static_cast<Scalar>(2) / order;
    out[50][2] = static_cast<Scalar>(0) / order;
    out[51][0] = static_cast<Scalar>(2) / order;
    out[51][1] = static_cast<Scalar>(1) / order;
    out[51][2] = static_cast<Scalar>(0) / order;
    out[52][0] = static_cast<Scalar>(1) / order;
    out[52][1] = static_cast<Scalar>(1) / order;
    out[52][2] = static_cast<Scalar>(1) / order;
    out[53][0] = static_cast<Scalar>(2) / order;
    out[53][1] = static_cast<Scalar>(1) / order;
    out[53][2] = static_cast<Scalar>(1) / order;
    out[54][0] = static_cast<Scalar>(1) / order;
    out[54][1] = static_cast<Scalar>(2) / order;
    out[54][2] = static_cast<Scalar>(1) / order;
    out[55][0] = static_cast<Scalar>(1) / order;
    out[55][1] = static_cast<Scalar>(1) / order;
    out[55][2] = static_cast<Scalar>(2) / order;
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
      out[0] = 5;
      out[1] = 0;
      out[2] = 0;
      break;
    case 2:
      out[0] = 0;
      out[1] = 5;
      out[2] = 0;
      break;
    case 3:
      out[0] = 0;
      out[1] = 0;
      out[2] = 5;
      break;
    case 4:
      out[0] = 1;
      out[1] = 0;
      out[2] = 0;
      break;
    case 5:
      out[0] = 2;
      out[1] = 0;
      out[2] = 0;
      break;
    case 6:
      out[0] = 3;
      out[1] = 0;
      out[2] = 0;
      break;
    case 7:
      out[0] = 4;
      out[1] = 0;
      out[2] = 0;
      break;
    case 8:
      out[0] = 4;
      out[1] = 1;
      out[2] = 0;
      break;
    case 9:
      out[0] = 3;
      out[1] = 2;
      out[2] = 0;
      break;
    case 10:
      out[0] = 2;
      out[1] = 3;
      out[2] = 0;
      break;
    case 11:
      out[0] = 1;
      out[1] = 4;
      out[2] = 0;
      break;
    case 12:
      out[0] = 0;
      out[1] = 4;
      out[2] = 0;
      break;
    case 13:
      out[0] = 0;
      out[1] = 3;
      out[2] = 0;
      break;
    case 14:
      out[0] = 0;
      out[1] = 2;
      out[2] = 0;
      break;
    case 15:
      out[0] = 0;
      out[1] = 1;
      out[2] = 0;
      break;
    case 16:
      out[0] = 0;
      out[1] = 0;
      out[2] = 1;
      break;
    case 17:
      out[0] = 0;
      out[1] = 0;
      out[2] = 2;
      break;
    case 18:
      out[0] = 0;
      out[1] = 0;
      out[2] = 3;
      break;
    case 19:
      out[0] = 0;
      out[1] = 0;
      out[2] = 4;
      break;
    case 20:
      out[0] = 4;
      out[1] = 0;
      out[2] = 1;
      break;
    case 21:
      out[0] = 3;
      out[1] = 0;
      out[2] = 2;
      break;
    case 22:
      out[0] = 2;
      out[1] = 0;
      out[2] = 3;
      break;
    case 23:
      out[0] = 1;
      out[1] = 0;
      out[2] = 4;
      break;
    case 24:
      out[0] = 0;
      out[1] = 4;
      out[2] = 1;
      break;
    case 25:
      out[0] = 0;
      out[1] = 3;
      out[2] = 2;
      break;
    case 26:
      out[0] = 0;
      out[1] = 2;
      out[2] = 3;
      break;
    case 27:
      out[0] = 0;
      out[1] = 1;
      out[2] = 4;
      break;
    case 28:
      out[0] = 1;
      out[1] = 0;
      out[2] = 1;
      break;
    case 29:
      out[0] = 3;
      out[1] = 0;
      out[2] = 1;
      break;
    case 30:
      out[0] = 1;
      out[1] = 0;
      out[2] = 3;
      break;
    case 31:
      out[0] = 2;
      out[1] = 0;
      out[2] = 1;
      break;
    case 32:
      out[0] = 2;
      out[1] = 0;
      out[2] = 2;
      break;
    case 33:
      out[0] = 1;
      out[1] = 0;
      out[2] = 2;
      break;
    case 34:
      out[0] = 1;
      out[1] = 3;
      out[2] = 1;
      break;
    case 35:
      out[0] = 1;
      out[1] = 1;
      out[2] = 3;
      break;
    case 36:
      out[0] = 3;
      out[1] = 1;
      out[2] = 1;
      break;
    case 37:
      out[0] = 1;
      out[1] = 2;
      out[2] = 2;
      break;
    case 38:
      out[0] = 2;
      out[1] = 1;
      out[2] = 2;
      break;
    case 39:
      out[0] = 2;
      out[1] = 2;
      out[2] = 1;
      break;
    case 40:
      out[0] = 0;
      out[1] = 1;
      out[2] = 1;
      break;
    case 41:
      out[0] = 0;
      out[1] = 1;
      out[2] = 3;
      break;
    case 42:
      out[0] = 0;
      out[1] = 3;
      out[2] = 1;
      break;
    case 43:
      out[0] = 0;
      out[1] = 1;
      out[2] = 2;
      break;
    case 44:
      out[0] = 0;
      out[1] = 2;
      out[2] = 2;
      break;
    case 45:
      out[0] = 0;
      out[1] = 2;
      out[2] = 1;
      break;
    case 46:
      out[0] = 1;
      out[1] = 1;
      out[2] = 0;
      break;
    case 47:
      out[0] = 1;
      out[1] = 3;
      out[2] = 0;
      break;
    case 48:
      out[0] = 3;
      out[1] = 1;
      out[2] = 0;
      break;
    case 49:
      out[0] = 1;
      out[1] = 2;
      out[2] = 0;
      break;
    case 50:
      out[0] = 2;
      out[1] = 2;
      out[2] = 0;
      break;
    case 51:
      out[0] = 2;
      out[1] = 1;
      out[2] = 0;
      break;
    case 52:
      out[0] = 1;
      out[1] = 1;
      out[2] = 1;
      break;
    case 53:
      out[0] = 2;
      out[1] = 1;
      out[2] = 1;
      break;
    case 54:
      out[0] = 1;
      out[1] = 2;
      out[2] = 1;
      break;
    case 55:
      out[0] = 1;
      out[1] = 1;
      out[2] = 2;
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
      idxs[2] = 4;
      idxs[3] = 5;
      idxs[4] = 6;
      idxs[5] = 7;
      break;
    case 1:
      idxs[0] = 1;
      idxs[1] = 2;
      idxs[2] = 8;
      idxs[3] = 9;
      idxs[4] = 10;
      idxs[5] = 11;
      break;
    case 2:
      idxs[0] = 2;
      idxs[1] = 0;
      idxs[2] = 12;
      idxs[3] = 13;
      idxs[4] = 14;
      idxs[5] = 15;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 16;
      idxs[3] = 17;
      idxs[4] = 18;
      idxs[5] = 19;
      break;
    case 4:
      idxs[0] = 1;
      idxs[1] = 3;
      idxs[2] = 20;
      idxs[3] = 21;
      idxs[4] = 22;
      idxs[5] = 23;
      break;
    case 5:
      idxs[0] = 2;
      idxs[1] = 3;
      idxs[2] = 24;
      idxs[3] = 25;
      idxs[4] = 26;
      idxs[5] = 27;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElTria>) {
    switch (subelement) {
    case 0:
      idxs[0] = 0;
      idxs[1] = 1;
      idxs[2] = 3;
      idxs[3] = 4;
      idxs[4] = 5;
      idxs[5] = 6;
      idxs[6] = 7;
      idxs[7] = 20;
      idxs[8] = 21;
      idxs[9] = 22;
      idxs[10] = 23;
      idxs[11] = 19;
      idxs[12] = 18;
      idxs[13] = 17;
      idxs[14] = 16;
      idxs[15] = 28;
      idxs[16] = 29;
      idxs[17] = 30;
      idxs[18] = 31;
      idxs[19] = 32;
      idxs[20] = 33;
      break;
    case 1:
      idxs[0] = 2;
      idxs[1] = 3;
      idxs[2] = 1;
      idxs[3] = 24;
      idxs[4] = 25;
      idxs[5] = 26;
      idxs[6] = 27;
      idxs[7] = 23;
      idxs[8] = 22;
      idxs[9] = 21;
      idxs[10] = 20;
      idxs[11] = 8;
      idxs[12] = 9;
      idxs[13] = 10;
      idxs[14] = 11;
      idxs[15] = 34;
      idxs[16] = 35;
      idxs[17] = 36;
      idxs[18] = 37;
      idxs[19] = 38;
      idxs[20] = 39;
      break;
    case 2:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 2;
      idxs[3] = 16;
      idxs[4] = 17;
      idxs[5] = 18;
      idxs[6] = 19;
      idxs[7] = 27;
      idxs[8] = 26;
      idxs[9] = 25;
      idxs[10] = 24;
      idxs[11] = 12;
      idxs[12] = 13;
      idxs[13] = 14;
      idxs[14] = 15;
      idxs[15] = 40;
      idxs[16] = 41;
      idxs[17] = 42;
      idxs[18] = 43;
      idxs[19] = 44;
      idxs[20] = 45;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 2;
      idxs[2] = 1;
      idxs[3] = 15;
      idxs[4] = 14;
      idxs[5] = 13;
      idxs[6] = 12;
      idxs[7] = 11;
      idxs[8] = 10;
      idxs[9] = 9;
      idxs[10] = 8;
      idxs[11] = 7;
      idxs[12] = 6;
      idxs[13] = 5;
      idxs[14] = 4;
      idxs[15] = 46;
      idxs[16] = 47;
      idxs[17] = 48;
      idxs[18] = 49;
      idxs[19] = 50;
      idxs[20] = 51;
      break;
    default:
      break;
    }
  }

  static void subelement_node_idxs(dim_t subelement, dim_t *idxs,
                                   meta::type_tag<mesh::RefElQuad>) {
    switch (subelement) {
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
