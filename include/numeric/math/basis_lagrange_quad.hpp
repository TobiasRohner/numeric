#ifndef NUMERIC_MATH_BASIS_LAGRANGE_QUAD_HPP_
#define NUMERIC_MATH_BASIS_LAGRANGE_QUAD_HPP_

#include <numeric/mesh/ref_el_cube.hpp>
#include <numeric/mesh/ref_el_point.hpp>
#include <numeric/mesh/ref_el_quad.hpp>
#include <numeric/mesh/ref_el_segment.hpp>
#include <numeric/mesh/ref_el_tetra.hpp>
#include <numeric/mesh/ref_el_tria.hpp>
#include <numeric/meta/meta.hpp>
#include <numeric/meta/type_tag.hpp>

namespace numeric::math {

template <> struct BasisLagrange<mesh::RefElQuad, 1> {
  using ref_el_t = mesh::RefElQuad;
  static constexpr dim_t order = 1;
  static constexpr dim_t num_basis_functions = 4;
  static constexpr dim_t num_interpolation_nodes = 4;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = x[1] - 1;
    const Scalar x1 = x[0] - 1;
    return coeffs[0] * x0 * x1 - coeffs[1] * x0 * x[0] +
           coeffs[2] * x[0] * x[1] - coeffs[3] * x1 * x[1];
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = x[1] - 1;
    out[0] = x0 * x1;
    out[1] = -x1 * x[0];
    out[2] = x[0] * x[1];
    out[3] = -x0 * x[1];
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = x[1] - 1;
    const Scalar x1 = x[0] - 1;
    out[0] =
        coeffs[0] * x0 - coeffs[1] * x0 + coeffs[2] * x[1] - coeffs[3] * x[1];
    out[1] =
        coeffs[0] * x1 - coeffs[1] * x[0] + coeffs[2] * x[0] - coeffs[3] * x1;
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[2]) {
    const Scalar x0 = x[1] - 1;
    const Scalar x1 = x[0] - 1;
    out[0][0] = x0;
    out[0][1] = x1;
    out[1][0] = -x0;
    out[1][1] = -x[0];
    out[2][0] = x[1];
    out[2][1] = x[0];
    out[3][0] = -x[1];
    out[3][1] = -x1;
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[2];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
  }

  template <typename Scalar>
  static constexpr void interpolation_nodes(Scalar (*out)[2]) {
    out[0][0] = static_cast<Scalar>(0) / order;
    out[0][1] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(1) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(1) / order;
    out[2][1] = static_cast<Scalar>(1) / order;
    out[3][0] = static_cast<Scalar>(0) / order;
    out[3][1] = static_cast<Scalar>(1) / order;
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
      break;
    case 1:
      out[0] = 1;
      out[1] = 0;
      break;
    case 2:
      out[0] = 1;
      out[1] = 1;
      break;
    case 3:
      out[0] = 0;
      out[1] = 1;
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
      idxs[0] = 3;
      idxs[1] = 2;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 3;
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

template <> struct BasisLagrange<mesh::RefElQuad, 2> {
  using ref_el_t = mesh::RefElQuad;
  static constexpr dim_t order = 2;
  static constexpr dim_t num_basis_functions = 9;
  static constexpr dim_t num_interpolation_nodes = 9;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = x[1] - 1;
    const Scalar x2 = x1 * x[0];
    const Scalar x3 = 2 * x[0] - 1;
    const Scalar x4 = x3 * x[1];
    const Scalar x5 = 2 * x[1] - 1;
    const Scalar x6 = x0 * x5;
    const Scalar x7 = (1.0 / 4.0) * x4;
    const Scalar x8 = (1.0 / 4.0) * x3;
    return 4 * coeffs[0] * x1 * x6 * x8 + 4 * coeffs[1] * x2 * x5 * x8 +
           4 * coeffs[2] * x5 * x7 * x[0] + 4 * coeffs[3] * x6 * x7 -
           4 * coeffs[4] * x2 * x6 - 4 * coeffs[5] * x2 * x4 -
           4 * coeffs[6] * x6 * x[0] * x[1] - 4 * coeffs[7] * x0 * x1 * x4 +
           16 * coeffs[8] * x0 * x2 * x[1];
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    const Scalar x0 = 2 * x[0] - 1;
    const Scalar x1 = 2 * x[1] - 1;
    const Scalar x2 = x0 * x1;
    const Scalar x3 = x[0] - 1;
    const Scalar x4 = x[1] - 1;
    const Scalar x5 = x3 * x4;
    const Scalar x6 = x2 * x[0];
    const Scalar x7 = x3 * x[1];
    const Scalar x8 = 4 * x[0];
    const Scalar x9 = x1 * x8;
    const Scalar x10 = x0 * x[1];
    out[0] = x2 * x5;
    out[1] = x4 * x6;
    out[2] = x6 * x[1];
    out[3] = x2 * x7;
    out[4] = -x5 * x9;
    out[5] = -x10 * x4 * x8;
    out[6] = -x7 * x9;
    out[7] = -4 * x10 * x5;
    out[8] = 16 * x5 * x[0] * x[1];
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = 4 * x[0];
    const Scalar x1 = x0 - 1;
    const Scalar x2 = x[1] - 1;
    const Scalar x3 = x2 * x[1];
    const Scalar x4 = x0 - 3;
    const Scalar x5 = 2 * x[0] - 1;
    const Scalar x6 = 4 * x[1];
    const Scalar x7 = 2 * x[1] - 1;
    const Scalar x8 = (1.0 / 4.0) * x7;
    const Scalar x9 = x8 * x[1];
    const Scalar x10 = x5 * x7;
    const Scalar x11 = x2 * x8;
    const Scalar x12 = x6 - 3;
    const Scalar x13 = x[0] - 1;
    const Scalar x14 = x13 * x[0];
    const Scalar x15 = x6 - 1;
    const Scalar x16 = (1.0 / 4.0) * x5;
    const Scalar x17 = x12 * x16;
    const Scalar x18 = x15 * x16;
    out[0] = 4 * coeffs[0] * x11 * x4 + 4 * coeffs[1] * x1 * x11 +
             4 * coeffs[2] * x1 * x9 + 4 * coeffs[3] * x4 * x9 -
             4 * coeffs[4] * x10 * x2 - 4 * coeffs[5] * x1 * x3 -
             4 * coeffs[6] * x10 * x[1] - 4 * coeffs[7] * x3 * x4 +
             4 * coeffs[8] * x2 * x5 * x6;
    out[1] = 4 * coeffs[0] * x13 * x17 + 4 * coeffs[1] * x17 * x[0] +
             4 * coeffs[2] * x18 * x[0] + 4 * coeffs[3] * x13 * x18 -
             4 * coeffs[4] * x12 * x14 - 4 * coeffs[5] * x10 * x[0] -
             4 * coeffs[6] * x14 * x15 - 4 * coeffs[7] * x10 * x13 +
             4 * coeffs[8] * x0 * x13 * x7;
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[2]) {
    const Scalar x0 = 4 * x[0];
    const Scalar x1 = x0 - 3;
    const Scalar x2 = x[1] - 1;
    const Scalar x3 = 2 * x[1] - 1;
    const Scalar x4 = x2 * x3;
    const Scalar x5 = x[0] - 1;
    const Scalar x6 = 2 * x[0] - 1;
    const Scalar x7 = 4 * x[1];
    const Scalar x8 = x7 - 3;
    const Scalar x9 = x6 * x8;
    const Scalar x10 = x0 - 1;
    const Scalar x11 = x3 * x[1];
    const Scalar x12 = x7 - 1;
    const Scalar x13 = x12 * x6;
    const Scalar x14 = 4 * x6;
    const Scalar x15 = x0 * x5;
    const Scalar x16 = x2 * x7;
    const Scalar x17 = x3 * x6;
    const Scalar x18 = x3 * x5;
    out[0][0] = x1 * x4;
    out[0][1] = x5 * x9;
    out[1][0] = x10 * x4;
    out[1][1] = x9 * x[0];
    out[2][0] = x10 * x11;
    out[2][1] = x13 * x[0];
    out[3][0] = x1 * x11;
    out[3][1] = x13 * x5;
    out[4][0] = -x14 * x4;
    out[4][1] = -x15 * x8;
    out[5][0] = -x10 * x16;
    out[5][1] = -x0 * x17;
    out[6][0] = -x17 * x7;
    out[6][1] = -x12 * x15;
    out[7][0] = -x1 * x16;
    out[7][1] = -x14 * x18;
    out[8][0] = 16 * x2 * x6 * x[1];
    out[8][1] = 16 * x18 * x[0];
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[2];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
  }

  template <typename Scalar>
  static constexpr void interpolation_nodes(Scalar (*out)[2]) {
    out[0][0] = static_cast<Scalar>(0) / order;
    out[0][1] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(2) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(2) / order;
    out[2][1] = static_cast<Scalar>(2) / order;
    out[3][0] = static_cast<Scalar>(0) / order;
    out[3][1] = static_cast<Scalar>(2) / order;
    out[4][0] = static_cast<Scalar>(1) / order;
    out[4][1] = static_cast<Scalar>(0) / order;
    out[5][0] = static_cast<Scalar>(2) / order;
    out[5][1] = static_cast<Scalar>(1) / order;
    out[6][0] = static_cast<Scalar>(1) / order;
    out[6][1] = static_cast<Scalar>(2) / order;
    out[7][0] = static_cast<Scalar>(0) / order;
    out[7][1] = static_cast<Scalar>(1) / order;
    out[8][0] = static_cast<Scalar>(1) / order;
    out[8][1] = static_cast<Scalar>(1) / order;
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
      break;
    case 1:
      out[0] = 2;
      out[1] = 0;
      break;
    case 2:
      out[0] = 2;
      out[1] = 2;
      break;
    case 3:
      out[0] = 0;
      out[1] = 2;
      break;
    case 4:
      out[0] = 1;
      out[1] = 0;
      break;
    case 5:
      out[0] = 2;
      out[1] = 1;
      break;
    case 6:
      out[0] = 1;
      out[1] = 2;
      break;
    case 7:
      out[0] = 0;
      out[1] = 1;
      break;
    case 8:
      out[0] = 1;
      out[1] = 1;
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
      idxs[0] = 3;
      idxs[1] = 2;
      idxs[2] = 6;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 7;
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

template <> struct BasisLagrange<mesh::RefElQuad, 3> {
  using ref_el_t = mesh::RefElQuad;
  static constexpr dim_t order = 3;
  static constexpr dim_t num_basis_functions = 16;
  static constexpr dim_t num_interpolation_nodes = 16;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = x[1] - 1;
    const Scalar x2 = x0 * x1 * x[0] * x[1];
    const Scalar x3 = 3 * x[0];
    const Scalar x4 = x3 - 2;
    const Scalar x5 = 3 * x[1];
    const Scalar x6 = x5 - 2;
    const Scalar x7 = x4 * x6;
    const Scalar x8 = x5 - 1;
    const Scalar x9 = x3 - 1;
    const Scalar x10 = x2 * x9;
    const Scalar x11 = x4 * x8;
    const Scalar x12 = x1 * x7 * x9;
    const Scalar x13 = x1 * x[0];
    const Scalar x14 = x9 * x[1];
    const Scalar x15 = (1.0 / 9.0) * x0;
    const Scalar x16 = x15 * x[1];
    const Scalar x17 = x7 * x8;
    const Scalar x18 = x8 * x[0];
    const Scalar x19 = x16 * x9;
    const Scalar x20 = (1.0 / 81.0) * x14 * x17;
    const Scalar x21 = x13 * x15;
    const Scalar x22 = (1.0 / 81.0) * x12;
    return (81.0 / 4.0) * coeffs[0] * x0 * x22 * x8 -
           81.0 / 4.0 * coeffs[10] * x12 * x16 +
           (81.0 / 4.0) * coeffs[11] * x1 * x11 * x19 +
           (81.0 / 4.0) * coeffs[12] * x2 * x7 -
           81.0 / 4.0 * coeffs[13] * x10 * x6 -
           81.0 / 4.0 * coeffs[14] * x11 * x2 +
           (81.0 / 4.0) * coeffs[15] * x10 * x8 -
           81.0 / 4.0 * coeffs[1] * x18 * x22 +
           (81.0 / 4.0) * coeffs[2] * x20 * x[0] -
           81.0 / 4.0 * coeffs[3] * x0 * x20 -
           81.0 / 4.0 * coeffs[4] * x17 * x21 +
           (81.0 / 4.0) * coeffs[5] * x21 * x6 * x8 * x9 +
           (9.0 / 4.0) * coeffs[6] * x12 * x[0] * x[1] -
           9.0 / 4.0 * coeffs[7] * x11 * x13 * x14 +
           (81.0 / 4.0) * coeffs[8] * x16 * x17 * x[0] -
           81.0 / 4.0 * coeffs[9] * x18 * x19 * x6;
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = 3 * x[1];
    const Scalar x2 = x1 - 1;
    const Scalar x3 = x0 * x2;
    const Scalar x4 = x[1] - 1;
    const Scalar x5 = 3 * x[0];
    const Scalar x6 = x5 - 1;
    const Scalar x7 = x5 - 2;
    const Scalar x8 = x1 - 2;
    const Scalar x9 = x4 * x6 * x7 * x8;
    const Scalar x10 = (1.0 / 4.0) * x9;
    const Scalar x11 = x2 * x[0];
    const Scalar x12 = x6 * x[1];
    const Scalar x13 = x11 * x12;
    const Scalar x14 = x7 * x8;
    const Scalar x15 = (1.0 / 4.0) * x14;
    const Scalar x16 = x12 * x3;
    const Scalar x17 = x14 * x4;
    const Scalar x18 = (9.0 / 4.0) * x[0];
    const Scalar x19 = x18 * x3;
    const Scalar x20 = x4 * x8;
    const Scalar x21 = x9 * x[1];
    const Scalar x22 = x4 * x7;
    const Scalar x23 = (9.0 / 4.0) * x22;
    const Scalar x24 = (81.0 / 4.0) * x[0];
    const Scalar x25 = x0 * x24;
    out[0] = x10 * x3;
    out[1] = -x10 * x11;
    out[2] = x13 * x15;
    out[3] = -x15 * x16;
    out[4] = -x17 * x19;
    out[5] = x19 * x20 * x6;
    out[6] = x18 * x21;
    out[7] = -x13 * x23;
    out[8] = x14 * x19 * x[1];
    out[9] = -x16 * x18 * x8;
    out[10] = -9.0 / 4.0 * x0 * x21;
    out[11] = x16 * x23;
    out[12] = x17 * x25 * x[1];
    out[13] = -x12 * x20 * x25;
    out[14] = -x22 * x24 * x3 * x[1];
    out[15] = x16 * x24 * x4;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = 3 * x[0];
    const Scalar x2 = x0 * x1;
    const Scalar x3 = x1 - 2;
    const Scalar x4 = x3 * x[0];
    const Scalar x5 = x0 * x3;
    const Scalar x6 = x2 + x4 + x5;
    const Scalar x7 = x[1] - 1;
    const Scalar x8 = 3 * x[1];
    const Scalar x9 = x8 - 2;
    const Scalar x10 = x9 * x[1];
    const Scalar x11 = x10 * x7;
    const Scalar x12 = x1 - 1;
    const Scalar x13 = x12 * x[0];
    const Scalar x14 = x0 * x12;
    const Scalar x15 = x13 + x14 + x2;
    const Scalar x16 = x8 - 1;
    const Scalar x17 = x16 * x[1];
    const Scalar x18 = x17 * x7;
    const Scalar x19 = (1.0 / 9.0) * x16;
    const Scalar x20 = x10 * x19;
    const Scalar x21 = x7 * x9;
    const Scalar x22 = x19 * x21;
    const Scalar x23 = x12 * x3;
    const Scalar x24 = x1 * x12 + x1 * x3 + x23;
    const Scalar x25 = (1.0 / 9.0) * x24;
    const Scalar x26 = (1.0 / 81.0) * x16;
    const Scalar x27 = x10 * x26;
    const Scalar x28 = 3 * x14 + x23 + 3 * x5;
    const Scalar x29 = (1.0 / 9.0) * x28;
    const Scalar x30 = x21 * x26;
    const Scalar x31 = x7 * x8;
    const Scalar x32 = x10 + x21 + x31;
    const Scalar x33 = x0 * x32;
    const Scalar x34 = x16 * x7;
    const Scalar x35 = x17 + x31 + x34;
    const Scalar x36 = x0 * x35;
    const Scalar x37 = (1.0 / 9.0) * x12;
    const Scalar x38 = x37 * x4;
    const Scalar x39 = x37 * x5;
    const Scalar x40 = (1.0 / 9.0) * x0;
    const Scalar x41 = x16 * x9;
    const Scalar x42 = x16 * x8 + x41 + x8 * x9;
    const Scalar x43 = x4 * x42;
    const Scalar x44 = x13 * x40;
    const Scalar x45 = (1.0 / 81.0) * x12;
    const Scalar x46 = x45 * x5;
    const Scalar x47 = 3 * x21 + 3 * x34 + x41;
    const Scalar x48 = x4 * x47;
    out[0] = (81.0 / 4.0) * coeffs[0] * x28 * x30 -
             81.0 / 4.0 * coeffs[10] * x11 * x29 +
             (81.0 / 4.0) * coeffs[11] * x18 * x29 +
             (81.0 / 4.0) * coeffs[12] * x11 * x6 -
             81.0 / 4.0 * coeffs[13] * x11 * x15 -
             81.0 / 4.0 * coeffs[14] * x18 * x6 +
             (81.0 / 4.0) * coeffs[15] * x15 * x18 -
             81.0 / 4.0 * coeffs[1] * x24 * x30 +
             (81.0 / 4.0) * coeffs[2] * x24 * x27 -
             81.0 / 4.0 * coeffs[3] * x27 * x28 -
             81.0 / 4.0 * coeffs[4] * x22 * x6 +
             (81.0 / 4.0) * coeffs[5] * x15 * x22 +
             (81.0 / 4.0) * coeffs[6] * x11 * x25 -
             81.0 / 4.0 * coeffs[7] * x18 * x25 +
             (81.0 / 4.0) * coeffs[8] * x20 * x6 -
             81.0 / 4.0 * coeffs[9] * x15 * x20;
    out[1] = (81.0 / 4.0) * coeffs[0] * x46 * x47 -
             81.0 / 4.0 * coeffs[10] * x32 * x39 +
             (81.0 / 4.0) * coeffs[11] * x35 * x39 +
             (81.0 / 4.0) * coeffs[12] * x33 * x4 -
             81.0 / 4.0 * coeffs[13] * x13 * x33 -
             81.0 / 4.0 * coeffs[14] * x36 * x4 +
             (81.0 / 4.0) * coeffs[15] * x13 * x36 -
             81.0 / 4.0 * coeffs[1] * x45 * x48 +
             (81.0 / 4.0) * coeffs[2] * x43 * x45 -
             81.0 / 4.0 * coeffs[3] * x42 * x46 -
             81.0 / 4.0 * coeffs[4] * x40 * x48 +
             (81.0 / 4.0) * coeffs[5] * x44 * x47 +
             (81.0 / 4.0) * coeffs[6] * x32 * x38 -
             81.0 / 4.0 * coeffs[7] * x35 * x38 +
             (81.0 / 4.0) * coeffs[8] * x40 * x43 -
             81.0 / 4.0 * coeffs[9] * x42 * x44;
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[2]) {
    const Scalar x0 = 3 * x[1];
    const Scalar x1 = x0 - 2;
    const Scalar x2 = x0 - 1;
    const Scalar x3 = x1 * x2;
    const Scalar x4 = (1.0 / 4.0) * x3;
    const Scalar x5 = x[1] - 1;
    const Scalar x6 = x[0] - 1;
    const Scalar x7 = 3 * x[0];
    const Scalar x8 = x7 - 2;
    const Scalar x9 = x6 * x8;
    const Scalar x10 = x7 - 1;
    const Scalar x11 = x10 * x6;
    const Scalar x12 = x10 * x8;
    const Scalar x13 = 3 * x11 + x12 + 3 * x9;
    const Scalar x14 = x13 * x5;
    const Scalar x15 = (1.0 / 4.0) * x12;
    const Scalar x16 = x1 * x5;
    const Scalar x17 = x2 * x5;
    const Scalar x18 = 3 * x16 + 3 * x17 + x3;
    const Scalar x19 = x18 * x6;
    const Scalar x20 = x10 * x7 + x12 + x7 * x8;
    const Scalar x21 = x20 * x4;
    const Scalar x22 = x15 * x[0];
    const Scalar x23 = x0 * x1 + x0 * x2 + x3;
    const Scalar x24 = x23 * x6;
    const Scalar x25 = x6 * x7;
    const Scalar x26 = x8 * x[0];
    const Scalar x27 = x25 + x26 + x9;
    const Scalar x28 = (9.0 / 4.0) * x5;
    const Scalar x29 = x28 * x3;
    const Scalar x30 = (9.0 / 4.0) * x19;
    const Scalar x31 = x10 * x[0];
    const Scalar x32 = x11 + x25 + x31;
    const Scalar x33 = x1 * x[1];
    const Scalar x34 = x20 * x28;
    const Scalar x35 = x0 * x5;
    const Scalar x36 = x16 + x33 + x35;
    const Scalar x37 = (9.0 / 4.0) * x12;
    const Scalar x38 = x37 * x[0];
    const Scalar x39 = x2 * x[1];
    const Scalar x40 = x17 + x35 + x39;
    const Scalar x41 = (9.0 / 4.0) * x3 * x[1];
    const Scalar x42 = (9.0 / 4.0) * x24;
    const Scalar x43 = (9.0 / 4.0) * x14;
    const Scalar x44 = x37 * x6;
    const Scalar x45 = (81.0 / 4.0) * x5;
    const Scalar x46 = x33 * x45;
    const Scalar x47 = (81.0 / 4.0) * x6;
    const Scalar x48 = x36 * x47;
    const Scalar x49 = x39 * x45;
    const Scalar x50 = x40 * x47;
    out[0][0] = x14 * x4;
    out[0][1] = x15 * x19;
    out[1][0] = -x21 * x5;
    out[1][1] = -x18 * x22;
    out[2][0] = x21 * x[1];
    out[2][1] = x22 * x23;
    out[3][0] = -x13 * x4 * x[1];
    out[3][1] = -x15 * x24;
    out[4][0] = -x27 * x29;
    out[4][1] = -x26 * x30;
    out[5][0] = x29 * x32;
    out[5][1] = x30 * x31;
    out[6][0] = x33 * x34;
    out[6][1] = x36 * x38;
    out[7][0] = -x34 * x39;
    out[7][1] = -x38 * x40;
    out[8][0] = x27 * x41;
    out[8][1] = x26 * x42;
    out[9][0] = -x32 * x41;
    out[9][1] = -x31 * x42;
    out[10][0] = -x33 * x43;
    out[10][1] = -x36 * x44;
    out[11][0] = x39 * x43;
    out[11][1] = x40 * x44;
    out[12][0] = x27 * x46;
    out[12][1] = x26 * x48;
    out[13][0] = -x32 * x46;
    out[13][1] = -x31 * x48;
    out[14][0] = -x27 * x49;
    out[14][1] = -x26 * x50;
    out[15][0] = x32 * x49;
    out[15][1] = x31 * x50;
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[2];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
  }

  template <typename Scalar>
  static constexpr void interpolation_nodes(Scalar (*out)[2]) {
    out[0][0] = static_cast<Scalar>(0) / order;
    out[0][1] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(3) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(3) / order;
    out[2][1] = static_cast<Scalar>(3) / order;
    out[3][0] = static_cast<Scalar>(0) / order;
    out[3][1] = static_cast<Scalar>(3) / order;
    out[4][0] = static_cast<Scalar>(1) / order;
    out[4][1] = static_cast<Scalar>(0) / order;
    out[5][0] = static_cast<Scalar>(2) / order;
    out[5][1] = static_cast<Scalar>(0) / order;
    out[6][0] = static_cast<Scalar>(3) / order;
    out[6][1] = static_cast<Scalar>(1) / order;
    out[7][0] = static_cast<Scalar>(3) / order;
    out[7][1] = static_cast<Scalar>(2) / order;
    out[8][0] = static_cast<Scalar>(1) / order;
    out[8][1] = static_cast<Scalar>(3) / order;
    out[9][0] = static_cast<Scalar>(2) / order;
    out[9][1] = static_cast<Scalar>(3) / order;
    out[10][0] = static_cast<Scalar>(0) / order;
    out[10][1] = static_cast<Scalar>(1) / order;
    out[11][0] = static_cast<Scalar>(0) / order;
    out[11][1] = static_cast<Scalar>(2) / order;
    out[12][0] = static_cast<Scalar>(1) / order;
    out[12][1] = static_cast<Scalar>(1) / order;
    out[13][0] = static_cast<Scalar>(2) / order;
    out[13][1] = static_cast<Scalar>(1) / order;
    out[14][0] = static_cast<Scalar>(1) / order;
    out[14][1] = static_cast<Scalar>(2) / order;
    out[15][0] = static_cast<Scalar>(2) / order;
    out[15][1] = static_cast<Scalar>(2) / order;
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
      break;
    case 1:
      out[0] = 3;
      out[1] = 0;
      break;
    case 2:
      out[0] = 3;
      out[1] = 3;
      break;
    case 3:
      out[0] = 0;
      out[1] = 3;
      break;
    case 4:
      out[0] = 1;
      out[1] = 0;
      break;
    case 5:
      out[0] = 2;
      out[1] = 0;
      break;
    case 6:
      out[0] = 3;
      out[1] = 1;
      break;
    case 7:
      out[0] = 3;
      out[1] = 2;
      break;
    case 8:
      out[0] = 1;
      out[1] = 3;
      break;
    case 9:
      out[0] = 2;
      out[1] = 3;
      break;
    case 10:
      out[0] = 0;
      out[1] = 1;
      break;
    case 11:
      out[0] = 0;
      out[1] = 2;
      break;
    case 12:
      out[0] = 1;
      out[1] = 1;
      break;
    case 13:
      out[0] = 2;
      out[1] = 1;
      break;
    case 14:
      out[0] = 1;
      out[1] = 2;
      break;
    case 15:
      out[0] = 2;
      out[1] = 2;
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
      idxs[0] = 3;
      idxs[1] = 2;
      idxs[2] = 8;
      idxs[3] = 9;
      break;
    case 3:
      idxs[0] = 0;
      idxs[1] = 3;
      idxs[2] = 10;
      idxs[3] = 11;
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

template <> struct BasisLagrange<mesh::RefElQuad, 4> {
  using ref_el_t = mesh::RefElQuad;
  static constexpr dim_t order = 4;
  static constexpr dim_t num_basis_functions = 25;
  static constexpr dim_t num_interpolation_nodes = 25;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = x[1] - 1;
    const Scalar x2 = x0 * x1;
    const Scalar x3 = 2 * x[1] - 1;
    const Scalar x4 = 4 * x[0];
    const Scalar x5 = x4 - 3;
    const Scalar x6 = 4 * x[1];
    const Scalar x7 = x6 - 3;
    const Scalar x8 = x3 * x5 * x7 * x[0];
    const Scalar x9 = 2 * x[0] - 1;
    const Scalar x10 = 256 * x9;
    const Scalar x11 = x4 - 1;
    const Scalar x12 = x11 * x2 * x[1];
    const Scalar x13 = 192 * x12;
    const Scalar x14 = x7 * x[0];
    const Scalar x15 = x12 * x3;
    const Scalar x16 = x6 - 1;
    const Scalar x17 = x16 * x9;
    const Scalar x18 = x17 * x2;
    const Scalar x19 = x5 * x[0];
    const Scalar x20 = x19 * x7;
    const Scalar x21 = x20 * x[1];
    const Scalar x22 = x14 * x17;
    const Scalar x23 = x19 * x3;
    const Scalar x24 = x0 * x[1];
    const Scalar x25 = 16 * x24;
    const Scalar x26 = x11 * x8;
    const Scalar x27 = 12 * x16 * x26;
    const Scalar x28 = x11 * x3;
    const Scalar x29 = 16 * x9;
    const Scalar x30 = x1 * x[1];
    const Scalar x31 = 12 * x17;
    const Scalar x32 = 16 * x17;
    const Scalar x33 = x11 * x16 * x3 * x5 * x7 * x9;
    const Scalar x34 = x33 * x[0];
    const Scalar x35 = x5 * x7;
    const Scalar x36 = 16 * x18;
    return (1.0 / 9.0) * coeffs[0] * x2 * x33 -
           1.0 / 9.0 * coeffs[10] * x17 * x25 * x8 +
           (1.0 / 9.0) * coeffs[11] * x24 * x27 -
           1.0 / 9.0 * coeffs[12] * x22 * x25 * x28 -
           1.0 / 9.0 * coeffs[13] * x15 * x29 * x35 +
           (1.0 / 9.0) * coeffs[14] * x12 * x31 * x35 -
           1.0 / 9.0 * coeffs[15] * x15 * x32 * x5 +
           (1.0 / 9.0) * coeffs[16] * x10 * x2 * x8 * x[1] -
           1.0 / 9.0 * coeffs[17] * x13 * x8 +
           (1.0 / 9.0) * coeffs[18] * x10 * x14 * x15 -
           64.0 / 3.0 * coeffs[19] * x18 * x21 +
           (1.0 / 9.0) * coeffs[1] * x1 * x34 +
           16 * coeffs[20] * x12 * x16 * x20 -
           1.0 / 9.0 * coeffs[21] * x13 * x22 +
           (256.0 / 9.0) * coeffs[22] * x18 * x23 * x[1] -
           1.0 / 9.0 * coeffs[23] * x13 * x16 * x23 +
           (256.0 / 9.0) * coeffs[24] * x15 * x17 * x[0] +
           (1.0 / 9.0) * coeffs[2] * x34 * x[1] +
           (1.0 / 9.0) * coeffs[3] * x24 * x33 -
           1.0 / 9.0 * coeffs[4] * x36 * x8 +
           (1.0 / 9.0) * coeffs[5] * x2 * x27 -
           1.0 / 9.0 * coeffs[6] * x14 * x28 * x36 -
           1.0 / 9.0 * coeffs[7] * x26 * x29 * x30 +
           (1.0 / 9.0) * coeffs[8] * x1 * x11 * x21 * x31 -
           1.0 / 9.0 * coeffs[9] * x11 * x23 * x30 * x32;
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = 4 * x[1];
    const Scalar x2 = x1 - 1;
    const Scalar x3 = x0 * x2;
    const Scalar x4 = x[1] - 1;
    const Scalar x5 = 2 * x[0] - 1;
    const Scalar x6 = 2 * x[1] - 1;
    const Scalar x7 = 4 * x[0];
    const Scalar x8 = x7 - 1;
    const Scalar x9 = x7 - 3;
    const Scalar x10 = x1 - 3;
    const Scalar x11 = x10 * x4 * x5 * x6 * x8 * x9;
    const Scalar x12 = (1.0 / 9.0) * x11;
    const Scalar x13 = x2 * x[0];
    const Scalar x14 = x8 * x[1];
    const Scalar x15 = x13 * x14;
    const Scalar x16 = x10 * x5 * x6 * x9;
    const Scalar x17 = (1.0 / 9.0) * x16;
    const Scalar x18 = x14 * x3;
    const Scalar x19 = (16.0 / 9.0) * x[0];
    const Scalar x20 = x3 * x4;
    const Scalar x21 = x19 * x20;
    const Scalar x22 = x6 * x9;
    const Scalar x23 = x10 * x[0];
    const Scalar x24 = x22 * x23;
    const Scalar x25 = (4.0 / 3.0) * x24;
    const Scalar x26 = x10 * x5;
    const Scalar x27 = x26 * x6;
    const Scalar x28 = x19 * x[1];
    const Scalar x29 = x15 * x4;
    const Scalar x30 = (4.0 / 3.0) * x26 * x9;
    const Scalar x31 = x22 * x5;
    const Scalar x32 = (16.0 / 9.0) * x31;
    const Scalar x33 = x0 * x[1];
    const Scalar x34 = x18 * x4;
    const Scalar x35 = (256.0 / 9.0) * x4;
    const Scalar x36 = x0 * x14;
    const Scalar x37 = x23 * x5;
    const Scalar x38 = (64.0 / 3.0) * x37;
    const Scalar x39 = x20 * x[1];
    const Scalar x40 = (256.0 / 9.0) * x[0];
    out[0] = x12 * x3;
    out[1] = x12 * x13;
    out[2] = x15 * x17;
    out[3] = x17 * x18;
    out[4] = -x16 * x21;
    out[5] = x20 * x25 * x8;
    out[6] = -x21 * x27 * x8;
    out[7] = -x11 * x28;
    out[8] = x29 * x30;
    out[9] = -x29 * x32;
    out[10] = -x16 * x28 * x3;
    out[11] = x18 * x25;
    out[12] = -x18 * x19 * x27;
    out[13] = -16.0 / 9.0 * x11 * x33;
    out[14] = x30 * x34;
    out[15] = -x32 * x34;
    out[16] = x16 * x33 * x35 * x[0];
    out[17] = -64.0 / 3.0 * x24 * x36 * x4;
    out[18] = x35 * x36 * x37 * x6;
    out[19] = -x38 * x39 * x9;
    out[20] = 16 * x23 * x34 * x9;
    out[21] = -x34 * x38;
    out[22] = x31 * x39 * x40;
    out[23] = -64.0 / 3.0 * x22 * x34 * x[0];
    out[24] = x34 * x40 * x5 * x6;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = 2 * x[0];
    const Scalar x1 = x0 - 1;
    const Scalar x2 = x[0] - 1;
    const Scalar x3 = 4 * x[0];
    const Scalar x4 = x2 * x3;
    const Scalar x5 = x1 * x4;
    const Scalar x6 = x3 - 3;
    const Scalar x7 = x0 * x2;
    const Scalar x8 = x1 * x6;
    const Scalar x9 = x8 * x[0];
    const Scalar x10 = x2 * x8;
    const Scalar x11 = x10 + x5 + x6 * x7 + x9;
    const Scalar x12 = 2 * x[1];
    const Scalar x13 = x12 - 1;
    const Scalar x14 = 4 * x[1];
    const Scalar x15 = x14 - 3;
    const Scalar x16 = x13 * x15;
    const Scalar x17 = x16 * x[1];
    const Scalar x18 = x[1] - 1;
    const Scalar x19 = 256 * x18;
    const Scalar x20 = x17 * x19;
    const Scalar x21 = x3 - 1;
    const Scalar x22 = x21 * x6;
    const Scalar x23 = x22 * x[0];
    const Scalar x24 = x2 * x22;
    const Scalar x25 = x21 * x4 + x23 + x24 + x4 * x6;
    const Scalar x26 = 192 * x18;
    const Scalar x27 = x25 * x26;
    const Scalar x28 = x1 * x21;
    const Scalar x29 = x28 * x[0];
    const Scalar x30 = x2 * x28;
    const Scalar x31 = x21 * x7 + x29 + x30 + x5;
    const Scalar x32 = x14 - 1;
    const Scalar x33 = x15 * x32;
    const Scalar x34 = x33 * x[1];
    const Scalar x35 = x26 * x34;
    const Scalar x36 = x18 * x34;
    const Scalar x37 = 144 * coeffs[20];
    const Scalar x38 = x13 * x32;
    const Scalar x39 = x38 * x[1];
    const Scalar x40 = x19 * x39;
    const Scalar x41 = x16 * x32;
    const Scalar x42 = x41 * x[1];
    const Scalar x43 = 16 * x42;
    const Scalar x44 = 12 * x25;
    const Scalar x45 = x18 * x41;
    const Scalar x46 = 16 * x45;
    const Scalar x47 = x1 * x22;
    const Scalar x48 = x0 * x22 + x28 * x3 + x3 * x8 + x47;
    const Scalar x49 = 16 * x18;
    const Scalar x50 = x48 * x49;
    const Scalar x51 = 12 * coeffs[8];
    const Scalar x52 = 4 * x10 + 2 * x24 + 4 * x30 + x47;
    const Scalar x53 = x49 * x52;
    const Scalar x54 = 12 * coeffs[14];
    const Scalar x55 = x14 * x18;
    const Scalar x56 = x13 * x55;
    const Scalar x57 = x15 * x18;
    const Scalar x58 = x16 * x18;
    const Scalar x59 = x12 * x57 + x17 + x56 + x58;
    const Scalar x60 = 256 * x2;
    const Scalar x61 = x59 * x60;
    const Scalar x62 = 192 * x2;
    const Scalar x63 = x23 * x62;
    const Scalar x64 = x18 * x33;
    const Scalar x65 = x14 * x57 + x32 * x55 + x34 + x64;
    const Scalar x66 = x62 * x65;
    const Scalar x67 = x2 * x23;
    const Scalar x68 = x18 * x38;
    const Scalar x69 = x12 * x18 * x32 + x39 + x56 + x68;
    const Scalar x70 = x60 * x69;
    const Scalar x71 = x47 * x[0];
    const Scalar x72 = 16 * x71;
    const Scalar x73 = 16 * x2;
    const Scalar x74 = x47 * x73;
    const Scalar x75 = x12 * x33 + x14 * x16 + x14 * x38 + x41;
    const Scalar x76 = x73 * x75;
    const Scalar x77 = 12 * x67;
    const Scalar x78 = x47 * x75;
    const Scalar x79 = x41 + 4 * x58 + 2 * x64 + 4 * x68;
    const Scalar x80 = x73 * x79;
    const Scalar x81 = x47 * x79;
    out[0] = (1.0 / 9.0) * coeffs[0] * x45 * x52 -
             1.0 / 9.0 * coeffs[10] * x11 * x43 +
             (1.0 / 9.0) * coeffs[11] * x42 * x44 -
             1.0 / 9.0 * coeffs[12] * x31 * x43 -
             1.0 / 9.0 * coeffs[13] * x17 * x53 -
             1.0 / 9.0 * coeffs[15] * x39 * x53 +
             (1.0 / 9.0) * coeffs[16] * x11 * x20 -
             1.0 / 9.0 * coeffs[17] * x17 * x27 +
             (1.0 / 9.0) * coeffs[18] * x20 * x31 -
             1.0 / 9.0 * coeffs[19] * x11 * x35 +
             (1.0 / 9.0) * coeffs[1] * x45 * x48 -
             1.0 / 9.0 * coeffs[21] * x31 * x35 +
             (1.0 / 9.0) * coeffs[22] * x11 * x40 -
             1.0 / 9.0 * coeffs[23] * x27 * x39 +
             (1.0 / 9.0) * coeffs[24] * x31 * x40 +
             (1.0 / 9.0) * coeffs[2] * x42 * x48 +
             (1.0 / 9.0) * coeffs[3] * x42 * x52 -
             1.0 / 9.0 * coeffs[4] * x11 * x46 +
             (1.0 / 9.0) * coeffs[5] * x44 * x45 -
             1.0 / 9.0 * coeffs[6] * x31 * x46 -
             1.0 / 9.0 * coeffs[7] * x17 * x50 -
             1.0 / 9.0 * coeffs[9] * x39 * x50 + (1.0 / 9.0) * x25 * x36 * x37 +
             (1.0 / 9.0) * x36 * x48 * x51 + (1.0 / 9.0) * x36 * x52 * x54;
    out[1] =
        (1.0 / 9.0) * coeffs[0] * x2 * x81 - 1.0 / 9.0 * coeffs[10] * x76 * x9 +
        (1.0 / 9.0) * coeffs[11] * x75 * x77 -
        1.0 / 9.0 * coeffs[12] * x29 * x76 -
        1.0 / 9.0 * coeffs[13] * x59 * x74 -
        1.0 / 9.0 * coeffs[15] * x69 * x74 +
        (1.0 / 9.0) * coeffs[16] * x61 * x9 -
        1.0 / 9.0 * coeffs[17] * x59 * x63 +
        (1.0 / 9.0) * coeffs[18] * x29 * x61 -
        1.0 / 9.0 * coeffs[19] * x66 * x9 +
        (1.0 / 9.0) * coeffs[1] * x81 * x[0] -
        1.0 / 9.0 * coeffs[21] * x29 * x66 +
        (1.0 / 9.0) * coeffs[22] * x70 * x9 -
        1.0 / 9.0 * coeffs[23] * x63 * x69 +
        (1.0 / 9.0) * coeffs[24] * x29 * x70 +
        (1.0 / 9.0) * coeffs[2] * x78 * x[0] +
        (1.0 / 9.0) * coeffs[3] * x2 * x78 - 1.0 / 9.0 * coeffs[4] * x80 * x9 +
        (1.0 / 9.0) * coeffs[5] * x77 * x79 -
        1.0 / 9.0 * coeffs[6] * x29 * x80 - 1.0 / 9.0 * coeffs[7] * x59 * x72 -
        1.0 / 9.0 * coeffs[9] * x69 * x72 + (1.0 / 9.0) * x2 * x47 * x54 * x65 +
        (1.0 / 9.0) * x37 * x65 * x67 + (1.0 / 9.0) * x51 * x65 * x71;
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[2]) {
    const Scalar x0 = 4 * x[1];
    const Scalar x1 = x0 - 1;
    const Scalar x2 = 2 * x[1];
    const Scalar x3 = x2 - 1;
    const Scalar x4 = x0 - 3;
    const Scalar x5 = x3 * x4;
    const Scalar x6 = x1 * x5;
    const Scalar x7 = (1.0 / 9.0) * x6;
    const Scalar x8 = x[1] - 1;
    const Scalar x9 = x[0] - 1;
    const Scalar x10 = 2 * x[0];
    const Scalar x11 = x10 - 1;
    const Scalar x12 = 4 * x[0];
    const Scalar x13 = x12 - 3;
    const Scalar x14 = x11 * x13;
    const Scalar x15 = x14 * x9;
    const Scalar x16 = x12 - 1;
    const Scalar x17 = x11 * x16;
    const Scalar x18 = x17 * x9;
    const Scalar x19 = x13 * x16;
    const Scalar x20 = x19 * x9;
    const Scalar x21 = x14 * x16;
    const Scalar x22 = 4 * x15 + 4 * x18 + 2 * x20 + x21;
    const Scalar x23 = x22 * x8;
    const Scalar x24 = (1.0 / 9.0) * x21;
    const Scalar x25 = x5 * x8;
    const Scalar x26 = x1 * x3;
    const Scalar x27 = x26 * x8;
    const Scalar x28 = x1 * x4;
    const Scalar x29 = x28 * x8;
    const Scalar x30 = 4 * x25 + 4 * x27 + 2 * x29 + x6;
    const Scalar x31 = x30 * x9;
    const Scalar x32 = x10 * x19 + x12 * x14 + x12 * x17 + x21;
    const Scalar x33 = x32 * x7;
    const Scalar x34 = x24 * x[0];
    const Scalar x35 = x0 * x26 + x0 * x5 + x2 * x28 + x6;
    const Scalar x36 = x35 * x9;
    const Scalar x37 = x12 * x9;
    const Scalar x38 = x11 * x37;
    const Scalar x39 = x13 * x9;
    const Scalar x40 = x14 * x[0];
    const Scalar x41 = x10 * x39 + x15 + x38 + x40;
    const Scalar x42 = x6 * x8;
    const Scalar x43 = (16.0 / 9.0) * x42;
    const Scalar x44 = (16.0 / 9.0) * x31;
    const Scalar x45 = x19 * x[0];
    const Scalar x46 = x12 * x39 + x16 * x37 + x20 + x45;
    const Scalar x47 = (4.0 / 3.0) * x46;
    const Scalar x48 = (4.0 / 3.0) * x45;
    const Scalar x49 = x17 * x[0];
    const Scalar x50 = x10 * x16 * x9 + x18 + x38 + x49;
    const Scalar x51 = x32 * x8;
    const Scalar x52 = x5 * x[1];
    const Scalar x53 = (16.0 / 9.0) * x52;
    const Scalar x54 = x0 * x8;
    const Scalar x55 = x3 * x54;
    const Scalar x56 = x4 * x8;
    const Scalar x57 = x2 * x56 + x25 + x52 + x55;
    const Scalar x58 = x21 * x[0];
    const Scalar x59 = (16.0 / 9.0) * x58;
    const Scalar x60 = x28 * x[1];
    const Scalar x61 = (4.0 / 3.0) * x60;
    const Scalar x62 = x0 * x56 + x1 * x54 + x29 + x60;
    const Scalar x63 = (4.0 / 3.0) * x62;
    const Scalar x64 = x26 * x[1];
    const Scalar x65 = (16.0 / 9.0) * x64;
    const Scalar x66 = x1 * x2 * x8 + x27 + x55 + x64;
    const Scalar x67 = x6 * x[1];
    const Scalar x68 = (16.0 / 9.0) * x67;
    const Scalar x69 = (16.0 / 9.0) * x36;
    const Scalar x70 = x21 * x9;
    const Scalar x71 = (16.0 / 9.0) * x70;
    const Scalar x72 = x52 * x8;
    const Scalar x73 = (256.0 / 9.0) * x72;
    const Scalar x74 = x57 * x9;
    const Scalar x75 = (256.0 / 9.0) * x74;
    const Scalar x76 = (64.0 / 3.0) * x46;
    const Scalar x77 = (64.0 / 3.0) * x45;
    const Scalar x78 = x60 * x8;
    const Scalar x79 = (64.0 / 3.0) * x78;
    const Scalar x80 = x62 * x9;
    const Scalar x81 = (64.0 / 3.0) * x80;
    const Scalar x82 = x64 * x8;
    const Scalar x83 = (256.0 / 9.0) * x82;
    const Scalar x84 = x66 * x9;
    const Scalar x85 = (256.0 / 9.0) * x84;
    out[0][0] = x23 * x7;
    out[0][1] = x24 * x31;
    out[1][0] = x33 * x8;
    out[1][1] = x30 * x34;
    out[2][0] = x33 * x[1];
    out[2][1] = x34 * x35;
    out[3][0] = x22 * x7 * x[1];
    out[3][1] = x24 * x36;
    out[4][0] = -x41 * x43;
    out[4][1] = -x40 * x44;
    out[5][0] = x42 * x47;
    out[5][1] = x31 * x48;
    out[6][0] = -x43 * x50;
    out[6][1] = -x44 * x49;
    out[7][0] = -x51 * x53;
    out[7][1] = -x57 * x59;
    out[8][0] = x51 * x61;
    out[8][1] = x58 * x63;
    out[9][0] = -x51 * x65;
    out[9][1] = -x59 * x66;
    out[10][0] = -x41 * x68;
    out[10][1] = -x40 * x69;
    out[11][0] = x47 * x67;
    out[11][1] = x36 * x48;
    out[12][0] = -x50 * x68;
    out[12][1] = -x49 * x69;
    out[13][0] = -x23 * x53;
    out[13][1] = -x57 * x71;
    out[14][0] = x23 * x61;
    out[14][1] = x63 * x70;
    out[15][0] = -x23 * x65;
    out[15][1] = -x66 * x71;
    out[16][0] = x41 * x73;
    out[16][1] = x40 * x75;
    out[17][0] = -x72 * x76;
    out[17][1] = -x74 * x77;
    out[18][0] = x50 * x73;
    out[18][1] = x49 * x75;
    out[19][0] = -x41 * x79;
    out[19][1] = -x40 * x81;
    out[20][0] = 16 * x46 * x78;
    out[20][1] = 16 * x45 * x80;
    out[21][0] = -x50 * x79;
    out[21][1] = -x49 * x81;
    out[22][0] = x41 * x83;
    out[22][1] = x40 * x85;
    out[23][0] = -x76 * x82;
    out[23][1] = -x77 * x84;
    out[24][0] = x50 * x83;
    out[24][1] = x49 * x85;
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[2];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
  }

  template <typename Scalar>
  static constexpr void interpolation_nodes(Scalar (*out)[2]) {
    out[0][0] = static_cast<Scalar>(0) / order;
    out[0][1] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(4) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(4) / order;
    out[2][1] = static_cast<Scalar>(4) / order;
    out[3][0] = static_cast<Scalar>(0) / order;
    out[3][1] = static_cast<Scalar>(4) / order;
    out[4][0] = static_cast<Scalar>(1) / order;
    out[4][1] = static_cast<Scalar>(0) / order;
    out[5][0] = static_cast<Scalar>(2) / order;
    out[5][1] = static_cast<Scalar>(0) / order;
    out[6][0] = static_cast<Scalar>(3) / order;
    out[6][1] = static_cast<Scalar>(0) / order;
    out[7][0] = static_cast<Scalar>(4) / order;
    out[7][1] = static_cast<Scalar>(1) / order;
    out[8][0] = static_cast<Scalar>(4) / order;
    out[8][1] = static_cast<Scalar>(2) / order;
    out[9][0] = static_cast<Scalar>(4) / order;
    out[9][1] = static_cast<Scalar>(3) / order;
    out[10][0] = static_cast<Scalar>(1) / order;
    out[10][1] = static_cast<Scalar>(4) / order;
    out[11][0] = static_cast<Scalar>(2) / order;
    out[11][1] = static_cast<Scalar>(4) / order;
    out[12][0] = static_cast<Scalar>(3) / order;
    out[12][1] = static_cast<Scalar>(4) / order;
    out[13][0] = static_cast<Scalar>(0) / order;
    out[13][1] = static_cast<Scalar>(1) / order;
    out[14][0] = static_cast<Scalar>(0) / order;
    out[14][1] = static_cast<Scalar>(2) / order;
    out[15][0] = static_cast<Scalar>(0) / order;
    out[15][1] = static_cast<Scalar>(3) / order;
    out[16][0] = static_cast<Scalar>(1) / order;
    out[16][1] = static_cast<Scalar>(1) / order;
    out[17][0] = static_cast<Scalar>(2) / order;
    out[17][1] = static_cast<Scalar>(1) / order;
    out[18][0] = static_cast<Scalar>(3) / order;
    out[18][1] = static_cast<Scalar>(1) / order;
    out[19][0] = static_cast<Scalar>(1) / order;
    out[19][1] = static_cast<Scalar>(2) / order;
    out[20][0] = static_cast<Scalar>(2) / order;
    out[20][1] = static_cast<Scalar>(2) / order;
    out[21][0] = static_cast<Scalar>(3) / order;
    out[21][1] = static_cast<Scalar>(2) / order;
    out[22][0] = static_cast<Scalar>(1) / order;
    out[22][1] = static_cast<Scalar>(3) / order;
    out[23][0] = static_cast<Scalar>(2) / order;
    out[23][1] = static_cast<Scalar>(3) / order;
    out[24][0] = static_cast<Scalar>(3) / order;
    out[24][1] = static_cast<Scalar>(3) / order;
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
      break;
    case 1:
      out[0] = 4;
      out[1] = 0;
      break;
    case 2:
      out[0] = 4;
      out[1] = 4;
      break;
    case 3:
      out[0] = 0;
      out[1] = 4;
      break;
    case 4:
      out[0] = 1;
      out[1] = 0;
      break;
    case 5:
      out[0] = 2;
      out[1] = 0;
      break;
    case 6:
      out[0] = 3;
      out[1] = 0;
      break;
    case 7:
      out[0] = 4;
      out[1] = 1;
      break;
    case 8:
      out[0] = 4;
      out[1] = 2;
      break;
    case 9:
      out[0] = 4;
      out[1] = 3;
      break;
    case 10:
      out[0] = 1;
      out[1] = 4;
      break;
    case 11:
      out[0] = 2;
      out[1] = 4;
      break;
    case 12:
      out[0] = 3;
      out[1] = 4;
      break;
    case 13:
      out[0] = 0;
      out[1] = 1;
      break;
    case 14:
      out[0] = 0;
      out[1] = 2;
      break;
    case 15:
      out[0] = 0;
      out[1] = 3;
      break;
    case 16:
      out[0] = 1;
      out[1] = 1;
      break;
    case 17:
      out[0] = 2;
      out[1] = 1;
      break;
    case 18:
      out[0] = 3;
      out[1] = 1;
      break;
    case 19:
      out[0] = 1;
      out[1] = 2;
      break;
    case 20:
      out[0] = 2;
      out[1] = 2;
      break;
    case 21:
      out[0] = 3;
      out[1] = 2;
      break;
    case 22:
      out[0] = 1;
      out[1] = 3;
      break;
    case 23:
      out[0] = 2;
      out[1] = 3;
      break;
    case 24:
      out[0] = 3;
      out[1] = 3;
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
      idxs[0] = 3;
      idxs[1] = 2;
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

template <> struct BasisLagrange<mesh::RefElQuad, 5> {
  using ref_el_t = mesh::RefElQuad;
  static constexpr dim_t order = 5;
  static constexpr dim_t num_basis_functions = 36;
  static constexpr dim_t num_interpolation_nodes = 36;

  template <typename Scalar>
  static constexpr Scalar eval(const Scalar *x, const Scalar *coeffs) {
    const Scalar x0 = 5 * x[0];
    const Scalar x1 = x0 - 4;
    const Scalar x2 = 5 * x[1];
    const Scalar x3 = x2 - 4;
    const Scalar x4 = x1 * x3;
    const Scalar x5 = x0 - 2;
    const Scalar x6 = x[0] - 1;
    const Scalar x7 = x[1] - 1;
    const Scalar x8 = x6 * x7;
    const Scalar x9 = x8 * x[1];
    const Scalar x10 = x5 * x9;
    const Scalar x11 = x2 - 3;
    const Scalar x12 = 625 * x11;
    const Scalar x13 = x10 * x12;
    const Scalar x14 = x0 - 3;
    const Scalar x15 = x2 - 2;
    const Scalar x16 = x14 * x15;
    const Scalar x17 = x16 * x[0];
    const Scalar x18 = x0 - 1;
    const Scalar x19 = x16 * x18;
    const Scalar x20 = x19 * x9;
    const Scalar x21 = 1250 * x11;
    const Scalar x22 = x21 * x[0];
    const Scalar x23 = x22 * x4;
    const Scalar x24 = x15 * x18;
    const Scalar x25 = x10 * x24;
    const Scalar x26 = x19 * x[0];
    const Scalar x27 = x26 * x3;
    const Scalar x28 = x2 - 1;
    const Scalar x29 = x1 * x28 * x3 * x[0];
    const Scalar x30 = x10 * x14;
    const Scalar x31 = x1 * x11 * x18 * x28 * x3 * x[0];
    const Scalar x32 = 2500 * x31;
    const Scalar x33 = x18 * x28 * x30;
    const Scalar x34 = 2500 * x29;
    const Scalar x35 = x10 * x28;
    const Scalar x36 = x1 * x35;
    const Scalar x37 = x1 * x28;
    const Scalar x38 = x7 * x[1];
    const Scalar x39 = x38 * x5;
    const Scalar x40 = 25 * x11;
    const Scalar x41 = x26 * x39 * x40;
    const Scalar x42 = x6 * x[1];
    const Scalar x43 = x16 * x42;
    const Scalar x44 = x40 * x5;
    const Scalar x45 = x29 * x44;
    const Scalar x46 = 50 * x31;
    const Scalar x47 = x46 * x5;
    const Scalar x48 = x15 * x47;
    const Scalar x49 = x27 * x28 * x44;
    const Scalar x50 = x1 * x11 * x14 * x15 * x18 * x28 * x3 * x5;
    const Scalar x51 = x50 * x[0];
    const Scalar x52 = x19 * x4;
    const Scalar x53 = x16 * x8;
    return (1.0 / 576.0) * coeffs[0] * x50 * x8 +
           (25.0 / 288.0) * coeffs[10] * x19 * x29 * x39 -
           1.0 / 576.0 * coeffs[11] * x37 * x41 +
           (1.0 / 576.0) * coeffs[12] * x43 * x45 -
           1.0 / 576.0 * coeffs[13] * x43 * x46 +
           (1.0 / 576.0) * coeffs[14] * x42 * x48 -
           1.0 / 576.0 * coeffs[15] * x42 * x49 -
           1.0 / 576.0 * coeffs[16] * x10 * x40 * x52 +
           (25.0 / 288.0) * coeffs[17] * x11 * x33 * x4 -
           25.0 / 288.0 * coeffs[18] * x35 * x52 +
           (1.0 / 576.0) * coeffs[19] * x19 * x36 * x40 -
           1.0 / 576.0 * coeffs[1] * x51 * x7 +
           (1.0 / 576.0) * coeffs[20] * x13 * x17 * x4 -
           1.0 / 576.0 * coeffs[21] * x20 * x23 +
           (1.0 / 576.0) * coeffs[22] * x23 * x25 -
           1.0 / 576.0 * coeffs[23] * x13 * x27 -
           1.0 / 576.0 * coeffs[24] * x21 * x29 * x30 +
           (1.0 / 576.0) * coeffs[25] * x14 * x32 * x9 -
           1.0 / 576.0 * coeffs[26] * x10 * x32 +
           (1.0 / 576.0) * coeffs[27] * x22 * x3 * x33 +
           (625.0 / 288.0) * coeffs[28] * x10 * x16 * x29 -
           1.0 / 576.0 * coeffs[29] * x20 * x34 +
           (1.0 / 576.0) * coeffs[2] * x51 * x[1] +
           (1.0 / 576.0) * coeffs[30] * x25 * x34 -
           625.0 / 288.0 * coeffs[31] * x27 * x35 -
           1.0 / 576.0 * coeffs[32] * x12 * x17 * x36 +
           (1.0 / 576.0) * coeffs[33] * x20 * x22 * x37 -
           1.0 / 576.0 * coeffs[34] * x22 * x24 * x36 +
           (1.0 / 576.0) * coeffs[35] * x12 * x26 * x35 -
           1.0 / 576.0 * coeffs[3] * x42 * x50 -
           1.0 / 576.0 * coeffs[4] * x45 * x53 +
           (1.0 / 576.0) * coeffs[5] * x46 * x53 -
           1.0 / 576.0 * coeffs[6] * x48 * x8 +
           (1.0 / 576.0) * coeffs[7] * x49 * x8 +
           (1.0 / 576.0) * coeffs[8] * x4 * x41 -
           1.0 / 576.0 * coeffs[9] * x14 * x38 * x47;
  }

  template <typename Scalar>
  static constexpr void eval_basis(const Scalar *x, Scalar *out) {
    const Scalar x0 = x[0] - 1;
    const Scalar x1 = 5 * x[1];
    const Scalar x2 = x1 - 1;
    const Scalar x3 = x0 * x2;
    const Scalar x4 = x[1] - 1;
    const Scalar x5 = 5 * x[0];
    const Scalar x6 = x5 - 1;
    const Scalar x7 = x5 - 4;
    const Scalar x8 = x1 - 4;
    const Scalar x9 = x5 - 3;
    const Scalar x10 = x1 - 3;
    const Scalar x11 = x5 - 2;
    const Scalar x12 = x1 - 2;
    const Scalar x13 = x10 * x11 * x12 * x4 * x6 * x7 * x8 * x9;
    const Scalar x14 = (1.0 / 576.0) * x13;
    const Scalar x15 = x2 * x[0];
    const Scalar x16 = x6 * x[1];
    const Scalar x17 = x15 * x16;
    const Scalar x18 = x10 * x11 * x12 * x7 * x8 * x9;
    const Scalar x19 = (1.0 / 576.0) * x18;
    const Scalar x20 = x16 * x3;
    const Scalar x21 = (25.0 / 576.0) * x[0];
    const Scalar x22 = x3 * x4;
    const Scalar x23 = x21 * x22;
    const Scalar x24 = x10 * x12 * x7 * x9;
    const Scalar x25 = x22 * x24;
    const Scalar x26 = x8 * x[0];
    const Scalar x27 = (25.0 / 288.0) * x26;
    const Scalar x28 = x27 * x6;
    const Scalar x29 = x10 * x7;
    const Scalar x30 = x11 * x12;
    const Scalar x31 = x29 * x30;
    const Scalar x32 = x8 * x9;
    const Scalar x33 = x30 * x32;
    const Scalar x34 = x10 * x33;
    const Scalar x35 = x21 * x[1];
    const Scalar x36 = (25.0 / 288.0) * x4;
    const Scalar x37 = x17 * x36;
    const Scalar x38 = x11 * x29;
    const Scalar x39 = x32 * x38;
    const Scalar x40 = x33 * x7;
    const Scalar x41 = x24 * x4;
    const Scalar x42 = (25.0 / 576.0) * x11 * x41;
    const Scalar x43 = x20 * x27;
    const Scalar x44 = x0 * x[1];
    const Scalar x45 = x20 * x36;
    const Scalar x46 = (625.0 / 576.0) * x4;
    const Scalar x47 = x46 * x[0];
    const Scalar x48 = (625.0 / 288.0) * x26;
    const Scalar x49 = x0 * x16;
    const Scalar x50 = x48 * x49;
    const Scalar x51 = x31 * x4;
    const Scalar x52 = x26 * x9;
    const Scalar x53 = x10 * x30;
    const Scalar x54 = x48 * x9;
    const Scalar x55 = x22 * x54 * x[1];
    const Scalar x56 = x20 * x4;
    const Scalar x57 = (625.0 / 144.0) * x56;
    const Scalar x58 = x52 * x57;
    const Scalar x59 = x26 * x57;
    const Scalar x60 = x54 * x56;
    const Scalar x61 = x30 * x7;
    const Scalar x62 = (625.0 / 288.0) * x20 * x[0];
    out[0] = x14 * x3;
    out[1] = -x14 * x15;
    out[2] = x17 * x19;
    out[3] = -x19 * x20;
    out[4] = -x18 * x23;
    out[5] = x25 * x28;
    out[6] = -x22 * x28 * x31;
    out[7] = x23 * x34 * x6;
    out[8] = x13 * x35;
    out[9] = -x37 * x39;
    out[10] = x37 * x40;
    out[11] = -x17 * x42;
    out[12] = x18 * x3 * x35;
    out[13] = -x24 * x43;
    out[14] = x31 * x43;
    out[15] = -x20 * x21 * x34;
    out[16] = -25.0 / 576.0 * x13 * x44;
    out[17] = x39 * x45;
    out[18] = -x40 * x45;
    out[19] = x20 * x42;
    out[20] = x18 * x44 * x47;
    out[21] = -x41 * x50;
    out[22] = x50 * x51;
    out[23] = -x46 * x49 * x52 * x53;
    out[24] = -x38 * x55;
    out[25] = x29 * x58;
    out[26] = -x38 * x59;
    out[27] = x10 * x11 * x60;
    out[28] = x55 * x61;
    out[29] = -x12 * x58 * x7;
    out[30] = x59 * x61;
    out[31] = -x30 * x60;
    out[32] = -625.0 / 576.0 * x11 * x25 * x[0] * x[1];
    out[33] = x41 * x62;
    out[34] = -x51 * x62;
    out[35] = x20 * x47 * x53 * x9;
  }

  template <typename Scalar>
  static constexpr void grad(const Scalar *x, const Scalar *coeffs,
                             Scalar *out) {
    const Scalar x0 = 5 * x[0];
    const Scalar x1 = x0 - 4;
    const Scalar x2 = x0 - 3;
    const Scalar x3 = x[0] - 1;
    const Scalar x4 = x0 * x3;
    const Scalar x5 = x2 * x4;
    const Scalar x6 = x1 * x5;
    const Scalar x7 = x0 - 2;
    const Scalar x8 = x4 * x7;
    const Scalar x9 = x1 * x8;
    const Scalar x10 = x2 * x7;
    const Scalar x11 = x10 * x4;
    const Scalar x12 = x1 * x10;
    const Scalar x13 = x12 * x[0];
    const Scalar x14 = x12 * x3;
    const Scalar x15 = x11 + x13 + x14 + x6 + x9;
    const Scalar x16 = 5 * x[1];
    const Scalar x17 = x16 - 4;
    const Scalar x18 = x16 - 3;
    const Scalar x19 = x16 - 2;
    const Scalar x20 = x18 * x19;
    const Scalar x21 = x17 * x20;
    const Scalar x22 = x21 * x[1];
    const Scalar x23 = x[1] - 1;
    const Scalar x24 = 625 * x23;
    const Scalar x25 = x22 * x24;
    const Scalar x26 = x0 - 1;
    const Scalar x27 = x1 * x26;
    const Scalar x28 = x27 * x4;
    const Scalar x29 = x26 * x5;
    const Scalar x30 = x2 * x27;
    const Scalar x31 = x30 * x[0];
    const Scalar x32 = x3 * x30;
    const Scalar x33 = x28 + x29 + x31 + x32 + x6;
    const Scalar x34 = 1250 * x23;
    const Scalar x35 = x22 * x34;
    const Scalar x36 = x26 * x8;
    const Scalar x37 = x27 * x7;
    const Scalar x38 = x37 * x[0];
    const Scalar x39 = x3 * x37;
    const Scalar x40 = x28 + x36 + x38 + x39 + x9;
    const Scalar x41 = x10 * x26;
    const Scalar x42 = x41 * x[0];
    const Scalar x43 = x3 * x41;
    const Scalar x44 = x11 + x29 + x36 + x42 + x43;
    const Scalar x45 = x16 - 1;
    const Scalar x46 = x17 * x45;
    const Scalar x47 = x18 * x46;
    const Scalar x48 = x47 * x[1];
    const Scalar x49 = x34 * x48;
    const Scalar x50 = 2500 * x23;
    const Scalar x51 = x48 * x50;
    const Scalar x52 = x19 * x46;
    const Scalar x53 = x52 * x[1];
    const Scalar x54 = x34 * x53;
    const Scalar x55 = x50 * x53;
    const Scalar x56 = x20 * x45;
    const Scalar x57 = x56 * x[1];
    const Scalar x58 = x24 * x57;
    const Scalar x59 = x34 * x57;
    const Scalar x60 = x20 * x46;
    const Scalar x61 = x60 * x[1];
    const Scalar x62 = 25 * x61;
    const Scalar x63 = 50 * x61;
    const Scalar x64 = x23 * x60;
    const Scalar x65 = 25 * x64;
    const Scalar x66 = 50 * x64;
    const Scalar x67 = x10 * x27;
    const Scalar x68 = x0 * x12 + x0 * x30 + x0 * x37 + x0 * x41 + x67;
    const Scalar x69 = 50 * x23;
    const Scalar x70 = x53 * x69;
    const Scalar x71 = 25 * x23;
    const Scalar x72 = x57 * x71;
    const Scalar x73 = x22 * x71;
    const Scalar x74 = x48 * x69;
    const Scalar x75 = 5 * x14 + 5 * x32 + 5 * x39 + 5 * x43 + x67;
    const Scalar x76 = x16 * x23;
    const Scalar x77 = x18 * x76;
    const Scalar x78 = x17 * x77;
    const Scalar x79 = x19 * x76;
    const Scalar x80 = x17 * x79;
    const Scalar x81 = x20 * x76;
    const Scalar x82 = x21 * x23;
    const Scalar x83 = x22 + x78 + x80 + x81 + x82;
    const Scalar x84 = 625 * x3;
    const Scalar x85 = x83 * x84;
    const Scalar x86 = 1250 * x3;
    const Scalar x87 = x83 * x86;
    const Scalar x88 = x46 * x76;
    const Scalar x89 = x45 * x77;
    const Scalar x90 = x23 * x47;
    const Scalar x91 = x48 + x78 + x88 + x89 + x90;
    const Scalar x92 = x86 * x91;
    const Scalar x93 = 2500 * x3;
    const Scalar x94 = x91 * x93;
    const Scalar x95 = x45 * x79;
    const Scalar x96 = x23 * x52;
    const Scalar x97 = x53 + x80 + x88 + x95 + x96;
    const Scalar x98 = x86 * x97;
    const Scalar x99 = x93 * x97;
    const Scalar x100 = x23 * x56;
    const Scalar x101 = x100 + x57 + x81 + x89 + x95;
    const Scalar x102 = x101 * x84;
    const Scalar x103 = x101 * x86;
    const Scalar x104 = x67 * x97;
    const Scalar x105 = x67 * x[0];
    const Scalar x106 = 25 * x101;
    const Scalar x107 = 25 * x3;
    const Scalar x108 = 50 * x3;
    const Scalar x109 = x16 * x21 + x16 * x47 + x16 * x52 + x16 * x56 + x60;
    const Scalar x110 = x107 * x109;
    const Scalar x111 = x108 * x109;
    const Scalar x112 = x109 * x67;
    const Scalar x113 = 5 * x100 + x60 + 5 * x82 + 5 * x90 + 5 * x96;
    const Scalar x114 = x107 * x113;
    const Scalar x115 = x108 * x113;
    const Scalar x116 = x113 * x67;
    out[0] = (1.0 / 576.0) * coeffs[0] * x64 * x75 +
             (1.0 / 576.0) * coeffs[10] * x68 * x70 -
             1.0 / 576.0 * coeffs[11] * x68 * x72 +
             (1.0 / 576.0) * coeffs[12] * x15 * x62 -
             1.0 / 576.0 * coeffs[13] * x33 * x63 +
             (1.0 / 576.0) * coeffs[14] * x40 * x63 -
             1.0 / 576.0 * coeffs[15] * x44 * x62 -
             1.0 / 576.0 * coeffs[16] * x73 * x75 +
             (1.0 / 576.0) * coeffs[17] * x74 * x75 -
             1.0 / 576.0 * coeffs[18] * x70 * x75 +
             (1.0 / 576.0) * coeffs[19] * x72 * x75 -
             1.0 / 576.0 * coeffs[1] * x64 * x68 +
             (1.0 / 576.0) * coeffs[20] * x15 * x25 -
             1.0 / 576.0 * coeffs[21] * x33 * x35 +
             (1.0 / 576.0) * coeffs[22] * x35 * x40 -
             1.0 / 576.0 * coeffs[23] * x25 * x44 -
             1.0 / 576.0 * coeffs[24] * x15 * x49 +
             (1.0 / 576.0) * coeffs[25] * x33 * x51 -
             1.0 / 576.0 * coeffs[26] * x40 * x51 +
             (1.0 / 576.0) * coeffs[27] * x44 * x49 +
             (1.0 / 576.0) * coeffs[28] * x15 * x54 -
             1.0 / 576.0 * coeffs[29] * x33 * x55 +
             (1.0 / 576.0) * coeffs[2] * x61 * x68 +
             (1.0 / 576.0) * coeffs[30] * x40 * x55 -
             1.0 / 576.0 * coeffs[31] * x44 * x54 -
             1.0 / 576.0 * coeffs[32] * x15 * x58 +
             (1.0 / 576.0) * coeffs[33] * x33 * x59 -
             1.0 / 576.0 * coeffs[34] * x40 * x59 +
             (1.0 / 576.0) * coeffs[35] * x44 * x58 -
             1.0 / 576.0 * coeffs[3] * x61 * x75 -
             1.0 / 576.0 * coeffs[4] * x15 * x65 +
             (1.0 / 576.0) * coeffs[5] * x33 * x66 -
             1.0 / 576.0 * coeffs[6] * x40 * x66 +
             (1.0 / 576.0) * coeffs[7] * x44 * x65 +
             (1.0 / 576.0) * coeffs[8] * x68 * x73 -
             1.0 / 576.0 * coeffs[9] * x68 * x74;
    out[1] = (1.0 / 576.0) * coeffs[0] * x116 * x3 +
             (25.0 / 288.0) * coeffs[10] * x104 * x[0] -
             1.0 / 576.0 * coeffs[11] * x105 * x106 +
             (1.0 / 576.0) * coeffs[12] * x110 * x13 -
             1.0 / 576.0 * coeffs[13] * x111 * x31 +
             (1.0 / 576.0) * coeffs[14] * x111 * x38 -
             1.0 / 576.0 * coeffs[15] * x110 * x42 -
             1.0 / 576.0 * coeffs[16] * x107 * x67 * x83 +
             (1.0 / 576.0) * coeffs[17] * x108 * x67 * x91 -
             1.0 / 576.0 * coeffs[18] * x104 * x108 +
             (1.0 / 576.0) * coeffs[19] * x106 * x3 * x67 -
             1.0 / 576.0 * coeffs[1] * x116 * x[0] +
             (1.0 / 576.0) * coeffs[20] * x13 * x85 -
             1.0 / 576.0 * coeffs[21] * x31 * x87 +
             (1.0 / 576.0) * coeffs[22] * x38 * x87 -
             1.0 / 576.0 * coeffs[23] * x42 * x85 -
             1.0 / 576.0 * coeffs[24] * x13 * x92 +
             (1.0 / 576.0) * coeffs[25] * x31 * x94 -
             1.0 / 576.0 * coeffs[26] * x38 * x94 +
             (1.0 / 576.0) * coeffs[27] * x42 * x92 +
             (1.0 / 576.0) * coeffs[28] * x13 * x98 -
             1.0 / 576.0 * coeffs[29] * x31 * x99 +
             (1.0 / 576.0) * coeffs[2] * x112 * x[0] +
             (1.0 / 576.0) * coeffs[30] * x38 * x99 -
             1.0 / 576.0 * coeffs[31] * x42 * x98 -
             1.0 / 576.0 * coeffs[32] * x102 * x13 +
             (1.0 / 576.0) * coeffs[33] * x103 * x31 -
             1.0 / 576.0 * coeffs[34] * x103 * x38 +
             (1.0 / 576.0) * coeffs[35] * x102 * x42 -
             1.0 / 576.0 * coeffs[3] * x112 * x3 -
             1.0 / 576.0 * coeffs[4] * x114 * x13 +
             (1.0 / 576.0) * coeffs[5] * x115 * x31 -
             1.0 / 576.0 * coeffs[6] * x115 * x38 +
             (1.0 / 576.0) * coeffs[7] * x114 * x42 +
             (25.0 / 576.0) * coeffs[8] * x105 * x83 -
             25.0 / 288.0 * coeffs[9] * x105 * x91;
  }

  template <typename Scalar>
  static constexpr void grad_basis(const Scalar *x, Scalar (*out)[2]) {
    const Scalar x0 = 5 * x[1];
    const Scalar x1 = x0 - 4;
    const Scalar x2 = x0 - 3;
    const Scalar x3 = x1 * x2;
    const Scalar x4 = x0 - 1;
    const Scalar x5 = x0 - 2;
    const Scalar x6 = x4 * x5;
    const Scalar x7 = x3 * x6;
    const Scalar x8 = (1.0 / 576.0) * x7;
    const Scalar x9 = x[1] - 1;
    const Scalar x10 = x[0] - 1;
    const Scalar x11 = 5 * x[0];
    const Scalar x12 = x11 - 2;
    const Scalar x13 = x11 - 4;
    const Scalar x14 = x11 - 3;
    const Scalar x15 = x13 * x14;
    const Scalar x16 = x12 * x15;
    const Scalar x17 = x10 * x16;
    const Scalar x18 = x11 - 1;
    const Scalar x19 = x15 * x18;
    const Scalar x20 = x10 * x19;
    const Scalar x21 = x12 * x18;
    const Scalar x22 = x13 * x21;
    const Scalar x23 = x10 * x22;
    const Scalar x24 = x14 * x21;
    const Scalar x25 = x10 * x24;
    const Scalar x26 = x15 * x21;
    const Scalar x27 = 5 * x17 + 5 * x20 + 5 * x23 + 5 * x25 + x26;
    const Scalar x28 = x27 * x9;
    const Scalar x29 = (1.0 / 576.0) * x26;
    const Scalar x30 = x3 * x5;
    const Scalar x31 = x30 * x9;
    const Scalar x32 = x3 * x4;
    const Scalar x33 = x32 * x9;
    const Scalar x34 = x1 * x6;
    const Scalar x35 = x34 * x9;
    const Scalar x36 = x2 * x6;
    const Scalar x37 = x36 * x9;
    const Scalar x38 = 5 * x31 + 5 * x33 + 5 * x35 + 5 * x37 + x7;
    const Scalar x39 = x10 * x38;
    const Scalar x40 = x11 * x16 + x11 * x19 + x11 * x22 + x11 * x24 + x26;
    const Scalar x41 = x40 * x8;
    const Scalar x42 = x29 * x[0];
    const Scalar x43 = x0 * x30 + x0 * x32 + x0 * x34 + x0 * x36 + x7;
    const Scalar x44 = x10 * x43;
    const Scalar x45 = x10 * x11;
    const Scalar x46 = x15 * x45;
    const Scalar x47 = x12 * x45;
    const Scalar x48 = x13 * x47;
    const Scalar x49 = x14 * x47;
    const Scalar x50 = x16 * x[0];
    const Scalar x51 = x17 + x46 + x48 + x49 + x50;
    const Scalar x52 = x7 * x9;
    const Scalar x53 = (25.0 / 576.0) * x52;
    const Scalar x54 = (25.0 / 576.0) * x39;
    const Scalar x55 = x18 * x45;
    const Scalar x56 = x13 * x55;
    const Scalar x57 = x14 * x55;
    const Scalar x58 = x19 * x[0];
    const Scalar x59 = x20 + x46 + x56 + x57 + x58;
    const Scalar x60 = (25.0 / 288.0) * x52;
    const Scalar x61 = (25.0 / 288.0) * x39;
    const Scalar x62 = x21 * x45;
    const Scalar x63 = x22 * x[0];
    const Scalar x64 = x23 + x48 + x56 + x62 + x63;
    const Scalar x65 = x24 * x[0];
    const Scalar x66 = x25 + x49 + x57 + x62 + x65;
    const Scalar x67 = x40 * x9;
    const Scalar x68 = x30 * x[1];
    const Scalar x69 = (25.0 / 576.0) * x68;
    const Scalar x70 = x0 * x9;
    const Scalar x71 = x3 * x70;
    const Scalar x72 = x5 * x70;
    const Scalar x73 = x1 * x72;
    const Scalar x74 = x2 * x72;
    const Scalar x75 = x31 + x68 + x71 + x73 + x74;
    const Scalar x76 = x26 * x[0];
    const Scalar x77 = (25.0 / 576.0) * x76;
    const Scalar x78 = x32 * x[1];
    const Scalar x79 = (25.0 / 288.0) * x67;
    const Scalar x80 = x4 * x70;
    const Scalar x81 = x1 * x80;
    const Scalar x82 = x2 * x80;
    const Scalar x83 = x33 + x71 + x78 + x81 + x82;
    const Scalar x84 = (25.0 / 288.0) * x76;
    const Scalar x85 = x34 * x[1];
    const Scalar x86 = x6 * x70;
    const Scalar x87 = x35 + x73 + x81 + x85 + x86;
    const Scalar x88 = x36 * x[1];
    const Scalar x89 = (25.0 / 576.0) * x88;
    const Scalar x90 = x37 + x74 + x82 + x86 + x88;
    const Scalar x91 = x7 * x[1];
    const Scalar x92 = (25.0 / 576.0) * x91;
    const Scalar x93 = (25.0 / 576.0) * x44;
    const Scalar x94 = (25.0 / 288.0) * x91;
    const Scalar x95 = (25.0 / 288.0) * x44;
    const Scalar x96 = x10 * x26;
    const Scalar x97 = (25.0 / 576.0) * x96;
    const Scalar x98 = (25.0 / 288.0) * x28;
    const Scalar x99 = (25.0 / 288.0) * x96;
    const Scalar x100 = x68 * x9;
    const Scalar x101 = (625.0 / 576.0) * x100;
    const Scalar x102 = x10 * x75;
    const Scalar x103 = (625.0 / 576.0) * x102;
    const Scalar x104 = (625.0 / 288.0) * x100;
    const Scalar x105 = (625.0 / 288.0) * x102;
    const Scalar x106 = x78 * x9;
    const Scalar x107 = (625.0 / 288.0) * x106;
    const Scalar x108 = x10 * x83;
    const Scalar x109 = (625.0 / 288.0) * x108;
    const Scalar x110 = (625.0 / 144.0) * x106;
    const Scalar x111 = (625.0 / 144.0) * x108;
    const Scalar x112 = x85 * x9;
    const Scalar x113 = (625.0 / 288.0) * x112;
    const Scalar x114 = x10 * x87;
    const Scalar x115 = (625.0 / 288.0) * x114;
    const Scalar x116 = (625.0 / 144.0) * x112;
    const Scalar x117 = (625.0 / 144.0) * x114;
    const Scalar x118 = x88 * x9;
    const Scalar x119 = (625.0 / 576.0) * x118;
    const Scalar x120 = x10 * x90;
    const Scalar x121 = (625.0 / 576.0) * x120;
    const Scalar x122 = (625.0 / 288.0) * x118;
    const Scalar x123 = (625.0 / 288.0) * x120;
    out[0][0] = x28 * x8;
    out[0][1] = x29 * x39;
    out[1][0] = -x41 * x9;
    out[1][1] = -x38 * x42;
    out[2][0] = x41 * x[1];
    out[2][1] = x42 * x43;
    out[3][0] = -x27 * x8 * x[1];
    out[3][1] = -x29 * x44;
    out[4][0] = -x51 * x53;
    out[4][1] = -x50 * x54;
    out[5][0] = x59 * x60;
    out[5][1] = x58 * x61;
    out[6][0] = -x60 * x64;
    out[6][1] = -x61 * x63;
    out[7][0] = x53 * x66;
    out[7][1] = x54 * x65;
    out[8][0] = x67 * x69;
    out[8][1] = x75 * x77;
    out[9][0] = -x78 * x79;
    out[9][1] = -x83 * x84;
    out[10][0] = x79 * x85;
    out[10][1] = x84 * x87;
    out[11][0] = -x67 * x89;
    out[11][1] = -x77 * x90;
    out[12][0] = x51 * x92;
    out[12][1] = x50 * x93;
    out[13][0] = -x59 * x94;
    out[13][1] = -x58 * x95;
    out[14][0] = x64 * x94;
    out[14][1] = x63 * x95;
    out[15][0] = -x66 * x92;
    out[15][1] = -x65 * x93;
    out[16][0] = -x28 * x69;
    out[16][1] = -x75 * x97;
    out[17][0] = x78 * x98;
    out[17][1] = x83 * x99;
    out[18][0] = -x85 * x98;
    out[18][1] = -x87 * x99;
    out[19][0] = x28 * x89;
    out[19][1] = x90 * x97;
    out[20][0] = x101 * x51;
    out[20][1] = x103 * x50;
    out[21][0] = -x104 * x59;
    out[21][1] = -x105 * x58;
    out[22][0] = x104 * x64;
    out[22][1] = x105 * x63;
    out[23][0] = -x101 * x66;
    out[23][1] = -x103 * x65;
    out[24][0] = -x107 * x51;
    out[24][1] = -x109 * x50;
    out[25][0] = x110 * x59;
    out[25][1] = x111 * x58;
    out[26][0] = -x110 * x64;
    out[26][1] = -x111 * x63;
    out[27][0] = x107 * x66;
    out[27][1] = x109 * x65;
    out[28][0] = x113 * x51;
    out[28][1] = x115 * x50;
    out[29][0] = -x116 * x59;
    out[29][1] = -x117 * x58;
    out[30][0] = x116 * x64;
    out[30][1] = x117 * x63;
    out[31][0] = -x113 * x66;
    out[31][1] = -x115 * x65;
    out[32][0] = -x119 * x51;
    out[32][1] = -x121 * x50;
    out[33][0] = x122 * x59;
    out[33][1] = x123 * x58;
    out[34][0] = -x122 * x64;
    out[34][1] = -x123 * x63;
    out[35][0] = x119 * x66;
    out[35][1] = x121 * x65;
  }

  template <typename Scalar> static constexpr void node(dim_t i, Scalar *out) {
    dim_t idxs[2];
    node_idxs(i, idxs);
    out[0] = static_cast<Scalar>(idxs[0]) / order;
    out[1] = static_cast<Scalar>(idxs[1]) / order;
  }

  template <typename Scalar>
  static constexpr void interpolation_nodes(Scalar (*out)[2]) {
    out[0][0] = static_cast<Scalar>(0) / order;
    out[0][1] = static_cast<Scalar>(0) / order;
    out[1][0] = static_cast<Scalar>(5) / order;
    out[1][1] = static_cast<Scalar>(0) / order;
    out[2][0] = static_cast<Scalar>(5) / order;
    out[2][1] = static_cast<Scalar>(5) / order;
    out[3][0] = static_cast<Scalar>(0) / order;
    out[3][1] = static_cast<Scalar>(5) / order;
    out[4][0] = static_cast<Scalar>(1) / order;
    out[4][1] = static_cast<Scalar>(0) / order;
    out[5][0] = static_cast<Scalar>(2) / order;
    out[5][1] = static_cast<Scalar>(0) / order;
    out[6][0] = static_cast<Scalar>(3) / order;
    out[6][1] = static_cast<Scalar>(0) / order;
    out[7][0] = static_cast<Scalar>(4) / order;
    out[7][1] = static_cast<Scalar>(0) / order;
    out[8][0] = static_cast<Scalar>(5) / order;
    out[8][1] = static_cast<Scalar>(1) / order;
    out[9][0] = static_cast<Scalar>(5) / order;
    out[9][1] = static_cast<Scalar>(2) / order;
    out[10][0] = static_cast<Scalar>(5) / order;
    out[10][1] = static_cast<Scalar>(3) / order;
    out[11][0] = static_cast<Scalar>(5) / order;
    out[11][1] = static_cast<Scalar>(4) / order;
    out[12][0] = static_cast<Scalar>(1) / order;
    out[12][1] = static_cast<Scalar>(5) / order;
    out[13][0] = static_cast<Scalar>(2) / order;
    out[13][1] = static_cast<Scalar>(5) / order;
    out[14][0] = static_cast<Scalar>(3) / order;
    out[14][1] = static_cast<Scalar>(5) / order;
    out[15][0] = static_cast<Scalar>(4) / order;
    out[15][1] = static_cast<Scalar>(5) / order;
    out[16][0] = static_cast<Scalar>(0) / order;
    out[16][1] = static_cast<Scalar>(1) / order;
    out[17][0] = static_cast<Scalar>(0) / order;
    out[17][1] = static_cast<Scalar>(2) / order;
    out[18][0] = static_cast<Scalar>(0) / order;
    out[18][1] = static_cast<Scalar>(3) / order;
    out[19][0] = static_cast<Scalar>(0) / order;
    out[19][1] = static_cast<Scalar>(4) / order;
    out[20][0] = static_cast<Scalar>(1) / order;
    out[20][1] = static_cast<Scalar>(1) / order;
    out[21][0] = static_cast<Scalar>(2) / order;
    out[21][1] = static_cast<Scalar>(1) / order;
    out[22][0] = static_cast<Scalar>(3) / order;
    out[22][1] = static_cast<Scalar>(1) / order;
    out[23][0] = static_cast<Scalar>(4) / order;
    out[23][1] = static_cast<Scalar>(1) / order;
    out[24][0] = static_cast<Scalar>(1) / order;
    out[24][1] = static_cast<Scalar>(2) / order;
    out[25][0] = static_cast<Scalar>(2) / order;
    out[25][1] = static_cast<Scalar>(2) / order;
    out[26][0] = static_cast<Scalar>(3) / order;
    out[26][1] = static_cast<Scalar>(2) / order;
    out[27][0] = static_cast<Scalar>(4) / order;
    out[27][1] = static_cast<Scalar>(2) / order;
    out[28][0] = static_cast<Scalar>(1) / order;
    out[28][1] = static_cast<Scalar>(3) / order;
    out[29][0] = static_cast<Scalar>(2) / order;
    out[29][1] = static_cast<Scalar>(3) / order;
    out[30][0] = static_cast<Scalar>(3) / order;
    out[30][1] = static_cast<Scalar>(3) / order;
    out[31][0] = static_cast<Scalar>(4) / order;
    out[31][1] = static_cast<Scalar>(3) / order;
    out[32][0] = static_cast<Scalar>(1) / order;
    out[32][1] = static_cast<Scalar>(4) / order;
    out[33][0] = static_cast<Scalar>(2) / order;
    out[33][1] = static_cast<Scalar>(4) / order;
    out[34][0] = static_cast<Scalar>(3) / order;
    out[34][1] = static_cast<Scalar>(4) / order;
    out[35][0] = static_cast<Scalar>(4) / order;
    out[35][1] = static_cast<Scalar>(4) / order;
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
      break;
    case 1:
      out[0] = 5;
      out[1] = 0;
      break;
    case 2:
      out[0] = 5;
      out[1] = 5;
      break;
    case 3:
      out[0] = 0;
      out[1] = 5;
      break;
    case 4:
      out[0] = 1;
      out[1] = 0;
      break;
    case 5:
      out[0] = 2;
      out[1] = 0;
      break;
    case 6:
      out[0] = 3;
      out[1] = 0;
      break;
    case 7:
      out[0] = 4;
      out[1] = 0;
      break;
    case 8:
      out[0] = 5;
      out[1] = 1;
      break;
    case 9:
      out[0] = 5;
      out[1] = 2;
      break;
    case 10:
      out[0] = 5;
      out[1] = 3;
      break;
    case 11:
      out[0] = 5;
      out[1] = 4;
      break;
    case 12:
      out[0] = 1;
      out[1] = 5;
      break;
    case 13:
      out[0] = 2;
      out[1] = 5;
      break;
    case 14:
      out[0] = 3;
      out[1] = 5;
      break;
    case 15:
      out[0] = 4;
      out[1] = 5;
      break;
    case 16:
      out[0] = 0;
      out[1] = 1;
      break;
    case 17:
      out[0] = 0;
      out[1] = 2;
      break;
    case 18:
      out[0] = 0;
      out[1] = 3;
      break;
    case 19:
      out[0] = 0;
      out[1] = 4;
      break;
    case 20:
      out[0] = 1;
      out[1] = 1;
      break;
    case 21:
      out[0] = 2;
      out[1] = 1;
      break;
    case 22:
      out[0] = 3;
      out[1] = 1;
      break;
    case 23:
      out[0] = 4;
      out[1] = 1;
      break;
    case 24:
      out[0] = 1;
      out[1] = 2;
      break;
    case 25:
      out[0] = 2;
      out[1] = 2;
      break;
    case 26:
      out[0] = 3;
      out[1] = 2;
      break;
    case 27:
      out[0] = 4;
      out[1] = 2;
      break;
    case 28:
      out[0] = 1;
      out[1] = 3;
      break;
    case 29:
      out[0] = 2;
      out[1] = 3;
      break;
    case 30:
      out[0] = 3;
      out[1] = 3;
      break;
    case 31:
      out[0] = 4;
      out[1] = 3;
      break;
    case 32:
      out[0] = 1;
      out[1] = 4;
      break;
    case 33:
      out[0] = 2;
      out[1] = 4;
      break;
    case 34:
      out[0] = 3;
      out[1] = 4;
      break;
    case 35:
      out[0] = 4;
      out[1] = 4;
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
      idxs[0] = 3;
      idxs[1] = 2;
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
